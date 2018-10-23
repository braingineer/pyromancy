# coding=utf-8
import gc
import json
import importlib
import logging
import os
import sys
import time
from copy import deepcopy

import redis
import torch
from rq import Queue, get_failed_queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.worker import SimpleWorker, WorkerStatus, setup_loghandlers, VERSION, \
    StopRequested


logging.getLogger(__name__).addHandler(logging.NullHandler())


class Trial(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self._kwargs = kwargs

    def deepcopy_and_modify_args(self, args):
        """
        Override this to modify args that require additional serialization logic
        :param args:
        :return:
        """
        args = deepcopy(args)
        for key, value in self._kwargs.items():
            setattr(args, key, value)
        return args


class Experiment(object):

    def __init__(self, name, trials):
        self.trials = trials
        self.name = name

    def run(self, args, worker_class, gpu_ids=None):
        raise NotImplementedError


class ExperimentWorker(object):
    """
    Worker class that ensures that experiment code is run on exactly one GPU
    """

    def run_once(self, args):
        """
        Override for experiment-specific worker
        :param args: argparse namespace-like object
        :return:
        """
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class RQExperimentWorker(SimpleWorker):

    def __init__(self, queues, name=None, default_result_ttl=None,
                 connection=None, exc_handler=None, exception_handlers=None,
                 default_worker_ttl=None, job_class=None, queue_class=None):
        """Overrides Worker.__init__ to load gpu id from worker name"""
        prefix, gpu_id, _ = name.split("_")
        self.gpu_id = int(gpu_id)
        super(RQExperimentWorker, self).__init__(queues,
                                                 name=name,
                                                 default_result_ttl=default_result_ttl,
                                                 connection=connection,
                                                 exc_handler=exc_handler,
                                                 exception_handlers=exception_handlers,
                                                 default_worker_ttl=default_worker_ttl,
                                                 job_class=job_class,
                                                 queue_class=queue_class)

    def main_work_horse(self, *args, **kwargs):
        super(RQExperimentWorker, self).main_work_horse(*args, **kwargs)

    def work(self, burst=False, logging_level="INFO"):
        """Hacky override of Worker.work that wraps calls to execute_job with torch.cuda.device
        Copied from rq.worker.Worker.work
        """
        setup_loghandlers(logging_level)
        self._install_signal_handlers()

        did_perform_work = False
        self.register_birth()
        self.log.info(f"RQ worker {self.key!r} started, version {VERSION}")
        self.set_state(WorkerStatus.STARTED)

        try:
            while True:
                try:
                    self.check_for_suspension(burst)

                    if self.should_run_maintenance_tasks:
                        self.clean_registries()

                    if self._stop_requested:
                        self.log.info("Stopping on request")
                        break

                    if burst:
                        timeout = None
                    else:
                        timeout = max(1, self.default_worker_ttl - 60)

                    result = self.dequeue_job_and_maintain_ttl(timeout)
                    if result is None:
                        if burst:
                            self.log.info(f"RQ worker {self.key!r} done, "
                                          "quitting")
                        break

                    job, queue = result

                    # Modified Section
                    if torch.cuda.is_available():
                        with torch.cuda.device(self.gpu_id):
                            self.execute_job(job, queue)
                    else:
                        self.execute_job(job, queue)

                    # Update Experiment status tracker / potentially requeue job
                    job.meta["attempts"] = job.meta["attempts"] + 1
                    job.save_meta()
                    if job.is_failed:
                        max_attempts = job.meta["max_attempts"]
                        if job.meta["attempts"] < max_attempts:
                            failed_queue = get_failed_queue(self.connection, job.__class__)
                            failed_queue.requeue(job.id)
                            self.log.info("Job %s failed on attempt %s of %s, re-queueing",
                                          job.id, job.meta["attempts"], job.meta["max_attempts"])
                    else:
                        job_key = RQExperiment.unpack_job_id(job.id)
                        self.connection.hset(job_key["experiment"], job_key["trial"], JobStatus.FINISHED)

                    gc.collect()

                    self.heartbeat()

                    did_perform_work = True

                    raise StopRequested
                    # End Modified Section

                except StopRequested:
                    break
        finally:
            if not self.is_horse:
                self.register_death()
        return did_perform_work


class RQExperiment(Experiment):
    """Experiment that uses RQ as simple worker queue

    For our purposes, "experiment.name|trial.name" == RQ job id
    """

    def __init__(self, name, trials, job_class=None,
                 redis_host="localhost", redis_port=6379, queue_name="default"):
        super(RQExperiment, self).__init__(name, trials)
        self.redis_conn = redis.Redis(redis_host, redis_port)
        self.job_class = job_class or RQExperimentJob
        self.job_queue = Queue(queue_name, connection=self.redis_conn, job_class=self.job_class)
        self.failed_queue = get_failed_queue(self.redis_conn, job_class=self.job_class)

    @staticmethod
    def make_job_id(experiment_name, trial_name):
        key = {"experiment": experiment_name, "trial": trial_name}
        return json.dumps(key, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def unpack_job_id(job_id):
        return json.loads(job_id)

    def run(self, cli_args, target_func, single_thread=False, gpu_id=None,
            queue_ttl=-1, result_ttl=-1, log_level=logging.INFO, job_timeout=86400,
            poll=False, poll_interval=5, requeue=False, max_attempts=3):
        """
        Run an Experiment using Redis Queue
        :param cli_args: Arguments that override trial default arguments
        :param target_func: Function (in its own module) to run for each Trial
        :param single_thread: If True, run trials in current thread
        :param gpu_id: GPU to use in single thread mode
        :param queue_ttl: Maximum time (in seconds) a Trial may stay in queue
        :param log_level: Logging level (for messages)
        :param result_ttl: Maximum time (in seconds) for any results to be stored in queue
        :param job_timeout: Maximum time (in seconds) a job may run before being terminated
        :param poll: If True, wait for jobs to complete
        :param poll_interval: The time (in seconds) to sleep between checks when polling
        :param requeue: If True, re-queue previously completed jobs
        :param max_attempts: Maximum number of attempts to re-run failed jobs
        :return:
        """
        logging.getLogger().setLevel(log_level)

        if requeue is False:
            trial_status = self.get_trial_status()
            finished = sum(1 for status in trial_status.values() if status == JobStatus.FINISHED)
            remaining = len(self.trials) - finished
            if remaining == 0:
                logging.info("Experiment completed!")
                return

        func_target = target_func
        func_module = os.path.abspath(target_func.__module__)
        func_description = f"{func_module}|{func_target.__name__}"

        for trial in self.trials:
            job_id = self.make_job_id(self.name, trial.name)
            runtime_args = trial.deepcopy_and_modify_args(cli_args)

            # Don't do any redis job tracking if running locally
            if single_thread:
                logging.info("Running job %s locally", job_id)
                # noinspection PyBroadException
                try:
                    if torch.cuda.is_available() and gpu_id is not None:
                        with torch.cuda.device(gpu_id):
                            _ = target_func(runtime_args)
                    else:
                        _ = target_func(runtime_args)
                    logging.info("Completed job %s", job_id)
                except KeyboardInterrupt:
                    logging.error('Caught interrupt, exiting')
                    exit(0)
                except Exception:  # super broad, but maybe reasonable in this case...
                    logging.error("Job %s failed", job_id, exc_info=True)
                continue

            job_meta = {"attempts": 0, "max_attempts": max_attempts}

            # Don"t queued jobs that are queued or running
            # TODO slack message with collective job submit status
            try:
                job = Job.fetch(job_id, connection=self.redis_conn)
                status = job.get_status()
                attempts = job.meta.get("attempts", 0)
                if attempts > 0:
                    job_meta["attempts"] = attempts
                if requeue and (status == JobStatus.FINISHED):
                    # Remove old job if it was previously completed
                    logging.info("Removing completed job %s", job_id)
                    job.delete()
                if status in (JobStatus.QUEUED, JobStatus.DEFERRED, JobStatus.STARTED):
                    logging.info("Experiment %s: trial %s already queued", self.name, trial.name)
                    self.redis_conn.hset(self.name, trial.name, status)
                    # TODO what about "zombie" jobs? Can they exist? Check TTL?
                    continue
                elif status == JobStatus.FAILED:
                    if attempts < job_meta["max_attempts"]:
                        self.failed_queue.requeue(job_id)
                        logging.info("Job %s has failed on %s of %s attempts, re-queued",
                                     job_id, attempts, job_meta["max_attempts"])
                        continue
                    else:
                        if requeue:
                            job.meta["attempts"] = 0
                            job.save_meta()
                            self.failed_queue.requeue(job_id)
                            logging.info("Re-queued previously failed job %s", job_id)
                        else:
                            logging.error("Job %s failed %s of %s times, skipping",
                                          job_id, attempts, job_meta["max_attempts"])
                        continue

            except NoSuchJobError:
                pass

            _ = self.job_queue.enqueue(func_description, runtime_args, ttl=queue_ttl,
                                       result_ttl=result_ttl, timeout=job_timeout,
                                       job_id=job_id, meta=job_meta)
            self.redis_conn.hset(self.name, trial.name, JobStatus.QUEUED)
            logging.info("Queued job %s", job_id)

        if poll and (single_thread is False):
            self.wait_for_jobs(poll_interval)

    def wait_for_jobs(self, poll_interval):
        while True:
            running = 0
            trial_status = self.get_trial_status()
            for trial_name, status in trial_status.items():
                if status == JobStatus.FINISHED:
                    continue
                else:
                    trial_name = trial_name.decode("utf-8")
                    job_id = self.make_job_id(self.name, trial_name)
                    try:
                        job = Job.fetch(job_id, connection=self.redis_conn)
                    except NoSuchJobError:
                        fail_queue = get_failed_queue(self.redis_conn, RQExperimentJob)
                        job = fail_queue.fetch_job(job_id)
                        if job is None:
                            logging.error("Could not find job %s", job_id)
                            raise

                    job_status = job.get_status()
                    if job_status == JobStatus.FINISHED:
                        logging.info("Experiment %s: completed trial %s", self.name, trial_name)
                    elif job_status == JobStatus.FAILED:
                        attempts = job.meta["attempts"]
                        max_attempts = job.meta["max_attempts"]
                        if attempts < max_attempts:
                            running += 1
                        else:
                            logging.error("Job %s failed after %s attempts", job_id, attempts)
                    elif job_status in (JobStatus.QUEUED, JobStatus.STARTED, JobStatus.DEFERRED):
                        running += 1
                    self.redis_conn.hset(self.name, trial_name, job_status)
            if running == 0:
                logging.info("Experiment %s: No more running jobs", self.name)
                break
            time.sleep(poll_interval)

    def get_trial_status(self):
        return self.redis_conn.hgetall(self.name)


class RQExperimentJob(Job):
    """"RQ Job Subclass with hacks for dynamic module injection"""

    def _unpickle_data(self):
        """
        Override that parses our "fully qualified" function name, adding the
        functions module to sys.path.
        :return:
        """
        super(RQExperimentJob, self)._unpickle_data()
        self._submitted_func_name = self._func_name
        module_path, func_name = self._func_name.split("|")
        module_dir = os.path.dirname(module_path)
        sys.path.insert(0, module_dir)
        self._module_dir = module_dir
        self._func_name = f"{os.path.basename(module_path)}.{func_name}"

        # Always reload module (in case it was previously imported and changes have been made)
        module_name, attribute = self._func_name.rsplit(".", 1)
        loaded_module = importlib.import_module(module_name)
        importlib.reload(loaded_module)

    def perform(self):
        """
        Override that cleans up sys.path and unloads our runtime-imported
        module after running the requested function.
        :return:
        """
        try:
            result = super(RQExperimentJob, self).perform()
            return result
        finally:
            # Cleanup modules
            _ = sys.path.remove(self._module_dir)
            del self._module_dir
            sys.stdout.flush()
