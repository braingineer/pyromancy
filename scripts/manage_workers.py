# coding=utf-8
import argparse
import datetime
import logging
import os
import subprocess
import time

import psutil
import redis
from rq import Worker
from rq.exceptions import NoSuchJobError

logging.getLogger().setLevel(logging.INFO)

DEFAULT_LOGDIR = '/tmp'
DEFAULT_WORKDIR = './'
DEFAULT_WORKER_CLASS = 'pyromancy.experiment.RQExperimentWorker'
DEFAULT_JOB_CLASS = 'pyromancy.experiment.RQExperimentJob'


def stop_worker(worker):
    """Stop the given RQ worker and mark it as dead."""
    try:
        job = worker.get_current_job()
        if job is not None:
            job.ended_at = datetime.datetime.utcnow()
            worker.failed_queue.quarantine(job, exc_info=('Worker killed', 'Moving job to Failed Queue'))
            logging.info('Stopped job %s on worker %s', job.id, worker.name)
    except NoSuchJobError:
        pass
    worker.register_death()
    logging.info('Stopped worker %s', worker.name)


def find_worker_process(worker_name):
    """Find any RQ worker with the given name.
    Using this to bypass complications with shell=True when calling Popen()
    """
    for process in psutil.process_iter():
        try:
            cmdline = ' '.join(process.cmdline())
            if 'rq' in cmdline and worker_name in cmdline:
                return process
        except (psutil.ZombieProcess, psutil.NoSuchProcess):
            continue


def main(args, redis_connection):
    command = args.command

    workers_by_name = {worker.name: worker for worker in Worker.all(redis_connection)}
    for gpu in args.gpu_ids:
        for n in args.n_procs:
            worker_name = 'worker_{}_{}'.format(gpu, n)
            log_file = '{}.log'.format(os.path.join(args.logdir, worker_name))

            proc = find_worker_process(worker_name)
            worker = workers_by_name.get(worker_name)
            if proc and worker:
                # Worker registered and process running
                if command == 'start':
                    logging.debug('Worker %s already running as pid %s', worker_name, proc.pid)
                    continue
                elif command in ('stop', 'restart'):
                    logging.info('Terminating worker %s (pid %s)', worker_name, proc.pid)
                    proc.kill()
                    stop_worker(worker)
            elif proc is None and worker:
                # Zombie worker: No process running
                logging.error('Worker %s is not associated with a running process, cleaning up', worker_name)
                stop_worker(worker)
            elif proc and not worker:
                # Zombie process: No worker registered
                logging.error('Process %s is not associated with a Worker, stopping', worker_name)
                proc.kill()
            elif proc is None and worker is None:
                # No workers / processes running
                pass

            if command in ('start', 'restart'):
                worker_command = f'nohup rq worker -w {args.worker_class} -j {args.job_class} -n {worker_name} >> {log_file} 2>&1 &'
                logging.debug('Starting worker %s with command %s', worker_name, worker_command)
                _ = subprocess.Popen(worker_command, cwd=args.workdir, shell=True, preexec_fn=os.setpgrp)
                time.sleep(1.0)
                proc = find_worker_process(worker_name)
                if proc:
                    logging.info('Started worker %s (pid %s)', worker_name, proc.pid)
                else:
                    logging.error('Failed to start worker %s, see %s', worker_name, log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage RQ workers')
    parser.add_argument('--gpu-ids', '-g', nargs='+', type=int,
                        help='Integer ids of GPUs to manage')
    parser.add_argument('--n-procs', '-n', nargs='+', type=int,
                        help='Number of workers per GPU')
    parser.add_argument('command', type=str, choices=('start', 'stop', 'restart', 'monitor'),
                        help='Command to pass workers')
    parser.add_argument('--redis-host', type=str, default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--worker-class', type=str, default=DEFAULT_WORKER_CLASS)
    parser.add_argument('--job-class', type=str, default=DEFAULT_JOB_CLASS)
    parser.add_argument('--workdir', type=str, default=DEFAULT_WORKDIR)
    parser.add_argument('--logdir', type=str, default=DEFAULT_LOGDIR)
    args = parser.parse_args()

    redis_connection = redis.Redis(args.redis_host, args.redis_port)
    if args.command == 'monitor':
        args.command = 'start'
        while True:
            main(args, redis_connection)
            time.sleep(5.0)
    else:
        main(args, redis_connection)
