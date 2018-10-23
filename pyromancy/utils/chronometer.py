import sqlite3
import pandas as pd
import os


class Chronometer(object):
    """
    Interface with the Chronometer Database.
    """
    def __init__(self, chronodb_info, db_type='sqlite3'):
        """
        Args:
            chronodb_info (dict): the information for the chronometer database
                depending on the type of db, this will effectively change.
                For example, sqlite3 only needs a path.
            db_type (str): the type of db the chronometer has been recording to
                Currently supported: 'sqlite3'
        """

        if db_type == 'sqlite3':
            if not os.path.exists(chronodb_info['path']):
                raise Exception(f"Cannot find {chronodb_info['path']}")
            self.conn = sqlite3.connect(chronodb_info['path'])

        self.df = pd.read_sql_query("select * from timing_events", self.conn)
        self._experiment_set = None
        self._experiment_index = None
        self._subdf = None

    def get_experiments(self):
        if self._experiment_set is None:
            self._experiment_set = set(self.df.experiment_name.unique())
        return self._experiment_set

    def get_experiment_index(self):
        if self._experiment_index is None:
            self._experiment_index = dict(enumerate(sorted(self.get_experiments())))
        return self._experiment_index

    def experiment_subset(self, index=None, name=None):
        if index is not None and name is not None:
            raise Exception(f"passed index ({index}) and name ({name}); need only one")
        elif index is None and name is None:
            raise Exception("index and name are both None; please pass one")
        elif index is not None:
            name = self._experiment_index[index]

        if name not in self.get_experiments():
            raise Exception(f"Not a valid experiment name: {name}")

        return self.df[self.df.experiment_name==name]

    def set_experiment_subset(self, index=None, name=None):
        self._subdf = self.experiment_subset(index, name)

    def _mean_x(self, x):
        if self._subdf is None:
            raise Exception("Please set the experiment subset first!")
        subdf = self._subdf[self._subdf.event_name==x]
        return subdf.duration.mean()

    def mean_epoch(self):
        return self._mean_x('epoch')

    def mean_batch(self):
        return self._mean_x('batch')
