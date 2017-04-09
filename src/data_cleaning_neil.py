import pandas as pd
import numpy as np

class DataCleaning(object):

    def __init__(self, instance='training'):
        if instance == 'training':
            self.df = pd.read_csv('../data/churn_train.csv')
        else:
            self.df = pd.read_csv('../data/churn_test.csv')

    # def make_target_variable(self):
    #     # convert last_trip_date to date_time
    #     self.df['last_trip_date'] = self.df['last_trip_date'].map(lambda x: pd.to_datetime(x))
    #     self.df['churn'] = (pd.to_datetime('2014-07-01') - self.df['last_trip_date'])
    #     self.df['churn'] = self.df['churn'].map(lambda x: 1 if x.days > 30 else 0)
    #     # Drop leaky column
    #     self.df.pop('last_trip_date')

    def make_target_variable(self):
        self.df['last_trip_date'] = pd.to_datetime(self.df['last_trip_date'])
        self.df['Churn'] = (self.df.last_trip_date < '2014-06-01').astype(int)


    def clean(self):
        self.make_target_variable()
        y = self.df.pop('Churn')
        X = self.df[['avg_dist', 'avg_surge']].values
        return X, y
