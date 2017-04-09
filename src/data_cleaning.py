import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataCleaning(object):

    def __init__(self, instance='training'):
        if instance == 'training':
            self.df = pd.read_csv('../data/churn_train.csv')
        else:
            self.df = pd.read_csv('../data/churn_test.csv')

    def make_target_variable(self):
        self.df['last_trip_date'] = pd.to_datetime(self.df['last_trip_date'])
        self.df['Churn'] = (self.df.last_trip_date < '2014-06-01').astype(int)

    def dummify(self, columns):
        dummies = pd.get_dummies(self.df[columns], prefix = columns)
        self.df = self.df.drop(columns, axis = 1)
        # return pd.concat([df,dummies], axis = 1)
        self.df = pd.concat([self.df,dummies], axis = 1)

    def drop_date_columns(self):
        self.df.drop('last_trip_date', axis=1, inplace=True)
        self.df.drop('signup_date', axis=1, inplace=True)

    def get_column_names(self):
        return list(self.df.columns.values)

    def cut_outliers(self, col):
       std = self.df[col].std()
       t_min = self.df[col].mean() - 3*std
       t_max = self.df[col].mean() + 3*std
       self.df = self.df[(self.df[col] >= t_min) & (self.df[col] <= t_max)]

    def total_distance(self):
        self.df['total_distance'] = self.df['avg_dist'] * self.df['trips_in_first_30_days']

    def drop_na(self):
       self.df = self.df.dropna(axis=0, how='any')

    def drop_avg_dist_and_trips(self):
        self.df = self.df.drop(['avg_dist', 'trips_in_first_30_days'], axis=1)

    def make_log_no_trips(self):
        #import ipdb; ipdb.set_trace()
        self.df['log_trips'] = self.df[(self.df['trips_in_first_30_days'] != 0)].trips_in_first_30_days.apply(np.log)
        self.df['log_trips'] = self.df['log_trips'].apply(lambda x: 0 if np.isnan(x) else x)
        #self.df["trips_in_first_30_days"] = np.log(self.df["trips_in_first_30_days"])+1

    def drop_low_log_columns(self):
        self.df = self.df.drop(['phone_iPhone', 'city_Astapor'], axis=1)

    def clean(self):
        self.make_target_variable()
        self.drop_date_columns()

        # making total distance
        # self.total_distance()
        # self.drop_avg_dist_and_trips()

        #self.make_log_no_trips()

        for column in ['avg_dist', 'trips_in_first_30_days', 'surge_pct']:
           self.cut_outliers(column)

        for column in ['city', 'phone']:
            self.dummify(column)
        #self.drop_low_log_columns()
        self.drop_na()
        y = self.df.pop('Churn')

        #drop low value coeffs


        ss = StandardScaler()

        X = self.df.values
        X = ss.fit_transform(X)
        return X, y
