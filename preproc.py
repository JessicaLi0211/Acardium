# module to preprocess data
# preprocess steps
# scale metric data using standard scaler
# cateogrical data leave as is but change datatype to obj for easy pooling with catboost

# import libs
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Preproc:
    def __init__(self, raw_data_file, metric_col, categorical_col, target_col):
        # read data
        self.raw_data = pd.read(raw_data_file)
        # using additional memory for feature
        self.feature_df = self.raw_data.copy()
        # target attribute
        self.label = self.raw_data[target_col]
        # drop target value
        self.feature_df = self.feature_df.drop(target_col, 1)
        # scale metric data
        self.scale_metric_data(metric_col)
        # label categorical data
        self.label_categorical_data(categorical_col)

    def scale_metric_data(self, metric_col):
        # standard scaler
        scaler = StandardScaler()
        self.feature_df[metric_col] = scaler.fit_transform(self.feature_df[metric_col])

    def label_categorical_data(self, categorical_col):
        # change dtype to object
        self.feature_df[categorical_col] = self.feature_df[categorical_col].astype(object)
