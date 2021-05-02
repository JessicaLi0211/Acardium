# module to preprocess data
# preprocess steps
# scale metric data using standard scaler
# change datatype of categorical features to object for easy pooling with catboost
# train/test split (stratified)

# import libs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class Preproc:
    def __init__(self, raw_data_file, metric_col, categorical_col, target_col, test_perc):
        # read data
        self.raw_data = pd.read_csv(raw_data_file)
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
        # train test split
        self.train_test_split(test_perc)

    def scale_metric_data(self, metric_col):
        # standard scaler
        print('Scale metric data using standard scaler:')
        scaler = StandardScaler()
        self.feature_df[metric_col] = scaler.fit_transform(self.feature_df[metric_col])

    def label_categorical_data(self, categorical_col):
        # change dtype to object
        print('Label categorical label for label encoding')
        self.feature_df[categorical_col] = self.feature_df[categorical_col].astype(str)

    def train_test_split(self, test_perc):
        print('Train test split:')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.feature_df, self.label,
                                                                                test_size=test_perc, random_state=42)
