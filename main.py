# main script to run model for buying classification
from model import train_best_model
from preproc import Preproc
from eval import eval_model
import catboost as cb
import numpy as np
import sys

# load configuration
# make it True if your want to use GPU for training
have_gpu = False

# params for preprocessing from EDA
raw_data_file = 'data/online_shoppers_intention.csv'
metric_col = ['Administrative_Duration',
              'Informational_Duration',
              'ProductRelated_Duration',
              'BounceRates',
              'ExitRates',
              'PageValues']
categorical_col = ['Administrative',
                   'Informational',
                   'ProductRelated',
                   'Month',
                   'OperatingSystems',
                   'Browser',
                   'Region',
                   'TrafficType',
                   'VisitorType',
                   'SpecialDay',
                   'Weekend']
target_col = 'Revenue'
test_perc = .2

# hyperparameter tuning
# skip hyperparameter tuning and just use default parameters
use_predefined_params = False
# number of iterations of hyper-parameter search
hyperopt_iterations = 30
# constant params for the catboost tree
const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'custom_metric': ['F1', 'BalancedAccuracy'],
    'iterations': 100,
    'random_seed': 42})
# tuning metric
tuning_metric = 'BalancedAccuracy'
# cv fold
k_fold = 5

# save model directory
save_model_dir = 'model.json'


# main function
def classification_model(raw_data_file,
                         metric_col,
                         categorical_col,
                         target_col,
                         test_perc,
                         hyperopt_iterations,
                         const_params,
                         use_predefined_params,
                         k_fold,
                         tuning_metric
                         ):
    # preprocess data
    print('preprocess data:')
    data_obj = Preproc(raw_data_file, metric_col, categorical_col, target_col, test_perc)

    # hyperparameter tuning train with best params
    print('hyperparams tuning and model fitting:')
    model, params = train_best_model(
        data_obj.X_train, data_obj.y_train,
        const_params,
        hyperopt_iterations,
        k_fold,
        tuning_metric,
        use_predefined_params)
    print('best params are {}'.format(params), file=sys.stdout)

    # evaluate model
    auc = eval_model(data_obj.X_test, data_obj.y_test, model)

    # save model
    model.save_model(
        save_model_dir,
        format="json",
        pool=cb.Pool(data_obj.X_train, data_obj.y_train, cat_features=np.where(data_obj.X_train.dtypes == object)[0]))


#  run the script.
if __name__ == '__main__':
    classification_model(raw_data_file,
                         metric_col,
                         categorical_col,
                         target_col,
                         test_perc,
                         hyperopt_iterations,
                         const_params,
                         use_predefined_params,
                         k_fold,
                         tuning_metric)
