# helper functions to use the hyperop framework for parameter tuning
# using objective function defined in hyperop (auc in this case)

import hyperopt
from hyperop import RevenueClassifierObjective
import catboost as cb
import numpy as np


# helper function for hyperparameter tuning
def hyperparam_tuning(dataset, const_params, max_eval, k_fold,tuning_metric):
    # optimize four parameters
    # details see hyperop.py
    parameter_space = {
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.01, 1.0),
        'depth': hyperopt.hp.randint('depth', 10),
        'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 1, 10),
        'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 1, 10)}

    # feeding objective function to hyperop as AUC
    objective = RevenueClassifierObjective(dataset=dataset, const_params=const_params, k_fold=k_fold,tuning_metric=tuning_metric)
    trials = hyperopt.Trials()
    # find the best combination of hyperparameters
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_eval,
        rstate=np.random.RandomState(seed=42))
    return best


# helper function to find the best model with cv with training data
def train_best_model(X, y, const_params, max_evals, k_fold, tuning_metric, use_default=False):
    # convert pandas.DataFrame to catboost.Pool
    # categorical feature is marked by the dtype = object in preproc
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes == object)[0])

    if use_default:
        # default parameters for catboost for the model (learning rate, depth, l2_leaf_reg)
        # scale_pos_weight = sum(neg)/sum(pos) -- imbalanced data
        best = {
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'scale_pos_weight': 5}
    else:
        best = hyperparam_tuning(dataset=dataset, const_params=const_params, max_eval=max_evals, k_fold=k_fold, tuning_metric= tuning_metric)

    # merge tuned hyperparameters with the predefined hyperparameters
    hyper_params = best.copy()
    hyper_params.update(const_params)

    # drop `use_best_model` because we are going to use entire dataset for training of the final model
    hyper_params.pop('use_best_model', None)

    # train final model using all training dta
    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)

    return model, hyper_params
