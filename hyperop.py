# hyper-parameter tuning under hyperop framework
# under this framework, we need to provide an objective to optimize
# as the use case is binary classification with imbalanced class, we will optimize AUC
# use cross validation, we will try to avoid over-fitting
# hyper-parameters to be tune:
# learning rate
# max tree depth
# l2 regularization strength
# rebalanced weight for positive class

# import libs
import sys
import catboost as cb
import numpy as np
import hyperopt


class RevenueClassifierObjective(object):
    def __init__(self, dataset, const_params, k_fold,tuning_metric):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._k_fold = k_fold
        self._evaluated_count = 0
        self._tuning_metric = tuning_metric

    def convert_catboost_params(self, hyper_params):
        # get params for tuning for catboost classifier learning rate: used for reducing gradient decent step size
        # depth: max_depth of the tree
        # l2_leaf_reg: coefficient at the L2 regularization term of the cost function
        # scale_pos_weight: The weight for class 1 in binary classification. The value is used as a multiplier for
        # the weights of objects from class 1.
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg'],
            'scale_pos_weight': hyper_params['scale_pos_weight']}

    # objective function for hyperopt to optimize
    def __call__(self, hyper_params):
        # join tunable and constant hyper-params
        params = self.convert_catboost_params(hyper_params)
        params.update(self._const_params)
        sys.stdout.flush()
        # options to optimize different metrics based on business use case
        metrics = {'AUC':'test-auc-mean',
                   'F1':'test-F1:use_weights=true-mean',
                   'BalancedAccuracy':'test-BalancedAccuracy:use_weights=true-mean'}
        tuning_metrics = metrics[self._tuning_metric]
        # k-fold cross-validation to avoid over-fitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._k_fold,
            partition_random_seed=42,
            verbose=False)

        # AUC per fold
        max_mean_score = np.max(scores[tuning_metrics])
        print('evaluated score={}'.format(max_mean_score), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)

        # minimize the negative auc --> maximize actual auc
        return {'loss': -max_mean_score, 'status': hyperopt.STATUS_OK}
