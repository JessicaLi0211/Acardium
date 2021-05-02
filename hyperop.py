import sys
import hyperopt
import catboost as cb
import numpy as np


class RevenueClassifierObjective(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0

    def convert_catboost_params(self, hyper_params):
        # get params for tuning for catboost classifier
        # learning rate: used for reducing gradient decent stpe
        # depth: max_depth of the tree
        # l2_leaf_reg: coefficient at the L2 regularization term of the cost function
        # auto_class_weights: battles imbalanced dataset
        return {
            'learning_rate': hyper_params['learning_rate'],
            'depth': hyper_params['depth'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg'],
            'auto_class_weights': hyper_params['auto_class_weights']}

    # objective function for hyperopt to optimize
    # here optimize log loss as we have a binary classification case
    def __call__(self, hyper_params):
        # join tunable and constant hyper-params
        params = self.convert_catboost_params(hyper_params)
        params.update(self._const_params)
        sys.stdout.flush()

        # k-fold cross-validation to avoid over-fitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=42,
            verbose=False)

        # AUC per fold
        max_mean_auc = np.max(scores['test-AUC-mean'])
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)

        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}
