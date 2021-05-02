# evaluate model performance

import sklearn.metrics
import matplotlib.pylab as plt
import catboost as cb
import catboost.utils as cbu
import numpy as np


def eval_model(X, y, model):
    # confusion matrix
    test_pool = cb.Pool(X, cat_features=np.where(X.dtypes == object)[0])
    predictions = model.predict_proba(test_pool)
    predictions = [1 if ele[1] > 0.5 else 0 for ele in predictions]
    print(sklearn.metrics.classification_report(y, predictions))
    # auc
    plt.style.use('ggplot')
    dataset = cb.Pool(X, y, cat_features=np.where(X.dtypes == object)[0])
    fpr, tpr, _ = cbu.get_roc_curve(model, dataset, plot=True)
    auc = sklearn.metrics.auc(fpr, tpr)
    print('auc: ', auc)
    return auc
