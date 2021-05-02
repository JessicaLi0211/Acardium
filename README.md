# online-transaction-classifier
## Code challenge

Data: 
- This data set contains transactions occurring in an online store (E-commerce). 
- Out of the 12,330 customer samples in the dataset, 84.5% (10,422) were negative class samples (i.e. customers who did not end up buying the product), and the rest (1908) were positive class samples (i.e. customers who ended up buying). 
- The dataset consists of 10 numerical and 8 categorical attributes. 
- The 'Revenue' attribute can be used as the class label. 

Goal: 
- Create a ML model that can predict buying behavior per session

## Getting started

Install packages in the requirments.txt 
```bash
pip install -r requirements.txt
```
Run the experiment
```bash
python main.py
```

Performance
```bash
              precision    recall  f1-score   support

       False       0.97      0.84      0.90      2055
        True       0.51      0.87      0.65       411

    accuracy                           0.84      2466
   macro avg       0.74      0.85      0.77      2466
weighted avg       0.89      0.84      0.86      2466

auc:  0.9275205569467384
```

## Model overview and rationale

After EDA in EDA.ipynb, a few observations guide the selection of model for this use case:
1. highly imbalanced target attribute distribution 
- consider resampling, SMOTE, or adjust class weights in the loss
2.  a few feature attributes are highly correlated 
- consider removing multicolinearity if linear model is used
3. there is no significant linear correlation amonst feature and target attribute 
- consider using non-linear models
4. though most attributes have numerical values, not all have cardinality 
- consider treating those as categorical features 
5. some categorical features take a wide range of noncrdinal values 
- consider using target encoding instead of one hot encoding

Model selection
[Catboost](https://catboost.ai) with ordered boosting, catboost encoding for cateogrical variables, adjusted class weights, together with hyperparameter tuning under [hyperop framework](https://github.com/hyperopt/hyperopt/blob/master/README.md) can effectively accomondate all points above

## Important adjustable parameters in the model

Addtional information for parameters can be found in [CatBoost documentation](https://catboost.ai/docs/concepts/parameter-tuning.html)

Constant hyper parameters:
- device_type: GPU or CPU
- loss_function: function for gradient descent (log loss or cross entropy loss)
- eval_metric : AUC, F1, accuracy, etc.
- iterations: number of iters for fitting
- k_fold: number of folds for cross validation in hyperparameter tuning

Tunable hyper parameters:
- learning rate: used for reducing gradient decent step size
- depth: max_depth of the tree
- l2_leaf_reg: coefficient at the L2 regularization term of the cost function
- scale_pos_weight: The weight for class 1 in binary classification. The value is used as a multiplier for the weights of objects from class 1

## Future work for improvement

- Hyperparameter tuning: 
  
  depending on the business value, the hyperparameters of the model can be optimized against different metrics such as precison of positive class, or combination of f1 of positive class and overall AUC, etc.
- Resampling:
  
  though class weight adjustment is effective, we can also synthesize positive class data using algorithms such as SMOTE, GAN etc. 
- Ensembling:
  
  it is possible to gain incremental performance with an ensemble of different models, the key is the trade off between performance gain and space/time needed for execution
- Additional featues:
  
  timestamp per session
  user transaction history with more details such as frequency, recency, monetary information
  sentiment of product mentioned (can be from google trend)
  and othe relevant information for this specific use case
    
