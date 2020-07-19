#! usr/bin/en python3

"""
  Project : xgboost reference
  Module : K-fold cross validation
  Description : perform cross validation on parameters indentified in model model selection.
  Cross validation is used to identify best set of hyperparameters for a selected set of
  model parameters. K-fold is good technice to do so.
"""
import xgboost as xgb

## Recording model parameters as a dict
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

## Cross validation model built using 3-folds using sgboost .cv()
cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=1)


## RMSE after the last round
print((cv_results["test-rmse-mean"]).tail(1))
