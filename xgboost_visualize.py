#! usr/bin/en python3

"""
  Project : xgboost reference
  Module : K-fold cross validation
  Description : perform cross validation on parameters indentified in model model selection.
  Cross validation is used to identify best set of hyperparameters for a selected set of
  model parameters. K-fold is good technice to do so.
"""

import xgboost as xgb
import cross_validation  # for parameters
import setup # for data matris
import matplotlib.pyplot as plt

## train an xgboost model for visualizing
xg_reg = xgb.train(params=params, dtrain=data_matrix, num_boost_round=10)

## leveraging xgb.plot_tree(). num_trees argument specified number of trees to visualized
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


## Another way to visualize your XGBoost models is to examine the importance of
## each feature column in the original dataset within the model. XGBoost has a
## plot_importance() function that allows you to do exactly this
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
