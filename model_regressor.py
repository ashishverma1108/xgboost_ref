#! usr/bin/en python3

"""
  Project : xgboost reference
  Module : Create an XGBoost regressor
  Description : Create and train a xgboost regressor, make the prediction,
  and calculate error
"""

import setup
import xgboost as xgb

## Setting up xgbosst regressor with basic/default parameter values
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

## Model training... use xg.train() instead of xg.XGBRegressor() and xg.fit()
xg_reg.fit(setup.X_train,setup.y_train)
## make prediction using newly trained model
preds = xg_reg.predict(setup.X_test)
## Calculate error based on new error
rmse = np.sqrt(mean_squared_error(setup.y_test, preds))
