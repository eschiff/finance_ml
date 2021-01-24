from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

from finance_ml.variants.linear_model.hyperparams import Hyperparams


def train_and_evaluate(hyperparams: Hyperparams, X_train, y_train, X_test, y_test):
    if hyperparams.MODEL is LinearRegression:
        model = LinearRegression(fit_intercept=True,
                                 normalize=True).fit(X_train, y_train)

        model.fit(X_train, y_train)

        for i, col in enumerate(X_train.columns):
            print(f'The coefficient for {col} is {model.coef_[i]}')
        print(f'The intercept for our model is {model.intercept_}')

    elif hyperparams.MODEL is XGBRegressor:
        booster = 'gbtree'  # 'dart'  #'gblinear' #'gbtree'

        model = XGBRegressor(seed=100,
                             n_estimators=100,
                             max_depth=3,
                             learning_rate=0.1,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             colsample_bylevel=1,
                             gamma=0,
                             booster=booster).fit(X_train, y_train)

        if booster == 'gblinear':
            for i, col in enumerate(X_train.columns):
                print(f'The coefficient for {col} is {model.coef_[i]}')
            print(f'The intercept for our model is {model.intercept_}')

    elif hyperparams.MODEL is LGBMRegressor:
        model = LGBMRegressor(boosting_type='gbdt',
                              num_leaves=hyperparams.NUM_LEAVES,
                              max_depth=hyperparams.MAX_DEPTH,
                              learning_rate=hyperparams.LEARNING_RATE,
                              n_estimators=hyperparams.N_ESTIMATORS,
                              random_state=hyperparams.RANDOM_SEED)

        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='l1',
                  early_stopping_rounds=hyperparams.EARLY_STOPPING_ROUNDS)

        feature_importance_dict = {
            feature: importance
            for feature, importance
            in sorted(zip(X_train.columns, model.feature_importances_),
                      key=lambda tup: tup[1])}

        print(f"Feature Importances: {feature_importance_dict}")

    y_pred = model.predict(X_test)

    metrics_dict = {
        'Mean Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
        'Mean Squared Error': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'Absolute Percentage Error': abs(y_pred - y_test) * 100.
    }

    return model, metrics_dict
