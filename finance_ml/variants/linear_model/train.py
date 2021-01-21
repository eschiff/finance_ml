from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

from finance_ml.variants.linear_model.hyperparams import Hyperparams


def train_and_evaluate(hyperparams: Hyperparams, X_train, y_train, X_test, y_test):
    if hyperparams.MODEL is LinearRegression:
        model = LinearRegression(fit_intercept=True,
                                 normalize=True).fit(X_train, y_train)
    elif hyperparams.MODEL is LGBMRegressor:
        model = LGBMRegressor(boosting_type='gbdt',
                              num_leaves=hyperparams.NUM_LEAVES,
                              max_depth=hyperparams.MAX_DEPTH,
                              learning_rate=hyperparams.LEARNING_RATE,
                              n_estimators=hyperparams.N_ESTIMATORS,
                              random_state=hyperparams.RANDOM_SEED)

        model.fit(X=X_train, y=y_train)

        print(f"Feature Importances: {model._feature_importances}")

    for i, col in enumerate(X_train.columns):
        print(f'The coefficient for {col} is {model.coef_[i]}')
    print(f'The intercept for our model is {model.intercept_}')

    score = model.score(X_test, y_test)
    print('*' * 50)
    print(f'The score of our model is {score}')

    y_pred = model.predict(X_test)

    metrics_dict = {
        'Mean Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
        'Mean Squared Error': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'Absolute Percentage Error': abs(y_pred - y_test) * 100.
    }

    return model, metrics_dict
