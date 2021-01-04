from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = LinearRegression(fit_intercept=True,
                             normalize=True).fit(X_train, y_train)

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
