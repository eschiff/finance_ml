from datetime import datetime, timedelta
import pandas as pd
from sklearn.pipeline import Pipeline

from finance_ml.utils.constants import (
    TARGET_COLUMN, QuarterlyColumns, CATEGORICAL_COLUMNS)
from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.transforms import (
    NumericalScaler, CategoricalToDummy, ColumnFilter, QuarterFilter,
    OutlierExtractor, CategoricalToNumeric, Splitter)

from finance_ml.variants.linear_model.config import FEATURE_COLUMNS
from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.train import train_and_evaluate


def main(hyperparams: Hyperparams):
    df = preprocess_data(hyperparams)

    columns_to_drop = list(set(df.columns).difference({*FEATURE_COLUMNS}))
    print(f"Dropping columns not in features: {columns_to_drop}")
    df.drop(columns=columns_to_drop, inplace=True)

    df[TARGET_COLUMN] = df.apply(_get_target_col_prediction, axis=1,
                                 df=df, hyperparams=hyperparams)

    # Get dataframe of rows to make predictions on (most recent rows)
    prediction_candidate_df = df[df[TARGET_COLUMN].isnull()]

    numerical_scaler = NumericalScaler(
        [col for col in df.columns if col not in CATEGORICAL_COLUMNS])
    categorical_to_dummy = CategoricalToDummy(CATEGORICAL_COLUMNS)
    categorical_to_numeric = CategoricalToNumeric(CATEGORICAL_COLUMNS)
    date_filter = QuarterFilter(
        start_date=hyperparams.START_DATE,
        end_date=hyperparams.END_DATE or datetime.now() - timedelta(days=90))

    data_pipeline = Pipeline(steps=[
        ('filter_dates', date_filter)
    ])

    if hyperparams.EXTRACT_OUTLIERS:
        data_pipeline.steps.append(('extract_outliers', OutlierExtractor(feature_columns)))

    if hyperparams.ONE_HOT_ENCODE:
        data_pipeline.steps.append(('one_hot_encode', categorical_to_dummy))
    elif hyperparams.NUMERIC_ENCODE_CATEGORIES:
        data_pipeline.steps.append(('cat_to_numeric', categorical_to_numeric))

    if hyperparams.SCALE_NUMERICS:
        data_pipeline.steps.append(('numeric_scaler', numerical_scaler))

    X_transformed = data_pipeline.fit_transform(df)
    X_train, y_train, X_test, y_test = Splitter().transform(X_transformed)

    model, result_dict = train_and_evaluate(X_train, y_train, X_test, y_test)

    print(result_dict)

    return model


def _get_target_col_prediction(row: pd.Series, df: pd.DataFrame, hyperparams: Hyperparams):
    try:
        target_prediction_index = QuarterlyIndex(*row.name).time_travel(
            hyperparams.N_QUARTERS_OUT_TO_PREDICT).to_tuple()

        prediction_data = df.loc[target_prediction_index]
        return prediction_data[
            f'{hyperparams.PREDICTION_TARGET_PREFIX}{QuarterlyColumns.PRICE_AVG}']
    except:
        return None
