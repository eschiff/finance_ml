from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline

from finance_ml.utils.constants import (
    TARGET_COLUMN, QuarterlyColumns, CATEGORICAL_COLUMNS, INDEX_COLUMNS)
from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.transforms import (
    IndexSwitchTransformer, NumericalScaler, CategoricalToDummy, ColumnFilter, DateFilter,
    OutlierExtractor, CategoricalToNumeric, Splitter)

from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.train import train_and_evaluate


def main(hyperparams: Hyperparams):
    df = preprocess_data(hyperparams)

    df.drop(columns=[QuarterlyColumns.PRICE_HI,
                     QuarterlyColumns.PRICE_LO,
                     QuarterlyColumns.PRICE_AVG,
                     QuarterlyColumns.PRICE_AT_END_OF_QUARTER,
                     QuarterlyColumns.AVG_RECOMMENDATION_SCORE,
                     QuarterlyColumns.SPLIT,
                     QuarterlyColumns.INVESTMENTS,
                     QuarterlyColumns.NET_BORROWINGS], inplace=True)

    feature_columns = list(set(df.columns).difference({TARGET_COLUMN, *CATEGORICAL_COLUMNS}))

    df[TARGET_COLUMN] = df.apply(_get_target_col_prediction, axis=1,
                                 df=df, hyperparams=hyperparams)

    # Get dataframe of rows to make predictions on (most recent rows)
    prediction_candidate_df = df[df[TARGET_COLUMN].isnull()]

    index_transformer = IndexSwitchTransformer(INDEX_COLUMNS)
    numerical_scaler = NumericalScaler(feature_columns)
    categorical_to_dummy = CategoricalToDummy(CATEGORICAL_COLUMNS)
    categorical_to_numeric = CategoricalToNumeric(CATEGORICAL_COLUMNS)
    date_filter = DateFilter(start_date=datetime(2000, 1, 1),
                             end_date=datetime(2020, 8, 1))

    data_pipeline = Pipeline(steps=[
        ('reset_index', index_transformer),
        ('filter_dates', date_filter),
        ('filter_columns', ColumnFilter(feature_columns + CATEGORICAL_COLUMNS + [TARGET_COLUMN]))
    ])

    if hyperparams.EXTRACT_OUTLIERS:
        data_pipeline.steps.append(('extract_outliers', OutlierExtractor(feature_columns)))

    if hyperparams.ONE_HOT_ENCODE:
        data_pipeline.steps.append(('one_hot_encode', categorical_to_dummy))
    else:
        data_pipeline.steps.append(('cat_to_numeric', categorical_to_numeric))

    data_pipeline.steps.extend([
        ('numeric_scaler', numerical_scaler),
        ('restore_index', index_transformer)
    ])

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
