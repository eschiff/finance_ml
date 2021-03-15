import copy
from datetime import datetime, timedelta, date
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Tuple

from finance_ml.utils.constants import (
    TARGET_COLUMN, CATEGORICAL_COLUMNS, INDEX_COLUMNS, QuarterlyColumns)
from finance_ml.utils.transforms import (
    NumericalScaler, CategoricalToDummy, QuarterFilter,
    OutlierExtractor, CategoricalToNumeric, Splitter)

from finance_ml.variants.linear_model.config import FEATURE_COLUMNS
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.train import train_and_evaluate


class FinanceMLMetamodel:
    def __init__(self, hyperparams: Hyperparams):
        self.model = None
        self.hyperparams = hyperparams
        self.data_pipeline = None
        self.model = None
        self.result_dict = None

    def fit(self, df: pd.DataFrame):
        data_pipeline = self._build_data_pipeline(self.hyperparams, df)
        X_transformed = data_pipeline.fit_transform(df)

        X_train, y_train, X_test, y_test = Splitter(
            test_size=self.hyperparams.TEST_SIZE).transform(X_transformed)

        self.model, self.result_dict = train_and_evaluate(self.hyperparams,
                                                          X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test)

    def predict(self, df: pd.DataFrame, start_date: date = None,
                end_date: date = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Predict on data (filtering by dates if provided)

        Returns tuple of (predictions made, avg price used for prediction)
        """
        assert (self.model is not None, "Model is not yet trained!")

        start_date = start_date or datetime.now().date() - timedelta(days=90)
        end_date = end_date or datetime.now().date()
        date_filter = QuarterFilter(start_date=start_date, end_date=end_date)

        data_pipeline = copy.deepcopy(self.data_pipeline)
        data_pipeline.steps[0] = ('date_filter', date_filter)

        df_transformed = data_pipeline.transform(df)

        predicted = self.model.predict(df_transformed)

        # Grab random row from df and use its index columns
        predict_df = df_transformed[FEATURE_COLUMNS[0]].reset_index()
        predict_df[TARGET_COLUMN] = predicted
        predict_df = predict_df.set_index(INDEX_COLUMNS).drop(columns=[FEATURE_COLUMNS[0]])

        return predict_df, df_transformed[QuarterlyColumns.PRICE_AVG]

    def _build_data_pipeline(self, hyperparams: Hyperparams, df: pd.DataFrame):
        if self.data_pipeline:
            return self.data_pipeline

        non_categorical_columns = list(set(df.columns).difference({*CATEGORICAL_COLUMNS}))

        numerical_scaler = NumericalScaler(columns=non_categorical_columns)
        outlier_extractor = OutlierExtractor(columns=non_categorical_columns)
        categorical_to_dummy = CategoricalToDummy(CATEGORICAL_COLUMNS)
        categorical_to_numeric = CategoricalToNumeric(CATEGORICAL_COLUMNS)
        date_filter = QuarterFilter(
            start_date=hyperparams.START_DATE,
            end_date=hyperparams.END_DATE or datetime.now() - timedelta(days=90))

        self.data_pipeline = Pipeline(steps=[('filter_dates', date_filter)])

        if hyperparams.EXTRACT_OUTLIERS:
            self.data_pipeline.steps.append(('extract_outliers', outlier_extractor))

        if hyperparams.ONE_HOT_ENCODE:
            self.data_pipeline.steps.append(('one_hot_encode', categorical_to_dummy))
        elif hyperparams.NUMERIC_ENCODE_CATEGORIES:
            self.data_pipeline.steps.append(('cat_to_numeric', categorical_to_numeric))

        if hyperparams.SCALE_NUMERICS:
            self.data_pipeline.steps.append(('numeric_scaler', numerical_scaler))

        return self.data_pipeline
