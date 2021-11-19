import asyncio
import copy
from datetime import datetime, timedelta
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import List
import yfinance_ez as yf

from finance_ml.utils.constants import (
    TARGET_COLUMN, CATEGORICAL_COLUMNS, INDEX_COLUMNS, QuarterlyColumns as QC)
from finance_ml.utils.transforms import (
    NumericalScaler, CategoricalToDummy, QuarterFilter,
    OutlierExtractor, CategoricalToNumeric, Splitter, DummyTransform)
from finance_ml.utils import QuarterlyIndex

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
        self.feature_importance_dict = None

    def fit(self, df: pd.DataFrame, current_qtr: QuarterlyIndex = None):
        print(f"\nFitting data for: {current_qtr}")

        self._build_data_pipeline(self.hyperparams, df, current_qtr)
        X_transformed = self.data_pipeline.fit_transform(df)

        X_train, y_train, X_test, y_test = Splitter(
            test_size=self.hyperparams.TEST_SIZE).transform(X_transformed)

        print(f"Train Size: {X_train.shape}, Test Size: {X_test.shape}")

        self.model, self.result_dict = train_and_evaluate(self.hyperparams,
                                                          X_train,
                                                          y_train,
                                                          X_test,
                                                          y_test)

        self.feature_importance_dict = {
            feature: importance
            for feature, importance
            in sorted(zip(X_train.columns, self.model.feature_importances_),
                      key=lambda tup: tup[1])}

    async def predict(self,
                      df: pd.DataFrame,
                      adjust_for_current_price: bool = True) -> pd.DataFrame:
        """
        Predict on data (filtering by dates if provided)

        Returns dataframe of predictions
        """
        assert (self.model is not None, "Model is not yet trained!")
        current_price_avg = None

        # Run data pipeline on df, but remove date filter
        data_pipeline = copy.deepcopy(self.data_pipeline)
        data_pipeline.steps[0] = ('date_filter', DummyTransform())
        df_transformed = data_pipeline.transform(df)

        if QC.PRICE_AVG in df_transformed.columns:
            current_price_avg = df_transformed[QC.PRICE_AVG]
            df_transformed.drop(columns=[QC.PRICE_AVG], inplace=True)

        predicted = self.model.predict(df_transformed)

        # Grab random row from df and use its index columns
        predict_df = df_transformed[FEATURE_COLUMNS[0]].reset_index()
        predict_df[TARGET_COLUMN] = predicted
        predict_df = predict_df.set_index(INDEX_COLUMNS).drop(columns=[FEATURE_COLUMNS[0]])

        if adjust_for_current_price:
            assert current_price_avg is not None, \
                "Unable to adjust for current price if AvgPrice is not in prediction DataFrame"

            ticker_symbols = predict_df.index.levels[0]
            avg_price_used_for_prediction = current_price_avg

            tickers_w_history = await self.get_history_multiple_tickers(
                ticker_symbols,
                start=datetime.now() - timedelta(days=5))

            most_recent_prices = pd.DataFrame({
                QC.TICKER_SYMBOL: [t.ticker for t in tickers_w_history],
                'CurrentPrice': [t.history.get('Close', [None])[-1] or
                                 avg_price_used_for_prediction[t.ticker] for t in tickers_w_history]
            })
            most_recent_prices = most_recent_prices.set_index([QC.TICKER_SYMBOL])

            predict_df[TARGET_COLUMN] -= \
                (most_recent_prices['CurrentPrice'] - avg_price_used_for_prediction
                 ) / most_recent_prices['CurrentPrice']

        return predict_df

    async def get_history_multiple_tickers(self,
                                           ticker_symbols: List[str],
                                           **kwargs) -> List[yf.Ticker]:
        tickers = [yf.Ticker(ticker_symbol) for ticker_symbol in ticker_symbols]

        await asyncio.gather(*[ticker.get_history_async(**kwargs) for ticker in tickers])

        return tickers

    def _build_data_pipeline(self,
                             hyperparams: Hyperparams,
                             df: pd.DataFrame,
                             current_quarter: QuarterlyIndex = None):
        if not current_quarter:
            current_quarter = QuarterlyIndex.from_date(datetime.now().date())

        non_categorical_columns = list(set(df.columns).difference({*CATEGORICAL_COLUMNS}))

        numerical_scaler = NumericalScaler(columns=non_categorical_columns)
        outlier_extractor = OutlierExtractor(columns=non_categorical_columns)
        categorical_to_dummy = CategoricalToDummy(CATEGORICAL_COLUMNS)
        categorical_to_numeric = CategoricalToNumeric(CATEGORICAL_COLUMNS)
        date_filter = QuarterFilter(
            start_qtr=current_quarter.time_travel(-hyperparams.NUM_QUARTERS_FOR_TRAINING),
            end_qtr=current_quarter)

        self.data_pipeline = Pipeline(steps=[('filter_dates', date_filter)])

        if hyperparams.EXTRACT_OUTLIERS:
            self.data_pipeline.steps.append(('extract_outliers', outlier_extractor))

        if hyperparams.ONE_HOT_ENCODE:
            self.data_pipeline.steps.append(('one_hot_encode', categorical_to_dummy))
        elif hyperparams.NUMERIC_ENCODE_CATEGORIES:
            self.data_pipeline.steps.append(('cat_to_numeric', categorical_to_numeric))

        if hyperparams.SCALE_NUMERICS:
            self.data_pipeline.steps.append(('numeric_scaler', numerical_scaler))
