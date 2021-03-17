import asyncio
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle
import yfinance_ez as yf
from typing import List

from finance_ml.utils.constants import (TARGET_COLUMN, QuarterlyColumns as QC)
from finance_ml.utils.quarterly_index import QuarterlyIndex

from finance_ml.variants.linear_model.config import FEATURE_COLUMNS
from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'models')


async def main(hyperparams: Hyperparams):
    df = preprocess_data(hyperparams)

    columns_to_drop = list(set(df.columns).difference({*FEATURE_COLUMNS}))
    df.drop(columns=columns_to_drop, inplace=True)
    df[TARGET_COLUMN] = df.apply(_get_target_col_prediction, axis=1,
                                 df=df, hyperparams=hyperparams)

    # Dividend Per Share was used in getting Predicted Appreciation, but is no longer needed
    df.drop(columns=[QC.DIVIDEND_PER_SHARE], inplace=True)

    # Get dataframe of rows to make predictions on (most recent rows)
    prediction_candidate_df = df[df[TARGET_COLUMN].isnull()].drop(columns=[TARGET_COLUMN])
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    metamodel = FinanceMLMetamodel(hyperparams)

    metamodel.fit(df)

    y_pred, avg_price_used_for_prediction = metamodel.predict(prediction_candidate_df)

    if hyperparams.ADJUST_FOR_CURRENT_PRICE:
        ticker_symbols = y_pred.index.levels[0]

        tickers_w_history = await get_history_multiple_tickers(
            ticker_symbols,
            start=datetime.now() - timedelta(days=5))

        most_recent_prices = pd.DataFrame({
            QC.TICKER_SYMBOL: [t.ticker for t in tickers_w_history],
            'CurrentPrice': [t.history.get('Close', [None])[-1] or
                             avg_price_used_for_prediction[t.ticker] for t in tickers_w_history]
        })
        most_recent_prices = most_recent_prices.set_index([QC.TICKER_SYMBOL])

        appreciation_since_model_ran = (most_recent_prices['CurrentPrice']
                                        - avg_price_used_for_prediction
                                        ) / most_recent_prices['CurrentPrice']

        y_pred[TARGET_COLUMN] -= appreciation_since_model_ran

    n_largest = y_pred.astype('float').nlargest(
        hyperparams.N_STOCKS_TO_BUY, columns=[TARGET_COLUMN])

    print(f"Predicting Top {len(n_largest)} stocks to purchase for"
          f" {hyperparams.N_QUARTERS_OUT_TO_PREDICT} Quarters in the future: \n{n_largest}")

    model_name = f'linear-model-{datetime.now().date()}.pickle'
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(metamodel, file=f)


def _get_target_col_prediction(row: pd.Series, df: pd.DataFrame, hyperparams: Hyperparams):
    try:
        current_price = row[QC.PRICE_AVG]
        dividends = 0

        if hyperparams.INCLUDE_DIVIDENDS_IN_PREDICTED_PRICE:
            for i in range(hyperparams.N_QUARTERS_OUT_TO_PREDICT):
                target_prediction_index = QuarterlyIndex(*row.name).time_travel(i).to_tuple()

                dividends += df.loc[target_prediction_index][QC.DIVIDEND_PER_SHARE] or 0

        target_prediction_index = QuarterlyIndex(*row.name).time_travel(
            hyperparams.N_QUARTERS_OUT_TO_PREDICT).to_tuple()

        prediction_data = df.loc[target_prediction_index]

        return prediction_data[f'{hyperparams.PREDICTION_TARGET_PREFIX}{QC.PRICE_AVG}'] + (
                dividends / current_price)
    except:
        return None


async def get_history_multiple_tickers(ticker_symbols: List[str],
                                       **kwargs) -> List[yf.Ticker]:
    tickers = [yf.Ticker(ticker_symbol) for ticker_symbol in ticker_symbols]

    await asyncio.gather(*[ticker.get_history_async(**kwargs) for ticker in tickers])

    return tickers
