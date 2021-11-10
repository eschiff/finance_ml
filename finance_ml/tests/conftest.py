import pandas as pd
import pytest
from typing import List

from finance_ml.utils.constants import QuarterlyColumns as QC, TARGET_COLUMN, Q_DELTA_PREFIX


class PriceIter:
    def __init__(self, start_price: float, growth_rate: float = .02):
        self._price = start_price
        self._growth_rate = growth_rate

    def __iter__(self):
        return self

    def next(self):
        output = self._price
        self._price *= (1 + self._growth_rate)
        return round(output, 2)


def build_ticker_data(ticker='A', quarter=1, year=2021, price_avg=100.0, market_index=False,
                      add_prediction_column=False, predicted_growth_rate=.02,
                      prev_quarter_growth_rate=.02):
    d = {
        QC.TICKER_SYMBOL: ticker,
        QC.QUARTER: quarter,
        QC.YEAR: year,
        QC.PRICE_AVG: price_avg,
        QC.PRICE_HI: price_avg * 1.1,
        QC.PRICE_LO: price_avg * 0.9,
    }

    if add_prediction_column:
        d[f"{Q_DELTA_PREFIX}{QC.PRICE_AVG}"] = prev_quarter_growth_rate
        d[TARGET_COLUMN] = price_avg * (1 + predicted_growth_rate)

    if not market_index:
        # adding a smattering of fields
        d[QC.SECTOR] = 'Technology'
        d[QC.EARNINGS] = 1000000
        d[QC.REVENUE] = 1000000
        d[QC.DIVIDENDS] = 0
        d[QC.CASH] = 1000000
        d[QC.ASSETS] = 1000000

    return d


@pytest.fixture
def tickers() -> List[str]:
    return ['A', 'AAPL']


@pytest.fixture
def growth_rate() -> float:
    return 0.02


@pytest.fixture
def quarterly_df_with_predictions(tickers, growth_rate) -> pd.DataFrame:
    data = []

    for ticker in tickers:
        price_iter = PriceIter(100.0, growth_rate=growth_rate)

        data += [build_ticker_data(ticker, q, y, price_iter.next(), add_prediction_column=True)
                 for y in (2020, 2021)
                 for q in range(1, 5)]

    df = pd.DataFrame(data)

    df = df.set_index([QC.TICKER_SYMBOL, QC.QUARTER, QC.YEAR])
    return df
