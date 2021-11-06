import pandas as pd
import pytest

from finance_ml.utils.constants import QuarterlyColumns as QC


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


def build_ticker_data(ticker='A', quarter=1, year=2021, price_avg=100.0, market_index=False):
    d = {
        QC.TICKER_SYMBOL: ticker,
        QC.QUARTER: quarter,
        QC.YEAR: year,
        QC.PRICE_AVG: price_avg,
        QC.PRICE_HI: price_avg * 1.1,
        QC.PRICE_LO: price_avg * 0.9,
    }

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
def mkt_idx() -> str:
    return "DJI"


@pytest.fixture
def yf_input_df() -> pd.DataFrame:
    A_price_iter = PriceIter(100.0)
    AAPL_price_iter = PriceIter(200.0)

    df = pd.DataFrame(
        [build_ticker_data('A', q, y, A_price_iter.next())
         for q in range(1, 5) for y in (2020, 2021)] +
        [build_ticker_data('AAPL', q, y, AAPL_price_iter.next())
         for q in range(1, 5) for y in (2020, 2021)]
    )

    df = df.set_index([QC.TICKER_SYMBOL, QC.QUARTER, QC.YEAR])
    return df


@pytest.fixture
def market_index_df(mkt_idx) -> pd.DataFrame:
    initial_price = 100.0

    def get_price():
        nonlocal initial_price
        initial_price = initial_price + 2.0
        return initial_price

    DJI_data = [build_ticker_data(mkt_idx, q, y, get_price(), market_index=True)
                for q in range(1, 5)
                for y in (2019, 2020, 2021)]

    df = pd.DataFrame(DJI_data)
    df = df.set_index([QC.TICKER_SYMBOL, QC.QUARTER, QC.YEAR])
    return df
