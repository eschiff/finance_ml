import pandas as pd
import pytest

from finance_ml.utils.constants import QuarterlyColumns as QC
from finance_ml.tests.conftest import PriceIter, build_ticker_data


@pytest.fixture
def mkt_idx() -> str:
    return "DJI"


@pytest.fixture
def yf_input_df() -> pd.DataFrame:
    A_price_iter = PriceIter(100.0)
    AAPL_price_iter = PriceIter(200.0)

    df = pd.DataFrame(
        [build_ticker_data('A', q, y, A_price_iter.next())
         for y in (2020, 2021) for q in range(1, 5)] +
        [build_ticker_data('AAPL', q, y, AAPL_price_iter.next())
         for y in (2020, 2021) for q in range(1, 5)]
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
                for y in (2019, 2020, 2021)
                for q in range(1, 5)
                ]

    df = pd.DataFrame(DJI_data)
    df = df.set_index([QC.TICKER_SYMBOL, QC.QUARTER, QC.YEAR])
    return df
