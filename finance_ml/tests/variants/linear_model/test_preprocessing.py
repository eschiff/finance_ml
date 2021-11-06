import pandas as pd
from pandas.util.testing import assert_almost_equal

from finance_ml.utils.constants import (
    QuarterlyColumns as QC, Q_DELTA_PREFIX, YOY_DELTA_PREFIX, VS_MKT_IDX
)
from finance_ml.variants.linear_model.preprocessing import (
    add_comparison_to_market_index, add_delta_columns
)


def test_add_delta_columns(yf_input_df):
    columns = [QC.PRICE_AVG, QC.EARNINGS]

    output = add_delta_columns(yf_input_df, columns=columns, num_quarters=1)
    output = add_delta_columns(output, columns=columns, num_quarters=4)

    assert len(yf_input_df) == len(output)

    for col in columns:
        for prefix in (Q_DELTA_PREFIX, YOY_DELTA_PREFIX):
            assert f"{prefix}{col}" in output.columns

    idx = pd.IndexSlice
    for t in ('A', 'AAPL'):
        for f in columns:
            # just checking Q4
            assert_almost_equal(output.loc[idx[t, 4, 2021]][f"{Q_DELTA_PREFIX}{f}"], (
                    output.loc[idx[t, 4, 2021]][f] - output.loc[idx[t, 3, 2021]][f]
            ) / output.loc[idx[t, 3, 2021]][f],
                                atol=0.01)

            assert_almost_equal(output.loc[idx[t, 4, 2021]][f"{YOY_DELTA_PREFIX}{f}"], (
                    output.loc[idx[t, 4, 2021]][f] - output.loc[idx[t, 4, 2020]][f]
            ) / output.loc[idx[t, 4, 2020]][f],
                                atol=0.01)


def test_add_comparison_to_market_index(yf_input_df, market_index_df, mkt_idx):
    output = add_comparison_to_market_index(yf_input_df,
                                            market_index_df=market_index_df,
                                            market_indices=[mkt_idx],
                                            columns=[QC.PRICE_AVG])

    assert len(yf_input_df) == len(output)
    assert f"{QC.PRICE_AVG}{VS_MKT_IDX}{mkt_idx}" in output.columns

    idx = pd.IndexSlice
    # picking a random quarter for each ticker
    assert output.loc[idx['A', 4, 2021]][f"{QC.PRICE_AVG}{VS_MKT_IDX}{mkt_idx}"] == (
            yf_input_df.loc[idx['A', 4, 2021]][QC.PRICE_AVG] /
            market_index_df.loc[idx[mkt_idx, 4, 2021]][QC.PRICE_AVG]
    )
    assert output.loc[idx['AAPL', 2, 2021]][f"{QC.PRICE_AVG}{VS_MKT_IDX}{mkt_idx}"] == (
            yf_input_df.loc[idx['AAPL', 2, 2021]][QC.PRICE_AVG] /
            market_index_df.loc[idx[mkt_idx, 2, 2021]][QC.PRICE_AVG]
    )
