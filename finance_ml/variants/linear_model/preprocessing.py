import numpy as np
import pandas as pd
import json
import sqlite3
from typing import Tuple, List, Dict, Callable

from finance_ml.utils.constants import (
    QuarterlyColumns as QC, StockPupColumns as SPC, STOCKPUP_TABLE_NAME, QUARTERLY_DB_FILE_PATH,
    YF_QUARTERLY_TABLE_NAME, INDEX_COLUMNS, MISSING_SECTOR, MISSING_INDUSTRY,
    STOCK_GENERAL_INFO_CSV, FORMULAE, Q_DELTA_PREFIX, YOY_DELTA_PREFIX, NUMERIC_COLUMNS,
    COLUMNS_TO_COMPARE_TO_MARKET_INDICES, QUARTER, YEAR, VS_MKT_IDX, CATEGORICAL_COLUMNS,
    TICKER_SYMBOL
)
from finance_ml.utils.quarterly_index import QuarterlyIndex

from finance_ml.variants.linear_model.hyperparams import Hyperparams


def preprocess_data(hyperparams: Hyperparams) -> pd.DataFrame:
    quarterly_df, market_index_df = preprocess_quarterly_data(hyperparams)
    stockpup_df = preprocess_stockpup_data()
    stock_info_df = read_stock_info()

    # Filter out all data in stockpup_df that exists in quarterly_df (by index)
    stockpup_df = stockpup_df[~stockpup_df.index.isin(quarterly_df.index)]

    quarterly_df = pd.concat([quarterly_df, stockpup_df])
    quarterly_df.sort_index(inplace=True)

    # Add general info
    quarterly_df = quarterly_df.join(stock_info_df, on=[QC.TICKER_SYMBOL])
    quarterly_df[QC.SECTOR].fillna(MISSING_SECTOR, inplace=True)
    quarterly_df[QC.INDUSTRY].fillna(MISSING_INDUSTRY, inplace=True)
    quarterly_df[QC.DEBT_SHORT].fillna(0, inplace=True)

    print(f"Initial combined data size: {quarterly_df.shape}")

    # Apply econ statistic formulae
    quarterly_df = apply_engineered_columns(quarterly_df,
                                            columns=NUMERIC_COLUMNS,
                                            formulae=FORMULAE)

    market_index_df = apply_engineered_columns(market_index_df,
                                               columns=[QC.VOLATILITY,
                                                        QC.PRICE_AVG],
                                               formulae={QC.VOLATILITY: FORMULAE[
                                                   QC.VOLATILITY]})

    market_index_df.sort_index(inplace=True)

    quarterly_df = add_comparison_to_market_index(df=quarterly_df,
                                                  market_index_df=market_index_df,
                                                  market_indices=hyperparams.MARKET_INDICES,
                                                  columns=COLUMNS_TO_COMPARE_TO_MARKET_INDICES)

    quarterly_df[QC.QUARTER] = quarterly_df.index.get_level_values(
        QC.QUARTER)

    quarterly_df[QC.DIVIDEND_PER_SHARE] = quarterly_df[
        QC.DIVIDEND_PER_SHARE].fillna(0)

    for col in CATEGORICAL_COLUMNS:
        try:
            quarterly_df[col] = quarterly_df[col].astype('category')
        except KeyError:
            pass

    return quarterly_df


def read_stock_info() -> pd.DataFrame:
    stock_info_df = pd.read_csv(STOCK_GENERAL_INFO_CSV)[['tickerSymbol', 'sector', 'industry']]

    stock_info_df.rename(columns={
        'tickerSymbol': QC.TICKER_SYMBOL,
        'sector': QC.SECTOR,
        'industry': QC.INDUSTRY
    }, inplace=True)

    stock_info_df.set_index([QC.TICKER_SYMBOL], inplace=True)

    return stock_info_df


def preprocess_quarterly_data(hyperparams: Hyperparams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
    quarterly_df = pd.read_sql_query(f'SELECT * FROM {YF_QUARTERLY_TABLE_NAME}', db_conn)
    db_conn.close()

    # This needs to be above the filters below otherwise we'll drop quarterly data!
    # (indices have no revenue)
    market_index_df = quarterly_df[
        quarterly_df[QC.TICKER_SYMBOL].isin(hyperparams.MARKET_INDICES)]

    market_index_df.dropna(subset=[QC.DATE,
                                   QC.PRICE_AVG,
                                   QC.PRICE_HI,
                                   QC.PRICE_LO,
                                   ],
                           inplace=True)
    market_index_df.set_index(INDEX_COLUMNS, inplace=True)

    quarterly_df.dropna(subset=[QC.DATE,
                                QC.REVENUE,
                                QC.PRICE_AVG,
                                QC.PRICE_HI,
                                QC.PRICE_LO,
                                QC.EARNINGS,
                                QC.MARKET_CAP],
                        inplace=True)

    quarterly_df = quarterly_df[((quarterly_df[QC.REVENUE] != 0) &
                                 (quarterly_df[QC.EARNINGS] != 0) &
                                 (quarterly_df[QC.MARKET_CAP] != 0) &
                                 (~quarterly_df[QC.TICKER_SYMBOL].isin(
                                     hyperparams.MARKET_INDICES)))]

    quarterly_df.set_index(INDEX_COLUMNS, inplace=True)

    quarterly_df[QC.AVG_RECOMMENDATION_SCORE] = quarterly_df.apply(
        get_avg_recommendation_score, axis=1)

    quarterly_df.sort_index(inplace=True)

    return quarterly_df, market_index_df


def preprocess_stockpup_data() -> pd.DataFrame:
    db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
    df = pd.read_sql_query(f'SELECT * FROM {STOCKPUP_TABLE_NAME}', db_conn)
    db_conn.close()

    df.dropna(subset=[SPC.SHARES,
                      SPC.SHARES_SPLIT_ADJUSTED,
                      SPC.FREE_CASH_FLOW_PER_SHARE,
                      SPC.EARNINGS,
                      SPC.SHAREHOLDER_EQUITY,
                      SPC.LIABILITIES,
                      SPC.PRICE],
              inplace=True)
    df = df[((df[SPC.REVENUE] != 0) &
             (df[SPC.EARNINGS] != 0))]

    df[SPC.QUARTER_END] = pd.to_datetime(df[SPC.QUARTER_END])

    df[QC.QUARTER] = df[SPC.QUARTER_END].apply(
        lambda r: QuarterlyIndex.from_date(r).quarter)
    df[QC.YEAR] = df[SPC.QUARTER_END].apply(
        lambda r: QuarterlyIndex.from_date(r).year)
    df[QC.DIVIDENDS] = df[SPC.DIVIDEND_PER_SHARE] * df[
        SPC.SHARES]
    df[QC.DATE] = df[SPC.QUARTER_END].apply(lambda r: str(r.date()))
    df[QC.OPERATING_INCOME] = df[SPC.FREE_CASH_FLOW_PER_SHARE] * df[
        SPC.SHARES]
    df[QC.MARKET_CAP] = df[SPC.SHARES_SPLIT_ADJUSTED] * df[
        SPC.PRICE]
    df[QC.DEBT_SHORT] = 0  # I think short long term debt is in long term debt?
    df[QC.EBIT] = df[SPC.REVENUE] - df[
        SPC.CAPITAL_EXPENDITURES]

    df.rename(columns={
        SPC.ASSETS: QC.ASSETS,
        SPC.REVENUE: QC.REVENUE,
        SPC.LIABILITIES: QC.LIABILITIES,
        SPC.LONG_TERM_DEBT: QC.DEBT_LONG,
        SPC.SHAREHOLDER_EQUITY: QC.STOCKHOLDER_EQUITY,
        SPC.CASH_AT_END_OF_PERIOD: QC.CASH,
        SPC.PRICE: QC.PRICE_AVG,
        SPC.PRICE_LOW: QC.PRICE_LO,
        SPC.PRICE_HIGH: QC.PRICE_HI,
        SPC.SPLIT_FACTOR: QC.SPLIT,
        SPC.SHARES_SPLIT_ADJUSTED: QC.COMMON_STOCK,
        SPC.EARNINGS: QC.EARNINGS  # These aren't exactly the same
    }, inplace=True)

    df[QC.PRICE_AVG] = df[QC.PRICE_AVG]
    df[QC.PRICE_LO] = df[QC.PRICE_LO]
    df[QC.PRICE_HI] = df[QC.PRICE_HI]

    df[QC.DEBT_LONG] = df[QC.DEBT_LONG].apply(lambda row: int(row))
    df[QC.NET_INCOME] = df[
        QC.EARNINGS]  # These aren't exactly the same...

    # Filter only to columns in QC
    df = df[[col for col in df.columns if col in QC.columns()]]

    df.set_index(INDEX_COLUMNS, inplace=True)
    df.sort_index(inplace=True)

    # Drop duplicates (occurs if quarter end dates are close to eachother)
    df = df.loc[~df.index.duplicated(keep='last')]

    return df


def add_delta_columns(df: pd.DataFrame, columns: list, num_quarters: int, dropna=False):
    tickers = set(df.index.levels[TICKER_SYMBOL])

    prefix = YOY_DELTA_PREFIX if num_quarters == 4 else Q_DELTA_PREFIX
    column_rename_dict = {col: f"{prefix}{col}" for col in columns}

    delta_df = pd.DataFrame()
    for ticker in tickers:
        ticker_df = df[df.index.isin([ticker], level=TICKER_SYMBOL)][columns]
        ticker_df = ticker_df.sort_index(level=YEAR, ascending=True)
        pct_change = ticker_df.pct_change(periods=num_quarters).rename(columns=column_rename_dict)
        delta_df = delta_df.append(pct_change)

    output = df.join(delta_df)
    if dropna:
        output = output.dropna(subset=set(column_rename_dict.values()))

    print(f"Output of delta {num_quarters} Quarters: {output.shape}")

    return output


def apply_engineered_columns(df: pd.DataFrame,
                             columns: List[str],
                             formulae: Dict[str, Callable]) -> pd.DataFrame:
    """
    Applies formulae to a dataframe, adding new columns.
    Also adds quarterly and year-over-year differential data

    Args:
        df: target df
        columns: column names for adding quarterly and year-over-year differential data
        formulae: list of formuale to apply

    Returns:
        New DataFrame
    """
    for col_name, fn in formulae.items():
        df[col_name] = df.apply(fn, axis=1)

    new_df = add_delta_columns(df, columns, 1)
    new_df = add_delta_columns(new_df, columns, 4)

    return new_df


def add_comparison_to_market_index(df: pd.DataFrame,
                                   market_index_df: pd.DataFrame,
                                   market_indices: List[str],
                                   columns: List[str],
                                   dropna=False):
    idx = pd.IndexSlice
    output = df.copy()

    for mkt_idx in market_indices:
        mkt_idx_df = market_index_df.loc[
            idx[mkt_idx, :, :]]  # removes mkt_idx from multiIndex so we can use as divisor

        for col in columns:
            vs_mkt_idx = df[col] / mkt_idx_df[col]
            vs_mkt_idx.name = f"{col}{VS_MKT_IDX}{mkt_idx}"

            output = output.join(vs_mkt_idx)

    if dropna:
        output = output.dropna(
            subset=[f"{col}{VS_MKT_IDX}{mkt_idx}" for mkt_idx in market_indices for col in columns])

    print(f"Output of comparison to market index: {output.shape}")

    return output


def get_avg_recommendation_score(row: pd.Series):
    if row[QC.AVG_RECOMMENDATIONS] is None or str(
            row[QC.AVG_RECOMMENDATIONS]) == 'nan':
        return pd.Series([0])

    avg_recommendation = np.mean(
        [float(v) for v in json.loads(row[QC.AVG_RECOMMENDATIONS]).values()])
    return pd.Series([avg_recommendation])
