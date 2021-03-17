import numpy as np
import pandas as pd
import json
import sqlite3
from typing import Tuple, List, Dict, Callable

from finance_ml.utils.constants import (
    QuarterlyColumns as QC, StockPupColumns as SPC, STOCKPUP_TABLE_NAME, QUARTERLY_DB_FILE_PATH,
    YF_QUARTERLY_TABLE_NAME, INDEX_COLUMNS, MISSING_SECTOR, MISSING_INDUSTRY,
    STOCK_GENERAL_INFO_CSV, FORMULAE, Q_DELTA_PREFIX, YOY_DELTA_PREFIX, NUMERIC_COLUMNS,
    COLUMNS_TO_COMPARE_TO_MARKET_INDICES, QUARTER, YEAR, VS_MKT_IDX, CATEGORICAL_COLUMNS
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

    quarterly_df = compare_to_market_indices(quarterly_df, market_index_df, hyperparams)

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

    quarterly_df.set_index([QC.TICKER_SYMBOL,
                            QC.QUARTER,
                            QC.YEAR],
                           inplace=True)

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


def _add_delta_columns(row: pd.Series, df: pd.DataFrame, columns: list, num_quarters: int):
    try:
        # row.name returns the multiIndex tuple
        prev_quarter_tuple = QuarterlyIndex(*row.name).time_travel(-num_quarters).to_tuple()
        prev_quarter_row = df.loc[prev_quarter_tuple]
    except Exception as e:
        # print(f'Unable to find quarterly info for {prev_quarter_tuple}')
        prev_quarter_row = pd.DataFrame()

    if not prev_quarter_row.empty:
        new_cols = []

        for col in columns:
            if prev_quarter_row[col] == 0:
                new_cols.append(0)
            else:
                # converting to float to get rid of index terms
                new_cols.append(float((row[col] - prev_quarter_row[col]) / prev_quarter_row[col]))

        return pd.Series(new_cols)

    return pd.Series([None] * len(columns))


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

    q_delta_col_names = [f'{Q_DELTA_PREFIX}{col}' for col in columns]
    yoy_delta_col_names = [f'{YOY_DELTA_PREFIX}{col}' for col in columns]

    df[q_delta_col_names] = df.apply(_add_delta_columns,
                                     axis=1, df=df, columns=columns, num_quarters=1)
    df[yoy_delta_col_names] = df.apply(_add_delta_columns,
                                       axis=1, df=df, columns=columns, num_quarters=4)

    return df


def _compare_to_market_index(
        row: pd.Series, market_indices: List[str], market_index_df: pd.DataFrame):
    new_cols = []
    for col in COLUMNS_TO_COMPARE_TO_MARKET_INDICES:
        for mkt_idx in market_indices:
            try:
                mkt_idx_row = market_index_df.loc[mkt_idx, row.name[QUARTER], row.name[YEAR]]
            except:
                # print(f'Unable to find {mkt_idx} Q{row.name[QUARTER]} {row.name[YEAR]}')
                mkt_idx_row = pd.DataFrame()

            if not mkt_idx_row.empty:
                if mkt_idx_row[col] == 0:
                    new_cols.append(0)
                else:
                    # converting to float to drop index terms
                    new_cols.append(float(row[col] / mkt_idx_row[col]))
            else:
                new_cols.append(None)

    return pd.Series(new_cols)


def compare_to_market_indices(df: pd.DataFrame,
                              market_index_df: pd.DataFrame,
                              hyperparams: Hyperparams) -> pd.DataFrame:
    vs_market_indices_col_names = [f'{col}{VS_MKT_IDX}{mkt_idx}'
                                   for col in COLUMNS_TO_COMPARE_TO_MARKET_INDICES
                                   for mkt_idx in hyperparams.MARKET_INDICES]

    df[vs_market_indices_col_names] = df.apply(_compare_to_market_index,
                                               axis=1,
                                               market_indices=hyperparams.MARKET_INDICES,
                                               market_index_df=market_index_df)

    return df


def get_avg_recommendation_score(row: pd.Series):
    if row[QC.AVG_RECOMMENDATIONS] is None or str(
            row[QC.AVG_RECOMMENDATIONS]) == 'nan':
        return pd.Series([0])

    avg_recommendation = np.mean(
        [float(v) for v in json.loads(row[QC.AVG_RECOMMENDATIONS]).values()])
    return pd.Series([avg_recommendation])
