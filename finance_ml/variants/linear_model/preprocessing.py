import pandas as pd
import sqlite3
from typing import Tuple, List, Dict, Callable

from finance_ml.utils.constants import (
    QuarterlyColumns, StockPupColumns, STOCKPUP_TABLE_NAME, QUARTERLY_DB_FILE_PATH,
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
    quarterly_df = quarterly_df.join(stock_info_df, on=[QuarterlyColumns.TICKER_SYMBOL])
    quarterly_df[QuarterlyColumns.SECTOR].fillna(MISSING_SECTOR, inplace=True)
    quarterly_df[QuarterlyColumns.INDUSTRY].fillna(MISSING_INDUSTRY, inplace=True)
    quarterly_df[QuarterlyColumns.DEBT_SHORT].fillna(0, inplace=True)

    # Apply econ statistic formulae
    quarterly_df = apply_engineered_columns(quarterly_df,
                                            columns=NUMERIC_COLUMNS,
                                            formulae=FORMULAE)

    market_index_df = apply_engineered_columns(market_index_df,
                                               columns=[QuarterlyColumns.VOLATILITY,
                                                        QuarterlyColumns.PRICE_AVG],
                                               formulae={QuarterlyColumns.VOLATILITY: FORMULAE[
                                                   QuarterlyColumns.VOLATILITY]})

    market_index_df.sort_index(inplace=True)

    quarterly_df = compare_to_market_indices(quarterly_df, market_index_df, hyperparams)

    for col in CATEGORICAL_COLUMNS:
        try:
            quarterly_df[col] = quarterly_df[col].astype('category')
        except KeyError:
            pass

    return quarterly_df


def read_stock_info() -> pd.DataFrame:
    stock_info_df = pd.read_csv(STOCK_GENERAL_INFO_CSV)[['tickerSymbol', 'sector', 'industry']]

    stock_info_df.rename(columns={
        'tickerSymbol': QuarterlyColumns.TICKER_SYMBOL,
        'sector': QuarterlyColumns.SECTOR,
        'industry': QuarterlyColumns.INDUSTRY
    }, inplace=True)

    stock_info_df.set_index([QuarterlyColumns.TICKER_SYMBOL], inplace=True)

    return stock_info_df


def preprocess_quarterly_data(hyperparams: Hyperparams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
    quarterly_df = pd.read_sql_query(f'SELECT * FROM {YF_QUARTERLY_TABLE_NAME}', db_conn)
    db_conn.close()

    # This needs to be above the filters below otherwise we'll drop quarterly data!
    # (indices have no revenue)
    market_index_df = quarterly_df[
        quarterly_df[QuarterlyColumns.TICKER_SYMBOL].isin(hyperparams.MARKET_INDICES)]

    market_index_df.dropna(subset=[QuarterlyColumns.DATE,
                                   QuarterlyColumns.PRICE_AVG,
                                   QuarterlyColumns.PRICE_HI,
                                   QuarterlyColumns.PRICE_LO,
                                   ],
                           inplace=True)
    market_index_df.set_index(INDEX_COLUMNS, inplace=True)

    quarterly_df.dropna(subset=[QuarterlyColumns.DATE,
                                QuarterlyColumns.REVENUE,
                                QuarterlyColumns.PRICE_AVG,
                                QuarterlyColumns.PRICE_HI,
                                QuarterlyColumns.PRICE_LO,
                                QuarterlyColumns.EARNINGS,
                                QuarterlyColumns.MARKET_CAP],
                        inplace=True)

    quarterly_df = quarterly_df[((quarterly_df[QuarterlyColumns.REVENUE] != 0) &
                                 (quarterly_df[QuarterlyColumns.EARNINGS] != 0) &
                                 (quarterly_df[QuarterlyColumns.MARKET_CAP] != 0) &
                                 (~quarterly_df[QuarterlyColumns.TICKER_SYMBOL].isin(
                                     hyperparams.MARKET_INDICES)))]

    quarterly_df.set_index([QuarterlyColumns.TICKER_SYMBOL,
                            QuarterlyColumns.QUARTER,
                            QuarterlyColumns.YEAR],
                           inplace=True)

    quarterly_df.sort_index(inplace=True)

    return quarterly_df, market_index_df


def preprocess_stockpup_data() -> pd.DataFrame:
    db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
    df = pd.read_sql_query(f'SELECT * FROM {STOCKPUP_TABLE_NAME}', db_conn)
    db_conn.close()

    df.dropna(subset=[StockPupColumns.SHARES,
                      StockPupColumns.SHARES_SPLIT_ADJUSTED,
                      StockPupColumns.FREE_CASH_FLOW_PER_SHARE,
                      StockPupColumns.EARNINGS,
                      StockPupColumns.SHAREHOLDER_EQUITY,
                      StockPupColumns.LIABILITIES,
                      StockPupColumns.PRICE],
              inplace=True)
    df = df[((df[StockPupColumns.REVENUE] != 0) &
             (df[StockPupColumns.EARNINGS] != 0))]

    df[StockPupColumns.QUARTER_END] = pd.to_datetime(df[StockPupColumns.QUARTER_END])

    df[QuarterlyColumns.QUARTER] = df[StockPupColumns.QUARTER_END].apply(
        lambda r: QuarterlyIndex.from_date(r).quarter)
    df[QuarterlyColumns.YEAR] = df[StockPupColumns.QUARTER_END].apply(
        lambda r: QuarterlyIndex.from_date(r).year)
    df[QuarterlyColumns.DIVIDENDS] = df[StockPupColumns.DIVIDEND_PER_SHARE] * df[
        StockPupColumns.SHARES]
    df[QuarterlyColumns.DATE] = df[StockPupColumns.QUARTER_END].apply(lambda r: str(r.date()))
    df[QuarterlyColumns.OPERATING_INCOME] = df[StockPupColumns.FREE_CASH_FLOW_PER_SHARE] * df[
        StockPupColumns.SHARES]
    df[QuarterlyColumns.MARKET_CAP] = df[StockPupColumns.SHARES_SPLIT_ADJUSTED] * df[
        StockPupColumns.PRICE]
    df[QuarterlyColumns.DEBT_SHORT] = 0  # I think short long term debt is in long term debt?
    df[QuarterlyColumns.EBIT] = df[StockPupColumns.REVENUE] - df[
        StockPupColumns.CAPITAL_EXPENDITURES]

    df.rename(columns={
        StockPupColumns.ASSETS: QuarterlyColumns.ASSETS,
        StockPupColumns.REVENUE: QuarterlyColumns.REVENUE,
        StockPupColumns.LIABILITIES: QuarterlyColumns.LIABILITIES,
        StockPupColumns.LONG_TERM_DEBT: QuarterlyColumns.DEBT_LONG,
        StockPupColumns.SHAREHOLDER_EQUITY: QuarterlyColumns.STOCKHOLDER_EQUITY,
        StockPupColumns.CASH_AT_END_OF_PERIOD: QuarterlyColumns.CASH,
        StockPupColumns.PRICE: QuarterlyColumns.PRICE_AVG,
        StockPupColumns.PRICE_LOW: QuarterlyColumns.PRICE_LO,
        StockPupColumns.PRICE_HIGH: QuarterlyColumns.PRICE_HI,
        StockPupColumns.SPLIT_FACTOR: QuarterlyColumns.SPLIT,
        StockPupColumns.SHARES_SPLIT_ADJUSTED: QuarterlyColumns.COMMON_STOCK,
        StockPupColumns.EARNINGS: QuarterlyColumns.EARNINGS  # These aren't exactly the same
    }, inplace=True)

    df[QuarterlyColumns.PRICE_AVG] = df[QuarterlyColumns.PRICE_AVG]
    df[QuarterlyColumns.PRICE_LO] = df[QuarterlyColumns.PRICE_LO]
    df[QuarterlyColumns.PRICE_HI] = df[QuarterlyColumns.PRICE_HI]

    df[QuarterlyColumns.DEBT_LONG] = df[QuarterlyColumns.DEBT_LONG].apply(lambda row: int(row))
    df[QuarterlyColumns.NET_INCOME] = df[
        QuarterlyColumns.EARNINGS]  # These aren't exactly the same...

    # Filter only to columns in QuarterlyColumns
    df = df[[col for col in df.columns if col in QuarterlyColumns.columns()]]

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
