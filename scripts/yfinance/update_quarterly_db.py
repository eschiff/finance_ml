from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import yfinance_ez as yf
from typing import Tuple, Dict, Union, List
import re
from functools import reduce

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, QuarterlyColumns as QC, StockPupColumns as SPC,
    YF_QUARTERLY_TABLE_NAME, MARKET_INDICES, STOCKPUP_TABLE_NAME)
from scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS)

from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.yf_utils import (
    get_average_price_over_time_period, build_split_data,
    get_average_recommendations_over_time_period)
from finance_ml.utils.utils import get_ticker_symbols


def _check_quarter_data(dates: Union[pd.Series, List]) -> Union[pd.Series, List]:
    """
    Make sure that quarter dates are sequential. For example, if we have quarter data in which
     the quarters look like: '1334', we correct it to: '1234'

    Assumes that there is only one incorrect quarter

    Returns a list of dates in the form xQyyyy
    """
    qi_data = dates.apply(lambda d: QuarterlyIndex.from_date(d)) if isinstance(dates, pd.Series) \
        else [QuarterlyIndex.from_date(d) for d in dates]

    for i in range(len(qi_data)):
        left_idx = i - 1 if i > 0 else len(qi_data) - 1
        right_idx = 0 if i == len(qi_data) - 1 else i + 1

        if qi_data[i] == qi_data[left_idx]:
            if qi_data[i].time_travel(1) == qi_data[right_idx]:
                qi_data[left_idx] = qi_data[left_idx].time_travel(-1)
            else:
                qi_data[i] = qi_data[i].time_travel(1)
            break

    return qi_data.apply(lambda qi: qi.to_xQyyyy()) if isinstance(dates, pd.Series) \
        else [qi.to_xQyyyy() for qi in qi_data]


def get_quarterly_price_history(ticker, start):
    price_history_df = get_average_price_over_time_period(ticker=ticker,
                                                          start=start,
                                                          time_period=13 * 7)

    qi_data = price_history_df[QC.DATE].apply(lambda d: QuarterlyIndex.from_date(d))

    price_history_df[QC.YEAR] = qi_data.apply(lambda qi: qi.year)
    price_history_df[QC.QUARTER] = qi_data.apply(lambda qi: qi.quarter)

    price_history_df[QC.SPLIT] = build_split_data(ticker, price_history_df[
        QC.DATE])

    return price_history_df


def _get_dividends(ticker: yf.Ticker):
    try:
        dividends = ticker.dividends.copy()
    except KeyError:
        return
    dividends = dividends.loc[
        dividends.index.to_pydatetime() > datetime.now() - timedelta(days=550)]
    df = pd.DataFrame(dividends)
    df.index = _check_quarter_data(df.index)
    df.rename({'Dividends': QC.DIVIDEND_PER_SHARE}, axis=1, inplace=True)
    return df


def get_quarterly_data(ticker: yf.Ticker) -> Tuple[Dict, pd.DataFrame]:
    """
    Get Quarterly Stock Data
    Args:
        ticker: (yf.Ticker)

    Returns:
        (Dict of general stock info, DataFrame of Quarterly Data)
    """
    # most recent data is first
    quarter_end_dates = [date.date() for date in ticker.quarterly_balance_sheet.columns]
    q_indexes = _check_quarter_data(quarter_end_dates)

    # Row Indexes are Quarter strings: '1Q2019'
    q_data = ticker.quarterly_earnings.copy()
    if q_data.empty:
        q_data = pd.DataFrame({col: [None] * len(q_indexes) for col in q_data.columns})
    q_data.index = q_indexes
    q_data[QC.TICKER_SYMBOL] = [ticker.ticker] * len(quarter_end_dates)
    q_data[QC.YEAR] = q_data.index.to_series().apply(lambda i: int(i[-4:]))
    q_data[QC.QUARTER] = q_data.index.to_series().apply(lambda i: int(i[0]))
    q_data[QC.SPLIT] = build_split_data(ticker, quarter_end_dates)

    dividends = _get_dividends(ticker)
    q_data = q_data.join(dividends)

    info = {feature: ticker.info.get(feature, '') for feature in INFO_KEYS}

    quarterly_balance_sheet = ticker.quarterly_balance_sheet.loc[
        [key for key in BALANCE_SHEET_KEYS if key in ticker.quarterly_balance_sheet.index]]
    financial_data = ticker.quarterly_financials.loc[
        [key for key in FINANCIAL_KEYS if key in ticker.quarterly_financials.index]]
    cashflow_data = ticker.quarterly_cashflow.loc[
        [key for key in CASHFLOW_KEYS if key in ticker.quarterly_cashflow.index]]
    combined_data = pd.concat([quarterly_balance_sheet, financial_data, cashflow_data])
    combined_data = combined_data.rename(
        columns={date: QuarterlyIndex.from_date(date).to_xQyyyy()
                 for date in combined_data.columns}).transpose()
    q_data = q_data.join(combined_data)

    # Add Average stock price data
    for fn in (get_average_price_over_time_period, get_average_recommendations_over_time_period):
        avg_data = fn(ticker=ticker,
                      start=quarter_end_dates[-1] - timedelta(days=13 * 7),
                      time_period=13 * 7)

        if avg_data.empty:
            continue
        avg_data['Quarter'] = _check_quarter_data(avg_data[QC.DATE])

        avg_data.drop(columns=[QC.DATE, QC.TICKER_SYMBOL], inplace=True)
        avg_data.set_index(['Quarter'], inplace=True)
        q_data = q_data.join(avg_data)

    q_data.reset_index(drop=True, inplace=True)

    q_data[QC.DATE] = quarter_end_dates
    q_data[QC.MARKET_CAP] = info.get('marketCap', 'NULL')

    # Remove duplicate columns
    q_data = q_data.loc[:, ~q_data.columns.duplicated()]

    return info, q_data


def update_quarterly_database(ticker_symbols=None):
    today = datetime.now()

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        if ticker_symbols is None:
            ticker_symbols = get_ticker_symbols()

        for ticker_symbol in ticker_symbols:
            print(f'Adding: {ticker_symbol}')
            ticker = yf.Ticker(ticker_symbol)

            if ticker_symbol in MARKET_INDICES:
                # Market Indices don't have balance sheets, etc. so stick to price data
                ticker_df = get_quarterly_price_history(
                    ticker, start=datetime.now() - timedelta(days=365))
            else:
                if ticker.quarterly_balance_sheet.empty:
                    print("quarterly balance sheet is empty! skipping!")
                    continue
                if today - ticker.quarterly_balance_sheet.columns[0] > timedelta(days=365):
                    print("quarterly balance sheet is older than 365 days! skipping!")
                    continue

                try:
                    ticker_info, ticker_df = get_quarterly_data(ticker)
                except:
                    print("Error fetching quarterly data")
                    continue

                # Shouldn't be necessary, but as a precaution this removes
                # all non alphanumeric characters from column names
                ticker_df = ticker_df.rename(
                    columns={col: re.compile('[\W_]+').sub('', col) for col in ticker_df.columns})

            dates_to_drop = []

            for i, row in ticker_df.iterrows():
                resp = db_conn.execute(f"""SELECT * FROM {YF_QUARTERLY_TABLE_NAME}
WHERE {QC.TICKER_SYMBOL}='{row[QC.TICKER_SYMBOL]}' AND
      {QC.QUARTER}={row[QC.QUARTER]} AND
      {QC.YEAR}={row[QC.YEAR]}""")

                if resp.fetchone():
                    dates_to_drop.append(row[QC.DATE])

            print(f'Dates already exist in table: {dates_to_drop}')
            ticker_df = ticker_df[
                ticker_df.apply(lambda r: r[QC.DATE] not in dates_to_drop, axis=1)]

            # Check for splits in ticker_df (rows don't yet exist in table)
            if not ticker_df[QC.SPLIT].isnull().all():
                split_factor = reduce(lambda x, y: x * y,
                                      ticker_df[QC.SPLIT].fillna(1.0))

                # Adjust all pre-existing split data to make it easy to compare prices!
                # (also since all new yf data added will be in split adjusted terms already)
                _apply_split_to_yf_table(ticker_symbol=ticker_symbol,
                                         split_factor=split_factor,
                                         year=ticker_df[QC.YEAR][0],
                                         quarter=ticker_df[QC.QUARTER][0],
                                         db_conn=db_conn)

                _apply_split_to_stockpup_table(ticker_symbol=ticker_symbol,
                                               split_factor=split_factor,
                                               db_conn=db_conn)

            if not ticker_df.empty:
                ticker_df.to_sql(name=YF_QUARTERLY_TABLE_NAME,
                                 con=db_conn,
                                 if_exists='append',
                                 index=False)
            else:
                print("NOTHING ADDED")

        db_conn.commit()

    finally:
        if db_conn:
            db_conn.close()


def _apply_split_to_yf_table(ticker_symbol, split_factor, year, quarter, db_conn):
    """
     Adjusts all values in the Yahoo Finance Table by the split factor
    """
    command = f'''UPDATE {YF_QUARTERLY_TABLE_NAME}  
SET "{QC.PRICE_AVG}" = "{QC.PRICE_AVG}" / {split_factor},
    "{QC.PRICE_HI}" = "{QC.PRICE_HI}" / {split_factor},
    "{QC.PRICE_LO}" = "{QC.PRICE_LO}" / {split_factor},
    "{QC.PRICE_AT_END_OF_QUARTER}" = "{QC.PRICE_AT_END_OF_QUARTER}" / {split_factor},
    "{QC.DIVIDEND_PER_SHARE}" = "{QC.DIVIDEND_PER_SHARE}" / {split_factor},
    "{QC.STOCK_ISSUED}" = "{QC.STOCK_ISSUED}" * {split_factor},
    "{QC.STOCK_REPURCHASED}" = "{QC.STOCK_REPURCHASED}" * {split_factor},
    "{QC.COMMON_STOCK}" = "{QC.COMMON_STOCK}" * {split_factor}
WHERE ("{QC.TICKER_SYMBOL}" = "{ticker_symbol}" AND (
    "{QC.YEAR}" < {year} OR (
    "{QC.YEAR}" = {year} AND "{QC.QUARTER}" < {quarter})))'''

    db_conn.execute(command)


def _apply_split_to_stockpup_table(ticker_symbol, split_factor, db_conn):
    """
     Adjusts all values in the Stockpup Table by the split factor
    """
    command = f'''UPDATE {STOCKPUP_TABLE_NAME}  
SET "{SPC.PRICE}" = "{SPC.PRICE}"/{split_factor},
    "{SPC.PRICE_HIGH}" = "{SPC.PRICE_HIGH}" / {split_factor},
    "{SPC.PRICE_LOW}" = "{SPC.PRICE_LOW}" / {split_factor},
    "{SPC.DIVIDEND_PER_SHARE}" = "{SPC.DIVIDEND_PER_SHARE}" / {split_factor},
    "{SPC.SHARES_SPLIT_ADJUSTED}" = "{SPC.SHARES_SPLIT_ADJUSTED}" * {split_factor},
    "{SPC.SPLIT_FACTOR}" = "{SPC.SPLIT_FACTOR}" * {split_factor}
WHERE "{QC.TICKER_SYMBOL}" = "{ticker_symbol}"'''

    db_conn.execute(command)


def add_dividends(ticker_symbols=None):
    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        if ticker_symbols is None:
            ticker_symbols = get_ticker_symbols()

        for ticker_symbol in ticker_symbols:
            print(f'Adding: {ticker_symbol}')
            ticker = yf.Ticker(ticker_symbol)

            if ticker_symbol in MARKET_INDICES:
                continue
            else:
                try:
                    dividends = ticker.dividends
                except KeyError:
                    continue

                # filter out rows before 1/1/2019
                dividends = dividends.loc[dividends.index.to_pydatetime() > datetime(2019, 1, 1)]

                df = pd.DataFrame(dividends)
                df['xQyyyy'] = _check_quarter_data(df.index)
                df[QC.YEAR] = df['xQyyyy'].apply(lambda i: int(i[2:]))
                df[QC.QUARTER] = df['xQyyyy'].apply(lambda i: int(i[0]))
                df[QC.TICKER_SYMBOL] = ticker.ticker
                df.drop(columns=['xQyyyy'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.rename({'Dividends': QC.DIVIDEND_PER_SHARE}, axis=1, inplace=True)

                for i, row in df.iterrows():
                    db_conn.execute(
                        f"""UPDATE {YF_QUARTERLY_TABLE_NAME} 
SET {QC.DIVIDEND_PER_SHARE}={row[QC.DIVIDEND_PER_SHARE]}
WHERE {QC.TICKER_SYMBOL}='{row[QC.TICKER_SYMBOL]}' AND
      {QC.QUARTER}={row[QC.QUARTER]} AND
      {QC.YEAR}={row[QC.YEAR]}""")

        db_conn.commit()

    finally:
        if db_conn:
            db_conn.close()


if __name__ == "__main__":
    update_quarterly_database()
