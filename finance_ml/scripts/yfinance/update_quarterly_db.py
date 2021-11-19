from datetime import datetime, timedelta, date
import os
import pandas as pd
import sqlite3
import yfinance_ez as yf
from typing import Tuple, Dict, List, Optional
import re

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, QuarterlyColumns as QC, StockPupColumns as SPC,
    YF_QUARTERLY_TABLE_NAME, MARKET_INDICES, STOCKPUP_TABLE_NAME,
    STOCK_GENERAL_INFO_CSV)
from finance_ml.scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS)

from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.yf_utils import (
    get_average_price_over_time_period, build_split_data,
    get_average_recommendations_over_time_period)
from finance_ml.utils.utils import get_ticker_symbols


def _check_quarter_data(input_dates) -> List[str]:
    """
    Make sure that quarter dates are sequential. For example, if we have quarter data in which
     the quarters look like: '1334', we correct it to: '1234'

    Assumes that there is only one incorrect quarter

    Returns a list of dates in the form xQyyyy
    """
    dates = list(input_dates)
    reverse = False
    if dates[0] < dates[1]:  # let's make sure that dates are in descending order
        dates.reverse()
        reverse = True

    qi_data = [QuarterlyIndex.from_date(d) for d in dates]
    last_idx = len(qi_data) - 1

    for i, qi in enumerate(qi_data):
        left_idx = 3 if i == 0 and len(qi_data) >= 4 else i - 1
        right_idx = i - 3 if i == last_idx and len(qi_data) >= 4 else i + 1

        # assert that dates are in the descending order
        if i != last_idx:
            assert qi >= qi_data[right_idx], f"{qi} not >= {qi_data[right_idx]}"

        if qi.quarter == qi_data[left_idx].quarter:
            if i == 0:
                # 3143 --> 2143
                if qi.time_travel(-1) == qi_data[right_idx]:
                    qi_data[left_idx] = qi_data[left_idx].time_travel(1)
                # 2142 --> 2143
                elif qi.time_travel(-2) == qi_data[right_idx]:
                    qi_data[i] = qi.time_travel(-1)
            elif i < 3:
                # 2234 --> 1234
                if qi.time_travel(1) == qi_data[right_idx]:
                    qi_data[left_idx] = qi_data[left_idx].time_travel(-1)
                # 1134 --> 1234
                elif qi.time_travel(2) == qi_data[right_idx]:
                    qi_data[i] = qi.time_travel(1)
            else:
                # 1233 --> 1234
                if qi == qi_data[left_idx - 1].time_travel(1):
                    qi_data[i] = qi.time_travel(1)
                # 1244 --> 1234
                elif qi == qi_data[left_idx - 1].time_travel(2):
                    qi_data[i - 1] = qi_data[i - 1].time_travel(-1)

    if reverse:
        qi_data.reverse()

    return qi_data.apply(lambda qi: qi.to_xQyyyy()) if isinstance(dates, pd.Series) \
        else [qi.to_xQyyyy() for qi in qi_data]


def get_quarterly_price_history(ticker, start) -> pd.DataFrame:
    price_history_df = get_average_price_over_time_period(ticker=ticker,
                                                          start=start,
                                                          time_period=13 * 7)

    qi_data = price_history_df[QC.DATE].apply(lambda d: QuarterlyIndex.from_date(d))

    price_history_df[QC.YEAR] = qi_data.apply(lambda qi: qi.year)
    price_history_df[QC.QUARTER] = qi_data.apply(lambda qi: qi.quarter)

    price_history_df[QC.SPLIT] = build_split_data(ticker, price_history_df[
        QC.DATE])

    return price_history_df


def _dt_to_index(dt, dates):
    """identify which date bucket to assign dt to"""
    # dates is in ascending order (last is most recent)
    for i, _dt in enumerate(dates):
        if dt < _dt:
            return len(dates) - i - 1


def get_dividends(ticker: yf.Ticker, dates: List[date], q_index):
    # dates is in descending order (first is most recent)
    try:
        dividends = ticker.dividends.copy()
    except KeyError:
        return pd.DataFrame({})

    if not dates:
        return pd.DataFrame({})

    earliest_date = datetime(dates[-1].year, dates[-1].month, dates[-1].day)
    latest_date = datetime(dates[0].year, dates[0].month, dates[0].day)

    dividends = dividends.loc[(
                                      dividends.index.to_pydatetime() > earliest_date - timedelta(
                                  days=13 * 7)) & (
                                      dividends.index.to_pydatetime() <= latest_date)]
    df = pd.DataFrame(dividends)
    df = df.reset_index()

    ascending_dates = list(dates)
    ascending_dates.reverse()

    # We could potentially have multiple dividends per quarter
    df['QuarterIndex'] = df.Date.apply(lambda dt: _dt_to_index(dt, ascending_dates))
    df = df.groupby('QuarterIndex').sum()

    # set dividends to 0 for quarters without dividends
    zero_df = pd.DataFrame({'QuarterIndex': range(len(dates)),
                            'Dividends': [0.0] * len(dates)})
    zero_df.set_index('QuarterIndex', inplace=True)
    df = zero_df.add(df, fill_value=0)

    # add "xQyyyy" index
    df.set_index(q_index, inplace=True)

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
    info = {feature: ticker.info.get(feature, '') for feature in INFO_KEYS}

    quarterly_balance_sheet = ticker.quarterly_balance_sheet.loc[
        [key for key in BALANCE_SHEET_KEYS if key in ticker.quarterly_balance_sheet.index]]
    financial_data = ticker.quarterly_financials.loc[
        [key for key in FINANCIAL_KEYS if key in ticker.quarterly_financials.index]]
    cashflow_data = ticker.quarterly_cashflow.loc[
        [key for key in CASHFLOW_KEYS if key in ticker.quarterly_cashflow.index]]
    combined_data = pd.concat([quarterly_balance_sheet, financial_data, cashflow_data])

    # Descending QuarterlyIndexes (first is most recent)
    q_indexes = [f"{qi[0]}Q{qi[-4:]}" for qi in ticker.quarterly_earnings.index]
    q_indexes.reverse()

    quarterly_dates = [dt.date() for dt in combined_data.columns][:len(q_indexes)]

    combined_data = combined_data.rename(
        columns={k: v for k, v in zip(combined_data.columns, q_indexes)}).transpose()

    # filter out q indexes that aren't in combined_data
    combined_data = combined_data.loc[q_indexes]

    q_data = combined_data.join(ticker.quarterly_earnings)

    # drop duplicate rows by index
    q_data = q_data[~q_data.index.duplicated(keep='first')]

    dividends = get_dividends(ticker, quarterly_dates, q_data.index)
    q_data = q_data.join(dividends)

    # Add Average stock price data
    for fn in (get_average_price_over_time_period, get_average_recommendations_over_time_period):
        if quarterly_dates:
            # data returned is in ascending order (last is most recent)
            avg_data = fn(ticker=ticker,
                          start=quarterly_dates[-1] - timedelta(days=13 * 7),
                          time_period=13 * 7)

            if avg_data.empty:
                continue

            new_index = list(combined_data.index)
            new_index.reverse()

            if len(avg_data) > len(new_index):
                # drop the most recent date(s) from avg_data
                avg_data.drop(avg_data.tail(len(avg_data) - len(new_index)).index, inplace=True)
            elif len(new_index) > len(avg_data):
                # drop the oldest quarter(s) from new_index
                new_index = new_index[len(new_index) - len(avg_data):]

            avg_data.index = new_index

            shared_columns = set(avg_data.columns).intersection(set(q_data.columns))
            avg_data.drop(columns=shared_columns, inplace=True)

            q_data = q_data.join(avg_data)

    q_data[QC.MARKET_CAP] = info.get('marketCap', 'NULL')
    q_data[QC.TICKER_SYMBOL] = ticker.ticker
    q_data[QC.YEAR] = q_data.index.to_series().apply(lambda i: int(i[-4:]))
    q_data[QC.QUARTER] = q_data.index.to_series().apply(lambda i: int(i[0]))
    q_data[QC.SPLIT] = build_split_data(ticker, quarterly_dates)

    # Remove duplicate columns
    q_data = q_data.loc[:, ~q_data.columns.duplicated()]

    return info, q_data


def _row_in_table(row, db_conn) -> bool:
    resp = db_conn.execute(f"""SELECT * FROM {YF_QUARTERLY_TABLE_NAME}
WHERE {QC.TICKER_SYMBOL}='{row[QC.TICKER_SYMBOL]}' AND
      {QC.QUARTER}={row[QC.QUARTER]} AND
      {QC.YEAR}={row[QC.YEAR]}""")

    if resp.fetchone():
        return True
    return False


def _delete_row(row, db_conn):
    print(f"Deleting row Q{row[QC.QUARTER]} {row[QC.YEAR]} from {YF_QUARTERLY_TABLE_NAME}")

    db_conn.execute(f"""DELETE FROM {YF_QUARTERLY_TABLE_NAME}
WHERE {QC.TICKER_SYMBOL}='{row[QC.TICKER_SYMBOL]}' AND
      {QC.QUARTER}={row[QC.QUARTER]} AND
      {QC.YEAR}={row[QC.YEAR]}""")


def update_quarterly_database(ticker_symbols: Optional[List[str]] = None,
                              auto_backfill_splits: bool = False):
    """
    Updates quarterly database with new quarterly ticker info

    If we encounter a split, we backfill all existing data accounting for the split

    Args:
        ticker_symbols: List[str] of ticker symbols. If None, defaults to using ticker symbols
            found in table already.
        auto_backfill_splits: bool - whether to auto backfill splits. If False, prompts for user
            input before backfilling.
    """
    today = datetime.now()

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        all_ticker_symbols = get_ticker_symbols()

        if ticker_symbols is None:
            print("Updating Quarterly DB using existing ticker symbols")
            ticker_symbols = all_ticker_symbols
        else:
            print(f"Updating Quarterly DB with tickers: {ticker_symbols}")
            # Update general info in case new ticker symbols are being added
            ticker_symbols = [ts.upper() for ts in ticker_symbols]
            update_general_info(set(all_ticker_symbols).union(ticker_symbols))

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
                except Exception as e:
                    print(f"Error fetching quarterly data: {e}")
                    raise
                    continue

                # Shouldn't be necessary, but as a precaution this removes
                # all non alphanumeric characters from column names
                ticker_df = ticker_df.rename(
                    columns={col: re.compile('[\W_]+').sub('', col) for col in ticker_df.columns})

            split_factor = 1

            for i, row in ticker_df.iterrows():
                if _row_in_table(row, db_conn):
                    _delete_row(row, db_conn)
                elif isinstance(row[QC.SPLIT], (int, float)) and not pd.isna(row[QC.SPLIT]):
                    # split adjustment only relevant if data not in table already.
                    # this loop sets the split quarter/year to the earliest instance of a split found
                    split_factor *= row[QC.SPLIT]
                    split_quarter = row[QC.QUARTER]
                    split_year = row[QC.YEAR]

            if split_factor != 1:
                if auto_backfill_splits:
                    do_backfill = 'y'
                else:
                    do_backfill = input(
                        f"Found stock split - {split_factor}x in Q{split_quarter} {split_year}. "
                        "Backfill existing tables w/ split data? [y/n]: ")
                if do_backfill.lower().strip() == 'y':
                    # Adjust all pre-existing split data to make it easy to compare prices!
                    # (also since all new yf data added will be in split adjusted terms already)
                    _apply_split_to_yf_table(ticker_symbol=ticker_symbol,
                                             split_factor=split_factor,
                                             year=split_year,
                                             quarter=split_quarter,
                                             db_conn=db_conn)

                    _apply_split_to_stockpup_table(ticker_symbol=ticker_symbol,
                                                   split_factor=split_factor,
                                                   db_conn=db_conn)

            if not ticker_df.empty:
                print(ticker_df)

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


def update_general_info(ticker_symbols):
    columns = ['shortName', 'sector', 'industry']
    ticker_symbols = [ts.upper() for ts in ticker_symbols]

    if os.path.exists(STOCK_GENERAL_INFO_CSV):
        df = pd.read_csv(STOCK_GENERAL_INFO_CSV)
        df = df[['tickerSymbol'] + columns]
        os.remove(STOCK_GENERAL_INFO_CSV)

    for ticker_symbol in ticker_symbols:
        if df.loc[df['tickerSymbol'] == ticker_symbol].empty:
            print(f"Adding General Info for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            try:
                info = {key: ticker.info.get(key, '') for key in columns}
                info['tickerSymbol'] = ticker_symbol

                df = df.append(info, ignore_index=True)
            except Exception as e:
                print(f"Exception occurred: {e}")
                pass

    df = df.set_index(['tickerSymbol'])

    with open(STOCK_GENERAL_INFO_CSV, 'w') as f:
        f.write(df.to_csv())


if __name__ == "__main__":
    update_quarterly_database()
