from datetime import datetime, timedelta, date
import pandas as pd
import sqlite3
import yfinance_ez as yf
from typing import Union, Tuple, Dict, List
import glob
import os
import re
import json

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, DATA_PATH, QuarterlyColumns, STOCKPUP_TABLE_NAME,
    YF_QUARTERLY_TABLE_NAME, STOCK_GENERAL_INFO_CSV, StockPupColumns, MARKET_INDICES)
from scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS, RECOMMENDATION_GRADE_MAPPING,
    YF_QUARTERLY_TABLE_SCHEMA)

from finance_ml.utils.yf_utils import (
    get_average_price_over_time_period, date_to_xQyyyy, build_split_data, get_average_recommendations_over_time_period)
from finance_ml.utils.utils import get_ticker_symbols


def get_quarterly_price_history(ticker, start):
    price_history_df = get_average_price_over_time_period(ticker=ticker,
                                                          start=start,
                                                          time_period=13 * 7)

    price_history_df[QuarterlyColumns.YEAR] = price_history_df[QuarterlyColumns.DATE].apply(
        lambda d: int(date_to_xQyyyy(d)[-4:]))
    price_history_df[QuarterlyColumns.QUARTER] = price_history_df[QuarterlyColumns.DATE].apply(
        lambda d: int(date_to_xQyyyy(d)[0]))
    price_history_df[QuarterlyColumns.SPLIT] = build_split_data(ticker, price_history_df[
        QuarterlyColumns.DATE])

    return price_history_df


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
    q_indexes = [f'{date_to_xQyyyy(date)}' for date in quarter_end_dates]

    # Row Indexes are Quarter strings: '1Q2019'
    q_data = ticker.quarterly_earnings.copy()
    if q_data.empty:
        q_data = pd.DataFrame({col: [None] * len(q_indexes) for col in q_data.columns})
    q_data.index = q_indexes
    q_data[QuarterlyColumns.TICKER_SYMBOL] = [ticker.ticker] * len(quarter_end_dates)
    q_data[QuarterlyColumns.YEAR] = q_data.index.to_series().apply(lambda i: int(i[-4:]))
    q_data[QuarterlyColumns.QUARTER] = q_data.index.to_series().apply(lambda i: int(i[0]))
    q_data[QuarterlyColumns.SPLIT] = build_split_data(ticker, quarter_end_dates)

    info = {feature: ticker.info.get(feature, '') for feature in INFO_KEYS}

    quarterly_balance_sheet = ticker.quarterly_balance_sheet.loc[
        [key for key in BALANCE_SHEET_KEYS if key in ticker.quarterly_balance_sheet.index]]
    financial_data = ticker.quarterly_financials.loc[
        [key for key in FINANCIAL_KEYS if key in ticker.quarterly_financials.index]]
    cashflow_data = ticker.quarterly_cashflow.loc[
        [key for key in CASHFLOW_KEYS if key in ticker.quarterly_cashflow.index]]
    combined_data = pd.concat([quarterly_balance_sheet, financial_data, cashflow_data])
    combined_data = combined_data.rename(
        columns={date: f'{date_to_xQyyyy(date)}'
                 for date in combined_data.columns}).transpose()

    q_data = q_data.join(combined_data)

    # Add Average stock price data
    for fn in (get_average_price_over_time_period, get_average_recommendations_over_time_period):
        avg_data = fn(ticker=ticker,
                      start=quarter_end_dates[-1] - timedelta(days=13 * 7),
                      time_period=13 * 7)

        if avg_data.empty:
            continue

        avg_data['Quarter'] = avg_data[QuarterlyColumns.DATE].apply(
            lambda d: f'{date_to_xQyyyy(d)}')
        avg_data.drop(columns=[QuarterlyColumns.DATE, QuarterlyColumns.TICKER_SYMBOL], inplace=True)
        avg_data.set_index(['Quarter'], inplace=True)

        q_data = q_data.join(avg_data)

    q_data.reset_index(drop=True, inplace=True)
    q_data[QuarterlyColumns.DATE] = quarter_end_dates
    q_data[QuarterlyColumns.MARKET_CAP] = info.get('marketCap', 'NULL')

    # Remove duplicate columns
    q_data = q_data.loc[:, ~q_data.columns.duplicated()]

    return info, q_data


def update_quarterly_database():
    today = datetime.now()

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        ticker_symbols = get_ticker_symbols()

        for ticker_symbol in ticker_symbols:
            print(f'Adding: {ticker_symbol}')
            ticker = yf.Ticker(ticker_symbol)

            if ticker_symbol in MARKET_INDICES:
                # Market Indices don't have balance sheets, etc. so stick to price data
                ticker_df = get_quarterly_price_history(ticker, start=datetime.now() - timedelta(days=365))
            else:
                if ticker.quarterly_balance_sheet.empty or today - \
                        ticker.quarterly_balance_sheet.columns[0] > timedelta(days=90):
                    continue

                ticker_info, ticker_df = get_quarterly_data(ticker)
                # Shouldn't be necessary, but this removes all non alphanumeric characters from column names
                ticker_df = ticker_df.rename(
                    columns={col: re.compile('[\W_]+').sub('', col) for col in ticker_df.columns})

            dates_to_drop = []

            for i, row in ticker_df.iterrows():
                resp = db_conn.execute(f"""SELECT * FROM {YF_QUARTERLY_TABLE_NAME}
WHERE {QuarterlyColumns.TICKER_SYMBOL}='{row[QuarterlyColumns.TICKER_SYMBOL]}' AND
      {QuarterlyColumns.QUARTER}={row[QuarterlyColumns.QUARTER]} AND
      {QuarterlyColumns.YEAR}={row[QuarterlyColumns.YEAR]}""")

                if resp.fetchone():
                    dates_to_drop.append(row[QuarterlyColumns.DATE])

            ticker_df = ticker_df[
                ticker_df.apply(lambda r: r[QuarterlyColumns.DATE] not in dates_to_drop, axis=1)]

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