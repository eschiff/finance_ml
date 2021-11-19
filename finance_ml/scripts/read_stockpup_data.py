import glob
import pandas as pd
import os
import sqlite3
import numpy as np

from finance_ml.utils.constants import (
    StockPupColumns, QuarterlyColumns, DATA_PATH,
    STOCKPUP_TABLE_NAME, QUARTERLY_DB_FILE_PATH, YF_QUARTERLY_TABLE_NAME)


def _(s):
    return s.replace(' ', '')


def build_stock_pup_database():
    data_files = glob.glob(os.path.join(DATA_PATH, 'stockpup_data', '*.csv'))

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        command = f'''CREATE TABLE {STOCKPUP_TABLE_NAME} (
    {QuarterlyColumns.TICKER_SYMBOL} TEXT,
    {StockPupColumns.QUARTER_END} TEXT,
    {StockPupColumns.SHARES} INT,
    {StockPupColumns.SHARES_SPLIT_ADJUSTED} INT,
    {StockPupColumns.SPLIT_FACTOR} INT,
    {StockPupColumns.ASSETS} NUMERIC,
    {StockPupColumns.CURRENT_ASSETS} NUMERIC ,
    {StockPupColumns.LIABILITIES} NUMERIC,
    {StockPupColumns.CURRENT_LIABILITIES} NUMERIC,
    {StockPupColumns.SHAREHOLDER_EQUITY} NUMERIC,
    {StockPupColumns.NON_CONTROLLING_INTEREST} NUMERIC,
    {StockPupColumns.PREFERRED_EQUITY} NUMERIC ,
    {StockPupColumns.GOODWILL_AND_INTANGIBLES} NUMERIC ,
    {StockPupColumns.LONG_TERM_DEBT} TEXT,
    {StockPupColumns.REVENUE} NUMERIC,
    {StockPupColumns.EARNINGS} NUMERIC,
    {StockPupColumns.EARNINGS_AVAILABLE_FOR_COMMON_STOCKHOLDERS} NUMERIC,
    {StockPupColumns.EPS_BASIC} NUMERIC,
    {StockPupColumns.EPS_DILUTED} NUMERIC,
    {StockPupColumns.DIVIDEND_PER_SHARE} NUMERIC,
    {StockPupColumns.CASH_FROM_OPERATING_ACTIVITES} NUMERIC,
    {StockPupColumns.CASH_FROM_INVESTING_ACTIVITIES} NUMERIC ,
    {StockPupColumns.CASH_FROM_FINANCING_ACTIVITES} NUMERIC ,
    {StockPupColumns.CASH_CHANGE_DURING_PERIOD} NUMERIC,
    {StockPupColumns.CASH_AT_END_OF_PERIOD} NUMERIC,
    {StockPupColumns.CAPITAL_EXPENDITURES} NUMERIC,
    {StockPupColumns.PRICE} NUMERIC,
    {StockPupColumns.PRICE_HIGH} NUMERIC,
    {StockPupColumns.PRICE_LOW} NUMERIC,
    {StockPupColumns.ROE} NUMERIC,
    {StockPupColumns.ROA} NUMERIC ,
    {StockPupColumns.BOOK_VALUE_OF_EQUITY_PER_SHARE} NUMERIC ,
    {StockPupColumns.P_B_RATIO} NUMERIC,
    {StockPupColumns.P_E_RATIO} NUMERIC,
    {StockPupColumns.CUM_DIVIDENDS_PER_SHARE} NUMERIC,
    {StockPupColumns.DIVIDEND_PAYOUT_RATIO} NUMERIC,
    {StockPupColumns.LONG_TERM_DEBT_TO_EQUITY_RATIO} NUMERIC,
    {StockPupColumns.EQUITY_TO_ASSETS_RATIO} NUMERIC,
    {StockPupColumns.NET_MARGIN} NUMERIC,
    {StockPupColumns.ASSET_TURNOVER} NUMERIC,
    {StockPupColumns.FREE_CASH_FLOW_PER_SHARE} NUMERIC ,
    {StockPupColumns.CURRENT_RATIO} NUMERIC ,
    PRIMARY KEY ({QuarterlyColumns.TICKER_SYMBOL}, {StockPupColumns.QUARTER_END})
);'''

        print(f"Executing command: {command}")
        db_conn.execute(command)

        for data_file_path in data_files:
            full_file_name = os.path.basename(data_file_path)
            file_name, ext = os.path.splitext(full_file_name)

            ticker_symbol = file_name.split('_')[-1]
            print(f"Found ticker symbol: {ticker_symbol}")
            ticker_df = pd.read_csv(data_file_path)

            ticker_df.replace('None', value=np.nan, inplace=True)

            # First column is just row number
            ticker_df.drop(columns=[ticker_df.columns[0]], inplace=True)

            ticker_df.to_sql(name=YF_QUARTERLY_TABLE_NAME,
                             con=db_conn,
                             if_exists='append',
                             index=False)

        db_conn.commit()

    finally:
        if db_conn:
            db_conn.close()
