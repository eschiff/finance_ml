from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import yfinance_ez as yf
import glob
import os
import re

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, DATA_PATH, QuarterlyColumns,
    YF_QUARTERLY_TABLE_NAME, STOCK_GENERAL_INFO_CSV, StockPupColumns, MONTH_TO_QUARTER)
from scripts.yahoo_finance_constants import (YF_QUARTERLY_TABLE_SCHEMA)

from finance_ml.utils.utils import get_ticker_symbols


def build_quarterly_database():
    today = datetime.now()

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        command = f'''CREATE TABLE IF NOT EXISTS {YF_QUARTERLY_TABLE_NAME} (
{YF_QUARTERLY_TABLE_SCHEMA},
    PRIMARY KEY ({QuarterlyColumns.TICKER_SYMBOL}, {QuarterlyColumns.QUARTER}, {QuarterlyColumns.YEAR})
);'''

        db_conn.execute(command)

        ticker_symbols = get_ticker_symbols() + ['']
        full_ticker_info = {}

        for ticker_symbol in ticker_symbols:
            ticker = yf.Ticker(ticker_symbol)

            if ticker.quarterly_balance_sheet.empty or (
                    today - ticker.quarterly_balance_sheet.columns[0] > timedelta(days=90)):
                continue

            ticker_info, ticker_df = get_quarterly_data(ticker)
            ticker_df = ticker_df.rename(
                columns={col: re.compile('[\W_]+').sub('', col) for col in ticker_df.columns})

            full_ticker_info[ticker_symbol] = ticker_info

            existing_cols = [col_info[1] for col_info in db_conn.execute(
                f"PRAGMA table_info({YF_QUARTERLY_TABLE_NAME})").fetchall()]

            new_cols = set(ticker_df.columns) - set(existing_cols)
            for new_col in new_cols:
                db_conn.execute(
                    f"ALTER TABLE {YF_QUARTERLY_TABLE_NAME} ADD COLUMN {new_col} INT")

            ticker_df.to_sql(name=YF_QUARTERLY_TABLE_NAME,
                             con=db_conn,
                             if_exists='append',
                             index=False)

        db_conn.commit()

        if os.path.exists(STOCK_GENERAL_INFO_CSV):
            os.remove(STOCK_GENERAL_INFO_CSV)

        general_df = pd.DataFrame({k: pd.Series(v) for k, v in full_ticker_info.items()})
        general_df = general_df.transpose()
        general_df.index.name = 'tickerSymbol'

        with open(STOCK_GENERAL_INFO_CSV, 'w') as f:
            f.write(general_df.to_csv())

    finally:
        if db_conn:
            db_conn.close()


def add_stockpup_data_to_db():
    data_files = glob.glob(os.path.join(DATA_PATH, 'stockpup_data', '*.csv'))

    db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

    for data_file_path in data_files:
        full_file_name = os.path.basename(data_file_path)
        file_name, ext = os.path.splitext(full_file_name)

        ticker_symbol = file_name.split('_')[-1]
        df = pd.read_csv(data_file_path)

        # First column is just row number
        df.drop(columns=[df.columns[0]], inplace=True)
        df[StockPupColumns.QUARTER_END] = pd.to_datetime(df[StockPupColumns.QUARTER_END])

        df[QuarterlyColumns.QUARTER] = df[StockPupColumns.QUARTER_END].apply(
            lambda r: MONTH_TO_QUARTER[r.month])
        df[QuarterlyColumns.YEAR] = df[StockPupColumns.QUARTER_END].apply(lambda r: r.year)
        df[QuarterlyColumns.TICKER_SYMBOL] = ticker_symbol
        df[QuarterlyColumns.DIVIDENDS] = df[StockPupColumns.DIVIDEND_PER_SHARE] * df[
            StockPupColumns.SHARES]
        df[QuarterlyColumns.DATE] = df[StockPupColumns.QUARTER_END].apply(lambda r: str(r.date()))
        df[QuarterlyColumns.OPERATING_INCOME] = df[StockPupColumns.FREE_CASH_FLOW_PER_SHARE] * df[
            StockPupColumns.SHARES]

        df.rename(columns={
            QuarterlyColumns.ASSETS: StockPupColumns.ASSETS,
            QuarterlyColumns.REVENUE: StockPupColumns.REVENUE,
            QuarterlyColumns.EARNINGS: StockPupColumns.EARNINGS,
            QuarterlyColumns.LIABILITIES: StockPupColumns.LIABILITIES,
            QuarterlyColumns.DEBT_LONG: StockPupColumns.LONG_TERM_DEBT,
            QuarterlyColumns.STOCKHOLDER_EQUITY: StockPupColumns.SHAREHOLDER_EQUITY,
            QuarterlyColumns.CASH: StockPupColumns.CASH_AT_END_OF_PERIOD,
            QuarterlyColumns.PRICE_AVG: StockPupColumns.PRICE,
            QuarterlyColumns.PRICE_LO: StockPupColumns.PRICE_LOW,
            QuarterlyColumns.PRICE_HI: StockPupColumns.PRICE_HIGH,
            QuarterlyColumns.SPLIT: StockPupColumns.SPLIT_FACTOR,
            QuarterlyColumns.COMMON_STOCK: StockPupColumns.SHARES_SPLIT_ADJUSTED
        }, inplace=True)

        # Filter only to columns in QuarterlyColumns
        df = df[[col for col in df.columns if col in [
            getattr(QuarterlyColumns, qc) for qc in dir(QuarterlyColumns) if qc[0] != '_']]]

        columns = ', '.join([c for c in df.columns])

        for row in df.iterrows():
            values = ', '.join(
                ["'" + row[1][col] + "'" if type(row[1][col]) is str else row[1][col]
                 for col in df.columns])

            command = f"""INSERT INTO yahoo_financial_data ({columns}) VALUES ({values})"""
            print(command)

            db_conn.execute(command)

    db_conn.commit()

    db_conn.close()
