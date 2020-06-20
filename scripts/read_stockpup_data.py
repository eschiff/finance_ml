import glob
import pandas as pd
import os
import sqlite3

from finance_ml.utils.constants import (
    StockPupColumns, QuarterlyColumns, DATA_PATH, QUARTERLY_DB_NAME,
    QUARTERLY_TABLE_NAME, QUARTERLY_DB_FILE_PATH)


def _(value):
    if value in ['', 'None']:
        return 'null'
    else:
        return str(value)


def _quarterly_data_column_map(row: pd.Series, ticker_symbol: str) -> dict:
    column_map = {
        QuarterlyColumns.TICKER_SYMBOL: f"'{ticker_symbol}'",
        QuarterlyColumns.DATE: f"'{_(row[StockPupColumns.QUARTER_END])}'",
        QuarterlyColumns.PRICE_AVG: _(row[StockPupColumns.PRICE]),
        QuarterlyColumns.PRICE_HI: _(row[StockPupColumns.PRICE_HIGH]),
        QuarterlyColumns.PRICE_LO: _(row[StockPupColumns.PRICE_LOW]),
        QuarterlyColumns.SHARES: _(row[StockPupColumns.SHARES]),
        QuarterlyColumns.ASSETS: _(row[StockPupColumns.ASSETS]),
        QuarterlyColumns.LIABILITIES: _(row[StockPupColumns.LIABILITIES]),
        QuarterlyColumns.LONG_TERM_DEBT: _(row[StockPupColumns.LONG_TERM_DEBT]),
        QuarterlyColumns.REVENUE: _(row[StockPupColumns.REVENUE]),
        QuarterlyColumns.EARNINGS: _(row[StockPupColumns.EARNINGS]),
        QuarterlyColumns.DIVIDENDS_PER_SHARE: _(row[StockPupColumns.DIVIDEND_PER_SHARE]),
        QuarterlyColumns.ASSET_TURNOVER: _(row[StockPupColumns.ASSET_TURNOVER]),
        QuarterlyColumns.EPS: _(row[StockPupColumns.EPS_BASIC]),
        QuarterlyColumns.P_E_RATIO: _(row[StockPupColumns.P_E_RATIO]),
        QuarterlyColumns.ROE: _(row[StockPupColumns.ROE]),
        QuarterlyColumns.ROA: _(row[StockPupColumns.ROA])
    }

    return column_map


def build_quarterly_database():
    data_files = glob.glob(os.path.join(DATA_PATH, 'stockpup_data', '*.csv'))

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        command = f'''
CREATE TABLE {QUARTERLY_TABLE_NAME} (
    {QuarterlyColumns.TICKER_SYMBOL} TEXT,
    {QuarterlyColumns.DATE} TEXT,
    {QuarterlyColumns.PRICE_AVG} NUMERIC,
    {QuarterlyColumns.PRICE_HI} NUMERIC,
    {QuarterlyColumns.PRICE_LO} NUMERIC,
    {QuarterlyColumns.SHARES} INT,
    {QuarterlyColumns.ASSETS} INT,
    {QuarterlyColumns.LIABILITIES} INT,
    {QuarterlyColumns.LONG_TERM_DEBT} INT,
    {QuarterlyColumns.REVENUE} INT,
    {QuarterlyColumns.EARNINGS} INT,
    {QuarterlyColumns.DIVIDENDS_PER_SHARE} NUMERIC,
    {QuarterlyColumns.ASSET_TURNOVER} INT,
    {QuarterlyColumns.EPS} NUMERIC,
    {QuarterlyColumns.P_E_RATIO} NUMERIC,
    {QuarterlyColumns.ROE} NUMERIC,
    {QuarterlyColumns.ROA} NUMERIC,
    PRIMARY KEY ({QuarterlyColumns.TICKER_SYMBOL}, {QuarterlyColumns.DATE})
);'''

        print(f"Executing command: {command}")
        db_conn.execute(command)

        for data_file_path in data_files:
            full_file_name = os.path.basename(data_file_path)
            file_name, ext = os.path.splitext(full_file_name)

            ticker_symbol = file_name.split('_')[-1]
            print(f"Found ticker symbol: {ticker_symbol}")

            df = pd.read_csv(data_file_path)

            # First column is just row number
            df.drop(columns=[df.columns[0]], inplace=True)

            df[StockPupColumns.QUARTER_END] = pd.to_datetime(df[StockPupColumns.QUARTER_END])

            for idx, row in df.iterrows():
                column_map = _quarterly_data_column_map(row, ticker_symbol)

                command = f'''
INSERT INTO {QUARTERLY_TABLE_NAME} ({', '.join(column_map.keys())})
VALUES({', '.join(column_map.values())}
);'''
                print(f"Executing command: {command}")
                db_conn.execute(command)

                db_conn.commit()

    finally:
        if db_conn:
            db_conn.close()
