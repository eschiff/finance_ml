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
    YF_QUARTERLY_TABLE_NAME, STOCK_GENERAL_INFO_CSV, StockPupColumns)
from scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS, RECOMMENDATION_GRADE_MAPPING,
    YF_QUARTERLY_TABLE_SCHEMA)


def get_ticker_symbols(source='DB'):
    if source == 'DB':
        try:
            db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
            command = f'''
    SELECT DISTINCT {QuarterlyColumns.TICKER_SYMBOL} FROM {STOCKPUP_TABLE_NAME}
    '''
            print(f"Executing command: {command}")
            result = db_conn.execute(command)
            return [symbol[0] for symbol in result.fetchall()]

        finally:
            if db_conn:
                db_conn.close()
    else:
        raise NotImplementedError