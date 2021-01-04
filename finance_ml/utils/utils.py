import pandas as pd
import sqlite3
from typing import Tuple

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, QuarterlyColumns, STOCKPUP_TABLE_NAME)


def get_ticker_symbols(source='DB'):
    if source == 'DB':
        try:
            db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
            command = f'''
    SELECT DISTINCT {QuarterlyColumns.TICKER_SYMBOL} FROM {STOCKPUP_TABLE_NAME}
    '''
            print(f"Executing command: {command}")
            result = db_conn.execute(command)
            tickers = [symbol[0] for symbol in result.fetchall()]
            tickers.extend(['SNOW', 'ABNB'])
            return tickers

        finally:
            if db_conn:
                db_conn.close()
    else:
        raise NotImplementedError


def split_feature_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split processed data into features and target.

    :param df: processed data for model training.
    :param target_col: target column name.
    :return: X: features for model training.
             y: targets for model training.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y
