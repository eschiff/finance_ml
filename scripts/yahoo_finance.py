from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import yfinance as yf
from typing import Union, Tuple, Dict, List
import os
import re
import json

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, QuarterlyColumns, STOCKPUP_TABLE_NAME, AVG_REC_PREFIX,
    YF_QUARTERLY_TABLE_NAME, STOCK_GENERAL_INFO_CSV)
from scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS, RECOMMENDATION_GRADE_MAPPING,
    YF_QUARTERLY_TABLE_SCHEMA)

MONTH_TO_QUARTER = {
    1: 4,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 2,
    8: 3,
    9: 3,
    10: 3,
    11: 4,
    12: 4
}


def _get_ticker_symbols_from_db():
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


def _build_split_data(ticker, quarter_end_dates) -> List[Union[float, None]]:
    """

    Args:
        ticker:
        quarter_end_dates: most recent data is first!

    Returns:
        List[Union[float, None]] of splits if they occurred within the quarter
    """
    split_dates = list(zip(ticker.splits.index, ticker.splits))  # Tuples of (date, split)
    split_date, split = split_dates.pop() if split_dates else (None, None)
    split_data = []
    for quarter_end_date in quarter_end_dates:
        if split is None:
            split_data.append(None)
            continue

        quarter_start_date = quarter_end_date - timedelta(days=13 * 7)

        if quarter_start_date < split_date.date() < quarter_end_date:
            split_data.append(split)
            continue

        while split_dates:
            split_date, split = split_dates.pop()  # Pops most recent split date
            if split_date.date() < quarter_end_date:
                break

        split_data.append(
            split if quarter_start_date < split_date.date() < quarter_end_date else None)

    return split_data


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
    q_indexes = [f'{MONTH_TO_QUARTER[date.month]}Q{date.year}' for date in quarter_end_dates]

    # Row Indexes are Quarter strings: '1Q2019'
    q_data = ticker.quarterly_earnings.copy()
    q_data.index = q_indexes  # Replacing indexes, since the original can contain duplicates...
    q_data[QuarterlyColumns.TICKER_SYMBOL] = [ticker.ticker] * len(quarter_end_dates)
    q_data[QuarterlyColumns.YEAR] = q_data.index.to_series().apply(lambda i: int(i[-4:]))
    q_data[QuarterlyColumns.QUARTER] = q_data.index.to_series().apply(lambda i: int(i[0]))
    q_data[QuarterlyColumns.SPLIT] = _build_split_data(ticker, quarter_end_dates)

    info = {feature: ticker.info.get(feature, '') for feature in INFO_KEYS}

    quarterly_balance_sheet = ticker.quarterly_balance_sheet.loc[
        [key for key in BALANCE_SHEET_KEYS if key in ticker.quarterly_balance_sheet.index]]
    financial_data = ticker.quarterly_financials.loc[
        [key for key in FINANCIAL_KEYS if key in ticker.quarterly_financials.index]]
    cashflow_data = ticker.quarterly_cashflow.loc[
        [key for key in CASHFLOW_KEYS if key in ticker.quarterly_cashflow.index]]
    combined_data = pd.concat([quarterly_balance_sheet, financial_data, cashflow_data])
    combined_data = combined_data.rename(
        columns={date: f'{MONTH_TO_QUARTER[date.month]}Q{date.year}'
                 for date in combined_data.columns}).transpose()

    q_data = q_data.join(combined_data)

    # Add Average stock price data
    for fn in (get_average_over_time_period, get_average_recommendations_over_time_period):
        avg_data = fn(ticker=ticker,
                      start=quarter_end_dates[-1] - timedelta(days=13 * 7),
                      time_period=13 * 7)

        if avg_data.empty:
            continue

        avg_data['Quarter'] = avg_data[QuarterlyColumns.DATE].apply(
            lambda d: f'{MONTH_TO_QUARTER[d.month]}Q{d.year}')
        avg_data.drop(columns=[QuarterlyColumns.DATE, QuarterlyColumns.TICKER_SYMBOL],
                      inplace=True)
        avg_data.set_index(['Quarter'], inplace=True)

        q_data = q_data.join(avg_data)

    q_data.reset_index(drop=True, inplace=True)
    q_data[QuarterlyColumns.DATE] = quarter_end_dates

    # Remove duplicate columns
    q_data = q_data.loc[:, ~q_data.columns.duplicated()]

    return info, q_data


def _get_start_end_time_period(start, end, time_period):
    now = datetime.now().date()
    start = now - timedelta(days=time_period) if (start is None and time_period is not None) \
        else start.date() if hasattr(start, 'date') else start
    end = now if end is None else end.date() if hasattr(end, 'date') else end
    time_period = (end - start).days if time_period is None else time_period
    return (start, end, time_period)


def get_average_over_time_period(ticker: yf.Ticker,
                                 start: datetime = None,
                                 end: datetime = None,
                                 time_period: Union[int, None] = None) -> pd.DataFrame:
    """
    Compute Average data over a time period
    Args:
        ticker: (yf.Ticker) ticker object
        start: (datetime) start date. Defaults to 1 time_period ago.
        end: (datetime) end date. Defaults to today.
        time_period: (int) # days to average data over. Defaults to # days between start and end
            if not given

    Returns:
        DataFrame with columns for Ticker Symbol, Date, Avg Price, Lo Price, Hi Price,
        End of Quarter Price, and shares
    """
    start, end, time_period = _get_start_end_time_period(start, end, time_period)

    ticker.get_history(start=start, end=end)

    output = pd.DataFrame()

    period_start = start
    period_end = period_start + timedelta(days=time_period)

    while period_end < end:
        period_data = ticker.history.loc[
            (ticker.history.index >= datetime(
                period_start.year, period_start.month, period_start.day)) & (
                    ticker.history.index < datetime(end.year, end.month, end.day))]

        period_data = pd.DataFrame({
            QuarterlyColumns.TICKER_SYMBOL: [ticker.ticker],
            QuarterlyColumns.DATE: [period_end],
            QuarterlyColumns.PRICE_AVG: [period_data.Close.mean()],
            QuarterlyColumns.PRICE_LO: [period_data.Close.min()],
            QuarterlyColumns.PRICE_HI: [period_data.Close.max()],
            QuarterlyColumns.PRICE_AT_END_OF_QUARTER: [period_data.Close[-1]],
            QuarterlyColumns.SHARES: [period_data.Volume[-1]]
        })

        output = pd.concat([output, period_data]).reset_index(drop=True)

        period_start = period_end
        period_end = period_start + timedelta(days=time_period)

    return output


def get_average_recommendations_over_time_period(
        ticker: yf.Ticker,
        start: datetime = None,
        end: datetime = None,
        time_period: Union[int, None] = None,
        collapse_ratings_to_single_column: bool = True) -> pd.DataFrame:
    """
    Get average recommendations by time period (from -1 (negative) to 1 (positive)
    Args:
        ticker: (yf.Ticker) ticker object
        start: (datetime) start date. Defaults to 1 time_period ago.
        end: (datetime) end date. Defaults to today.
        time_period: (int) # days to average data over. Defaults to # days between start and end
            if not given
        collapse_ratings_to_single_column: (bool) Whether to collapse ratings to a single column

    Returns:
        DataFrame with column(s) for Average Recommendation Grades by Firm, Date, Ticker
    """
    output = pd.DataFrame()
    if ticker.recommendations.empty:
        return output

    start, end, time_period = _get_start_end_time_period(start, end, time_period)

    period_start = start
    period_end = period_start + timedelta(days=time_period)

    while period_end <= end:
        recs = ticker.recommendations[
            (ticker.recommendations.index >= datetime(start.year, start.month, start.day)) & (
                    ticker.recommendations.index < datetime(end.year, end.month, end.day))]

        if not recs.empty:
            recs['Grade'] = recs[yf.RecommendationColumns.ToGrade].apply(
                lambda r: RECOMMENDATION_GRADE_MAPPING.get(r, 0))
            firm_avg_grades_df = recs[[yf.RecommendationColumns.Firm, 'Grade']].groupby(
                [yf.RecommendationColumns.Firm]).mean().transpose()
            firm_avg_grades_df.rename(
                columns={firm: f'{firm.replace(" ", "")}' for firm in firm_avg_grades_df.columns},
                inplace=True)
            firm_avg_grades_df[QuarterlyColumns.DATE] = [period_end]
            firm_avg_grades_df[QuarterlyColumns.TICKER_SYMBOL] = [ticker.ticker]

            if collapse_ratings_to_single_column:
                ratings_json = json.dumps(
                    firm_avg_grades_df[firm_avg_grades_df.columns.difference(
                        [QuarterlyColumns.DATE, QuarterlyColumns.TICKER_SYMBOL]
                    )].transpose().to_dict()['Grade'])
                firm_avg_grades_df = firm_avg_grades_df[
                    [QuarterlyColumns.DATE, QuarterlyColumns.TICKER_SYMBOL]]
                firm_avg_grades_df[QuarterlyColumns.AVG_RECOMMENDATIONS] = [ratings_json]

            output = pd.concat([output, firm_avg_grades_df]).reset_index(drop=True)

        period_start = period_end
        period_end = period_start + timedelta(days=time_period)

    return output


def build_quarterly_database():
    today = datetime.now()

    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)

        command = f'''CREATE TABLE IF NOT EXISTS {YF_QUARTERLY_TABLE_NAME} (
{YF_QUARTERLY_TABLE_SCHEMA},
    PRIMARY KEY ({QuarterlyColumns.TICKER_SYMBOL}, {QuarterlyColumns.QUARTER}, {QuarterlyColumns.YEAR})
);'''

        db_conn.execute(command)

        ticker_symbols = _get_ticker_symbols_from_db() + ['']
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

    return full_ticker_info
