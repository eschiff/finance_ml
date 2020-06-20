from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import yfinance as yf
from typing import Union, Tuple, Dict, List

from finance_ml.utils.constants import (
    QUARTERLY_DB_FILE_PATH, QuarterlyColumns, QUARTERLY_TABLE_NAME, AVG_REC_PREFIX)
from scripts.yahoo_finance_constants import (
    INFO_KEYS, FINANCIAL_KEYS, BALANCE_SHEET_KEYS, CASHFLOW_KEYS, RECOMMENDATION_GRADE_MAPPING)

MONTH_TO_QUARTER = {3: '1',
                    4: '1',
                    6: '2',
                    7: '2',
                    9: '3',
                    10: '3',
                    12: '4',
                    1: '4'}


def _get_ticker_symbols_from_db():
    try:
        db_conn = sqlite3.connect(QUARTERLY_DB_FILE_PATH)
        command = f'''
SELECT DISTINCT {QuarterlyColumns.TICKER_SYMBOL} FROM {QUARTERLY_TABLE_NAME}
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

    # Row Indexes are Quarter strings: '1Q2019'
    q_data = ticker.quarterly_earnings.copy()
    q_data['Year'] = q_data.index.to_series().apply(lambda i: i[-4])
    q_data['Quarter'] = q_data.index.to_series().apply(lambda i: i[0])
    q_data['Split'] = _build_split_data(ticker, quarter_end_dates)

    info = {feature: ticker.info[feature] for feature in INFO_KEYS}

    quarterly_balance_sheet = ticker.quarterly_balance_sheet.loc[
        [column for column in BALANCE_SHEET_KEYS]]
    financial_data = ticker.quarterly_financials.loc[
        [column for column in FINANCIAL_KEYS]]
    cashflow_data = ticker.quarterly_cashflow.loc[
        [column for column in CASHFLOW_KEYS]]
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
        avg_data['Quarter'] = avg_data[QuarterlyColumns.DATE].apply(
            lambda d: f'{MONTH_TO_QUARTER[d.month]}Q{d.year}')
        avg_data.drop(columns=[QuarterlyColumns.DATE], inplace=True)
        avg_data.set_index(['Quarter'], inplace=True)

        q_data.join(avg_data)

    q_data.reset_index(drop=True, inplace=True)

    return info, q_data


def _get_start_end_time_period(start, end, time_period):
    now = datetime.now().date()
    start = now - timedelta(days=time_period) if (start is None and time_period is not None) \
        else getattr(start, 'date', start)
    end = now if end is None else getattr(end, 'date', end)
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
        time_period: Union[int, None] = None) -> pd.DataFrame:
    """
    Get average recommendations by time period (from -1 (negative) to 1 (positive)
    Args:
        ticker: (yf.Ticker) ticker object
        start: (datetime) start date. Defaults to 1 time_period ago.
        end: (datetime) end date. Defaults to today.
        time_period: (int) # days to average data over. Defaults to # days between start and end
            if not given

    Returns:
        DataFrame with columns for Average Recommendation Grades by Firm, Date
    """
    start, end, time_period = _get_start_end_time_period(start, end, time_period)

    output = pd.DataFrame()

    period_start = start
    period_end = period_start + timedelta(days=time_period)

    while period_end < end:
        recs = ticker.recommendations[
            (ticker.recommendations.index >= datetime(start.year, start.month, start.day)) & (
                    ticker.recommendations.index < datetime(end.year, end.month, end.day))]

        recs['Grade'] = recs[yf.RecommendationColumns.ToGrade].apply(
            lambda r: RECOMMENDATION_GRADE_MAPPING.get(r, 0))
        firm_avg_grades_df = recs[[yf.RecommendationColumns.Firm, 'Grade']].groupby(
            [yf.RecommendationColumns.Firm]).mean().transpose()
        firm_avg_grades_df.rename(
            columns={firm: f'{AVG_REC_PREFIX}{firm.replace(" ", "")}'
                     for firm in firm_avg_grades_df.columns},
            inplace=True)
        firm_avg_grades_df[QuarterlyColumns.DATE] = [period_end]

        output = pd.concat([output, firm_avg_grades_df]).reset_index(drop=True)

        period_start = period_end
        period_end = period_start + timedelta(days=time_period)

    return output
