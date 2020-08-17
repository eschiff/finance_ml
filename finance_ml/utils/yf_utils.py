from datetime import datetime, timedelta
import yfinance_ez as yf
from typing import Union, List, Tuple
import pandas as pd
import json


from finance_ml.utils.constants import QuarterlyColumns, MONTH_TO_QUARTER


def date_to_xQyyyy(dt: datetime.date):
    q = MONTH_TO_QUARTER[dt.month]
    year = dt.year - 1 if dt.month == 1 else dt.year
    return f'{q}Q{year}'


def _get_start_end_time_period(start: Union[datetime, None], 
                               end: Union[datetime, None], 
                               time_period: int):
    now = datetime.now().date()
    start = now - timedelta(days=time_period) if (start is None and time_period is not None) \
        else start.date() if hasattr(start, 'date') else start
    end = now if end is None else end.date() if hasattr(end, 'date') else end
    time_period = (end - start).days if time_period is None else time_period
    return (start, end, time_period)


def get_average_price_over_time_period(ticker: yf.Ticker,
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
                    ticker.history.index < datetime(period_end.year,
                                                    period_end.month,
                                                    period_end.day))]

        period_data = pd.DataFrame({
            QuarterlyColumns.TICKER_SYMBOL: [ticker.ticker],
            QuarterlyColumns.DATE: [period_end],
            QuarterlyColumns.PRICE_AVG: [period_data.Close.mean()],
            QuarterlyColumns.PRICE_LO: [period_data.Close.min()],
            QuarterlyColumns.PRICE_HI: [period_data.Close.max()],
            QuarterlyColumns.PRICE_AT_END_OF_QUARTER: [period_data.Close[-1]],
            QuarterlyColumns.VOLUME: [period_data.Volume[-1]]
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


def build_split_data(ticker, quarter_end_dates) -> List[Union[float, None]]:
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