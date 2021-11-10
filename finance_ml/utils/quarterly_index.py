from datetime import datetime, date, timedelta

from finance_ml.utils.constants import MONTH_TO_QUARTER


class QuarterlyIndex:
    def __init__(self, ticker, quarter, year):
        assert isinstance(ticker, str)
        assert isinstance(quarter, int)
        assert isinstance(year, int)

        self.ticker = ticker
        self.quarter = quarter
        self.year = year

    def __repr__(self):
        return f"QuarterlyIndex {self.ticker} Q{self.quarter} {self.year}"

    def __lt__(self, qi):
        return self.quarter < qi.quarter if self.year == qi.year else self.year < qi.year

    def __gt__(self, qi):
        return self.quarter > qi.quarter if self.year == qi.year else self.year > qi.year

    def __le__(self, qi):
        return self.quarter <= qi.quarter if self.year == qi.year else self.year <= qi.year

    def __ge__(self, qi):
        return self.quarter >= qi.quarter if self.year == qi.year else self.year >= qi.year

    def __eq__(self, qi):
        return qi.ticker == self.ticker and qi.quarter == self.quarter and qi.year == self.year

    def __ne__(self, qi):
        return qi.ticker != self.ticker or qi.quarter != self.quarter or qi.year != self.year

    def __hash__(self):
        return self.to_tuple().__hash__()

    def time_travel(self, n: int):
        time_quarters = abs(n)

        num_years, num_quarters = divmod(time_quarters, 4)

        if n >= 0:
            new_quarter = self.quarter + num_quarters
            extra_year = 0
            if new_quarter > 4:
                extra_year, new_quarter = divmod(new_quarter, 4)
            new_year = self.year + num_years + extra_year
        else:
            new_quarter = self.quarter - num_quarters
            extra_year = 0
            if new_quarter < 1:
                new_quarter = 4 + new_quarter
                extra_year = 1
            new_year = self.year - num_years - extra_year

        return QuarterlyIndex(self.ticker, new_quarter, new_year)

    @classmethod
    def from_date(cls, dt: date):
        # shift date into the next month if it's at the end of a month
        new_date = dt + timedelta(days=7) if dt.day > 24 else dt

        q = MONTH_TO_QUARTER[new_date.month]

        year = new_date.year - 1 if new_date.month == 1 else new_date.year
        return cls('', q, year)

    def to_tuple(self):
        return self.ticker, self.quarter, self.year

    def to_date(self, day=1):
        return datetime(self.year, self.quarter * 3, day).date()

    def to_xQyyyy(self) -> str:
        return f"{self.quarter}Q{self.year}"

    def copy(self, ticker: str = ""):
        # allow creating a new QuarterlyIndex from  the existing one
        return QuarterlyIndex(ticker or self.ticker, self.quarter, self.year)
