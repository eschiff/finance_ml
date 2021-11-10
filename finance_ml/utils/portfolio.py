import pandas as pd
from typing import Dict
from collections import OrderedDict

from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.constants import (
    QuarterlyColumns as QC, Q_DELTA_PREFIX)


class Portfolio:
    """
    Class to track portfolio performance
    """

    def __init__(self,
                 df: pd.DataFrame,
                 start_quarter_idx: QuarterlyIndex,
                 allocation_fn: str = 'equal',
                 quarters_to_hold: int = 4,
                 starting_cash: float = 1000):
        """

        Args:
            df: Quarterly data dataframe
            start_quarter_idx: start quarter QuarterlyIndex
            allocation_fn: (str) indicating how to buy stocks each quarter. Currently supporing
                `equal` and `fibonacci`
            quarters_to_hold: how long to hold a share
            starting_cash: starting cash
        """
        self.df = df
        self.current_quarter_idx = start_quarter_idx
        self.quarters_to_hold = quarters_to_hold
        self.cash = starting_cash
        self.allocation_fn = allocation_fn.lower()
        self.age = 0  # num quarters portfolio has been around (num purchases made)

        # Portfolio is a Dict of { Q Index of Purchase : current value }
        self.portfolio: Dict[QuarterlyIndex, float] = {}

        self.performance: Dict[str, float] = OrderedDict(
            {self.current_quarter_idx.to_xQyyyy(): self.cash})

    def update(self, sell_all: bool = False):
        """
        Update portfolio values after a quarter has elapsed.
        Sells shares in the portfolio after `sell_after_num_quarters` have elapsed
        """
        self.current_quarter_idx = self.current_quarter_idx.time_travel(1)
        total_value = self.cash  # this will usually be 0

        current_portfolio_indexes = list(self.portfolio.keys())

        for idx in current_portfolio_indexes:
            value = self.portfolio[idx]
            # Update value of stocks held in portfolio
            new_quarter_idx = self.current_quarter_idx.copy(ticker=idx.ticker)
            try:
                change_in_price = self.df.loc[new_quarter_idx.to_tuple()][
                    f'{Q_DELTA_PREFIX}{QC.PRICE_AVG}']
                self.portfolio[idx] = value * (1 + change_in_price)
                total_value += value * (1 + change_in_price)
            except:
                print(f'Failed to find data for {new_quarter_idx}')
                self.portfolio[idx] = 0

            # Sell off stocks that we've held for the desired period
            if sell_all or idx.time_travel(self.quarters_to_hold) == new_quarter_idx:
                self.cash += self.portfolio.pop(idx)

        self.performance[self.current_quarter_idx.to_xQyyyy()] = round(total_value, 2)

    def purchase(self, n_largest: pd.Series):
        """
            Allocate cash amongst tickers in n_largest

            Args:
                n_largest: index is QuarterlyIndex columns, data is predicted appreciation
                    (irrelevant)
            """
        n = len(n_largest)
        pct_cash_to_allocate = 1.0 if self.age >= self.quarters_to_hold else \
            1.0 / (self.quarters_to_hold - self.age)
        cash_to_allocate = self.cash * pct_cash_to_allocate

        if self.allocation_fn == 'equal':
            for idx, row in n_largest.items():
                self.portfolio[QuarterlyIndex(*idx)] = cash_to_allocate / n

        elif self.allocation_fn == 'fibonacci':
            fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
                             2584, 4181, 6765, 10946][:n]
            fib_sum = sum(fibonacci_seq)
            i = n - 1
            for idx, row in n_largest.items():
                self.portfolio[QuarterlyIndex(*idx)] = cash_to_allocate * fibonacci_seq[i] / fib_sum
                i -= 1

        else:
            raise Exception("Unrecognized allocation function")

        self.cash -= cash_to_allocate
        self.age += 1
        print(f"Current portfolio: {self.portfolio}")
