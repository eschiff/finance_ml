import pandas as pd
from typing import Dict, List

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
                 sell_after_n_quarters: int = 4,
                 starting_cash: float = 1000):
        self.df = df
        self.current_quarter_idx = start_quarter_idx
        self.sell_after_n_quarters = sell_after_n_quarters
        self.cash = starting_cash
        self.allocation_fn = allocation_fn.lower()
        self.age = 0  # num quarters portfolio has been around

        # Portfolio is a Dict of { Q Index of Purchase : current value }
        self.portfolio: Dict[QuarterlyIndex, float] = {}

        self.performance: List[float] = []

    def update(self, sell_all: bool = False):
        """
        Update portfolio values after a quarter has elapsed.
        Sells shares in the portfolio after `sell_after_num_quarters` have elapsed
        """
        self.current_quarter_idx = self.current_quarter_idx.time_travel(1)
        self.age += 1
        total_value = self.cash  # this will usually be 0

        for idx, value in self.portfolio.items():
            # Update value of stocks held in portfolio
            new_quarter_idx = self.current_quarter_idx.copy(ticker=idx.ticker)
            try:
                change_in_price = self.df.loc[new_quarter_idx.to_tuple()][
                    f'{Q_DELTA_PREFIX}{QC.PRICE_AVG}']
            except:
                print(f'Failed to find data for {new_quarter_idx}')
                change_in_price = 0

            self.portfolio[idx] = value * (1 + change_in_price)
            total_value += value * (1 + change_in_price)

            # Sell off stocks that we've held for the desired period
            if sell_all or idx.time_travel(self.sell_after_n_quarters) == new_quarter_idx:
                self.cash += self.portfolio.pop(idx)

        self.performance.append(total_value)

    def purchase(self, n_largest_df: pd.DataFrame):
        """
            Allocate cash evenly amongst shares in n_largest_df

            Args:
                n_largest_df: index is QuarterlyIndex columns, data is predicted appreciation
                    (irrelevant)
            """
        n = n_largest_df.shape[0]
        pct_cash_to_allocate = 1.0 if self.age >= self.sell_after_n_quarters else \
            1.0 / self.sell_after_n_quarters
        cash_to_allocate = self.cash * pct_cash_to_allocate

        if self.allocation_fn == 'equal':
            for idx, row in n_largest_df.iterrows():
                self.portfolio[QuarterlyIndex(*idx)] = cash_to_allocate / n

        elif self.allocation_fn == 'fibonacci':
            fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
                             2584, 4181, 6765, 10946][:n]
            fib_sum = sum(fibonacci_seq)
            i = n - 1
            for idx, row in n_largest_df.iterrows():
                self.portfolio[QuarterlyIndex(*idx)] = cash_to_allocate * fibonacci_seq[i] / fib_sum
                i -= 1

        else:
            raise Exception("Unrecognized allocation function")

        self.cash -= cash_to_allocate
