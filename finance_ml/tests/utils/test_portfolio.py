from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.portfolio import Portfolio
from finance_ml.utils.constants import TARGET_COLUMN, QUARTER, YEAR


def test_portfolio_hold_for_one_quarter(quarterly_df_with_predictions, growth_rate, tickers):
    n = len(tickers)
    starting_cash = 1000
    qtr, yr = 1, 2020
    portfolio = Portfolio(df=quarterly_df_with_predictions,
                          start_quarter_idx=QuarterlyIndex('', qtr, yr),
                          allocation_fn='equal',
                          quarters_to_hold=1,
                          starting_cash=starting_cash)

    portfolio.purchase(n_largest=quarterly_df_with_predictions[
        (quarterly_df_with_predictions.index.get_level_values(QUARTER) == qtr) &
        (quarterly_df_with_predictions.index.get_level_values(YEAR) == yr)][TARGET_COLUMN])

    assert portfolio.portfolio == {QuarterlyIndex('A', qtr, yr): starting_cash / n,
                                   QuarterlyIndex('AAPL', qtr, yr): starting_cash / n}

    portfolio.update()
    assert portfolio.portfolio == {}
    assert portfolio.cash == starting_cash * (1 + growth_rate)
    assert portfolio.performance == [starting_cash,
                                     starting_cash * (1 + growth_rate)]


def test_portfolio_hold_for_four_quarters(quarterly_df_with_predictions, growth_rate, tickers):
    n = len(tickers)
    starting_cash = 1000
    starting_cash_per_share = starting_cash / (4 * n)
    qtr, yr = 1, 2020
    portfolio = Portfolio(df=quarterly_df_with_predictions,
                          start_quarter_idx=QuarterlyIndex('', qtr, yr),
                          allocation_fn='equal',
                          quarters_to_hold=4,
                          starting_cash=starting_cash)

    # Verify that first purchase is correct
    n_largest = quarterly_df_with_predictions[
        (quarterly_df_with_predictions.index.get_level_values(QUARTER) == qtr) &
        (quarterly_df_with_predictions.index.get_level_values(YEAR) == yr)][TARGET_COLUMN]

    portfolio.purchase(n_largest)
    assert portfolio.portfolio == {QuarterlyIndex('A', qtr, yr): starting_cash_per_share,
                                   QuarterlyIndex('AAPL', qtr, yr): starting_cash_per_share}

    portfolio.update()
    assert portfolio.portfolio == {
        QuarterlyIndex('A', qtr, yr): (starting_cash_per_share) * (1 + growth_rate),
        QuarterlyIndex('AAPL', qtr, yr): (starting_cash_per_share) * (1 + growth_rate)}
    assert portfolio.cash == starting_cash * 3 / 4

    # Verify that second purchase updates value of and leaves existing shares in portfolio
    n_largest = quarterly_df_with_predictions[
        (quarterly_df_with_predictions.index.get_level_values(QUARTER) == qtr + 1) &
        (quarterly_df_with_predictions.index.get_level_values(YEAR) == yr)][TARGET_COLUMN]

    portfolio.purchase(n_largest)
    portfolio.update()
    assert portfolio.portfolio == {
        QuarterlyIndex('A', qtr, yr): (starting_cash_per_share) * ((1 + growth_rate) ** 2),
        QuarterlyIndex('A', qtr + 1, yr): (starting_cash_per_share) * (1 + growth_rate),
        QuarterlyIndex('AAPL', qtr, yr): (starting_cash_per_share) * ((1 + growth_rate) ** 2),
        QuarterlyIndex('AAPL', qtr + 1, yr): (starting_cash_per_share) * (1 + growth_rate)}
    assert portfolio.cash == starting_cash * 1 / 2
    assert portfolio.performance == [starting_cash,
                                     starting_cash * (3 / 4) +
                                     n * starting_cash_per_share * (1 + growth_rate),
                                     starting_cash * (1 / 2) +
                                     n * starting_cash_per_share * (1 + growth_rate) +
                                     n * starting_cash_per_share * (1 + growth_rate) ** 2
                                     ]
