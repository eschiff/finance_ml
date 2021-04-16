from datetime import date, datetime, timedelta
import pandas as pd

from finance_ml.utils import Portfolio, QuarterlyIndex
from finance_ml.utils.constants import TARGET_COLUMN
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel
from finance_ml.variants.linear_model.hyperparams import Hyperparams


async def compute_performance(df: pd.DataFrame,
                              start_date: date,
                              hyperparams: Hyperparams,
                              end_date: date = None,
                              ) -> Portfolio:
    end_date = end_date if end_date is not None else (datetime.now() - timedelta(days=90)).date()
    end_q_index = QuarterlyIndex.from_date(end_date)
    sell_after_n_quarters = hyperparams.N_QUARTERS_OUT_TO_PREDICT

    portfolio = Portfolio(df,
                          start_quarter_idx=QuarterlyIndex.from_date(start_date),
                          allocation_fn='equal',
                          sell_after_n_quarters=sell_after_n_quarters)

    while portfolio.current_quarter_idx < end_q_index:
        if portfolio.age > 0:
            portfolio.update()

        metamodel: FinanceMLMetamodel =  # Train metamodel
        y_pred = await metamodel.predict(prediction_candidate_df, adjust_for_current_price=False)
        n_largest_df = y_pred.astype('float').nlargest(hyperparams.N_STOCKS_TO_BUY,
                                                       columns=[TARGET_COLUMN])

        portfolio.purchase(n_largest_df)

    portfolio.update(sell_all=True)

    return portfolio
