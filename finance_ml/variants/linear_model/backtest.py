from datetime import date, datetime, timedelta
import pandas as pd

from finance_ml.utils import Portfolio, QuarterlyIndex
from finance_ml.utils.constants import TARGET_COLUMN
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.utils.transforms import QuarterFilter


async def compute_performance(df: pd.DataFrame,
                              start_date: date,
                              hyperparams: Hyperparams,
                              end_date: date = None,
                              ) -> Portfolio:
    end_date = end_date if end_date is not None else (datetime.now() - timedelta(days=90)).date()
    end_q_index = QuarterlyIndex.from_date(end_date)

    portfolio = Portfolio(df,
                          start_quarter_idx=QuarterlyIndex.from_date(start_date),
                          allocation_fn='equal',
                          sell_after_n_quarters=hyperparams.N_QUARTERS_OUT_TO_PREDICT)

    while portfolio.current_quarter_idx < end_q_index:
        if portfolio.age > 0:
            portfolio.update()

        metamodel = FinanceMLMetamodel(hyperparams)
        # Train on data starting at the previous quarter
        metamodel.fit(df, current_qtr=portfolio.current_quarter_idx.time_travel(-1))

        date_filter = QuarterFilter(start_qtr=portfolio.current_quarter_idx,
                                    end_qtr=portfolio.current_quarter_idx.time_travel(1))
        prediction_candidate_df = date_filter.transform(df.drop(columns=[TARGET_COLUMN]))

        y_pred = await metamodel.predict(prediction_candidate_df, adjust_for_current_price=False)
        n_largest_df = y_pred.astype('float').nlargest(hyperparams.N_STOCKS_TO_BUY,
                                                       columns=[TARGET_COLUMN])

        portfolio.purchase(n_largest_df)

    portfolio.update(sell_all=True)

    return portfolio
