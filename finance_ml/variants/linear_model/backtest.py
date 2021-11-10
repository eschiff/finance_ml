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
    """
    Trains a model with data before `start_date` and buys shares (and adds them to the portfolio)
    using rules defined in hyperparams. Each quarter, we sell shares in the portfolio,
    train a new model, and buy additional shares.
    Args:
        df:
        start_date:
        hyperparams:
        end_date:

    Returns:
        Final Portfolio object
    """
    end_date = end_date or (datetime.now() - timedelta(days=270)).date()
    end_q_index = QuarterlyIndex.from_date(end_date)

    portfolio = Portfolio(df,
                          start_quarter_idx=QuarterlyIndex.from_date(start_date),
                          allocation_fn=hyperparams.ALLOCATION_FN,
                          quarters_to_hold=hyperparams.N_QUARTERS_OUT_TO_PREDICT)

    while portfolio.current_quarter_idx < end_q_index:
        if portfolio.age > 0:
            portfolio.update()  # updates portfolio.current_quarter_idx

        metamodel = FinanceMLMetamodel(hyperparams)
        metamodel.fit(df.dropna(subset=[TARGET_COLUMN]),
                      current_qtr=portfolio.current_quarter_idx)

        qtr_filter = QuarterFilter(start_qtr=portfolio.current_quarter_idx,
                                   end_qtr=portfolio.current_quarter_idx.time_travel(1))
        prediction_candidate_df = qtr_filter.transform(df.drop(columns=[TARGET_COLUMN]))

        if prediction_candidate_df.empty:
            print(f"Prediction candidate DF empty. Stopping backtest")
            break

        y_pred = await metamodel.predict(prediction_candidate_df, adjust_for_current_price=False)
        n_largest = y_pred.astype('float').nlargest(hyperparams.N_STOCKS_TO_BUY,
                                                    columns=[TARGET_COLUMN])[TARGET_COLUMN]

        portfolio.purchase(n_largest)

    portfolio.update(sell_all=True)

    return portfolio
