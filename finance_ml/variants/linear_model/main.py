import pandas as pd

from finance_ml.utils.constants import (
    TARGET_COLUMN, QuarterlyColumns)
from finance_ml.utils.quarterly_index import QuarterlyIndex

from finance_ml.variants.linear_model.config import FEATURE_COLUMNS
from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel


def main(hyperparams: Hyperparams):
    df = preprocess_data(hyperparams)

    columns_to_drop = list(set(df.columns).difference({*FEATURE_COLUMNS}))
    df.drop(columns=columns_to_drop, inplace=True)

    df[TARGET_COLUMN] = df.apply(_get_target_col_prediction, axis=1,
                                 df=df, hyperparams=hyperparams)

    # Get dataframe of rows to make predictions on (most recent rows)
    prediction_candidate_df = df[df[TARGET_COLUMN].isnull()].drop(columns=[TARGET_COLUMN])
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    metamodel = FinanceMLMetamodel(hyperparams)

    metamodel.fit(df)

    y_pred = metamodel.predict(prediction_candidate_df)

    n_largest = y_pred.nlargest(hyperparams.N_STOCKS_TO_BUY, columns=[TARGET_COLUMN])

    print(f"Predicting Top {len(n_largest)} stocks to purchase for"
          f" {hyperparams.N_QUARTERS_OUT_TO_PREDICT} Quarters in the future: \n{n_largest}")


def _get_target_col_prediction(row: pd.Series, df: pd.DataFrame, hyperparams: Hyperparams):
    try:
        target_prediction_index = QuarterlyIndex(*row.name).time_travel(
            hyperparams.N_QUARTERS_OUT_TO_PREDICT).to_tuple()

        prediction_data = df.loc[target_prediction_index]
        return prediction_data[
            f'{hyperparams.PREDICTION_TARGET_PREFIX}{QuarterlyColumns.PRICE_AVG}']
    except:
        return None
