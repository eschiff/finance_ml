from datetime import datetime, timedelta
import os
import pandas as pd
import pickle

from finance_ml.utils.constants import (TARGET_COLUMN, QuarterlyColumns as QC)
from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.transforms import QuarterFilter

from finance_ml.variants.linear_model.config import FEATURE_COLUMNS
from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'models')


async def main(hyperparams: Hyperparams):
    df = preprocess_data(hyperparams)

    print(f"data preprocessed: {df.shape}")

    columns_to_drop = list(set(df.columns).difference({*FEATURE_COLUMNS}))
    # need to keep price avg and dividends per share to compute target column
    if QC.DIVIDEND_PER_SHARE in columns_to_drop:
        columns_to_drop.remove(QC.DIVIDEND_PER_SHARE)
    if QC.PRICE_AVG in columns_to_drop:
        columns_to_drop.remove(QC.PRICE_AVG)
    df.drop(columns=columns_to_drop, inplace=True)

    df[TARGET_COLUMN] = df.apply(_create_target_column, axis=1,
                                 df=df, hyperparams=hyperparams)

    # Columns were used in getting Predicted Appreciation, but are no longer needed
    df.drop(columns=[QC.DIVIDEND_PER_SHARE], inplace=True)

    # Get dataframe of rows to make predictions on (most recent rows)
    # We want to keep PriceAvg in the dataframe for adjustment
    prediction_candidate_df = df[df[TARGET_COLUMN].isnull()].drop(columns=[TARGET_COLUMN])

    df.drop(columns=[QC.PRICE_AVG], inplace=True)

    # Drop NA's in target column for prediction purposes
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    metamodel = FinanceMLMetamodel(hyperparams)
    metamodel.fit(df)

    model_name = f'linear-model-{datetime.now().date()}.pickle'
    model_path = os.path.join(MODEL_DIR, model_name)
    print(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(metamodel, file=f)

    # Filter prediction DF to recent dates only, then transform with Pipeline
    date_filter = QuarterFilter(start_date=datetime.now().date() - timedelta(days=90),
                                end_date=datetime.now().date())
    prediction_candidate_df = date_filter.transform(prediction_candidate_df)

    # Make Prediction
    y_pred = await metamodel.predict(prediction_candidate_df,
                                     hyperparams.ADJUST_FOR_CURRENT_PRICE)

    n_largest = y_pred.astype('float').nlargest(hyperparams.N_STOCKS_TO_BUY,
                                                columns=[TARGET_COLUMN])

    print(f"Predicting Top {len(n_largest)} stocks to purchase for"
          f" {hyperparams.N_QUARTERS_OUT_TO_PREDICT} Quarters in the future: \n{n_largest}")

    return y_pred, df


def _create_target_column(row: pd.Series, df: pd.DataFrame, hyperparams: Hyperparams):
    try:
        current_price = row[QC.PRICE_AVG]
        dividends = 0

        if hyperparams.INCLUDE_DIVIDENDS_IN_PREDICTED_PRICE:
            for i in range(hyperparams.N_QUARTERS_OUT_TO_PREDICT):
                target_prediction_index = QuarterlyIndex(*row.name).time_travel(i).to_tuple()

                dividends += df.loc[target_prediction_index][QC.DIVIDEND_PER_SHARE] or 0

        target_prediction_index = QuarterlyIndex(*row.name).time_travel(
            hyperparams.N_QUARTERS_OUT_TO_PREDICT).to_tuple()

        prediction_data = df.loc[target_prediction_index]

        return prediction_data[f'{hyperparams.PREDICTION_TARGET_PREFIX}{QC.PRICE_AVG}'] + (
                dividends / current_price)
    except:
        return None
