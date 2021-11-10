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
    df, prediction_candidate_df = preprocess_data(hyperparams)
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"data preprocessed: {df.shape}")

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
