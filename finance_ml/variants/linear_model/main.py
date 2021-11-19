from datetime import datetime, timedelta
import os
import pathlib
import pickle

from finance_ml.utils.constants import TARGET_COLUMN, QuarterlyColumns as QC
from finance_ml.utils.transforms import QuarterFilter

from finance_ml.variants.linear_model.preprocessing import preprocess_data
from finance_ml.variants.linear_model.hyperparams import Hyperparams
from finance_ml.variants.linear_model.metamodel import FinanceMLMetamodel

MODEL_DIR = os.path.join(
    pathlib.Path(__file__).parent.absolute(), os.pardir, os.pardir, os.pardir, 'models')


async def main(hyperparams: Hyperparams):
    print("Processing data")
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

    n_largest = y_pred.astype('float').reset_index().nlargest(
        hyperparams.N_STOCKS_TO_BUY, columns=[TARGET_COLUMN])

    print(f"Predicting Top {hyperparams.N_STOCKS_TO_BUY} stocks to purchase for"
          f" {hyperparams.N_QUARTERS_OUT_TO_PREDICT} Quarters in the future: \n")
    for i, ticker in enumerate(n_largest[QC.TICKER_SYMBOL]):
        print(f"\t{i + 1}. {ticker}")
