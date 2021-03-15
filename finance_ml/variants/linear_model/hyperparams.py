from dataclasses import dataclass
from datetime import datetime
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from finance_ml.utils.constants import YOY_DELTA_PREFIX, Q_DELTA_PREFIX


@dataclass
class Hyperparams:
    MARKET_INDICES = ['^DJI']  # , 'VTSAX', '^IXIC', '^GSPC', '^RUT', '^NYA']

    MODEL = LGBMRegressor

    N_STOCKS_TO_BUY = 10

    # Hyperparams for LGBM
    RANDOM_SEED = 32
    NUM_LEAVES = 31
    MAX_DEPTH = -1  # -1 is no max depth
    LEARNING_RATE = 0.1
    N_ESTIMATORS = 100
    EARLY_STOPPING_ROUNDS = 5

    # What year to start training data at
    START_DATE = datetime(2000, 1, 1)
    # end date to use for training data. None if today
    END_DATE = None

    # How many quarters out to predict market price
    N_QUARTERS_OUT_TO_PREDICT = 4

    PREDICTION_TARGET_PREFIX = YOY_DELTA_PREFIX
    assert PREDICTION_TARGET_PREFIX in [YOY_DELTA_PREFIX, Q_DELTA_PREFIX]

    EXTRACT_OUTLIERS = False
    ONE_HOT_ENCODE = False
    NUMERIC_ENCODE_CATEGORIES = False
    SCALE_NUMERICS = False
    TEST_SIZE = 0.2

    # Whether to adjust predicted appreciation based on appreciation since model data
    # being used for prediction was obtained
    ADJUST_FOR_CURRENT_PRICE = True
