from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from typing import List

from finance_ml.utils.constants import YOY_DELTA_PREFIX, Q_DELTA_PREFIX


@dataclass
class Hyperparams:
    MARKET_INDICES: List[str] = field(
        default_factory=lambda: ['^DJI'])  # , 'VTSAX', '^IXIC', '^GSPC', '^RUT', '^NYA']

    MODEL = LGBMRegressor

    N_STOCKS_TO_BUY: int = 7

    # Hyperparams for LGBM
    RANDOM_SEED: int = 32
    NUM_LEAVES: int = 31
    MAX_DEPTH: int = -1  # -1 is no max depth
    LEARNING_RATE: float = 0.1
    N_ESTIMATORS: int = 100
    EARLY_STOPPING_ROUNDS: int = 5

    # How many quarters back to use to train on
    NUM_QUARTERS_FOR_TRAINING: int = 12

    INCLUDE_DIVIDENDS_IN_PREDICTED_PRICE: bool = True

    # How many quarters out to predict market price
    N_QUARTERS_OUT_TO_PREDICT: int = 4

    PREDICTION_TARGET_PREFIX: str = YOY_DELTA_PREFIX
    assert PREDICTION_TARGET_PREFIX in [YOY_DELTA_PREFIX, Q_DELTA_PREFIX]

    EXTRACT_OUTLIERS: bool = False
    ONE_HOT_ENCODE: bool = False
    NUMERIC_ENCODE_CATEGORIES: bool = False
    SCALE_NUMERICS: bool = False
    TEST_SIZE: float = 0.2

    # Whether to adjust predicted appreciation based on appreciation since model data
    # being used for prediction was obtained
    ADJUST_FOR_CURRENT_PRICE: bool = True

    ALLOCATION_FN: str = 'equal'
