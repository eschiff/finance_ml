from dataclasses import dataclass
from datetime import datetime

from finance_ml.utils.constants import YOY_DELTA_PREFIX, Q_DELTA_PREFIX


@dataclass
class Hyperparams:
    MARKET_INDICES = ['^DJI']  # , 'VTSAX', '^IXIC', '^GSPC', '^RUT', '^NYA']

    # What year to start training data at
    START_DATE = datetime(2000, 1, 1)

    # How many quarters out to predict market price
    N_QUARTERS_OUT_TO_PREDICT = 4

    PREDICTION_TARGET_PREFIX = YOY_DELTA_PREFIX
    assert PREDICTION_TARGET_PREFIX in [YOY_DELTA_PREFIX, Q_DELTA_PREFIX]

    EXTRACT_OUTLIERS = True
    ONE_HOT_ENCODE = True
