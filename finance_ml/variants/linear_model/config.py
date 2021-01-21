from finance_ml.utils.constants import (
    NUMERIC_COLUMNS, FORMULAE, Q_DELTA_PREFIX, YOY_DELTA_PREFIX,
    COLUMNS_TO_COMPARE_TO_MARKET_INDICES, VS_MKT_IDX, QuarterlyColumns, CATEGORICAL_COLUMNS)
from finance_ml.variants.linear_model.hyperparams import Hyperparams

ALL_NUMERIC_COLUMNS = NUMERIC_COLUMNS + list(FORMULAE.keys())

VS_MARKET_INDICES_COLUMNS = [f"{col}{VS_MKT_IDX}{mkt_idx}"
                             for col in COLUMNS_TO_COMPARE_TO_MARKET_INDICES
                             for mkt_idx in Hyperparams.MARKET_INDICES]

# all feature columns to use in the model. This list is comprised of all delta numeric columns
# and all formulae, since those columns should be agnostic to the size of the company, plus
# a few other columns
FEATURE_COLUMNS = [f"{Q_DELTA_PREFIX}{col_name}"
                   for prefix in [Q_DELTA_PREFIX, YOY_DELTA_PREFIX]
                   for col_name in ALL_NUMERIC_COLUMNS] + \
                  list(FORMULAE.keys()) + \
                  VS_MARKET_INDICES_COLUMNS + \
                  CATEGORICAL_COLUMNS + \
                  [QuarterlyColumns.AVG_RECOMMENDATION_SCORE]
