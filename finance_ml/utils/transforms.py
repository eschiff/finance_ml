import numpy as np
import pandas as pd

from datetime import date
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from sklearn.compose import ColumnTransformer

from finance_ml.utils.quarterly_index import QuarterlyIndex
from finance_ml.utils.constants import TARGET_COLUMN, INDEX_COLUMNS, QuarterlyColumns
from finance_ml.utils.utils import split_feature_target


class OutlierExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns, num_std_threshold=3):
        self.columns = columns
        self.threshold = num_std_threshold

    def fit(self, X):
        return self

    def transform(self, X):
        outlier_filter = np.abs(stats.zscore(X[self.columns]))
        np.nan_to_num(outlier_filter, 0)
        X_new = X[(outlier_filter < self.threshold).all(axis=1)]
        print(f'OutlierExtractor removed {X.shape[0] - X_new.shape[0]} rows')
        print(f"OutlierExtractor output size: {X_new.shape}")
        return X_new


class ColumnFilter(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], dropna=True):
        """
        This transform keeps only columns in 'columns' and can drop NA's
        Args:
            columns: list of columns to keep
            dropna: (bool) whether to drop NA's
        """
        self.columns = columns
        self.dropna = dropna

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = X.copy()[self.columns]
        if self.dropna:
            X_new.dropna(subset=self.columns, inplace=True)
        print(f"Column Filter output size: {X_new.shape}")
        return X_new


class CategoricalToDummy(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_columns, drop_one=True, drop_original=True):
        """
        Transforms categorical columns to one-hot-encoded columns named as such:
            '<original column name>_<category name>'

        Args
            categorical_columns: List[str] of categorical columns in X
            drop_one: (bool) when converting to categorical columns, whether to drop the
                first category name. Eg. if a column has categories ['A', 'B'],
                drop category 'A'. (since 'B' == 1 or 0 is sufficient)
            drop_original: (bool) when transforming categorical columns in X,
                whether to drop the original columns
        """
        self.categorical_columns = categorical_columns
        self.drop_one = drop_one
        self.drop_original = drop_original

    def fit(self, X, y=None):
        X_temp = X.copy()
        self.dummy_values = {}
        self.base_category_by_col = {}

        for col in self.categorical_columns:
            self.dummy_values[col] = X_temp[col].unique()
            if self.drop_one:
                base_temp = [x for x in self.dummy_values[col] if str(x) != 'nan']
                self.base_category_by_col[col] = base_temp[0]

        return self

    def transform(self, X, y=None):
        X_temp = X.copy()

        for col in self.categorical_columns:
            for cat in self.dummy_values[col]:
                if str(cat) == 'nan':
                    X_temp[str(col) + '_' + str(cat)] = X_temp[col].isnull().astype(int)
                else:
                    X_temp[str(col) + '_' + str(cat)] = (X_temp[col] == cat).astype(int)
            if self.drop_one:
                X_temp.drop(labels=[str(col) + '_' + str(self.base_category_by_col[col])], axis=1,
                            inplace=True)

        if self.drop_original:
            X_temp.drop(labels=self.categorical_columns, axis=1, inplace=True)

        print(f"Categorical Transform output size: {X_temp.shape}")

        return X_temp


class CategoricalToNumeric(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_columns):
        """
        Transforms categorical columns to one-hot-encoded columns named as such:
            '<original column name>_<category name>'

        Args
            categorical_columns: List[str] of categorical columns in X
        """
        self.categorical_columns = categorical_columns
        self.category_codes = {}  # {column: {cat: code}}

    def fit(self, X, y=None):
        for col in self.categorical_columns:
            self.category_codes[col] = X[col].cat.codes.to_dict()

        return self

    def transform(self, X, y=None):
        X_temp = X.copy()

        for col in self.categorical_columns:
            X_temp[col] = X[col].cat.codes

        print(f"Categorical To Numeric Transform output size: {X_temp.shape}")

        return X_temp


class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List, **kwargs):
        self.columns = columns
        self.kwargs = kwargs

    def fit(self, X):
        self.column_transformer = ColumnTransformer(transformers=[
            ('scaler', StandardScaler(), self.columns)
        ])
        self.column_transformer.fit_transform(X[self.columns])
        return self

    def transform(self, X):
        X_temp = X.copy()
        X_temp[self.columns] = self.column_transformer.transform(X)
        print(f"Scaler output size: {X_temp.shape}")
        return X_temp


class Splitter(BaseEstimator, TransformerMixin):
    def __init__(self, test_size=0.2, target_col=TARGET_COLUMN):
        """
        Transforms dataset into X_train, y_train, X_test, y_test
        Args:
            test_size:
            target_col:
        """

        self.test_size = test_size
        self.target_col = target_col

    def fit(self, X):
        return self

    def transform(self, X):
        n_test = int(len(X) * self.test_size)
        X_test, y_test = split_feature_target(
            X.sample(n=n_test, random_state=7), self.target_col)

        X_train, y_train = split_feature_target(
            X.loc[~X.index.isin(X_test.index)], self.target_col)

        return X_train, y_train, X_test, y_test


class IndexSwitchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, index_columns=INDEX_COLUMNS):
        """
        Stores the indexes of the dataset upon fitting
        Transforming resets the index if the index is set - or reapplies the index if it isn't.
        Args:
            index_columns: columns making up the index columns
        """
        self.index_columns = index_columns
        self.index_df = None

    def fit(self, X):
        if self.index_df is None:
            X_temp = X.reset_index()
            self.index_df = X_temp[self.index_columns]
        return self

    def transform(self, X):
        if all([col in self.index_columns for col in X.index.names]):
            X_new = X.reset_index()
        else:
            X_new = X.join(self.index_df, how='left')
            X_new = X_new.set_index(self.index_columns)
        return X_new


class DateFilter(BaseEstimator, TransformerMixin):
    def __init__(self, start_date: date, end_date: date, date_column: str = QuarterlyColumns.DATE):
        """
        Filters dates to be within a date window.
        Args:
            start_date: dates must start here
            end_date: dates must end before this date
            date_column: date column name
        """
        self.start_date = start_date
        self.end_date = end_date
        self.date_column = date_column

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = X[(pd.to_datetime(X[self.date_column]) >= self.start_date) & (
                pd.to_datetime(X[self.date_column]) < self.end_date)]
        print(f'DateFilter removed {X.shape[0] - X_new.shape[0]} rows')
        print(f"DateFilter output size: {X_new.shape}")
        return X_new


class QuarterFilter(BaseEstimator, TransformerMixin):
    def __init__(self, start_date: date, end_date: date):
        start_quarter = QuarterlyIndex.from_date(start_date)
        end_quarter = QuarterlyIndex.from_date(end_date)
        self.q_start = start_quarter.quarter
        self.y_start = start_quarter.year
        self.q_end = end_quarter.quarter
        self.y_end = end_quarter.year

    def fit(self, X):
        return self

    def transform(self, X):
        X_new = X[(
            (X.index.get_level_values(QuarterlyColumns.YEAR) > self.y_start) | (
                (X.index.get_level_values(QuarterlyColumns.QUARTER) >= self.q_start) & (
                    X.index.get_level_values(QuarterlyColumns.YEAR) == self.y_start)
                )
        ) & (
            (X.index.get_level_values(QuarterlyColumns.YEAR) < self.y_end) | (
                (X.index.get_level_values(QuarterlyColumns.QUARTER) < self.q_end) & (
                X.index.get_level_values(QuarterlyColumns.YEAR) == self.y_end)
            )
        )]

        print(f'QuarterFilter removed {X.shape[0] - X_new.shape[0]} rows')
        print(f"QuarterFilter output size: {X_new.shape}")
        return X_new
