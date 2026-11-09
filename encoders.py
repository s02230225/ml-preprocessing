import numpy as np
import pandas as pd

_MISSING = object()


def _key(value):
    return _MISSING if pd.isna(value) else value


def _frame(X):
    X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if X.shape[1] == 0 or not X.columns.is_unique:
        raise ValueError("Expected at least one uniquely named feature")
    return X


def _target(Y, size):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 1 or len(Y) != size or size == 0 or not np.isfinite(Y).all():
        raise ValueError("Expected a finite target for each training row")
    return Y


class Preprocessor:
    def fit(self, X, Y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, Y=None):
        return self.fit(X, Y).transform(X)


class MyOneHotEncoder(Preprocessor):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y=None):
        X = _frame(X)
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        self.columns_ = X.columns.copy()
        self.categories_ = []
        for col in X:
            values = list(pd.unique(X[col]))
            try:
                values.sort()
            except TypeError:
                pass
            self.categories_.append(values)
        return self

    def transform(self, X):
        if not hasattr(self, 'categories_'):
            raise ValueError("Call fit before transform")
        X = _frame(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("Columns differ from training data")
        columns = []
        for col, values in zip(X, self.categories_):
            for value in values:
                mask = X[col].isna() if pd.isna(value) else X[col].eq(value)
                columns.append(mask.to_numpy(dtype=self.dtype))
        return np.column_stack(columns).astype(self.dtype, copy=False)

    def get_params(self, deep=True):
        return {'dtype': self.dtype}


class SimpleCounterEncoder(Preprocessor):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        X = _frame(X)
        Y = _target(Y, len(X))
        self.columns_ = X.columns.copy()
        self.success_rates, self.frequencies = [], []
        for col in X:
            totals, counts = {}, {}
            for value, target in zip(X[col], Y):
                key = _key(value)
                totals[key] = totals.get(key, 0.0) + target
                counts[key] = counts.get(key, 0) + 1
            self.success_rates.append({k: totals[k] / count for k, count in counts.items()})
            self.frequencies.append({k: count / len(X) for k, count in counts.items()})
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        if not hasattr(self, 'columns_'):
            raise ValueError("Call fit before transform")
        X = _frame(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("Columns differ from training data")
        if a < 0 or b <= 0 or not np.isfinite([a, b]).all():
            raise ValueError("Expected finite a >= 0 and b > 0")
        result = np.zeros((len(X), 3 * X.shape[1]), dtype=self.dtype)
        for i, col in enumerate(X):
            for row, value in enumerate(X[col]):
                key = _key(value)
                success = self.success_rates[i].get(key, 0.0)
                frequency = self.frequencies[i].get(key, 0.0)
                result[row, 3*i:3*i+3] = success, frequency, (success+a)/(frequency+b)
        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        return self.fit(X, Y).transform(X, a, b)

    def get_params(self, deep=True):
        return {'dtype': self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    if not 2 <= n_splits <= size:
        raise ValueError("Expected 2 <= n_splits <= size")
    indices = np.random.RandomState(seed).permutation(size)
    for valid in np.array_split(indices, n_splits):
        train = indices[~np.isin(indices, valid)]
        yield valid, train


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.n_folds = n_folds
        self.dtype = dtype

    def fit(self, X, Y, seed=1):
        X = _frame(X)
        Y = _target(Y, len(X))
        self.folds_info = list(group_k_fold(len(X), self.n_folds, seed))
        self.encoder_ = SimpleCounterEncoder(self.dtype).fit(X, Y)
        return self

    def transform(self, X, a=1e-5, b=1e-5):
        if not hasattr(self, 'encoder_'):
            raise ValueError("Call fit before transform")
        return self.encoder_.transform(X, a, b)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5, seed=1):
        X = _frame(X)
        Y = _target(Y, len(X))
        self.fit(X, Y, seed)
        result = np.zeros((len(X), X.shape[1] * 3), dtype=self.dtype)
        for valid, train in self.folds_info:
            encoder = SimpleCounterEncoder(self.dtype).fit(X.iloc[train], Y[train])
            result[valid] = encoder.transform(X.iloc[valid], a, b)
        return result


def weights(x, y):
    x, y = np.asarray(x), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected matching one-dimensional arrays")
    return np.array([y[x == value].mean() for value in np.unique(x)], dtype=float)
