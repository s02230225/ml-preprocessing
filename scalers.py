import numpy as np


def _matrix(data):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("Expected a matrix with at least one feature")
    if not np.isfinite(data).all():
        raise ValueError("Features must be finite numbers")
    return data


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, data):
        data = _matrix(data)
        if len(data) == 0:
            raise ValueError("Cannot fit on an empty matrix")
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, data):
        if self.min_ is None:
            raise ValueError("Call fit before transform")
        data = _matrix(data)
        if data.shape[1] != len(self.min_):
            raise ValueError("Feature count differs from training data")
        return (data - self.min_) / self.range_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        data = _matrix(data)
        if len(data) == 0:
            raise ValueError("Cannot fit on an empty matrix")
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, data):
        if self.mean_ is None:
            raise ValueError("Call fit before transform")
        data = _matrix(data)
        if data.shape[1] != len(self.mean_):
            raise ValueError("Feature count differs from training data")
        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        return self.fit(data).transform(data)
