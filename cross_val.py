from copy import deepcopy
from numbers import Integral

import numpy as np


def kfold_split(num_objects, num_folds):
    if (not isinstance(num_objects, Integral) or isinstance(num_objects, bool)
            or not isinstance(num_folds, Integral) or isinstance(num_folds, bool)
            or not 2 <= num_folds <= num_objects):
        raise ValueError("Expected 2 <= num_folds <= num_objects")
    indices = np.arange(num_objects)
    folds = []
    for valid in np.array_split(indices, num_folds):
        train = indices[~np.isin(indices, valid)]
        folds.append((train, valid))
    return folds


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not folds:
        raise ValueError("Expected matching X, y and nonempty folds")
    for train, valid in folds:
        if len(train) == 0 or len(valid) == 0 or np.intersect1d(train, valid).size:
            raise ValueError("Training and validation indices must be nonempty and disjoint")
        if min(np.min(train), np.min(valid)) < 0 or max(np.max(train), np.max(valid)) >= len(X):
            raise ValueError("Fold index outside the dataset")
    scores = {}
    for normalizer, name in parameters['normalizers']:
        for neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    values = []
                    for train, valid in folds:
                        X_train, X_valid = X[train], X[valid]
                        if normalizer is not None:
                            scaler = deepcopy(normalizer)
                            scaler.fit(X_train)
                            X_train = scaler.transform(X_train)
                            X_valid = scaler.transform(X_valid)
                        model = knn_class(n_neighbors=neighbors, metric=metric, weights=weight)
                        model.fit(X_train, y[train])
                        values.append(score_function(y[valid], model.predict(X_valid)))
                    scores[(name, neighbors, metric, weight)] = float(np.mean(values))
    return scores
