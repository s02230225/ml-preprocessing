import numpy as np
import pandas as pd
import pytest

from scalers import MinMaxScaler, StandardScaler
from cross_val import kfold_split, knn_cv_score
from encoders import MyOneHotEncoder, SimpleCounterEncoder, FoldCounters, group_k_fold


@pytest.mark.parametrize('scaler', [MinMaxScaler, StandardScaler])
def test_scalers_constant_features_and_no_mutation(scaler):
    X = np.array([[1., 7.], [3., 7.]])
    before = X.copy()
    model = scaler().fit(X)
    result = model.transform(X)
    assert np.array_equal(X, before)
    assert np.isfinite(result).all()
    assert np.array_equal(result[:, 1], [0, 0])
    assert model.transform(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize('scaler', [MinMaxScaler, StandardScaler])
def test_scaler_errors(scaler):
    with pytest.raises(ValueError):
        scaler().transform([[1, 2]])
    with pytest.raises(ValueError):
        scaler().fit([])
    with pytest.raises(ValueError):
        scaler().fit([[np.nan]])
    with pytest.raises(ValueError):
        scaler().fit([[1, 2]]).transform([[1]])


def test_kfold_balanced_disjoint_complete():
    folds = kfold_split(11, 4)
    assert sorted(len(valid) for _, valid in folds) == [2, 3, 3, 3]
    assert sorted(np.concatenate([valid for _, valid in folds])) == list(range(11))
    for train, valid in folds:
        assert not np.intersect1d(train, valid).size
        assert sorted(np.r_[train, valid]) == list(range(11))


@pytest.mark.parametrize('n,k', [(0, 2), (3, 4), (3, 1), (3, 0)])
def test_bad_folds(n, k):
    with pytest.raises(ValueError):
        kfold_split(n, k)


def test_onehot_missing_and_unknown():
    X = pd.DataFrame({'a': ['x', None, 'x'], 'b': [1, 2, 1]})
    encoder = MyOneHotEncoder().fit(X)
    result = encoder.transform(pd.DataFrame({'a': ['unknown', None], 'b': [1, 3]}))
    assert result.shape == (2, 4)
    assert result[0].sum() == 1
    assert result[1].sum() == 1
    assert encoder.transform(X.iloc[:0]).shape == (0, 4)


def test_counter_uses_row_positions_not_index_labels():
    X = pd.DataFrame({'a': ['x', 'x', None]}, index=[10, 20, 30])
    y = pd.Series([0., 1., 1.], index=[7, 8, 9])
    result = SimpleCounterEncoder().fit(X, y).transform(X)
    assert np.allclose(result[:, 0], [.5, .5, 1.])
    assert np.allclose(result[:, 1], [2/3, 2/3, 1/3])


def test_out_of_fold_has_no_self_target():
    X = pd.DataFrame({'id': ['a', 'b', 'c', 'd']})
    y = np.array([0., 1., 0., 1.])
    encoder = FoldCounters(n_folds=2)
    result = encoder.fit_transform(X, y)
    assert np.array_equal(result[:, :2], np.zeros((4, 2)))
    unseen = encoder.transform(pd.DataFrame({'id': ['new']}))
    assert unseen.shape == (1, 3)
    assert np.isfinite(unseen).all()


def test_rng_state_unchanged():
    np.random.seed(123)
    before = np.random.get_state()
    list(group_k_fold(10, 3, seed=5))
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_cv_fits_normalizer_only_on_training_fold():
    class Recorder:
        fits = []
        def fit(self, X):
            self.fits.append(X.copy())
        def transform(self, X):
            return X
    class Model:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            self.value = y[0]
        def predict(self, X):
            return np.full(len(X), self.value)
    X = np.arange(8).reshape(4, 2)
    folds = kfold_split(4, 2)
    parameters = {'normalizers': [(Recorder(), 'record')], 'n_neighbors': [1], 'metrics': ['euclidean'], 'weights': ['uniform']}
    result = knn_cv_score(X, np.zeros(4), parameters, lambda a, b: np.mean(a == b), folds, Model)
    assert result[('record', 1, 'euclidean', 'uniform')] == 1
    assert len(Recorder.fits) == 2
    assert all(len(values) == 2 for values in Recorder.fits)
