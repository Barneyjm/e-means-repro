import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from emeans_sklearn import EMeans


@parametrize_with_checks([EMeans()])
def test_sklearn_api(estimator, check):
    check(estimator)


def test_close_to_kmeans_on_blobs():
    X, _ = make_blobs(n_samples=20000, centers=50, n_features=16, random_state=0)
    X = X.astype(np.float32)
    km = KMeans(50, init="random", n_init=1, max_iter=25, random_state=0).fit(X)
    em = EMeans(50, random_state=0).fit(X)
    assert em.inertia_ < 1.1 * km.inertia_


def test_reproducible_and_pipeline():
    X, _ = make_blobs(n_samples=5000, centers=10, n_features=8, random_state=1)
    a = EMeans(10, random_state=3).fit(X)
    b = EMeans(10, random_state=3).fit(X)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.predict(X), a.labels_)
    assert a.transform(X).shape == (5000, 10)
    assert make_pipeline(StandardScaler(), EMeans(10, random_state=0)).fit_predict(X).shape == (5000,)
