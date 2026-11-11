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


def test_balance_evens_out_cluster_sizes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 16)).astype(np.float32) * rng.uniform(0.5, 2.0, 16).astype(np.float32)
    def imbalance(m): c = np.bincount(m.labels_, minlength=256); return 256 * (c.astype(float) ** 2).sum() / len(X) ** 2
    plain = EMeans(256, random_state=0).fit(X)
    faded = EMeans(256, balance=0.3, random_state=0).fit(X)
    held = EMeans(256, balance=0.3, balance_end=1.0, random_state=0).fit(X)
    assert imbalance(held) < imbalance(faded) < imbalance(plain)
    assert held.inertia_ < 1.02 * plain.inertia_


def test_step_of_one_stays_finite():
    X, _ = make_blobs(n_samples=4000, centers=20, n_features=8, random_state=2)
    m = EMeans(200, step_size=1.0, batch_size=400, random_state=0).fit(X.astype(np.float32))
    assert np.isfinite(m.cluster_centers_).all()
