import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from online_gmm_sklearn import OnlineGaussianMixture


@parametrize_with_checks([OnlineGaussianMixture(), OnlineGaussianMixture(3, covariance_type="spherical")])
def test_sklearn_api(estimator, check):
    check(estimator)


def _mixture(n=60000, K=24, d=16, seed=0):
    """Overlapping components with unequal variances, so no start is trivially right."""
    rng = np.random.RandomState(seed)
    mu = rng.normal(0, 1.0, (K, d))
    sd = np.exp(rng.uniform(np.log(0.4), np.log(1.5), (K, d)))
    z = rng.randint(K, size=n)
    return (mu[z] + rng.standard_normal((n, d)) * sd[z]).astype(np.float32)


@pytest.mark.parametrize("cov", ["diag", "spherical"])
def test_scores_match_sklearn_for_the_same_model(cov):
    X = _mixture(5000)
    g = OnlineGaussianMixture(8, covariance_type=cov, random_state=0).fit(X)
    ref = GaussianMixture(8, covariance_type=cov)
    ref.weights_, ref.means_, ref.covariances_ = g.weights_, g.means_, g.covariances_
    ref.precisions_cholesky_ = g.precisions_cholesky_
    X64 = X.astype(np.float64)
    np.testing.assert_allclose(g.score_samples(X64), ref.score_samples(X64), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(g.predict_proba(X64), ref.predict_proba(X64), atol=1e-9)
    np.testing.assert_array_equal(g.predict(X64), ref.predict(X64))
    assert g.bic(X64) == pytest.approx(ref.bic(X64))
    assert g.covariances_.shape == ref.covariances_.shape == ((8,) if cov == "spherical" else (8, 16))


def test_close_to_full_em_on_overlapping_mixture():
    # single runs land in different local optima for both methods, so compare the average over seeds
    X = _mixture()
    Xtr, Xte = X[:50000], X[50000:]
    full = [GaussianMixture(24, covariance_type="diag", random_state=s).fit(Xtr).score(Xte) for s in range(4)]
    mine = [OnlineGaussianMixture(24, n_epochs=10, random_state=s).fit(Xtr) for s in range(4)]
    assert np.mean([m.score(Xte) for m in mine]) > np.mean(full) - 0.1
    assert all(abs(m.weights_.sum() - 1) < 1e-9 and (m.covariances_ > 0).all() for m in mine)


def test_float32_and_float64_agree():
    X = _mixture(20000) + 1000.0          # a large offset is where float32 sums of squares go wrong
    a = OnlineGaussianMixture(8, random_state=0).fit(X.astype(np.float32))
    b = OnlineGaussianMixture(8, random_state=0).fit(X.astype(np.float64))
    assert a.score(X.astype(np.float64)) == pytest.approx(b.score(X.astype(np.float64)), abs=0.02)


def test_reproducible_pipeline_clone_and_search():
    X = _mixture(8000)
    a = OnlineGaussianMixture(6, random_state=3).fit(X)
    b = clone(a).fit(X)
    np.testing.assert_array_equal(a.means_, b.means_)
    np.testing.assert_array_equal(a.fit_predict(X), a.predict(X))
    Xs, comp = a.sample(500)
    assert Xs.shape == (500, 16) and comp.shape == (500,) and a.score(Xs) > a.score(X) - 1.0
    assert make_pipeline(StandardScaler(), OnlineGaussianMixture(6, random_state=0)).fit(X).score(X) < 0
    gs = GridSearchCV(OnlineGaussianMixture(random_state=0), {"n_components": [2, 6]}, cv=2).fit(X)
    assert gs.best_params_["n_components"] == 6


def test_revival_keeps_components_alive():
    X, _ = make_blobs(n_samples=30000, centers=40, n_features=8, cluster_std=0.3, random_state=0)
    on = OnlineGaussianMixture(40, init_params="random_from_data", random_state=0).fit(X)
    off = OnlineGaussianMixture(40, init_params="random_from_data", revive=False, random_state=0).fit(X)
    assert on.n_revived_ > 0 and on.score(X) > off.score(X)


def test_starts_on_well_separated_clusters():
    # random rows often put two means in one cluster and none in another, and the fit does not recover;
    # k-means++ seeds (the default up to 64 components) do not have that problem
    X, _ = make_blobs(n_samples=20000, centers=12, n_features=20, cluster_std=1.0, random_state=0)
    ref = GaussianMixture(12, covariance_type="diag", random_state=0).fit(X).score(X)
    seeded = [OnlineGaussianMixture(12, random_state=s).fit(X).score(X) for s in range(3)]          # auto = k-means++ here
    rand = [OnlineGaussianMixture(12, init_params="random_from_data", random_state=s).fit(X).score(X) for s in range(3)]
    assert min(seeded) > ref - 0.05 and np.mean(rand) < np.mean(seeded)
    assert OnlineGaussianMixture(12, init_params="emeans", random_state=0).fit(X).means_.shape == (12, 20)
    centres = np.vstack([X[:1000].mean(0) + i for i in range(3)])
    assert OnlineGaussianMixture(3, means_init=centres, random_state=0).fit(X).means_.shape == (3, 20)
    with pytest.raises(ValueError):
        OnlineGaussianMixture(3, means_init=centres[:2]).fit(X)
    with pytest.raises(ValueError):
        OnlineGaussianMixture(3, init_params="kmeans").fit(X)


def test_auto_batch_size_rule():
    from emeans_sklearn import batch_law
    rng = np.random.RandomState(0)
    wide, narrow = rng.standard_normal((20000, 32)), rng.standard_normal((20000, 8))
    law = batch_law(20000, 10, 3)
    assert OnlineGaussianMixture(10, random_state=0).fit(wide).batch_size_ == law
    assert OnlineGaussianMixture(10, random_state=0).fit(narrow).batch_size_ == law // 8
    assert OnlineGaussianMixture(3, random_state=0).fit(narrow[:150]).batch_size_ == batch_law(150, 3, 3)   # floor of 4 * K
    assert OnlineGaussianMixture(10, batch_size=500, random_state=0).fit(narrow).batch_size_ == 500


def test_bad_arguments():
    X = _mixture(200)
    with pytest.raises(ValueError):
        OnlineGaussianMixture(3, covariance_type="full").fit(X)
    with pytest.raises(ValueError):
        OnlineGaussianMixture(500).fit(X)
    with pytest.raises(ValueError):
        OnlineGaussianMixture(3, step_size=1.5).fit(X)
