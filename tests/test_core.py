"""Unit tests that need no isochrone grid (run with ``pytest``)."""

import numpy as np
import pytest
from astropy.table import Table

from ocnsfit import Cluster, FitResult, Gaussian, Priors, Uniform
from ocnsfit import constants as C
from ocnsfit.priors import PriorTransform
from ocnsfit.result import eta_fit


def test_uniform_and_gaussian_ppf():
    assert Uniform(1, 3).ppf(0.5) == 2
    g = Gaussian(0.0, 1.0, -1.0, 1.0)
    assert g.ppf(0.5) == pytest.approx(0.0)
    assert -1 <= g.ppf(1e-9) and g.ppf(1 - 1e-9) <= 1


def test_priors_defaults_and_bounds():
    p = Priors()
    assert isinstance(p.logage, Uniform) and (p.logage.low, p.logage.high) == C.LOGAGE_BOUNDS
    p = Priors(ebprp=(0.3, 10.0))
    assert p.ebprp.low == 0.0 and p.ebprp.high == 5.0


def test_prior_table_explicit_and_implicit():
    t = Table(rows=[("A", "uniform", np.nan, np.nan, 7.0, 8.0, 0.1, 0.05, "gauss"),
                    ("B", "", 9.0, 0.5, np.nan, np.nan, np.nan, np.nan, "")],
              names=("name", "logAge_type", "logAge_mu", "logAge_sigma", "logAge_lo",
                     "logAge_hi", "E_BPRP_mu", "E_BPRP_sigma", "E_BPRP_type"))
    a = Priors.from_table(t, "A")
    assert isinstance(a.logage, Uniform) and a.logage.high == 8.0
    assert isinstance(a.ebprp, Gaussian) and a.ebprp.mu == 0.1
    b = Priors.from_table(t, "B")
    assert isinstance(b.logage, Gaussian) and b.logage.mu == 9.0
    assert isinstance(b.ebprp, Uniform)
    with pytest.warns(UserWarning):
        Priors.from_table(t, "missing")


def _toy_cluster(n=50, seed=0):
    rng = np.random.default_rng(seed)
    plx = rng.normal(1.0, 0.05, n)
    return Cluster("toy", bp_rp=rng.uniform(0.5, 1.5, n), gmag=rng.uniform(10, 18, n),
                   e_bp_rp=np.full(n, 0.01), e_gmag=np.full(n, 0.005), parallax=plx,
                   e_parallax=np.full(n, 0.02), prob=np.r_[np.ones(n - 5), np.zeros(5)],
                   ra=rng.uniform(10, 10.1, n), dec=rng.uniform(20, 20.1, n))


def test_cluster_selection_and_distance_prior():
    cl = _toy_cluster()
    assert cl.n_members == 45
    d = cl.distance_prior
    assert d.mu == pytest.approx(1000, rel=0.05)
    assert C.DIST_SIGMA_FLOOR_FRAC * d.mu <= d.sigma <= C.DIST_SIGMA_CAP_FRAC * d.mu
    assert cl.coord.ra.deg == pytest.approx(10.05, abs=0.05)


def test_cluster_from_qtable_units():
    import astropy.units as u
    cl = _toy_cluster()
    t = cl.to_table()
    t["parallax"] = t["parallax"].to(u.arcsec)
    t2 = Cluster.from_table(t, "toy", columns=dict(
        ra="ra", dec="dec", parallax="parallax", e_parallax="e_parallax", gmag="gmag",
        bp_rp="bp_rp", e_gmag="e_gmag", e_bp_rp="e_bp_rp", prob="prob"))
    np.testing.assert_allclose(t2.parallax, cl.parallax)


def test_prior_transform_ranges():
    cl = _toy_cluster()
    pt = PriorTransform(Priors(feh=Uniform(-0.2, 0.1)), cl.distance_prior)
    for u in np.random.default_rng(1).uniform(size=(200, 4)):
        la, z, d, e = pt(u)
        assert 6 <= la <= 10 and d > 0 and 0 <= e <= 5
        assert -0.2 - 1e-9 <= np.log10(z / C.Z_SUN) <= 0.1 + 1e-9


def test_eta_fit_and_result_roundtrip(tmp_path):
    iso_c, iso_g = np.linspace(0, 2, 100), np.linspace(10, 20, 100)
    assert eta_fit(iso_c + 0.1, iso_g, iso_c, iso_g) == 1.0
    s = np.c_[np.full(10, 9.0), np.full(10, C.Z_SUN), np.full(10, 1000.), np.full(10, 0.2)]
    r = FitResult("x", s, np.full(10, 0.1), log_z=-3.0, log_z_err=0.1, eta_fit=0.9)
    r.save(tmp_path / "x.h5")
    r2 = FitResult.load(tmp_path / "x.h5")
    assert r2.name == "x" and r2.eta_fit == 0.9
    q = r2.quantiles()
    assert q["FeH"][1] == pytest.approx(0.0) and q["A_G"][1] == pytest.approx(0.378)
