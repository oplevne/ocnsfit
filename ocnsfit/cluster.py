"""The :class:`Cluster` object: member data, distance prior and ``fit()``."""

import multiprocessing as mp
import os
import warnings

import numpy as np

from . import constants as C
from .likelihood import CMDLikelihood
from .priors import Gaussian, Priors, PriorTransform
from .result import FitResult, eta_fit

__all__ = ["Cluster", "UCC_COLUMNS", "GAIA_COLUMNS"]

# Column names of the UCC member files (Perren et al. 2023; Gaia DR3 via VizieR).
UCC_COLUMNS = dict(ra="RA_ICRS", dec="DE_ICRS", parallax="Plx", e_parallax="e_Plx",
                   gmag="Gmag", bp_rp="BP-RP", e_gmag="e_Gmag", e_bp_rp="e_BP-RP",
                   prob="probs")
# Column names of the Gaia archive (gaia_source); magnitude errors must be added.
GAIA_COLUMNS = dict(ra="ra", dec="dec", parallax="parallax", e_parallax="parallax_error",
                    gmag="phot_g_mean_mag", bp_rp="bp_rp", e_gmag="e_gmag",
                    e_bp_rp="e_bp_rp", prob="prob")


def _values(col, unit=None):
    """Plain float array from a Column / Quantity / MaskedColumn / list."""
    if unit is not None and hasattr(col, "unit") and col.unit is not None:
        try:
            col = col.to(unit) if hasattr(col, "to") else (col.quantity.to(unit))
        except Exception:
            pass
    if hasattr(col, "filled"):
        col = col.filled(np.nan)
    col = getattr(col, "value", col)
    return np.asarray(col, dtype=float)


# Worker-global objects: set once per process, avoids pickling the grid per call.
_WORKER = {}


def _init_worker(loglike, ptform):
    _WORKER["loglike"] = loglike
    _WORKER["ptform"] = ptform


def _loglike(theta):
    return _WORKER["loglike"](theta)


def _ptform(u):
    return _WORKER["ptform"](u)


class Cluster:
    """An open cluster defined by its members.

    Parameters
    ----------
    name : str
    bp_rp, gmag : array_like
        Gaia BP-RP colour and G magnitude [mag].
    e_bp_rp, e_gmag : array_like
        Their uncertainties [mag]; stars with missing values are dropped,
        non-positive values are set to ``DEFAULT_MAG_ERR``.
    parallax, e_parallax : array_like or Quantity
        Parallaxes and uncertainties (mas if plain numbers).
    prob : array_like, optional
        Membership probabilities (default 1).
    ra, dec : array_like, optional
        Member coordinates [deg]; used for :attr:`coord`.
    priors : Priors, optional
    min_prob : float
        Members with ``prob < min_prob`` are ignored.

    Member selection follows Plevne & Akbaba (2026): ``prob >= min_prob``,
    ``parallax > 0``, ``-1 < BP-RP < 4`` and ``0 < G < 20``.
    """

    def __init__(self, name, bp_rp, gmag, e_bp_rp, e_gmag, parallax, e_parallax,
                 prob=None, ra=None, dec=None, priors=None, min_prob=C.MIN_PROB):
        import astropy.units as u
        self.name = str(name)
        plx = _values(parallax, u.mas)
        n = len(plx)
        cols = dict(bp_rp=_values(bp_rp, u.mag), gmag=_values(gmag, u.mag),
                    e_bp_rp=_values(e_bp_rp, u.mag), e_gmag=_values(e_gmag, u.mag),
                    parallax=plx, e_parallax=_values(e_parallax, u.mas),
                    prob=np.ones(n) if prob is None else _values(prob))
        if ra is not None and dec is not None:
            cols["ra"], cols["dec"] = _values(ra, u.deg), _values(dec, u.deg)

        ok = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        mask = (ok & (cols["prob"] >= min_prob) & (cols["parallax"] > 0)
                & (C.COLOR_RANGE[0] < cols["bp_rp"]) & (cols["bp_rp"] < C.COLOR_RANGE[1])
                & (C.G_RANGE[0] < cols["gmag"]) & (cols["gmag"] < C.G_RANGE[1]))
        for k, v in cols.items():
            if k.startswith("e_") and k != "e_parallax":
                v = np.where(v > 0, v, C.DEFAULT_MAG_ERR)
            setattr(self, k, v[mask])
        self.n_input = n
        self.min_prob = min_prob
        if self.n_members < C.MIN_MEMBERS:
            warnings.warn(f"{self.name}: only {self.n_members} members pass the selection")
        self.priors = priors if priors is not None else Priors()

    # ------------------------------------------------------- constructors
    @classmethod
    def from_table(cls, table, name, columns=None, **kwargs):
        """Build a cluster from an astropy Table / QTable or pandas DataFrame.

        ``columns`` maps the keys of :data:`UCC_COLUMNS` (``ra, dec, parallax,
        e_parallax, gmag, bp_rp, e_gmag, e_bp_rp, prob``) to column names. By
        default the UCC names are tried first, then the Gaia-archive names.
        Units attached to columns (e.g. parallax in arcsec) are converted.
        """
        if not hasattr(table, "colnames"):
            from astropy.table import Table
            table = Table.from_pandas(table)
        names = set(table.colnames)
        base = UCC_COLUMNS if UCC_COLUMNS["parallax"] in names else GAIA_COLUMNS
        cmap = {**base, **(columns or {})}

        def col(key, required=True):
            c = cmap.get(key)
            if c in names:
                return table[c]
            if required:
                raise KeyError(f"column for {key!r} not found (looked for {c!r}); "
                               "pass columns={...}")
            return None

        return cls(name, bp_rp=col("bp_rp"), gmag=col("gmag"), e_bp_rp=col("e_bp_rp"),
                   e_gmag=col("e_gmag"), parallax=col("parallax"),
                   e_parallax=col("e_parallax"), prob=col("prob", False),
                   ra=col("ra", False), dec=col("dec", False), **kwargs)

    @classmethod
    def from_ucc(cls, members, name, member_name=None, **kwargs):
        """Select one cluster from the UCC member file (column ``name``)."""
        key = (member_name or name).lower().replace(" ", "").replace("_", "")
        if hasattr(members, "colnames"):
            sel = np.char.lower(np.asarray(members["name"]).astype(str)) == key
            sub = members[sel]
        else:
            sub = members[members["name"].str.lower() == key]
        if len(sub) == 0:
            raise KeyError(f"no members named {key!r}")
        return cls.from_table(sub, name, columns=UCC_COLUMNS, **kwargs)

    # ------------------------------------------------------------ props
    @property
    def n_members(self):
        return len(self.gmag)

    @property
    def coord(self):
        """SkyCoord of the median member position (ICRS), or None."""
        if not hasattr(self, "ra"):
            return None
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        return SkyCoord(np.median(self.ra) * u.deg, np.median(self.dec) * u.deg, frame="icrs")

    @property
    def distance_prior(self):
        """Gaussian distance prior [pc] from the member parallaxes.

        Centre: median of 1000/parallax over members with
        sigma_parallax/parallax <= 0.2 (relaxed to 0.5, then to all members if
        fewer than three qualify). Width: 1.4826 MAD / sqrt(N) (standard error
        of the median), clipped to 2-50 per cent of the centre.
        """
        rel = self.e_parallax / np.abs(self.parallax)
        for lim in (0.2, 0.5, np.inf):
            sel = rel <= lim
            if sel.sum() >= 3:
                break
        d = 1000.0 / self.parallax[sel]
        mu = float(np.median(d))
        sig = 1.4826 * float(np.median(np.abs(d - mu))) / np.sqrt(len(d))
        sig = float(np.clip(sig, C.DIST_SIGMA_FLOOR_FRAC * mu, C.DIST_SIGMA_CAP_FRAC * mu))
        return Gaussian(mu, sig, 0.0, np.inf)

    def set_priors(self, priors=None, **kwargs):
        """Set priors: a :class:`Priors` object or keywords passed to it."""
        self.priors = priors if priors is not None else Priors(**kwargs)
        return self

    def to_table(self):
        """Selected members as an astropy QTable with units."""
        import astropy.units as u
        from astropy.table import QTable
        t = QTable()
        if hasattr(self, "ra"):
            t["ra"], t["dec"] = self.ra * u.deg, self.dec * u.deg
        t["parallax"], t["e_parallax"] = self.parallax * u.mas, self.e_parallax * u.mas
        t["gmag"], t["bp_rp"] = self.gmag * u.mag, self.bp_rp * u.mag
        t["e_gmag"], t["e_bp_rp"] = self.e_gmag * u.mag, self.e_bp_rp * u.mag
        t["prob"] = self.prob
        return t

    def __repr__(self):
        c = self.coord
        pos = "" if c is None else f", (ra, dec) = ({c.ra.deg:.3f}, {c.dec.deg:.3f}) deg"
        d = self.distance_prior if self.n_members >= C.MIN_MEMBERS else None
        dp = "" if d is None else f", d_prior = {d.mu:.0f} +/- {d.sigma:.0f} pc"
        return f"Cluster({self.name!r}, {self.n_members} members{pos}{dp})"

    # -------------------------------------------------------------- fit
    def fit(self, grid, priors=None, nlive=C.NLIVE, nlive_batch=C.NLIVE_BATCH,
            dlogz=C.DLOGZ_INIT, maxiter=C.MAXITER, bound=C.BOUND, sample=C.SAMPLE,
            workers=None, seed=None, print_progress=True, compute_eta=True,
            err_inflate=C.ERR_INFLATE, chi2_max=C.CHI2_MAX):
        """Sample the posterior of (logAge, [Fe/H], distance, E(BP-RP)).

        Runs ``dynesty.DynamicNestedSampler`` with the settings of
        Plevne & Akbaba (2026) and returns a :class:`~ocnsfit.FitResult`.

        Parameters
        ----------
        grid : IsochroneGrid
        priors : Priors, optional
            Overrides :attr:`priors` for this fit.
        workers : int, optional
            Number of processes (default: all CPUs; 1 disables multiprocessing).
        seed : int, optional
            Seed of the sampler's random generator.
        """
        from dynesty import DynamicNestedSampler

        if self.n_members < C.MIN_MEMBERS:
            raise ValueError(f"{self.name}: too few members ({self.n_members})")
        priors = priors or self.priors
        loglike = CMDLikelihood(grid, self.bp_rp, self.gmag, self.e_bp_rp, self.e_gmag,
                                self.prob, err_inflate=err_inflate, chi2_max=chi2_max)
        ptform = PriorTransform(priors, self.distance_prior)
        workers = workers or os.cpu_count() or 1
        rstate = np.random.default_rng(seed)
        kw = dict(ndim=4, nlive=nlive, bound=bound, sample=sample, rstate=rstate)
        run = dict(nlive_init=nlive, nlive_batch=nlive_batch, dlogz_init=dlogz,
                   maxiter=maxiter, print_progress=print_progress)

        if workers > 1:
            # 'fork' shares the grid with the workers without pickling it (the
            # interpolator is ~125 MB); fall back to the platform default elsewhere.
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else None)
            _init_worker(loglike, ptform)
            with ctx.Pool(workers, initializer=_init_worker,
                          initargs=(loglike, ptform)) as pool:
                sampler = DynamicNestedSampler(_loglike, _ptform, pool=pool,
                                               queue_size=workers, **kw)
                sampler.run_nested(**run)
        else:
            sampler = DynamicNestedSampler(loglike, ptform, **kw)
            sampler.run_nested(**run)

        res = FitResult.from_dynesty(self.name, sampler.results, cluster=self, grid=grid,
                                     meta={"priors": repr(priors),
                                           "distance_prior": str(ptform.distance)})
        if compute_eta:
            res.eta_fit = self.eta_fit(grid, res.median_theta)
        return res

    def eta_fit(self, grid, theta, band=C.ETA_FIT_BAND, source="grid"):
        """Quality metric eta_fit for parameters ``theta``.

        ``source='grid'`` (default) uses the nearest grid isochrone (phases 1-3);
        ``source='parsec'`` downloads the full PARSEC isochrone (all phases up
        to the early AGB, label <= 7), as done for the published catalogue.
        """
        la, z, d, e = theta
        if source == "parsec":
            from .parsec import ParsecQuery
            iso = ParsecQuery().query_isochrones(
                t=10 ** la, Z=z, photsys_file="YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat")
            m = np.asarray(iso["label"]) <= 7
            dm = 5 * np.log10(d) - 5
            col = np.asarray(iso["G_BPmag"][m] - iso["G_RPmag"][m], float) + e
            g = np.asarray(iso["Gmag"][m], float) + C.K_AG_EBPRP * e + dm
        else:
            iso = grid.isochrone(la, z, d, e)
            col, g = np.asarray(iso["bp_rp"]), np.asarray(iso["Gmag"])
        return eta_fit(self.bp_rp, self.gmag, col, g, band=band)
