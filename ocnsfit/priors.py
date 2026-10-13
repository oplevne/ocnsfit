"""Prior distributions and the prior transform used by the nested sampler.

Each of the four parameters (log age, [Fe/H], distance, E(BP-RP)) takes either
a :class:`Uniform` or a (truncated) :class:`Gaussian` prior. Priors can be set
by hand or read from a prior file with :meth:`Priors.from_file`; OCNSFIT never
builds priors itself (e.g. from dust maps), so any calibration such as the
Schlafly & Finkbeiner (2011) rescaling of SFD reddenings belongs in the file.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import truncnorm

from . import constants as C

__all__ = ["Uniform", "Gaussian", "Priors", "PriorTransform"]


@dataclass(frozen=True)
class Uniform:
    """Uniform prior on [low, high]."""

    low: float
    high: float

    def __post_init__(self):
        if not self.high > self.low:
            raise ValueError(f"Uniform prior needs high > low, got {self}")

    def ppf(self, u):
        return self.low + u * (self.high - self.low)

    def clip(self, low, high):
        return Uniform(max(self.low, low), min(self.high, high))

    def __str__(self):
        return f"Uniform[{self.low:g}, {self.high:g}]"


@dataclass(frozen=True)
class Gaussian:
    """Gaussian prior N(mu, sigma), truncated to [low, high]."""

    mu: float
    sigma: float
    low: float = -np.inf
    high: float = np.inf
    _a: float = field(init=False, repr=False)
    _b: float = field(init=False, repr=False)

    def __post_init__(self):
        if not self.sigma > 0:
            raise ValueError(f"Gaussian prior needs sigma > 0, got {self}")
        object.__setattr__(self, "_a", (self.low - self.mu) / self.sigma)
        object.__setattr__(self, "_b", (self.high - self.mu) / self.sigma)

    def ppf(self, u):
        return float(truncnorm.ppf(u, self._a, self._b, loc=self.mu, scale=self.sigma))

    def clip(self, low, high):
        return Gaussian(self.mu, self.sigma, max(self.low, low), min(self.high, high))

    def __str__(self):
        return (f"Gaussian(mu={self.mu:g}, sigma={self.sigma:g}) "
                f"truncated to [{self.low:g}, {self.high:g}]")


def _make(kind, mu=None, sigma=None, low=None, high=None):
    kind = str(kind).strip().lower()
    if kind in ("gauss", "gaussian", "normal"):
        return Gaussian(mu, sigma,
                        -np.inf if low is None else low,
                        np.inf if high is None else high)
    if kind in ("uniform", "flat"):
        return Uniform(low, high)
    raise ValueError(f"Unknown prior type {kind!r} (use 'gauss' or 'uniform')")


def _finite(x):
    try:
        return x is not None and not np.ma.is_masked(x) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class Priors:
    """Container for the priors of one cluster.

    Parameters
    ----------
    logage, feh, ebprp, distance : Uniform, Gaussian, tuple or None
        Prior on log(age/yr), [Fe/H], E(BP-RP) [mag] and distance [pc].
        A tuple ``(mu, sigma)`` is shorthand for a Gaussian, ``None`` selects
        the default: Uniform over the model bounds for age, [Fe/H] and
        E(BP-RP); for the distance, the parallax-based Gaussian computed by
        :class:`~ocnsfit.Cluster`.

    All priors are truncated to the model bounds in :mod:`ocnsfit.constants`.
    """

    def __init__(self, logage=None, feh=None, ebprp=None, distance=None):
        self.logage = self._coerce(logage, Uniform(*C.LOGAGE_BOUNDS)).clip(*C.LOGAGE_BOUNDS)
        self.feh = self._coerce(feh, Uniform(*C.FEH_BOUNDS))
        if isinstance(self.feh, Gaussian) and not (math.isfinite(self.feh.low)
                                                   and math.isfinite(self.feh.high)):
            self.feh = self.feh.clip(*C.FEH_BOUNDS)
        self.ebprp = self._coerce(ebprp, Uniform(*C.EBPRP_BOUNDS)).clip(*C.EBPRP_BOUNDS)
        self.distance = None if distance is None else \
            self._coerce(distance, None).clip(0.0, np.inf)

    @staticmethod
    def _coerce(p, default):
        if p is None:
            return default
        if isinstance(p, (Uniform, Gaussian)):
            return p
        if isinstance(p, (tuple, list)) and len(p) == 2:
            return Gaussian(float(p[0]), float(p[1]))
        raise TypeError(f"Cannot interpret {p!r} as a prior")

    def __repr__(self):
        d = "from parallaxes" if self.distance is None else str(self.distance)
        return (f"Priors(\n  logAge   : {self.logage}\n  [Fe/H]   : {self.feh}\n"
                f"  E(BP-RP) : {self.ebprp}\n  distance : {d}\n)")

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_row(cls, row):
        """Build priors from one row (mapping) of a prior table.

        Two column conventions are understood (see ``docs/priors.md``):

        * explicit: ``<par>_type`` ('gauss' / 'uniform'), ``<par>_mu``,
          ``<par>_sigma``, ``<par>_lo``, ``<par>_hi`` for
          ``<par>`` in ``logAge``, ``feh``, ``E_BPRP``, ``dist``;
        * implicit (format of the published run): a parameter gets a
          Gaussian when ``<par>_mu`` and ``<par>_sigma`` are finite, and a
          Uniform on ``[<par>_lo, <par>_hi]`` otherwise.
        """
        keys = {k.lower(): k for k in row.keys()}

        def get(name):
            k = keys.get(name.lower())
            return row[k] if k is not None and _finite(row[k]) else None

        def get_type(name):
            k = keys.get(name.lower())
            if k is None or row[k] is None or np.ma.is_masked(row[k]):
                return None
            s = str(row[k]).strip()
            return s or None

        def parse(par, default_bounds, positive_mu=False):
            kind = get_type(f"{par}_type")
            mu, sig = get(f"{par}_mu"), get(f"{par}_sigma")
            lo, hi = get(f"{par}_lo"), get(f"{par}_hi")
            if kind is None:
                gauss = mu is not None and sig is not None and (not positive_mu or mu > 0)
                kind = "gauss" if gauss else ("uniform" if lo is not None or hi is not None
                                              else None)
            if kind is None:
                return None
            if kind.lower().startswith("u"):
                lo = default_bounds[0] if lo is None else lo
                hi = default_bounds[1] if hi is None else hi
                return Uniform(lo, hi)
            return _make(kind, mu, sig,
                         default_bounds[0] if lo is None else lo,
                         default_bounds[1] if hi is None else hi)

        return cls(logage=parse("logAge", C.LOGAGE_BOUNDS, positive_mu=True),
                   feh=parse("feh", C.FEH_BOUNDS),
                   ebprp=parse("E_BPRP", C.EBPRP_BOUNDS, positive_mu=True),
                   distance=parse("dist", (0.0, np.inf)))

    @classmethod
    def from_file(cls, path, name, name_col=None, **read_kwargs):
        """Read the priors of cluster ``name`` from a table file.

        Any format readable by :meth:`astropy.table.Table.read` works
        (ECSV, FITS, CSV, VOTable, ...); ``.parquet`` is also accepted.
        ``name_col`` defaults to the first of ``name``, ``cluster``,
        ``display_name`` present in the table. If the cluster is absent,
        default priors are returned with a warning.
        """
        tab = read_prior_table(path, **read_kwargs)
        return cls.from_table(tab, name, name_col=name_col)

    @classmethod
    def from_table(cls, table, name, name_col=None):
        """Like :meth:`from_file` for an already loaded table."""
        name_col = name_col or _find_name_col(table.colnames)
        names = np.char.strip(np.asarray(table[name_col]).astype(str))
        idx = np.flatnonzero(names == str(name).strip())
        if len(idx) == 0:
            warnings.warn(f"{name!r} not found in prior table; using default priors")
            return cls()
        row = table[int(idx[0])]
        return cls.from_row({c: row[c] for c in table.colnames})


def _find_name_col(colnames):
    for c in ("name", "cluster", "Cluster", "display_name"):
        if c in colnames:
            return c
    raise KeyError("No cluster-name column found; pass name_col=")


def read_prior_table(path, **read_kwargs):
    """Read a prior table as an :class:`astropy.table.Table`."""
    from astropy.table import Table
    if str(path).endswith(".parquet"):
        import pandas as pd
        return Table.from_pandas(pd.read_parquet(path))
    return Table.read(path, **read_kwargs)


class PriorTransform:
    """Map the unit hypercube to (logAge, Zini, distance_pc, E_BPRP).

    The object is picklable, so it can be shipped to multiprocessing workers.
    """

    def __init__(self, priors, distance_prior):
        self.logage = priors.logage
        self.feh = priors.feh
        self.ebprp = priors.ebprp
        self.distance = priors.distance or distance_prior
        self._feh_lo = self.feh.low
        self._feh_hi = self.feh.high

    def __call__(self, u):
        theta = np.empty(4)
        theta[0] = self.logage.ppf(u[0])
        feh = np.clip(self.feh.ppf(u[1]), self._feh_lo, self._feh_hi)
        theta[1] = C.Z_SUN * 10.0 ** feh
        theta[2] = max(self.distance.ppf(u[2]), C.DISTANCE_MIN_PC)
        theta[3] = np.clip(self.ebprp.ppf(u[3]), *C.EBPRP_BOUNDS)
        return theta
