"""OCNSFIT -- Open Cluster Nested Sampling Fit.

Bayesian estimation of age, metallicity, distance and reddening of open
clusters by fitting PARSEC isochrones to Gaia colour-magnitude diagrams with
dynamic nested sampling (Plevne & Akbaba 2026, MNRAS).

Quick start::

    from ocnsfit import Cluster, IsochroneGrid, Priors, Gaussian

    grid = IsochroneGrid.load("isochrones/")
    cl = Cluster.from_table(members, name="NGC6791")
    cl.set_priors(feh=Gaussian(0.3, 0.35), ebprp=(0.15, 0.20))
    result = cl.fit(grid)
    print(result.summary())
"""

from .cluster import GAIA_COLUMNS, UCC_COLUMNS, Cluster
from .isochrones import IsochroneGrid, train_interpolator
from .priors import Gaussian, Priors, Uniform
from .result import FitResult

__version__ = "1.0.0"

__all__ = ["Cluster", "IsochroneGrid", "Priors", "Uniform", "Gaussian", "FitResult",
           "train_interpolator", "UCC_COLUMNS", "GAIA_COLUMNS", "__version__"]
