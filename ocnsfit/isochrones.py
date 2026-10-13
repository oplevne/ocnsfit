"""PARSEC isochrone grid: loading, nearest-node lookup and model CMDs.

The grid is a table of PARSEC v1.2S isochrones in the Gaia EDR3 passbands
(columns ``logAge, Zini, Mini, label, Gmag, bp_rp``), produced by
``scripts/build_grid.py``. A fully grown decision tree trained on the grid
(``scripts/train_interpolator.py``) returns absolute G and intrinsic BP-RP as a
function of (logAge, Zini, Mini). During the fit the sampled (Zini, logAge) is
snapped to the nearest grid node with a k-d tree, and the model CMD is
evaluated on that node's mass sampling.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import constants as C

__all__ = ["IsochroneGrid", "train_interpolator"]

GRID_FILE = "isochrone_grid.parquet"
INTERP_FILE = "iso_interpolator.pkl"
GRID_COLUMNS = ["logAge", "Zini", "Mini", "label", "Gmag", "bp_rp"]


class IsochroneGrid:
    """In-memory isochrone grid with its interpolator.

    Parameters
    ----------
    grid : pandas.DataFrame or astropy.table.Table
        Isochrone points with at least the columns in ``GRID_COLUMNS``.
    interpolator : object with ``predict(X) -> (n, 2)``
        Maps ``(logAge, Zini, Mini)`` to ``(Gmag, bp_rp)``.
    labels : tuple of int
        PARSEC evolutionary phases kept for the fit.
    """

    def __init__(self, grid, interpolator, labels=C.LABELS_FIT):
        if not isinstance(grid, pd.DataFrame):
            grid = grid.to_pandas()
        grid = grid[grid["label"].isin(labels)].reset_index(drop=True)
        self.grid = grid
        # Predictions are small and run inside sampler workers: no nested parallelism.
        if hasattr(interpolator, "n_jobs"):
            interpolator.n_jobs = 1
        self.interpolator = interpolator
        self.tree = cKDTree(np.c_[grid["Zini"].to_numpy(), grid["logAge"].to_numpy()])
        self._zini = grid["Zini"].to_numpy()
        self._logage = grid["logAge"].to_numpy()
        # (Zini, logAge) -> rows; avoids DataFrame filtering inside the likelihood
        self._mini = {}
        self._rows = {}
        for key, g in grid.groupby(["Zini", "logAge"], sort=False):
            self._mini[key] = g["Mini"].to_numpy()
            self._rows[key] = g.index.to_numpy()

    @classmethod
    def load(cls, directory, grid_file=GRID_FILE, interp_file=INTERP_FILE, **kwargs):
        """Load ``grid_file`` and ``interp_file`` from ``directory``."""
        grid = pd.read_parquet(os.path.join(directory, grid_file), columns=GRID_COLUMNS)
        with open(os.path.join(directory, interp_file), "rb") as f:
            interp = pickle.load(f)
        return cls(grid, interp, **kwargs)

    def __len__(self):
        return len(self._mini)

    def __repr__(self):
        feh = np.log10(np.unique(self._zini) / C.Z_SUN)
        return (f"IsochroneGrid({len(self)} isochrones, {len(self.grid):,} points; "
                f"logAge {self._logage.min():.2f}-{self._logage.max():.2f}, "
                f"log(Z/Zsun) {feh.min():+.2f}-{feh.max():+.2f})")

    # ---------------------------------------------------------------- lookup
    def nearest_node(self, logage, zini):
        """Return the (Zini, logAge) of the grid node closest to the input."""
        _, i = self.tree.query([zini, logage], k=1)
        return float(self._zini[i]), float(self._logage[i])

    def model_cmd(self, theta):
        """Model (BP-RP, G) of the isochrone for ``theta``.

        ``theta = (logAge, Zini, distance_pc, E_BPRP)``. Returns two empty arrays
        if the node has fewer than two mass points.
        """
        logage, zini, distance_pc, ebprp = theta
        z, la = self.nearest_node(logage, zini)
        mini = self._mini.get((z, la))
        if mini is None or len(mini) < 2:
            return np.array([]), np.array([])
        pred = self.interpolator.predict(
            np.c_[np.full(len(mini), la), np.full(len(mini), z), mini])
        dist_mod = 5.0 * np.log10(distance_pc) - 5.0
        color = pred[:, 1] + ebprp
        gmag = pred[:, 0] + C.K_AG_EBPRP * ebprp + dist_mod
        return color, gmag

    def isochrone(self, logage, zini, distance_pc=10.0, ebprp=0.0):
        """Grid isochrone (nearest node) shifted to the given distance/reddening.

        Returns an :class:`astropy.table.Table` with ``Mini, label, bp_rp, Gmag``
        taken directly from the grid (no interpolation).
        """
        from astropy.table import Table
        z, la = self.nearest_node(logage, zini)
        g = self.grid.loc[self._rows[(z, la)]]
        dist_mod = 5.0 * np.log10(distance_pc) - 5.0
        t = Table({"Mini": g["Mini"].to_numpy(), "label": g["label"].to_numpy(),
                   "bp_rp": g["bp_rp"].to_numpy() + ebprp,
                   "Gmag": g["Gmag"].to_numpy() + C.K_AG_EBPRP * ebprp + dist_mod})
        t.meta.update(logAge=la, Zini=z)
        return t


def train_interpolator(grid, labels=C.LABELS_FIT, mini_max=15.0, n_jobs=-1):
    """Train the (logAge, Zini, Mini) -> (Gmag, bp_rp) interpolator.

    A fully grown decision tree reproduces every grid point exactly; because the
    sampler always evaluates on grid nodes, no generalisation off the grid is
    needed. This is the interpolator used in Plevne & Akbaba (2026).
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.tree import DecisionTreeRegressor

    if not isinstance(grid, pd.DataFrame):
        grid = grid.to_pandas()
    train = grid[grid["label"].isin(labels) & (grid["Mini"] < mini_max)]
    model = MultiOutputRegressor(
        DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=42),
        n_jobs=n_jobs)
    model.fit(train[["logAge", "Zini", "Mini"]].to_numpy(),
              train[["Gmag", "bp_rp"]].to_numpy())
    return model
