"""Membership-weighted CMD likelihood.

For each star the squared, error-normalised distance to the closest point of
the model isochrone is computed in (BP-RP, G); it is clipped at ``CHI2_MAX``
for robustness, and the stars are combined with weights proportional to their
membership probabilities (Plevne & Akbaba 2026, Section 3.3).
"""

import numpy as np

from . import constants as C

__all__ = ["CMDLikelihood"]


class CMDLikelihood:
    """Picklable log-likelihood ``L(theta)`` for one cluster.

    Parameters
    ----------
    grid : IsochroneGrid
    bp_rp, gmag, e_bp_rp, e_gmag, prob : array_like
        Photometry, uncertainties and membership probabilities of the members.
    err_inflate, chi2_max : float
        Error inflation factor and per-star chi^2 clipping value.
    """

    def __init__(self, grid, bp_rp, gmag, e_bp_rp, e_gmag, prob,
                 err_inflate=C.ERR_INFLATE, chi2_max=C.CHI2_MAX):
        self.grid = grid
        self.bp_rp = np.asarray(bp_rp, float)
        self.gmag = np.asarray(gmag, float)
        self.e_c = np.asarray(e_bp_rp, float) * err_inflate
        self.e_g = np.asarray(e_gmag, float) * err_inflate
        self.chi2_max = chi2_max
        w = np.clip(np.asarray(prob, float), 0, None)
        self.w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        self._log_norm = np.log(2 * np.pi * self.e_c * self.e_g)

    def __call__(self, theta):
        color, gmag = self.grid.model_cmd(theta)
        if len(color) < 2:
            return -1e300
        d2 = ((color[:, None] - self.bp_rp[None, :]) ** 2 / self.e_c ** 2
              + (gmag[:, None] - self.gmag[None, :]) ** 2 / self.e_g ** 2)
        chi2 = np.minimum(d2.min(axis=0), self.chi2_max)
        return float(-0.5 * np.sum(self.w * (chi2 + self._log_norm)))
