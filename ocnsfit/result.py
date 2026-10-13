"""Fit results: posterior summaries, quality metric, HDF5 I/O and plots."""

import h5py
import numpy as np

from . import constants as C

__all__ = ["FitResult", "weighted_quantile", "eta_fit"]


def weighted_quantile(values, weights, q):
    """Quantile(s) ``q`` of weighted samples (linear interpolation of the CDF)."""
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return np.interp(q, cdf, values[order])


def eta_fit(bp_rp, gmag, iso_bp_rp, iso_g, band=C.ETA_FIT_BAND):
    """Fraction of stars whose colour lies within ``band`` of the isochrone.

    Each star is matched to the closest isochrone point in (BP-RP, G); the star
    counts as an inlier if ``|Delta(BP-RP)| < band`` at that point.
    """
    bp_rp, gmag = np.asarray(bp_rp, float), np.asarray(gmag, float)
    d2 = (iso_bp_rp[None, :] - bp_rp[:, None]) ** 2 + (iso_g[None, :] - gmag[:, None]) ** 2
    nearest = np.argmin(d2, axis=1)
    return float(np.mean(np.abs(bp_rp - iso_bp_rp[nearest]) < band))


class FitResult:
    """Posterior of one cluster fit.

    Attributes
    ----------
    name : str
    samples : ndarray (N, 4)
        Nested-sampling samples of ``(logAge, Zini, distance_pc, E_BPRP)``.
    weights : ndarray (N,)
        Normalised importance weights; always use them for statistics.
    log_wts, logl : ndarray (N,)
    log_z, log_z_err : float
        Log-evidence and its uncertainty.
    eta_fit : float or None
        Quality metric (fraction of members within 0.35 mag in colour of the
        median-parameter isochrone).
    cluster : Cluster or None
        The fitted cluster (not stored in HDF5).
    """

    def __init__(self, name, samples, weights, log_wts=None, logl=None,
                 log_z=np.nan, log_z_err=np.nan, eta_fit=None, cluster=None,
                 grid=None, meta=None):
        self.name = name
        self.samples = np.asarray(samples)
        self.weights = np.asarray(weights)
        self.log_wts = log_wts
        self.logl = logl
        self.log_z = float(log_z)
        self.log_z_err = float(log_z_err)
        self.eta_fit = eta_fit
        self.cluster = cluster
        self.grid = grid
        self.meta = dict(meta or {})

    @classmethod
    def from_dynesty(cls, name, results, **kwargs):
        log_wts = results.logwt
        w = np.exp(log_wts - log_wts.max())
        return cls(name, results.samples, w / w.sum(), log_wts=log_wts,
                   logl=results.logl, log_z=results.logz[-1],
                   log_z_err=results.logzerr[-1], **kwargs)

    # ------------------------------------------------------------ derived
    @property
    def derived(self):
        """Dict of sample arrays of all reported quantities."""
        s = self.samples
        return {"logAge": s[:, 0],
                "Age_Gyr": 10 ** s[:, 0] / 1e9,
                "FeH": np.log10(s[:, 1] / C.Z_SUN),
                "Zini": s[:, 1],
                "distance_pc": s[:, 2],
                "E_BPRP": s[:, 3],
                "A_G": C.K_AG_EBPRP * s[:, 3]}

    def quantiles(self, q=(0.16, 0.5, 0.84)):
        """``{quantity: array of quantiles}``."""
        return {k: weighted_quantile(v, self.weights, q) for k, v in self.derived.items()}

    @property
    def median_theta(self):
        """Weighted median of ``(logAge, Zini, distance_pc, E_BPRP)``."""
        return np.array([weighted_quantile(self.samples[:, i], self.weights, 0.5)
                         for i in range(4)])

    _UNITS = {"logAge": "dex", "Age_Gyr": "Gyr", "FeH": "dex", "Zini": "",
              "distance_pc": "pc", "E_BPRP": "mag", "A_G": "mag"}

    def summary(self):
        """Astropy QTable with median and 16th/84th percentiles per quantity."""
        import astropy.units as u
        from astropy.table import QTable
        rows = self.quantiles()
        units = {"Gyr": u.Gyr, "pc": u.pc, "mag": u.mag, "dex": u.dex, "": u.one}
        t = QTable()
        t["quantity"] = list(rows)
        for j, col in enumerate(("p16", "median", "p84")):
            t[col] = [rows[k][j] for k in rows]
        t["unit"] = [self._UNITS[k] for k in rows]
        t.meta.update(name=self.name, log_z=self.log_z, eta_fit=self.eta_fit)
        return t

    def to_row(self):
        """One-row astropy QTable (catalogue format) with units."""
        import astropy.units as u
        from astropy.table import QTable
        q = self.quantiles()
        unit = {"Age_Gyr": u.Gyr, "distance_pc": u.pc, "E_BPRP": u.mag, "A_G": u.mag}
        t = QTable()
        t["name"] = [self.name]
        for k, (lo, med, hi) in q.items():
            un = unit.get(k, 1)
            t[f"{k}"] = [med] * un
            t[f"{k}_p16"] = [lo] * un
            t[f"{k}_p84"] = [hi] * un
        t["log_z"] = [self.log_z]
        t["log_z_err"] = [self.log_z_err]
        t["eta_fit"] = [np.nan if self.eta_fit is None else self.eta_fit]
        if self.cluster is not None:
            t["n_members"] = [self.cluster.n_members]
        return t

    @property
    def coord(self):
        """SkyCoord of the cluster with the posterior-median distance."""
        if self.cluster is None or self.cluster.coord is None:
            return None
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        c = self.cluster.coord
        return SkyCoord(ra=c.ra, dec=c.dec, distance=self.median_theta[2] * u.pc, frame="icrs")

    def __repr__(self):
        q = self.quantiles()
        lines = [f"FitResult({self.name!r}, ln Z = {self.log_z:.2f} +/- {self.log_z_err:.2f}"
                 + ("" if self.eta_fit is None else f", eta_fit = {self.eta_fit:.2f}") + ")"]
        for k in ("logAge", "FeH", "distance_pc", "E_BPRP", "A_G"):
            lo, md, hi = q[k]
            lines.append(f"  {k:12s} {md:10.4g}  -{md - lo:.3g} +{hi - md:.3g}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- I/O
    def save(self, path):
        """Write the posterior in the HDF5 format of the published chains."""
        with h5py.File(path, "w") as f:
            g = f.create_group("ns")
            g.create_dataset("samples", data=self.samples)
            g.create_dataset("weights", data=self.weights)
            if self.log_wts is not None:
                g.create_dataset("log_wts", data=self.log_wts)
            if self.logl is not None:
                g.create_dataset("logl", data=self.logl)
            g.create_dataset("log_z", data=np.array([self.log_z]))
            g.create_dataset("log_z_err", data=np.array([self.log_z_err]))
            f.attrs["name"] = self.name
            f.attrs["parameters"] = ",".join(C.PARAM_NAMES)
            if self.eta_fit is not None:
                f.attrs["eta_fit"] = self.eta_fit

    @classmethod
    def load(cls, path, name=None):
        """Read a chain written by :meth:`save` (or a published Zenodo chain)."""
        import os
        with h5py.File(path, "r") as f:
            g = f["ns"]
            opt = {k: g[k][:] for k in ("log_wts", "logl") if k in g}
            name = name or f.attrs.get("name") or \
                os.path.basename(path).replace("_ns_ucc.h5", "").replace(".h5", "")
            return cls(name, g["samples"][:], g["weights"][:],
                       log_z=g["log_z"][0], log_z_err=g["log_z_err"][0],
                       eta_fit=f.attrs.get("eta_fit"), **opt)

    # -------------------------------------------------------------- plots
    def plot_corner(self, **kwargs):
        """Corner plot of logAge, Z_ini, distance and E(BP-RP)."""
        import corner
        d = self.derived
        x = np.c_[d["logAge"], d["Zini"], d["distance_pc"], d["E_BPRP"]]
        kw = dict(labels=[r"$\log(\mathrm{Age/yr})$", r"$Z_\mathrm{ini}$", r"$d$ (pc)",
                          r"$E(G_\mathrm{BP}-G_\mathrm{RP})$"],
                  quantiles=[0.16, 0.5, 0.84], show_titles=True)
        kw.update(kwargs)
        return corner.corner(x, weights=self.weights, **kw)

    def plot(self, grid=None, filename=None, dpi=300):
        """Summary figure in the style of Plevne & Akbaba (2026).

        Corner plot of (log Age, Z_ini, d, E(BP-RP)) with median and 16th/84th
        percentile titles, and the dereddened CMD of the members in the upper
        right with the median isochrone (orange) and the isochrones at the
        16th/84th age percentiles (green, dashed).
        """
        import corner
        import matplotlib.pyplot as plt

        grid = grid or self.grid
        d = self.derived
        x = np.c_[d["logAge"], d["Zini"], d["distance_pc"], d["E_BPRP"]]
        q = np.array([weighted_quantile(x[:, i], self.weights, [0.16, 0.5, 0.84])
                      for i in range(4)])
        labels = [r"$\log(\mathrm{Age/yr})$", r"$Z_\mathrm{ini}$", r"$d\,(\mathrm{pc})$",
                  r"$E(G_\mathrm{BP}-G_\mathrm{RP})$"]
        fmts = [".3f", ".5f", ".0f", ".3f"]
        rng = []
        for i in range(4):
            lo, hi = np.percentile(x[:, i], [0.5, 99.5])
            rng.append((lo, hi) if hi > lo else (lo - 1e-6, hi + 1e-6))

        fig = plt.figure(figsize=(10, 10))
        corner.corner(x, weights=self.weights, labels=labels, smooth=1.25, smooth1d=1.25,
                      quantiles=[0.16, 0.5, 0.84], truths=q[:, 1], truth_color="C1",
                      range=rng, fig=fig, label_kwargs={"fontsize": "x-large"})
        axes = np.array(fig.get_axes()).reshape(4, 4)
        for i in range(4):
            lo, md, hi = q[i]
            f = fmts[i]
            axes[i, i].set_title(f"{labels[i]} = {md:{f}}" + r"$^{+" + f"{hi - md:{f}}"
                                 + r"}_{-" + f"{md - lo:{f}}" + r"}$", fontsize="large")

        if self.cluster is not None and grid is not None:
            cl = self.cluster
            la, z, dist, e = self.median_theta
            ax = fig.add_axes([0.70, 0.56, 0.22, 0.40])
            o = np.argsort(cl.prob)
            sc = ax.scatter(cl.bp_rp[o] - e, cl.gmag[o] - C.K_AG_EBPRP * e, c=cl.prob[o],
                            cmap="coolwarm", vmin=0.5, vmax=1.0, edgecolor="black",
                            linewidths=0.5, s=20, zorder=50)
            age_lo, age_hi = q[0, 0], q[0, 2]
            for lage, col, ls, lab, zo in ((la, "orange", "-", "median", 100),
                                           (age_lo, "green", "--", r"$-1\sigma$ age", 90),
                                           (age_hi, "green", ":", r"$+1\sigma$ age", 90)):
                iso = grid.isochrone(lage, z, dist, 0.0)
                ax.plot(iso["bp_rp"], iso["Gmag"], color=col, ls=ls, lw=1.4, zorder=zo,
                        label=lab)
            g0 = cl.gmag - C.K_AG_EBPRP * e
            c0 = cl.bp_rp - e
            ax.set_xlim(c0.min() - 0.3, c0.max() + 0.3)
            ax.set_ylim(g0.max() + 0.5, g0.min() - 1.0)
            ax.set_xlabel(r"$(G_\mathrm{BP}-G_\mathrm{RP})_0$ (mag)")
            ax.set_ylabel(r"$G_0$ (mag)")
            ax.set_title(f"{self.name}  ($N$ = {cl.n_members})")
            ax.legend(loc="upper left", fontsize="small", framealpha=0.8)
            if self.eta_fit is not None:
                ax.text(0.96, 0.04, r"$\eta_\mathrm{fit}$" + f" = {self.eta_fit:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
            cax = fig.add_axes([0.935, 0.56, 0.015, 0.40])
            fig.colorbar(sc, cax=cax).set_label(r"$P_\mathrm{mem}$")

        if filename:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        return fig

    def plot_cmd(self, ax=None, grid=None):
        """Members on the CMD with the median-parameter grid isochrone."""
        import matplotlib.pyplot as plt
        grid = grid or self.grid
        if self.cluster is None or grid is None:
            raise ValueError("plot_cmd needs the fitted cluster and the isochrone grid")
        ax = ax or plt.gca()
        cl = self.cluster
        o = np.argsort(cl.prob)  # most probable members on top
        sc = ax.scatter(cl.bp_rp[o], cl.gmag[o], c=cl.prob[o], s=8, cmap="viridis",
                        vmin=cl.min_prob, vmax=1)
        la, z, d, e = self.median_theta
        iso = grid.isochrone(la, z, d, e)
        ax.plot(iso["bp_rp"], iso["Gmag"], "r-", lw=1.2,
                label=f"logAge={la:.2f}, [Fe/H]={np.log10(z / C.Z_SUN):+.2f}")
        # limit the axes to the data (the isochrone extends far below the members)
        cpad, gpad = 0.1 * np.ptp(cl.bp_rp) + 0.1, 0.05 * np.ptp(cl.gmag) + 0.3
        ax.set_xlim(cl.bp_rp.min() - cpad, cl.bp_rp.max() + cpad)
        ax.set_ylim(cl.gmag.max() + gpad, cl.gmag.min() - gpad)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(r"$G_\mathrm{BP}-G_\mathrm{RP}$ (mag)")
        ax.set_ylabel(r"$G$ (mag)")
        ax.set_title(self.name)
        plt.colorbar(sc, ax=ax, label="membership probability")
        return ax
