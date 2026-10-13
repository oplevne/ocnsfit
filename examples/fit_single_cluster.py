"""Fit a single cluster end to end.

Members are read from a local table (e.g. the cluster's rows of the UCC
member file of Perren et al. 2023) and fitted with the published settings.

    python examples/fit_single_cluster.py NGC6791 --grid isochrones/ --members members.fits
"""

import argparse

import matplotlib.pyplot as plt
from astropy.table import Table

from ocnsfit import Cluster, Gaussian, IsochroneGrid, Priors, Uniform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--grid", default="isochrones")
    ap.add_argument("--members", required=True, help="member table (FITS/ECSV/CSV/...)")
    ap.add_argument("--priors", help="optional prior table")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    grid = IsochroneGrid.load(a.grid)
    cl = Cluster.from_table(Table.read(a.members), a.name)
    print(cl)

    if a.priors:
        cl.set_priors(Priors.from_file(a.priors, a.name))
    else:
        # Example of hand-set priors: flat age, Gaussian metallicity and reddening.
        cl.set_priors(logage=Uniform(6.0, 10.0), feh=Gaussian(0.0, 0.35, -0.5, 0.5),
                      ebprp=Gaussian(0.3, 0.35))
    print(cl.priors)

    res = cl.fit(grid, workers=a.workers)
    print(res)
    res.summary().pprint_all()
    res.save(f"{a.name}_ns.h5")

    fig, ax = plt.subplots(figsize=(5, 6))
    res.plot_cmd(ax=ax)
    fig.savefig(f"{a.name}_cmd.png", dpi=150, bbox_inches="tight")
    res.plot_corner().savefig(f"{a.name}_corner.png", dpi=150)


if __name__ == "__main__":
    main()
