#!/usr/bin/env python
"""Train the isochrone interpolator on the downloaded grid.

Input : ``<dir>/isochrone_grid.parquet`` (from ``build_grid.py``)
Output: ``<dir>/iso_interpolator.pkl``

Usage::

    python scripts/train_interpolator.py --dir isochrones/
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from ocnsfit.isochrones import GRID_FILE, INTERP_FILE, train_interpolator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="isochrones")
    a = ap.parse_args()

    grid = pd.read_parquet(os.path.join(a.dir, GRID_FILE))
    t0 = time.time()
    model = train_interpolator(grid)
    print(f"trained in {time.time() - t0:.0f} s")

    # Sanity check: the tree must reproduce grid points exactly.
    s = grid.sample(min(5000, len(grid)), random_state=1)
    pred = model.predict(s[["logAge", "Zini", "Mini"]].to_numpy())
    err = np.abs(pred - s[["Gmag", "bp_rp"]].to_numpy()).max()
    print(f"max |error| on grid points: {err:.2e} mag")

    out = os.path.join(a.dir, INTERP_FILE)
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
