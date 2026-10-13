#!/usr/bin/env python
"""Download the PARSEC isochrone grid used by OCNSFIT.

Grid of Plevne & Akbaba (2026, Section 3.1):

* PARSEC v1.2S isochrones (CMD 3.7 web interface, default tracks), Gaia EDR3
  passbands (G, G_BP, G_RP);
* [M/H] = -1.5 ... +0.4 dex in steps of 0.025 dex
  (Z_ini ~ 0.0005-0.038 for Z_sun = 0.0152);
* log(Age/yr) = 6.0 ... 8.5 in steps of 0.05 and 8.6 ... 10.0 in steps of 0.10;
* phases PMS/MS, SGB and RGB (PARSEC labels 1-3).

The CMD server returns at most ~400 isochrones per request, so the grid is
fetched in [M/H] batches. Output: ``<outdir>/isochrone_grid.parquet``.
Then run ``train_interpolator.py``.

Usage::

    python scripts/build_grid.py --outdir isochrones/
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from ocnsfit import constants as C
from ocnsfit.isochrones import GRID_FILE
from ocnsfit.parsec import ParsecQuery

PHOTSYS = "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat"
MAX_PER_QUERY = 400


def grid_axes(meh_min=-1.5, meh_max=0.4, dmeh=0.025):
    meh = np.round(np.arange(meh_min, meh_max + dmeh / 2, dmeh), 4)
    young = np.round(np.arange(6.00, 8.50 + 0.025, 0.05), 3)
    old = np.round(np.arange(8.60, 10.00 + 0.05, 0.10), 3)
    return meh, young, old


def fetch(cmd, lgt, meh, retries=3):
    for attempt in range(1, retries + 1):
        try:
            tab = cmd.query_isochrones(lgt=lgt, MeH=meh, photsys_file=PHOTSYS)
            df = tab.to_pandas()
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as exc:  # network / server errors
            print(f"    attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="isochrones")
    ap.add_argument("--meh-min", type=float, default=-1.5)
    ap.add_argument("--meh-max", type=float, default=0.4)
    ap.add_argument("--dmeh", type=float, default=0.025)
    a = ap.parse_args()

    meh, young, old = grid_axes(a.meh_min, a.meh_max, a.dmeh)
    print(f"[M/H]: {len(meh)} values ({meh[0]:+.3f} .. {meh[-1]:+.3f})")
    print(f"logAge: {len(young)} young (6.00-8.50) + {len(old)} old (8.6-10.0)")
    print(f"nominal grid: {len(meh) * (len(young) + len(old))} isochrones")

    cmd = ParsecQuery()
    frames = []
    for band, lgt in (("young", young), ("old", old)):
        batch = max(1, MAX_PER_QUERY // len(lgt))
        for i in range(0, len(meh), batch):
            mb = meh[i:i + batch]
            print(f"  {band}: [M/H] {mb[0]:+.3f}..{mb[-1]:+.3f}", flush=True)
            frames.append(fetch(cmd, lgt, mb))
            time.sleep(1)

    grid = pd.concat(frames, ignore_index=True)
    grid = grid[grid["label"].isin(C.LABELS_FIT)].copy()
    grid["bp_rp"] = grid["G_BPmag"] - grid["G_RPmag"]
    grid = grid.dropna(subset=["Gmag", "bp_rp", "Mini", "logAge", "Zini"])
    grid = grid.reset_index(drop=True)

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, GRID_FILE)
    grid.to_parquet(out, index=False)
    n_iso = grid.groupby(["Zini", "logAge"]).ngroups
    print(f"wrote {out}: {len(grid):,} points, {n_iso} isochrones")


if __name__ == "__main__":
    main()
