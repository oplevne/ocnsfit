#!/usr/bin/env python
"""Fit many clusters from a member table, optionally split into chunks.

Each cluster is written to ``<outdir>/chains/<name>_ns_ucc.h5``; clusters whose
chain already exists are skipped, so interrupted runs can simply be restarted.
A summary table ``<outdir>/results[_chunkNNNN].ecsv`` is written at the end.

Usage::

    python scripts/run_batch.py members.parquet --grid isochrones/ \\
        --priors priors.ecsv --outdir run/ --chunk 0 --total-chunks 64

``members`` must contain a cluster-name column (``--name-col``, default
``name``) and the photometric/astrometric columns of the UCC member file
(see ``ocnsfit.UCC_COLUMNS``).
"""

import argparse
import os
import traceback

import numpy as np
from astropy.table import Table, vstack

from ocnsfit import Cluster, IsochroneGrid, Priors
from ocnsfit.priors import read_prior_table


def read_members(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return Table.from_pandas(pd.read_parquet(path))
    return Table.read(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("members")
    ap.add_argument("--grid", default="isochrones")
    ap.add_argument("--priors", help="prior table (optional)")
    ap.add_argument("--outdir", default="run")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--only", nargs="*", help="fit only these clusters")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--total-chunks", type=int, default=1)
    ap.add_argument("--nlive", type=int, default=400)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    members = read_members(a.members)
    names = np.unique(np.asarray(members[a.name_col]).astype(str))
    if a.only:
        names = np.array([n for n in names if n in set(a.only)])
    size = -(-len(names) // a.total_chunks)
    names = names[a.chunk * size:(a.chunk + 1) * size]
    print(f"{len(names)} clusters in chunk {a.chunk}/{a.total_chunks}")

    grid = IsochroneGrid.load(a.grid)
    prior_tab = read_prior_table(a.priors) if a.priors else None
    chain_dir = os.path.join(a.outdir, "chains")
    os.makedirs(chain_dir, exist_ok=True)

    rows, failed = [], []
    col = np.asarray(members[a.name_col]).astype(str)
    for name in names:
        h5 = os.path.join(chain_dir, f"{name}_ns_ucc.h5")
        if os.path.exists(h5):
            print(f"[skip] {name}: chain exists")
            continue
        try:
            cl = Cluster.from_table(members[col == name], name)
            if prior_tab is not None:
                cl.set_priors(Priors.from_table(prior_tab, name))
            print(cl, cl.priors, sep="\n")
            res = cl.fit(grid, nlive=a.nlive, workers=a.workers, print_progress=False)
            res.save(h5)
            rows.append(res.to_row())
            print(res)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    sfx = f"_chunk{a.chunk:04d}" if a.total_chunks > 1 else ""
    if rows:
        out = os.path.join(a.outdir, f"results{sfx}.ecsv")
        vstack(rows).write(out, overwrite=True)
        print(f"wrote {out}")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
