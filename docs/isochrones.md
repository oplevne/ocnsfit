# Isochrone grid and interpolator

## Grid (`scripts/build_grid.py`)

| | Value |
|---|---|
| Models | PARSEC v1.2S (Bressan et al. 2012; Chen et al. 2014; Marigo et al. 2017), CMD 3.7 web interface |
| Photometry | *Gaia* EDR3 passbands (`YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat`) |
| [M/H] | −1.5 to +0.4 dex, step 0.025 dex (Z<sub>ini</sub> ≈ 0.0005–0.038) |
| log(Age/yr) | 6.0–8.5 (step 0.05), 8.6–10.0 (step 0.10) |
| Phases | PARSEC labels 1–3 (PMS/MS, SGB, RGB) |

Run:

```bash
python scripts/build_grid.py --outdir isochrones/ [--meh-min -1.5 --meh-max 0.4 --dmeh 0.025]
```

The CMD server accepts about 400 isochrones per request, so the script sends
several batched queries. Failed requests are retried. The output is
`isochrones/isochrone_grid.parquet`, with all PARSEC columns plus `bp_rp`.

## Interpolator (`scripts/train_interpolator.py`)

A fully grown decision tree (scikit-learn) maps (log Age, Z<sub>ini</sub>,
M<sub>ini</sub>) to (G, BP−RP). It reproduces every grid point exactly. Since
the sampler snaps each proposal to the nearest grid node with a k-d tree, no
off-grid generalisation is needed. The result is a fast lookup (below 1 ms per
likelihood call) that keeps the native PARSEC mass sampling along each
isochrone.

```bash
python scripts/train_interpolator.py --dir isochrones/
```

The output is `isochrones/iso_interpolator.pkl`, about 125 MB.

## Loading

```python
from ocnsfit import IsochroneGrid
grid = IsochroneGrid.load("isochrones/")
grid.isochrone(9.6, 0.0152, distance_pc=1000, ebprp=0.3)  # astropy Table
```

Use `grid_file=` and `interp_file=` to load files with other names.
