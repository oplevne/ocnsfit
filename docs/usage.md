# Usage

## Building a `Cluster`

A cluster is defined by its members.

```python
from ocnsfit import Cluster

# 1. From an astropy Table / QTable or a pandas DataFrame
cl = Cluster.from_table(members, name="NGC6791")

# 2. From arrays or Quantities
cl = Cluster("NGC6791", bp_rp=..., gmag=..., e_bp_rp=..., e_gmag=...,
             parallax=plx * u.mas, e_parallax=eplx * u.mas, prob=p,
             ra=ra, dec=dec)

# 3. From the UCC member file (Perren et al. 2023)
cl = Cluster.from_ucc(ucc_members, "NGC6791")
```

### Column names

`from_table` first looks for the UCC/VizieR column names
(`ocnsfit.UCC_COLUMNS`), then for the *Gaia* archive names
(`ocnsfit.GAIA_COLUMNS`). You can override any of them:

```python
cl = Cluster.from_table(tab, "Melotte22",
                        columns=dict(gmag="phot_g_mean_mag", e_gmag="g_err",
                                     bp_rp="bp_rp", e_bp_rp="bp_rp_err",
                                     prob="membership"))
```

Columns with units are converted: parallaxes to mas, coordinates to deg,
magnitudes to mag. The *Gaia* archive gives flux errors, not magnitude errors;
compute them first, e.g. σ<sub>G</sub> ≈ 1.0857 / `phot_g_mean_flux_over_error`.

### Member selection

Stars are kept if:

* `prob ≥ min_prob` (default 0.75),
* ϖ > 0,
* −1 < BP−RP < 4 and 0 < G < 20,
* all used values are finite.

Non-positive magnitude errors are replaced by 0.05 mag.
`cl.n_members`, `cl.to_table()` and `cl.coord` (a `SkyCoord` of the median
member position) describe the selected sample.

## Fitting

```python
from ocnsfit import IsochroneGrid
grid = IsochroneGrid.load("isochrones/")
result = cl.fit(grid, nlive=400, workers=8, seed=42)
```

`fit` runs `dynesty.DynamicNestedSampler` with the paper settings:

* `nlive=400`,
* `bound="multi"`, `sample="rwalk"`,
* `dlogz=0.01`, `maxiter=50000`.

With `workers > 1` the likelihood is evaluated in a process pool. A fit of a
rich cluster takes a few minutes on 8 cores.

## Results

```python
print(result)              # compact summary
result.summary()           # QTable: p16, median, p84 for each quantity
result.to_row()            # one catalogue row (QTable, with units)
result.coord               # SkyCoord with posterior-median distance
result.eta_fit             # quality metric
result.quantiles()         # dict of arrays
result.derived["A_G"]      # weighted samples of derived quantities
result.save("x.h5"); FitResult.load("x.h5")
result.plot(filename="NGC6791.png")   # paper-style: corner + dereddened CMD
result.plot_cmd(); result.plot_corner()
```

Reported quantities:

* `logAge`, `Age_Gyr`
* `FeH`, `Zini`
* `distance_pc`
* `E_BPRP`
* `A_G` (= 1.890 E(BP−RP))

The samples are **weighted**: use `result.weights` for any statistic.
