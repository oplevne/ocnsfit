# Priors

Every parameter has either a **uniform** or a **truncated Gaussian** prior.
OCNSFIT never builds priors itself, for example from dust maps or literature
catalogues; you provide them. Any calibration therefore belongs in your prior
values. In Plevne & Akbaba (2026), for instance, the SFD reddening was rescaled
by 0.86 (Schlafly & Finkbeiner 2011) inside the prior file.

| Parameter | Keyword | Unit | Default | Hard bounds |
|---|---|---|---|---|
| log(age/yr) | `logage` | dex | Uniform[6, 10] | [6, 10] |
| [Fe/H] | `feh` | dex | Uniform[−0.5, +0.5] | your `low`/`high` |
| E(BP−RP) | `ebprp` | mag | Uniform[0, 5] | [0, 5] |
| distance | `distance` | pc | Gaussian from parallaxes | > 0 |

[Fe/H] is converted to the PARSEC initial metallicity with
Z<sub>ini</sub> = 0.0152 × 10<sup>[Fe/H]</sup>.

## In code

```python
from ocnsfit import Priors, Uniform, Gaussian

priors = Priors(
    logage=Uniform(8.0, 9.5),
    feh=Gaussian(0.0, 0.35, low=-0.5, high=0.5),   # truncated Gaussian
    ebprp=(0.42, 0.20),                            # tuple = Gaussian(mu, sigma)
    distance=None,                                 # parallax-based (default)
)
cluster.set_priors(priors)
# or directly
cluster.set_priors(feh=Gaussian(0.0, 0.35), ebprp=Uniform(0.2, 1.0))
```

Gaussians without explicit `low`/`high` are truncated to the hard bounds.

### Default distance prior

When `distance=None`, the cluster builds a Gaussian prior from its members:

* **Centre:** median of 1000/ϖ over members with σ<sub>ϖ</sub>/ϖ ≤ 0.2. If
  fewer than 3 members qualify, the limit is relaxed to 0.5, then dropped.
* **Width:** 1.4826 × MAD / √N, i.e. the standard error of the median, clipped
  to 2–50 per cent of the centre.

The prior can be inspected with `cluster.distance_prior`.

## Prior files

`Priors.from_file(path, name)` reads the row of cluster `name` from any table
that `astropy.table.Table.read` understands (ECSV, FITS, CSV, VOTable), or
from a `.parquet` file. The name column is the first one found among `name`,
`cluster`, `Cluster` and `display_name`; you can override it with `name_col=`.

Columns per parameter `<par>` ∈ {`logAge`, `feh`, `E_BPRP`, `dist`}:

| Column | Meaning |
|---|---|
| `<par>_type` | `gauss` or `uniform` (optional, see below) |
| `<par>_mu`, `<par>_sigma` | Gaussian centre and width |
| `<par>_lo`, `<par>_hi` | Uniform range, or Gaussian truncation |

When `<par>_type` is absent or empty, the type is inferred:

* finite `_mu` and `_sigma` → Gaussian;
* otherwise, a finite `_lo` or `_hi` → Uniform;
* otherwise → the default prior.

This inference also reads the prior file of the published run directly.

Example (`priors.ecsv`):

```
# %ECSV 1.0
# ---
# datatype:
# - {name: name, datatype: string}
# - {name: logAge_type, datatype: string}
# - {name: logAge_mu, datatype: float64}
# - {name: logAge_sigma, datatype: float64}
# - {name: feh_type, datatype: string}
# - {name: feh_mu, datatype: float64}
# - {name: feh_sigma, datatype: float64}
# - {name: feh_lo, datatype: float64}
# - {name: feh_hi, datatype: float64}
# - {name: E_BPRP_type, datatype: string}
# - {name: E_BPRP_mu, datatype: float64}
# - {name: E_BPRP_sigma, datatype: float64}
name logAge_type logAge_mu logAge_sigma feh_type feh_mu feh_sigma feh_lo feh_hi E_BPRP_type E_BPRP_mu E_BPRP_sigma
NGC6791 uniform nan nan gauss 0.32 0.35 -0.5 0.5 gauss 0.17 0.35
NGC2682 gauss 9.6 0.5 uniform nan nan -0.5 0.5 gauss 0.04 0.20
```

If a cluster is missing from the file, a warning is issued and the default
priors are used.
