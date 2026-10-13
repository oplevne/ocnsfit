# Output

## HDF5 chain format

`FitResult.save()` writes the same format as the published Zenodo chains. All
datasets are in the group `ns/`:

| Dataset | Shape | Description |
|---|---|---|
| `samples` | (N, 4) | `logAge` [dex], `Zini`, `distance_pc` [pc], `E_BPRP` [mag] |
| `weights` | (N,) | normalised importance weights |
| `log_wts` | (N,) | log importance weights |
| `logl` | (N,) | log-likelihood of each sample |
| `log_z`, `log_z_err` | (1,) | log-evidence and its uncertainty |

Files written by OCNSFIT also carry the attributes `name`, `parameters` and
`eta_fit`.

```python
from ocnsfit import FitResult
res = FitResult.load("NGC6791_ns_ucc.h5")     # works for the Zenodo chains too
res.summary()
```

## Conversions

| Quantity | Formula |
|---|---|
| [Fe/H] | log<sub>10</sub>(Z<sub>ini</sub> / 0.0152) |
| A<sub>G</sub> | 1.890 E(BP−RP) |
| A<sub>V</sub> | 3.1 / 1.305 × E(BP−RP) |

The extinction coefficients are from Wang & Chen (2019).
