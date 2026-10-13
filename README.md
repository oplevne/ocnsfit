# OCNSFIT — Open Cluster Nested Sampling Fit

OCNSFIT estimates the **age, metallicity, distance and reddening** of open star
clusters by fitting PARSEC isochrones to *Gaia* colour–magnitude diagrams
(CMDs) with Bayesian **dynamic nested sampling** (`dynesty`).

It is the code behind the homogeneous catalogue of 5056 open clusters of
[Plevne & Akbaba (2026, MNRAS)](#citation). The full posterior chains of that
catalogue are available on Zenodo
([doi:10.5281/zenodo.23349618](https://doi.org/10.5281/zenodo.23349618)).

```python
from astropy.table import Table
from ocnsfit import Cluster, IsochroneGrid, Gaussian, Uniform

grid = IsochroneGrid.load("isochrones/")              # PARSEC grid + interpolator

members = Table.read("NGC6791_members.fits")           # Gaia DR3 members
cl = Cluster.from_table(members, name="NGC6791")       # selects prob >= 0.75
cl.set_priors(feh=Gaussian(0.3, 0.35), ebprp=Gaussian(0.17, 0.35),
              logage=Uniform(9.0, 10.0))

result = cl.fit(grid)                                  # nested sampling
print(result)                                          # medians and 16/84 percentiles
result.summary()                                       # astropy QTable with units
result.save("NGC6791_ns.h5")
result.plot(filename="NGC6791.png")              # corner + CMD summary figure
```

## Features

* **Four-parameter model:** log(age), [Fe/H] (via *Z*<sub>ini</sub>), distance and
  *E*(*G*<sub>BP</sub>−*G*<sub>RP</sub>), with *A*<sub>G</sub> = 1.890 *E*(BP−RP)
  (Wang & Chen 2019).
* **Membership-weighted, outlier-robust CMD likelihood:** clipped χ², inflated
  photometric errors.
* **Flexible priors:** each parameter can take a uniform or a truncated Gaussian
  prior, set in code or read from a prior table. The distance prior is built
  automatically from member parallaxes.
* **Astropy-native:** clusters can be built from `Table`/`QTable` (units
  respected) or pandas objects. Results come back as `QTable`s with units and a
  `SkyCoord` carrying the posterior distance.
* **Quality metric η<sub>fit</sub>:** the fraction of members within 0.35 mag
  in colour of the best-fitting isochrone.
* **Batch mode:** chunked, restartable runs over thousands of clusters.

## Installation

```bash
git clone https://github.com/oplevne/ocnsfit
cd ocnsfit
pip install -e .
```

Requirements: Python ≥ 3.9, numpy, scipy, pandas, astropy, scikit-learn,
dynesty, h5py, pyarrow, matplotlib and corner. The grid download also needs
requests and lxml.

## Setting up the isochrone grid (once)

The grid files (~150 MB) are not part of the repository. Build them with:

```bash
python scripts/build_grid.py --outdir isochrones/        # download from PARSEC (~10 min)
python scripts/train_interpolator.py --dir isochrones/   # train interpolator (~1 min)
```

This downloads PARSEC v1.2S isochrones in the *Gaia* EDR3 passbands over the
grid of the paper: [M/H] from −1.5 to +0.4 dex in steps of 0.025 dex, and
log(Age/yr) from 6.0 to 10.0 in steps of 0.05 dex (≤ 8.5) and 0.10 dex
(> 8.5). See [docs/isochrones.md](docs/isochrones.md).

## Documentation

| Topic | File |
|---|---|
| Getting started, the `Cluster` object | [docs/usage.md](docs/usage.md) |
| Priors and the prior-file format | [docs/priors.md](docs/priors.md) |
| Isochrone grid and interpolator | [docs/isochrones.md](docs/isochrones.md) |
| Model, likelihood and η<sub>fit</sub> | [docs/method.md](docs/method.md) |
| Output and HDF5 chain format | [docs/output.md](docs/output.md) |
| Running thousands of clusters | [docs/batch.md](docs/batch.md) |

## Citation

If you use OCNSFIT, please cite:

> Plevne, O. & Akbaba, F. 2026, MNRAS, *Astrophysical Parameters of 5056 Open
> Star Clusters from Bayesian Nested Sampling with PARSEC Isochrones*

```bibtex
@ARTICLE{Plevne2026,
       author = {{Plevne}, Olcay and {Akbaba}, Furkan},
        title = "{Astrophysical Parameters of 5056 Open Star Clusters from Bayesian Nested Sampling with PARSEC Isochrones}",
      journal = {Monthly Notices of the Royal Astronomical Society},
         year = 2026,
       volume = {XXX},
        pages = {XXX},
          doi = {10.1093/mnras/XXXX},
}
```

Please also cite the software DOI (see [`CITATION.cff`](CITATION.cff)) and the
underlying tools: PARSEC (Bressan et al. 2012), dynesty (Speagle 2020) and
*Gaia* DR3.

## Authors

* **Olcay Plevne** — Faculty of Science, Department of Astronomy and Space
  Sciences, Istanbul University, Istanbul, Türkiye; Institute for Astronomy,
  University of Edinburgh, Royal Observatory, Edinburgh, UK
  ([ORCID 0000-0002-0435-4493](https://orcid.org/0000-0002-0435-4493);
  olcayplevne@istanbul.edu.tr)
* **Furkan Akbaba** — Institute of Graduate Studies in Science, Istanbul
  University, Istanbul, Türkiye
  ([ORCID 0000-0002-9993-7244](https://orcid.org/0000-0002-9993-7244))

## Licence

MIT — see [LICENSE](LICENSE). `ocnsfit/parsec.py` is adapted from the PARSEC
query script by Zhaozhou Li
([syrte/query_isochrone](https://github.com/syrte/query_isochrone), MIT License).
