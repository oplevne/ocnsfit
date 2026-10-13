# Method

Full description: Plevne & Akbaba (2026), Section 3.

## Parameters

θ = (log Age, Z<sub>ini</sub>, d, E(BP−RP)). For a given θ the model CMD is the
nearest-node grid isochrone shifted by:

* colour: BP−RP + E(BP−RP)
* magnitude: G + 1.890 E(BP−RP) + 5 log<sub>10</sub> d − 5

The extinction coefficients are from Wang & Chen (2019).

## Likelihood

For each member *i* with colour *c<sub>i</sub>*, magnitude *G<sub>i</sub>* and
errors σ inflated by a factor 1.5:

χ²<sub>i</sub> = min<sub>k</sub> [ (c<sub>k</sub> − c<sub>i</sub>)²/σ<sub>c,i</sub>² + (G<sub>k</sub> − G<sub>i</sub>)²/σ<sub>G,i</sub>² ]

Here *k* runs over the isochrone points. χ²<sub>i</sub> is clipped at 50, and

ln L = −½ Σ<sub>i</sub> w<sub>i</sub> [ χ²<sub>i</sub> + ln(2π σ<sub>c,i</sub> σ<sub>G,i</sub>) ]

The weights w<sub>i</sub> are the membership probabilities, normalised to unit
sum. The clipping and the error inflation make the fit robust to field
contaminants and photometric outliers.

## Sampling

Dynamic nested sampling with `dynesty` (multi-ellipsoid bounds, random walk,
400 live points, Δln Z = 0.01). The output is the weighted posterior and the
evidence ln Z. Reported values are weighted medians with 16th/84th
percentiles.

## Quality metric η<sub>fit</sub>

η<sub>fit</sub> is the fraction of members whose colour lies within 0.35 mag of
the posterior-median isochrone, measured at the isochrone point nearest to each
star in the CMD. In the paper, η<sub>fit</sub> ≥ 0.80 defines the high-quality
subset.

* **Default** (`source="grid"`): uses the grid isochrone, phases PMS/MS, SGB
  and RGB.
* **Paper catalogue** (`cluster.eta_fit(grid, theta, source="parsec")`):
  downloads the full PARSEC isochrone, including later phases up to the early
  AGB. Values can differ slightly for clusters with many evolved members.

## Known limitation

The likelihood is symmetric about the isochrone. Unresolved binaries,
differential reddening and residual contamination all move stars redwards, so
the fitted isochrone tends to follow the ridge of the main sequence rather than
its blue edge (see the paper, Section 5.6). An asymmetric colour-residual
likelihood or an explicit binary-sequence component are natural extensions.
