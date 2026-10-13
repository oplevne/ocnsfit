"""Physical constants and default settings used throughout OCNSFIT.

The defaults reproduce the configuration of Plevne & Akbaba (2026, MNRAS).
"""

# Solar metallicity of the PARSEC v1.2S models (Bressan et al. 2012).
Z_SUN = 0.0152

# Gaia extinction coefficients (Wang & Chen 2019, Table 3).
K_AG_EBPRP = 1.890   # A_G / E(BP-RP)
K_EBPRP_EBV = 1.305  # E(BP-RP) / E(B-V)

# Hard parameter bounds of the model.
LOGAGE_BOUNDS = (6.0, 10.0)   # log(age / yr)
FEH_BOUNDS = (-0.5, 0.5)      # default [Fe/H] range when no prior is given
EBPRP_BOUNDS = (0.0, 5.0)     # mag
DISTANCE_MIN_PC = 1.0         # lower floor on sampled distances

# Likelihood settings.
ERR_INFLATE = 1.5   # multiplicative inflation of photometric errors
CHI2_MAX = 50.0     # per-star chi^2 clipping (robustness against outliers)
DEFAULT_MAG_ERR = 0.05  # used when a photometric error is missing or non-positive

# Member selection.
MIN_PROB = 0.75               # minimum membership probability
COLOR_RANGE = (-1.0, 4.0)     # accepted BP-RP range
G_RANGE = (0.0, 20.0)         # accepted G range
MIN_MEMBERS = 3

# Distance prior from member parallaxes.
DIST_SIGMA_FLOOR_FRAC = 0.02  # sigma_d >= 2 per cent of d
DIST_SIGMA_CAP_FRAC = 0.50    # sigma_d <= 50 per cent of d

# Isochrone phases used in the fit (PARSEC labels: 1 = PMS/MS, 2 = SGB, 3 = RGB).
LABELS_FIT = (1, 2, 3)

# Quality metric eta_fit: fraction of members within this colour band of the
# best-fitting isochrone.
ETA_FIT_BAND = 0.35  # mag

# Nested sampling (dynesty) defaults.
NLIVE = 400
NLIVE_BATCH = 5000
DLOGZ_INIT = 0.01
MAXITER = 50000
BOUND = "multi"
SAMPLE = "rwalk"

# Order of the sampled parameters.
PARAM_NAMES = ("logAge", "Zini", "distance_pc", "E_BPRP")
