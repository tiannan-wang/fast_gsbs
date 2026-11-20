"""Small fast_gsbs example on synthetic data:  python example.py"""
import numpy as np

import fast_gsbs
from fast_gsbs import GSBS

# 120 timepoints x 50 features with 6 planted states
rng = np.random.default_rng(0)
lengths = [15, 25, 20, 10, 30, 20]
labels = np.repeat(np.arange(len(lengths)), lengths)
X = rng.standard_normal((len(lengths), 50))[labels] * 0.8 + rng.standard_normal((labels.size, 50))

g = GSBS(kmax=20, x=X)
g.fit(showProgressBar=False)
print("fast_gsbs", fast_gsbs.__version__)
print("planted boundaries :", np.flatnonzero(np.diff(labels)) + 1)
print("nstates            :", g.nstates)
print("bounds (positions) :", np.flatnonzero(g.bounds))
print("tdists[2:8]        :", np.round(g.tdists[2:8], 3))
print("get_bounds(4)      :", np.flatnonzero(g.get_bounds(4)))

# NaN/Inf values and constant timepoints are rejected
bad = X.copy()
bad[10, 3] = np.nan
try:
    GSBS(kmax=20, x=bad)
except ValueError as e:
    print("rejected:", str(e).split(".")[0])
