"""fast_gsbs: a faster implementation of GSBS (Greedy State Boundary Search).

    from fast_gsbs import GSBS
    g = GSBS(kmax=30, x=X)          # X: timepoints x features
    g.fit(showProgressBar=False)
    g.nstates, g.bounds, g.tdists

Same arguments and outputs as statesegmentation.GSBS 0.0.6. The README lists the limitations.
"""
from .core import DegenerateStateWarning, GSBSFast

GSBS = GSBSFast
__version__ = "1.0.0"
__all__ = ["GSBS", "GSBSFast", "DegenerateStateWarning", "__version__"]
