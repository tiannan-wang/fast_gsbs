"""GSBS with a running-sum evaluation of candidate boundaries.

GSBSFast subclasses the statesegmentation 0.0.6 class, which is copied into _reference_base.py.
The search loop, fine-tuning, the two-boundary step and the t-distance run from that code
unchanged. Only the three methods that score candidate boundaries are replaced here.

The reference scores a segmentation as the mean, over timepoints, of the correlation between x[t]
and the mean pattern of the state that contains t. Write x[t] - mean(x[t]) = sd[t] * z[t], where
z[t] is x[t] z-scored across the V features. For a state S = [a, b), with

    w = sum of sd[t] * z[t] over S        s = sum of z[t] over S

the correlations within S add up to (w . s) / (||w|| * sqrt(V - 1)). Both w and s are differences
of running sums over time, so a state costs O(V) instead of O(T * V). A candidate boundary splits
one state, so its score is base - contrib(S) + contrib(left part) + contrib(right part). The
score of each interval is cached.

The scores agree with the reference to rounding, not bit for bit, and boundaries are picked by
argmax. Two consequences are known (see the README):
  - candidates with exactly tied scores can be resolved differently from the reference;
  - a state whose mean pattern is constant across features (a "zero-mean state") scores NaN in the
    reference, and the NaN is handled differently here.

x is converted to float64. NaN or Inf values and constant timepoints raise ValueError. The running
sums and the cache stay on the fitted object.
"""
import warnings
from typing import Optional

import numpy as np
from numpy import arange, all, array, equal, max, ndarray, nonzero, ones, sum, unique, unravel_index, where, zeros
from ._reference_base import GSBS

# Warn when the rounding-error bound of a state's centred mean pattern reaches this fraction of
# the pattern's norm.
DEGENERATE_REL_ERR = 1e-6
_EPS = np.finfo(np.float64).eps


class DegenerateStateWarning(UserWarning):
    """A state's centred mean pattern is too close to zero for its score to be reliable."""


def _gamma(n):
    ne = n * _EPS
    return ne / (1.0 - ne)


def _check_finite(arr, name):
    a = np.asarray(arr)
    if a.ndim != 2 or a.dtype.kind not in "biufc":
        return
    bad = ~np.isfinite(a)
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        shown = ", ".join(str(int(i)) for i in rows[:10]) + (", ..." if rows.size > 10 else "")
        raise ValueError(
            f"{name} has {int(bad.sum())} non-finite value(s) (NaN or Inf) in row(s) {shown}. "
            f"Remove or interpolate them before fitting.")


def _check_constant_timepoints(arr, name):
    a = np.asarray(arr)
    if a.ndim != 2:
        return
    const = np.flatnonzero(np.all(a == a[:, :1], axis=1))
    if const.size:
        shown = ", ".join(str(int(i)) for i in const[:10]) + (", ..." if const.size > 10 else "")
        raise ValueError(
            f"{name} has {const.size} constant timepoint(s) (row(s) {shown}). Every timepoint must "
            f"vary across features. If every row is listed, check that the array is timepoints x "
            f"features with at least two features.")


class GSBSFast(GSBS):
    """GSBS with the constructor, methods and properties of statesegmentation.GSBS 0.0.6.

    kmax: largest number of states the search tries. Half the number of timepoints is a reasonable
        choice.
    x: timepoints x features array, for example timepoints x voxels.
    statewise_detection: if True (default), each step adds one boundary or two boundaries inside
        one existing state, whichever gives the higher t-distance. The result can then have
        kmax + 1 states. If False, one boundary is added per step.
    finetune: number of timepoints around each boundary that are searched when the boundaries are
        re-optimised after each step (default 1). 0 switches this off; a negative value searches
        all timepoints, which is slow.
    finetune_order: if True (default), finetune from the weakest boundary to the strongest.
    y: optional second array. If given, the boundaries come from x and the number of states is
        chosen with the t-distance computed on y.
    blocksize: smallest block of timepoints in which the search first looks for local optima
        (default 50).
    dmin: number of timepoints around the diagonal of the time x time correlation matrix that are
        left out of the t-distance (default 1).
    """

    _WARN_DEGENERATE = True

    def __init__(self, kmax: int, x: ndarray, statewise_detection: Optional[bool] = True, finetune: Optional[int] = 1,
                 finetune_order: Optional[bool] = True, y: Optional[ndarray] = None, blocksize: Optional[int] = 50,
                 dmin: Optional[int] = 1) -> None:
        """The reference constructor, then the input checks."""
        super().__init__(kmax, x, statewise_detection, finetune, finetune_order, y, blocksize, dmin)
        self._check_input_contract()

    def _check_input_contract(self):
        """Raise ValueError if x or y has a NaN or Inf value or a constant timepoint."""
        _check_finite(self.x, "x")
        _check_constant_timepoints(self.x, "x")
        if self.y is not None:
            _check_finite(self.y, "y")
            _check_constant_timepoints(self.y, "y")

    def fit(self, showProgressBar=True) -> None:
        """Find the state boundaries and the optimal number of states.

        Returns None: read the results from the properties (nstates, bounds, tdists, ...).
        showProgressBar=True writes a one-line counter to stderr.
        """
        self._check_input_contract()
        x = np.asarray(self.x, dtype=float)
        sd = x.std(1, keepdims=True, ddof=1)
        z = (x - x.mean(1, keepdims=True)) / sd
        T, V = x.shape
        # running sums over time of the centred rows (sd * z) and of the z-scores
        self._W = np.zeros((T + 1, V), float)
        self._Z = np.zeros((T + 1, V), float)
        np.cumsum(sd * z, axis=0, out=self._W[1:])
        np.cumsum(z, axis=0, out=self._Z[1:])
        self._sqrtV1 = np.sqrt(V - 1.0)
        self._cache = {}
        if self._WARN_DEGENERATE:
            # scalars for the rounding-error bound in _degenerate_check
            A = np.cumsum(np.abs(sd * z), axis=0)
            self._nA = np.zeros(T + 1)
            self._nA[1:] = np.sqrt(np.einsum("ij,ij->i", A, A))
            self._R = np.zeros(T + 1)
            np.cumsum(np.abs(x).max(1), out=self._R[1:])
            self._sqrtV = np.sqrt(V)
            self._degenerate_warned = False
            # err(a, b) in _degenerate_check never exceeds errmax, so a state whose norm is above
            # this screen cannot warn and skips the check
            errmax = (_gamma(T + 1) * np.max(self._nA) + _gamma(T + 1) * np.max(self._nA)
                      + self._sqrtV * (_gamma(T + 1) + 2.0 * _gamma(V + 2)) * self._R[T])
            self._degenerate_screen = float(2.0 * errmax / DEGENERATE_REL_ERR)
        else:
            self._degenerate_screen = -np.inf
        return super().fit(showProgressBar=showProgressBar)

    def _contrib(self, a, b):
        """Sum over t in [a, b) of corr(mean pattern of [a, b), x[t]). a and b are global indices."""
        key = (a, b)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        w = self._W[b] - self._W[a]
        s = self._Z[b] - self._Z[a]
        nw = float(np.sqrt(w @ w))
        val = float(w @ s) / (nw * self._sqrtV1)
        if not nw > self._degenerate_screen:
            self._degenerate_check(a, b, nw)
        self._cache[key] = val
        return val

    def _degenerate_check(self, a, b, nw):
        """Warn once per fit if the centred mean pattern of [a, b) is too close to zero.

        err bounds the rounding error of w. If the norm of w is below err / DEGENERATE_REL_ERR, the
        score of the state is not reliable. This only warns: nothing computed depends on it.
        """
        if not self._WARN_DEGENERATE or self._degenerate_warned:
            return
        m = b - a
        V = self._W.shape[1]
        err = (_gamma(b + 1) * self._nA[b] + _gamma(a + 1) * self._nA[a]
               + self._sqrtV * (_gamma(m + 1) + 2.0 * _gamma(V + 2)) * (self._R[b] - self._R[a]))
        if not nw > err / DEGENERATE_REL_ERR:
            self._degenerate_warned = True
            self._degenerate_screen = -np.inf
            warnings.warn(
                f"state [{a}, {b}) has an almost constant mean pattern (norm {nw:.3g}, rounding-error "
                f"bound {float(err):.3g}). Its score is unreliable and the result may differ from "
                f"statesegmentation 0.0.6. See 'Zero-mean states' in the README.",
                DegenerateStateWarning, stacklevel=2)

    def _ranges(self, states):
        """[a, b) ranges of the states, in order, in local indices."""
        s = np.asarray(states)
        cuts = np.flatnonzero(s[1:] != s[:-1]) + 1
        edges = np.concatenate(([0], cuts, [len(s)]))
        return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]

    def _base(self, states, off):
        rng = self._ranges(states)
        return float(np.sum([self._contrib(off + a, off + b) for a, b in rng])), rng

    @staticmethod
    def _find(rng, i):
        for a, b in rng:
            if a <= i < b:
                return a, b
        raise IndexError(i)

    def _wdists_impl(self, deltas, states, off, boundopt=None):
        """Scores of the one-boundary candidates of a block that starts at global timepoint off.
        The loop, the boundopt filter and the -1 fill are those of the reference's _wdists."""
        n = len(deltas)
        wdists = -ones(n, float)
        if boundopt is None:
            boundopt = arange(1, n)
        base, rng = self._base(states, off)
        for i in boundopt:
            if deltas[i] == 0:
                a, b = self._find(rng, int(i))
                wdists[i] = (base - self._contrib(off + a, off + b)
                             + self._contrib(off + a, off + i)
                             + self._contrib(off + i, off + b)) / n
        return wdists

    def _wdists_state_impl(self, deltas, states, off, stateopt=None):
        """Scores of the two-boundary candidates (a new state inside one existing state).
        The loops, the stateopt filters, the -1 fill and the None result are those of the
        reference's _wdists_state."""
        n = len(deltas)
        wdists = -ones((n, n), float)
        base, rng = self._base(states, off)
        for i in arange(1, n):
            if deltas[i] == 0:
                if stateopt is not None and max(sum(equal(stateopt, array([[i, 0]])))) < 1:
                    continue
                state = nonzero(states[i] == states)[0]
                a, b = int(state[0]), int(state[-1]) + 1
                cab = self._contrib(off + a, off + b)
                cai = self._contrib(off + a, off + i)
                for j in state:
                    if j > i:
                        if stateopt is not None and max(sum(equal(stateopt, array([[i, j]])))) < 2:
                            continue
                        wdists[i, j] = (base - cab + cai
                                        + self._contrib(off + i, off + j)
                                        + self._contrib(off + j, off + b)) / n
        if len(unique(wdists)) == 1 and wdists[0, 0] == -1:
            wdists = None
        return wdists

    # the reference's _finetune calls self._wdists(...) with the full series
    def _wdists(self, deltas, states, x, z, boundopt=None):
        assert len(deltas) == self.x.shape[0], "_wdists called on a slice; use _wdists_impl"
        return self._wdists_impl(deltas, states, 0, boundopt)

    def _wdists_state(self, deltas, states, x, z, stateopt=None):
        assert len(deltas) == self.x.shape[0], "_wdists_state called on a slice"
        return self._wdists_state_impl(deltas, states, 0, stateopt)

    def _wdists_blocks(self, deltas, states, x, z, statewise, blocksize):
        """The reference's _wdists_blocks, with its two evaluator calls redirected."""
        if len(unique(states)) > 1:
            boundopt = zeros(max(states) + 1)
            stateopt = zeros((max(states) + 1, 2))

            prevstate = -1
            for s in unique(states):
                state = where((states > prevstate) & (states <= s))[0]
                numt = state.shape[0]
                if numt > blocksize or s == max(states):
                    off = int(state[0])
                    if statewise:
                        wdists_s = self._wdists_state_impl(deltas[state], states[state], off)
                        if wdists_s is None:
                            stateopt[s, :] = [0, 0]
                        else:
                            stateopt[s, :] = unravel_index(
                                wdists_s.argmax(), (wdists_s.shape[0], wdists_s.shape[0])) + state[0]

                    wd = self._wdists_impl(deltas[state], states[state], off)
                    boundopt[s] = wd.argmax() + state[0]
                    prevstate = s

            if statewise and not (stateopt is None):
                stateopt = stateopt[~all(stateopt == 0, axis=1)]
                stateopt = stateopt.astype(int)
            boundopt = boundopt[boundopt > 0]
            boundopt = boundopt.astype(int)
        else:
            boundopt = None
            stateopt = None

        wdists = self._wdists_impl(deltas, states, 0, boundopt)
        wdists_s = self._wdists_state_impl(deltas, states, 0, stateopt) if statewise else 0
        return wdists, wdists_s
