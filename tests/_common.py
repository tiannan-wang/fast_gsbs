"""Shared test helpers.

Importing this module checks that statesegmentation 0.0.6 is installed. If it is not, the run stops
with exit status 2 and a message that says how to install it.
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

REF_SHA = "ac8d4dae3a2d72791f33699871f05f077f119ddd640a3904f09ed1fccb6e90de"    # gsbs.py of 0.0.6
INSTALL_HINT = ("Install the test requirements from the root of the repository:\n"
                "    pip install -r requirements-test.txt")


def _abort(msg):
    # unittest turns a SystemExit raised during import into a test error, so exit directly
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    os._exit(2)


def _check_reference():
    try:
        spec = importlib.util.find_spec("statesegmentation.gsbs")
        problem = "it is not installed"
    except ImportError as e:        # find_spec imports the parent package first
        spec, problem = None, "it cannot be imported (%s)" % e
    if spec is None:
        _abort("ABORT: the tests compare fast_gsbs with the reference implementation "
               "statesegmentation==0.0.6, and %s.\n%s" % (problem, INSTALL_HINT))
    with open(spec.origin, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    if h != REF_SHA:
        _abort("ABORT: installed statesegmentation/gsbs.py has SHA256 %s, expected %s (the file of "
               "statesegmentation 0.0.6).\n%s" % (h, REF_SHA, INSTALL_HINT))


_check_reference()

from statesegmentation.gsbs import GSBS as RefGSBS   # noqa: E402

PUBLIC_GETTERS = ("bounds", "deltas", "tdists", "nstates", "states", "strengths", "state_patterns")


def planted(T, V, nb, snr, seed):
    """T x V input with nb planted states: state patterns scaled by snr, plus unit noise."""
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(3, T - 3), nb - 1, replace=False))
    lab = np.searchsorted(cuts, np.arange(T), side="right")
    return rng.standard_normal((nb, V))[lab] * snr + rng.standard_normal((T, V))


def outputs_of(g, kmax):
    """Every public output of a fitted object, as arrays."""
    out = {name: np.array(getattr(g, name)) for name in PUBLIC_GETTERS}
    out["all_bounds"] = np.array(g.all_bounds)
    out["get_bounds_k"] = np.stack([g.get_bounds(k) for k in range(1, kmax + 1)])
    out["get_deltas_k"] = np.stack([g.get_deltas(k) for k in range(1, kmax + 1)])
    return out


def fit_outputs(cls, x, kmax, **opts):
    g = cls(kmax=kmax, x=x.copy(), **opts)
    g.fit(showProgressBar=False)
    return outputs_of(g, kmax)


def bitwise_equal(a, b):
    """True if every output has the same dtype, shape and bytes."""
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                    and a[k].tobytes() == b[k].tobytes() for k in a)


# name, arguments of planted(), kmax, options
CASES = [
    ("defaults", (40, 20, 5, 0.8, 11), 10, {}),
    ("statewise_false", (40, 20, 5, 0.8, 12), 10, {"statewise_detection": False}),
    ("finetune_0", (40, 20, 5, 0.8, 13), 10, {"finetune": 0}),
    ("finetune_-1", (40, 20, 5, 0.8, 14), 10, {"finetune": -1}),
    ("finetune_order_false", (40, 20, 5, 0.8, 15), 10, {"finetune_order": False}),
    ("blocksize_10", (60, 20, 6, 0.8, 16), 12, {"blocksize": 10}),
    ("dmin_3", (40, 20, 5, 0.8, 17), 10, {"dmin": 3}),
]


# ---- inputs for the two limitations in the README ------------------------------------------------

def zero_mean_planted():
    """30 x 20 input with x[1] = -x[0]: the state made of timepoints 0 and 1 has a mean pattern that
    is constant across features (a zero-mean state)."""
    x = planted(30, 20, 4, 1.0, 7)
    x[1] = -x[0]
    return x


def zero_mean_inputs():
    """(name, x, kmax, opts) for four inputs with a zero-mean state. fast_gsbs differs from the
    reference on each; ZERO_MEAN_EXPECTED holds the difference."""
    # the last three timepoints sum to a pattern that is constant across the two features
    out = [("two_features", np.array([[0.1, -0.1], [0.6, 0.1], [-0.5, 0.4], [1.3, 0.9]]), 3,
            dict(statewise_detection=False, finetune=0))]
    x = np.random.default_rng(0).standard_normal((30, 10)); x[1] = -x[0]
    out.append(("zero_mean_start", x, 4, {"statewise_detection": False}))
    x = np.random.default_rng(0).standard_normal((30, 10)); x[15] = -x[14]
    out.append(("zero_mean_mid", x, 8, {}))
    out.append(("zero_mean_planted", zero_mean_planted(), 6, {}))
    return out


# name: (nstates of the reference, nstates of fast_gsbs)
ZERO_MEAN_EXPECTED = {
    "two_features": (0, 0),
    "zero_mean_start": (2, 4),
    "zero_mean_mid": (5, 6),
    "zero_mean_planted": (2, 5),
}
# two_features only: timepoint of the first boundary, reference and fast_gsbs
TWO_FEATURES_FIRST_BOUNDARY = (1, 2)


def tie_inputs():
    """(name, x, kmax, opts) for four inputs with tied candidates. Whether fast_gsbs picks the
    reference's boundary depends on rounding, and so on the NumPy build."""
    kw = dict(statewise_detection=False, finetune=0)
    p = np.array([[-0.7, -0.2, 1.7], [0.7, -1.6, 0.0]])
    out = [("piecewise_constant", p[[0, 0, 1, 1]], 4, kw)]
    h = np.random.default_rng(0).standard_normal((15, 10))
    out.append(("mirror_symmetric", np.vstack([h, h[::-1]]), 3, kw))
    rng = np.random.default_rng(200)
    lab = np.minimum(np.arange(40) * 5 // 40, 4)
    x = rng.standard_normal((5, 20))[lab] * 0.6 + rng.standard_normal((40, 20)) + 1e6 * rng.standard_normal(20)
    out.append(("large_feature_offset", x, 3, dict(statewise_detection=False, finetune=0, blocksize=10_000)))
    H = planted(20, 25, 3, 0.8, 200)
    out.append(("planted_mirror_symmetric", np.vstack([H, H[::-1]]), 6, {}))
    return out


def result_problems(out, shape, kmax, statewise=True):
    """What makes a fitted result invalid, as a list of strings (empty for a valid result).

    Checks the shapes, that no boundary sits at timepoint 0, the number of boundaries in each row of
    all_bounds, that nstates is consistent with bounds, deltas and states, and that no t-distance is
    NaN. An infinite t-distance is accepted: the reference returns +inf on noise-free input.
    """
    T, V = shape
    problems = []
    want = {"all_bounds": (kmax + 2, T), "tdists": (kmax + 2,), "bounds": (T,), "deltas": (T,), "states": (T,),
            "strengths": (T,), "get_bounds_k": (kmax, T), "get_deltas_k": (kmax, T), "nstates": ()}
    for name, shp in want.items():
        if out[name].shape != shp:
            problems.append("%s has shape %s, expected %s" % (name, out[name].shape, shp))
    if problems:
        return problems
    ab, ns = out["all_bounds"], int(out["nstates"])
    if (ab < 0).any() or ab[:, 0].any():
        problems.append("all_bounds has a negative entry or a boundary at timepoint 0")
    # rows 0 and 1 hold no boundary; row k holds k - 1, or k when the statewise step added two at once
    counts = (ab > 0).sum(axis=1)
    for k in range(kmax + 2):
        if k < 2:
            allowed = {0}
        elif k == kmax + 1:
            allowed = {0, kmax} if statewise else {0}
        else:
            allowed = {k - 1, k} if statewise else {k - 1}
        if int(counts[k]) not in allowed:
            problems.append("row %d of all_bounds has %d boundaries, expected %s" % (k, counts[k], sorted(allowed)))
    if not 0 <= ns <= kmax + 1:
        problems.append("nstates %d is outside 0..%d" % (ns, kmax + 1))
    else:
        if not np.array_equal(out["bounds"], ab[ns]):
            problems.append("bounds is not row nstates of all_bounds")
        if not np.array_equal(out["deltas"], (ab[ns] > 0) * 1):
            problems.append("deltas does not mark the boundaries of row nstates")
        if not np.array_equal(out["states"], np.cumsum(ab[ns] > 0) + 1):
            problems.append("states does not follow from the boundaries of row nstates")
        elif out["state_patterns"].shape != (int(out["states"].max()), V):
            problems.append("state_patterns has shape %s, expected %s"
                            % (out["state_patterns"].shape, (int(out["states"].max()), V)))
    if np.isnan(out["tdists"]).any():
        problems.append("tdists contains NaN")
    return problems
