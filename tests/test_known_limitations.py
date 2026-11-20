"""The two limitations in the README, on eight small inputs.

Zero-mean states (four inputs): fast_gsbs differs from statesegmentation 0.0.6 on them. Each test
checks the difference itself: the number of states each implementation returns, and that fast_gsbs
warns exactly once.

Ties (four inputs): which boundary is chosen depends on rounding in the NumPy build. These tests
only check that fast_gsbs returns a valid result. At the end they print, for each input, whether
fast_gsbs matched the reference on this machine.
"""
import sys
import unittest
import warnings

import numpy as np

from tests._common import (TWO_FEATURES_FIRST_BOUNDARY, ZERO_MEAN_EXPECTED, RefGSBS, bitwise_equal, fit_outputs,
                           result_problems, tie_inputs, zero_mean_inputs)
import fast_gsbs
from fast_gsbs import GSBS as FastGSBS

_ZERO_MEAN = {name: (x, kmax, opts) for name, x, kmax, opts in zero_mean_inputs()}
_TIES = {name: (x, kmax, opts) for name, x, kmax, opts in tie_inputs()}


def _fit_both(x, kmax, opts):
    """Outputs of fast_gsbs and of the reference, and the DegenerateStateWarnings of fast_gsbs."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        fast = fit_outputs(FastGSBS, x, kmax, **opts)
    warned = [w for w in wl if issubclass(w.category, fast_gsbs.DegenerateStateWarning)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = fit_outputs(RefGSBS, x, kmax, **opts)
    return fast, ref, warned


def _mismatch(case, what, expected, found):
    return ("%s: expected %s = %s, found %s. This input has a zero-mean state (README, Limitations). "
            "A mismatch on another machine or NumPy build is a difference in that edge case, not a "
            "broken installation." % (case, what, expected, found))


class TestZeroMeanStates(unittest.TestCase):
    def _check(self, name):
        x, kmax, opts = _ZERO_MEAN[name]
        fast, ref, warned = _fit_both(x, kmax, opts)
        found = (int(ref["nstates"]), int(fast["nstates"]))
        if found != ZERO_MEAN_EXPECTED[name]:
            self.fail(_mismatch(name, "nstates (reference, fast_gsbs)", ZERO_MEAN_EXPECTED[name], found))
        if bitwise_equal(fast, ref):
            self.fail(_mismatch(name, "outputs that differ from the reference", "a difference",
                                "bitwise identical outputs"))
        if len(warned) != 1:
            self.fail(_mismatch(name, "number of DegenerateStateWarning", 1, len(warned)))
        return fast, ref

    def test_two_features(self):
        fast, ref = self._check("two_features")
        found = (np.flatnonzero(ref["get_bounds_k"][1]).tolist(), np.flatnonzero(fast["get_bounds_k"][1]).tolist())
        expected = ([TWO_FEATURES_FIRST_BOUNDARY[0]], [TWO_FEATURES_FIRST_BOUNDARY[1]])
        if found != expected:
            self.fail(_mismatch("two_features", "timepoint of the first boundary (reference, fast_gsbs)",
                                expected, found))

    def test_zero_mean_start(self):
        self._check("zero_mean_start")

    def test_zero_mean_mid(self):
        self._check("zero_mean_mid")

    def test_zero_mean_planted(self):
        self._check("zero_mean_planted")


class TestTies(unittest.TestCase):
    _report = []

    def _check(self, name):
        x, kmax, opts = _TIES[name]
        fast, ref, _ = _fit_both(x, kmax, opts)
        problems = result_problems(fast, x.shape, kmax, opts.get("statewise_detection", True))
        self.assertEqual(problems, [], "%s: fast_gsbs did not return a valid result" % name)
        if bitwise_equal(fast, ref):
            outcome = "yes"
        else:
            rows = np.flatnonzero((fast["all_bounds"] != ref["all_bounds"]).any(axis=1))
            where = "boundaries differ from k = %d" % rows[0] if rows.size else "same boundaries, other outputs differ"
            outcome = "no (%s; nstates %d, reference %d)" % (where, int(fast["nstates"]), int(ref["nstates"]))
        self._report.append("tie case %s: valid result; equal to the reference on this machine: %s" % (name, outcome))

    @classmethod
    def tearDownClass(cls):
        sys.stderr.write("\n" + "".join(line + "\n" for line in cls._report))
        sys.stderr.flush()
        del cls._report[:]

    def test_piecewise_constant(self):
        self._check("piecewise_constant")

    def test_mirror_symmetric(self):
        self._check("mirror_symmetric")

    def test_large_feature_offset(self):
        self._check("large_feature_offset")

    def test_planted_mirror_symmetric(self):
        self._check("planted_mirror_symmetric")


if __name__ == "__main__":
    unittest.main()
