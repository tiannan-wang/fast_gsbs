"""fast_gsbs against statesegmentation 0.0.6 on small inputs.

Run from the root of the repository:   python -m unittest discover -s tests -t .
Needs the test requirements (requirements-test.txt).
"""
import contextlib
import inspect
import io
import os
import re
import subprocess
import sys
import unittest
import warnings

import numpy as np

from tests._common import (CASES, RefGSBS, bitwise_equal, fit_outputs, outputs_of, planted,   # noqa: I100
                           zero_mean_planted)
import fast_gsbs                                    # noqa: E402
from fast_gsbs import GSBS as FastGSBS              # noqa: E402


class TestInterface(unittest.TestCase):
    def test_default_class(self):
        from fast_gsbs import core
        self.assertIs(FastGSBS, core.GSBSFast)
        self.assertIs(fast_gsbs.GSBSFast, core.GSBSFast)

    def test_constructor_signature_matches_reference(self):
        self.assertEqual(inspect.signature(FastGSBS.__init__), inspect.signature(RefGSBS.__init__))

    def test_fit_signature_matches_reference(self):
        self.assertEqual(inspect.signature(FastGSBS.fit), inspect.signature(RefGSBS.fit))

    def test_public_names_match_reference(self):
        pub = lambda c: {n for n in dir(c) if not n.startswith("_")}
        self.assertEqual(pub(FastGSBS), pub(RefGSBS))

    def test_runtime_imports_exclude_reference_and_tqdm(self):
        code = ("import sys, fast_gsbs; "
                "print(int('statesegmentation' in sys.modules), int('tqdm' in sys.modules))")
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(r.stdout.split(), ["0", "0"])

    def test_version_matches_pyproject(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
            text = f.read()
        # read the [project] table without tomllib, which Python 3.9 and 3.10 do not have
        project = text.split("[project]", 1)[1].split("\n[", 1)[0]
        m = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project, re.M)
        self.assertIsNotNone(m, "no version line in the [project] table of pyproject.toml")
        self.assertEqual(fast_gsbs.__version__, m.group(1))


class TestSmallCasesBitwise(unittest.TestCase):
    def test_option_cases_bitwise_equal_to_reference(self):
        for name, gen, kmax, opts in CASES:
            with self.subTest(name):
                self.assertTrue(bitwise_equal(fit_outputs(FastGSBS, planted(*gen), kmax, **opts),
                                              fit_outputs(RefGSBS, planted(*gen), kmax, **opts)))

    def test_y_given_bitwise_equal_to_reference(self):
        x = planted(40, 20, 5, 0.8, 18)
        y = x + 0.8 * np.random.default_rng(19).standard_normal(x.shape)
        self.assertTrue(bitwise_equal(fit_outputs(FastGSBS, x, 10, y=y), fit_outputs(RefGSBS, x, 10, y=y)))


class TestInputChecks(unittest.TestCase):
    """NaN/Inf values and constant timepoints are rejected, at construction and again in fit()."""

    def _x(self):
        return planted(40, 20, 5, 0.8, 30)

    def test_constant_row_in_x_rejected(self):
        for value in (0.0, 7.5):
            x = self._x()
            x[12] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "^x has .*constant timepoint"):
                    FastGSBS(kmax=10, x=x)

    def test_constant_row_in_y_rejected(self):
        x = self._x()
        y = x.copy()
        y[0] = 1.0
        with self.assertRaisesRegex(ValueError, "^y has .*constant timepoint"):
            FastGSBS(kmax=10, x=x, y=y)

    def test_reference_accepts_constant_row(self):
        """The reference fits such input without an error; its t-distances are NaN."""
        x = self._x()
        x[12] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = RefGSBS(kmax=10, x=x)
            g.fit(showProgressBar=False)
        self.assertTrue(np.isnan(g.tdists).any())

    def test_nonfinite_in_x_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            x = self._x()
            x[5, 3] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "^x has .*non-finite"):
                    FastGSBS(kmax=10, x=x)

    def test_nonfinite_in_y_rejected(self):
        for value in (np.nan, np.inf):
            x = self._x()
            y = x.copy()
            y[7, 0] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "^y has .*non-finite"):
                    FastGSBS(kmax=10, x=x, y=y)

    def test_inputs_modified_after_construction_rejected_at_fit(self):
        for what, value, pattern in (("x", 0.0, "^x has .*constant timepoint"), ("x", np.nan, "^x has .*non-finite"),
                                     ("y", 1.0, "^y has .*constant timepoint"), ("y", np.inf, "^y has .*non-finite")):
            with self.subTest(what=what, value=value):
                x = self._x()
                y = x + 0.5
                g = FastGSBS(kmax=10, x=x, y=y)       # valid at construction
                if what == "x":
                    x[12] = value                     # same array object, modified in place
                else:
                    y[3] = value
                with self.assertRaisesRegex(ValueError, pattern):
                    g.fit(showProgressBar=False)


class TestDegenerateStateWarning(unittest.TestCase):
    def _fit(self, cls, x, mode):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter(mode)
            out = fit_outputs(cls, x, 6)
        return out, [w for w in wl if issubclass(w.category, fast_gsbs.DegenerateStateWarning)]

    def test_warned_once(self):
        _, w = self._fit(FastGSBS, zero_mean_planted(), "always")
        self.assertEqual(len(w), 1)
        self.assertIn("may differ from statesegmentation 0.0.6", str(w[0].message))

    def test_warning_changes_nothing(self):
        """Same outputs with the warning code switched off, not only with the warning hidden."""
        class WithoutWarning(FastGSBS):
            _WARN_DEGENERATE = False

        a, wa = self._fit(FastGSBS, zero_mean_planted(), "always")
        b, wb = self._fit(WithoutWarning, zero_mean_planted(), "always")
        self.assertEqual((len(wa), len(wb)), (1, 0))
        self.assertTrue(bitwise_equal(a, b))

    def test_no_warning_on_ordinary_input(self):
        _, w = self._fit(FastGSBS, planted(40, 20, 5, 0.8, 11), "always")
        self.assertEqual(len(w), 0)


class TestProgressDisplay(unittest.TestCase):
    def test_default_fit_prints_a_counter_and_changes_no_output(self):
        x = planted(40, 20, 5, 0.8, 11)                 # the "defaults" case, kmax 10
        quiet = fit_outputs(FastGSBS, x, 10)
        g = FastGSBS(kmax=10, x=x.copy())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            g.fit()                                     # showProgressBar=True is the default
        self.assertTrue(bitwise_equal(outputs_of(g, 10), quiet))
        self.assertTrue(err.getvalue().endswith("GSBS: 9/9\n"), repr(err.getvalue()[-40:]))


if __name__ == "__main__":
    unittest.main()
