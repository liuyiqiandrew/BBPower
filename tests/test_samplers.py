"""Tests for bbpower.samplers — sampler dispatch and registry."""

from __future__ import annotations

import numpy as np
import pytest

from bbpower.samplers import SAMPLERS, run_singlepoint, run_timing


class TestSamplersDict:
    """Tests for the SAMPLERS registry."""

    def test_keys(self):
        """All expected sampler backends are registered."""
        expected = {
            "emcee",
            "polychord",
            "maximum_likelihood",
            "fisher",
            "single_point",
            "timing",
        }
        assert expected == set(SAMPLERS.keys())

    def test_callable(self):
        """All registered values are callable."""
        for name, func in SAMPLERS.items():
            assert callable(func), f"{name} is not callable"

    def test_unknown_not_in_dict(self):
        """Nonexistent sampler is not registered."""
        assert "nonexistent" not in SAMPLERS


class TestRunSinglepoint:
    """Test the single-point chi-squared evaluation."""

    def test_writes_output(self, tmp_path):
        """run_singlepoint writes an npz file with chi2 and ndof."""

        class MockParams:
            p0 = np.array([0.0])
            p_free_names = ["r"]

            def lnprior(self, par):
                return 0.0

            def build_params(self, par):
                return {"r": par[0]}

        class MockLikelihood:
            params = MockParams()
            invcov = np.eye(3)

            def lnprob(self, par):
                return -5.0

        lik = MockLikelihood()
        chi2 = run_singlepoint(lik, {}, str(tmp_path))
        assert chi2 == pytest.approx(10.0)  # -2 * (-5.0)
        out = np.load(tmp_path / "single_point.npz")
        assert "chi2" in out
        assert "ndof" in out


class TestRunTiming:
    """Test the timing benchmark."""

    def test_returns_positive_time(self, tmp_path):
        """run_timing returns positive elapsed times."""

        class MockParams:
            p0 = np.array([0.0])
            p_free_names = ["r"]

            def lnprior(self, par):
                return 0.0

            def build_params(self, par):
                return {"r": par[0]}

        class MockLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -1.0

        lik = MockLikelihood()
        total, per_eval = run_timing(lik, {}, str(tmp_path), n_eval=5)
        assert total > 0
        assert per_eval > 0
        out = np.load(tmp_path / "timing.npz")
        assert "timing" in out
