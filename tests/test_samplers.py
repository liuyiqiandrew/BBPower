"""Tests for bbpower.samplers — sampler dispatch and registry."""

from __future__ import annotations

import fcntl
import sys
import types

import numpy as np
import pytest

import bbpower.samplers as samplers
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


class TestRunEmcee:
    """Test the emcee backend wiring."""

    def test_backend_lock_raises_for_concurrent_writer(self, monkeypatch, tmp_path):
        """A second writer gets a clear error before touching the HDF backend."""

        def fail_lock(fd, flags):
            if flags & fcntl.LOCK_UN:
                return None
            raise BlockingIOError

        monkeypatch.setattr(fcntl, "flock", fail_lock)

        with pytest.raises(RuntimeError, match="already using"):
            with samplers._emcee_backend_lock(str(tmp_path / "emcee.npz.h5")):
                pass

    def test_worker_count_uses_env_and_caps_to_walkers(self, monkeypatch):
        """Worker count respects the useful parallel limit."""
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "32")
        assert samplers._get_emcee_nworkers(40) == 20
        assert samplers._get_emcee_nworkers(8) == 4

        monkeypatch.delenv("BBPOWER_EMCEE_WORKERS")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
        assert samplers._get_emcee_nworkers(40) == 6

    def test_default_pool_mode_is_thread(self, monkeypatch):
        """Thread pools are the safe default for BBCompSep likelihoods."""
        monkeypatch.delenv("BBPOWER_EMCEE_POOL", raising=False)
        assert samplers._get_emcee_pool_mode() == "thread"

        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "process")
        assert samplers._get_emcee_pool_mode() == "process"

    def test_passes_thread_pool_to_ensemble_sampler(self, monkeypatch, tmp_path):
        """run_emcee wires the thread pool into emcee."""

        class MockParams:
            p0 = np.array([0.0, 1.0])
            p_free_names = ["r", "A_lens"]

        class MockLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -1.0

        calls: dict[str, object] = {}

        class FakeBackend:
            def __init__(self, path):
                calls["backend_path"] = path

            def get_chain(self):
                raise AttributeError

            def reset(self, nwalkers, ndim):
                calls["reset"] = (nwalkers, ndim)

        class FakeSampler:
            def __init__(
                self,
                nwalkers,
                ndim,
                log_prob_fn,
                pool=None,
                backend=None,
                **kwargs,
            ):
                calls["pool"] = pool
                calls["backend"] = backend
                calls["kwargs"] = kwargs
                self.chain = np.zeros((nwalkers, 1, ndim))

            def run_mcmc(self, pos, nsteps, store=True, progress=False):
                calls["run_mcmc"] = (len(pos), nsteps, store, progress)

        class FakePool:
            def __init__(self, processes=None):
                calls["processes"] = processes

            def __enter__(self):
                calls["pool_obj"] = self
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_emcee = types.SimpleNamespace(
            backends=types.SimpleNamespace(HDFBackend=FakeBackend),
            EnsembleSampler=FakeSampler,
        )

        monkeypatch.setitem(sys.modules, "emcee", fake_emcee)
        monkeypatch.setattr("multiprocessing.pool.ThreadPool", FakePool)
        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "3")
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "thread")

        out = samplers.run_emcee(
            MockLikelihood(),
            {"nwalkers": 4, "n_iters": 2},
            str(tmp_path),
        )

        assert calls["processes"] == 2
        assert calls["pool"] is calls["pool_obj"]
        assert calls["reset"] == (4, 2)
        assert calls["run_mcmc"] == (4, 2, True, False)
        assert out["chain"].shape == (4, 1, 2)

        saved = np.load(tmp_path / "emcee.npz")
        assert saved["chain"].shape == (4, 1, 2)
        assert saved["names"].tolist() == ["r", "A_lens"]

    def test_thread_pool_handles_local_likelihood(self, monkeypatch, tmp_path):
        """Thread parallelism works with local likelihood objects."""

        class MockParams:
            p0 = np.array([0.1, -0.2])
            p_free_names = ["r", "A_lens"]

        class LocalLikelihood:
            params = MockParams()

            def lnprob(self, par):
                return -0.5 * np.dot(par, par)

        monkeypatch.setenv("BBPOWER_EMCEE_WORKERS", "2")
        monkeypatch.setenv("BBPOWER_EMCEE_POOL", "thread")

        out = samplers.run_emcee(
            LocalLikelihood(),
            {"nwalkers": 6, "n_iters": 3},
            str(tmp_path),
        )

        assert out["chain"].shape == (6, 3, 2)
