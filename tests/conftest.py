"""Shared fixtures for the BBPower test suite.

Provides mock ``bbpipe`` and ``fgbuster`` modules so that all ``bbpower``
submodules can be imported even when those packages are not installed.
Also provides reusable configuration dictionaries and synthetic data factories.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock bbpipe.PipelineStage
# ---------------------------------------------------------------------------
class _MockPipelineStage:
    """Minimal stand-in for ``bbpipe.PipelineStage``.

    Provides enough of the interface so that ``BBCompSep`` (and the other
    stage classes) can be *defined* (class body + ``__init_subclass__``)
    and instantiated without the real bbpipe.
    """

    pipeline_stages: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls.pipeline_stages[cls.name] = (cls, None)

    def __init__(self, args=None):
        self._configs = {}
        self._inputs = {}
        self._outputs = {}

    @property
    def config(self):
        return self._configs

    def get_input(self, tag):
        return self._inputs.get(tag)

    def get_output(self, tag):
        return self._outputs.get(tag)


# ---------------------------------------------------------------------------
# Mock fgbuster.component_model
# ---------------------------------------------------------------------------
class _MockSEDBase:
    """Trivial SED that returns nu**power."""

    _power = 0.0

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def eval(self, nu, *args):
        return np.ones_like(np.asarray(nu, dtype=float))


class _MockCMB(_MockSEDBase):
    def __init__(self, units="K_RJ"):
        self._units = units

    def eval(self, nu, *args):
        return np.ones_like(np.asarray(nu, dtype=float))


class _MockDust(_MockSEDBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, nu, *args):
        return (np.asarray(nu, dtype=float) / 353.0) ** 1.5


class _MockSynchrotron(_MockSEDBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def eval(self, nu, *args):
        return (np.asarray(nu, dtype=float) / 23.0) ** (-3.0)


# ---------------------------------------------------------------------------
# Inject mocks into sys.modules (before any bbpower import)
# ---------------------------------------------------------------------------
def _inject_mock_bbpipe():
    if "bbpipe" not in sys.modules:
        mod = types.ModuleType("bbpipe")
        mod.PipelineStage = _MockPipelineStage
        sys.modules["bbpipe"] = mod


def _inject_mock_fgbuster():
    if "fgbuster" not in sys.modules:
        fgb = types.ModuleType("fgbuster")
        fgc = types.ModuleType("fgbuster.component_model")
        fgc.CMB = _MockCMB
        fgc.Dust = _MockDust
        fgc.Synchrotron = _MockSynchrotron
        fgb.component_model = fgc
        sys.modules["fgbuster"] = fgb
        sys.modules["fgbuster.component_model"] = fgc


# Run injections at import time so they are available before collection
_inject_mock_bbpipe()
_inject_mock_fgbuster()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bb_only_config():
    """Minimal B-only config with dust + synchrotron components."""
    return {
        "pol_channels": ["B"],
        "l_min": 30,
        "l_max": 120,
        "bands": "all",
        "likelihood_type": "h&l",
        "sampler": "maximum_likelihood",
        "nwalkers": 8,
        "n_iters": 10,
        "cmb_model": {
            "params": {
                "r_tensor": ["r_tensor", "tophat", [-0.1, 0.0, 0.1]],
                "A_lens": ["A_lens", "tophat", [0.0, 1.0, 2.0]],
            }
        },
        "fg_model": {
            "component_1": {
                "name": "Dust",
                "sed": "Dust",
                "cl": {("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_d": ["beta_d", "Gaussian", [1.59, 0.11]],
                    "temp_d": ["temp", "fixed", [19.6]],
                    "nu0_d": ["nu0", "fixed", [353.0]],
                },
                "cl_parameters": {
                    ("B", "B"): {
                        "amp_d_bb": ["amp", "tophat", [0.0, 5.0, 100.0]],
                        "alpha_d_bb": ["alpha", "tophat", [-1.0, -0.2, 0.0]],
                        "l0_d_bb": ["ell0", "fixed", [80.0]],
                    }
                },
                "cross": {"epsilon_ds": ["component_2", "tophat", [-1.0, 0.0, 1.0]]},
            },
            "component_2": {
                "name": "Synchrotron",
                "sed": "Synchrotron",
                "cl": {("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_s": ["beta_pl", "Gaussian", [-3.0, 0.3]],
                    "nu0_s": ["nu0", "fixed", [23.0]],
                },
                "cl_parameters": {
                    ("B", "B"): {
                        "amp_s_bb": ["amp", "tophat", [0.0, 2.0, 10.0]],
                        "alpha_s_bb": ["alpha", "tophat", [-1.0, -0.4, 0.0]],
                        "l0_s_bb": ["ell0", "fixed", [80.0]],
                    }
                },
            },
        },
    }


@pytest.fixture(scope="session")
def eb_config():
    """Minimal E+B config with dust + synchrotron components."""
    return {
        "pol_channels": ["E", "B"],
        "l_min": 30,
        "l_max": 120,
        "bands": "all",
        "likelihood_type": "chi2",
        "sampler": "maximum_likelihood",
        "nwalkers": 8,
        "n_iters": 10,
        "cmb_model": {
            "params": {
                "r_tensor": ["r_tensor", "tophat", [-0.1, 0.0, 0.1]],
                "A_lens": ["A_lens", "tophat", [0.0, 1.0, 2.0]],
            }
        },
        "fg_model": {
            "component_1": {
                "name": "Dust",
                "sed": "Dust",
                "cl": {("E", "E"): "ClPowerLaw", ("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_d": ["beta_d", "Gaussian", [1.59, 0.11]],
                    "temp_d": ["temp", "fixed", [19.6]],
                    "nu0_d": ["nu0", "fixed", [353.0]],
                },
                "cl_parameters": {
                    ("E", "E"): {
                        "amp_d_ee": ["amp", "tophat", [0.0, 10.0, 100.0]],
                        "alpha_d_ee": ["alpha", "tophat", [-1.0, -0.42, 0.0]],
                        "l0_d_ee": ["ell0", "fixed", [80.0]],
                    },
                    ("B", "B"): {
                        "amp_d_bb": ["amp", "tophat", [0.0, 5.0, 100.0]],
                        "alpha_d_bb": ["alpha", "tophat", [-1.0, -0.2, 0.0]],
                        "l0_d_bb": ["ell0", "fixed", [80.0]],
                    },
                },
                "cross": {"epsilon_ds": ["component_2", "tophat", [-1.0, 0.0, 1.0]]},
            },
            "component_2": {
                "name": "Synchrotron",
                "sed": "Synchrotron",
                "cl": {("E", "E"): "ClPowerLaw", ("B", "B"): "ClPowerLaw"},
                "sed_parameters": {
                    "beta_s": ["beta_pl", "Gaussian", [-3.0, 0.3]],
                    "nu0_s": ["nu0", "fixed", [23.0]],
                },
                "cl_parameters": {
                    ("E", "E"): {
                        "amp_s_ee": ["amp", "tophat", [0.0, 4.0, 20.0]],
                        "alpha_s_ee": ["alpha", "tophat", [-1.0, -0.6, 0.0]],
                        "l0_s_ee": ["ell0", "fixed", [80.0]],
                    },
                    ("B", "B"): {
                        "amp_s_bb": ["amp", "tophat", [0.0, 2.0, 10.0]],
                        "alpha_s_bb": ["alpha", "tophat", [-1.0, -0.4, 0.0]],
                        "l0_s_bb": ["ell0", "fixed", [80.0]],
                    },
                },
            },
        },
    }


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def make_bandpass():
    """Factory fixture for creating ``Bandpass`` objects with synthetic data."""
    from bbpower.bandpasses import Bandpass

    def _factory(
        nu_center: float = 150.0,
        bandwidth: float = 30.0,
        n_points: int = 11,
        bp_number: int = 1,
        config: dict | None = None,
    ) -> Bandpass:
        if config is None:
            config = {}
        nu = np.linspace(nu_center - bandwidth / 2, nu_center + bandwidth / 2, n_points)
        bnu = np.ones_like(nu)
        dnu = np.ones_like(nu) * (nu[1] - nu[0])
        return Bandpass(nu, dnu, bnu, bp_number, config)

    return _factory
