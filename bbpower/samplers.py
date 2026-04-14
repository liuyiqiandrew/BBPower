"""Sampler backends for BBCompSep.

Each function takes a ``Likelihood`` object (and configuration) and runs a
specific inference or evaluation strategy.  They are registered in
``SAMPLERS`` and dispatched by name from ``BBCompSep.run()``.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .likelihood import Likelihood


def run_emcee(likelihood: Likelihood, config: dict, output_dir: str) -> dict:
    """Run an MCMC using emcee.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration (must contain ``nwalkers``, ``n_iters``).
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Keys: ``chain``, ``names``, ``time``.
    """
    import emcee
    from multiprocessing import Pool

    fname_temp = os.path.join(output_dir, "emcee.npz.h5")
    backend = emcee.backends.HDFBackend(fname_temp)

    nwalkers = config["nwalkers"]
    n_iters = config["n_iters"]
    ndim = len(likelihood.params.p0)
    found_file = os.path.isfile(fname_temp)

    try:
        nchain = len(backend.get_chain())
    except AttributeError:
        found_file = False

    if not found_file:
        backend.reset(nwalkers, ndim)
        pos = [
            likelihood.params.p0 + 1.0e-3 * np.random.randn(ndim)
            for _ in range(nwalkers)
        ]
        nsteps_use = n_iters
    else:
        print("Restarting from previous run")
        pos = None
        nsteps_use = max(n_iters - nchain, 0)

    with Pool() as pool:
        start = time.time()
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, likelihood.lnprob, backend=backend
        )
        if nsteps_use > 0:
            sampler.run_mcmc(pos, nsteps_use, store=True, progress=False)
        elapsed = time.time() - start

    out_path = os.path.join(output_dir, "emcee.npz")
    np.savez(
        out_path,
        chain=sampler.chain,
        names=likelihood.params.p_free_names,
        time=elapsed,
    )
    print(f"Finished sampling {elapsed}")
    return {
        "chain": sampler.chain,
        "names": likelihood.params.p_free_names,
        "time": elapsed,
    }


def run_polychord(likelihood: Likelihood, config: dict, output_dir: str) -> Any:
    """Run nested sampling using PolyChord.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration (must contain ``nlive``, ``nrepeat``).
    output_dir : str
        Directory for output files.

    Returns
    -------
    object
        PolyChord output object.
    """
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    from pypolychord.priors import UniformPrior, GaussianPrior

    ndim = len(likelihood.params.p0)
    nder = 0

    def pc_likelihood(theta):
        return likelihood.lnlike(theta), [0]

    def pc_prior(hypercube):
        prior = []
        for h, pr in zip(hypercube, likelihood.params.p_free_priors):
            if pr[1] == "Gaussian":
                prior.append(GaussianPrior(float(pr[2][0]), float(pr[2][1]))(h))
            else:
                prior.append(UniformPrior(float(pr[2][0]), float(pr[2][2]))(h))
        return prior

    def dumper(live, dead, logweights, logZ, logZerr):
        print("Last dead point:", dead[-1])

    settings = PolyChordSettings(ndim, nder)
    settings.base_dir = os.path.join(output_dir, "polychord")
    settings.file_root = "pch"
    settings.nlive = config["nlive"]
    settings.num_repeats = config["nrepeat"]
    settings.do_clustering = False
    settings.boost_posterior = 10
    settings.nprior = 200
    settings.maximise = True
    settings.read_resume = False
    settings.feedback = 2

    output = pypolychord.run_polychord(
        pc_likelihood, ndim, nder, settings, pc_prior, dumper
    )
    print("Finished sampling")
    return output


def run_minimizer(likelihood: Likelihood, config: dict, output_dir: str) -> np.ndarray:
    """Find the maximum-likelihood point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    np.ndarray
        Best-fit parameter vector.
    """
    from scipy.optimize import minimize

    def chi2(par):
        return -2 * likelihood.lnprob(par)

    res = minimize(chi2, likelihood.params.p0, method="Powell")
    best_fit = res.x

    chi2_val = -2 * likelihood.lnprob(best_fit)
    out_path = os.path.join(output_dir, "chi2.npz")
    np.savez(
        out_path,
        params=best_fit,
        names=likelihood.params.p_free_names,
        chi2=chi2_val,
        ndof=len(likelihood.invcov),
    )

    print("Best fit:")
    for n, p in zip(likelihood.params.p_free_names, best_fit):
        print(f"{n} = {p:.3E}")
    print(f"Chi2: {chi2_val:.3E}")
    return best_fit


def run_fisher(
    likelihood: Likelihood, config: dict, output_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fisher matrix at the best-fit point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    tuple
        ``(best_fit_params, fisher_matrix)``.
    """
    import numdifftools as nd
    from scipy.optimize import minimize

    def chi2(par):
        return -2 * likelihood.lnprob(par)

    res = minimize(chi2, likelihood.params.p0, method="Powell")
    best_fit = res.x

    def lnprobd(p):
        val = likelihood.lnprob(p)
        if val == -np.inf:
            val = -1e100
        return val

    fisher = -nd.Hessian(lnprobd)(best_fit)
    cov = np.linalg.inv(fisher)

    for i, (n, p) in enumerate(zip(likelihood.params.p_free_names, best_fit)):
        print(f"{n} = {p:.3E} +- {np.sqrt(cov[i, i]):.3E}")

    out_path = os.path.join(output_dir, "fisher.npz")
    np.savez(
        out_path, params=best_fit, fisher=fisher, names=likelihood.params.p_free_names
    )
    return best_fit, fisher


def run_singlepoint(likelihood: Likelihood, config: dict, output_dir: str) -> float:
    """Evaluate the chi-squared at the fiducial point.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.

    Returns
    -------
    float
        Chi-squared value.
    """
    chi2 = -2 * likelihood.lnprob(likelihood.params.p0)
    out_path = os.path.join(output_dir, "single_point.npz")
    np.savez(
        out_path,
        chi2=chi2,
        ndof=len(likelihood.invcov),
        names=likelihood.params.p_free_names,
    )
    print("Chi2:", chi2, len(likelihood.invcov))
    return chi2


def run_timing(
    likelihood: Likelihood, config: dict, output_dir: str, n_eval: int = 300
) -> tuple[float, float]:
    """Benchmark likelihood evaluation speed.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.
    n_eval : int
        Number of evaluations to run.

    Returns
    -------
    tuple
        ``(total_time, time_per_eval)``.
    """
    start = time.time()
    for _ in range(n_eval):
        likelihood.lnprob(likelihood.params.p0)
    elapsed = time.time() - start

    out_path = os.path.join(output_dir, "timing.npz")
    np.savez(out_path, timing=elapsed / n_eval, names=likelihood.params.p_free_names)
    print("Total time:", elapsed)
    print("Time per eval:", elapsed / n_eval)
    return elapsed, elapsed / n_eval


def run_predicted_spectra(
    likelihood: Likelihood, compsep: Any, config: dict, output_dir: str
) -> None:
    """Evaluate model at the MAP and save predicted spectra.

    Parameters
    ----------
    likelihood : Likelihood
        Configured likelihood object.
    compsep : BBCompSep
        The pipeline stage (needed for model evaluation and SACC I/O).
    config : dict
        Stage configuration.
    output_dir : str
        Directory for output files.
    """
    import sacc

    at_min = config.get("predict_at_minimum", True)
    save_npz = not config.get("predict_to_sacc", False)

    if at_min:
        from scipy.optimize import minimize

        def chi2(par):
            return -2 * likelihood.lnprob(par)

        res = minimize(chi2, likelihood.params.p0, method="Powell")
        p = np.array(res.x)
    else:
        p = likelihood.params.p0

    pars = likelihood.params.build_params(p)
    print(pars)
    model_cls = compsep.model(pars)

    if config["bands"] == "all":
        tr_names = sorted(list(compsep.s.tracers.keys()))
    else:
        tr_names = config["bands"]

    if save_npz:
        np.savez(
            os.path.join(output_dir, "cells_model.npz"),
            tracers=tr_names,
            ls=compsep.ell_b,
            dls=model_cls,
        )
        print("Predicted spectra saved")
        return

    s = sacc.Sacc()
    for tn in tr_names:
        t = compsep.s.tracers[tn]
        s.add_tracer(
            "NuMap",
            tn,
            quantity="cmb_polarization",
            spin=2,
            nu=t.nu,
            bandpass=t.bandpass,
            ell=t.ell,
            beam=t.beam,
            nu_unit="GHz",
            map_unit="uK_CMB",
        )
    for b1, b2, p1, p2, m1, m2, ind in compsep._freq_pol_iterator():
        cl = model_cls[:, m1, m2]
        t1 = tr_names[b1]
        t2 = tr_names[b2]
        pol1 = compsep.pols[p1].lower()
        pol2 = compsep.pols[p2].lower()
        cltyp = f"cl_{pol1}{pol2}"
        win = sacc.BandpowerWindow(compsep.bpw_l, compsep.windows[ind].T)
        s.add_ell_cl(cltyp, t1, t2, compsep.ell_b, cl, window=win)
    s.add_covariance(compsep.bbcovar)
    s.save_fits(os.path.join(output_dir, "cells_model.fits"), overwrite=True)
    print("Predicted spectra saved")


SAMPLERS = {
    "emcee": run_emcee,
    "polychord": run_polychord,
    "maximum_likelihood": run_minimizer,
    "fisher": run_fisher,
    "single_point": run_singlepoint,
    "timing": run_timing,
}
