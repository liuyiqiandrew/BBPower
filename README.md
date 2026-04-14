# BBPower

Power-spectrum-based component separation pipeline for constraining primordial B-modes from multi-frequency CMB polarization data.

BBPower performs a maps-to-parameters analysis: it computes cross-frequency bandpower spectra from HEALPix maps, coadds splits, estimates covariances from simulations, fits a parametric foreground + CMB model, and produces diagnostic plots. The pipeline is built on the [BBPipe](https://github.com/simonsobs/BBPipe) framework.

## Installation

```bash
# Core (component separation and plotting only)
pip install -e .

# Full (includes map-level stages: healpy, pymaster, fgbuster, getdist)
pip install -e ".[all]"
```

Requires Python >= 3.10. See [pyproject.toml](pyproject.toml) for the full dependency list.

## Quick Start

The fastest way to run BBPower is on pre-computed power spectra (skipping the map-level stages):

```bash
# 1. Generate synthetic Simons Observatory bandpowers
mkdir -p output
python examples/generate_SO_spectra.py output

# 2. Run component separation (maximum-likelihood fit)
python -m bbpower BBCompSep \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_fiducial=output/cls_fid.fits \
  --cells_coadded_cov=output/cls_coadd.fits \
  --output_dir=output \
  --config_copy=output/config_copy.yml \
  --config=test/test_config_sampling.yml

# 3. Generate diagnostic plots
python -m bbpower BBPlotter \
  --cells_coadded_total=output/cls_coadd.fits \
  --cells_coadded=output/cls_coadd.fits \
  --cells_noise=output/cls_noise.fits \
  --cells_null=output/cls_coadd.fits \
  --cells_fiducial=output/cls_fid.fits \
  --param_chains=output/chi2.npz \
  --plots=output/plots.dir \
  --plots_page=output/plots_page.html \
  --config=test/test_config_sampling.yml
```

## Pipeline Stages

BBPower has four stages that run in sequence. Each reads typed inputs and produces typed outputs in [SACC](https://github.com/LSSTDESC/sacc) format.

| Stage | Module | Purpose |
|---|---|---|
| **BBPowerSpecter** | `power_specter.py` | Compute cross-frequency bandpower spectra from HEALPix Q/U maps using NaMaster |
| **BBPowerSummarizer** | `power_summarizer.py` | Coadd splits, compute noise/null spectra, estimate covariances from simulations |
| **BBCompSep** | `compsep.py` | Fit a parametric CMB + foreground model to the bandpowers |
| **BBPlotter** | `plotter.py` | Generate diagnostic plots and an HTML summary page |

You can run the full pipeline (maps to parameters) or enter at any stage with pre-computed inputs. See [docs/architecture.md](docs/architecture.md) for the data flow and module interactions.

## Configuration

BBPower uses two YAML files:

1. **Pipeline file** (e.g., `test/test_sampling.yml`) -- declares stages, input file paths, and output directories for BBPipe orchestration.
2. **Stage config file** (e.g., `test/test_config_sampling.yml`) -- defines the physical model (CMB templates, foreground components, priors) and sampler settings.

The stage config has a `global` section (shared by all stages) and per-stage sections:

```yaml
global:
  nside: 64
  compute_dell: true

BBCompSep:
  sampler: 'maximum_likelihood'     # emcee | polychord | fisher | single_point | timing
  likelihood_type: 'h&l'            # chi2 | h&l
  pol_channels: ['E', 'B']
  l_min: 30
  l_max: 300

  cmb_model:
    cmb_templates:
      - "./examples/data/camb_lens_nobb.dat"
      - "./examples/data/camb_lens_r1.dat"
    params:
      r_tensor: ['r_tensor', 'tophat', [-0.1, 0.0, 0.1]]
      A_lens:   ['A_lens',   'tophat', [0.0,  1.0, 2.0]]

  fg_model:
    component_1:
      name: Dust
      sed: Dust
      cl: { EE: ClPowerLaw, BB: ClPowerLaw }
      sed_parameters:
        beta_d: ['beta_d', 'Gaussian', [1.59, 0.11]]
        temp_d: ['temp',   'fixed',    [19.6]]
        nu0_d:  ['nu0',    'fixed',    [353.]]
      cl_parameters:
        BB:
          amp_d_bb:   ['amp',   'tophat', [0., 5., 10.]]
          alpha_d_bb: ['alpha', 'tophat', [-1., -0.2, 0.]]
          l0_d_bb:    ['ell0',  'fixed',  [80.]]
```

See [docs/configuration.md](docs/configuration.md) for the complete reference.

## Parameter Format

Every model parameter is defined as a three-element list:

```yaml
param_name: ['internal_name', 'prior_type', [prior_args]]
```

| Prior type | Args | Description |
|---|---|---|
| `fixed` | `[value]` | Not sampled; held constant |
| `tophat` | `[lower, center, upper]` | Uniform prior; `center` is the initial value |
| `Gaussian` | `[mean, sigma]` | Gaussian prior; `mean` is the initial value |

## Samplers

| Name | Config key | Extra options | Output |
|---|---|---|---|
| emcee MCMC | `emcee` | `nwalkers`, `n_iters` | `emcee.npz` (chain, names) |
| PolyChord nested sampling | `polychord` | `nlive`, `nrepeat` | `polychord/` directory |
| Maximum likelihood | `maximum_likelihood` | -- | `chi2.npz` (best-fit params) |
| Fisher matrix | `fisher` | -- | `fisher.npz` (params, fisher) |
| Single-point chi2 | `single_point` | -- | `single_point.npz` |
| Timing benchmark | `timing` | -- | `timing.npz` |

## Tests

Tests are shell scripts in `test/`. The fastest is:

```bash
bash test/run_sampling_test.sh
```

See [docs/examples.md](docs/examples.md) for descriptions of all test scripts and example workflows.

## Documentation

- [docs/architecture.md](docs/architecture.md) -- Module interactions, data flow, and class relationships
- [docs/configuration.md](docs/configuration.md) -- Complete configuration reference
- [docs/examples.md](docs/examples.md) -- Step-by-step usage examples and test descriptions

## Credits

Developed by the Simons Observatory BB Analysis Working Group. Questions and contributions welcome -- contact Max Abitbol (mabitbol), David Alonso (damonge), or open an issue.

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
