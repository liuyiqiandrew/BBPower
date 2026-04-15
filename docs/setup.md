# Setup and Entry Points

This page is the quickest way to get BBPower running with the fewest moving parts. It focuses on:

- which dependencies are needed for which stages
- what files each stage expects
- what to run first if something is missing

## 1. Create an environment

Use any Python environment manager you prefer. A plain virtualenv works:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

BBPower requires Python >= 3.10.

## 2. Install only what you need

The package is deliberately split into extras so you do not need heavy map-level dependencies unless you are running those stages.

| Goal | Stages | Install command | Extra notes |
|---|---|---|---|
| Inspect configs, use shared helpers, run lightweight pieces | Base package, `BBPowerSummarizer` | `pip install -e .` | Does **not** include `fgbuster`, `getdist`, `pymaster`, or `pyshtools` |
| Run component separation on pre-computed spectra | `BBCompSep` | `pip install -e ".[compsep]"` | This is the most common setup |
| Run component separation and make plots | `BBCompSep`, `BBPlotter` | `pip install -e ".[compsep,plotting]"` | Needed for triangle plots via `getdist` |
| Use moment-expanded foreground models or Fisher runs | `BBCompSep` | `pip install -e ".[compsep,plotting,sampling]"` | Adds `pyshtools` and `numdifftools` |
| Run the full maps-to-parameters pipeline | All four stages | `pip install -e ".[all]"` | Includes `healpy` and `pymaster` |

Practical notes:

- `BBCompSep` always needs `fgbuster`.
- `BBPlotter` only needs `getdist` if you want likelihood contours from `emcee` chains.
- `BBCompSep` with `fg_model.use_moments: true` needs `pyshtools` for Wigner 3-j calculations.
- `sampler: polychord` requires a separate PolyChord installation that is not provided by `pyproject.toml`.

## 3. Verify the install

```bash
python -m bbpower --help
python -c "import bbpower; print(bbpower.__file__)"
```

If you are working with multiple clones or installs, the second command is the fastest way to confirm which checkout Python is importing.

## 4. Pick the lowest-friction entry point

Most users do **not** need to start from maps. If you already have SACC spectra, start at `BBCompSep`.

| Starting point | When to use it | Required inputs |
|---|---|---|
| `BBPowerSpecter` | You only have HEALPix Q/U maps and simulations | `splits_list`, `masks_apodized`, `bandpasses_list`, `beams_list`, `sims_list` |
| `BBPowerSummarizer` | You already have split-level spectra from maps/sims | `cells_all_splits`, `cells_all_sims`, `splits_list`, `bandpasses_list` |
| `BBCompSep` | You already have coadded spectra and covariance | `cells_coadded`, `cells_noise`, `cells_coadded_cov`, config |
| `BBPlotter` | You already have BBPower outputs and want diagnostics | `cells_coadded*`, `cells_fiducial`, `param_chains`, plot paths |

## 5. Minimal file checklist for `BBCompSep`

To run:

```bash
python -m bbpower BBCompSep \
  --cells_coadded=... \
  --cells_noise=... \
  --cells_fiducial=... \
  --cells_coadded_cov=... \
  --output_dir=... \
  --config_copy=... \
  --config=...
```

You need:

- `cells_coadded`: coadded signal estimate in SACC format
- `cells_noise`: noise spectra in SACC format
- `cells_coadded_cov`: covariance in SACC format
- `config`: stage config YAML
- `output_dir`: an existing writable directory

Important detail:

- `cells_fiducial` is still a required CLI / stage input today for both likelihood modes.
- In practice, it is only used by `likelihood_type: h&l`.
- For `likelihood_type: chi2`, you still need to pass a path, even though the stage does not use the file contents at runtime.

## 6. What each stage writes

| Stage | Main outputs |
|---|---|
| `BBPowerSpecter` | `cells_all_splits.fits`, `cells_all_sims.txt`, workspace files under the `mcm` prefix |
| `BBPowerSummarizer` | `cells_coadded.fits`, `cells_coadded_total.fits`, `cells_noise.fits`, `cells_null.fits` |
| `BBCompSep` | sampler-specific files in `output_dir` such as `emcee.npz`, `chi2.npz`, `single_point.npz`, `fisher.npz`, `cells_model.npz`, plus `config_copy.yml` |
| `BBPlotter` | `plots.dir/`, `plots_page.html`, and optionally `triangle.png` |

## 7. Recommended smoke tests

These are the fastest ways to confirm a given install actually runs the stages you care about.

```bash
# Component separation only
bash test/run_compsep_test.sh

# Component separation + plot generation
bash test/run_sampling_test.sh

# Predicted spectra mode
bash test/run_predicted_spectra_test.sh
```

If you installed the full map-level stack:

```bash
bash test/run_power_specter_test.sh
```

## 8. Common setup failures

### `ModuleNotFoundError: fgbuster`

You installed the base package only. Reinstall with:

```bash
pip install -e ".[compsep]"
```

### `ModuleNotFoundError: getdist`

You are trying to make triangle plots without the plotting extra:

```bash
pip install -e ".[plotting]"
```

### `ModuleNotFoundError: pyshtools`

You are using moment-expanded foreground models:

```bash
pip install -e ".[sampling]"
```

### `ModuleNotFoundError: pymaster` or `healpy`

You are trying to run `BBPowerSpecter` without the map-level dependencies:

```bash
pip install -e ".[power-spectra]"
```

### `BBCompSep` fails when copying `config_copy.yml`

Make sure `--output_dir` already exists. BBPower writes into that directory but does not create every intermediate parent path for you.

## 9. Where to go next

- [README.md](../README.md) for the main project overview
- [architecture.md](architecture.md) for stage internals and data flow
- [configuration.md](configuration.md) for all YAML options
- [examples.md](examples.md) for concrete workflows and commands
