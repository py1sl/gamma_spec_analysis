# gamma_spec_analysis
Tools for Gamma spec analysis

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/py1sl/gamma_spec_analysis.git
cd gamma_spec_analysis
pip install -e ".[dev]"
```

The `-e` flag installs the package in *editable* mode so that changes to the
source files are immediately reflected without re-installing.

### Runtime-only install

```bash
pip install .
```

### Legacy / no-packaging install

If you prefer not to install the package you can still run the code by adding
the `src/` directory to your Python path:

```bash
export PYTHONPATH=/path/to/gamma_spec_analysis/src:$PYTHONPATH
```

## Dependencies

Runtime dependencies are declared in `pyproject.toml` and mirrored in
`requirements.txt`:

| Package | Minimum version |
|---------|----------------|
| numpy | 1.21.1 |
| scipy | 1.7.0 |
| matplotlib | 3.4.2 |
| pandas | 1.3.0 |

Development / test extras (`pip install -e ".[dev]"`):
* `pytest >= 7.0`
* `pytest-cov >= 4.0`
* `flake8 >= 6.0`

## Modules

| Module | Purpose |
|--------|---------|
| `ph_spectrum` | `PhSpectrum` dataclass – the core data model |
| `gs_spe_reading` | Read ORTEC `.Spe` / `.spe` files into `PhSpectrum` objects |
| `gs_analysis` | Peak finding, background subtraction, efficiency fitting |
| `gs_plotting` | Matplotlib helpers for visualising spectra and fits |
| `gs_creator` | Synthesise `PhSpectrum` objects from peak lists (useful for testing) |

## Running the tests

```bash
pytest --cov=src --cov-report=term-missing
```

All test data lives in `test_data/`, and pytest can be run from the repository
root.

## Examples

Jupyter notebooks with worked examples are in the `examples/` directory:

* `examples/gs_creator_co60_example.ipynb` – creating a synthetic Co-60 spectrum
* `examples/peak_find_example.ipynb` – automated peak finding

The notebooks add `src/` to the Python path when run from the `examples/`
directory.
