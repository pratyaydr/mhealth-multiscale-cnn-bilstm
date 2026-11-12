# Validation record

Validation was performed during conversion on 9 November 2026 (Asia/Kolkata). See [the environment snapshot](../evidence/validation_environment.json) for exact Python/package versions. It records Linux CPU validation with TensorFlow CPU 2.20.0 and Keras 3.13.2; it is not a recovered Kaggle environment lock.

| Check | Executed outcome |
| --- | --- |
| Editable package installation/imports | Passed |
| pytest suite | 18 passed, including optional TensorFlow tests |
| Ruff lint and format | Passed on source, tests, and scripts |
| Original recorded metrics | All five raw confusion matrices reproduce aggregate CSV test metrics to absolute tolerance 1e-12 |
| Supplied CSV preparation | 3,127 / 582 / 987 windows with original split sizes |
| Original run assignments / labels | Exactly equal |
| Standardized float32 arrays | Not bit-identical; maximum absolute difference 1.9073486328125e-6, within tested tolerance 3e-6 |
| Saved checkpoint inference | All five experiments reproduce every aggregate test metric on archived test arrays |
| Refactored architecture parity | Same layer types, weight shapes, and parameter counts; original weights yield identical probabilities on the first 32 test windows for every archived model |
| Synthetic model optimization | One real optimization step, finite loss, then save/load parity |
| CLI training/evaluation | One synthetic 12-class epoch plus explicit test evaluation passed; no synthetic score is used as a benchmark |
| New evidence walkthrough | All 8 code cells executed sequentially in-process with captured tables/images |
| Notebook structure / code | New notebook validated with nbformat; original and new code cells parse |
| HTML export | Succeeded; full browser layout inspection was blocked by the environment's socket restriction |
| Figures | All three generated figures visually inspected; original notebook figures reviewed as a contact sheet |
| Markdown links | Local destinations checked; external GitHub/Kaggle availability is not asserted |
| Source preservation and publication contents | Notebook/artifact hashes and archive contents checked; raw data and model binaries excluded |
| Full training from initialization | Not performed |
| Original multi-experiment notebook execution | Not repeated; historical outputs preserved |
| Python 3.11 / Windows / GPU execution | Not performed |
| GitHub-hosted Actions / Markdown / Mermaid rendering | Pending publication |

## Important execution distinctions

The standard `jupyter nbconvert --execute` attempt could not start its ZeroMQ kernel because this environment forbids the required socket creation. The walkthrough was instead executed with:

```bash
python scripts/execute_walkthrough.py
```

That transparent IPython fallback runs all cells in order, stops on exceptions, and saves captured rich outputs and execution-method metadata. It is not validation of Jupyter kernel transport. Normal environments and CI can use:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/reproducible_walkthrough.ipynb
```

The exported HTML's browser launch was separately blocked by socket creation. Figure PNGs were visually inspected, but no claim is made that the entire rendered notebook or README has been visually checked in GitHub. NBConvert's HTML export emitted missing-alt-text warnings for embedded image outputs; the README images have descriptive alt text.

The test environment emitted upstream Keras/NumPy and pytest-plugin deprecation warnings. Tests passed; warnings are not represented as failures or hidden by broad suppression.

## Repeat the checks

From the repository root:

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/check_repository.py
python -m mhealth_har audit
python scripts/make_figures.py
python scripts/validate_artifacts.py --csv /path/to/mHealth.csv --results /path/to/results.zip --checkpoints
```

[Preprocessing evidence](../evidence/preprocessing_validation.json) records exact labels/assignments and observed array differences. [Checkpoint evidence](../evidence/checkpoint_validation.json) records parameter counts, reproduced scores, and refactor parity. StandardScaler numerical differences across environments are observed; their exact cause has not been established. Never relabel numerical agreement as bitwise equality or saved-weight inference as full retraining.
