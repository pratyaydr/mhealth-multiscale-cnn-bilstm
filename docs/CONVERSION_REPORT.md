# Repository conversion report

## Changed

Extracted the duplicated notebook pipeline into a compact installable `mhealth_har` package: data preparation, shared experiment configs, five recorded architectures/task variants plus one unexecuted historical variant, training, evaluation, aggregate evidence auditing, and reproducible plotting. Added portable commands, validation errors, output protection, provenance/hashes, environment records, synthetic tests, and CI configuration. A shorter executed notebook guides readers through the reusable modules and supplied evidence.

## Preserved

The original uploaded notebook is byte-preserved with all outputs. Aggregate CSV/JSON artifacts are copied without editing. The primary 12-class hybrid, 12-class baselines, and distinct 11-class task retain their source methodology. Original data ordering, run allocation, standardization fit population, windowing, weights, models, optimizer, callbacks, and selection rules are retained. No scientific correction is mixed into the historical result table.

## Tested

See [validation evidence](VALIDATION.md) for exact outcomes. Tests cover temporal boundaries and overlap, deterministic complete-run allocation, train-only scaler fitting, invalid inputs, label mapping, output protection, confusion-count metric reconciliation, model dimensions/parameter counts, a real synthetic optimization step, and checkpoint save/load. The supplied CSV was processed, original run assignments/labels reconciled, and all five original saved checkpoints evaluated on their archived test arrays. Refactored models receiving the original weights matched checked original probabilities exactly.

## Unverified

Full end-to-end retraining from initialization, seed-to-seed variability, exact historical package installation, GPU parity, actual GitHub rendering/Actions execution, current Kaggle derivative license/access metadata, and unseen-subject performance. The original Kaggle notebook was not rerun top-to-bottom because it launches multiple long training experiments; its runnable portable replacement is the package and walkthrough. The live Kaggle page was inaccessible, so the uploaded snapshot controls the audit.

## Reproduce and publish

Use the README to install, run the included evidence audit, rebuild the exact local CSV, train an isolated experiment, and evaluate explicitly. `scripts/validate_artifacts.py` checks the supplied CSV/ZIP and optionally checkpoint inference. [Publishing instructions](PUBLISHING.md) provide repository metadata and initial commit/push commands. No remote publication was performed. Raw data, processed arrays, trained weights, report PDF, and local caches are excluded from the bundle. No software license is assumed.

## Final quality review

| Category | Evidence and outcome |
| --- | --- |
| Scientific accuracy | Historical task variants separated; saved checkpoint metrics reproduced; no new benchmark claims |
| Code quality | Compact package, portable configs, docstrings, lint and formatting passed |
| Reproducibility | Input hashes, environment snapshot, data instructions, executed artifact validator; full retraining remains unverified |
| Documentation | README, scientific audit, data provenance, validation record, and publication guide complete |
| Security | No detected credential patterns; raw CSV/arrays/weights and report omitted; original previews disclosed |
| Portfolio quality | Demonstrates actual temporal modeling, evaluation, and engineering work with explicit limitations |
| GitHub presentation | Local links and source code checked; remote rendering and Actions await publication |
