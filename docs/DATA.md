# Data provenance and access

The supplied CSV is a derivative, not the complete source release. Obtain the exact file from [Kaggle](https://www.kaggle.com/datasets/pratyaydhar/mhealth), then place it in `data/raw/mHealth.csv`. Authentication may be required. No downloader is included because availability, current derivative license metadata, and the conversion recipe could not be verified. Use the supplied file if the live resource is unavailable. Do not embed credentials in commands, notebooks, or Git.

[UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) is the authoritative source for collection details and the CC BY 4.0 license. It contains per-subject logs, including null labels; those logs are useful for a future subject-aware experiment but will not reproduce this CSV's 308,021-row experiment without the missing derivation recipe. The CSV has only labels 1–12 and no subject/session ID. A–W are numerical channel names; exact physical channel semantics in the derivative are not independently confirmed.

## Exact input identity

Use [input_manifest.json](../evidence/input_manifest.json) for original upload filenames, byte sizes, and SHA-256 hashes. The notebook is renamed in the repository but its bytes are unchanged. Use [artifact_manifest.json](../evidence/artifact_manifest.json) for every file inside `results.zip`, its hash, and whether it is included. The hidden source notebook inside the ZIP is an auxiliary archive item; the explicitly uploaded notebook is the canonical original.

Compute a local hash with `python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('data/raw/mHealth.csv').read_bytes()).hexdigest())"`. Do not change CSV row order: run detection and all downstream partitions depend on it.

## Target labels

| Original ID | Activity | Model index, 12-class task |
| --- | --- | --- |
| 1 | Standing still | 0 |
| 2 | Sitting and relaxing | 1 |
| 3 | Lying down | 2 |
| 4 | Walking | 3 |
| 5 | Climbing stairs | 4 |
| 6 | Waist bends forward | 5 |
| 7 | Frontal elevation of arms | 6 |
| 8 | Knees bending | 7 |
| 9 | Cycling | 8 |
| 10 | Jogging | 9 |
| 11 | Running | 10 |
| 12 | Jump front & back | 11 |

Processed NPZ files preserve original activity IDs. Model loading maps them to indices 0–11. The legacy 11-class variants remove original activity 2 **after** preprocessing, then map the retained activity IDs to 0–10. They therefore retain the original scaler fit rather than refitting it after class removal.

## Publication boundaries

Raw CSV, processed arrays, saved scalers, and trained weights are ignored. Aggregate run intervals and metrics are included for auditing; the preserved notebook contains a limited sensor preview. `scaler.npz` stores numerical means/scales without executable pickle deserialization. The original pickle is not loaded or redistributed. The report PDF is omitted, avoiding unnecessary redistribution of institutional and personal report pages.
