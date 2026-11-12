# GitHub publication

The authenticated destination account is `pratyaydr`. This document provides the publication instructions for `https://github.com/pratyaydr/mhealth-multiscale-cnn-bilstm`. Creating the remote repository and pushing still require completion and verification.

**Recommended name:** `mhealth-multiscale-cnn-bilstm`

**Description:** Reproducible wearable activity recognition with multi-scale CNN–BiLSTM, run-separated preprocessing, baseline comparisons, and audited experimental evidence.

**Suggested topics:** `human-activity-recognition`, `time-series-classification`, `wearable-sensors`, `deep-learning`, `tensorflow`, `keras`, `cnn`, `bilstm`, `mhealth`, `reproducible-research`.

**Initial commit message:** `feat: package mHealth HAR experiments with reproducible pipeline and audited results`

## Publish the prepared project

1. Extract the bundle and enter the project directory. Install/test as documented in the README.
2. The verified GitHub URL and README links are already filled. Leave LinkedIn blank until a verified URL is supplied.
3. Choose a software license for your own work if desired. No LICENSE file is included without a choice; keep dataset attribution separate.
4. Create a new empty GitHub repository with the recommended name. Do not initialize it with unrelated contents.
5. Run the commands below from the extracted project root:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "feat: package mHealth HAR experiments with reproducible pipeline and audited results"
git remote add origin https://github.com/pratyaydr/mhealth-multiscale-cnn-bilstm.git
git push -u origin main
```

These commands are a handoff, not operations already performed. Configure your local Git name/email if Git asks. Inspect the staged files before committing; no dataset or model binary should appear. For an existing repository, inspect its history and files first and integrate on a branch; do not force-push or replace unrelated work.

After publication, verify GitHub's README/Mermaid rendering and first Actions run. Local Markdown link validation and tests have been executed; GitHub-hosted rendering and CI remain pending. Optionally pin the repository on your profile and link the original Kaggle experiment. Avoid state-of-the-art or deployment-readiness claims.
