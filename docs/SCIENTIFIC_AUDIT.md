# Scientific audit

Scope: all 32 notebook cells, their saved outputs and figures, the supplied CSV, named files in `results.zip`, and the accompanying report as secondary context. The uploaded notebook and saved CSV/checkpoint evidence control reported results when prose conflicts with code.

## Findings before refactoring

1. The primary executed model is **12-class**, despite an 11-class Markdown introduction and legacy `cnn_multiscale_bilstm_11class` directory. The archived folder is retained verbatim; refactored config names remove the ambiguity.
2. Activity-2-removed experiments solve a different **11-class** task. Their 893 test windows must not be compared directly with the 987-window task as though the target were unchanged.
3. Hybrid code uses recurrent dropout **0.0** and AdamW; the adjacent Markdown describes recurrent dropout 0.25 and Adam. Code and checkpoint topology are authoritative.
4. Preprocessing exports balanced class weights, but the hybrid and CNN use **manual** weights (1.0 except jumping 1.6). Stacked recurrent baselines compute balanced weights. The report's broad claim that all model weights are training-derived is therefore incomplete.
5. Hybrid, LSTM, and BiLSTM select the first strict maximum of validation Macro-F1. CNN cells save the **final executed epoch** with `restore_best_weights=False`, not a best-Macro-F1 checkpoint. No checkpoint correction was silently applied.
6. Recurrent baselines allow 60 epochs at learning rate 2e-4 and L2 1e-4; hybrid/CNN allow 35 at 2.5e-4 and L2 1.5e-4. Comparisons cannot isolate recurrence or receptive-field effects under a common protocol.
7. Run targets 70/15/15 yield actual window proportions approximately **66.59/12.39/21.02%**. Rounding per activity is preserved, not replaced by a new split algorithm.
8. Label-run separation prevents the same recorded row/run entering multiple partitions. Missing subject/session metadata prevents subject-independent claims and may hide same-label recording boundaries. A run should not be called a verified recording session.
9. Window generation uses target labels to avoid transitions. This label-dependent segmentation differs from deployment on an unsegmented stream.
10. Test windows overlap and are dependent. No repeated-seed variance, independent external validation, calibration, or significance evidence is available. Model-selection code uses validation only, but artifact review cannot verify all prior human development choices.
11. The final 11-class recurrent cell has no execution count/output. Empty cells and `print("test")` add clutter. Duplicated imports, helpers, feature tables, and plotting code are moved to reusable modules in the portable version; original bytes remain intact.
12. Original runtime facts conflict: notebook metadata says GPU disabled, while execution outputs show two Tesla T4 devices. Keras checkpoint metadata records 3.13.2. Python metadata records 3.12.13. A complete original environment cannot be recovered from these facts alone.

## Experiment lineage

Cell numbers below are zero-based indices in the uploaded notebook.

| Source cell | Task/model | Archived directory | Refactored config | Result status |
| --- | --- | --- | --- | --- |
| 3 | Run-based preprocessing | Root of `evidence/original` | `preprocessing.json` | Assignments and labels match; float arrays within tolerance |
| 7 | 12-class multi-scale CNN + BiLSTM | `cnn_multiscale_bilstm_11class` | `hybrid.json` | Saved inference and all test metrics reproduced |
| 9 | 12-class CNN | `cnn_only_12class` | `cnn.json` | Saved inference and all test metrics reproduced |
| 10 | 11-class CNN, activity 2 removed | `cnn_only_11class` | `cnn11.json` | Saved inference and all test metrics reproduced |
| 12 | 12-class stacked LSTM | `lstm_only_generalization_v1` | `lstm.json` | Saved inference and all test metrics reproduced |
| 13 | 12-class stacked BiLSTM | `bilstm_only_generalization_v1` | `bilstm.json` | Saved inference and all test metrics reproduced |
| 22 | 11-class single-layer BiLSTM, titled LSTM | No saved artifact | `bilstm11.json` | Architecture/config retained; no performance claimed |

The CNN's final-epoch behavior is intentional historical compatibility in the default config, not a recommendation for future methodology. A prospective common-selection experiment should use a new config/output folder and report new metrics separately.

## Reconciled observations

All five archived raw confusion matrices reproduce their corresponding aggregate test CSV metrics to absolute tolerance 1e-12. The primary matrix contains 917 correct and 70 incorrect predictions. Its error directions are sitting → knees bending (47), knees bending → stairs (10), running → jogging (11), and jumping → jogging (2). Some report prose overinterprets the physical similarity of these classes; the repository reports counts without assigning causal explanations.

The report rounds the hybrid-over-BiLSTM accuracy difference to 5.68 points by subtracting displayed percentages. Using unrounded CSV values gives **5.673758865 points**, displayed here as **5.67**. No underlying score is changed.

## Refactor decisions

Preserved: raw class IDs, float32 sensor representation, activity order, RNG and split allocation, train-only StandardScaler, window/stride, architectures and regularization, original weights, optimizer, epoch budgets, callback ordering, and checkpoint criteria.

Changed for engineering quality: portable paths/configs, explicit validation exceptions, prevention of accidental output overwrites, lazy optional TensorFlow imports, safe numerical scaler storage, run/window provenance, environment/selection records, isolated test evaluation, executable tests/CI, and aggregate plotting. AdamW is required explicitly instead of silently falling back to Adam; this preserves the optimizer observed in the executed notebook.

No architectural, class-selection, split-policy, or benchmark-improvement experiment was introduced. One synthetic training step and a synthetic CLI training run validate plumbing; neither is evidence of predictive performance.
