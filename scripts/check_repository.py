"""Validate notebook code and local Markdown destinations without network access."""

import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for notebook_path in (ROOT / "notebooks").glob("*.ipynb"):
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
for document in ROOT.rglob("*.md"):
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        assert (document.parent / target).exists(), (document, target)
print("Notebook code and local Markdown links passed.")

# Source snapshots must remain byte-faithful to the supplied upload/archive.
inputs = json.loads((ROOT / "evidence/input_manifest.json").read_text())
original = next(item for item in inputs if item["filename"].endswith(".ipynb"))
notebook = ROOT / "notebooks/original_experiment.ipynb"
assert hashlib.sha256(notebook.read_bytes()).hexdigest() == original["sha256"]
for item in json.loads((ROOT / "evidence/artifact_manifest.json").read_text()):
    if item["included"]:
        relative = Path(item["path"]).relative_to("mHealth_preprocessed")
        path = ROOT / "evidence/original" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
print("Original notebook and included evidence hashes passed.")
