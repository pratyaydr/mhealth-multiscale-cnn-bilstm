"""Regenerate README figures from immutable recorded CSV evidence."""

from pathlib import Path

from mhealth_har.figures import make_figures

root = Path(__file__).resolve().parents[1]
for path in make_figures(root / "evidence/original", root / "assets/figures"):
    print(path.relative_to(root))
