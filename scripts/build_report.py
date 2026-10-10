"""
FILE PURPOSE: builds docs/PROJECT_REPORT.md from docs/PROJECT_REPORT.template.md by inserting the
CURRENT evaluation/report.txt (so the report never contains hand-typed or stale numbers).
USAGE: python scripts/build_report.py      CALLED BY: humans after running the evaluation.
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
tpl = (root / "docs" / "PROJECT_REPORT.template.md").read_text(encoding="utf-8")
rep = root / "evaluation" / "report.txt"
body = rep.read_text(encoding="utf-8") if rep.exists() else "Not evaluated yet"
(root / "docs" / "PROJECT_REPORT.md").write_text(tpl.replace("<<EVALUATION_REPORT>>", body), encoding="utf-8")
print("wrote docs/PROJECT_REPORT.md")
