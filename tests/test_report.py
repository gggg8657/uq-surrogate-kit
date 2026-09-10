"""Structural checks on the report generator itself.

Two defects of this shape have already reached `RESULTS.md` in this repo, both
from sections being appended by successive rounds of work:

1. **A shadowed function.** Two `def sec_h15(...)` existed; Python kept the
   second, so the H16 and H17 sections were present as code and absent from the
   document for as long as that lasted.
2. **A duplicated emission.** Two consecutive blocks guarded by the same
   `scj.get("wtol")` condition each appended an H16 and an H17 verdict row, so
   the KPI table carried each of them twice.

Neither is caught by any numerical test: the document generates fine, the
numbers in it are right, and it is simply wrong in a way only a reader notices.
So they are checked here as properties of the file and of its output.

The third check is the one that matters most for the repo's central rule. A
report generator is the one place where a hand-typed number is invisible --
`RESULTS.md` says it was regenerated from JSON either way. The H17 verdict row
carried its width-factor range as a string literal (`54.7-107.6x -> 0.96-1.19x`)
until it was replaced by `_amp_range`, computed from `required_factor`. The
literal happened to be *correct*, which is exactly why it survived review, and
exactly why the check has to be mechanical rather than a matter of noticing.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "scripts" / "report.py"


def test_no_duplicate_module_level_defs():
    """Every top-level function in report.py is defined exactly once."""
    tree = ast.parse(REPORT.read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    dupes = {k: v for k, v in Counter(names).items() if v > 1}
    assert not dupes, f"shadowed function definitions in report.py: {dupes}"
    print(f"  {len(names)} top-level functions, no shadowing")


def test_each_section_heading_emitted_once():
    """No `### n.` or `#### ` heading appears twice in the generated document.

    Read from `RESULTS.md` if it is present, since that is the artefact a
    reader sees. A heading emitted twice means either a duplicated call in
    `main()` or two blocks behind the same guard.
    """
    doc = ROOT / "RESULTS.md"
    if not doc.exists():
        print("  RESULTS.md absent -- skipped")
        return
    heads = [l.strip() for l in doc.read_text().split("\n")
             if re.match(r"^#{3,4} ", l)]
    dupes = {k: v for k, v in Counter(heads).items() if v > 1}
    assert not dupes, f"duplicated headings in RESULTS.md: {list(dupes)}"
    # the numbered section labels have to be unique too, even if the titles
    # differ -- two "### 1f." sections is the same bug wearing a new title
    labels = [m.group(1) for l in heads
              if (m := re.match(r"^#{3,4} (\d+[a-z]?)\.", l))]
    ldupes = {k: v for k, v in Counter(labels).items() if v > 1}
    assert not ldupes, f"duplicated section labels in RESULTS.md: {ldupes}"
    print(f"  {len(heads)} headings, {len(labels)} numbered, all unique")


def test_no_duplicated_table_rows():
    """The KPI verdict table has no two identical rows."""
    doc = ROOT / "RESULTS.md"
    if not doc.exists():
        print("  RESULTS.md absent -- skipped")
        return
    rows = [l.strip() for l in doc.read_text().split("\n")
            if l.startswith("| ") and not set(l) <= set("|- ")]
    dupes = {k: v for k, v in Counter(rows).items() if v > 1}
    assert not dupes, ("identical rows in RESULTS.md:\n"
                       + "\n".join(f"  x{v}: {k[:110]}" for k, v in dupes.items()))
    print(f"  {len(rows)} table rows, none identical")


#: Numeric literals a report generator legitimately contains: the KPI band and
#: target, percentage and permille conversions, quantile levels, indices,
#: rounding, and the alpha the whole repo runs at.
_ALLOWED = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 32, 40, 42,
            49, 50, 100, 110, 128, 256, 512, 1000, 1024,
            0.1, 0.5, 0.88, 0.9, 0.92, 0.05, 0.95, 1.0, 1.5, 2.0, 100.0, 90.0}


def test_no_suspicious_measured_looking_literals():
    """No float literal that looks like a *measurement* is typed into report.py.

    The heuristic is deliberately narrow, because a report generator is full of
    legitimate constants: it flags floats outside `_ALLOWED` that have two or
    more decimal places, which is the shape a copied measurement takes
    (`54.7`, `107.6`, `0.96`, `1.19`) and not the shape a format width or a
    quantile level takes.
    """
    tree = ast.parse(REPORT.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            v = node.value
            if v in _ALLOWED:
                continue
            if round(v, 1) == v and abs(v) < 10:
                continue          # one-decimal small constants are formatting
            bad.append((node.lineno, v))
    assert not bad, ("measurement-shaped float literals in report.py "
                     "(compute these from the run JSON instead): "
                     + ", ".join(f"line {l}: {v}" for l, v in bad))
    print("  no measurement-shaped float literals")


def test_report_module_imports_and_exposes_its_sections():
    """report.py imports cleanly and `main` wires every `sec_*` it defines."""
    spec = importlib.util.spec_from_file_location("uqkit_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    secs = {n for n in dir(mod) if n.startswith("sec_")}
    src = REPORT.read_text()
    main_src = src[src.index("def main("):]
    unused = sorted(n for n in secs if f"{n}(" not in main_src)
    assert not unused, f"sec_* functions never called from main(): {unused}"
    print(f"  {len(secs)} sec_* functions, all called from main()")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all report tests passed")
