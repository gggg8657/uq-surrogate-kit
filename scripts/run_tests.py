"""Run every test file, so CI and a human have the same one-liner."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fail = 0
for t in sorted((root / "tests").glob("test_*.py")):
    print(f"\n=== {t.name} ===", flush=True)
    r = subprocess.run([sys.executable, str(t)], cwd=root,
                       env={"PYTHONPATH": str(root), **dict(__import__("os").environ)})
    fail += r.returncode != 0
sys.exit(1 if fail else 0)
