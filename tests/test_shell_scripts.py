from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_shell_scripts_have_valid_syntax():
    scripts = sorted(ROOT.rglob("*.sh"))
    assert scripts
    for script in scripts:
        completed = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
