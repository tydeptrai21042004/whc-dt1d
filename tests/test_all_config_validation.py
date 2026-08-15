from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_all_self_contained_configs_validate():
    done = subprocess.run([sys.executable, str(ROOT / "tools/validate_all_configs.py")], cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((ROOT / "reproducibility/config_validation.json").read_text())
    assert report["status"] == "PASS"
    assert report["proposal"]["method_key"] == "dt1d"
    assert report["main_experiments"] == 15
    assert report["reviewer_ablation_experiments"] == 2
