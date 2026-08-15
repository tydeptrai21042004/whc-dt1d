#!/usr/bin/env python3
"""Launch one or more self-contained classification experiments.

This is a convenience wrapper around ``tools/run_experiment.py``.  The clean
release commits one YAML per experiment under ``configs/experiments`` and does
not generate per-seed YAML configuration fragments.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG_DIR=ROOT/"configs"/"experiments"
INDEX=CONFIG_DIR/"index.yaml"

def parse_names(value:str)->list[str]:
    index=yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    if value=="all":
        return list(index["main_experiments"])
    if value=="ablations":
        return list(index["reviewer_ablations"])
    if value=="all-with-ablations":
        return list(index["main_experiments"])+list(index["reviewer_ablations"])
    return [x.strip() for x in value.split(",") if x.strip()]

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target",default="all",help="name(s), all, ablations, or all-with-ablations")
    ap.add_argument("--methods",default=None)
    ap.add_argument("--seeds",default=None)
    ap.add_argument("--output-root",type=Path,default=ROOT/"outputs"/"experiments")
    ap.add_argument("--data-path",type=Path,default=ROOT/"data")
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("--plan-only",action="store_true")
    ap.add_argument("--skip-if-complete",action="store_true")
    ns=ap.parse_args()
    for name in parse_names(ns.target):
        config=CONFIG_DIR/f"{name}.yaml"
        if not config.is_file():
            raise SystemExit(f"Unknown experiment {name!r}: {config} not found")
        cmd=[
            sys.executable,str(ROOT/"tools"/"run_experiment.py"),str(config),
            "--output-root",str(ns.output_root),
            "--data-path",str(ns.data_path),
            "--device",ns.device,
        ]
        if ns.methods: cmd += ["--methods",ns.methods]
        if ns.seeds: cmd += ["--seeds",ns.seeds]
        if ns.smoke: cmd.append("--smoke")
        if ns.dry_run: cmd.append("--dry-run")
        if ns.plan_only: cmd.append("--plan-only")
        if ns.skip_if_complete: cmd.append("--skip-if-complete")
        print(" ".join(cmd),flush=True)
        rc=subprocess.call(cmd,cwd=ROOT)
        if rc: return rc
    return 0

if __name__=="__main__":
    raise SystemExit(main())
