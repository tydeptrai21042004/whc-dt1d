#!/usr/bin/env python3
"""Validate the core runtime before launching paper experiments."""
from __future__ import annotations
import json, platform, sys
from importlib.metadata import PackageNotFoundError, version

required = {
    "torch": "2.6.0", "torchvision": "0.21.0", "timm": "1.0.15",
    "PyYAML": "6.0.2", "numpy": "1.26.4", "pandas": "2.2.3",
}
report = {"python": sys.version, "platform": platform.platform(), "packages": {}}
errors = []
for name, expected in required.items():
    try:
        observed = version(name)
    except PackageNotFoundError:
        observed = None
    report["packages"][name] = {"expected": expected, "observed": observed}
    if observed != expected:
        errors.append(f"{name}: expected {expected}, observed {observed}")
print(json.dumps(report, indent=2))
if errors:
    raise SystemExit("Environment mismatch:\n" + "\n".join(errors))
