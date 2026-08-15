#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('run_dir',type=Path); a=p.parse_args()
root=a.run_dir.resolve(); rows={}
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'SHA256SUMS.json':
        rows[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'SHA256SUMS.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(root/'SHA256SUMS.json')
