#!/usr/bin/env python3
"""Create manuscript Figure 1 from exactly one representative training seed."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load_history(root: Path, seed: int):
    matches=[]
    for path in sorted(root.rglob("history.json")):
        mp=path.parent/"run_metadata.json"; sp=path.parent/"run_status.json"
        if not mp.is_file() or not sp.is_file(): continue
        meta=json.loads(mp.read_text()); status=json.loads(sp.read_text())
        if status.get("return_code")!=0: continue
        if meta.get("target")=="figure_01" and meta.get("method_preset")=="dt1d" and int(meta.get("independent_seed",-1))==seed: matches.append(path)
    if len(matches)!=1: raise SystemExit(f"Expected exactly one successful Figure-1 history for seed {seed}; found {len(matches)}")
    return json.loads(matches[0].read_text())
def series(history,key): return np.asarray([float(row.get(key,np.nan)) for row in history],dtype=float)
def maybe_percent(v):
    finite=v[np.isfinite(v)]; return v*100.0 if finite.size and float(np.nanmax(np.abs(finite)))<=1.5 else v
def save(fig,output):
    output.parent.mkdir(parents=True,exist_ok=True); png=output.with_suffix('.png'); pdf=output.with_suffix('.pdf'); fig.savefig(png,dpi=300,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight'); print(png); print(pdf)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT/'outputs/cnn_paper_revision'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--output',type=Path,default=ROOT/'outputs/cnn_paper_revision/figures/figure_01_seed0.png'); a=ap.parse_args()
    root=a.root if a.root.is_absolute() else ROOT/a.root; h=load_history(root,a.seed)
    ta=maybe_percent(series(h,'train_class_acc')); va=maybe_percent(series(h,'val_acc1')); tl=series(h,'train_loss'); vl=series(h,'val_loss'); n=min(map(len,[ta,va,tl,vl])); e=np.arange(1,n+1)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6)); axes[0].plot(e,ta[:n],label='Train accuracy'); axes[0].plot(e,va[:n],label='Validation accuracy'); axes[0].set(xlabel='Epoch',ylabel='Accuracy (%)',title=f'Accuracy behavior (representative seed {a.seed})'); axes[0].legend(); axes[0].grid(True,alpha=.25)
    axes[1].plot(e,tl[:n],label='Train loss'); axes[1].plot(e,vl[:n],label='Validation loss'); axes[1].set(xlabel='Epoch',ylabel='Loss',title=f'Loss behavior (representative seed {a.seed})'); axes[1].legend(); axes[1].grid(True,alpha=.25)
    fig.suptitle('Caltech101 + ResNet-18 + DT1D-Adapter'); fig.tight_layout(); save(fig,a.output); plt.close(fig); return 0
if __name__=='__main__': raise SystemExit(main())
