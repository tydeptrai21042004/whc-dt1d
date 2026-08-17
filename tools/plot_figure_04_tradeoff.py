#!/usr/bin/env python3
"""Create manuscript Figure 4 from exactly one representative seed."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read_json(p): return json.loads(p.read_text()) if p.is_file() else {}
def save(fig,output):
    output.parent.mkdir(parents=True,exist_ok=True); png=output.with_suffix('.png'); pdf=output.with_suffix('.pdf'); fig.savefig(png,dpi=300,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight'); print(png); print(pdf)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT/'outputs/cnn_paper_revision'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--output',type=Path,default=ROOT/'outputs/cnn_paper_revision/figures/figure_04_seed0.png'); a=ap.parse_args(); root=a.root if a.root.is_absolute() else ROOT/a.root; rows=[]
    for mp in sorted((root/'figure_04').rglob('run_metadata.json')):
        rd=mp.parent; meta=read_json(mp); status=read_json(rd/'run_status.json'); test=read_json(rd/'test_summary.json'); conv=read_json(rd/'convergence_summary.json')
        if status.get('return_code')!=0 or 'acc1' not in test or int(meta.get('independent_seed',-1))!=a.seed: continue
        params=conv.get('n_trainable_parameters');
        if params is None: raise SystemExit(f'Missing trainable parameters: {rd}')
        rows.append((str(meta.get('method_label',meta.get('method_preset'))),float(params),float(test['acc1'])))
    if not rows: raise SystemExit(f'No successful Figure-4 seed-{a.seed} runs under {root}')
    labels=[r[0] for r in rows]
    if len(labels)!=len(set(labels)): raise SystemExit(f'Duplicate Figure-4 method labels for seed {a.seed}')
    fig,ax=plt.subplots(figsize=(9.5,6.0))
    for label,params,acc in sorted(rows): ax.scatter(params,acc); ax.annotate(label,(params,acc),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xscale('log'); ax.set_xlabel('Trainable parameters (log scale)'); ax.set_ylabel('Test top-1 accuracy (%)'); ax.set_title(f'Accuracy–parameter trade-off (representative seed {a.seed})'); ax.grid(True,alpha=.25); fig.tight_layout(); save(fig,a.output); plt.close(fig); return 0
if __name__=='__main__': raise SystemExit(main())
