#!/usr/bin/env python3
"""Generate a deterministic spectral illustration for DT1D-Adapter."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def coarse_response(w,beta=(0.28,0.18,0.12,0.07)):
    r=np.full_like(w,beta[0],dtype=float)
    for c,o in zip(beta[1:],(1,2,4)): r += 2*c*np.cos(o*w)
    return r
def normalize(v):
    m=np.max(np.abs(v)); return v/m if m else v
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'outputs/cnn_paper_three_seed/figures/figure_02_dt1d_spectral.png'); a=ap.parse_args()
    w=np.linspace(-np.pi,np.pi,1200); base=normalize(coarse_response(w));
    # Illustrative nonzero lambda only for visualizing the learned weighted-shift direction; training initializes lambda=0.
    lam=.35; multiplier=(1-lam)+lam*np.cos(2*w); weighted=normalize(coarse_response(w)*multiplier)
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8)); axes[0].plot(w,base,label='Compact symmetric base {1,2,4}'); axes[0].set_title('Compact base response')
    axes[1].plot(w,multiplier,label='Weighted-shift multiplier (1−λ)+λ cos(2ω)'); axes[1].plot(w,weighted,label='Illustrative weighted response'); axes[1].set_title('p=2 weighted-shift spectral modulation')
    for ax in axes:
        ax.axhline(0,linewidth=.8); ax.set_xlabel('Frequency ω'); ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],['−π','−π/2','0','π/2','π']); ax.grid(True,alpha=.25); ax.legend()
    axes[0].set_ylabel('Normalized response'); fig.suptitle('DT1D-Adapter spectral interpretation (p=2 weighted shift)',fontsize=13); fig.tight_layout(); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=300,bbox_inches='tight'); fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig); print(a.output); return 0
if __name__=='__main__': raise SystemExit(main())
