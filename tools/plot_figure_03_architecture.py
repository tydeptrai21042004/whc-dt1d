#!/usr/bin/env python3
"""Generate the final DT1D-Adapter architecture figure."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
ROOT = Path(__file__).resolve().parents[1]
def box(ax,x,y,w,h,text,fontsize=9):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",fill=False,linewidth=1.2); ax.add_patch(patch)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize)
def arrow(ax,x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="->",mutation_scale=12,linewidth=1.1))
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'outputs/cnn_paper_three_seed/figures/figure_03_dt1d_architecture.png'); a=ap.parse_args()
    fig,ax=plt.subplots(figsize=(14.5,5.9)); ax.set_xlim(0,14.5); ax.set_ylim(0,6.5); ax.axis('off')
    box(ax,.2,2.65,1.5,1,'Frozen visual\nbackbone block'); box(ax,1.95,2.65,1.35,1,'Feature map\nB×C×H×W')
    box(ax,3.65,4.45,2.0,.85,'Group-shared base\n0, ±1, ±2, ±4'); box(ax,3.65,3.15,2.0,.85,'Zero-mean channel\ncontrast, group 16'); box(ax,3.65,1.85,2.0,.85,'Normalized ψ₄\ndetail correction')
    box(ax,6.05,3.25,2.1,1.05,'Weighted shift p=2\n(1−λ)k + λ/2\n(S₋₂k + S₊₂k)'); box(ax,8.55,3.25,1.55,1.05,'Joint H/W\nL1 projection')
    box(ax,10.45,4.15,1.6,.85,'13×1 depthwise\nheight filter'); box(ax,10.45,2.35,1.6,.85,'1×13 depthwise\nwidth filter')
    box(ax,12.4,3.25,.65,1.0,'Σ'); box(ax,13.35,3.25,.9,1.0,'learned γ\ninit 0.01 + skip')
    arrow(ax,1.7,3.15,1.95,3.15); arrow(ax,3.3,3.15,3.65,4.85); arrow(ax,3.3,3.15,3.65,3.55); arrow(ax,3.3,3.15,3.65,2.25)
    arrow(ax,5.65,4.85,6.05,3.9); arrow(ax,5.65,3.55,6.05,3.75); arrow(ax,5.65,2.25,6.05,3.55); arrow(ax,8.15,3.78,8.55,3.78)
    arrow(ax,10.1,3.8,10.45,4.55); arrow(ax,10.1,3.7,10.45,2.75); arrow(ax,12.05,4.55,12.4,3.8); arrow(ax,12.05,2.75,12.4,3.55); arrow(ax,13.05,3.75,13.35,3.75)
    ax.plot([2.6,2.6,13.8],[2.65,.62,.62],linestyle='--',linewidth=1.0); arrow(ax,13.8,.62,13.8,3.25)
    ax.text(7.25,5.95,'DT1D-Adapter: R124-P2-G16-Axis-LearnedGate',ha='center',fontsize=14,fontweight='bold')
    ax.text(7.25,.18,'Two fused axial convolutions; learned per-axis λ; learned residual gate; no pointwise mixer',ha='center',fontsize=10)
    fig.tight_layout(); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=300,bbox_inches='tight'); fig.savefig(a.output.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig); print(a.output); return 0
if __name__=='__main__': raise SystemExit(main())
