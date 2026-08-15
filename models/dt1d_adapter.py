"""DT1D-Adapter: the single canonical proposal used by the paper.

The architecture is frozen and identical across datasets/backbones:

    axes                  = H + W
    group size            = 16
    symmetric support     = {1, 2, 4}
    channel detail        = normalized psi4
    weighted shift p      = 2
    shift lambda          = learned independently for H and W
    lambda init / bound   = 0 / [-0.5, 0.5]
    joint H/W L1 cap      = 1
    residual gate         = learned, initialized at 0.01
    pointwise mixing      = off
    padding               = replicate
    effective kernel      = 13
    depthwise conv calls  = 2

Ablation controls live in :mod:`models.dt1d_ablation_adapter`; the proposal
itself has no dataset-specific or stage-specific architecture switches.
"""
from __future__ import annotations

from ._dt1d_weighted_core import _DT1DWeightedCore


class DT1DAdapter(_DT1DWeightedCore):
    """Frozen DT1D-Adapter proposal."""

    method_name = "DT1D-Adapter"
    proposal_name = "DT1D-Adapter"
    architecture_name = "R124-P2-G16-Axis-LearnedGate"

    def __init__(self, C: int, *, cache_kernel: bool = False) -> None:
        super().__init__(
            C=C,
            axis="hw",
            alpha_group=16,
            residual_scale=1.0,
            gate_init=0.01,
            gate_mode="learned",
            padding_mode="replicate",
            contrast_split=8,
            detail_basis="orth",
            detail_components="offset4",
            active_offsets="1,2,4",
            use_pointwise=False,
            pointwise_ratio=32,
            pointwise_groups=4,
            use_bn=False,
            cache_kernel=cache_kernel,
            project_l1=True,
            joint_l1_cap=1.0,
            shift_p=2,
            shift_lambda_mode="learned",
            shift_lambda_scope="axis",
            shift_lambda_init=0.0,
            shift_lambda_max=0.5,
            shift_normalization="mean",
        )
        self.is_dt1d_adapter = True
        self.is_canonical_dt1d_adapter = True
        self.implementation = "dt1d_r124_p2_g16_axis_learnedgate"

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, method=DT1D-Adapter, architecture={self.architecture_name}, "
            f"axes=hw, group=16, offsets={self.active_offsets}, p=2, "
            f"baseK={self.base_kernel_size}, effectiveK={self.effective_kernel_size}, "
            f"lambda=learned-axis, lambda_max={self.shift_lambda_max}, "
            f"joint_l1_cap=1, gate=learned@0.01, pointwise=False, padding=replicate"
        )


__all__ = ["DT1DAdapter"]
