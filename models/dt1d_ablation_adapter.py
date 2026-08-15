"""Reviewer-only one-factor-at-a-time DT1D ablation implementation.

This class is intentionally not exposed as a proposal. It keeps the same
weighted axial core as the canonical DT1D-Adapter while allowing the exact
components requested by the reviewers to be disabled or altered.
"""
from __future__ import annotations

from typing import Sequence
from ._dt1d_weighted_core import _DT1DWeightedCore


class DT1DAblationAdapter(_DT1DWeightedCore):
    """Configurable DT1D core for controlled ablations only."""

    method_name = "DT1D reviewer ablation"
    proposal_name = "reviewer_ablation_only"

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        alpha_group: int = 16,
        gate_init: float = 0.01,
        gate_mode: str = "learned",
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_components: str = "offset4",
        active_offsets: str | Sequence[int] = "1,2,4",
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        cache_kernel: bool = False,
        project_l1: bool = True,
        joint_l1_cap: float = 1.0,
        shift_p: int = 2,
        shift_lambda_mode: str = "learned",
        shift_lambda_scope: str = "axis",
        shift_lambda_init: float = 0.0,
        shift_lambda_max: float = 0.5,
        shift_normalization: str = "mean",
    ) -> None:
        super().__init__(
            C=C,
            axis=axis,
            alpha_group=alpha_group,
            residual_scale=1.0,
            gate_init=gate_init,
            gate_mode=gate_mode,
            padding_mode=padding_mode,
            contrast_split=contrast_split,
            detail_basis="orth",
            detail_components=detail_components,
            active_offsets=active_offsets,
            use_pointwise=use_pointwise,
            pointwise_ratio=pointwise_ratio,
            pointwise_groups=pointwise_groups,
            use_bn=use_bn,
            cache_kernel=cache_kernel,
            project_l1=project_l1,
            joint_l1_cap=joint_l1_cap,
            shift_p=shift_p,
            shift_lambda_mode=shift_lambda_mode,
            shift_lambda_scope=shift_lambda_scope,
            shift_lambda_init=shift_lambda_init,
            shift_lambda_max=shift_lambda_max,
            shift_normalization=shift_normalization,
        )
        self.is_dt1d_ablation = True
        self.is_reviewer_ablation = True


__all__ = ["DT1DAblationAdapter"]
