# models/tuning_modules/__init__.py

from .prompter import PadPrompter
from .conv_adapter import ConvAdapter, LinearAdapter
from .program_module import ProgramModule

# Baselines
from .ssf import SSF
from .lora_conv import LoRAConv2d, apply_lora_conv2d
from .bam_adapter import BAMAdapter


def set_tuning_config(tuning_method, args):
    """
    Return a small config dict describing the chosen tuning method.
    Normalize the public method names used by the clean release.
    """
    alias = {
        "conv": "conv_adapt",
        "conv-adapter": "conv_adapt",
        "conv_adapter": "conv_adapt",
        "dt": "dt1d",
        "dt1d": "dt1d",
        "dt1d_adapter": "dt1d",
        "dt1d-adapter": "dt1d",
        "dt1d-ablation": "dt1d_ablation",
        "bam_adapter": "bam",
        "bam-tuning": "bam",
        "bam_tuning": "bam",
        "residual_adapter": "residual",
        "residual_adapters": "residual",
        "ra": "residual",
        "side-tuning": "sidetune",
        "sidetuning": "sidetune",
        "side_tune": "sidetune",
        "plain-axial": "plain_axial",
        "axial-dw": "plain_axial",
        "reviewer-routing": "reviewer_routing",
        "routing_ablation": "reviewer_routing",
        "lora": "lora_conv",
        "lora-conv": "lora_conv",
    }
    tm = alias.get(str(tuning_method), str(tuning_method))

    if tm in ("conv_adapt", "conv_adapt_norm", "conv_adapt_bias"):
        return {
            "method": tm,
            "kernel_size": getattr(args, "kernel_size", 3),
            "adapt_size": getattr(args, "adapt_size", 8),
            "adapt_scale": getattr(args, "adapt_scale", 1.0),
        }

    if tm == "prompt":
        return {"method": tm, "prompt_size": getattr(args, "prompt_size", 10)}

    if tm in ("full", "linear", "norm", "repnet", "repnet_bias", "bias", "bitfit"):
        return {"method": tm}

    if tm == "dt1d":
        return {
            "method": "dt1d",
            "architecture": "R124-P2-G16-Axis-LearnedGate",
            "cache_kernel": getattr(args, "dt_cache_kernel", False),
        }


    if tm == "dt1d_ablation":
        return {
            "method": "dt1d_ablation",
            "reviewer_control": True,
            "axis": getattr(args, "dt_axis", "hw"),
            "alpha_group": getattr(args, "dt_alpha_group", 16),
            "active_offsets": getattr(args, "ablation_active_offsets", "1,2,4"),
            "shift_p": getattr(args, "ablation_shift_p", 2),
            "shift_lambda_mode": getattr(args, "ablation_shift_lambda_mode", "learned"),
            "shift_lambda_scope": getattr(args, "ablation_shift_lambda_scope", "axis"),
            "gate_mode": getattr(args, "ablation_gate_mode", "learned"),
            "use_pointwise": getattr(args, "dt_use_pointwise", False),
        }

    if tm == "plain_axial":
        return {
            "method": "plain_axial",
            "axis": getattr(args, "dt_axis", "hw"),
            "kernel_mode": getattr(args, "axial_kernel_mode", "unrestricted"),
            "padding_mode": getattr(args, "dt_padding", "replicate"),
        }

    if tm == "reviewer_routing":
        return {
            "method": "reviewer_routing",
            "axis": getattr(args, "dt_axis", "hw"),
            "order": getattr(args, "reviewer_order", 2),
            "dilations": getattr(args, "reviewer_dilations", "1,2,4"),
            "group_size": getattr(args, "reviewer_group_size", 16),
            "shifted": getattr(args, "reviewer_shifted", True),
            "routing": getattr(args, "reviewer_routing", "learned_softmax"),
            "temperature": getattr(args, "reviewer_temperature", 1.0),
        }

    if tm == "bam":
        return {
            "method": "bam",
            "reduction": getattr(args, "bam_reduction", 16),
            "dilation": getattr(args, "bam_dilation", 4),
            "gate_init": getattr(args, "bam_gate_init", 0.0),
            "use_bn": getattr(args, "bam_use_bn", True),
            "insert": getattr(args, "bam_insert", "stage"),
            "stages": getattr(args, "bam_stages", "1,2,3,4"),
        }

    if tm == "residual":
        return {
            "method": "residual",
            "mode": getattr(args, "ra_mode", "parallel"),
            "reduction": getattr(args, "ra_reduction", 16),
            "norm": getattr(args, "ra_norm", "bn"),
            "act": getattr(args, "ra_act", "relu"),
            "gate_init": getattr(args, "ra_gate_init", 0.0),
            "stages": getattr(args, "ra_stages", "1,2,3,4"),
        }

    if tm == "sidetune":
        return {
            "method": "sidetune",
            "alpha": getattr(args, "sidetune_alpha", 0.5),
            "learn_alpha": getattr(args, "sidetune_learn_alpha", True),
            "side_width": getattr(args, "sidetune_width", 64),
            "side_depth": getattr(args, "sidetune_depth", 3),
        }

    if tm == "ssf":
        return {
            "method": "ssf",
            "init_scale": getattr(args, "ssf_init_scale", 1.0),
            "init_shift": getattr(args, "ssf_init_shift", 0.0),
        }

    if tm == "lora_conv":
        return {
            "method": "lora_conv",
            "r": getattr(args, "lora_r", 4),
            "alpha": getattr(args, "lora_alpha", 1.0),
            "target": getattr(args, "lora_target", "all"),
        }

    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")
