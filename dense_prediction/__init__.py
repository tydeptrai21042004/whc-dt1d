"""Dense-prediction support for DT1D-Adapter.

The package contains the segmentation and detection pipelines reported in the
manuscript submitted to *The Visual Computer*.
"""

from .datasets import build_dense_datasets, detection_collate_fn
from .models import build_dense_model, configure_dense_trainability

__all__ = [
    "build_dense_datasets",
    "detection_collate_fn",
    "build_dense_model",
    "configure_dense_trainability",
]
