"""Detection evaluation utilities for Phase 2+ model benchmarking."""

from sentinel.evaluation.metrics import (
    Box,
    DetectionRecord,
    GroundTruthRecord,
    compute_ap50,
    compute_precision_recall_f1,
    evaluate_image,
    evaluate_dataset,
    pairwise_iou,
)

__all__ = [
    "Box",
    "DetectionRecord",
    "GroundTruthRecord",
    "compute_ap50",
    "compute_precision_recall_f1",
    "evaluate_image",
    "evaluate_dataset",
    "pairwise_iou",
]
