"""
Standard object-detection evaluation metrics for a single class.

Uses COCO-style greedy matching at a fixed IoU threshold (default 0.5) and
reports precision, recall, F1, and AP@0.5 (average precision at IoU=0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class GroundTruthRecord:
    image_id: str
    class_name: str
    bbox: Box


@dataclass(frozen=True)
class DetectionRecord:
    image_id: str
    class_name: str
    bbox: Box
    confidence: float


@dataclass(frozen=True)
class ImageEvalResult:
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class DatasetEvalResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    ap50: float
    num_images: int
    num_ground_truths: int
    num_predictions: int


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height


def pairwise_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = box_area(box_a) + box_area(box_b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    precision_den = true_positives + false_positives
    recall_den = true_positives + false_negatives

    precision = true_positives / precision_den if precision_den else 0.0
    recall = true_positives / recall_den if recall_den else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_image(
    ground_truths: Sequence[GroundTruthRecord],
    detections: Sequence[DetectionRecord],
    iou_threshold: float = 0.5,
) -> ImageEvalResult:
    """
    Greedy one-to-one matching at fixed IoU threshold (confidence order).
    """
    unmatched_gt = list(range(len(ground_truths)))
    tp = 0
    fp = 0

    sorted_preds = sorted(detections, key=lambda d: d.confidence, reverse=True)
    for pred in sorted_preds:
        best_idx = None
        best_iou = 0.0
        for gt_idx in unmatched_gt:
            iou = pairwise_iou(pred.bbox, ground_truths[gt_idx].bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            unmatched_gt.remove(best_idx)
        else:
            fp += 1

    fn = len(unmatched_gt)
    return ImageEvalResult(true_positives=tp, false_positives=fp, false_negatives=fn)


def compute_ap50(
    ground_truths: Sequence[GroundTruthRecord],
    detections: Sequence[DetectionRecord],
    iou_threshold: float = 0.5,
) -> float:
    """
    Average precision at IoU=0.5 using the 101-point interpolated PR curve
    (PASCAL VOC / COCO-style sampling over recall in [0, 1]).
    """
    if not ground_truths:
        return 0.0

    preds_by_image: dict[str, list[DetectionRecord]] = {}
    gts_by_image: dict[str, list[GroundTruthRecord]] = {}

    for gt in ground_truths:
        gts_by_image.setdefault(gt.image_id, []).append(gt)
    for det in detections:
        preds_by_image.setdefault(det.image_id, []).append(det)

    all_preds = sorted(detections, key=lambda d: d.confidence, reverse=True)
    total_gt = len(ground_truths)
    tp = 0
    fp = 0

    precisions: list[float] = []
    recalls: list[float] = []

    matched_gt: dict[str, set[int]] = {image_id: set() for image_id in gts_by_image}

    for pred in all_preds:
        image_gts = gts_by_image.get(pred.image_id, [])
        best_iou = 0.0
        best_gt_idx = None
        for gt_idx, gt in enumerate(image_gts):
            if gt_idx in matched_gt[pred.image_id]:
                continue
            iou = pairwise_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt[pred.image_id].add(best_gt_idx)
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_gt
        precisions.append(precision)
        recalls.append(recall)

    # 101-point interpolated AP
    ap = 0.0
    for recall_threshold in [i / 100 for i in range(101)]:
        max_precision = 0.0
        for precision, recall in zip(precisions, recalls):
            if recall >= recall_threshold:
                max_precision = max(max_precision, precision)
        ap += max_precision
    return ap / 101


def evaluate_dataset(
    ground_truths: Sequence[GroundTruthRecord],
    detections: Sequence[DetectionRecord],
    iou_threshold: float = 0.5,
) -> DatasetEvalResult:
    gts_by_image: dict[str, list[GroundTruthRecord]] = {}
    dets_by_image: dict[str, list[DetectionRecord]] = {}

    for gt in ground_truths:
        gts_by_image.setdefault(gt.image_id, []).append(gt)
    for det in detections:
        dets_by_image.setdefault(det.image_id, []).append(det)

    image_ids = sorted(set(gts_by_image) | set(dets_by_image))
    tp = fp = fn = 0
    for image_id in image_ids:
        result = evaluate_image(
            gts_by_image.get(image_id, []),
            dets_by_image.get(image_id, []),
            iou_threshold=iou_threshold,
        )
        tp += result.true_positives
        fp += result.false_positives
        fn += result.false_negatives

    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    ap50 = compute_ap50(ground_truths, detections, iou_threshold=iou_threshold)

    return DatasetEvalResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        ap50=ap50,
        num_images=len(image_ids),
        num_ground_truths=len(ground_truths),
        num_predictions=len(detections),
    )
