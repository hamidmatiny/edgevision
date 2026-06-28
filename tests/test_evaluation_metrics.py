"""Unit tests for detection evaluation metrics (Phase 2 Step 1)."""

import pytest

from sentinel.evaluation.metrics import (
    DetectionRecord,
    GroundTruthRecord,
    compute_ap50,
    compute_precision_recall_f1,
    evaluate_image,
    pairwise_iou,
)


def _gt(image_id: str, bbox, class_name: str = "person") -> GroundTruthRecord:
    return GroundTruthRecord(image_id=image_id, class_name=class_name, bbox=bbox)


def _det(image_id: str, bbox, confidence: float, class_name: str = "person") -> DetectionRecord:
    return DetectionRecord(
        image_id=image_id,
        class_name=class_name,
        bbox=bbox,
        confidence=confidence,
    )


class TestIoU:
    def test_identical_boxes_have_iou_one(self):
        box = (0.0, 0.0, 10.0, 10.0)
        assert pairwise_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_have_iou_zero(self):
        assert pairwise_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap_hand_computed(self):
        # A=[0,0,10,10] area=100; B=[5,0,15,10] overlap=[5,0,10,10] area=50
        # union=100+100-50=150 -> IoU=50/150=1/3
        iou = pairwise_iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert iou == pytest.approx(1 / 3, rel=1e-6)


class TestImageMatching:
    def test_perfect_match_counts_tp(self):
        gt = [_gt("img1", (0, 0, 100, 100))]
        det = [_det("img1", (0, 0, 100, 100), confidence=0.95)]
        result = evaluate_image(gt, det, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0
        p, r, f1 = compute_precision_recall_f1(
            result.true_positives, result.false_positives, result.false_negatives
        )
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_low_iou_counts_fp_and_fn(self):
        gt = [_gt("img1", (0, 0, 100, 100))]
        det = [_det("img1", (80, 80, 180, 180), confidence=0.9)]
        result = evaluate_image(gt, det, iou_threshold=0.5)
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_two_predictions_one_gt_one_tp_one_fp(self):
        gt = [_gt("img1", (0, 0, 100, 100))]
        dets = [
            _det("img1", (0, 0, 100, 100), confidence=0.9),
            _det("img1", (200, 200, 300, 300), confidence=0.8),
        ]
        result = evaluate_image(gt, dets, iou_threshold=0.5)
        assert result.true_positives == 1
        assert result.false_positives == 1
        assert result.false_negatives == 0
        p, r, _ = compute_precision_recall_f1(
            result.true_positives, result.false_positives, result.false_negatives
        )
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(1.0)


class TestAP50:
    def test_single_gt_perfect_top_prediction_has_ap_one(self):
        gt = [_gt("img1", (0, 0, 50, 50))]
        dets = [
            _det("img1", (0, 0, 50, 50), confidence=0.95),
            _det("img1", (100, 100, 150, 150), confidence=0.10),
        ]
        ap = compute_ap50(gt, dets, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_hand_computed_two_step_pr_curve(self):
        """
        2 GT boxes (different images), 2 predictions:
          - high-conf pred matches img1 -> TP
          - low-conf pred matches img2  -> TP
        AP@0.5 should be 1.0.
        """
        gts = [
            _gt("img1", (0, 0, 10, 10)),
            _gt("img2", (0, 0, 10, 10)),
        ]
        dets = [
            _det("img1", (0, 0, 10, 10), confidence=0.9),
            _det("img2", (0, 0, 10, 10), confidence=0.4),
        ]
        ap = compute_ap50(gts, dets, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_no_predictions_yields_zero_ap(self):
        gt = [_gt("img1", (0, 0, 10, 10))]
        assert compute_ap50(gt, [], iou_threshold=0.5) == 0.0
