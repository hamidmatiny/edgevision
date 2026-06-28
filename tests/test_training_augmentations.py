"""Tests for Phase 2 low-light augmentation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.training.augmentations import (
    LOW_LIGHT_RECIPES,
    apply_low_light_recipe,
    yolo_label_line,
)


def test_low_light_recipes_preserve_at_least_one_box():
    rng = np.random.default_rng(42)
    image = rng.integers(40, 220, size=(432, 768, 3), dtype=np.uint8)
    boxes = [(100.0, 80.0, 220.0, 380.0), (400.0, 90.0, 520.0, 390.0)]

    for recipe_name, _ in LOW_LIGHT_RECIPES:
        result = apply_low_light_recipe(image, boxes, recipe_name)
        assert result is not None, f"{recipe_name} dropped all boxes"
        aug_img, aug_boxes = result
        assert aug_img.shape == image.shape
        assert len(aug_boxes) >= 1
        for box in aug_boxes:
            x1, y1, x2, y2 = box
            assert x2 > x1
            assert y2 > y1


def test_yolo_label_line_normalized():
    line = yolo_label_line((0.0, 0.0, 100.0, 200.0), img_w=200, img_h=400)
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, w, h = map(float, parts[1:])
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0


def test_unknown_recipe_raises():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(KeyError):
        apply_low_light_recipe(image, [(10, 10, 50, 80)], "nonexistent_recipe")
