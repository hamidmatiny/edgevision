"""
Low-light Albumentations recipes for Phase 2 synthetic training data.

Labels are applied on bright source frames, then bbox coordinates are
transformed alongside the image. Does NOT use Ultralytics sample images.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 pixel coords

BBOX_PARAMS = A.BboxParams(
    format="pascal_voc",
    label_fields=["category_ids"],
    min_visibility=0.3,
)


def _recipe(name: str, transforms: list[A.BasicTransform]) -> tuple[str, A.Compose]:
    return name, A.Compose(transforms, bbox_params=BBOX_PARAMS)


# Four distinct low-light profiles (~4× expansion per labeled parent frame).
LOW_LIGHT_RECIPES: tuple[tuple[str, A.Compose], ...] = (
    _recipe(
        "underexposed_gamma",
        [
            A.RandomGamma(gamma_limit=(15, 45), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.55, -0.25),
                contrast_limit=(-0.35, -0.05),
                p=1.0,
            ),
            A.GaussNoise(std_range=(0.05, 0.2), p=0.8),
        ],
    ),
    _recipe(
        "cool_cast_noise",
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, -0.2),
                contrast_limit=(-0.4, -0.1),
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-8, 8),
                sat_shift_limit=(-40, -10),
                val_shift_limit=(-50, -20),
                p=1.0,
            ),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.9),
        ],
    ),
    _recipe(
        "warm_glare_compression",
        [
            A.RandomGamma(gamma_limit=(20, 55), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=(5, 20),
                sat_shift_limit=(-20, 10),
                val_shift_limit=(-45, -15),
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.45, -0.15),
                contrast_limit=(-0.5, -0.15),
                p=1.0,
            ),
            A.GaussNoise(std_range=(0.03, 0.15), p=0.6),
        ],
    ),
    _recipe(
        "motion_blur_dark",
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, -0.3),
                contrast_limit=(-0.3, 0.0),
                p=1.0,
            ),
            A.MotionBlur(blur_limit=(5, 11), p=0.7),
            A.GaussNoise(std_range=(0.08, 0.25), p=0.8),
            A.RandomGamma(gamma_limit=(25, 60), p=0.8),
        ],
    ),
)


def boxes_to_albumentations(
    boxes: list[Box],
    class_ids: list[int] | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Convert (x1,y1,x2,y2) boxes to Albumentations pascal_voc format."""
    if class_ids is None:
        class_ids = [0] * len(boxes)
    alb_boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes]
    return alb_boxes, class_ids


def boxes_from_albumentations(alb_boxes: list[list[float]]) -> list[Box]:
    return [tuple(map(float, b)) for b in alb_boxes]


def apply_low_light_recipe(
    image_bgr: np.ndarray,
    boxes: list[Box],
    recipe_name: str,
) -> tuple[np.ndarray, list[Box]] | None:
    """
    Apply a named low-light recipe, returning augmented image + transformed boxes.

    Returns None if all boxes are filtered out (visibility too low after transform).
    """
    recipe_map = dict(LOW_LIGHT_RECIPES)
    if recipe_name not in recipe_map:
        raise KeyError(f"Unknown recipe: {recipe_name}. Available: {list(recipe_map)}")

    compose = recipe_map[recipe_name]
    alb_boxes, category_ids = boxes_to_albumentations(boxes)
    if not alb_boxes:
        return None

    result: dict[str, Any] = compose(
        image=image_bgr,
        bboxes=alb_boxes,
        category_ids=category_ids,
    )
    out_boxes = boxes_from_albumentations(result["bboxes"])
    if not out_boxes:
        return None
    return result["image"], out_boxes


def yolo_label_line(box: Box, img_w: int, img_h: int, class_id: int = 0) -> str:
    """YOLO normalized label: class cx cy w h."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return ""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def write_yolo_label(path: Path, boxes: list[Box], img_w: int, img_h: int) -> bool:
    from pathlib import Path as PathType

    path = PathType(path)
    lines = [
        line
        for box in boxes
        if (line := yolo_label_line(box, img_w, img_h))
    ]
    if not lines:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
