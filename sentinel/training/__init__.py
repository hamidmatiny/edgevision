"""Commercial-safe synthetic training data utilities (Phase 2)."""

from sentinel.training.augmentations import LOW_LIGHT_RECIPES, apply_low_light_recipe
from sentinel.training.intel_sources import INTEL_TRAINING_VIDEOS, INTEL_VIDEO_LICENSE

__all__ = [
    "INTEL_TRAINING_VIDEOS",
    "INTEL_VIDEO_LICENSE",
    "LOW_LIGHT_RECIPES",
    "apply_low_light_recipe",
]
