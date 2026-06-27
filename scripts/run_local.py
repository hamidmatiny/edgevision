"""
run_local.py — CLI entrypoint for SENTINEL Phase 1

Usage:
    python scripts/run_local.py --source <path_or_rtsp_url> --camera-id cam1

Examples:
    # Local video file (dev/testing):
    python scripts/run_local.py --source /path/to/test_video.mp4 --camera-id cam1

    # RTSP stream:
    python scripts/run_local.py --source rtsp://192.168.1.100:554/stream --camera-id cam1

    # Override frame skip and model:
    python scripts/run_local.py --source test.mp4 --camera-id cam1 --frame-skip 2 --model yolo11s.pt

    # Limit to 200 frames (useful for quick smoke tests):
    python scripts/run_local.py --source test.mp4 --camera-id cam1 --max-frames 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.pipeline import CameraPipeline


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTINEL — Edge Security Analytics (Phase 1 local runner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", required=True,
        help="RTSP URL or local video file path",
    )
    parser.add_argument(
        "--camera-id", default="cam1",
        help="Camera identifier (must match a key in config/zones.yaml)",
    )
    parser.add_argument(
        "--zones-config", default="config/zones.yaml",
        help="Path to zones.yaml (default: config/zones.yaml)",
    )
    parser.add_argument(
        "--evidence-dir", default="evidence",
        help="Directory to write event records and clips (default: evidence/)",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=3,
        help="Process every Nth frame (default: 3)",
    )
    parser.add_argument(
        "--model", default="yolo11n.pt",
        help="YOLO model name or path (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4,
        help="Detection confidence threshold 0..1 (default: 0.4)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Inference device: cpu, cuda, cuda:0, mps (default: cpu)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after N processed frames (useful for smoke tests)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("sentinel.run_local")
    logger.info("Starting SENTINEL pipeline")
    logger.info("  source     : %s", args.source)
    logger.info("  camera_id  : %s", args.camera_id)
    logger.info("  zones      : %s", args.zones_config)
    logger.info("  evidence   : %s", args.evidence_dir)
    logger.info("  frame_skip : %d", args.frame_skip)
    logger.info("  model      : %s", args.model)
    logger.info("  device     : %s", args.device)

    pipeline = CameraPipeline(
        camera_id=args.camera_id,
        source=args.source,
        zones_yaml=args.zones_config,
        evidence_dir=args.evidence_dir,
        frame_skip=args.frame_skip,
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
