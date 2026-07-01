#!/usr/bin/env python3
"""
Phase 3.1 — VLM latency spike on local candidate crops.

Measures wall-clock ms per yes/no verification query on sample images
(reuses ExDark FP example exports — eval-only, not training).

Usage:
    python scripts/spike_vlm_latency.py
    python scripts/spike_vlm_latency.py --model moondream2 --devices cpu mps
    python scripts/spike_vlm_latency.py --model qwen2-vl-2b --devices cpu
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = sorted((REPO_ROOT / "evaluation" / "fp_examples").glob("fp_*.jpg"))

VERIFY_PROMPT = (
    "Is there a clearly visible real person in this image? "
    "Answer YES or NO only."
)

MAX_CROP_PX = 512
MOONDREAM_MAX_TOKENS = 16
MOONDREAM_SETTINGS = {
    "max_tokens": MOONDREAM_MAX_TOKENS,
    "temperature": 0.1,
    "top_p": 0.3,
    "variant": None,
}


@dataclass
class LatencyResult:
    model: str
    device: str
    n_runs: int
    warmup_runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    load_ms: float
    sample_answers: list[str]


def load_images(paths: list[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        scale = min(1.0, MAX_CROP_PX / max(h, w))
        if scale < 1.0:
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        images.append(img)
    return images


def bgr_to_pil(bgr: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_moondream(device: str):
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device in ("mps", "cuda") else torch.float32,
    )
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
        device = "cpu"
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    return model, device, load_ms


def infer_moondream(model, pil_image, device: str) -> tuple[str, float]:
    import torch

    t0 = time.perf_counter()
    with torch.inference_mode():
        result = model.query(
            pil_image,
            VERIFY_PROMPT,
            settings=MOONDREAM_SETTINGS,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    answer = result.get("answer", str(result))
    return answer.strip()[:200], elapsed_ms


def load_qwen2_vl(device: str):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    t0 = time.perf_counter()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
        device = "cpu"
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    return model, processor, device, load_ms


def infer_qwen2_vl(model, processor, pil_image, device: str) -> tuple[str, float]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": VERIFY_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return answer.strip()[:200], elapsed_ms


def run_spike(
    model_name: str,
    device: str,
    images_bgr: list[np.ndarray],
    warmup: int = 2,
    runs_per_image: int = 3,
) -> LatencyResult:
    pil_images = [bgr_to_pil(img) for img in images_bgr]
    latencies: list[float] = []
    answers: list[str] = []

    if model_name == "moondream2":
        model, device, load_ms = load_moondream(device)
        infer_fn = lambda pil: infer_moondream(model, pil, device)
    elif model_name == "qwen2-vl-2b":
        model, processor, device, load_ms = load_qwen2_vl(device)
        infer_fn = lambda pil: infer_qwen2_vl(model, processor, pil, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Warmup
    for i in range(warmup):
        infer_fn(pil_images[i % len(pil_images)])

    # Timed runs — cycle through sample crops
    total_runs = runs_per_image * len(pil_images)
    for i in range(total_runs):
        answer, ms = infer_fn(pil_images[i % len(pil_images)])
        latencies.append(ms)
        if i < 3:
            answers.append(answer)

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]

    return LatencyResult(
        model=model_name,
        device=device,
        n_runs=total_runs,
        warmup_runs=warmup,
        mean_ms=statistics.mean(latencies),
        p50_ms=p50,
        p95_ms=p95,
        min_ms=min(latencies),
        max_ms=max(latencies),
        load_ms=load_ms,
        sample_answers=answers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["moondream2", "qwen2-vl-2b", "all"],
        default="all",
    )
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs-per-image", type=int, default=3)
    args = parser.parse_args()

    if not args.images:
        print("No sample images found. Expected evaluation/fp_examples/fp_*.jpg")
        sys.exit(1)

    images = load_images(args.images)
    models = ["moondream2", "qwen2-vl-2b"] if args.model == "all" else [args.model]

    import torch

    print("Phase 3.1 VLM latency spike")
    print(f"  Sample crops: {len(images)} (ExDark FP exports — eval-only)")
    print(f"  Prompt: {VERIFY_PROMPT[:80]}...")
    print(f"  SPEC budget: ~200–500 ms per candidate\n")

    results: list[LatencyResult] = []
    for model_name in models:
        for device in args.devices:
            if device == "mps" and not torch.backends.mps.is_available():
                print(f"[skip] {model_name} on mps — MPS not available")
                continue
            print(f"Running {model_name} on {device} ...")
            try:
                result = run_spike(
                    model_name=model_name,
                    device=device,
                    images_bgr=images,
                    warmup=args.warmup,
                    runs_per_image=args.runs_per_image,
                )
                results.append(result)
                print(
                    f"  load={result.load_ms:.0f}ms  "
                    f"mean={result.mean_ms:.0f}ms  p50={result.p50_ms:.0f}ms  "
                    f"p95={result.p95_ms:.0f}ms  (n={result.n_runs})"
                )
                for i, ans in enumerate(result.sample_answers, 1):
                    print(f"  sample[{i}]: {ans[:120]}")
                print()
            except Exception as exc:
                print(f"  FAILED: {exc}\n")

    if not results:
        sys.exit(1)

    out_path = REPO_ROOT / "evaluation" / "vlm_latency_spike.json"
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
