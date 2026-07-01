# Phase 3 — VLM Contextual Verifier

**Status:** Step 3.1 complete (model selection + license review + latency spike). **Stopped for your review before integration code.**

## Goal

Replace `VLMVerifierStub` in `sentinel/pipeline.py` with a real Stage 3 verifier that confirms or rejects zone-engine **candidates** before evidence clips are written. Primary target: **dark-scene hallucinations and shape confusions** identified in Phase 1/2 FP spot-checks — not re-solving the ExDark recall gap via detector fine-tuning.

Stock **`yolo11n.pt` remains the Stage 1 detector** (Phase 2 negative result).

---

## Step 3.1 — Model selection, license, latency (this document)

### Candidates reviewed (commercial license scrutiny)

Same standard as ExDark flag in Phase 2: state license explicitly; do not assume.

| Model | Hugging Face ID | License | Commercial deployment | Restrictions / notes |
|---|---|---|---|---|
| **Moondream 2** (recommended) | `vikhyatk/moondream2` @ revision **`2025-06-21`** | **Apache 2.0** | **Permitted** | Pin revision for production; attribution + NOTICE in `THIRD_PARTY_NOTICES.md`. Local weights only (no cloud API). |
| SmolVLM-256M-Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` | **Apache 2.0** | **Permitted** | Faster in spike (~607 ms p50 MPS) but **much smaller**; likely weaker on subtle dark-scene FP reasoning. Fallback if Moondream quality insufficient. |
| Qwen2-VL-2B-Instruct | `Qwen/Qwen2-VL-2B-Instruct` | **Apache 2.0** | **Permitted** | Alibaba copyright in LICENSE file. **Not recommended:** MPS OOM on test crops; heavier than needed. Larger Qwen variants add MAU thresholds (not relevant at our scale). |
| SmolVLM2-2.2B-Instruct | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | **Apache 2.0** | **Permitted** | Not spike-tested; likely slower than 256M. |
| Moondream 3 Preview | `moondream/moondream3-preview` | Check model card | **Defer** | Preview / moving target; transformers compat issues. |
| Llama 3.2 Vision | Meta | **Llama Community License** | Conditional | Custom license; **EU availability restrictions**; MAU thresholds at scale. Not chosen. |
| Ultralytics / COCO sample assets | — | Various | **Excluded** | Same policy as Phase 2 training. |
| ExDark images | — | **Non-commercial research only** | **Eval only** | Approved for verifier **evaluation** manifest only — never training. |

**Moondream cloud API** (`moondream` pip package → `api.moondream.ai`) is **not acceptable** for SENTINEL: footage must stay on-prem. Phase 3 uses **local HF weights only**.

**Integration dependency note:** Moondream2 custom HF code currently loads with **`transformers==4.56.1`** on this machine. Transformers 5.x fails with `'HfMoondream' object has no attribute 'all_tied_weights_keys'` until upstream Moondream merges transformers-5 compat. Phase 3 build will pin this explicitly in optional deps.

### Latency spike — your machine (Apple M3 Max)

Script: `scripts/spike_vlm_latency.py`  
Sample inputs: **5 crops** from `evaluation/fp_examples/` (ExDark FP exports — **eval-only**, not training)  
Prompt: yes/no person verification  
SPEC budget ([`SPEC.md`](SPEC.md)): **~200–500 ms per candidate**

| Model | Device | Config | p50 | p95 | vs SPEC |
|---|---|---|---:|---:|---|
| Moondream2 | MPS | default (~768 token cap) | **1287 ms** | 1789 ms | ❌ ~2.6× over max |
| Moondream2 | MPS | tuned: 512px crop, max_tokens=16 | **678 ms** | 1310 ms | ❌ ~1.4× over max |
| Moondream2 | MPS | tuned: 384px crop, max_tokens=8 | **671 ms** | 679 ms | ❌ ~1.3× over max (latency floor) |
| SmolVLM-256M | MPS | 384px crop, max_new_tokens=8 | **607 ms** | 611 ms | ❌ ~1.2× over max |
| Moondream2 | CPU | — | — | — | ❌ **Failed** (`Passed CPU tensor to MPS op`) |
| Qwen2-VL-2B | MPS | full-res crops | — | — | ❌ **Failed** (MPS buffer OOM) |

Machine-readable: [`evaluation/vlm_latency_spike.json`](evaluation/vlm_latency_spike.json)

**Honest read:** On your M3 Max, **no candidate hit the 200–500 ms budget** in eager PyTorch inference. Tuning (short yes/no answers, resized crops) brought Moondream2 from ~1.3 s down to a **~670 ms floor** — dominated by vision encoding, not token generation. This does **not** block Phase 3, but it **does** require a decision before integration:

1. **Accept ~650–700 ms Stage 3 latency** on this hardware class (still within SPEC **<2 s end-to-end** with Stage 1+2) and optimize later (ONNX, batch=1 compile, smaller crop), or
2. **Revise the Stage 3 latency target** in SPEC for edge CPU/MPS class hardware, or
3. **Pivot to SmolVLM-256M** for speed at quality risk, or
4. **Defer Phase 3** until a faster runtime path is validated.

**Recommendation for 3.2+:** Proceed with **Moondream2 @ `2025-06-21`**, local MPS/CPU, tuned yes/no settings — best balance of **Apache 2.0 certainty**, API fit (`model.query`), and verification quality potential. Re-spike ONNX/quantized path if you want sub-500 ms before integration.

### Proposed verification prompt (yes/no)

```
Is there a clearly visible real person in this image?
Answer YES or NO only.
```

Production will pass a **bbox crop with margin** (not full frame) plus structured context in the system portion of the prompt (zone name, dwell, detector confidence).

---

## Design decisions (approved direction — includes your corrections)

### 1. Rejected candidates are recorded (not silently dropped)

When Stage 3 rejects a candidate, the pipeline will write a **lightweight rejected-candidate record** — separate from confirmed incidents:

| | Confirmed incident | Rejected candidate |
|---|---|---|
| Path | `evidence/<camera>_<event_id>.json` (+ clip) | `evidence/rejected/<camera>_<event_id>.json` |
| Video clip | Yes (rolling buffer clip) | **No** (optional small thumbnail later) |
| Purpose | Operator alert + audit | FP suppression audit + Phase 7 learning |

**Proposed schema (`record_type: "rejected_candidate"`):**

```json
{
  "schema_version": "1.0",
  "record_type": "rejected_candidate",
  "event_id": "<uuid>",
  "camera_id": "cam1",
  "rejected_at_stage": "stage3_vlm",
  "rejected_at_iso": "2026-06-28T12:00:00+00:00",
  "candidate": {
    "zone_name": "restricted_storage",
    "track_id": 3,
    "detection_class": "person",
    "confidence": 0.88,
    "dwell_elapsed_seconds": 5.2,
    "centroid": [640.0, 240.0],
    "bbox": [500, 100, 700, 400]
  },
  "vlm_verification": {
    "confirmed": false,
    "model": "vikhyatk/moondream2@2025-06-21",
    "reasoning": "NO — dark vertical structure, not a person",
    "latency_ms": 671
  },
  "audit_chain": {
    "stage1_confidence": 0.88,
    "stage2_zone_rule": "zone='restricted_storage' dwell=5.20s",
    "stage3_vlm": "NO — dark vertical structure, not a person",
    "final_decision": "rejected"
  }
}
```

Confirmed incidents update the same `vlm_verification` / `audit_chain` fields with `"final_decision": "confirmed"`.

This enables: **(a)** measurable FP suppression counts, **(b)** compliance audit trail, **(c)** Phase 7 continuous learning without retrofit.

### 2. ExDark in verifier eval — eval-only (approved)

ExDark FP example crops may appear in the **verifier evaluation manifest** (same as detector benchmarking). They must **not** appear in any training or fine-tuning run.

---

## Planned evaluation harness (Step 3.4 — not built yet)

**Sample size (honest):** Initial harness will likely start with **on the order of 10–20 labeled candidate crops**, including:

- **5** ExDark FP exports already in `evaluation/fp_examples/` (eval-only)
- Additional TPs/FPs from `smoke_test_loiter.mp4` and Intel `test_video.mp4` runs (commercial-safe sources)

That is **too small for tight precision/recall confidence intervals**. `README_PHASE3.md` results will report **raw counts and point estimates only**, with explicit “small-n” caveats — same honesty standard as Phase 2 findings. Expanding the labeled set is a follow-up, not a reason to overstate Phase 3 metrics.

**Metrics:**

- Verifier-only: TP/FP/TN/FN on labeled yes/no person presence in crop
- End-to-end: Stage 1+2 vs Stage 1+2+3 candidate-to-confirmed rate on the same labeled set
- **FP suppression count** = rejected-candidate records where GT = no person

---

## Planned build steps (after your 3.1 approval)

| Step | Deliverable |
|---|---|
| 3.2 | `sentinel/verification/vlm_verifier.py` + config; replace stub |
| 3.3 | Rejected-candidate writer + confirmed incident schema wiring |
| 3.4 | Eval harness + small labeled manifest |
| 3.5 | Tests (mocked VLM in CI) + reproduce doc |

**Stop point:** You verify real verifier metrics and pipeline output before Phase 4.

---

## Reproduce Step 3.1 spike

```bash
pip install transformers==4.56.1 accelerate pillow   # Moondream2 HF compat pin
python scripts/spike_vlm_latency.py --model moondream2 --devices mps
cat evaluation/vlm_latency_spike.json
```
