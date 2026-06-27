# SENTINEL — Phase 0 Spec

## MVP Use Cases (v1 — 3 only)

| # | Use Case | Trigger | False-positive source being suppressed |
|---|---|---|---|
| 1 | **Perimeter intrusion** | Person/vehicle crosses defined boundary after hours | Wind, shadows, animals, headlights |
| 2 | **Loitering / unauthorized presence** | Person inside restricted zone > N seconds | Walk-through foot traffic |
| 3 | **PPE / safety-zone compliance** | Person without hard hat/vest in designated zone | Lighting variation, partial occlusions |

## Ideal Customer Profile (v1 pilot)

Mid-market industrial site (warehouse, construction yard, utility yard) with:
- 4–20 existing IP cameras (RTSP/ONVIF-compliant)
- No dedicated 24/7 SOC
- Currently relying on legacy motion-detection alerts or manual review
- Not willing to rip out existing camera hardware for an all-in-one contract

## Success Metrics (numerically defined)

| Metric | Target | Measurement method |
|---|---|---|
| False positive rate | **< 5%** (industry benchmark) | FP / (FP + TP) on labeled test set |
| Detection-to-alert latency | **< 2 seconds** | Timestamp delta: frame capture → event JSON written |
| Pipeline uptime | **> 99%** (24h unattended) | RTSP reconnect handling, graceful degradation test |
| True positive rate | **> 90%** on benchmark test set | Precision/recall on labeled true events |

## Non-Goals for v1 (explicitly out of scope)

- Facial recognition or identity matching
- Weapon detection
- Multi-site federated fleet management at scale
- Mobile app
- Billing / subscription infrastructure
- Cloud dashboard (Phase 5)

## Pilot Environments

- **Primary (dev):** Local video files from public anomaly datasets (UCF-Crime, ShanghaiTech) + self-recorded webcam footage
- **Secondary (target):** Live RTSP stream from a real industrial site (TBD — in progress)

## Latency Budget

```
Frame capture           →  YOLO inference:        ~50–150ms (CPU, INT8)
YOLO output             →  Zone engine eval:       <1ms
Candidate event         →  Event JSON written:     <5ms
Total (Stage 1+2+4):    →  Target <500ms end-to-end (CPU)
Stage 3 VLM (Phase 3):  →  Add ~200–500ms per candidate (runs on <5% of frames)
Total with VLM:         →  Target <2s end-to-end
```
