# Task 11 — Proctoring Hardening (Start)
**AI/ML Engineer · AI-ML Trust Layer (Offers & E-Sign) · Week 4, Phase 2**

## Objective
Begin hardening the proctoring component of the e-sign trust layer by reducing
false-positive violation flags — i.e., not penalizing a legitimate signer for
normal, momentary face-detection dropouts (blinks, brief head turns, minor
camera noise).

## What was built
A webcam-based face-presence proctoring module (`src/proctor.py`) using
OpenCV's Haar cascade face detector, with a sliding-window majority-vote
mechanism layered on top of raw per-frame detection.

- **Raw detection**: each frame is independently checked for a face.
- **Hardening layer**: the last 15 frames are kept in a rolling window. A
  violation is only flagged if a majority (≥60%) of frames in that window
  show no face — a single missed frame is not enough.
- **Cooldown**: once flagged, a 3-second cooldown prevents duplicate flags
  for the same sustained absence.
- **Logging**: every confirmed violation is timestamped and written to
  `violations_log.csv` for auditability.
- **Session summary**: on exit, the script reports total frames processed,
  raw no-face frame count/percentage, and confirmed violation count.

## Test results (sample session)
| Metric | Value |
|---|---|
| Total frames processed | 762 |
| Frames with no face detected (raw) | 185 (24.3%) |
| Confirmed violations logged (after hardening) | 4 |

**Interpretation:** Without the windowed voting, each of the 185 raw no-face
frames could have triggered a separate false alarm. With hardening applied,
only 4 sustained absences were flagged — filtering out blink- and
glance-level noise while still catching genuine, prolonged face-absence
events.

## Next steps
- Replace Haar cascade with a more robust detector (e.g. MediaPipe).
- Add identity continuity checks (face embedding match).
- Tie violation logs to a session/offer ID rather than a flat CSV.
- Formally tune window size / vote threshold against test recordings.