# Task 11 — Proctoring Hardening (Start)

## Step-by-step setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install opencv-python==4.10.0.84
```

### 3. Run the proctoring script
```bash
python src/proctor.py
```
- Opens webcam, shows FACE OK / NO FACE live.
- Press `q` to quit.
- On exit, prints a session summary (total frames, no-face %, confirmed violations).
- Confirmed violations are logged to `violations_log.csv`.

### 4. Run the debug tool (optional)
```bash
python src/debug_face.py
```
Shows raw face detection with a bounding box, useful for checking camera/lighting quality.

## What "hardening" / "false-positive reduction" means here
Instead of flagging a violation the instant one frame misses a face (which
happens constantly from blinks, glare, or head turns), the script:
- Keeps a rolling window of the last 15 frames.
- Only flags a violation if a majority of frames in that window show no face.
- Adds a cooldown so one sustained absence doesn't spam multiple violation logs.

## Sample result
762 frames processed, 24.3% of individual frames had no face detected (raw
blinks/turns), but only 4 confirmed violations were logged — showing the
windowed voting filters out short-term noise while still catching genuine
sustained absence.

## Next steps
- Swap Haar cascade for a more robust detector (e.g. MediaPipe) for lighting/angle robustness.
- Add face-identity continuity checks (not just presence).
- Tie violation logs to a session/offer ID instead of a flat CSV.
- Formally tune window size / vote threshold against test recordings.

## Project structure
```
proctoring-trust-layer/
├── README.md
├── TASK11_SUMMARY.md
├── requirements.txt
└── src/
    ├── proctor.py
    └── debug_face.py
```