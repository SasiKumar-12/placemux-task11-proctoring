"""
Proctoring Hardening - Task 11
AI-ML Trust Layer (Offers & E-Sign)

Goal: detect whether a face is present/verified during an e-sign session,
while reducing false positives caused by single bad frames (blinks,
lighting flicker, brief head turns).

Approach: sliding-window majority vote instead of trusting one frame.
Violations are logged to a CSV file for evidence/demo purposes.
"""

import cv2
from collections import deque
import time
import csv
import os

# ---------- Config ----------
WINDOW_SIZE = 15          # number of recent frames to consider
VOTE_THRESHOLD = 0.6      # fraction of frames in window that must show a face
MIN_FACE_SIZE = (80, 80)  # ignore tiny/far-away detections (noise)
FLAG_COOLDOWN_SEC = 3.0   # don't re-flag the same violation repeatedly
LOG_FILE = "violations_log.csv"

# ---------- Setup ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

history = deque(maxlen=WINDOW_SIZE)  # 1 = face seen, 0 = not seen
last_flag_time = 0.0
total_frames = 0
frames_no_face = 0
total_violations = 0

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "event"])


def detect_face(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )
    return len(faces) > 0


def is_violation(history: deque) -> bool:
    """Majority vote over the window, not a single frame."""
    if len(history) < history.maxlen:
        return False  # not enough data yet, don't flag during warm-up
    seen_ratio = sum(history) / len(history)
    return seen_ratio < (1 - VOTE_THRESHOLD)


def main():
    global last_flag_time, total_frames, frames_no_face, total_violations

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permissions/index.")
        return

    print("Proctoring started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face_present = detect_face(frame)
        history.append(1 if face_present else 0)

        total_frames += 1
        if not face_present:
            frames_no_face += 1

        status = "FACE OK" if face_present else "NO FACE"
        color = (0, 200, 0) if face_present else (0, 0, 200)

        if is_violation(history) and (time.time() - last_flag_time) > FLAG_COOLDOWN_SEC:
            last_flag_time = time.time()
            total_violations += 1
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] VIOLATION: sustained face-absence over last {WINDOW_SIZE} frames")
            with open(LOG_FILE, "a", newline="") as f:
                csv.writer(f).writerow([ts, "face_absence_violation"])

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Proctoring - Task 11", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    no_face_rate = (frames_no_face / total_frames * 100) if total_frames else 0
    print("\n--- Session Summary ---")
    print(f"Total frames processed:      {total_frames}")
    print(f"Frames with no face:         {frames_no_face} ({no_face_rate:.1f}%)")
    print(f"Confirmed violations logged: {total_violations}")
    print(f"(A high 'no face' % but low violation count shows the ")
    print(f" windowed voting is filtering out short/blink-level noise.)")


if __name__ == "__main__":
    main()