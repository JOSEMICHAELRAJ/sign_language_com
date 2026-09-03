import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 28
LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime sign-letter recognition from webcam with temporal smoothing."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Webcam index (default: 0)"
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=8,
        help="Rolling prediction window for smoothing",
    )
    parser.add_argument(
        "--hold_frames",
        type=int,
        default=20,
        help="Consecutive stable frames required before appending a letter",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.65,
        help="Minimum confidence for stable detections",
    )
    return parser.parse_args()


def preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    # Keep preprocessing aligned with model training inputs.
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=(0, -1))


def get_center_roi(frame_shape: tuple[int, int, int], box_size: int = 220) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2
    half = box_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions/index.")

    prob_window: deque[np.ndarray] = deque(maxlen=max(1, args.smooth_window))
    message = ""

    last_stable_letter = ""
    stable_count = 0
    appended_for_current_hold = False

    print("Controls: 'q' quit, 'c' clear message")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        x1, y1, x2, y2 = get_center_roi(frame.shape)
        roi = frame[y1:y2, x1:x2]

        inp = preprocess_roi(roi)
        probs = model.predict(inp, verbose=0)[0]
        prob_window.append(probs)

        # Average recent probability vectors to suppress single-frame noise spikes.
        smooth_probs = np.mean(np.stack(prob_window, axis=0), axis=0)
        pred_idx = int(np.argmax(smooth_probs))
        pred_conf = float(smooth_probs[pred_idx])
        pred_letter = LETTERS[pred_idx] if pred_idx < len(LETTERS) else "?"

        if pred_conf >= args.conf_threshold:
            if pred_letter == last_stable_letter:
                stable_count += 1
            else:
                last_stable_letter = pred_letter
                stable_count = 1
                appended_for_current_hold = False

            # Append once per hold event to avoid repeated characters while user keeps same sign.
            if stable_count >= args.hold_frames and not appended_for_current_hold:
                message += pred_letter
                appended_for_current_hold = True
        else:
            last_stable_letter = ""
            stable_count = 0
            appended_for_current_hold = False

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Prediction: {pred_letter} ({pred_conf * 100:.1f}%)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Stable Frames: {stable_count}/{args.hold_frames}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display_message = message[-40:] if message else ""
        cv2.putText(
            frame,
            f"Message: {display_message}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign Language Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            message = ""
            stable_count = 0
            last_stable_letter = ""
            appended_for_current_hold = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
