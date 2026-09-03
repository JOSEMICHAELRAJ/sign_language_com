import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 28
LETTERS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k sign-letter predictions for a single image file."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    return parser.parse_args()


def preprocess_image(image_path: Path) -> np.ndarray:
    # Match training input format: grayscale 28x28 with [0,1] normalization.
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=(0, -1))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = tf.keras.models.load_model(model_path)
    inp = preprocess_image(image_path)

    probs = model.predict(inp, verbose=0)[0]
    topk = max(1, min(args.topk, len(probs)))
    top_indices = np.argsort(probs)[::-1][:topk]

    print(f"Top-{topk} predictions for {image_path}:")
    for rank, idx in enumerate(top_indices, start=1):
        letter = LETTERS[idx] if idx < len(LETTERS) else f"class_{idx}"
        conf = probs[idx] * 100.0
        print(f"{rank}. {letter}: {conf:.2f}%")


if __name__ == "__main__":
    main()
