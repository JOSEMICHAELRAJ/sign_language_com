import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

IMG_SIZE = 28
NUM_CLASSES = 26
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN on the Sign Language MNIST CSV dataset and save the model."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing sign_mnist_train.csv and optionally sign_mnist_test.csv",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts",
        help="Directory where model/checkpoints/plots are saved",
    )
    return parser.parse_args()


def set_reproducibility(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_csv(data_dir: Path, preferred_names: list[str]) -> Optional[Path]:
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return None
    return csv_files[0]


def load_sign_mnist_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {csv_path}")

    labels = df["label"].to_numpy(dtype=np.int32)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    if pixels.shape[1] != IMG_SIZE * IMG_SIZE:
        raise ValueError(
            f"Expected {IMG_SIZE * IMG_SIZE} pixel columns, found {pixels.shape[1]} in {csv_path}"
        )

    images = pixels.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return images, labels


def build_model() -> tf.keras.Model:
    # Keep augmentation inside the model so the saved graph reflects training-time robustness.
    data_aug = models.Sequential(
        [
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.10, 0.10),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = data_aug(inputs)

    # Three convolutional blocks requested for progressively richer feature extraction.
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="sign_language_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, out_path: Path) -> None:
    hist = history.history
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("accuracy", []), label="train")
    plt.plot(hist.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_reproducibility()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = find_csv(data_dir, ["sign_mnist_train.csv", "train.csv"])
    if train_csv is None:
        raise FileNotFoundError(
            f"No training CSV found in {data_dir}. Expected sign_mnist_train.csv or train.csv."
        )

    print(f"Loading training data from: {train_csv}")
    images, labels = load_sign_mnist_csv(train_csv)

    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.15,
        random_state=SEED,
        stratify=labels,
    )

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)

    model = build_model()
    model.summary()

    checkpoint_path = out_dir / "best_model.keras"
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=1,
    )

    final_model_path = out_dir / "sign_language_cnn.keras"
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    curve_path = out_dir / "training_curves.png"
    plot_history(history, curve_path)
    print(f"Saved training curves to: {curve_path}")

    test_csv = find_csv(data_dir, ["sign_mnist_test.csv", "test.csv"])
    if test_csv is not None and test_csv != train_csv:
        print(f"Evaluating held-out test data from: {test_csv}")
        x_test, y_test = load_sign_mnist_csv(test_csv)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
        test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
    else:
        print("No separate test CSV found. Skipping held-out test evaluation.")


if __name__ == "__main__":
    main()
