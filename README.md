# Sign Language Image Classification for Assistive Emergency Communication

This project trains and runs a sign language image classifier based on the
Sign Language MNIST dataset. It is designed as a foundation for assistive
emergency communication workflows for deaf-blind users, where prediction
stability and confidence handling are safety-critical.

## Project Files

- `train_model.py`: Train a CNN model from Sign Language MNIST CSV files.
- `predict_image.py`: Predict top-k letters for one image.
- `realtime_webcam.py`: Run live webcam inference with temporal smoothing.
- `requirements.txt`: Python dependencies.

## 1. Environment Setup

From the workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download Dataset (Sign Language MNIST)

Dataset source (Kaggle):

- https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Download and place CSV files in a `data/` directory like this:

```text
sign_language_com/
	data/
		sign_mnist_train.csv
		sign_mnist_test.csv    # optional but recommended
```

CSV format expected:

- One `label` column (class index).
- 784 pixel columns for flattened 28x28 grayscale image values.

## 3. Train the Model

Example:

```powershell
python train_model.py --data_dir data --epochs 40 --batch_size 64 --out_dir artifacts
```

Outputs in `artifacts/`:

- `best_model.keras` (best checkpoint)
- `sign_language_cnn.keras` (final model)
- `training_curves.png` (accuracy/loss plot)

If `sign_mnist_test.csv` is present, the script also evaluates held-out test performance.

## 4. Predict a Single Image

Example:

```powershell
python predict_image.py --model artifacts/sign_language_cnn.keras --image sample.png --topk 3
```

Behavior:

- Converts image to grayscale.
- Resizes to 28x28.
- Normalizes pixels to [0,1].
- Prints top-k letter predictions with confidence percentages.

## 5. Run Realtime Webcam Inference

Example:

```powershell
python realtime_webcam.py --model artifacts/sign_language_cnn.keras
```

Optional tuning:

```powershell
python realtime_webcam.py --model artifacts/sign_language_cnn.keras --smooth_window 8 --hold_frames 20 --conf_threshold 0.65
```

Controls:

- Press `q` to quit.
- Press `c` to clear the spelled message.

Runtime behavior:

- Displays a fixed ROI box for hand placement.
- Smooths predictions over a rolling window (default 8 frames).
- Requires a stable letter over consecutive frames (default 20) before appending.

## 6. Limitations and Safety Notes

- This model targets static, single-frame handshapes only.
- Motion-based letters/gestures (including J and Z in ASL fingerspelling) are not handled.
- It does not recognize full ASL words, grammar, or context.
- In an emergency-alert pipeline, confidence thresholding and temporal smoothing are critical to reduce false triggers from noisy frames, motion blur, background clutter, and transient misclassifications.

This repository is suitable as a baseline component. A production-grade emergency system should include multi-sensor confirmation, user feedback loops, and human-in-the-loop escalation safeguards.