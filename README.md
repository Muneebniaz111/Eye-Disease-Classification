---
title: Eye Disease Classifier
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Eye Disease Classification (Retinal Fundus Images)

A Flask + TensorFlow app that trains a CNN on the `Fundus_diseases/`
image dataset and classifies new, unseen retinal fundus photographs into
one of four categories: **cataract**, **diabetic_retinopathy**,
**glaucoma**, or **normal**.

## Pipeline

```
Fundus_diseases/ dataset -> Data Preprocessing -> Model Training ->
Validation/Testing (held-out split) -> Saved Model (models/) ->
User uploads a new retinal image -> Validate it's actually a fundus
image -> Prediction -> Result
```

- **Dataset**: `Fundus_diseases/<class_name>/*.jpg|png` — one folder per
  class: `cataract`, `diabetic_retinopathy`, `glaucoma`, `normal`
  (~1,000-1,100 real fundus photographs each, ~4,200 total). Class names
  are discovered automatically from the folder names.
- **Training** (`train_model.py`): loads every image from
  `Fundus_diseases/`, splits it into train/validation/test (80/10/10,
  stratified), trains a CNN, and saves:
  - `models/best_cnn_model.keras` — the trained model
  - `models/class_names.json` — the exact label order the model outputs,
    so the app never has to guess or hardcode class names
  - `models/confusion_matrix.png` and `models/classification_report.txt`
    — evaluated on the held-out **test** split only (images the model
    never saw during training)
- **Inference** (`app1.py`): a Flask app that loads the saved model +
  `class_names.json`, validates the upload actually looks like a retinal
  fundus photo (see below), preprocesses it the same way training images
  were preprocessed, and returns the predicted class with per-class
  probabilities. The uploaded image is used only for prediction — it is
  never added to the training set.

## Web interface & PDF reports

The frontend (`templates/index.html`) is a single-page dark-themed UI:
drag-and-drop (or click-to-browse) image upload, live model-status
indicator, an animated scan effect while a prediction is running, and a
results view with a confidence score and a probability breakdown bar per
class.

After a successful screening, a **"Download PDF report"** button appears.
This calls a new `/generate_report` endpoint that formats the
already-computed prediction (condition, confidence, full probability
breakdown, and the uploaded image as a thumbnail) into a formatted PDF,
complete with a report ID, timestamp, and a medical disclaimer. It does
**not** re-run the model or re-validate the image -- it only turns
results the browser already has into a downloadable file, so it can't
affect prediction behavior. It requires the `reportlab` package (see
Setup above).

## Rejecting non-retinal images

`validate_retinal_image()` in `app1.py` runs a lightweight check before
any image reaches the model: minimum resolution, aspect ratio, color
richness, edge/texture detail, and brightness range, calibrated against
this dataset's actual fundus photos (see the comments in that function
for the exact thresholds and how they were derived). If an upload fails
this check — e.g. a screenshot, selfie, document scan, or otherwise
non-fundus image — the app returns an error instead of a prediction,
rather than silently guessing. This is a heuristic gate, not a certified
medical-image classifier, so treat it as a helpful filter rather than a
guarantee.

## Model accuracy (current trained model)

Evaluated on the held-out test set (422 images, never seen during
training):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cataract | 0.72 | 0.63 | 0.67 |
| diabetic_retinopathy | 0.75 | 0.78 | 0.77 |
| glaucoma | 0.64 | 0.16 | 0.25 |
| normal | 0.51 | 0.92 | 0.66 |

**Overall test accuracy: 63.0%** (chance = 25% for 4 classes) · **macro F1: 0.59**

Cataract and diabetic retinopathy are detected reasonably well.
**Glaucoma recall is weak (16%)** — the model frequently misses it,
often predicting "normal" instead (see `models/confusion_matrix.png`).
This is a real, honest limitation, not a display issue. To improve it:
add more glaucoma-labeled training images, train longer with a lower
learning rate specifically once glaucoma recall plateaus, or — most
effective — use transfer learning with pretrained ImageNet weights
(e.g. MobileNetV2/EfficientNet) if you run this in an environment with
internet access; this sandboxed environment could not download
pretrained weights, so the model was trained from scratch.

## Setup

1. Install Python 3.11.
2. Create/activate a virtual environment: `./venv/Scripts/activate` (Windows) or `source venv/bin/activate` (macOS/Linux).
3. `pip install --upgrade pip`
4. `pip install numpy flask pillow tensorflow scikit-learn matplotlib reportlab`

## Train the model

Place your dataset at `Fundus_diseases/<class_name>/*` next to
`train_model.py`, then:

```
python train_model.py
```

This reads `Fundus_diseases/`, trains, evaluates on a held-out test set,
and writes `models/best_cnn_model.keras` + `models/class_names.json`.
Training ~4,200 images on CPU takes a while (expect a few minutes per
epoch on a modest machine); it uses early stopping so it won't run
longer than necessary.

## Run the app

```
python app1.py
```

Then open the app in your browser and upload a retinal fundus image to
get a prediction. The app will not start if `models/best_cnn_model.keras`
or `models/class_names.json` is missing — run `train_model.py` first.
