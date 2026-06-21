# License Plate Recognition

## Overview

This module is the first independent GRIDLOCK traffic violation detection
component. It detects a license plate in an input vehicle image, segments the
plate into individual characters, and recognizes each character with a CNN.

The implementation follows the attached reference notebook's workflow:

- Haar cascade plate localization where a cascade file is available.
- Contour-based plate localization fallback for environments without the
  Indian plate cascade XML.
- Character segmentation from a resized plate image using thresholding,
  morphology, contour filtering, and left-to-right sorting.
- A Keras CNN based on the notebook architecture for `0-9A-Z` character
  recognition.

## Architecture

```text
Image
  |
  v
Preprocessing
  |
  v
Plate Detection
  |
  v
Character Segmentation
  |
  v
CNN Recognition
  |
  v
Plate Number
```

## Folder Layout

```text
modules/license_plate_recognition/
|-- README.md
|-- requirements.txt
|-- config.py
|-- inference.py
|-- train.py
|-- evaluate.py
|-- utils.py
|-- preprocessing.py
|-- plate_detector.py
|-- character_segmenter.py
|-- character_recognizer.py
|-- models/
|-- datasets/
|-- outputs/
`-- tests/
```

## Installation

```bash
cd modules/license_plate_recognition
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Dataset Format

Training and evaluation expect labelled character folders. Directory names must
match the supported classes: `0-9` and `A-Z`.

```text
datasets/
|-- train/
|   |-- 0/
|   |-- A/
|   `-- Z/
|-- val/
|   |-- 0/
|   |-- A/
|   `-- Z/
`-- test/
    |-- 0/
    |-- A/
    `-- Z/
```

The training script also accepts the reference notebook/Kaggle layout:

```text
datasets/
`-- data/
    `-- data/
        |-- train/
        `-- val/
```

If you only have one folder of labelled class directories, pass that folder with
`--data-dir`; the script will use the configured validation split.

If you have the Indian plate Haar cascade from the notebook dataset, place it at:

```text
datasets/indian_license_plate.xml
```

When that file is missing, the detector attempts OpenCV's bundled plate cascade
and then falls back to contour-based localization.

## Training

```bash
python train.py
```

Optional paths:

```bash
python train.py --data-dir datasets --output-model models/license_plate_cnn.h5
python train.py --data-dir C:\path\to\ai-indian-license-plate-recognition-data\data\data
```

The script saves:

```text
models/license_plate_cnn.h5
```

## Inference

```bash
python inference.py --image path/to/vehicle.jpg
```

Optional custom model path:

```bash
python inference.py --image path/to/vehicle.jpg --model models/license_plate_cnn.h5
```

Example JSON output:

```json
{
  "plate_number": "MH12AB1234",
  "confidence": 0.96
}
```

The annotated image is saved to:

```text
outputs/annotated_image.jpg
```

## Evaluation

```bash
python evaluate.py
```

Optional paths:

```bash
python evaluate.py --data-dir datasets/test --model models/license_plate_cnn.h5
python evaluate.py --data-dir C:\path\to\archive\data\data\val
```

If `datasets/test` is missing, the script automatically tries `datasets/val`
and the notebook layout `datasets/data/data/val`.

The evaluation module reports accuracy, precision, recall, and F1 score, then
saves:

```text
outputs/evaluation_report.json
```

## Configuration

All paths, thresholds, image sizes, model names, and detection settings are
defined in `config.py`. Update `LPRConfig` instead of hardcoding values in the
pipeline modules.

## Notes

- `preprocessing.py` handles low light, shadows, blur, and noisy inputs with
  denoising, CLAHE, gamma correction, resizing, grayscale conversion, and
  adaptive thresholding.
- `plate_detector.py` returns detection dictionaries with a bounding box and
  confidence: `{"bbox": [x1, y1, x2, y2], "confidence": 0.91}`.
- `character_segmenter.py` returns sorted character image crops.
- `character_recognizer.py` loads the CNN and returns plate text such as
  `MH12AB1234`.
