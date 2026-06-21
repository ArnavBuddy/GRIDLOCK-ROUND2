# Evidence Generation

## Overview

This module produces violation evidence for the GRIDLOCK traffic violation
detection system. It takes a source image plus structured violation records,
draws annotations around the violating objects, and stores timestamped metadata.

It is independent of the license plate recognition module and writes only inside
`modules/evidence_generation`.

## Architecture

```text
Image + Violation Metadata
  |
  v
Annotation
  |
  v
Annotated Evidence Image
  |
  v
Metadata JSON + JSONL Manifest
```

## Installation

```bash
cd modules/evidence_generation
pip install -r requirements.txt
```

## Violation Input Format

Create a JSON file with one or more violation records:

```json
{
  "violations": [
    {
      "violation_type": "red_light_jump",
      "bbox": [50, 80, 220, 180],
      "confidence": 0.94,
      "track_id": "vehicle-12",
      "description": "Vehicle crossed stop line after signal turned red",
      "metadata": {
        "speed_kmph": 48
      }
    }
  ]
}
```

Required fields:

- `violation_type`
- `bbox` as `[x1, y1, x2, y2]`

Optional fields:

- `confidence`
- `track_id`
- `description`
- `timestamp`
- `metadata`

## Usage

```bash
python cli.py --image path/to/frame.jpg --violations path/to/violations.json
```

On Windows, wrap paths in quotes when they contain spaces:

```bash
python cli.py --image "C:\Users\Arnav Majithia'\Downloads\archive\car.jpg" --violations "C:\Users\Arnav Majithia'\Downloads\violations.json"
```

Optional context:

```bash
python cli.py \
  --image path/to/frame.jpg \
  --violations path/to/violations.json \
  --camera-id cam-01 \
  --frame-id frame-000123 \
  --location "Main Road Junction"
```

## Outputs

The module writes:

```text
outputs/
|-- evidence_<id>.jpg
|-- evidence_<id>.json
`-- violation_events.jsonl
```

The JSON file stores:

- evidence ID
- source image path
- annotated image path
- generated timestamp
- camera/frame/location context
- violation metadata and bounding boxes

The JSONL manifest appends one record per generated evidence item so downstream
services can ingest violation evidence incrementally.
