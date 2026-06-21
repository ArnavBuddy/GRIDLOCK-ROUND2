import os
import sys
import shutil
import tempfile
import cv2
import easyocr
import pandas as pd
from collections import Counter
from ultralytics import YOLO
from pathlib import Path

# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

HELMET_MODEL_PATH = BASE_DIR / "models" / "helmet_detector.pt"
PLATE_MODEL_PATH  = BASE_DIR / "models" / "plate_detector.pt"
VIDEO_PATH        = BASE_DIR / "test_video.mp4"
OUTPUT_VIDEO      = BASE_DIR / "outputs" / "output_video.mp4"
OUTPUT_CSV        = BASE_DIR / "outputs" / "violations.csv"
PLATE_FOLDER      = BASE_DIR / "outputs" / "video_plates"

# Class IDs from your helmet model (train-9 dataset)
# 0 = bicyclist, 1 = driver, 2 = helmet, 3 = no-helmet
NO_HELMET_CLASS = 3

# Minimum plate detection confidence
PLATE_CONF = 0.60

# Minimum plate crop size (pixels)
MIN_PLATE_W = 40
MIN_PLATE_H = 20

# Padding around detected plate
PLATE_PAD = 10

# OCR upscale factor
OCR_SCALE = 8

# Minimum characters for a valid plate read
MIN_PLATE_LEN = 4
MIN_FILTERED_LEN = 8

print("=" * 50)
print("Helmet Violation System")
print("=" * 50)
print(f"BASE_DIR : {BASE_DIR}")
print(f"VIDEO    : {VIDEO_PATH}")
print(f"HELMET   : {HELMET_MODEL_PATH}")
print(f"PLATE    : {PLATE_MODEL_PATH}")

# ==================================================
# PRE-FLIGHT CHECKS
# ==================================================

errors = []

if not HELMET_MODEL_PATH.exists():
    errors.append(f"[ERROR] Helmet model not found: {HELMET_MODEL_PATH}")
if not PLATE_MODEL_PATH.exists():
    errors.append(f"[ERROR] Plate model not found: {PLATE_MODEL_PATH}")
if not VIDEO_PATH.exists():
    errors.append(f"[ERROR] Video not found: {VIDEO_PATH}")

if errors:
    for e in errors:
        print(e)
    print("\nMake sure your folder structure looks like:")
    print("  helmet_violation_system/")
    print("  ├── helmet_violation_system.py")
    print("  ├── test_video.mp4")
    print("  ├── models/")
    print("  │   ├── helmet_detector.pt")
    print("  │   └── plate_detector.pt")
    print("  └── outputs/")
    sys.exit(1)

print("\nHelmet model : EXISTS")
print("Plate model  : EXISTS")
print("Video        : EXISTS")

# ==================================================
# SETUP OUTPUT DIRS
# ==================================================

os.makedirs(PLATE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VIDEO.parent, exist_ok=True)

# Clear old plate crops
for f in PLATE_FOLDER.iterdir():
    try:
        f.unlink()
    except Exception:
        pass

# ==================================================
# WORKAROUND: copy models to apostrophe-free temp path
# Windows/PyTorch's C++ file loader strips apostrophes
# from paths, so we stage the files in a clean temp dir.
# ==================================================

_tmp_dir = Path(tempfile.mkdtemp(prefix="gridlock_"))
HELMET_MODEL_SAFE = _tmp_dir / "helmet_detector.pt"
PLATE_MODEL_SAFE  = _tmp_dir / "plate_detector.pt"

shutil.copy2(HELMET_MODEL_PATH, HELMET_MODEL_SAFE)
shutil.copy2(PLATE_MODEL_PATH,  PLATE_MODEL_SAFE)

print(f"\nModels staged at: {_tmp_dir}")

# ==================================================
# LOAD MODELS
# ==================================================

print("Loading models...")

try:
    helmet_model = YOLO(str(HELMET_MODEL_SAFE))
except Exception as e:
    print(f"[ERROR] Failed to load helmet model: {e}")
    sys.exit(1)

try:
    plate_model = YOLO(str(PLATE_MODEL_SAFE))
except Exception as e:
    print(f"[ERROR] Failed to load plate model: {e}")
    sys.exit(1)

try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    print(f"[ERROR] Failed to init EasyOCR: {e}")
    sys.exit(1)

print("Models loaded successfully.")

# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo info: {width}x{height} @ {fps}fps, {total_frames} frames")

# ==================================================
# VIDEO WRITER
# ==================================================

if OUTPUT_VIDEO.exists():
    OUTPUT_VIDEO.unlink()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))

if not out.isOpened():
    print(f"[ERROR] Cannot create output video: {OUTPUT_VIDEO}")
    cap.release()
    sys.exit(1)

# ==================================================
# MAIN PROCESSING LOOP
# ==================================================

frame_no   = 0
crop_count = 0
violation_frames = 0

print("\nProcessing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    if frame_no % 50 == 0:
        pct = (frame_no / total_frames * 100) if total_frames else 0
        print(f"  Frame {frame_no}/{total_frames} ({pct:.1f}%)")

    # --- Helmet detection ---
    helmet_results = helmet_model(frame, verbose=False)
    annotated = helmet_results[0].plot()

    no_helmet_found = any(
        int(box.cls[0]) == NO_HELMET_CLASS
        for r in helmet_results
        for box in r.boxes
    )

    if no_helmet_found:
        violation_frames += 1

        # --- Plate detection ---
        plate_results = plate_model(frame, conf=PLATE_CONF, verbose=False)

        for r in plate_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                if w < MIN_PLATE_W or h < MIN_PLATE_H:
                    continue

                # Padded crop
                px1 = max(0, x1 - PLATE_PAD)
                py1 = max(0, y1 - PLATE_PAD)
                px2 = min(frame.shape[1], x2 + PLATE_PAD)
                py2 = min(frame.shape[0], y2 + PLATE_PAD)

                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    continue

                crop_path = PLATE_FOLDER / f"crop_{crop_count:05d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crop_count += 1

                # Draw plate box on annotated frame
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(
                    annotated, "NUMBER PLATE",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2
                )

    out.write(annotated)

cap.release()
out.release()

print(f"\nVideo processing complete.")
print(f"  Frames processed  : {frame_no}")
print(f"  Violation frames  : {violation_frames}")
print(f"  Plate crops saved : {crop_count}")

# ==================================================
# OCR ON PLATE CROPS
# ==================================================

print("\nRunning OCR on plate crops...")

all_texts = []
ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

for crop_file in sorted(PLATE_FOLDER.iterdir()):
    if not crop_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        continue

    img = cv2.imread(str(crop_file))
    if img is None:
        continue

    try:
        # Upscale + preprocess
        img = cv2.resize(img, None, fx=OCR_SCALE, fy=OCR_SCALE,
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = reader.readtext(
            thresh,
            detail=0,
            paragraph=True,
            allowlist=ALLOWED
        )

        if result:
            text = "".join(result).strip()
            if len(text) >= MIN_PLATE_LEN:
                all_texts.append(text)

    except Exception as e:
        print(f"  [WARN] OCR failed on {crop_file.name}: {e}")

print(f"Raw OCR reads: {all_texts}")

# ==================================================
# NORMALIZATION & VOTING
# ==================================================

def normalize(text: str) -> str:
    """Uppercase, strip spaces, fix common OCR confusions."""
    text = text.upper().replace(" ", "")
    # Common OCR substitutions for plates
    text = text.replace("O", "0")
    text = text.replace("I", "1")
    text = text.replace("S", "5")
    return text

filtered = []
for text in all_texts:
    clean = normalize(text)
    if len(clean) < MIN_FILTERED_LEN:
        continue
    if not any(c.isalpha() for c in clean):
        continue
    if not any(c.isdigit() for c in clean):
        continue
    filtered.append(clean)

counter = Counter(filtered)
print(f"\nOCR voting results: {counter.most_common(10)}")

final_plate = counter.most_common(1)[0][0] if counter else "UNKNOWN"
print(f"Final plate number : {final_plate}")

# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame([{
    "plate": final_plate,
    "violation": "No Helmet",
    "violation_frames": violation_frames,
    "total_frames": frame_no,
    "plate_crops": crop_count,
}])

df.to_csv(str(OUTPUT_CSV), index=False)

# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 50)
print("DONE")
print("=" * 50)
print(f"Output video : {OUTPUT_VIDEO}")
print(f"CSV report   : {OUTPUT_CSV}")
print(f"Plate number : {final_plate}")
print(df.to_string(index=False))

# Cleanup temp model copies
shutil.rmtree(_tmp_dir, ignore_errors=True)