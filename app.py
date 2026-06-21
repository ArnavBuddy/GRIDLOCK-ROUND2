import os
import shutil
import tempfile
import threading
import time
import uuid
import math
import torch
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import traffic_guardian_fsd as tg


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 750 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()
model = None
model_lock = threading.Lock()
model_stage_dir = None

CLASS_MAP = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

LANE_POLYGONS = [
    np.array([[120, 850], [400, 850], [460, 240], [390, 240]], dtype=np.float32),
    np.array([[400, 850], [680, 850], [550, 240], [460, 240]], dtype=np.float32),
    np.array([[680, 850], [960, 850], [710, 240], [550, 240]], dtype=np.float32),
]

LANE_COLORS = [(255, 100, 0), (0, 200, 0), (255, 0, 150)]


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_model():
    global model, model_stage_dir
    with model_lock:
        if model is None:
            source_weights = BASE_DIR / "yolov8n.pt"
            if not source_weights.exists():
                raise FileNotFoundError(f"YOLO weights not found: {source_weights}")

            model_stage_dir = Path(tempfile.mkdtemp(prefix="gridlock_web_model_"))
            staged_weights = model_stage_dir / "yolov8n.pt"
            shutil.copy2(source_weights, staged_weights)
            model = YOLO(str(staged_weights))
    return model


def set_job(job_id, **updates):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def get_job(job_id):
    with jobs_lock:
        return dict(jobs.get(job_id, {}))


def public_job(job):
    data = dict(job)
    data.pop("latest_frame", None)
    data.pop("latest_fsd_frame", None)
    return data


def frame_stream(job_id, frame_key):
    while True:
        job = get_job(job_id)
        if not job:
            break

        frame = job.get(frame_key)
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        if job.get("status") in {"complete", "stopped", "error"} and frame is None:
            break

        time.sleep(0.04)


def scaled_lane_polygons(width, height):
    scale_x = width / 1024.0
    scale_y = height / 1024.0
    return [
        np.array([[int(x * scale_x), int(y * scale_y)] for x, y in lane], dtype=np.int32)
        for lane in LANE_POLYGONS
    ]


def lane_index_for_point(cx, cy, lanes):
    for idx, lane in enumerate(lanes):
        if cv2.pointPolygonTest(lane, (float(cx), float(cy)), False) >= 0:
            return idx
    return -1


def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.48, 1)
    y1 = max(0, y - th - baseline - 10)
    cv2.rectangle(frame, (x, y1), (min(frame.shape[1] - 1, x + tw + 12), y), (5, 9, 18), -1)
    cv2.rectangle(frame, (x, y1), (min(frame.shape[1] - 1, x + tw + 12), y1 + 3), color, -1)
    cv2.putText(frame, text, (x + 6, y - baseline - 4), font, 0.48, (245, 247, 255), 1, cv2.LINE_AA)


def draw_dashboard_overlay(frame, metrics, progress):
    h, w = frame.shape[:2]
    lanes = scaled_lane_polygons(w, h)
    overlay = frame.copy()
    for idx, lane in enumerate(lanes):
        cv2.fillPoly(overlay, [lane], LANE_COLORS[idx])
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    panel_h = 76
    cv2.rectangle(frame, (0, 0), (w, panel_h), (7, 12, 24), -1)
    cv2.putText(frame, "AI TRAFFIC GUARDIAN", (22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 148, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Upload analysis active  Progress {progress:.0f}%", (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 185, 205), 1, cv2.LINE_AA)

    stat_text = (
        f"VEHICLES {metrics['vehicles']}   WRONG WAY {metrics['wrong_way']}   "
        f"HELMET {metrics['helmet']}   SEATBELT {metrics['seatbelt']}   FPS {metrics['fps']:.1f}"
    )
    (tw, _), _ = cv2.getTextSize(stat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, stat_text, (max(20, w - tw - 22), 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 247, 255), 1, cv2.LINE_AA)

    return lanes


def encode_jpeg(frame, quality=82):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


def draw_fsd_map(detections, metrics, width=960, height=540):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (7, 13, 24)

    cv2.putText(img, "3D FSD MAP", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (245, 247, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Realtime projection from uploaded video", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 160, 180), 1, cv2.LINE_AA)

    horizon_y = int(height * 0.22)
    road_bottom_y = height + 80
    road_top_w = int(width * 0.22)
    road_bottom_w = int(width * 0.82)
    cx = width // 2

    road = np.array([
        [cx - road_bottom_w // 2, road_bottom_y],
        [cx + road_bottom_w // 2, road_bottom_y],
        [cx + road_top_w // 2, horizon_y],
        [cx - road_top_w // 2, horizon_y],
    ], dtype=np.int32)
    cv2.fillPoly(img, [road], (12, 22, 38))
    cv2.polylines(img, [road], True, (40, 55, 80), 2, cv2.LINE_AA)

    for lane_frac in [0.33, 0.5, 0.67]:
        bottom_x = int(cx - road_bottom_w / 2 + road_bottom_w * lane_frac)
        top_x = int(cx - road_top_w / 2 + road_top_w * lane_frac)
        cv2.line(img, (bottom_x, road_bottom_y), (top_x, horizon_y), (0, 220, 255), 2, cv2.LINE_AA)

    for det in detections:
        lane = max(0, min(2, det.get("lane", 1)))
        depth = np.clip(det.get("depth", 0.5), 0.0, 1.0)
        lateral = np.clip(det.get("lateral", 0.5), 0.0, 1.0)

        y = int(horizon_y + (height * 0.70 - horizon_y) * depth)
        lane_center = (lane + 0.5) / 3.0
        x_norm = lane_center + (lateral - 0.5) * 0.18
        road_w_at_y = road_top_w + (road_bottom_w - road_top_w) * depth
        x = int(cx - road_w_at_y / 2 + road_w_at_y * x_norm)
        size = int(10 + 24 * depth)
        color = (0, 255, 80) if det.get("type") != "pedestrian" else (255, 100, 255)

        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - size, y - size), (x + size, y + size), (245, 247, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(det.get("id", "")), (x - size, y - size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    stat = f"VEHICLES {metrics['vehicles']}  DENSITY {metrics['density']:.2f}  FPS {metrics['fps']:.1f}"
    cv2.rectangle(img, (0, height - 48), (width, height), (10, 16, 28), -1)
    cv2.putText(img, stat, (28, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 247, 255), 1, cv2.LINE_AA)
    return img


def process_video(job_id, input_path, output_path):
    from collections import defaultdict, deque
    from datetime import datetime
    import sqlite3
    
    started = time.time()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        set_job(job_id, status="error", error="Could not open uploaded video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        set_job(job_id, status="error", error="Could not create annotated video.")
        return

    detector = get_model()
    
    # History buffers
    trajectories = defaultdict(lambda: deque(maxlen=24))
    speed_store = defaultdict(lambda: deque(maxlen=8))
    last_positions_3d = {}
    lane_history = defaultdict(lambda: deque(maxlen=15))
    rider_history = defaultdict(lambda: deque(maxlen=5))
    wrong_way_states = {}
    
    # Caches to avoid running checks on every frame
    helmet_status_cache = {}
    seatbelt_status_cache = {}
    last_helmet_detections = []
    processed_plates = {}
    ocr_attempts = {}
    logged_violations = defaultdict(set)
    
    # 3D Visualizer Ground
    visualizer_3d = tg.FSD3DVisualizer(960, 540)
    
    metrics = {
        "vehicles": 0,
        "violations": 0,
        "wrong_way": 0,
        "helmet": 0,
        "seatbelt": 0,
        "parking": 0,
        "density": 0.0,
        "fps": 0.0,
    }
    violation_feed = []
    evidence = []
    frame_idx = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tg.init_db()

    try:
        set_job(job_id, status="processing", progress=0)
        while True:
            frame_started = time.time()
            job = get_job(job_id)
            if job.get("cancel_requested"):
                set_job(job_id, status="stopped")
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            progress = (frame_idx / total_frames * 100.0) if total_frames else 0.0
            
            enhanced_frame = frame.copy()
            lanes = draw_dashboard_overlay(enhanced_frame, metrics, progress)
            
            results = detector.track(
                frame,
                persist=True,
                conf=0.25,
                iou=0.5,
                classes=[0, 1, 2, 3, 5, 7],
                verbose=False,
            )
            
            current_ids = set()
            pedestrians = []
            vehicles = []
            
            # Parse tracked bounding boxes
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    xyxy_boxes = boxes.xyxy.int().cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    
                    for track_id, bbox, cls_id, conf in zip(track_ids, xyxy_boxes, class_ids, confs):
                        current_ids.add(track_id)
                        class_name = CLASS_MAP.get(cls_id, "car")
                        
                        # Correct YOLO nano's class confusion (confusing SUVs/pickups with buses)
                        if class_name == "bus":
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if bbox_area < 25000:
                                class_name = "suv"
                                
                        obj = {
                            "track_id": track_id,
                            "bbox": bbox,
                            "cls_id": cls_id,
                            "class_name": class_name,
                            "conf": conf
                        }
                        if cls_id == 0:
                            pedestrians.append(obj)
                        else:
                            vehicles.append(obj)

            # Build vehicle snapshots and ahead flags for wrong way logic
            vehicle_snapshots = []
            vehicle_snapshot_by_id = {}
            for veh in vehicles:
                track_id = veh["track_id"]
                x1, y1, x2, y2 = veh["bbox"]
                cx = (x1 + x2) // 2
                cy = y2
                X_3D, Y_3D = tg.project_to_fsd_coords(cx, cy)
                polygon_lane_idx = tg.get_lane_index_by_polygon(cx, cy)
                lane_idx = tg.resolve_lane_index(cx, cy, X_3D)
                snapshot = {
                    "track_id": track_id,
                    "point": (cx, cy),
                    "X_3D": X_3D,
                    "Y_3D": Y_3D,
                    "polygon_lane_idx": polygon_lane_idx,
                    "lane_idx": lane_idx,
                }
                vehicle_snapshots.append(snapshot)
                vehicle_snapshot_by_id[track_id] = snapshot

            vehicle_ahead_flags = tg.build_vehicle_ahead_flags(vehicle_snapshots)

            vehicle_data = []
            violating_vehicles = set()
            
            # Map 3D Visualizer Ground
            map_img = np.zeros((540, 960, 3), dtype=np.uint8)
            visualizer_3d.draw_road(map_img)

            # Loop through vehicles to calculate metrics & violations
            for veh in vehicles:
                track_id = veh["track_id"]
                x1, y1, x2, y2 = veh["bbox"]
                snapshot = vehicle_snapshot_by_id[track_id]
                cx, cy = snapshot["point"]
                
                # Append vehicle coordinates to trajectory
                trajectories[track_id].append((cx, cy))
                
                # 3D Math projection
                X_3D = snapshot["X_3D"]
                Y_3D = snapshot["Y_3D"]
                
                # Speed calculation
                if track_id in last_positions_3d:
                    x_prev, y_prev = last_positions_3d[track_id]
                    dist = math.sqrt((X_3D - x_prev)**2 + (Y_3D - y_prev)**2)
                    raw_speed = dist * fps * 3.6
                    if raw_speed < 1.5:
                        raw_speed = 0.0
                    speed_store[track_id].append(raw_speed)
                    speed_kmh = sum(speed_store[track_id]) / len(speed_store[track_id])
                else:
                    speed_kmh = 0.0
                    
                last_positions_3d[track_id] = (X_3D, Y_3D)
                
                # Retrieve current lane index
                lane_idx = snapshot["lane_idx"]
                polygon_lane_idx = snapshot["polygon_lane_idx"]
                        
                # Determine stable lane assignment (using robust lane_idx fallback)
                stable_lane = tg.check_lane_stability(track_id, lane_idx, lane_history)
                
                # Perform wrong-side driving detection
                wrong_way_result = tg.check_wrong_way(
                    track_id=track_id,
                    stable_lane=stable_lane,
                    bottom_center=(cx, cy),
                    speed_kmh=speed_kmh,
                    frame_count=frame_idx,
                    wrong_way_states=wrong_way_states,
                    current_lane=lane_idx,
                    has_vehicle_ahead=vehicle_ahead_flags.get(track_id, False)
                )
                is_wrong_way = wrong_way_result["confirmed"]
                
                # Perform triple-riding detection
                is_triple_riding = False
                if veh["class_name"] in ["motorcycle", "bicycle"]:
                    is_triple_riding = tg.check_triple_riding(
                        track_id=track_id,
                        veh_bbox=veh["bbox"],
                        pedestrians=pedestrians,
                        rider_history_dict=rider_history
                    )
                
                is_violating = False
                violation_type = ""
                violation_status = ""
                n_persons_found = 0
                
                # Caching/checking of helmet and seatbelt every 6 frames
                should_check = (frame_idx % 6 == (track_id % 6))
                
                if veh["class_name"] in ["motorcycle", "bicycle"]:
                    violation_type = "HELMET"
                    if should_check or track_id not in helmet_status_cache:
                        zoomed = tg.zoom_crop_vehicle(frame, veh["bbox"], pad_ratio=0.20, min_size=640)
                        if zoomed is not None:
                            status = tg.check_helmet_on_bike(zoomed, track_id)
                            if status != "UNKNOWN":
                                helmet_status_cache[track_id] = status
                            elif track_id not in helmet_status_cache:
                                helmet_status_cache[track_id] = "SCANNING"
                    
                    cached = helmet_status_cache.get(track_id, "SCANNING")
                    violation_status = cached
                    if cached == "VIOLATION":
                        is_violating = True
                         
                elif veh["class_name"] in ["car", "bus", "truck", "suv"]:
                    violation_type = "SEATBELT"
                    if should_check or track_id not in seatbelt_status_cache:
                        zoomed = tg.zoom_crop_vehicle(frame, veh["bbox"], pad_ratio=0.10, min_size=640)
                        if zoomed is not None:
                            status, n_found = tg.check_seatbelt_zoomed(zoomed, detector, track_id, device=device)
                            if status != "UNKNOWN":
                                seatbelt_status_cache[track_id] = (status, n_found)
                            elif track_id not in seatbelt_status_cache:
                                seatbelt_status_cache[track_id] = ("SCANNING", 0)
                    
                    cached = seatbelt_status_cache.get(track_id, ("SCANNING", 0))
                    violation_status = cached[0]
                    n_persons_found = cached[1]
                    if cached[0] == "VIOLATION":
                        is_violating = True
                
                if is_violating:
                    violating_vehicles.add(track_id)
                    
                vehicle_data.append({
                    "track_id": track_id,
                    "bbox": veh["bbox"],
                    "class_name": veh["class_name"],
                    "X_3D": X_3D,
                    "Y_3D": Y_3D,
                    "lane_idx": lane_idx,
                    "is_violating": is_violating,
                    "violation_type": violation_type,
                    "violation_status": violation_status,
                    "n_persons": n_persons_found,
                    "is_wrong_way": is_wrong_way,
                    "wrong_way_status": wrong_way_result["status"],
                    "wrong_way_counter": wrong_way_result["counter"],
                    "wrong_way_direction": wrong_way_result["movement_direction"],
                    "wrong_way_confidence": wrong_way_result["confidence"],
                    "is_triple_riding": is_triple_riding,
                })

            # Full-frame helmet scanning
            helmet_frame_detections = []
            associated_no_helmets = set()
            if tg.helmet_model is not None and (frame_idx % 5 == 0):
                try:
                    h_results = tg.helmet_model(frame, conf=0.30, verbose=False)
                    if len(h_results) > 0 and h_results[0].boxes is not None:
                        h_boxes = h_results[0].boxes
                        h_xyxy = h_boxes.xyxy.int().cpu().tolist()
                        h_clss = h_boxes.cls.int().cpu().tolist()
                        h_confs = h_boxes.conf.cpu().tolist()
                        for hbox, hcls, hconf in zip(h_xyxy, h_clss, h_confs):
                            cls_name = {0: "BICYCLIST", 1: "DRIVER", 2: "HELMET", 3: "NO-HELMET"}.get(hcls, "?")
                            helmet_frame_detections.append({
                                "bbox": hbox,
                                "cls_id": hcls,
                                "cls_name": cls_name,
                                "conf": hconf,
                            })
                except Exception:
                    pass
            
            if frame_idx % 5 == 0:
                last_helmet_detections = helmet_frame_detections

            # Association of Violations
            violating_pedestrians = set()
            for ped in pedestrians:
                ped_track_id = ped["track_id"]
                px1, py1, px2, py2 = ped["bbox"]
                pcx, pcy = (px1 + px2) // 2, py2
                ped_X_3D, ped_Y_3D = tg.project_to_fsd_coords(pcx, pcy)
                
                # Check overlap with no-helmet detections
                has_no_helmet_overlap = False
                for idx, h_det in enumerate(last_helmet_detections):
                    if h_det["cls_id"] == tg.HELMET_CLS_NO_HELMET:
                        overlap = tg.get_overlap_ratio(ped["bbox"], h_det["bbox"])
                        if overlap > 0.40 or tg.center_inside(h_det["bbox"], ped["bbox"]):
                            has_no_helmet_overlap = True
                            associated_no_helmets.add(idx)
                            break
                            
                is_rider = False
                for idx, h_det in enumerate(last_helmet_detections):
                    if h_det["cls_id"] in [tg.HELMET_CLS_BICYCLIST, tg.HELMET_CLS_DRIVER, tg.HELMET_CLS_HELMET, tg.HELMET_CLS_NO_HELMET]:
                        overlap = tg.get_overlap_ratio(ped["bbox"], h_det["bbox"])
                        if overlap > 0.35:
                            is_rider = True
                            break
                            
                ped["is_rider"] = is_rider
                if has_no_helmet_overlap:
                    violating_pedestrians.add(ped_track_id)

            # Combine violations for each vehicle
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                class_name = v_info["class_name"]
                active_viols = []
                
                if v_info.get("is_wrong_way", False):
                    active_viols.append("WRONG WAY")
                    
                if v_info.get("is_triple_riding", False):
                    active_viols.append("TRIPLE RIDING")
                    
                if class_name in ["motorcycle", "bicycle"]:
                    if v_info.get("violation_status") == "VIOLATION":
                        active_viols.append("HELMET VIOLATION")
                elif class_name in ["car", "bus", "truck", "suv"]:
                    if v_info.get("violation_status") == "VIOLATION":
                        active_viols.append("SEATBELT VIOLATION")
                        
                if active_viols:
                    v_info["is_violating"] = True
                    v_info["violation_type"] = " | ".join(active_viols)
                    v_info["violation_status"] = "VIOLATION"
                    violating_vehicles.add(track_id)
                else:
                    v_info["is_violating"] = False
                    v_info["violation_type"] = ""
                    if class_name in ["motorcycle", "bicycle"]:
                        v_info["violation_status"] = helmet_status_cache.get(track_id, "SCANNING")
                    else:
                        v_info["violation_status"] = seatbelt_status_cache.get(track_id, ("SCANNING", 0))[0]

            # Compute totals for dashboard metrics
            frame_wrong_way = 0
            frame_helmet = 0
            frame_seatbelt = 0
            for v_info in vehicle_data:
                if v_info["is_violating"]:
                    if "WRONG WAY" in v_info["violation_type"]:
                        frame_wrong_way += 1
                    if "HELMET VIOLATION" in v_info["violation_type"]:
                        frame_helmet += 1
                    if "SEATBELT VIOLATION" in v_info["violation_type"]:
                        frame_seatbelt += 1
            
            metrics["wrong_way"] = frame_wrong_way
            metrics["helmet"] = frame_helmet
            metrics["seatbelt"] = frame_seatbelt
            
            # DB logging of new violations
            new_violations_logged = []
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                if v_info["is_violating"]:
                    active_viols = v_info["violation_type"].split(" | ")
                    logged_types = logged_violations[track_id]
                    unlogged_viols = [v for v in active_viols if v not in logged_types]
                    
                    if not unlogged_viols:
                        continue
                        
                    plate_number = None
                    if track_id not in processed_plates and ocr_attempts.get(track_id, 0) < 5:
                        ocr_attempts[track_id] = ocr_attempts.get(track_id, 0) + 1
                        plate_number = tg.run_lpr_on_vehicle(frame, v_info["bbox"], track_id, v_info["class_name"], frame_idx)
                        if plate_number:
                            processed_plates[track_id] = plate_number
                            
                    plate_to_log = processed_plates.get(track_id, "UNKNOWN" if ocr_attempts.get(track_id, 0) >= 5 else None)
                    if plate_to_log:
                        logged_violations[track_id].update(active_viols)
                        if "WRONG WAY" in active_viols and track_id in wrong_way_states:
                            wrong_way_states[track_id]["logged_wrong_way"] = True
                        new_violations_logged.append({
                            "track_id": track_id,
                            "class_name": v_info["class_name"],
                            "violation_type": v_info["violation_type"] or "SAFETY VIOLATION",
                            "plate_number": plate_to_log,
                            "bbox": v_info["bbox"],
                            "lane_idx": v_info["lane_idx"],
                        })

            # Save evidence photos & insert in SQLite DB
            if new_violations_logged:
                evidence_dir = Path("evidence")
                evidence_dir.mkdir(exist_ok=True)
                for viol in new_violations_logged:
                    track_id = viol["track_id"]
                    class_name = viol["class_name"]
                    v_type = viol["violation_type"]
                    plate_number = viol["plate_number"]
                    
                    clean_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    evidence_filename = f"violation_{class_name}_{track_id}_{clean_time}.jpg"
                    evidence_path = evidence_dir / evidence_filename
                    
                    evidence_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    evidence_img = tg.build_evidence_image(enhanced_frame, viol, evidence_timestamp)
                    cv2.imwrite(str(evidence_path), evidence_img)
                    
                    try:
                        conn = sqlite3.connect("traffic_violations.db")
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO violations (timestamp, track_id, vehicle_type, violation_type, plate_number, frame_number, evidence_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?);
                        """, (
                            evidence_timestamp,
                            track_id,
                            class_name,
                            v_type,
                            plate_number,
                            frame_idx,
                            str(evidence_path.resolve())
                        ))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        print(f"Database write error: {e}")
                        
                    # Add to realtime lists for JSON response
                    time_str = datetime.now().strftime("%H:%M:%S")
                    violation_feed.insert(0, {
                        "time": time_str,
                        "vehicle": class_name.upper(),
                        "tracker": f"ID:{track_id}",
                        "violation": v_type,
                        "plate": plate_number,
                        "confidence": "100%",
                    })
                    evidence.insert(0, {
                        "violation": v_type,
                        "vehicle": class_name.upper(),
                        "tracker": f"ID:{track_id}",
                    })

            metrics["violations"] = len(violation_feed)

            # Draw 3D Vehicles on 3D map
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                class_name = v_info["class_name"]
                is_violating = v_info["is_violating"]
                wrong_way_status = v_info.get("wrong_way_status", "normal")
                
                if is_violating:
                    color = (0, 0, 255) # COLOR_RED
                elif wrong_way_status == "potential":
                    color = (0, 165, 255) # COLOR_ORANGE
                elif wrong_way_status == "observing":
                    color = (0, 255, 255) # COLOR_YELLOW
                else:
                    color = (0, 255, 0) # COLOR_GREEN
                    
                visualizer_3d.draw_vehicle_3d(map_img, v_info["X_3D"], v_info["Y_3D"], class_name, track_id, color)

            # Draw 2D camera overlays on enhanced_frame
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                x1, y1, x2, y2 = v_info["bbox"]
                class_name = v_info["class_name"]
                lane_idx = v_info["lane_idx"]
                is_violating = v_info["is_violating"]
                v_type = v_info["violation_type"]
                v_status = v_info["violation_status"]
                wrong_way_status = v_info.get("wrong_way_status", "normal")
                
                if is_violating:
                    color = (0, 0, 255)
                elif wrong_way_status == "potential":
                    color = (0, 165, 255)
                elif wrong_way_status == "observing":
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                
                thickness = 3 if is_violating else 2
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                
                # Check wrong way line vector
                if v_info.get("is_wrong_way", False):
                    strobe_color = (0, 69, 255) if int(time.time() * 6) % 2 == 0 else (0, 0, 255)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.line(enhanced_frame, (enhanced_frame.shape[1] // 2, 0), (cx, cy), strobe_color, 2, cv2.LINE_AA)
                
                # Trajectory trails
                pts = list(trajectories[track_id])
                for idx, pt in enumerate(pts):
                    alpha = (idx + 1) / max(len(pts), 1)
                    dot_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(enhanced_frame, pt, max(2, int(4 * alpha)), dot_color, -1, cv2.LINE_AA)
                    
                # Label
                if is_violating:
                    label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | {v_type}]"
                else:
                    if class_name in ["motorcycle", "bicycle"]:
                        label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | HELMET: {v_status}]"
                    else:
                        label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | SEATBELT: {v_status}]"
                
                draw_label(enhanced_frame, label_text, x1, max(24, y1), color)

            # Update general frame metrics
            frame_vehicle_count = len(vehicles)
            metrics["vehicles"] = max(metrics["vehicles"], frame_vehicle_count)
            metrics["density"] = round(min(1.0, frame_vehicle_count / 18.0), 2)
            elapsed = max(time.time() - started, 0.001)
            metrics["fps"] = frame_idx / elapsed
            
            # Write annotated frame
            writer.write(enhanced_frame)
            
            # Encode frames for streaming
            camera_jpeg = encode_jpeg(enhanced_frame)
            fsd_jpeg = encode_jpeg(map_img)
            
            update_payload = {
                "status": "processing",
                "progress": round(progress, 1),
                "metrics": metrics,
                "violations": violation_feed,
                "evidence": evidence,
            }
            if camera_jpeg is not None:
                update_payload["latest_frame"] = camera_jpeg
            if fsd_jpeg is not None:
                update_payload["latest_fsd_frame"] = fsd_jpeg
            set_job(job_id, **update_payload)

            # Frame rate sync
            frame_delay = max(0.0, (1.0 / max(fps, 1.0)) - (time.time() - frame_started))
            time.sleep(min(frame_delay, 0.05))
            
            # Clean stale buffers
            stale_keys = [k for k in list(trajectories.keys()) if k not in current_ids]
            for k in stale_keys:
                del trajectories[k]
                if k in speed_store:
                    del speed_store[k]
                if k in last_positions_3d:
                    del last_positions_3d[k]
                if k in helmet_status_cache:
                    del helmet_status_cache[k]
                if k in seatbelt_status_cache:
                    del seatbelt_status_cache[k]
                if k in lane_history:
                    del lane_history[k]
                if k in rider_history:
                    del rider_history[k]
                if k in wrong_way_states:
                    del wrong_way_states[k]
                if k in logged_violations:
                    del logged_violations[k]

    except Exception as exc:
        set_job(job_id, status="error", error=str(exc))
        return
    finally:
        cap.release()
        writer.release()

    if get_job(job_id).get("status") != "stopped":
        set_job(
            job_id,
            status="complete",
            progress=100,
            metrics=metrics,
            violations=violation_feed,
            evidence=evidence,
            output_url=f"/processed/{output_path.name}",
        )


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.post("/api/upload")
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "Upload a traffic video first."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Upload a traffic video first."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed traffic video types: MP4, AVI, MOV, MKV."}), 400

    job_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename)
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    output_path = PROCESSED_DIR / f"annotated_{job_id}.mp4"
    file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "filename": safe_name,
            "metrics": {
                "vehicles": 0,
                "violations": 0,
                "wrong_way": 0,
                "helmet": 0,
                "seatbelt": 0,
                "parking": 0,
                "density": 0.0,
                "fps": 0.0,
            },
            "violations": [],
            "evidence": [],
            "output_url": None,
            "error": None,
            "cancel_requested": False,
            "latest_frame": None,
            "latest_fsd_frame": None,
        }

    thread = threading.Thread(target=process_video, args=(job_id, input_path, output_path), daemon=True)
    thread.start()
    return jsonify(public_job(get_job(job_id)))


@app.get("/api/status/<job_id>")
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(public_job(job))


@app.post("/api/stop/<job_id>")
def stop_job(job_id):
    if not get_job(job_id):
        return jsonify({"error": "Job not found."}), 404
    set_job(job_id, cancel_requested=True)
    return jsonify(public_job(get_job(job_id)))


@app.route("/api/stream/<job_id>")
def stream_camera(job_id):
    return Response(frame_stream(job_id, "latest_frame"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/fsd/<job_id>")
def stream_fsd(job_id):
    return Response(frame_stream(job_id, "latest_fsd_frame"), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/processed/<path:filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True, threaded=True)
