"""
traffic_guardian_fsd.py - Integrated AI Traffic Guardian & Tesla FSD 3D Map Visualizer

Features:
1. Camera View with custom lane overlays, speed breaker, crosswalk, active HUD telemetry.
2. Tesla FSD 3D Map showing projected 3D bounding boxes and Google Maps GPS locked telemetry.
3. Spatial overlap reasoning to associate pedestrians with vehicles (inside car -> seatbelt check, on bike -> helmet check).
4. Auto-color violating vehicles in RED across both windows.
5. Hide speeds and speed breaker behaviors from labels.
"""

import os
import sys
import time
import math
import glob
import tempfile
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
import easyocr
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Colors & Presets
# ---------------------------------------------------------------------------
COLOR_BLUE = (255, 100, 0)
COLOR_GREEN = (0, 200, 0)
COLOR_PURPLE = (255, 0, 150)
COLOR_YELLOW = (0, 215, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (240, 240, 240)
COLOR_BLACK = (15, 15, 15)
COLOR_GREEN_BRIGHT = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 140, 255)

CLASS_MAP = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CLASS_COLORS = {
    "pedestrian": (255, 100, 255),
    "bicycle": (255, 255, 0),
    "car": (255, 100, 0),
    "suv": (255, 100, 0),
    "motorcycle": (0, 165, 255),
    "bus": (0, 200, 0),
    "truck": (0, 0, 220)
}

def get_track_color(track_id):
    palette = [
        (255, 120, 0),    # vibrant azure / blue
        (0, 165, 255),    # vibrant orange
        (50, 220, 240),   # yellow-orange
        (255, 50, 150),   # hot pink
        (0, 200, 100),    # vibrant green
        (200, 50, 200),   # bright purple
        (255, 150, 0),    # sky blue
        (0, 100, 250),    # amber/orange
        (220, 20, 60),    # deep red
        (240, 230, 140),  # light yellow
        (147, 112, 219),  # medium purple
        (0, 255, 127),    # spring green
    ]
    return palette[track_id % len(palette)]

# ---------------------------------------------------------------------------
# Preprocessing / Image Enhancement functions
# ---------------------------------------------------------------------------
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_shadow_reduction(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    _, shadow_mask = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shadow_mask_blur = cv2.GaussianBlur(shadow_mask, (21, 21), 0) / 255.0
    l_float = l.astype(np.float32)
    l_boosted = l_float + shadow_mask_blur * (128.0 - l_float) * 0.4
    l_new = np.clip(l_boosted, 0, 255).astype(np.uint8)
    lab_new = cv2.merge([l_new, a, b])
    return cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int32)
    return cv2.filter2D(img, -1, kernel)

def apply_denoise(img):
    return cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

def apply_white_balance(img):
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b > 0 and avg_g > 0 and avg_r > 0:
        scale_b = avg_gray / avg_b
        scale_g = avg_gray / avg_g
        scale_r = avg_gray / avg_r
        out_b = np.clip(img[:, :, 0] * scale_b, 0, 255).astype(np.uint8)
        out_g = np.clip(img[:, :, 1] * scale_g, 0, 255).astype(np.uint8)
        out_r = np.clip(img[:, :, 2] * scale_r, 0, 255).astype(np.uint8)
        return cv2.merge([out_b, out_g, out_r])
    return img.copy()

def apply_contrast_stretch(img):
    channels = cv2.split(img)
    out_channels = []
    for ch in channels:
        min_val, max_val, _, _ = cv2.minMaxLoc(ch)
        if max_val > min_val:
            stretched = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX)
            out_channels.append(stretched)
        else:
            out_channels.append(ch.copy())
    return cv2.merge(out_channels)

def enhance_frame(img, preproc_status):
    frame = img.copy()
    gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray_check))

    # Adaptive night detection: auto-boost filters
    is_night = mean_brightness < 80
    
    clahe_clip = 4.0 if is_night else 2.0
    gamma_val = 2.0 if is_night else 1.2
    
    # White Balance
    if preproc_status["wb"] or (is_night and preproc_status["shadow"]):
        frame = apply_white_balance(frame)
        
    # Contrast Stretch
    if preproc_status["dehaze"] or is_night:
        frame = apply_contrast_stretch(frame)

    # Shadow Reduction
    if preproc_status["shadow"] or is_night:
        frame = apply_shadow_reduction(frame)

    # CLAHE
    if preproc_status["clahe"]:
        frame = apply_clahe(frame, clip_limit=clahe_clip)

    # Gamma
    if preproc_status["gamma"]:
        frame = apply_gamma(frame, gamma=gamma_val)

    # Denoise
    if preproc_status["denoise"]:
        frame = apply_denoise(frame)

    # Sharpen
    if preproc_status["sharpen"]:
        frame = apply_sharpen(frame)

    return frame

# ---------------------------------------------------------------------------
# Telemetry HUD Utilities
# ---------------------------------------------------------------------------
class TrafficPhaseManager:
    def __init__(self):
        self.start_time = time.time()
        self.phases = ["NS", "EW"]
        self.phase_idx = 0
        self.phase_duration = [30.0, 25.0]  # Duration per phase in seconds
        self.yellow_duration = 4.0
        self.state = "GREEN"  # GREEN, YELLOW
        
    def update(self):
        elapsed = time.time() - self.start_time
        curr_duration = self.phase_duration[self.phase_idx]
        
        if elapsed < curr_duration - self.yellow_duration:
            self.state = "GREEN"
        elif elapsed < curr_duration:
            self.state = "YELLOW"
        else:
            self.phase_idx = (self.phase_idx + 1) % len(self.phases)
            self.start_time = time.time()
            self.state = "GREEN"
            
    def get_status(self):
        elapsed = time.time() - self.start_time
        curr_duration = self.phase_duration[self.phase_idx]
        return self.phases[self.phase_idx], self.state, elapsed, curr_duration

def draw_hazard_bar(img, y_start, y_end, x_start, x_end):
    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), COLOR_YELLOW, -1)
    stripe_width = 25
    for x in range(x_start - 30, x_end + 30, stripe_width * 2):
        pts = np.array([
            [x, y_start],
            [x + stripe_width, y_start],
            [x + stripe_width - 15, y_end],
            [x - 15, y_end]
        ], dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], x_start, x_end)
        cv2.fillPoly(img, [pts], COLOR_BLACK)

def draw_crosswalk(img, y_start, y_end, x_start, x_end):
    block_width = 45
    gap = 25
    for x in range(x_start, x_end, block_width + gap):
        cv2.rectangle(img, (x, y_start), (x + block_width, y_end), COLOR_WHITE, -1)

def draw_hud_panels(img, elapsed_time, active_imgsz, preproc_status, phase_mgr, lane_counts, lane_pressures, total_violations):
    overlay = img.copy()
    
    # Panel 1: CSIN LIVE TELEMETRY (Top Left)
    p1_x1, p1_y1 = 30, 40
    p1_x2, p1_y2 = 340, 260
    cv2.rectangle(overlay, (p1_x1, p1_y1), (p1_x2, p1_y2), COLOR_BLACK, -1)
    cv2.rectangle(overlay, (p1_x1, p1_y1), (p1_x2, p1_y2), COLOR_CYAN, 1)
    
    # Panel 2: AI TRAFFIC GUARDIAN (Bottom Left)
    p2_x1, p2_y1 = 30, 280
    p2_x2, p2_y2 = 340, 520
    cv2.rectangle(overlay, (p2_x1, p2_y1), (p2_x2, p2_y2), COLOR_BLACK, -1)
    cv2.rectangle(overlay, (p2_x1, p2_y1), (p2_x2, p2_y2), COLOR_CYAN, 1)
    
    # Alpha blend
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    
    # P1 Text
    cv2.putText(img, "CSIN LIVE TELEMETRY", (p1_x1 + 15, p1_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)
    
    phase_name, phase_state, elapsed_p, hold_p = phase_mgr.get_status()
    cv2.putText(img, f"ACTIVE PHASE: {phase_name}", (p1_x1 + 15, p1_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    
    badge_color = COLOR_GREEN_BRIGHT if phase_state == "GREEN" else COLOR_YELLOW
    cv2.rectangle(img, (p1_x2 - 85, p1_y1 + 38), (p1_x2 - 15, p1_y1 + 55), badge_color, -1)
    cv2.putText(img, phase_state, (p1_x2 - 80 if phase_state == "GREEN" else p1_x2 - 83, p1_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_BLACK, 2, cv2.LINE_AA)
    
    timer_str = f"Hold: {hold_p:.1f}s (Elapsed: {elapsed_p:.1f}s)"
    cv2.putText(img, timer_str, (p1_x1 + 15, p1_y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    
    cv2.putText(img, "PHASE  QUEUE  WAIT(max)  PRESSURE", (p1_x1 + 15, p1_y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1, cv2.LINE_AA)
    
    ns_queue = f"{lane_counts[1]:02d}"
    ns_pressure = f"{lane_pressures[1]:.2f}"
    ns_wait = f"{elapsed_p:.1f}s" if phase_name == "NS" else "0.0s"
    cv2.putText(img, f"NS      {ns_queue}     {ns_wait}      {ns_pressure}", (p1_x1 + 15, p1_y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    
    ew_queue = f"{(lane_counts[0] + lane_counts[2]):02d}"
    ew_pressure = f"{(lane_pressures[0] + lane_pressures[2])/2.0:.2f}"
    ew_wait = f"{elapsed_p:.1f}s" if phase_name == "EW" else "0.0s"
    cv2.putText(img, f"EW      {ew_queue}     {ew_wait}      {ew_pressure}", (p1_x1 + 15, p1_y1 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    
    cv2.putText(img, "EVPS: ADAPTIVE REGULATION", (p1_x1 + 15, p1_y1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GREEN_BRIGHT, 1, cv2.LINE_AA)
    
    # P2 Text
    cv2.putText(img, "AI TRAFFIC GUARDIAN", (p2_x1 + 15, p2_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "SYSTEM MODE: TESLA-VISION SPATIAL INTERFACE", (p2_x1 + 15, p2_y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GREEN_BRIGHT, 1, cv2.LINE_AA)
    cv2.putText(img, "ENFORCEMENT: AUTO ROAD SCANNING ACTIVE", (p2_x1 + 15, p2_y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"RESOLUTION : YOLO-{active_imgsz}px", (p2_x1 + 15, p2_y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)
    
    filters_on = [k.upper() for k, v in preproc_status.items() if v]
    f_str1 = " | ".join(filters_on[:3])
    f_str2 = " | ".join(filters_on[3:])
    cv2.putText(img, f"FILTERS 1  : {f_str1 if f_str1 else 'NONE'}", (p2_x1 + 15, p2_y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"FILTERS 2  : {f_str2 if f_str2 else 'NONE'}", (p2_x1 + 15, p2_y1 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)
    
    viol_str = "STANDBY (NO VIOLATIONS)" if total_violations == 0 else f"ALERT: {total_violations} SAFETY VIOLATIONS"
    viol_color = COLOR_WHITE if total_violations == 0 else COLOR_RED
    cv2.putText(img, f"TARGET TRAIL: {viol_str}", (p2_x1 + 15, p2_y1 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, viol_color, 1, cv2.LINE_AA)
    cv2.putText(img, "ZONE STATE : SCANNING FOR VIOLATIONS", (p2_x1 + 15, p2_y1 + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Tesla FSD 3D Map Rendering Engine
# ---------------------------------------------------------------------------
class FSD3DVisualizer:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.horizon_y = 120
        self.CamZ = 160
        self.focal_length = 400
        
    def project(self, x, y, z):
        if y <= 0:
            y = 0.1
        scale = self.focal_length / y
        px = int(self.center_x + x * scale)
        py = int(self.horizon_y + (self.CamZ - z) * scale)
        return px, py
        
    def draw_road(self, img):
        img.fill(12)  # solid dark charcoal background
        
        # Horizon Line
        cv2.line(img, (0, self.horizon_y), (self.width, self.horizon_y), (35, 35, 35), 1)
        
        # Road Surface Polygon
        nl = self.project(-90, 60, 0)
        nr = self.project(90, 60, 0)
        fl = self.project(-90, 700, 0)
        fr = self.project(90, 700, 0)
        
        road_poly = np.array([nl, fl, fr, nr], dtype=np.int32)
        cv2.fillPoly(img, [road_poly], (24, 24, 26))
        
        # Distance Grid Lines
        for y_grid in range(80, 700, 45):
            gl = self.project(-120, y_grid, 0)
            gr = self.project(120, y_grid, 0)
            alpha = max(0.0, 1.0 - (y_grid / 600.0))
            color_val = int(45 * alpha)
            cv2.line(img, gl, gr, (color_val, color_val, color_val), 1)
            
        # Left & Right Solid Lane Edges
        cv2.line(img, nl, fl, (70, 70, 70), 2, cv2.LINE_AA)
        cv2.line(img, nr, fr, (70, 70, 70), 2, cv2.LINE_AA)
        
        # Dashed lane separators (X = -30, X = 30)
        for y_dash in range(60, 700, 20):
            if (y_dash // 20) % 2 == 0:
                p1_s = self.project(-30, y_dash, 0)
                p1_e = self.project(-30, y_dash + 10, 0)
                cv2.line(img, p1_s, p1_e, (120, 120, 120), 1, cv2.LINE_AA)
                
                p2_s = self.project(30, y_dash, 0)
                p2_e = self.project(30, y_dash + 10, 0)
                cv2.line(img, p2_s, p2_e, (120, 120, 120), 1, cv2.LINE_AA)
                


        # Crosswalk Zebra Crossing (Y = 90 to 110)
        for x_strip in range(-80, 90, 20):
            s_nl = self.project(x_strip - 5, 90, 0)
            s_nr = self.project(x_strip + 5, 90, 0)
            s_fl = self.project(x_strip - 5, 110, 0)
            s_fr = self.project(x_strip + 5, 110, 0)
            s_poly = np.array([s_nl, s_fl, s_fr, s_nr], dtype=np.int32)
            cv2.fillPoly(img, [s_poly], (180, 180, 180))
            
        # Lane Labels (L1, L2, L3)
        l1 = self.project(-60, 70, 0)
        l2 = self.project(0, 70, 0)
        l3 = self.project(60, 70, 0)
        cv2.putText(img, "L1", (l1[0] - 8, l1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, "L2", (l2[0] - 8, l2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, "L3", (l3[0] - 8, l3[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        
    def draw_vehicle_3d(self, img, x_3d, y_3d, class_name, track_id, color):
        # 3D Dimensions per vehicle category
        if class_name in ["truck", "bus"]:
            w, l, h = 26, 52, 26
        elif class_name in ["motorcycle", "bicycle"]:
            w, l, h = 10, 22, 14
        elif class_name in ["person", "pedestrian"]:
            w, l, h = 6, 6, 16
        else:
            w, l, h = 20, 36, 16
            
        # Vertices in 3D
        c3d = [
            (x_3d - w/2, y_3d - l/2, 0),  # BLB
            (x_3d + w/2, y_3d - l/2, 0),  # BRB
            (x_3d + w/2, y_3d + l/2, 0),  # FRB
            (x_3d - w/2, y_3d + l/2, 0),  # FLB
            (x_3d - w/2, y_3d - l/2, h),  # BLT
            (x_3d + w/2, y_3d - l/2, h),  # BRT
            (x_3d + w/2, y_3d + l/2, h),  # FRT
            (x_3d - w/2, y_3d + l/2, h),  # FLT
        ]
        
        pts = [self.project(cx, cy, cz) for cx, cy, cz in c3d]
        
        # Cull faces behind horizon
        for pt in pts:
            if pt[1] < self.horizon_y:
                return
                
        # Draw translucent glass faces
        overlay = img.copy()
        faces = [
            [pts[0], pts[1], pts[2], pts[3]],  # Bottom
            [pts[4], pts[5], pts[6], pts[7]],  # Top
            [pts[0], pts[3], pts[7], pts[4]],  # Left
            [pts[1], pts[2], pts[6], pts[5]],  # Right
            [pts[3], pts[2], pts[6], pts[7]],  # Front
            [pts[0], pts[1], pts[5], pts[4]],  # Back
        ]
        
        # Darker transparent glass face color based on outline color
        fill_color = tuple(max(0, int(c * 0.7)) for c in color)
        border_color = color
        
        for face in faces:
            cv2.fillPoly(overlay, [np.array(face, dtype=np.int32)], fill_color)
            
        cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)
        
        # Draw wireframe edges
        # Bottom Face Outline
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], border_color, 1, cv2.LINE_AA)
        # Top Face Outline
        for i in range(4):
            cv2.line(img, pts[i + 4], pts[((i + 1) % 4) + 4], border_color, 1, cv2.LINE_AA)
        # Vertical Pillars
        for i in range(4):
            cv2.line(img, pts[i], pts[i + 4], border_color, 1, cv2.LINE_AA)
            
        # Draw 3D track ID floating badge
        top_cx = int(sum(pt[0] for pt in pts[4:8]) / 4)
        top_cy = int(sum(pt[1] for pt in pts[4:8]) / 4)
        
        label = f"ID:{track_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        
        rx1, ry1 = top_cx - tw // 2 - 4, top_cy - th // 2 - 3
        rx2, ry2 = top_cx + tw // 2 + 4, top_cy + th // 2 + 3
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (10, 10, 10), -1)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), border_color, 1)
        cv2.putText(img, label, (top_cx - tw // 2, top_cy + th // 2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

def draw_gps_panel(img, width, height):
    px1, py1 = width - 240, 40
    px2, py2 = width - 30, 150
    
    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_BLACK, -1)
    cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_CYAN, 1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Pin icon
    pin_cx, pin_cy = px1 + 25, py1 + 45
    cv2.circle(img, (pin_cx, pin_cy), 7, (0, 0, 255), -1, cv2.LINE_AA)
    pin_pts = np.array([
        [pin_cx - 7, pin_cy + 3],
        [pin_cx + 7, pin_cy + 3],
        [pin_cx, pin_cy + 15]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pin_pts], (0, 0, 255))
    cv2.circle(img, (pin_cx, pin_cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)
    
    cv2.putText(img, "GOOGLE MAPS", (px1 + 45, py1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(img, "Lat: 28.6139* N", (px1 + 45, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "Lon: 77.2090* E", (px1 + 45, py1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "GPS LOCK: STABLE", (px1 + 15, py1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_GREEN_BRIGHT, 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Spatial Reasoning & Heuristics Utilities
# ---------------------------------------------------------------------------
def get_overlap_ratio(boxA, boxB):
    """Calculate what ratio of boxA (the pedestrian) is inside boxB (the vehicle)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return interArea / max(float(areaA), 1.0)

def center_inside(boxA, boxB):
    """Check if the center of boxA falls inside boxB."""
    cx = (boxA[0] + boxA[2]) / 2.0
    cy = (boxA[1] + boxA[3]) / 2.0
    return boxB[0] <= cx <= boxB[2] and boxB[1] <= cy <= boxB[3]

def proximity_score(boxA, boxB):
    """Calculate normalized proximity between two boxes (0=far, 1=overlapping)."""
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0
    dist = math.sqrt((cxA - cxB)**2 + (cyA - cyB)**2)
    diag = math.sqrt((boxB[2] - boxB[0])**2 + (boxB[3] - boxB[1])**2)
    return max(0.0, 1.0 - dist / max(diag, 1.0))

# Lane configuration structure
LANES_CONFIG = [
    {
        "id": 0,
        "name": "EW_in",
        "polygon": np.array([[120, 850], [400, 850], [460, 240], [390, 240]], dtype=np.int32),
        "direction": "inbound",
        "color": COLOR_BLUE,
        "text_pos": (220, 645)
    },
    {
        "id": 1,
        "name": "NS_out",
        "polygon": np.array([[400, 850], [680, 850], [550, 240], [460, 240]], dtype=np.int32),
        "direction": "outbound",
        "color": COLOR_GREEN,
        "text_pos": (490, 645)
    },
    {
        "id": 2,
        "name": "W_out",
        "polygon": np.array([[680, 850], [960, 850], [710, 240], [550, 240]], dtype=np.int32),
        "direction": "outbound",
        "color": COLOR_PURPLE,
        "text_pos": (780, 645)
    }
]

# ---------------------------------------------------------------------------
# Helper functions for Wrong-Way and Triple Riding Violations
# ---------------------------------------------------------------------------

WRONG_WAY_HISTORY_LEN = 24
WRONG_WAY_MIN_HISTORY = 12
WRONG_WAY_MIN_ACCUMULATED_PX = 35.0
WRONG_WAY_MIN_NET_PX = 45.0
WRONG_WAY_MIN_SPEED_KMH = 5.0
WRONG_WAY_CONFIRM_FRAMES = 9
WRONG_WAY_LANE_STABLE_FRAMES = 8
WRONG_WAY_QUEUE_AHEAD_PX = 130.0
WRONG_WAY_JITTER_PX = 3.0
WRONG_WAY_MIN_REVERSE_PX = 30.0

def get_expected_lane_direction(polygon, direction_str):
    """Derive normalized expected lane direction vector from a 4-point polygon."""
    bottom_center = (polygon[0] + polygon[1]) / 2.0
    top_center = (polygon[2] + polygon[3]) / 2.0
    
    if direction_str == "inbound":
        vec = bottom_center - top_center
    else:
        vec = top_center - bottom_center
        
    mag = math.sqrt(vec[0]**2 + vec[1]**2)
    if mag == 0:
        return (0.0, 1.0)
    return (vec[0]/mag, vec[1]/mag)

def get_camera_forward_lane_direction(polygon):
    """Expected travel direction in this camera view: far end of lane to near end."""
    bottom_center = (polygon[0] + polygon[1]) / 2.0
    top_center = (polygon[2] + polygon[3]) / 2.0
    vec = bottom_center - top_center
    mag = math.sqrt(vec[0]**2 + vec[1]**2)
    if mag == 0:
        return (0.0, 1.0)
    return (vec[0]/mag, vec[1]/mag)

def get_lane_index_by_polygon(cx, cy):
    """Test if bottom center point is inside any lane polygon. Returns lane ID or -1."""
    point = (float(cx), float(cy))
    for lane in LANES_CONFIG:
        dist = cv2.pointPolygonTest(lane["polygon"], point, False)
        if dist >= 0:
            return lane["id"]
    return -1

def get_lane_config(lane_id):
    for lane in LANES_CONFIG:
        if lane["id"] == lane_id:
            return lane
    return None

def project_to_fsd_coords(cx, cy):
    cy_norm = np.clip((cy - 240.0) / (850.0 - 240.0), 0.0, 1.0)
    Y_3D = 80.0 + (1.0 - cy_norm) ** 1.8 * 520.0
    left_edge = 390.0 + cy_norm * (120.0 - 390.0)
    right_edge = 710.0 + cy_norm * (960.0 - 710.0)
    road_width = max(right_edge - left_edge, 1.0)
    x_rel = (cx - left_edge) / road_width
    X_3D = (x_rel - 0.5) * 180.0
    return X_3D, Y_3D

def resolve_lane_index(cx, cy, X_3D):
    lane_idx = get_lane_index_by_polygon(cx, cy)
    if lane_idx != -1:
        return lane_idx
    if X_3D < -30:
        return 0
    if X_3D <= 30:
        return 1
    return 2

def check_lane_stability(track_id, current_lane, lane_history_dict):
    """Maintain history of assigned lanes and return the stable lane ID via majority vote.
    
    Requires Counter(history).most_common(1)[0][1] >= 12 inside a 15-frame window.
    """
    from collections import Counter
    history = lane_history_dict[track_id]
    history.append(current_lane if current_lane in [0, 1, 2] else None)
    
    if len(history) < 15:
        return None

    valid_lanes = [lane for lane in history if lane is not None]
    if len(valid_lanes) < 12:
        return None

    most_common = Counter(valid_lanes).most_common(1)[0]
    if most_common[1] >= 12:
        return most_common[0]
    return None

def create_wrong_way_state():
    return {
        "positions": deque(maxlen=WRONG_WAY_HISTORY_LEN),
        "accumulated_distance": 0.0,
        "current_lane": None,
        "lane_consistent_frames": 0,
        "consecutive_wrong_way": 0,
        "logged_wrong_way": False,
        "last_seen_frame": 0,
        "last_position": None,
        "status": "normal",
        "movement_direction": "UNKNOWN",
        "confidence": 0.0,
        "speed_kmh": 0.0,
    }

def build_vehicle_ahead_flags(vehicle_snapshots):
    """Mark vehicles with another tracked vehicle directly ahead in the same lane."""
    flags = {snap["track_id"]: False for snap in vehicle_snapshots}
    by_lane = defaultdict(list)

    for snap in vehicle_snapshots:
        lane = get_lane_config(snap["lane_idx"])
        if lane is None:
            continue
        expected = get_camera_forward_lane_direction(lane["polygon"])
        cx, cy = snap["point"]
        projection = (cx * expected[0]) + (cy * expected[1])
        by_lane[snap["lane_idx"]].append((projection, snap["track_id"]))

    for lane_items in by_lane.values():
        lane_items.sort(key=lambda item: item[0])
        for idx in range(len(lane_items) - 1):
            projection, track_id = lane_items[idx]
            next_projection, _ = lane_items[idx + 1]
            if 0 < next_projection - projection <= WRONG_WAY_QUEUE_AHEAD_PX:
                flags[track_id] = True

    return flags

def check_wrong_way(
    track_id,
    stable_lane,
    bottom_center,
    speed_kmh,
    frame_count,
    wrong_way_states,
    current_lane=None,
    has_vehicle_ahead=False
):
    """Tracker-stable, congestion-aware wrong-way detection."""
    state = wrong_way_states.setdefault(track_id, create_wrong_way_state())
    state["last_seen_frame"] = frame_count
    state["speed_kmh"] = speed_kmh

    if state["logged_wrong_way"]:
        state["status"] = "logged"
        state["consecutive_wrong_way"] = 0
        return {
            "confirmed": False,
            "status": "logged",
            "counter": 0,
            "movement_direction": state["movement_direction"],
            "confidence": state["confidence"],
        }

    if current_lane != state["current_lane"]:
        state["current_lane"] = current_lane
        state["lane_consistent_frames"] = 1 if current_lane in [0, 1, 2] else 0
        state["consecutive_wrong_way"] = 0
        state["accumulated_distance"] = 0.0
        state["positions"].clear()
        state["last_position"] = None
    elif current_lane in [0, 1, 2]:
        state["lane_consistent_frames"] += 1

    if state["last_position"] is not None:
        lx, ly = state["last_position"]
        step_dist = math.sqrt((bottom_center[0] - lx) ** 2 + (bottom_center[1] - ly) ** 2)
        if step_dist >= WRONG_WAY_JITTER_PX:
            state["accumulated_distance"] += step_dist

    state["last_position"] = bottom_center
    state["positions"].append(bottom_center)

    result = {
        "confirmed": False,
        "status": "normal",
        "counter": state["consecutive_wrong_way"],
        "movement_direction": "UNKNOWN",
        "confidence": 0.0,
    }

    if len(state["positions"]) >= 2:
        first = state["positions"][0]
        last = state["positions"][-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        result["movement_direction"] = f"dx={dx:.1f}, dy={dy:.1f}"
        state["movement_direction"] = result["movement_direction"]

    lane_ready = (
        stable_lane is not None
        and current_lane == stable_lane
        and state["lane_consistent_frames"] >= WRONG_WAY_LANE_STABLE_FRAMES
    )

    net_dist = 0.0
    if len(state["positions"]) >= 2:
        first = state["positions"][0]
        last = state["positions"][-1]
        net_dist = math.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)

    motion_ready = (
        len(state["positions"]) >= WRONG_WAY_MIN_HISTORY
        and state["accumulated_distance"] >= WRONG_WAY_MIN_ACCUMULATED_PX
        and net_dist >= WRONG_WAY_MIN_NET_PX
        and speed_kmh >= WRONG_WAY_MIN_SPEED_KMH
    )

    if not lane_ready:
        state["status"] = "normal"
        state["consecutive_wrong_way"] = 0
        return result

    if not motion_ready or has_vehicle_ahead:
        state["status"] = "observing" if len(state["positions"]) >= WRONG_WAY_MIN_HISTORY else "normal"
        state["consecutive_wrong_way"] = 0
        result["status"] = state["status"]
        return result

    lane = get_lane_config(stable_lane)
    if lane is None:
        state["status"] = "normal"
        state["consecutive_wrong_way"] = 0
        return result

    first = state["positions"][0]
    last = state["positions"][-1]
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    expected = get_camera_forward_lane_direction(lane["polygon"])
    movement_along_expected = (dx * expected[0]) + (dy * expected[1])
    confidence = min(1.0, abs(movement_along_expected) / max(WRONG_WAY_MIN_ACCUMULATED_PX * 2.0, 1.0))

    result["movement_direction"] = f"dx={dx:.1f}, dy={dy:.1f}"
    result["confidence"] = confidence
    state["movement_direction"] = result["movement_direction"]
    state["confidence"] = confidence

    if movement_along_expected <= -WRONG_WAY_MIN_REVERSE_PX:
        state["consecutive_wrong_way"] += 1
        state["status"] = "potential"
    else:
        state["consecutive_wrong_way"] = 0
        state["status"] = "normal"

    result["counter"] = state["consecutive_wrong_way"]
    result["status"] = state["status"]

    if state["consecutive_wrong_way"] >= WRONG_WAY_CONFIRM_FRAMES:
        state["status"] = "confirmed"
        result["status"] = "confirmed"
        result["confirmed"] = True

    return result

def check_triple_riding(track_id, veh_bbox, pedestrians, rider_history_dict):
    """Stably detect triple riding using tracker IDs over 5 consecutive frames.
    
    Requires overlapping count >= 3 for 5 consecutive frames, and intersection of IDs >= 2.
    """
    current_peds = set()
    for ped in pedestrians:
        ped_track_id = ped["track_id"]
        overlap = get_overlap_ratio(ped["bbox"], veh_bbox)
        if overlap > 0.25 or center_inside(ped["bbox"], veh_bbox):
            current_peds.add(ped_track_id)
            
    history = rider_history_dict[track_id]
    history.append(current_peds)
    
    if len(history) < 5:
        return False
        
    # 1. Count must be >= 3 in all 5 frames
    if not all(len(s) >= 3 for s in history):
        return False
        
    # 2. Intersection must have >= 2 persistent track IDs
    common_riders = set.intersection(*history)
    if len(common_riders) >= 2:
        return True
    return False

# Staged models cleanup tracker
staged_temp_dir = None
ocr_reader = None

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    db_path = "traffic_violations.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                track_id INTEGER,
                vehicle_type TEXT,
                violation_type TEXT,
                plate_number TEXT,
                frame_number INTEGER,
                evidence_path TEXT
            );
        """)
        conn.commit()
        conn.close()
        print(f"SQLite Database initialized at {db_path}")
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")

def load_staged_models():
    """Stage and load the friend's helmet and plate detector weights securely."""
    global staged_temp_dir
    helmet_src = Path("GRIDLOCK-ROUND-2-feature-helmet-violation-system/modules/helmet_violation_system/models/helmet_detector.pt")
    plate_src = Path("GRIDLOCK-ROUND-2-feature-helmet-violation-system/modules/helmet_violation_system/models/plate_detector.pt")
    
    helmet_model = None
    plate_model = None
    
    try:
        staged_temp_dir = Path(tempfile.mkdtemp(prefix="gridlock_"))
        
        if helmet_src.exists():
            helmet_safe = staged_temp_dir / "helmet_detector.pt"
            shutil.copy2(helmet_src, helmet_safe)
            print(f"Loaded helmet detector from safe stage: {helmet_safe}")
            helmet_model = YOLO(str(helmet_safe))
        else:
            print(f"WARNING: Helmet detector model not found at {helmet_src}")
            
        if plate_src.exists():
            plate_safe = staged_temp_dir / "plate_detector.pt"
            shutil.copy2(plate_src, plate_safe)
            print(f"Loaded plate detector from safe stage: {plate_safe}")
            plate_model = YOLO(str(plate_safe))
        else:
            print(f"WARNING: Plate detector model not found at {plate_src}")
            
    except Exception as e:
        print(f"Failed to stage models: {e}")
        
    return helmet_model, plate_model

# Load the staged models
print("Staging weights...")
helmet_model, plate_model = load_staged_models()

# Initialize EasyOCR reader
print("Initializing EasyOCR reader...")
try:
    use_gpu = torch.cuda.is_available()
    ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    print(f"EasyOCR initialized successfully (GPU: {use_gpu}).")
except Exception as e:
    print(f"WARNING: Failed to initialize EasyOCR: {e}")
    ocr_reader = None

# Helmet model classes from friend's train-9 dataset:
# 0 = bicyclist, 1 = driver, 2 = helmet, 3 = no-helmet
HELMET_CLS_BICYCLIST = 0
HELMET_CLS_DRIVER = 1
HELMET_CLS_HELMET = 2
HELMET_CLS_NO_HELMET = 3

# ---------------------------------------------------------------------------
# ZOOM-IN DETECTION APPROACH
# Instead of detecting pedestrians and associating them with vehicles,
# we CROP each vehicle, ZOOM/UPSCALE it, and run detection DIRECTLY on it.
# This works because YOLO can't see people inside cars from a traffic camera,
# but if we zoom into the car crop 3-4x, we can find the driver/passengers.
# ---------------------------------------------------------------------------

def zoom_crop_vehicle(frame, bbox, pad_ratio=0.15, min_size=480):
    """Crop a vehicle region from the frame with padding, then upscale for detection.
    
    Args:
        frame: Full frame image
        bbox: [x1, y1, x2, y2] bounding box of the vehicle
        pad_ratio: Extra padding as ratio of box dimensions
        min_size: Minimum dimension of the output crop
    
    Returns:
        Zoomed/upscaled crop of the vehicle region
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    
    # Add padding around the vehicle
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(fw, x2 + pad_x)
    cy2 = min(fh, y2 + pad_y)
    
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    
    # Upscale to at least min_size on the largest dimension
    ch, cw = crop.shape[:2]
    max_dim = max(ch, cw)
    if max_dim < min_size:
        scale = min_size / max_dim
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return crop

def check_helmet_on_bike(zoomed_crop, track_id):
    """Run the friend's helmet model on a ZOOMED vehicle crop.
    
    The model detects: 0=bicyclist, 1=driver, 2=helmet, 3=no-helmet
    We upscale the bike region so the model can see helmet details clearly.
    """
    if helmet_model is None or zoomed_crop is None or zoomed_crop.size == 0:
        return "UNKNOWN"
    try:
        h, w = zoomed_crop.shape[:2]
        # Ensure minimum 640px for the helmet model to work well
        min_dim = max(h, w)
        if min_dim < 640:
            scale = 640.0 / min_dim
            zoomed_crop = cv2.resize(zoomed_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        h_res = helmet_model(zoomed_crop, conf=0.30, verbose=False)
        if len(h_res) > 0 and h_res[0].boxes is not None:
            boxes = h_res[0].boxes
            clss = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            
            found_helmet = False
            found_no_helmet = False
            found_rider = False
            helmet_conf = 0.0
            no_helmet_conf = 0.0
            
            for cls_val, conf_val in zip(clss, confs):
                cls_int = int(cls_val)
                if cls_int == HELMET_CLS_NO_HELMET:
                    found_no_helmet = True
                    no_helmet_conf = max(no_helmet_conf, conf_val)
                elif cls_int == HELMET_CLS_HELMET:
                    found_helmet = True
                    helmet_conf = max(helmet_conf, conf_val)
                elif cls_int in [HELMET_CLS_BICYCLIST, HELMET_CLS_DRIVER]:
                    found_rider = True
            
            # Priority: no-helmet detection is a clear violation
            if found_no_helmet and no_helmet_conf > 0.30:
                return "VIOLATION"
            if found_helmet and helmet_conf > 0.30:
                return "OK"
            # Rider detected but no helmet class found → assume no helmet
            if found_rider:
                return "VIOLATION"
        
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def check_seatbelt_zoomed(zoomed_car_crop, yolo_model, track_id, device="cpu"):
    """Zoom into car and check seatbelt using YOLO person detection & torso edge-heuristics.
    
    Returns: ("OK", n_persons) or ("VIOLATION", n_persons) or ("SCANNING", 0)
    """
    if zoomed_car_crop is None or zoomed_car_crop.size == 0:
        return "SCANNING", 0
    
    try:
        h, w = zoomed_car_crop.shape[:2]
        # Ensure large enough for detection models
        min_dim = max(h, w)
        if min_dim < 640:
            scale = 640.0 / min_dim
            zoomed_car_crop = cv2.resize(zoomed_car_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h, w = zoomed_car_crop.shape[:2]
            
        # Run YOLO person detection on zoomed car crop
        person_results = yolo_model(zoomed_car_crop, conf=0.20, classes=[0], verbose=False)
        
        person_boxes = []
        if len(person_results) > 0 and person_results[0].boxes is not None:
            boxes = person_results[0].boxes
            if boxes.xyxy is not None and len(boxes.xyxy) > 0:
                person_boxes = boxes.xyxy.int().cpu().tolist()
        
        if not person_boxes:
            return "SCANNING", 0
        
        # For each detected person, check seatbelt status
        n_persons = len(person_boxes)
        seatbelt_found_count = 0
        no_seatbelt_count = 0
        
        for px1, py1, px2, py2 in person_boxes:
            # Crop the person from the zoomed car image
            person_crop = zoomed_car_crop[max(0,py1):min(h,py2), max(0,px1):min(w,px2)]
            if person_crop.size == 0:
                continue
            
            ph, pw = person_crop.shape[:2]
            # Upscale person crop further for seatbelt detail
            if max(ph, pw) < 200:
                pscale = 200.0 / max(ph, pw)
                person_crop = cv2.resize(person_crop, None, fx=pscale, fy=pscale, interpolation=cv2.INTER_CUBIC)
                ph, pw = person_crop.shape[:2]
            
            # Analyze torso region for seatbelt strap
            torso_y1 = int(ph * 0.10)
            torso_y2 = int(ph * 0.80)
            torso_x1 = int(pw * 0.05)
            torso_x2 = int(pw * 0.95)
            torso = person_crop[torso_y1:torso_y2, torso_x1:torso_x2]
            
            if torso.size == 0:
                continue
            
            belt_score = 0
            
            # Feature 1: Diagonal line detection (Hough)
            gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
            clahe_sb = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe_sb.apply(gray)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 25, 100)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=8, maxLineGap=6)
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    length = math.sqrt((lx2-lx1)**2 + (ly2-ly1)**2)
                    angle = math.degrees(math.atan2(abs(ly2 - ly1), abs(lx2 - lx1)))
                    if 20 < angle < 70 and length > 6:
                        belt_score += 1
            
            # Feature 2: Dark diagonal band detection
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            dark_mask = cv2.inRange(hsv, (0, 0, 10), (180, 80, 100))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 5:
                    rect = cv2.minAreaRect(cnt)
                    (_, (rw, rh), rect_angle) = rect
                    aspect = max(rw, rh) / max(min(rw, rh), 1)
                    if aspect > 2.0 and max(rw, rh) > 10:
                        norm_angle = rect_angle % 180
                        if 15 < norm_angle < 75 or 105 < norm_angle < 165:
                            belt_score += 1
            
            # Feature 3: Gradient orientation analysis
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx**2 + sobely**2)
            angle_map = np.degrees(np.arctan2(np.abs(sobely), np.abs(sobelx)))
            strong_mask = mag > np.percentile(mag, 70)
            if np.any(strong_mask):
                diag_angles = angle_map[strong_mask]
                diag_ratio = np.sum((diag_angles > 20) & (diag_angles < 70)) / max(len(diag_angles), 1)
                if diag_ratio > 0.20:
                    belt_score += 1
            
            if belt_score >= 2:
                seatbelt_found_count += 1
            else:
                no_seatbelt_count += 1
        
        if no_seatbelt_count > 0:
            return "VIOLATION", n_persons
        elif seatbelt_found_count > 0:
            return "OK", n_persons
        else:
            return "SCANNING", n_persons
            
    except Exception:
        return "SCANNING", 0

def run_lpr_on_vehicle(frame, bbox, track_id, class_name, frame_count):
    """Run plate detection and OCR on a vehicle crop."""
    if plate_model is None or ocr_reader is None:
        return None
        
    try:
        # Crop vehicle
        zoomed = zoom_crop_vehicle(frame, bbox, pad_ratio=0.10, min_size=640)
        if zoomed is None:
            return None
            
        # Detect plate
        plate_results = plate_model(zoomed, conf=0.45, verbose=False)
        if len(plate_results) == 0 or plate_results[0].boxes is None or len(plate_results[0].boxes) == 0:
            return None
            
        # Get highest confidence plate box
        best_box = None
        best_conf = 0.0
        for box in plate_results[0].boxes:
            conf = float(box.conf[0].item())
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().tolist()
                
        if best_box is None:
            return None
            
        # Crop plate from zoomed vehicle crop
        px1, py1, px2, py2 = map(int, best_box)
        plate_crop = zoomed[max(0, py1):min(zoomed.shape[0], py2), max(0, px1):min(zoomed.shape[1], px2)]
        if plate_crop.size == 0:
            return None
            
        # Preprocess plate crop for OCR (similar to friend's code)
        plate_crop = cv2.resize(plate_crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR
        ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ocr_results = ocr_reader.readtext(thresh, detail=0, paragraph=True, allowlist=ALLOWED_CHARS)
        
        if ocr_results:
            raw_text = "".join(ocr_results).strip()
            # Normalize
            normalized = raw_text.upper().replace(" ", "")
            normalized = normalized.replace("O", "0").replace("I", "1").replace("S", "5")
            
            # Basic plate format check: must have at least 4 characters
            if len(normalized) >= 4:
                return normalized
                
        return None
    except Exception as e:
        print(f"LPR error on vehicle {track_id}: {e}")
        return None

def build_evidence_image(frame_img, viol, timestamp_text):
    """Create an evidence frame with full scene, vehicle crop, and violation metadata."""
    bbox = viol.get("bbox")
    if bbox is None:
        return frame_img

    h, w = frame_img.shape[:2]
    panel_w = 360
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame_img
    canvas[:, w:] = (18, 18, 18)
    cv2.rectangle(canvas, (w, 0), (w + panel_w - 1, h - 1), COLOR_CYAN, 1)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame_img[y1:y2, x1:x2]

    panel_x = w + 18
    panel_y = 28
    cv2.putText(canvas, "EVIDENCE", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, timestamp_text, (panel_x, panel_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_CYAN, 1, cv2.LINE_AA)

    if crop.size > 0:
        max_crop_w = panel_w - 36
        max_crop_h = 240
        ch, cw = crop.shape[:2]
        scale = min(max_crop_w / max(cw, 1), max_crop_h / max(ch, 1))
        resized = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        crop_y = panel_y + 54
        canvas[crop_y:crop_y + rh, panel_x:panel_x + rw] = resized
        cv2.rectangle(canvas, (panel_x, crop_y), (panel_x + rw, crop_y + rh), COLOR_YELLOW, 1)
        meta_y = crop_y + rh + 30
    else:
        meta_y = panel_y + 90

    lane_idx = viol.get("lane_idx")
    lane_text = f"L{lane_idx + 1}" if lane_idx is not None else "UNKNOWN"
    confidence = viol.get("wrong_way_confidence", 0.0)
    metadata_lines = [
        f"Track ID: {viol.get('track_id', 'UNKNOWN')}",
        f"Violation: {viol.get('violation_type', 'UNKNOWN')}",
        f"Lane: {lane_text}",
        f"Direction: {viol.get('wrong_way_direction', 'UNKNOWN')}",
        f"Confidence: {confidence:.2f}",
        f"Plate: {viol.get('plate_number', 'UNKNOWN')}",
    ]

    for line in metadata_lines:
        cv2.putText(canvas, line, (panel_x, meta_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
        meta_y += 24

    return canvas

# ---------------------------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------------------------
def main():
    # Detect videos in local directory
    videos_dir = os.path.join(".", "videos")
    patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for pat in patterns:
        video_files.extend(glob.glob(os.path.join(videos_dir, pat)))
    
    video_files.sort()
    
    if not video_files:
        print("ERROR: No video files found in ./videos directory!")
        sys.exit(1)
        
    import random
    current_video_idx = random.randint(0, len(video_files) - 1)
    video_path = video_files[current_video_idx]
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        for path in video_files:
            if arg.lower() in os.path.basename(path).lower():
                video_path = path
                current_video_idx = video_files.index(path)
                break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
        
    print(f"\n=======================================================")
    print("AI TRAFFIC GUARDIAN & TESLA FSD 3D MAP ENGINE")
    print("-------------------------------------------------------")
    print(f"Device fallback: {device.upper()}")
    print(f"Staged Helmet Model: {'LOADED' if helmet_model is not None else 'FAILED'}")
    print(f"Available Videos in workspace:")
    for idx, path in enumerate(video_files):
        print(f"  [{idx + 1}] {os.path.basename(path)}")
    print("-------------------------------------------------------")
    print("KEYBOARD CONTROLS (OpenCV windows active):")
    print("  Q : Quit Visualizer")
    print("  P : Pause / Resume Video")
    print("  C : Toggle CLAHE Contrast Stretch")
    print("  G : Toggle Gamma Correction")
    print("  S : Toggle Shadow Reduction")
    print("  H : Toggle Laplacian Sharpening")
    print("  D : Toggle Bilateral Denoising")
    print("  W : Toggle White Balance")
    print("  1-7 : Hot-switch videos instantly")
    print("=======================================================\n")
    
    print("Loading YOLO Model...")
    model = YOLO("yolov8n.pt")
    print("YOLO Model loaded successfully.")
    
    # Preprocessing toggles
    preproc_status = {
        "clahe": True,
        "gamma": True,
        "denoise": False,
        "sharpen": True,
        "shadow": True,
        "dehaze": False,
        "wb": False
    }
    
    active_imgsz = 640
    
    # Initialize UI Windows
    cam_win = "Camera View - AI Traffic Guardian"
    map_win = "Tesla FSD 3D Map"
    
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cam_win, 1024, 768)
    
    cv2.namedWindow(map_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(map_win, 800, 800)
    
    init_db()
    phase_mgr = TrafficPhaseManager()
    visualizer_3d = FSD3DVisualizer(800, 800)
    
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
    last_helmet_detections = []  # Full-frame helmet model detections (persisted across frames)
    processed_plates = {}        # Cache of recognized license plates
    ocr_attempts = {}            # Track LPR attempts per vehicle/rider to save performance
    logged_violations = defaultdict(set)  # Track which violation types have been logged per track ID
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    
    is_paused = False
    t_start = time.time()
    frame_count = 0
    
    try:
        while True:
            t_frame_start = time.perf_counter()
            
            # Keyboard interaction check
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('p') or key == ord('P'):
                is_paused = not is_paused
                print(f"Playback {'PAUSED' if is_paused else 'RESUMED'}")
            elif key == ord('c') or key == ord('C'):
                preproc_status["clahe"] = not preproc_status["clahe"]
            elif key == ord('g') or key == ord('G'):
                preproc_status["gamma"] = not preproc_status["gamma"]
            elif key == ord('s') or key == ord('S'):
                preproc_status["shadow"] = not preproc_status["shadow"]
            elif key == ord('h') or key == ord('H'):
                preproc_status["sharpen"] = not preproc_status["sharpen"]
            elif key == ord('d') or key == ord('D'):
                preproc_status["denoise"] = not preproc_status["denoise"]
            elif key == ord('w') or key == ord('W'):
                preproc_status["wb"] = not preproc_status["wb"]
            elif ord('1') <= key <= ord('7'):
                selected_idx = key - ord('1')
                if selected_idx < len(video_files):
                    current_video_idx = selected_idx
                    video_path = video_files[current_video_idx]
                    print(f"\n🔄 Switching Video to: {os.path.basename(video_path)}")
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_delay = 1.0 / fps
                    trajectories.clear()
                    speed_store.clear()
                    last_positions_3d.clear()
                    lane_history.clear()
                    rider_history.clear()
                    wrong_way_states.clear()
                    helmet_status_cache.clear()
                    seatbelt_status_cache.clear()
                    processed_plates.clear()
                    ocr_attempts.clear()
                    logged_violations.clear()
                    t_start = time.time()
                    frame_count = 0
                    
            if is_paused:
                time.sleep(0.05)
                continue
                
            ret, frame = cap.read()
            if not ret:
                import random
                current_video_idx = random.randint(0, len(video_files) - 1)
                video_path = video_files[current_video_idx]
                print(f"\n🔄 Video ended. Switching to random Video: {os.path.basename(video_path)}")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_delay = 1.0 / fps
                trajectories.clear()
                speed_store.clear()
                last_positions_3d.clear()
                lane_history.clear()
                rider_history.clear()
                wrong_way_states.clear()
                helmet_status_cache.clear()
                seatbelt_status_cache.clear()
                processed_plates.clear()
                ocr_attempts.clear()
                logged_violations.clear()
                last_helmet_detections = []
                t_start = time.time()
                frame_count = 0
                continue
                
            frame_count += 1
            phase_mgr.update()
            
            # Enhance frame
            enhanced_frame = enhance_frame(frame, preproc_status)
            enhanced_frame = cv2.resize(enhanced_frame, (1024, 1024))
            
            # Run YOLO Tracking
            results = model.track(source=enhanced_frame, persist=True, conf=0.25, iou=0.5,
                                  classes=[0, 1, 2, 3, 5, 7], device=device, verbose=False)
            
            boxes = results[0].boxes if (results and results[0].boxes is not None) else None
            
            lane_counts = [0, 0, 0]  # L1, L2, L3 vehicle counts
            lane_pressures = [0.0, 0.0, 0.0]
            
            # Draw overlay polygons on Window 1
            overlay = enhanced_frame.copy()
            for lane in LANES_CONFIG:
                cv2.fillPoly(overlay, [lane["polygon"]], lane["color"])
            cv2.addWeighted(overlay, 0.22, enhanced_frame, 0.78, 0, enhanced_frame)
            
            # Lanes Boundaries
            cv2.line(enhanced_frame, (400, 850), (460, 240), COLOR_CYAN, 1, cv2.LINE_AA)
            cv2.line(enhanced_frame, (680, 850), (550, 240), COLOR_CYAN, 1, cv2.LINE_AA)
            for lane in LANES_CONFIG:
                cv2.putText(enhanced_frame, lane["name"], lane["text_pos"], cv2.FONT_HERSHEY_SIMPLEX, 0.40, lane["color"], 1, cv2.LINE_AA)
            

            
            # Crosswalk (Y = 750)
            draw_crosswalk(enhanced_frame, 750, 780, 180, 920)
            
            # Map 3D Visualizer Ground
            map_img = np.zeros((800, 800, 3), dtype=np.uint8)
            visualizer_3d.draw_road(map_img)
            
            current_ids = set()
            pedestrians = []
            vehicles = []
            
            # Parse tracked bounding boxes
            if boxes is not None and boxes.id is not None:
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
            

            # =====================================================================
            # ZOOM-IN VIOLATION DETECTION (No pedestrian-vehicle association needed)
            # For each vehicle: CROP → ZOOM/UPSCALE → RUN DETECTION DIRECTLY
            # =====================================================================
            violating_vehicles = set()
            total_violations = 0
            
            vehicle_snapshots = []
            vehicle_snapshot_by_id = {}
            for veh in vehicles:
                track_id = veh["track_id"]
                x1, y1, x2, y2 = veh["bbox"]
                cx = (x1 + x2) // 2
                cy = y2
                X_3D, Y_3D = project_to_fsd_coords(cx, cy)
                polygon_lane_idx = get_lane_index_by_polygon(cx, cy)
                lane_idx = resolve_lane_index(cx, cy, X_3D)
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

            vehicle_ahead_flags = build_vehicle_ahead_flags(vehicle_snapshots)

            vehicle_data = []
            for veh in vehicles:
                track_id = veh["track_id"]
                x1, y1, x2, y2 = veh["bbox"]
                snapshot = vehicle_snapshot_by_id[track_id]
                cx, cy = snapshot["point"]
                
                # Append vehicle coordinates to trajectory at the beginning of the loop
                trajectories[track_id].append((cx, cy))
                
                # 3D Math projection
                X_3D = snapshot["X_3D"]
                Y_3D = snapshot["Y_3D"]
                
                # Speed calculation (kept for pressure simulation)
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
                
                # Retrieve current lane index with polygon containment (fallback to X_3D heuristic)
                lane_idx = snapshot["lane_idx"]
                polygon_lane_idx = snapshot["polygon_lane_idx"]
                        
                # Determine stable lane assignment via Counter majority vote
                stable_lane = check_lane_stability(track_id, polygon_lane_idx, lane_history)
                
                # Perform wrong-side driving detection
                wrong_way_result = check_wrong_way(
                    track_id=track_id,
                    stable_lane=stable_lane,
                    bottom_center=(cx, cy),
                    speed_kmh=speed_kmh,
                    frame_count=frame_count,
                    wrong_way_states=wrong_way_states,
                    current_lane=polygon_lane_idx,
                    has_vehicle_ahead=vehicle_ahead_flags.get(track_id, False)
                )
                is_wrong_way = wrong_way_result["confirmed"]
                
                # Perform triple-riding detection on motorcycles/bicycles
                is_triple_riding = False
                if veh["class_name"] in ["motorcycle", "bicycle"]:
                    is_triple_riding = check_triple_riding(
                        track_id=track_id,
                        veh_bbox=veh["bbox"],
                        pedestrians=pedestrians,
                        rider_history_dict=rider_history
                    )
                    
                lane_counts[lane_idx] += 1
                lane_pressures[lane_idx] += 0.03 + (speed_kmh * 0.002)
                
                # =============================================================
                # DIRECT ZOOM-IN VIOLATION CHECK (no pedestrian association)
                # =============================================================
                is_violating = False
                violation_type = ""  # "HELMET" or "SEATBELT"
                violation_status = ""  # "OK", "VIOLATION", or ""
                n_persons_found = 0
                
                # Only run detection every N frames per vehicle (cache results)
                should_check = (frame_count % 6 == (track_id % 6))
                
                if veh["class_name"] in ["motorcycle", "bicycle"]:
                    # ---- HELMET CHECK: Zoom into bike, run helmet model ----
                    violation_type = "HELMET"
                    if should_check or track_id not in helmet_status_cache:
                        zoomed = zoom_crop_vehicle(enhanced_frame, veh["bbox"], pad_ratio=0.20, min_size=640)
                        if zoomed is not None:
                            status = check_helmet_on_bike(zoomed, track_id)
                            if status != "UNKNOWN":
                                helmet_status_cache[track_id] = status
                            elif track_id not in helmet_status_cache:
                                helmet_status_cache[track_id] = "SCANNING"
                    
                    cached = helmet_status_cache.get(track_id, "SCANNING")
                    violation_status = cached
                    if cached == "VIOLATION":
                        is_violating = True
                        
                elif veh["class_name"] in ["car", "bus", "truck", "suv"]:
                    # ---- SEATBELT CHECK: Zoom into car, find persons, check seatbelt ----
                    violation_type = "SEATBELT"
                    if should_check or track_id not in seatbelt_status_cache:
                        zoomed = zoom_crop_vehicle(enhanced_frame, veh["bbox"], pad_ratio=0.10, min_size=640)
                        if zoomed is not None:
                            status, n_found = check_seatbelt_zoomed(zoomed, model, track_id, device=device)
                            n_persons_found = n_found
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

            # =====================================================================
            # FULL-FRAME HELMET MODEL SCAN
            # The friend's model detects bicyclists/drivers/helmets/no-helmets
            # DIRECTLY on the full frame — no need for YOLO to detect bikes first.
            # This is how the friend's original code works: helmet_model(frame)
            # =====================================================================
            helmet_frame_detections = []  # List of dicts with bbox, class, conf
            
            if helmet_model is not None and (frame_count % 5 == 0):
                try:
                    # Run on full enhanced frame
                    h_results = helmet_model(enhanced_frame, conf=0.30, verbose=False)
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
                except Exception as e:
                    pass  # Don't crash if helmet model fails
            
            # Cache the last helmet frame detections for drawing on non-scan frames
            if frame_count % 5 == 0:
                last_helmet_detections = helmet_frame_detections

            # =====================================================================
            # ASSOCIATE FULL-FRAME HELMET DETECTIONS & COMPUTE TOTAL VIOLATIONS
            # =====================================================================
            total_violations = 0
            associated_no_helmets = set()
            violating_pedestrians = set()

            # Pass 1: Associate with vehicles (motorcycles/bicycles)
            for v_info in vehicle_data:
                if v_info["class_name"] in ["motorcycle", "bicycle"]:
                    v_box = v_info["bbox"]
                    has_no_helmet_overlap = False
                    has_helmet_overlap = False
                    
                    for idx, h_det in enumerate(last_helmet_detections):
                        h_box = h_det["bbox"]
                        h_cls = h_det["cls_id"]
                        
                        overlap = get_overlap_ratio(h_box, v_box)
                        is_inside = center_inside(h_box, v_box)
                        
                        if overlap > 0.25 or is_inside:
                            if h_cls == HELMET_CLS_NO_HELMET:
                                has_no_helmet_overlap = True
                                associated_no_helmets.add(idx)
                            elif h_cls == HELMET_CLS_HELMET:
                                has_helmet_overlap = True
                                
                    if v_info["violation_status"] != "VIOLATION":
                        if has_no_helmet_overlap:
                            v_info["is_violating"] = True
                            v_info["violation_type"] = "HELMET"
                            v_info["violation_status"] = "VIOLATION"
                            violating_vehicles.add(v_info["track_id"])
                        elif has_helmet_overlap:
                            v_info["is_violating"] = False
                            v_info["violation_type"] = "HELMET"
                            v_info["violation_status"] = "OK"

            # Pass 2: Associate with pedestrians (who are actually riders on bikes)
            for ped in pedestrians:
                ped_box = ped["bbox"]
                
                # Check if pedestrian overlaps with any vehicle (e.g. they are in a car or on a detected bike)
                is_inside_vehicle = False
                for v_info in vehicle_data:
                    overlap_with_vehicle = get_overlap_ratio(ped_box, v_info["bbox"])
                    if overlap_with_vehicle > 0.3 or center_inside(ped_box, v_info["bbox"]):
                        is_inside_vehicle = True
                        break
                        
                if is_inside_vehicle:
                    ped["is_inside_vehicle"] = True
                    continue
                
                ped["is_inside_vehicle"] = False
                has_no_helmet_overlap = False
                is_rider = False
                
                for idx, h_det in enumerate(last_helmet_detections):
                    h_box = h_det["bbox"]
                    h_cls = h_det["cls_id"]
                    
                    overlap = get_overlap_ratio(h_box, ped_box)
                    is_inside = center_inside(h_box, ped_box)
                    
                    if overlap > 0.25 or is_inside:
                        if h_cls == HELMET_CLS_NO_HELMET:
                            has_no_helmet_overlap = True
                            associated_no_helmets.add(idx)
                        if h_cls in [HELMET_CLS_BICYCLIST, HELMET_CLS_DRIVER, HELMET_CLS_HELMET, HELMET_CLS_NO_HELMET]:
                            is_rider = True
                        
                ped["is_rider"] = is_rider
                if has_no_helmet_overlap:
                    violating_pedestrians.add(ped["track_id"])

            # Aggregation Pass: Combine all active violations for each vehicle
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                class_name = v_info["class_name"]
                active_viols = []
                
                # 1. Check wrong way
                if v_info.get("is_wrong_way", False):
                    active_viols.append("WRONG WAY")
                    
                # 2. Check triple riding
                if v_info.get("is_triple_riding", False):
                    active_viols.append("TRIPLE RIDING")
                    
                # 3. Check helmet violation (zoom-in or full-frame association)
                if class_name in ["motorcycle", "bicycle"]:
                    if v_info.get("violation_status") == "VIOLATION":
                        active_viols.append("HELMET VIOLATION")
                # 4. Check seatbelt violation (zoom-in)
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
                    # Retain original OK or SCANNING status
                    if class_name in ["motorcycle", "bicycle"]:
                        v_info["violation_status"] = helmet_status_cache.get(track_id, "SCANNING")
                    else:
                        v_info["violation_status"] = seatbelt_status_cache.get(track_id, ("SCANNING", 0))[0]

            # Pass 3: Compute total violations
            # 3.1. Violating vehicles
            for v_info in vehicle_data:
                if v_info["is_violating"]:
                    total_violations += 1
            # 3.2. Violating pedestrians
            total_violations += len(violating_pedestrians)
            # 3.3. Unassociated NO-HELMET detections
            for idx, h_det in enumerate(last_helmet_detections):
                if h_det["cls_id"] == HELMET_CLS_NO_HELMET and idx not in associated_no_helmets:
                    total_violations += 1

            # Draw Vehicles (Window 1 & Window 2)
            for v_info in vehicle_data:
                track_id = v_info["track_id"]
                x1, y1, x2, y2 = v_info["bbox"]
                class_name = v_info["class_name"]
                lane_idx = v_info["lane_idx"]
                is_violating = v_info["is_violating"]
                v_type = v_info["violation_type"]
                v_status = v_info["violation_status"]
                n_persons = v_info["n_persons"]
                wrong_way_status = v_info.get("wrong_way_status", "normal")
                
                # Wrong-way observation colors: green, yellow, orange, red.
                if is_violating:
                    color = COLOR_RED
                elif wrong_way_status == "potential":
                    color = COLOR_ORANGE
                elif wrong_way_status == "observing":
                    color = COLOR_YELLOW
                else:
                    color = COLOR_GREEN_BRIGHT
                
                # Draw on Camera View — thicker border for violations
                thickness = 3 if is_violating else 2
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                
                # Double glow border for violations (pulsing effect)
                if is_violating:
                    cv2.rectangle(enhanced_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 180), 1, cv2.LINE_AA)
                
                # Laser line vector for wrong-way vehicles (similar to CSIN overhead view)
                is_wrong_way = v_info.get("is_wrong_way", False)
                if is_wrong_way:
                    strobe_color = (0, 69, 255) if int(time.time() * 6) % 2 == 0 else (0, 0, 255)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.line(enhanced_frame, (enhanced_frame.shape[1] // 2, 0), (cx, cy), strobe_color, 2, cv2.LINE_AA)
                    cv2.circle(enhanced_frame, (cx, cy), 6, strobe_color, -1, cv2.LINE_AA)
                
                # Trajectory trails
                pts = list(trajectories[track_id])
                n_pts = len(pts)
                for idx, pt in enumerate(pts):
                    alpha = (idx + 1) / max(n_pts, 1)
                    dot_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(enhanced_frame, pt, max(2, int(4 * alpha)), dot_color, -1, cv2.LINE_AA)
                    
                cv2.circle(enhanced_frame, ((x1+x2)//2, y2), 4, color, -1, cv2.LINE_AA)
                
                # Build label with violation status
                if is_violating:
                    label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | VIOLATIONS: {v_type}]"
                else:
                    if class_name in ["motorcycle", "bicycle"]:
                        label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | HELMET: {v_status}]"
                    else:
                        label_text = f"{class_name.upper()}#{track_id} [L{lane_idx+1} | SEATBELT: {v_status}]"
                
                # Append license plate number if available in cache
                cached_plate = processed_plates.get(track_id)
                if cached_plate:
                    label_text += f" | PLATE: {cached_plate}"
                
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                pill_x1, pill_y1 = x1, max(0, y1 - th - baseline - 6)
                pill_x2, pill_y2 = x1 + tw + 8, y1
                cv2.rectangle(enhanced_frame, (pill_x1, pill_y1), (pill_x2, pill_y2), COLOR_BLACK, -1)
                
                # Color-coded top bar: green=OK, red=VIOLATION, cyan=scanning
                bar_color = color
                if v_status == "OK":
                    bar_color = COLOR_GREEN_BRIGHT
                elif v_status == "VIOLATION":
                    bar_color = COLOR_RED
                elif v_status == "SCANNING":
                    bar_color = COLOR_CYAN
                cv2.rectangle(enhanced_frame, (pill_x1, pill_y1), (pill_x2, pill_y1 + 2), bar_color, -1)
                
                cv2.putText(enhanced_frame, label_text, (pill_x1 + 4, pill_y2 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1, cv2.LINE_AA)
                
                # Draw on 3D Map
                visualizer_3d.draw_vehicle_3d(map_img, v_info["X_3D"], v_info["Y_3D"], class_name, track_id, color)

            # =====================================================================
            # DRAW HELMET MODEL FULL-FRAME DETECTIONS
            # These are direct detections from the friend's model on the full frame
            # =====================================================================
            for h_det in last_helmet_detections:
                hx1, hy1, hx2, hy2 = h_det["bbox"]
                h_cls = h_det["cls_id"]
                h_name = h_det["cls_name"]
                h_conf = h_det["conf"]
                
                # Color based on detection class
                if h_cls == HELMET_CLS_NO_HELMET:
                    h_color = (0, 0, 255)    # RED for no helmet
                    h_label = f"NO-HELMET ({h_conf:.0%})"
                elif h_cls == HELMET_CLS_HELMET:
                    h_color = (0, 255, 0)    # GREEN for helmet
                    h_label = f"HELMET ({h_conf:.0%})"
                elif h_cls == HELMET_CLS_BICYCLIST:
                    h_color = (255, 165, 0)  # ORANGE for bicyclist
                    h_label = f"BICYCLIST ({h_conf:.0%})"
                elif h_cls == HELMET_CLS_DRIVER:
                    h_color = (255, 255, 0)  # CYAN for driver
                    h_label = f"RIDER ({h_conf:.0%})"
                else:
                    continue
                
                # Draw detection box
                cv2.rectangle(enhanced_frame, (hx1, hy1), (hx2, hy2), h_color, 2, cv2.LINE_AA)
                
                # Violation flash for no-helmet
                if h_cls == HELMET_CLS_NO_HELMET:
                    cv2.rectangle(enhanced_frame, (hx1-3, hy1-3), (hx2+3, hy2+3), (0, 0, 180), 2, cv2.LINE_AA)
                
                # Label background
                (ltw, lth), lb = cv2.getTextSize(h_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                lbg_y1 = max(0, hy1 - lth - lb - 8)
                lbg_y2 = hy1
                cv2.rectangle(enhanced_frame, (hx1, lbg_y1), (hx1 + ltw + 10, lbg_y2), (0, 0, 0), -1)
                cv2.rectangle(enhanced_frame, (hx1, lbg_y1), (hx1 + ltw + 10, lbg_y1 + 3), h_color, -1)
                cv2.putText(enhanced_frame, h_label, (hx1 + 5, lbg_y2 - lb - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, h_color, 2, cv2.LINE_AA)

            # Draw Pedestrians (Window 1 & Window 2)
            for ped in pedestrians:
                if ped.get("is_inside_vehicle", False):
                    continue
                track_id = ped["track_id"]
                x1, y1, x2, y2 = ped["bbox"]
                cx = (x1 + x2) // 2
                cy = y2
                
                # Determine Lane index
                cy_norm = np.clip((cy - 240.0) / (850.0 - 240.0), 0.0, 1.0)
                left_edge = 390.0 + cy_norm * (120.0 - 390.0)
                right_edge = 710.0 + cy_norm * (960.0 - 710.0)
                road_width = max(right_edge - left_edge, 1.0)
                x_rel = (cx - left_edge) / road_width
                X_3D = (x_rel - 0.5) * 180.0
                lane_idx = 0 if X_3D < -30 else (1 if X_3D <= 30 else 2)
                
                is_violating = track_id in violating_pedestrians
                is_rider = ped.get("is_rider", False)
                class_lbl = "RIDER" if is_rider else "PEDESTRIAN"
                
                if is_violating:
                    color = COLOR_RED
                    label_text = f"{class_lbl}#{track_id} [L{lane_idx+1} | HELMET: VIOLATION]"
                else:
                    color = (255, 100, 255)  # magenta for pedestrians
                    label_text = f"{class_lbl}#{track_id} [Lane {lane_idx+1}]"
                    
                # Append license plate number if available in cache
                cached_plate = processed_plates.get(track_id)
                if cached_plate:
                    label_text += f" | PLATE: {cached_plate}"
                    
                # Draw pedestrian bbox on Window 1
                thickness = 3 if is_violating else 2
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
                cv2.circle(enhanced_frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
                
                # Double glow border for violations (pulsing effect)
                if is_violating:
                    cv2.rectangle(enhanced_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 180), 1, cv2.LINE_AA)
                
                # Trajectories for pedestrians
                trajectories[track_id].append((cx, cy))
                pts = list(trajectories[track_id])
                for idx, pt in enumerate(pts):
                    alpha = (idx + 1) / max(len(pts), 1)
                    dot_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(enhanced_frame, pt, max(2, int(3 * alpha)), dot_color, -1, cv2.LINE_AA)
                
                # BBox labels
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                pill_x1, pill_y1 = x1, max(0, y1 - th - baseline - 6)
                pill_x2, pill_y2 = x1 + tw + 8, y1
                cv2.rectangle(enhanced_frame, (pill_x1, pill_y1), (pill_x2, pill_y2), COLOR_BLACK, -1)
                
                bar_color = COLOR_RED if is_violating else (255, 100, 255)
                cv2.rectangle(enhanced_frame, (pill_x1, pill_y1), (pill_x2, pill_y1 + 2), bar_color, -1)
                cv2.putText(enhanced_frame, label_text, (pill_x1 + 4, pill_y2 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)
                            
                # Draw on 3D Map
                visualizer_3d.draw_vehicle_3d(map_img, X_3D, Y_3D, "pedestrian", track_id, color)

            # =====================================================================
            # LPR AND DATABASE LOGGING PASS (WITH EVIDENCE SNAPSHOTS)
            # =====================================================================
            new_violations_logged = []
            
            # Check violating vehicles
            for v_info in vehicle_data:
                if v_info["is_violating"]:
                    track_id = v_info["track_id"]
                    
                    # Split active violations and check against logged ones
                    active_viols = [v.strip() for v in v_info["violation_type"].split("|")]
                    logged_types = logged_violations[track_id]
                    unlogged_viols = [v for v in active_viols if v not in logged_types]
                    
                    if not unlogged_viols:
                        continue
                        
                    # 1. Attempt LPR
                    plate_number = None
                    if track_id not in processed_plates and ocr_attempts.get(track_id, 0) < 5:
                        ocr_attempts[track_id] = ocr_attempts.get(track_id, 0) + 1
                        plate_number = run_lpr_on_vehicle(frame, v_info["bbox"], track_id, v_info["class_name"], frame_count)
                        if plate_number:
                            processed_plates[track_id] = plate_number
                            
                    # 2. Log if plate found, or attempts exhausted
                    if track_id in processed_plates:
                        plate_to_log = processed_plates[track_id]
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
                            "wrong_way_direction": v_info.get("wrong_way_direction", "UNKNOWN"),
                            "wrong_way_confidence": v_info.get("wrong_way_confidence", 0.0),
                        })
                    elif ocr_attempts.get(track_id, 0) >= 5:
                        logged_violations[track_id].update(active_viols)
                        if "WRONG WAY" in active_viols and track_id in wrong_way_states:
                            wrong_way_states[track_id]["logged_wrong_way"] = True
                        new_violations_logged.append({
                            "track_id": track_id,
                            "class_name": v_info["class_name"],
                            "violation_type": v_info["violation_type"] or "SAFETY VIOLATION",
                            "plate_number": "UNKNOWN",
                            "bbox": v_info["bbox"],
                            "lane_idx": v_info["lane_idx"],
                            "wrong_way_direction": v_info.get("wrong_way_direction", "UNKNOWN"),
                            "wrong_way_confidence": v_info.get("wrong_way_confidence", 0.0),
                        })

            # Check violating pedestrians (standalone riders)
            for ped in pedestrians:
                if ped.get("is_inside_vehicle", False):
                    continue
                track_id = ped["track_id"]
                if track_id in violating_pedestrians:
                    active_viols = ["HELMET VIOLATION"]
                    logged_types = logged_violations[track_id]
                    unlogged_viols = [v for v in active_viols if v not in logged_types]
                    
                    if not unlogged_viols:
                        continue
                        
                    # 1. Attempt LPR
                    plate_number = None
                    if track_id not in processed_plates and ocr_attempts.get(track_id, 0) < 5:
                        ocr_attempts[track_id] = ocr_attempts.get(track_id, 0) + 1
                        plate_number = run_lpr_on_vehicle(frame, ped["bbox"], track_id, "rider", frame_count)
                        if plate_number:
                            processed_plates[track_id] = plate_number
                            
                    # 2. Log if plate found, or attempts exhausted
                    if track_id in processed_plates:
                        plate_to_log = processed_plates[track_id]
                        logged_violations[track_id].update(active_viols)
                        new_violations_logged.append({
                            "track_id": track_id,
                            "class_name": "rider" if ped.get("is_rider", False) else "pedestrian",
                            "violation_type": "HELMET VIOLATION",
                            "plate_number": plate_to_log,
                        })
                    elif ocr_attempts.get(track_id, 0) >= 5:
                        logged_violations[track_id].update(active_viols)
                        new_violations_logged.append({
                            "track_id": track_id,
                            "class_name": "rider" if ped.get("is_rider", False) else "pedestrian",
                            "violation_type": "HELMET VIOLATION",
                            "plate_number": "UNKNOWN",
                        })

            # Save evidence photos for new violations logged this frame
            if new_violations_logged:
                # Create evidence directory
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
                    
                    # Save the annotated full frame with crop and metadata when available.
                    evidence_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    evidence_img = build_evidence_image(enhanced_frame, viol, evidence_timestamp)
                    cv2.imwrite(str(evidence_path), evidence_img)
                    print(f"📸 Saved evidence image: {evidence_path}")
                    
                    # Insert into Database
                    try:
                        conn = sqlite3.connect("traffic_violations.db")
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO violations (timestamp, track_id, vehicle_type, violation_type, plate_number, frame_number, evidence_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?);
                        """, (
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            track_id,
                            class_name,
                            v_type,
                            plate_number,
                            frame_count,
                            str(evidence_path.resolve())
                        ))
                        conn.commit()
                        conn.close()
                        print(f"💾 Logged violation for {class_name}#{track_id} to DB. Plate: {plate_number}")
                    except Exception as e:
                        print(f"Database write error: {e}")

            # Normalize pressures
            for i in range(3):
                lane_pressures[i] = min(1.0, lane_pressures[i])
                
            # Clean stale trajectories & caches
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
                    
            # 4. Draw HUD overlays (Commented out to remove the big panels)
            # draw_hud_panels(enhanced_frame, time.time() - t_start, active_imgsz, preproc_status,
            #                 phase_mgr, lane_counts, lane_pressures, total_violations)
            draw_gps_panel(map_img, 800, 800)
            
            # Show displays
            cv2.imshow(cam_win, enhanced_frame)
            cv2.imshow(map_win, map_img)
            
            # Frame rate sync
            t_loop = time.perf_counter() - t_frame_start
            time.sleep(max(0.001, frame_delay - t_loop))
            
            # Check window visibility close click
            if cv2.getWindowProperty(cam_win, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(map_win, cv2.WND_PROP_VISIBLE) < 1:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup staged model copies
        if staged_temp_dir is not None and staged_temp_dir.exists():
            shutil.rmtree(staged_temp_dir, ignore_errors=True)
            
        print("Dual visualizer resources closed cleanly.")

if __name__ == "__main__":
    main()
