"""Smart parking vision: detect empty / occupied slots on the drawn parking sheet
from a phone camera at (almost) any angle, and show every processing step.

Pipeline (all shown live in the dashboard window):
  1. Find the white sheet in the camera frame (largest bright, low-saturation blob -> 4 corners)
  2. Perspective-warp it to a flat top-down view, so the camera angle stops mattering
  3. Normalise lighting using the lane pixels (no cars there) as a brightness reference
  4. Compare with a baseline photo of the EMPTY sheet (colour change + new edges)
  5. Per slot: changed pixels / slot pixels = occupancy ratio
  6. Debounce with hysteresis so a slot only flips after several consistent frames
  7. Digital logic: occupied = sum, available = N - occupied, FULL = AND, gate = ENTRY AND NOT FULL

Phone as camera (pick one):
  Android : app "IP Webcam"  -> python parking_vision.py --source http://PHONE_IP:8080/video
  Any     : DroidCam / Iriun / Camo (appears as a normal webcam) -> --source 1
  Test    : a saved photo -> --source photo.jpg

Keys:  B baseline (sheet EMPTY)   L lock/unlock sheet tracking   F flip 180 deg
       R rotate 90 deg (if view looks squashed)   E edit slot corners   S screenshot   Q quit
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Layout. Slots are stored as normalised (0..1) polygons of the sheet, so they
# work for any camera angle. Defaults were traced from the hand-drawn sketch.
# ----------------------------------------------------------------------------
SHEET_W, SHEET_H = 560, 792          # top-down size (A4 ratio)
SKETCH_W, SKETCH_H = 707.0, 1000.0   # sketch pixel space used for the defaults below

_DEFAULT_SKETCH = [
    ("L1", [(12, 115), (212, 215), (212, 350), (7, 230)]),
    ("L2", [(7, 230), (212, 350), (222, 485), (7, 345)]),
    ("L3", [(7, 345), (222, 485), (207, 640), (2, 495)]),
    ("L4", [(2, 495), (207, 640), (182, 805), (2, 680)]),
    ("L5", [(2, 680), (182, 805), (167, 965), (2, 860)]),
    ("R1", [(542, 175), (702, 235), (697, 365), (537, 295)]),
    ("R2", [(537, 295), (697, 365), (702, 495), (552, 410)]),
    ("R3", [(552, 410), (702, 495), (697, 640), (552, 545)]),
    ("R4", [(552, 545), (697, 640), (702, 805), (552, 700)]),
    ("R5", [(552, 700), (702, 805), (702, 975), (542, 865)]),
    ("ENTRY", [(397, 20), (537, 20), (537, 170), (397, 170)]),   # top of the down-lane
]
DEFAULT_ZONES = [(n, [(x / SKETCH_W, y / SKETCH_H) for x, y in p]) for n, p in _DEFAULT_SKETCH]

DST = np.float32([[0, 0], [SHEET_W, 0], [SHEET_W, SHEET_H], [0, SHEET_H]])

# Detection thresholds (OpenCV 8-bit LAB units)
L_THR = 45        # getting brighter by this much counts as "something is here"
DARK_RATIO = 0.62  # getting darker than this fraction of the baseline counts (shadows are lighter than this)
AB_THR = 22       # colour change that counts as "something is here"


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def order_cyclic(pts):
    """Sort 4 points clockwise (on screen) around their centre."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


FOCAL_RATIO = 0.85   # assumed phone focal length as a fraction of the longer frame side (+-40% is fine)


def aspect_from_quad(quad, shape, focal_ratio=FOCAL_RATIO):
    """Real width/height of a rectangle seen in perspective, quad = [tl, tr, br, bl].
    Zhang & He's method with an assumed focal length. (Estimating the focal length from the
    quad itself is degenerate for the most common pose: phone tilted straight forward.)"""
    tl, tr, br, bl = [np.float64(p) for p in quad]
    plain = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    h_img, w_img = shape[:2]
    f = focal_ratio * max(h_img, w_img)
    m1, m2, m3, m4 = [np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br)]
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
        n2, n3 = k2 * m2 - m1, k3 * m3 - m1
        Ai = np.linalg.inv(np.array([[f, 0, w_img / 2.0], [0, f, h_img / 2.0], [0, 0, 1.0]]))
        M = Ai.T @ Ai
        r = (n2 @ M @ n2) / (n3 @ M @ n3)
        return float(np.sqrt(r)) if np.isfinite(r) and r > 0 else float(plain)
    except (ZeroDivisionError, np.linalg.LinAlgError, FloatingPointError):
        return float(plain)


def choose_corners(quad, shape, prev=None, swap=False):
    """Order the sheet corners so the warped view is portrait. Uses the real aspect ratio,
    so it works even when perspective makes the sheet look square or landscape."""
    cyc = order_cyclic(quad)
    ratio = aspect_from_quad(cyc, shape)
    landscape = (ratio > 1.0) != swap
    base = 1 if landscape else 0
    cands = [np.roll(cyc, -base, axis=0), np.roll(cyc, -(base + 2), axis=0)]   # differ by 180 degrees
    if prev is not None:
        best = min(cands, key=lambda c: np.linalg.norm(c - prev, axis=1).mean())   # stay consistent
    else:
        best = min(cands, key=lambda c: c[0].sum())                                # top-left-most first
    return np.float32(best), ratio


def refine_quad(contour, quad):
    """Replace the rounded corners from the blob outline with the intersections of 4 straight
    lines fitted to the sheet edges. Much more accurate, and accuracy is what keeps the
    painted lines aligned with the empty-sheet baseline."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 5:
            return quad
        d = d / L
        rel = pts - a
        t = rel @ d
        dist = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        sel = pts[(dist < 0.04 * L + 3) & (t > 0.12 * L) & (t < 0.88 * L)]
        if len(sel) < 10:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]       # corner i = edge (i-1) meets edge i
        A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float32)
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        t = np.linalg.solve(A, p2 - p1)
        out.append(p1 + t[0] * d1)
    out = np.float32(out)
    diag = np.linalg.norm(quad[0] - quad[2])
    return out if np.linalg.norm(out - quad, axis=1).max() < 0.12 * diag else quad


def find_paper(frame):
    """Find the white sheet. Returns 4 corners in frame coordinates, or None."""
    h, w = frame.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s) if s < 1.0 else frame
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    _, bright = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_sat = (hsv[..., 1] < 70).astype(np.uint8) * 255
    mask = cv2.bitwise_and(bright, low_sat)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # fills pen lines and toy cars
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # NONE: keep every edge pixel for line fitting
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.08 * small.shape[0] * small.shape[1]:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = refine_quad(c, approx.reshape(4, 2).astype(np.float32))
            return quad / s
    return None


# ----------------------------------------------------------------------------
# Debounced decision per slot
# ----------------------------------------------------------------------------
class Debounced:
    """Smooth the ratio, then flip state only after `frames` consistent frames (hysteresis)."""

    def __init__(self, on=0.12, off=0.05, frames=8, alpha=0.5):
        self.on, self.off, self.frames, self.alpha = on, off, frames, alpha
        self.ema, self.state, self.count = None, False, 0

    def update(self, ratio):
        self.ema = ratio if self.ema is None else self.alpha * ratio + (1 - self.alpha) * self.ema
        wants_change = (self.ema > self.on) if not self.state else (self.ema < self.off)
        self.count = self.count + 1 if wants_change else 0
        if self.count >= self.frames:
            self.state, self.count = not self.state, 0
        return self.state

    def reset(self):
        self.ema, self.state, self.count = None, False, 0


# ----------------------------------------------------------------------------
# Main engine (no GUI code in here, so it can be tested headless)
# ----------------------------------------------------------------------------
def load_zones(path):
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return [(z["name"], [tuple(p) for p in z["poly"]]) for z in data["zones"]]
    return DEFAULT_ZONES


def save_zones(path, zones):
    with open(path, "w") as f:
        json.dump({"zones": [{"name": n, "poly": [list(p) for p in poly]} for n, poly in zones]}, f, indent=1)


class ParkingVision:
    def __init__(self, zones, on=0.12, off=0.05, frames=8):
        self.on, self.off, self.frames = on, off, frames
        self.set_zones(zones)
        self.corners = None
        self.locked = False
        self.swap = False
        self.aspect = None
        self.paper_found = False
        self.baseline_lab = None
        self.baseline_edges = None
        self.baseline_img = None
        self.band_k = np.ones((9, 9), np.uint8)

    # -- zones ---------------------------------------------------------------
    def set_zones(self, zones):
        self.zones = zones
        self.polys_px, self.masks, self.areas = {}, {}, {}
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        union = np.zeros((SHEET_H, SHEET_W), np.uint8)
        for name, poly in zones:
            px = (np.float32(poly) * [SHEET_W, SHEET_H]).astype(np.int32)
            m = np.zeros((SHEET_H, SHEET_W), np.uint8)
            cv2.fillPoly(m, [px], 255)
            union |= m
            m = cv2.erode(m, k)            # keep away from the painted slot lines
            self.polys_px[name], self.masks[name] = px, m > 0
            self.areas[name] = max(int(np.count_nonzero(m)), 1)
        probe = cv2.dilate(union, k, iterations=2) == 0   # lane / margin pixels, never covered by cars
        probe[:14, :] = probe[-14:, :] = False
        probe[:, :14] = probe[:, -14:] = False
        self.probe = probe
        self.slot_names = [n for n, _ in zones if n != "ENTRY"]
        self.states = {n: Debounced(self.on, self.off, self.frames) for n, _ in zones}

    # -- sheet tracking --------------------------------------------------------
    def update_paper(self, frame):
        quad = None if self.locked else find_paper(frame)
        self.paper_found = quad is not None
        if quad is None:
            return
        quad, self.aspect = choose_corners(quad, frame.shape, self.corners, self.swap)
        if self.corners is not None:
            diag = np.linalg.norm(self.corners[0] - self.corners[2])
            if np.linalg.norm(quad - self.corners, axis=1).mean() < 0.15 * diag:
                quad = 0.6 * self.corners + 0.4 * quad     # smooth jitter, follow slow motion
        self.corners = quad.astype(np.float32)

    def flip_180(self):
        if self.corners is not None:
            self.corners = np.roll(self.corners, 2, axis=0)

    def swap_90(self):
        self.swap = not self.swap
        self.corners = None

    # -- baseline ----------------------------------------------------------------
    @staticmethod
    def _prep(rect):
        blur = cv2.GaussianBlur(rect, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), 60, 140)
        return lab, edges

    def set_baseline(self, rect):
        self.baseline_img = rect.copy()
        self.baseline_lab, edges = self._prep(rect)
        self.baseline_edges = cv2.dilate(edges, np.ones((13, 13), np.uint8))   # +-6 px alignment tolerance
        for s in self.states.values():
            s.reset()

    # -- one frame ----------------------------------------------------------------
    def process(self, frame):
        t0 = time.time()
        self.update_paper(frame)
        out = {"frame": frame, "paper_found": self.paper_found, "rect": None, "mask": None,
               "ratios": {}, "occ": {}, "gain": None, "H": None}
        if self.corners is None:
            return out
        H = cv2.getPerspectiveTransform(self.corners, DST)
        rect = cv2.warpPerspective(frame, H, (SHEET_W, SHEET_H), flags=cv2.INTER_LINEAR)
        out.update(rect=rect, H=H)

        if self.baseline_lab is not None:
            lab, edges = self._prep(rect)
            ref = self.baseline_lab
            sel = self.probe & (ref[..., 0] > 120)          # bright paper in lane areas
            gain = 1.0
            if np.count_nonzero(sel) > 500:
                gain = float(np.clip(np.median(ref[..., 0][sel]) / max(np.median(lab[..., 0][sel]), 1.0), 0.6, 1.6))
            # Tolerant difference: compare against the min..max band of the baseline in a 9x9 window,
            # so a 1-3 px alignment wobble of the painted lines is not mistaken for a car.
            cur = lab.copy()
            cur[..., 0] *= gain
            lo, hi = cv2.erode(ref, self.band_k), cv2.dilate(ref, self.band_k)
            excess = np.maximum(lo - cur, cur - hi)                     # how far outside the band, per channel
            # Darker only counts when it is MUCH darker (a car), so a soft shadow or a hand's shade is ignored.
            much_darker = cur[..., 0] < lo[..., 0] * DARK_RATIO - 6
            brighter = (cur[..., 0] - hi[..., 0]) > L_THR
            changed = much_darker | brighter | (np.maximum(excess[..., 1], excess[..., 2]) > AB_THR)
            new_edges = (edges > 0) & (self.baseline_edges == 0)    # catches white-on-white cars
            new_edges = cv2.dilate(new_edges.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            new_edges = cv2.morphologyEx(new_edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            cnts, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:                                           # fill outlines, even with gaps (flat light cars)
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 400 and min(cv2.minAreaRect(hull)[1]) >= 18:   # compact blob, not a thin line
                    cv2.drawContours(new_edges, [hull], -1, 255, cv2.FILLED)
            mask = changed.astype(np.uint8) * 255 | new_edges
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop speckle noise
            for name, _ in self.zones:
                ratio = np.count_nonzero(mask[self.masks[name]]) / self.areas[name]
                out["ratios"][name] = ratio
                out["occ"][name] = self.states[name].update(ratio)
            out.update(mask=mask, gain=gain)

        out["ms"] = (time.time() - t0) * 1000
        return out

    # -- digital logic ------------------------------------------------------------------
    def logic(self, occ):
        bits = [int(occ.get(n, False)) for n in self.slot_names]
        occupied = sum(bits)
        total = len(bits)
        full = int(all(bits)) if bits else 0
        entry = int(occ.get("ENTRY", False))
        return {"bits": bits, "occupied": occupied, "available": total - occupied, "total": total,
                "full": full, "entry": entry, "gate_open": int(entry and not full)}


# ----------------------------------------------------------------------------
# Frame source: webcam index, phone stream URL, or still image. Reads on a
# background thread so a slow phone stream never builds up lag.
# ----------------------------------------------------------------------------
class FrameSource:
    def __init__(self, source):
        self.image = None
        if os.path.isfile(source):
            self.image = cv2.imread(source)
            if self.image is None:
                sys.exit(f"Could not read image {source}")
            self.cap = None
            return
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            sys.exit(f"Could not open source '{source}'. Check the index/URL and that phone and laptop share Wi-Fi.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.ok, self.running = None, False, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            self.ok = ok
            if ok:
                self.frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        if self.image is not None:
            return self.image.copy()
        return None if self.frame is None else self.frame.copy()

    def close(self):
        if self.cap is not None:
            self.running = False
            time.sleep(0.1)
            self.cap.release()


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
GREEN, RED, YELLOW, CYAN, WHITE, GREY = (80, 200, 60), (60, 60, 230), (0, 220, 255), (255, 200, 0), (240, 240, 240), (150, 150, 150)


def text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h, title=None):
    canvas = np.full((h, w, 3), 25, np.uint8)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        s = min(w / img.shape[1], h / img.shape[0])
        r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    if title:
        cv2.rectangle(canvas, (0, 0), (w, 20), (0, 0, 0), -1)
        text(canvas, title, (6, 15), 0.45, YELLOW)
    return canvas


def zone_color(name, occ):
    if name == "ENTRY":
        return CYAN if occ.get(name) else GREY
    return RED if occ.get(name) else GREEN


def draw_zones(img, polys, occ, ratios, alpha=0.30, labels=True):
    over = img.copy()
    for name, poly in polys.items():
        cv2.fillPoly(over, [poly.astype(np.int32)], zone_color(name, occ))
    img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for name, poly in polys.items():
        cv2.polylines(img, [poly.astype(np.int32)], True, zone_color(name, occ), 2, cv2.LINE_AA)
        if labels:
            cx, cy = poly.mean(axis=0).astype(int)
            tag = name if name not in ratios else f"{name} {ratios[name] * 100:.0f}%"
            text(img, tag, (cx - 28, cy + 4), 0.45)
    return img


def build_dashboard(vision, out, fps, source_name):
    occ, ratios = out["occ"], out["ratios"]
    lg = vision.logic(occ)

    # Panel A: camera view with sheet outline and projected slots
    cam = out["frame"].copy()
    if vision.corners is not None:
        cv2.polylines(cam, [vision.corners.astype(np.int32)], True, YELLOW, 3, cv2.LINE_AA)
        for i, p in enumerate(vision.corners):
            cv2.circle(cam, tuple(int(v) for v in p), 7, YELLOW, -1)
            text(cam, "TL TR BR BL".split()[i], (int(p[0]) + 8, int(p[1]) - 8), 0.6, YELLOW, 2)
        if out["H"] is not None:
            Hinv = np.linalg.inv(out["H"])
            polys = {n: cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)
                     for n, p in vision.polys_px.items()}
            cam = draw_zones(cam, polys, occ, {}, 0.25, labels=False)
            for n, p in polys.items():
                cx, cy = p.mean(axis=0).astype(int)
                text(cam, n, (cx - 12, cy + 5), 0.55, WHITE, 2)
    panelA = fit(cam, 640, 360, "1) CAMERA  - sheet outline (yellow) and slots projected back")

    # Panel B: flat top-down view
    if out["rect"] is not None:
        rb = draw_zones(out["rect"], vision.polys_px, occ, ratios)
    else:
        rb = None
    panelB = fit(rb, 320, 453, "2) TOP-DOWN VIEW (warped)")

    # Panel C: change mask
    if out["mask"] is not None:
        heat = cv2.cvtColor(out["mask"], cv2.COLOR_GRAY2BGR)
        heat[out["mask"] > 0] = (60, 90, 255)
        for n, p in vision.polys_px.items():
            cv2.polylines(heat, [p], True, zone_color(n, occ), 2, cv2.LINE_AA)
        panelC = fit(heat, 320, 453, "3) WHAT CHANGED vs EMPTY BASELINE")
    else:
        panelC = fit(vision.baseline_img, 320, 453, "3) baseline not set (press B)")

    # Panel D: logic + per-slot bars
    d = np.full((360, 640, 3), 30, np.uint8)
    paper = "LOCKED" if vision.locked else ("TRACKING" if out["paper_found"] else "LOST (using last)")
    if vision.corners is None:
        paper = "SEARCHING - show the whole white sheet"
    text(d, f"Source: {source_name}   {fps:.0f} fps", (8, 18), 0.5, GREY)
    asp = f"   aspect est {vision.aspect:.2f} (A4 = 0.71)" if vision.aspect else ""
    text(d, f"Sheet: {paper}{asp}", (8, 40), 0.55, YELLOW if out["paper_found"] or vision.locked else RED, 1)
    if vision.baseline_lab is None:
        text(d, "Baseline: NOT SET - empty all slots, then press B", (8, 62), 0.55, RED, 1)
    else:
        text(d, "Baseline: set", (8, 62), 0.55, GREEN, 1)
        text(d, f"OCCUPIED = {' + '.join(str(b) for b in lg['bits'])} = {lg['occupied']}", (8, 92), 0.55)
        text(d, f"AVAILABLE = {lg['total']} - {lg['occupied']} = {lg['available']}", (8, 114), 0.55,
             GREEN if lg["available"] else RED, 1)
        text(d, f"FULL = AND(all slots) = {lg['full']}", (8, 136), 0.55, RED if lg["full"] else WHITE)
        text(d, f"ENTRY car = {lg['entry']}   GATE OPEN = ENTRY AND NOT FULL = {lg['gate_open']}", (8, 158), 0.55,
             GREEN if lg["gate_open"] else WHITE, 1)
        for i, name in enumerate(vision.slot_names):
            col, row = divmod(i, 5)
            x0, y = 8 + col * 318, 188 + row * 24
            r = ratios.get(name, 0)
            text(d, name, (x0, y + 12), 0.5)
            cv2.rectangle(d, (x0 + 34, y), (x0 + 184, y + 14), (70, 70, 70), -1)
            cv2.rectangle(d, (x0 + 34, y), (x0 + 34 + int(min(r, 1) * 150), y + 14), RED if occ.get(name) else GREEN, -1)
            for t in (vision.on, vision.off):
                cv2.line(d, (x0 + 34 + int(t * 150), y - 2), (x0 + 34 + int(t * 150), y + 16), YELLOW, 1)
            text(d, f"{r * 100:3.0f}%  {'OCCUPIED' if occ.get(name) else 'empty'}", (x0 + 192, y + 12), 0.45,
                 RED if occ.get(name) else GREEN)
    text(d, "B baseline  L lock  F flip180  R rot90  E edit slots  S shot  Q quit", (8, 350), 0.42, GREY)
    panelD = fit(d, 640, 360, None)

    # Panel E: pipeline explanation with live numbers
    e = np.full((267, 640, 3), 38, np.uint8)
    lines = [
        ("PIPELINE", YELLOW),
        (f"1 Find sheet: bright + low-saturation blob -> 4 corners   [{'OK' if vision.corners is not None else 'no'}]", WHITE),
        (f"2 Homography: warp to {SHEET_W}x{SHEET_H} flat view (angle-independent)", WHITE),
        (f"3 Lighting gain from lane pixels: x{out['gain']:.2f}" if out["gain"] else "3 Lighting gain: waiting for baseline", WHITE),
        (f"4 Change = much darker/brighter, colour>{AB_THR} or new edges vs baseline", WHITE),
        ("5 Slot ratio = changed px / slot px (slot mask shrunk off the lines)", WHITE),
        (f"6 Debounce: ON>{vision.on * 100:.0f}%  OFF<{vision.off * 100:.0f}%  for {vision.frames} frames", WHITE),
        ("7 Logic: sum, 4-AND (FULL), ENTRY AND NOT FULL -> gate", WHITE),
        (f"Processing time: {out.get('ms', 0):.0f} ms/frame", GREY),
    ]
    for i, (s, c) in enumerate(lines):
        text(e, s, (8, 22 + i * 26), 0.5 if i else 0.6, c)
    panelE = fit(e, 640, 267, None)

    col1 = np.vstack([panelA, panelD])                       # 640 x 720
    col23 = np.vstack([np.hstack([panelB, panelC]), panelE])  # 640 x 720
    return np.hstack([col1, col23])


# ----------------------------------------------------------------------------
# Slot editor: click the 4 corners of each zone on a frozen top-down view
# ----------------------------------------------------------------------------
def edit_zones(rect, zones, path):
    scale = min(1.0, 700.0 / SHEET_H)
    base = cv2.resize(rect, None, fx=scale, fy=scale)
    state = {"i": 0, "pts": [], "new": []}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(zones):
            state["pts"].append((x, y))
            if len(state["pts"]) == 4:
                n = [(px / (SHEET_W * scale), py / (SHEET_H * scale)) for px, py in state["pts"]]
                state["new"].append((zones[state["i"]][0], n))
                state["i"] += 1
                state["pts"] = []

    win = "Edit slots"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = base.copy()
        for name, poly in state["new"]:
            px = (np.float32(poly) * [SHEET_W * scale, SHEET_H * scale]).astype(np.int32)
            cv2.polylines(img, [px], True, GREEN, 2)
        for p in state["pts"]:
            cv2.circle(img, p, 4, YELLOW, -1)
        if state["i"] < len(zones):
            text(img, f"Click 4 corners of {zones[state['i']][0]}   (N = keep old, Z = undo point, ESC = cancel)", (8, 18), 0.5, YELLOW)
        else:
            text(img, "Done - press ENTER to save", (8, 18), 0.6, GREEN)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(win)
            return None
        if k in (13, 10) and state["i"] >= len(zones):
            save_zones(path, state["new"])
            cv2.destroyWindow(win)
            return state["new"]
        if k in (ord("z"), ord("Z")) and state["pts"]:
            state["pts"].pop()
        if k in (ord("n"), ord("N")) and state["i"] < len(zones):
            state["new"].append(zones[state["i"]])
            state["i"] += 1
            state["pts"] = []


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Smart parking vision dashboard")
    ap.add_argument("--source", default="0", help="webcam index, phone stream URL, or image file")
    ap.add_argument("--zones", default="slots.json", help="slot layout file (created by the editor)")
    ap.add_argument("--baseline", default="baseline.png", help="empty-sheet baseline image")
    ap.add_argument("--on", type=float, default=0.12, help="occupied threshold: fraction of the slot a car must cover")
    ap.add_argument("--off", type=float, default=0.05, help="empty threshold: must fall below this to become empty")
    ap.add_argument("--frames", type=int, default=8, help="frames needed to change state")
    ap.add_argument("--scale", type=float, default=1.0, help="dashboard window scale")
    args = ap.parse_args()

    vision = ParkingVision(load_zones(args.zones), args.on, args.off, args.frames)
    if os.path.exists(args.baseline):
        vision.set_baseline(cv2.resize(cv2.imread(args.baseline), (SHEET_W, SHEET_H)))
        print(f"Loaded baseline {args.baseline} (press B to replace it)")
    src = FrameSource(args.source)
    win = "Smart Parking Vision"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    fps, t_prev, shot = 0.0, time.time(), 0

    while True:
        frame = src.read()
        if frame is None:
            time.sleep(0.02)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        out = vision.process(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        dash = build_dashboard(vision, out, fps, args.source)
        if args.scale != 1.0:
            dash = cv2.resize(dash, None, fx=args.scale, fy=args.scale)
        cv2.imshow(win, dash)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("b"), ord("B")) and out["rect"] is not None:
            vision.set_baseline(out["rect"])
            cv2.imwrite(args.baseline, out["rect"])
            print("Baseline saved ->", args.baseline)
        elif k in (ord("l"), ord("L")):
            vision.locked = not vision.locked
        elif k in (ord("f"), ord("F")):
            vision.flip_180()
        elif k in (ord("r"), ord("R")):
            vision.swap_90()
        elif k in (ord("e"), ord("E")) and out["rect"] is not None:
            new = edit_zones(out["rect"], vision.zones, args.zones)
            if new:
                vision.set_zones(new)
                print("Saved", args.zones)
        elif k in (ord("s"), ord("S")):
            shot += 1
            cv2.imwrite(f"dashboard_{shot}.png", dash)
            print(f"Saved dashboard_{shot}.png")

    src.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
