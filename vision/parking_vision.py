"""Smart parking vision: detect empty / occupied slots on the drawn parking sheet
from a phone camera at (almost) any angle, and show every processing step.

Pipeline (all shown live in the dashboard window):
  1. Find the white sheet AUTOMATICALLY (no clicks): many candidate regions are proposed (brightness /
     colourlessness thresholds, smooth-surface masks, edge loops) and each is scored on uniformity,
     visible edge on every side, rectangularity, A4 aspect (perspective-corrected), corner angles and
     size. Patterned bedsheets and other bright objects fail those tests. No confident sheet -> warning.
  2. Perspective-warp it to a flat top-down view, so the camera angle stops mattering
  3. Normalise lighting with a smooth gain field estimated from the lane pixels (no cars there)
  4. Compare with a baseline photo of the EMPTY sheet (colour change + new edges)
  5. Per slot (own crop, no leakage): largest changed blob / slot pixels = occupancy ratio
  6. Debounce with hysteresis (EMPTY / OCCUPIED / UNCERTAIN); unreliable frames keep the old state
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

# Slot layout measured on the real sheet photo (orange dots mark the ends of the 6 slot lines per side).
# Pixel numbers are in the 560 x 792 top-down view; each line is (y at sheet edge, y at lane side).
_LEFT_LINES = [(118, 124), (240, 238), (366, 361), (473, 478), (600, 597), (720, 723)]
_RIGHT_LINES = [(119, 117), (241, 241), (372, 370), (490, 493), (607, 606), (726, 726)]
_XL_OUT, _XL_IN, _XR_IN, _XR_OUT = 3, 170, 388, 557


def _slots(prefix, lines, x_out, x_in):
    out = []
    for k in range(5):
        (yo0, yi0), (yo1, yi1) = lines[k], lines[k + 1]
        out.append((f"{prefix}{k + 1}", [(x_out, yo0), (x_in, yi0), (x_in, yi1), (x_out, yo1)]))
    return out


_DEFAULT_PX = (_slots("L", _LEFT_LINES, _XL_OUT, _XL_IN) + _slots("R", _RIGHT_LINES, _XR_OUT, _XR_IN)
               + [("ENTRY", [(180, 28), (245, 28), (245, 112), (180, 112)])])      # top of the lane with the down arrow
DEFAULT_ZONES = [(n, [(x / 560.0, y / 792.0) for x, y in p]) for n, p in _DEFAULT_PX]

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


A4_ASPECT = 210.0 / 297.0
PAPER_ACCEPT = 0.55    # minimum score to lock onto a NEW paper candidate
PAPER_KEEP = 0.40      # minimum score to keep following a sheet that is already tracked
DETECT_W = 480         # paper search runs on a frame downscaled to this long side


def _paper_features(small):
    """Per-pixel cues used to tell white paper from everything else."""
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    white = lab[..., 0] - 2.0 * chroma                        # bright AND colourless
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    return lab, chroma, white, grad, gray


def _quad_from_contour(c):
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    if peri < 40:
        return None
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _candidate_contours(F, shape):
    """Several independent ways of proposing 'paper-like' regions; scoring decides which is real."""
    lab, chroma, white, grad, gray = F
    h, w = shape[:2]
    min_area = 0.05 * h * w
    w8 = np.clip(white, 0, 255).astype(np.uint8)
    t1, _ = cv2.threshold(w8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ths = [t1]
    hi = w8[w8 > t1]
    if hi.size > 500:                                   # 2nd Otsu: splits paper from pale bedsheet
        t2, _ = cv2.threshold(hi.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ths.append(t2)
    ths += [float(np.percentile(w8, p)) for p in (50, 68, 84)]
    flat = cv2.blur(grad, (9, 9)) < 22                  # smooth surface (paper) vs patterned cloth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    masks, seen = [], []
    for t in ths:
        if any(abs(t - s) < 5 for s in seen):
            continue
        seen.append(t)
        m = (w8 > t)
        masks.append(m)
        masks.append(m & flat)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)       # independent of brightness
    e = cv2.morphologyEx(cv2.dilate(edges, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, k)
    out = []
    for m in masks:
        m8 = m.astype(np.uint8) * 255
        m8 = cv2.morphologyEx(cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = sorted([c for c in cs if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)
        out += cs[:3]
    cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)   # holes enclosed by closed edge chains
    cs = sorted([c for c in cs if min_area <= cv2.contourArea(c) <= 0.9 * h * w], key=cv2.contourArea, reverse=True)
    return out + cs[:4]


def _merge_lines(segs):
    """Cluster Hough segments that lie on the same straight line into one long line with an extent."""
    items = []
    for x1, y1, x2, y2 in segs:
        d = np.array([x2 - x1, y2 - y1], np.float64)
        L = np.linalg.norm(d)
        if L < 1:
            continue
        d /= L
        if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        items.append((L, d, np.array([x1, y1], np.float64), np.array([x2, y2], np.float64)))
    items.sort(key=lambda t: -t[0])
    clusters = []
    for L, d, a, b in items:
        n = np.array([-d[1], d[0]])
        rho = float(a @ n)
        for c in clusters:
            dot = abs(float(d @ c["d"]))
            if dot > np.cos(np.radians(4)) and abs(float(a @ c["n"]) - c["rho"]) < 6 and abs(float(b @ c["n"]) - c["rho"]) < 6:
                ts = [float(a @ c["d"]), float(b @ c["d"])]
                c["t0"], c["t1"] = min(c["t0"], *ts), max(c["t1"], *ts)
                break
        else:
            ts = [float(a @ d), float(b @ d)]
            clusters.append({"d": d, "n": n, "rho": rho, "t0": min(ts), "t1": max(ts)})
    return clusters


def _line_quads(gray, shape, max_lines=22, top=14):
    """Fallback proposals that do not depend on colour at all: long straight edges -> rectangles.
    Works even when the white paper touches equally bright bedsheet."""
    h, w = shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 90)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=int(0.16 * min(h, w)), maxLineGap=8)
    if segs is None:
        return []
    lines = _merge_lines(segs.reshape(-1, 4))
    lines = sorted(lines, key=lambda c: -(c["t1"] - c["t0"]))[:max_lines]
    mind = 0.15 * min(h, w)

    def inter(a, b):
        A = np.array([a["n"], b["n"]])
        if abs(np.linalg.det(A)) < 0.15:
            return None
        return np.linalg.solve(A, [a["rho"], b["rho"]])

    def fold(a, b):                      # angle between two lines, 0..90 deg
        return np.degrees(np.arccos(min(1.0, abs(float(a["d"] @ b["d"])))))

    def parallel_pairs():
        out = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if fold(a, b) > 25:
                    continue
                mid = b["d"] * (b["t0"] + b["t1"]) / 2 + b["n"] * b["rho"]
                if abs(float(mid @ a["n"]) - a["rho"]) < mind:
                    continue
                out.append((i, j))
        return out

    pp = parallel_pairs()
    quads = []
    for x in range(len(pp)):
        A, B = lines[pp[x][0]], lines[pp[x][1]]
        for y in range(x + 1, len(pp)):
            ids = set(pp[x]) | set(pp[y])
            if len(ids) < 4:
                continue
            C, D = lines[pp[y][0]], lines[pp[y][1]]
            if fold(A, C) < 40:
                continue
            pts = [inter(A, C), inter(C, B), inter(B, D), inter(D, A)]
            if any(p is None for p in pts):
                continue
            pts = np.float32(pts)
            if (pts[:, 0] < -0.12 * w).any() or (pts[:, 0] > 1.12 * w).any() or (pts[:, 1] < -0.12 * h).any() or (pts[:, 1] > 1.12 * h).any():
                continue
            if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
                continue
            fa = cv2.contourArea(pts) / (h * w)
            if fa < 0.05 or fa > 0.88:
                continue
            sup = []
            for L_, P, Q in ((A, pts[0], pts[3]), (C, pts[0], pts[1]), (B, pts[1], pts[2]), (D, pts[2], pts[3])):
                tp, tq = float(P @ L_["d"]), float(Q @ L_["d"])
                lo, hi = min(tp, tq), max(tp, tq)
                sup.append(max(0.0, min(hi, L_["t1"]) - max(lo, L_["t0"])) / max(hi - lo, 1e-6))
            if np.mean(sup) < 0.5 or min(sup) < 0.25:
                continue
            quads.append((float(np.mean(sup)), pts))
    quads.sort(key=lambda t: -t[0])
    return [q for _, q in quads[:top]]


def _score_quad(quad, cont_area, F, shape, prev):
    """0..1 'how much does this quad look like a sheet of white A4 paper' + reasons. Combines:
    colour/brightness uniformity, smoothness, a visible edge on every side, rectangularity,
    perspective-corrected aspect ratio, corner angles, size and border contact."""
    lab, chroma, white, grad, gray = F
    h, w = shape[:2]
    cyc = order_cyclic(quad)
    qa = float(cv2.contourArea(cyc))
    fa = qa / (h * w)
    info = {"area": round(fa, 2)}
    if fa < 0.05 or fa > 0.88:
        return 0.0, dict(info, why="size (too small / fills whole frame)")
    mx, my = 0.01 * w, 0.01 * h
    if ((cyc[:, 0] < mx) | (cyc[:, 0] > w - mx) | (cyc[:, 1] < my) | (cyc[:, 1] > h - my)).any():
        return 0.0, dict(info, why="sheet is cut off by the frame edge - move camera back")
    sides = [float(np.linalg.norm(cyc[(i + 1) % 4] - cyc[i])) for i in range(4)]
    if min(sides) < 0.2 * max(sides):
        return 0.0, dict(info, why="degenerate shape")
    dev = 0.0
    for i in range(4):
        a, b = cyc[i - 1] - cyc[i], cyc[(i + 1) % 4] - cyc[i]
        ang = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1)))
        dev = max(dev, abs(ang - 90))
    if dev > 62:
        return 0.0, dict(info, why="corner angles not rectangular")
    ratio = aspect_from_quad(cyc, shape)
    a_err = abs(np.log(max(ratio, 1 / ratio) / (1 / A4_ASPECT)))
    if a_err > 0.5:
        return 0.0, dict(info, why=f"aspect {ratio:.2f} is not A4")
    fill = cont_area / qa
    if fill < 0.72:
        return 0.0, dict(info, why="irregular outline (not rectangular)")

    pad = 16
    x0, y0 = max(int(cyc[:, 0].min()) - pad, 0), max(int(cyc[:, 1].min()) - pad, 0)
    x1, y1 = min(int(cyc[:, 0].max()) + pad, w), min(int(cyc[:, 1].max()) + pad, h)
    q = cyc - [x0, y0]
    sh = (y1 - y0, x1 - x0)
    M = np.zeros(sh, np.uint8)
    cv2.fillPoly(M, [q.astype(np.int32)], 255)
    rw = int(np.clip(0.035 * np.sqrt(qa), 4, 14))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rw + 1, 2 * rw + 1))
    inner_all = cv2.erode(M, k) > 0
    C, Wh, G = (a[y0:y1, x0:x1] for a in (chroma, white, grad))
    if np.count_nonzero(inner_all) < 200:
        return 0.0, dict(info, why="too thin")
    Lc = lab[y0:y1, x0:x1]
    med = np.array([np.median(Lc[..., c][inner_all]) for c in range(3)])
    dE = np.sqrt((0.4 * (Lc[..., 0] - med[0])) ** 2 + (Lc[..., 1] - med[1]) ** 2 + (Lc[..., 2] - med[2]) ** 2)
    uniform = float(np.mean(dE[inner_all] < 14))
    dens = float(np.mean(G[inner_all] > 40))
    flat = 1.0 - float(np.clip((dens - 0.12) / 0.25, 0, 1))
    lowchroma = float(np.clip(1 - (np.median(C[inner_all]) - 8) / 30, 0, 1))
    info.update(uniform=round(uniform, 2), flat=round(flat, 2), chroma=round(float(np.median(C[inner_all])), 1))
    if uniform < 0.5:
        return 0.0, dict(info, why="not uniform (patterned surface)")

    # edge visible on every side: brighter inside than outside, and a real gradient on the boundary
    inner_band = (cv2.erode(M, np.ones((5, 5), np.uint8)) > 0) & ~inner_all
    outer_ring = (cv2.dilate(M, k) > 0) & (M == 0)
    side_scores = []
    for i in range(4):
        side = np.zeros(sh, np.uint8)
        cv2.line(side, tuple(int(v) for v in q[i]), tuple(int(v) for v in q[(i + 1) % 4]), 255, 2 * rw + 1)
        o = outer_ring & (side > 0)
        n_i = inner_band & (side > 0)
        if np.count_nonzero(o) < 0.25 * sides[i] * rw or np.count_nonzero(n_i) < 20:
            continue                                        # this side is at the frame edge: no evidence either way
        dW = float(np.median(Wh[n_i])) - Wh[o]
        cf = float(np.mean(dW > 8))
        edge = np.zeros(sh, np.uint8)
        cv2.line(edge, tuple(int(v) for v in q[i]), tuple(int(v) for v in q[(i + 1) % 4]), 255, 2 * max(2, rw // 2) + 1)
        ef = float(np.mean(G[edge > 0] > 30))
        side_scores.append(0.5 * cf + 0.5 * ef)
    if len(side_scores) < 2:
        bscore = 0.3
    else:
        bscore = 0.5 * float(np.mean(side_scores)) + 0.5 * float(np.min(side_scores))
        if bscore < 0.25:
            return 0.0, dict(info, why="no clear paper edge on all sides")
    info["edge"] = round(bscore, 2)

    aspect_s = 1 - float(np.clip((a_err - 0.15) / 0.2, 0, 1))      # a square card / towel scores ~0; strong tilt (est. error ~0.2) stays ok
    angle_s = 1 - float(np.clip((dev - 25) / 35, 0, 1))
    fill_s = float(np.clip((fill - 0.75) / 0.2, 0, 1))
    size_s = float(np.clip((fa - 0.04) / 0.10, 0, 1) * np.clip((0.90 - fa) / 0.10, 0, 1))
    s = (0.20 * uniform + 0.10 * flat + 0.26 * bscore + 0.20 * aspect_s + 0.06 * fill_s + 0.08 * angle_s + 0.10 * lowchroma) * size_s
    s *= 0.6 + 0.4 * aspect_s
    return float(np.clip(s, 0, 1)), dict(info, why="ok")


def detect_paper(frame, prev=None, keep=False):
    """Automatic detection of the white A4 sheet. Returns (quad | None, confidence, debug).
    No clicks: several candidate generators + geometric/appearance validation."""
    h, w = frame.shape[:2]
    s = min(1.0, DETECT_W / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else frame
    F = _paper_features(small)
    prev_s = None if prev is None else np.float32(prev) * s
    need = PAPER_KEEP if keep else PAPER_ACCEPT
    st = {"best": None, "prevq": None, "rej": None, "seen": []}

    def consider(quad, cont_area, contour, fallback=False):
        key = (tuple((quad.mean(0) / 6).astype(int)), int(cv2.contourArea(quad) / (0.01 * small.shape[0] * small.shape[1])))
        if not fallback:
            if key in st["seen"]:
                return
            st["seen"].append(key)       # (the fallback copy of the previous quad must not hide the fresh one)
        sc, info = _score_quad(quad, cont_area, F, small.shape, prev_s)
        if st["rej"] is None or sc > st["rej"][1]:
            st["rej"] = (quad / s, sc, info.get("why", ""))
        if info.get("why") == "ok":
            slot = "prevq" if fallback else "best"
            if st.get(slot) is None or sc > st[slot][1]:
                st[slot] = (quad, sc, contour, info)

    if prev_s is not None:                                   # the sheet we were already following
        consider(order_cyclic(prev_s), 0.95 * cv2.contourArea(order_cyclic(prev_s)), None, fallback=True)
    for c in _candidate_contours(F, small.shape):
        quad = _quad_from_contour(c)
        if quad is not None:
            consider(quad, cv2.contourArea(c), c)
    if st["best"] is None or st["best"][1] < PAPER_ACCEPT:   # colour merged paper with cloth -> use straight edges
        for quad in _line_quads(F[4], small.shape):
            consider(order_cyclic(quad), 0.95 * cv2.contourArea(quad), None)
    best = st["best"]
    if (best is None or best[1] < need) and st.get("prevq") is not None:
        best = st["prevq"]              # nothing better found: keep following the sheet we had (e.g. briefly hidden by a hand)
    dbg = {"best_rejected": st["rej"], "info": None if best is None else best[3]}
    if best is None or best[1] < need:
        return None, (0.0 if best is None else best[1]), dbg
    quad = best[0] if best[2] is None else refine_quad(best[2], best[0])
    return (quad / s).astype(np.float32), best[1], dbg


def find_paper(frame):
    """Backward-compatible wrapper: 4 corners in frame coordinates, or None."""
    return detect_paper(frame)[0]


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Debounced decision per slot
# ----------------------------------------------------------------------------
class Debounced:
    """Three-state decision per slot: EMPTY / OCCUPIED / UNCERTAIN.
    Smooths the ratio, then changes state only after `frames` consecutive votes (hysteresis:
    ON threshold above OFF threshold). Ratios between OFF and ON are 'no vote' (grey zone).
    Unreliable frames are skipped, so the previous stable state is preserved."""

    def __init__(self, on=0.12, off=0.05, frames=8, alpha=0.5, init_frames=None, uncertain_frames=None):
        self.on, self.off, self.frames, self.alpha = on, off, frames, alpha
        self.init_frames = init_frames or max(2, frames // 2)
        self.uncertain_frames = uncertain_frames or 2 * frames
        self.reset()

    def reset(self):
        self.ema, self.state, self.pending = None, None, None
        self.count = self.gray = self.stable = self.unreliable = 0
        self.conf = 0.0

    @property
    def occupied(self):
        return self.state == 1

    @property
    def status(self):
        if self.state is None or self.gray >= self.uncertain_frames or self.unreliable > 60:
            return "UNCERTAIN"
        return "OCCUPIED" if self.state == 1 else "EMPTY"

    def _confidence(self):
        r, on, off = self.ema, self.on, self.off
        if r >= on:
            c = 0.6 + 0.4 * min(1.0, (r - on) / max(on, 1e-6))
        elif r <= off:
            c = 0.6 + 0.4 * min(1.0, (off - r) / max(off, 1e-6))
        else:
            c = 0.2 + 0.35 * abs(r - (on + off) / 2) / ((on - off) / 2 + 1e-6)
        if self.state is not None:
            c *= 0.6 + 0.4 * min(1.0, self.stable / (2.0 * self.frames))
        return float(min(c, 1.0)) if self.state is not None else float(min(c, 0.5))

    def update(self, ratio, reliable=True):
        if not reliable or ratio is None:
            self.unreliable += 1                       # keep the previous stable state
            return self.occupied
        self.unreliable = 0
        self.ema = ratio if self.ema is None else self.alpha * ratio + (1 - self.alpha) * self.ema
        vote = 1 if self.ema >= self.on else (0 if self.ema <= self.off else None)
        self.gray = self.gray + 1 if vote is None else 0
        need = self.init_frames if self.state is None else self.frames
        if vote is not None and vote != self.state:
            self.count = self.count + 1 if vote == self.pending else 1
            self.pending = vote
            if self.count >= need:
                self.state, self.count, self.pending, self.stable = vote, 0, None, 0
        else:
            self.count, self.pending = 0, None
        self.stable += 1
        self.conf = self._confidence()
        return self.occupied


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


MOVE_THR = 0.02       # sheet corners moving more than this (fraction of diagonal) in one frame = camera/sheet moved
SETTLE = 6            # frames to wait after movement before trusting detections again
LOST_MAX = 15         # frames without a confident sheet before occupancy is suspended
ECC_OK, ECC_EVERY, EDGE_OK = 0.45, 12, 0.55   # min registration correlation vs baseline; frames between refinements
GAIN_LIMITS = (0.5, 2.2)   # lighting change beyond this = frame is not trusted
DISTURB_MAX = 0.12         # more than this fraction of the lane area changed = hand/occlusion, frame not trusted


class ParkingVision:
    def __init__(self, zones, on=0.12, off=0.05, frames=8):
        self.on, self.off, self.frames = on, off, frames
        self.set_zones(zones)
        self.corners = None
        self.locked = False
        self.swap = False
        self.aspect = None
        self.paper_found = False          # sheet confidently detected in THIS frame
        self.paper_conf = 0.0
        self.paper_dbg = None
        self.lost = 0
        self.pending, self.pending_n = None, 0
        self.settle = 0
        self.moved_hold = 0
        self.shift = 0.0
        self.frame_idx = 0
        self.baseline_lab = self.baseline_edges = self.baseline_img = None
        self.baseline_ok = False
        self.align, self.next_align = None, 0
        self.reg, self.reg_bad, self.need_orient = np.eye(3), 0, False
        self.band_k = np.ones((9, 9), np.uint8)

    # -- zones ---------------------------------------------------------------
    def set_zones(self, zones):
        self.zones = zones
        self.polys_px, self.masks, self.areas, self.regions, self.boxes = {}, {}, {}, {}, {}
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        union = np.zeros((SHEET_H, SHEET_W), np.uint8)
        cover = np.zeros((SHEET_H, SHEET_W), np.uint8)
        regs = {}
        for name, poly in zones:
            px = (np.float32(poly) * [SHEET_W, SHEET_H]).astype(np.int32)
            m = np.zeros((SHEET_H, SHEET_W), np.uint8)
            cv2.fillPoly(m, [px], 255)
            union |= m
            cover += (m > 0).astype(np.uint8)
            regs[name] = (px, m)
        overlap = cover > 1                                    # pixels claimed by two zones belong to nobody
        for name, (px, m) in regs.items():
            reg = (m > 0) & ~overlap
            cnt = cv2.erode(reg.astype(np.uint8) * 255, k) > 0   # keep away from the painted slot lines
            self.polys_px[name], self.regions[name], self.masks[name] = px, reg, cnt
            self.areas[name] = max(int(np.count_nonzero(cnt)), 1)
            x, y, bw, bh = cv2.boundingRect(px)
            self.boxes[name] = (max(x - 2, 0), max(y - 2, 0), min(x + bw + 2, SHEET_W), min(y + bh + 2, SHEET_H))
        probe = cv2.dilate(union, k, iterations=2) == 0   # lane / margin pixels, never covered by cars
        probe[:14, :] = probe[-14:, :] = False
        probe[:, :14] = probe[:, -14:] = False
        self.probe = probe
        ol = np.zeros((SHEET_H, SHEET_W), np.uint8)
        for n_, px_ in self.polys_px.items():
            if n_ != "ENTRY":
                cv2.polylines(ol, [px_], True, 255, 1)
        self.reg_mask = (probe.astype(np.uint8) * 255) | cv2.dilate(ol, np.ones((9, 9), np.uint8))
        self.slot_names = [n for n, _ in zones if n != "ENTRY"]
        # slot-specific thresholds: a toy car covers a fixed AREA, so bigger slots need a smaller fraction
        med = float(np.median([self.areas[n] for n in self.slot_names])) if self.slot_names else 1.0
        self.thr = {}
        for n, _ in zones:
            f = float(np.clip(med / self.areas[n], 0.6, 1.6))
            self.thr[n] = (self.on * f, self.off * f)
        self.states = {n: Debounced(*self.thr[n], frames=self.frames) for n, _ in zones}

    # -- sheet tracking --------------------------------------------------------
    @property
    def paper_state(self):
        if self.corners is None:
            return "SEARCHING"
        if self.locked or self.lost == 0:
            return "DETECTED"
        return "HOLDING" if self.lost <= LOST_MAX else "LOST"

    def update_paper(self, frame):
        self.shift = 0.0
        if self.locked:
            self.paper_found = self.corners is not None
            return
        run = self.corners is None or self.lost > 0 or self.pending is not None or self.frame_idx % 2 == 0
        if not run:
            return
        quad, conf, self.paper_dbg = detect_paper(frame, self.corners, keep=self.corners is not None)
        self.paper_found = quad is not None
        self.paper_conf = conf if quad is not None else 0.0
        if quad is None:
            self.lost += 1
            return
        quad, self.aspect = choose_corners(quad, frame.shape, self.corners, self.swap)
        if self.corners is None:
            self.corners, self.lost, self.next_align, self.need_orient = quad.astype(np.float32), 0, 0, True
            return
        diag = np.linalg.norm(self.corners[0] - self.corners[2])
        d = np.linalg.norm(quad - self.corners, axis=1).mean()
        if d < 0.15 * diag:
            self.pending, self.pending_n, self.lost = None, 0, 0
            self.shift = d / diag
            if self.shift > MOVE_THR:                       # moved: follow immediately, then re-verify
                self.settle, self.moved_hold, self.next_align, self.need_orient = SETTLE, 45, 0, True
                self.corners = quad.astype(np.float32)
            else:
                self.corners = (0.6 * self.corners + 0.4 * quad).astype(np.float32)   # smooth jitter
        else:                                               # a DIFFERENT region: accept only if it persists
            if self.pending is not None and np.linalg.norm(quad - self.pending, axis=1).mean() < 0.05 * diag:
                self.pending_n += 1
            else:
                self.pending, self.pending_n = quad, 1
            if self.pending_n >= 3 or self.lost > LOST_MAX:
                self.corners, self.pending, self.pending_n, self.lost = quad.astype(np.float32), None, 0, 0
                self.settle, self.moved_hold, self.next_align, self.need_orient = SETTLE, 45, 0, True
                self.shift = 1.0

    def flip_180(self):
        if self.corners is not None:
            self.corners = np.roll(self.corners, 2, axis=0)
            self.next_align, self.need_orient = 0, True

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

    def set_baseline(self, rect, trusted=True):
        """trusted=False (baseline loaded from disk): it must first prove it still lines up with the sheet."""
        self.baseline_img = rect.copy()
        self.baseline_lab, edges = self._prep(rect)
        self.baseline_edges = cv2.dilate(edges, np.ones((13, 13), np.uint8))   # +-6 px alignment tolerance
        self.base_e7 = cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0
        self.base_lo = cv2.erode(self.baseline_lab, self.band_k)
        self.base_hi = cv2.dilate(self.baseline_lab, self.band_k)
        self.baseline_ok = bool(trusted)
        g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        self.base_pyr = [self._ecc_prep(g, sc) for sc in (0.25, 0.5)]
        pm = self.reg_mask
        self.probe_pyr = [cv2.resize(pm, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST) for sc in (0.25, 0.5)]
        self.reg, self.reg_bad, self.need_orient = np.eye(3), 0, not trusted
        self.align, self.next_align = (1.0 if trusted else None), self.frame_idx + (ECC_EVERY if trusted else 0)
        for s_ in self.states.values():
            s_.reset()

    @staticmethod
    def _ecc_prep(gray, sc):
        return cv2.GaussianBlur(cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), (0, 0), 1.5).astype(np.float32)

    def _warp(self, frame, corners):
        return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners, DST), (SHEET_W, SHEET_H), flags=cv2.INTER_LINEAR)

    def _apply_reg(self, rect0):
        if np.allclose(self.reg, np.eye(3), atol=1e-6):
            return rect0
        return cv2.warpPerspective(rect0, self.reg, (SHEET_W, SHEET_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)

    def _align_score(self, rect):
        """Fallback check: fraction of the lane/margin line work that coincides with the baseline's (+-6 px)."""
        gray = cv2.cvtColor(cv2.GaussianBlur(rect, (5, 5), 0), cv2.COLOR_BGR2GRAY)
        e = (cv2.Canny(gray, 60, 140) > 0) & self.probe
        n = int(np.count_nonzero(e))
        if n < 150:
            return None
        return float(np.count_nonzero(e & (self.baseline_edges > 0)) / n)

    @staticmethod
    def _sane(W):
        c = np.float32([[0, 0], [SHEET_W, 0], [SHEET_W, SHEET_H], [0, SHEET_H]]).reshape(-1, 1, 2)
        try:
            return float(np.linalg.norm(cv2.perspectiveTransform(c, W).reshape(-1, 2) - c.reshape(-1, 2), axis=1).max()) < 70
        except cv2.error:
            return False

    def _ecc(self, rect, init):
        """Register the current top-down view to the baseline using only lane / printed-line pixels.
        Gain/offset invariant (lighting does not matter) and outlier-robust: pixels that disagree strongly
        with the baseline (a car touching a line, a hand) are masked out and the fit is repeated.
        Returns (correlation, W) or (None, init)."""
        g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        W = init.astype(np.float64).copy()
        crit = lambda n: (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n, 1e-4)
        inp = [self._ecc_prep(g, sc) for sc in (0.25, 0.5)]
        masks = [m.copy() for m in self.probe_pyr]
        cc = None

        def run(lvl, sc, n, motion=cv2.MOTION_AFFINE):
            nonlocal W, cc
            S = np.diag([sc, sc, 1.0])
            Ws = (S @ W @ np.linalg.inv(S))[:2].astype(np.float32)
            cc, Ws = cv2.findTransformECC(self.base_pyr[lvl], inp[lvl], Ws, motion, crit(n), masks[lvl], 5)
            W = np.linalg.inv(S) @ np.vstack([Ws.astype(np.float64), [0, 0, 1]]) @ S
            W /= W[2, 2]

        def drop_outliers(lvl, sc):
            S = np.diag([sc, sc, 1.0])
            Ws = (S @ W @ np.linalg.inv(S))[:2]
            h_, w_ = self.base_pyr[lvl].shape
            ia = cv2.warpAffine(inp[lvl], Ws, (w_, h_), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            b, m = self.base_pyr[lvl], self.probe_pyr[lvl] > 0
            if np.count_nonzero(m) < 200:
                return
            z = lambda x: (x - x[m].mean()) / (x[m].std() + 1e-6)
            r = np.abs(z(ia) - z(b))
            out = ((r > max(1.3, 3 * float(np.median(r[m])))) & m).astype(np.uint8) * 255
            keep = self.probe_pyr[lvl] & ~cv2.dilate(out, np.ones((9, 9), np.uint8))
            if np.count_nonzero(keep) > 0.35 * np.count_nonzero(m):
                masks[lvl] = keep

        try:
            run(0, 0.25, 50, cv2.MOTION_TRANSLATION)    # staged, so repeating structure (5 equal slots) cannot trap the fit
            run(0, 0.25, 50, cv2.MOTION_EUCLIDEAN)
            drop_outliers(1, 0.5)
            run(1, 0.5, 30, cv2.MOTION_EUCLIDEAN)
            run(1, 0.5, 30, cv2.MOTION_AFFINE)
            drop_outliers(1, 0.5)                       # re-judge with the better alignment, then final fit
            run(1, 0.5, 30, cv2.MOTION_AFFINE)

        except cv2.error as err:
            self.last_ecc_error = str(err)[:120]
            return None, init
        return float(cc), W

    def _maintain_registration(self, frame, rect0):
        """Keeps the live view locked to the baseline WITHOUT recapturing it: refines the sheet position
        every few frames, and after any movement / re-acquisition / 180 deg turn finds the correct orientation
        by correlating each candidate orientation with the baseline. Returns True if the orientation changed."""
        if self.settle > 0 or self.frame_idx < self.next_align:
            return False
        full = self.need_orient or not self.baseline_ok
        if not full:
            cc, W = self._ecc(rect0, self.reg)
            if cc is not None and cc >= ECC_OK and self._sane(W):
                self.reg, self.align, self.reg_bad = W, cc, 0
            elif (self._align_score(self._apply_reg(rect0)) or 0) >= EDGE_OK:      # fit failed but lines still coincide
                self.reg_bad = 0
            else:
                self.reg_bad += 1
                self.need_orient = self.need_orient or self.reg_bad >= 2
            self.next_align = self.frame_idx + (ECC_EVERY if self.reg_bad == 0 else 1)
            return False
        res, match = {}, {}
        for k in (0, 2, 1, 3):                          # 0 / 180 first; 90 only if neither fits
            if k in (1, 3) and any(v[0] >= ECC_OK for v in res.values()):
                continue
            rk = rect0 if k == 0 else self._warp(frame, np.roll(self.corners, k, axis=0))
            cc, W = self._ecc(rk, np.eye(3))
            if cc is None or not self._sane(W):
                es = self._align_score(rk)                  # image fit did not converge: plain line-coincidence test
                cc, W = (None, np.eye(3)) if es is None or es < EDGE_OK else (es, np.eye(3))
            elif cc < ECC_OK:
                es = self._align_score(rk) or 0
                if es >= EDGE_OK:
                    cc, W = es, np.eye(3)
            if cc is not None:
                res[k] = (cc, W)
                # Orientation evidence from the WHOLE sheet (printed P letters, arrows, hatching...), not only lanes:
                # how much of the aligned view's line work lies on the baseline's line work.
                ra = cv2.warpPerspective(rk, W, (SHEET_W, SHEET_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                e = cv2.Canny(cv2.cvtColor(cv2.GaussianBlur(ra, (5, 5), 0), cv2.COLOR_BGR2GRAY), 60, 140)[14:-14, 14:-14] > 0
                match[k] = float(np.count_nonzero(e & self.base_e7[14:-14, 14:-14]) / max(np.count_nonzero(e), 1))
        score = {k: res[k][0] + 0.5 * match.get(k, 0.0) for k in res}
        best = None
        if res:
            kb = max(score, key=score.get)
            if kb != 0 and 0 in score and score[kb] < score[0] + 0.05:
                kb = 0                                  # no clear winner: stay with the current orientation
            best = (kb, res[kb])
        if best is not None and best[1][0] >= ECC_OK:
            k, (cc, W) = best
            self.reg, self.align, self.baseline_ok, self.need_orient, self.reg_bad = W, cc, True, False, 0
            self.next_align = self.frame_idx + ECC_EVERY
            if k:
                self.corners = np.roll(self.corners, k, axis=0).astype(np.float32)
                for s_ in self.states.values():
                    s_.reset()
                return True
        else:
            self.baseline_ok = False
            self.align = best[1][0] if best else None
            self.next_align = self.frame_idx + 5
        return False

    # -- change detection ----------------------------------------------------------
    def _lighting_field(self, lab):
        """Smooth multiplicative lighting correction estimated from lane pixels only (cars never cover them)."""
        ref = self.baseline_lab
        ratio = ref[..., 0] / np.maximum(lab[..., 0], 1.0)
        sel = self.probe & (ref[..., 0] > 120) & (lab[..., 0] > 40)
        raw, shift = 1.0, np.zeros(2, np.float32)
        if np.count_nonzero(sel) > 500:
            raw = float(np.median(ratio[sel]))
            sel &= np.abs(ratio - raw) < 0.3 * raw            # ignore hands / cars driving in the lane
            if np.count_nonzero(sel) > 300:
                shift = np.array([np.median((ref[..., c] - lab[..., c])[sel]) for c in (1, 2)], np.float32)
        g = float(np.clip(raw, *GAIN_LIMITS))
        s = 4
        sz = (SHEET_W // s, SHEET_H // s)
        w = sel.astype(np.float32)
        num = cv2.GaussianBlur(cv2.resize(ratio * w, sz, interpolation=cv2.INTER_AREA), (0, 0), 10)
        den = cv2.GaussianBlur(cv2.resize(w, sz, interpolation=cv2.INTER_AREA), (0, 0), 10)
        eps = 0.05                                             # far from lane pixels the field relaxes to the global gain
        field = (num + eps * g) / (den + eps)
        field = np.clip(cv2.resize(field.astype(np.float32), (SHEET_W, SHEET_H), interpolation=cv2.INTER_LINEAR), *GAIN_LIMITS)
        return field, raw, shift

    def _diff(self, rect):
        lab, edges = self._prep(rect)
        field, raw_gain, ab_shift = self._lighting_field(lab)
        cur = lab.copy()
        cur[..., 0] *= field
        cur[..., 1] += ab_shift[0]
        cur[..., 2] += ab_shift[1]
        lo, hi = self.base_lo, self.base_hi
        # Tolerant difference: compare against the min..max band of the baseline in a 9x9 window,
        # so a 1-3 px alignment wobble of the painted lines is not mistaken for a car.
        excess = np.maximum(lo - cur, cur - hi)
        much_darker = cur[..., 0] < lo[..., 0] * DARK_RATIO - 6      # a soft shadow is lighter than this
        brighter = (cur[..., 0] - hi[..., 0]) > L_THR
        changed = much_darker | brighter | (np.maximum(excess[..., 1], excess[..., 2]) > AB_THR)
        new_edges = (edges > 0) & (self.baseline_edges == 0)            # texture: catches white-on-white cars
        k5, k9, k21 = (np.ones((5, 5), np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        mask = np.zeros((SHEET_H, SHEET_W), np.uint8)
        ratios, cover, slot_m = {}, {}, {}
        for name, _ in self.zones:                      # every slot is processed on its own crop: no leakage into neighbours
            x0, y0, x1, y1 = self.boxes[name]
            reg = self.regions[name][y0:y1, x0:x1]
            cnt = self.masks[name][y0:y1, x0:x1]
            ne = cv2.dilate((new_edges[y0:y1, x0:x1] & reg).astype(np.uint8) * 255, k5)
            ne = cv2.morphologyEx(ne, cv2.MORPH_CLOSE, k21)
            cs, _ = cv2.findContours(ne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:                                # fill outlines, even with gaps (flat light cars)
                hull = cv2.convexHull(c)
                if cv2.contourArea(hull) > 400 and min(cv2.minAreaRect(hull)[1]) >= 18:   # compact blob, not a thin line
                    cv2.drawContours(ne, [hull], -1, 255, cv2.FILLED)
            m = ((changed[y0:y1, x0:x1] & reg).astype(np.uint8) * 255) | ne
            m &= reg.astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k13)            # thin strokes (P letters, misaligned line slivers) vanish; cars survive
            m &= cnt.astype(np.uint8) * 255
            slot_m[name] = m
            mask[y0:y1, x0:x1] |= m
        # A tall toy car leans into the neighbouring slot (parallax). Join foreground pieces across the thin gaps
        # between slots; each joined object is credited to the slot holding most of it, and a neighbour keeps
        # its share only if it holds a substantial part of the object itself (two real cars stay two cars).
        _, comp = cv2.connectedComponents(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))))
        share = {}
        for name, m in slot_m.items():
            x0, y0, x1, y1 = self.boxes[name]
            ids, cn = np.unique(comp[y0:y1, x0:x1][m > 0], return_counts=True)
            share[name] = dict(zip(ids.tolist(), cn.tolist()))
        for name, m in slot_m.items():
            x0, y0, x1, y1 = self.boxes[name]
            for cid, cn in share[name].items():
                top = max(sh.get(cid, 0) for sh in share.values())
                if cn < 0.5 * top:                         # just the overhang of a car parked in another slot
                    m[(comp[y0:y1, x0:x1] == cid)] = 0
            n, _, stats, _ = cv2.connectedComponentsWithStats(m)
            blob = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0   # largest coherent foreground object
            ratios[name] = blob / self.areas[name]
            cover[name] = int(np.count_nonzero(m)) / self.areas[name]
            mask[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] & ~(self.regions[name][y0:y1, x0:x1].astype(np.uint8) * 255)) | m
        lane = (changed | cv2.dilate(new_edges.astype(np.uint8) * 255, k5).astype(bool)) & self.probe
        lane = cv2.morphologyEx(lane.astype(np.uint8) * 255, cv2.MORPH_OPEN, k5)
        disturb = float(np.count_nonzero(lane) / max(np.count_nonzero(self.probe), 1))
        mask |= lane
        gp = field[self.probe]
        return dict(mask=mask, ratios=ratios, cover=cover, gain=float(np.median(gp)), gain_min=float(gp.min()),
                    gain_max=float(gp.max()), raw_gain=raw_gain, disturb=disturb)

    # -- one frame ----------------------------------------------------------------
    def process(self, frame):
        t0 = time.time()
        self.frame_idx += 1
        self.update_paper(frame)
        names = [n for n, _ in self.zones]
        out = {"frame": frame, "paper_found": self.paper_found, "paper_state": self.paper_state, "paper_conf": self.paper_conf,
               "rect": None, "mask": None, "ratios": {}, "cover": {}, "occ": {n: False for n in names},
               "status": {n: "UNCERTAIN" for n in names}, "conf": {n: 0.0 for n in names},
               "stable": {n: 0 for n in names}, "thr": dict(self.thr), "gain": None, "H": None,
               "valid": False, "reason": "", "disturb": 0.0, "moved": self.moved_hold > 0 or self.settle > 0,
               "baseline_ok": self.baseline_ok, "align": self.align, "frame_ok": False}
        self.moved_hold = max(self.moved_hold - 1, 0)
        if self.corners is None:
            out["reason"] = "PAPER NOT DETECTED - show the whole white sheet"
            out["ms"] = (time.time() - t0) * 1000
            return out
        rect0 = self._warp(frame, self.corners)
        tracking = self.paper_state in ("DETECTED", "HOLDING")
        if tracking and self.baseline_lab is not None:
            if self._maintain_registration(frame, rect0):
                rect0 = self._warp(frame, self.corners)
        rect = self._apply_reg(rect0) if self.baseline_lab is not None else rect0
        H = cv2.getPerspectiveTransform(self.corners, DST)
        out.update(rect=rect, H=H, baseline_ok=self.baseline_ok, align=self.align)
        if not tracking:
            out["reason"] = "PAPER LOST - occupancy suspended"
            for s in self.states.values():
                s.reset()
        elif self.baseline_lab is None:
            out["reason"] = "BASELINE NOT SET - empty all slots, press B"
        elif not self.baseline_ok:
            out["reason"] = "BASELINE DOES NOT MATCH SHEET (moved / old) - empty the sheet, press B"
            for s in self.states.values():
                s.reset()
        else:
            d = self._diff(rect)
            out.update(mask=d["mask"], ratios=d["ratios"], cover=d["cover"], gain=d["gain"], disturb=d["disturb"],
                       gain_range=(d["gain_min"], d["gain_max"]), raw_gain=d["raw_gain"])
            reliable, why = True, ""
            if self.settle > 0:
                self.settle -= 1
                reliable, why = False, "CAMERA MOVING - settling"
            elif self.paper_state != "DETECTED":
                reliable, why = False, "paper briefly lost - holding last state"
            elif not (GAIN_LIMITS[0] <= d["raw_gain"] <= GAIN_LIMITS[1]):
                reliable, why = False, "LIGHTING changed too much"
            elif d["disturb"] > DISTURB_MAX:
                reliable, why = False, "large change in lanes (hand / occlusion)"
            out["frame_ok"] = reliable
            for n in names:
                st = self.states[n]
                st.update(d["ratios"][n], reliable)
                out["occ"][n] = st.status == "OCCUPIED"
                out["status"][n], out["conf"][n], out["stable"][n] = st.status, st.conf, st.stable
            out["valid"] = reliable
            out["reason"] = why
        out["ms"] = (time.time() - t0) * 1000
        return out

    # -- digital logic ------------------------------------------------------------------
    def logic(self, occ, status=None, valid=True):
        """Uncertain slots are counted as neither occupied nor free."""
        if status is None:
            status = {n: ("OCCUPIED" if occ.get(n) else "EMPTY") for n in occ}
        st = [status.get(n, "UNCERTAIN") for n in self.slot_names]
        bits = [int(s == "OCCUPIED") for s in st]
        occupied, total = sum(bits), len(bits)
        free = sum(s == "EMPTY" for s in st)
        unc = sum(s == "UNCERTAIN" for s in st)
        full = int(total > 0 and occupied == total)
        entry = int(status.get("ENTRY") == "OCCUPIED")
        return {"bits": bits, "occupied": occupied, "available": total - occupied, "free_confirmed": free,
                "uncertain": unc, "total": total, "full": full, "entry": entry, "valid": bool(valid),
                "gate_open": int(bool(valid) and entry and not full and free > 0)}


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
        self.frame, self.ok, self.running, self.last_ok = None, False, True, 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            self.ok = ok
            if ok:
                self.frame, self.last_ok = frame, time.time()
            else:
                time.sleep(0.05)

    def status(self):
        if self.image is not None:
            return "IMAGE FILE"
        if self.frame is None:
            return "CONNECTING"
        return "CONNECTED" if time.time() - self.last_ok < 2.0 else "NO SIGNAL"

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


STATUS_COLOR = {"EMPTY": GREEN, "OCCUPIED": RED, "UNCERTAIN": YELLOW}


def zone_color(name, status):
    s = status.get(name, "UNCERTAIN")
    if name == "ENTRY" and s == "EMPTY":
        return GREY
    return CYAN if (name == "ENTRY" and s == "OCCUPIED") else STATUS_COLOR[s]


def draw_zones(img, polys, status, ratios, alpha=0.30, labels=True):
    over = img.copy()
    for name, poly in polys.items():
        cv2.fillPoly(over, [poly.astype(np.int32)], zone_color(name, status))
    img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for name, poly in polys.items():
        cv2.polylines(img, [poly.astype(np.int32)], True, zone_color(name, status), 2, cv2.LINE_AA)
        if labels:
            cx, cy = poly.mean(axis=0).astype(int)
            tag = name if name not in ratios else f"{name} {ratios[name] * 100:.0f}%"
            text(img, tag, (cx - 28, cy + 4), 0.45)
    return img


def build_dashboard(vision, out, fps, source_name, cam_status="CONNECTED"):
    status, ratios = out["status"], out["ratios"]
    lg = vision.logic(out["occ"], status, out["valid"])
    pstate = out["paper_state"]
    p_ok = pstate in ("DETECTED", "HOLDING")

    # Panel A: camera view with detected paper outline, corners, FPS and connection status
    cam = out["frame"].copy()
    if vision.corners is not None:
        col = YELLOW if pstate == "DETECTED" else (0, 140, 255)
        cv2.polylines(cam, [vision.corners.astype(np.int32)], True, col, 3, cv2.LINE_AA)
        for i, p in enumerate(vision.corners):
            cv2.circle(cam, tuple(int(v) for v in p), 7, col, -1)
            text(cam, "TL TR BR BL".split()[i], (int(p[0]) + 8, int(p[1]) - 8), 0.6, col, 2)
        if out["H"] is not None and p_ok:
            Hinv = np.linalg.inv(out["H"])
            polys = {n: cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)
                     for n, p in vision.polys_px.items()}
            cam = draw_zones(cam, polys, status, {}, 0.25, labels=False)
            for n, p in polys.items():
                cx, cy = p.mean(axis=0).astype(int)
                text(cam, n, (cx - 12, cy + 5), 0.55, WHITE, 2)
    else:
        dbg = vision.paper_dbg and vision.paper_dbg.get("best_rejected")
        if dbg is not None:                                  # show what was rejected and why
            cv2.polylines(cam, [dbg[0].astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
            text(cam, f"rejected ({dbg[1]:.2f}): {dbg[2]}", (10, cam.shape[0] - 14), 0.7, (0, 120, 255), 2)
    if not p_ok:
        cv2.rectangle(cam, (0, 0), (cam.shape[1], 46), (0, 0, 160), -1)
        text(cam, "WARNING: PAPER NOT DETECTED - occupancy suspended", (12, 32), 0.9, WHITE, 2)
    text(cam, f"{fps:.0f} FPS   camera: {cam_status}", (12, cam.shape[0] - 40), 0.8,
         GREEN if cam_status in ("CONNECTED", "IMAGE FILE") else RED, 2)
    panelA = fit(cam, 640, 360, "1) CAMERA - detected paper (yellow), corners, slots projected back")

    # Panel B: flat top-down view
    rb = draw_zones(out["rect"], vision.polys_px, status, ratios) if out["rect"] is not None else None
    panelB = fit(rb, 320, 453, "2) TOP-DOWN VIEW  green=empty red=occ yellow=?")

    # Panel C: change mask
    if out["mask"] is not None:
        heat = cv2.cvtColor(out["mask"], cv2.COLOR_GRAY2BGR)
        heat[out["mask"] > 0] = (60, 90, 255)
        for n, p in vision.polys_px.items():
            cv2.polylines(heat, [p], True, zone_color(n, status), 2, cv2.LINE_AA)
        panelC = fit(heat, 320, 453, "3) FOREGROUND vs EMPTY BASELINE")
    else:
        panelC = fit(vision.baseline_img, 320, 453, "3) no comparison (see status)")

    # Panel D: system status + logic
    d = np.full((360, 640, 3), 30, np.uint8)
    text(d, "SYSTEM STATUS", (8, 18), 0.55, YELLOW, 1)
    conf_s = f" conf {out['paper_conf']:.2f}" if out["paper_conf"] else ""
    rows = [
        ("Paper detection", ("DETECTED" + conf_s) if pstate == "DETECTED" else ("NOT DETECTED (" + pstate + ")"), GREEN if pstate == "DETECTED" else RED),
        ("Perspective correction", "ACTIVE" if (p_ok and out["rect"] is not None) else "FAILED", GREEN if p_ok else RED),
        ("Baseline", "NOT SET" if vision.baseline_lab is None else (f"SET, aligned {out['align']:.2f}" if out["baseline_ok"] else "SET but MISALIGNED"),
         RED if (vision.baseline_lab is None or not out["baseline_ok"]) else GREEN),
        ("Camera movement", "MOVED - re-checking" if out["moved"] else "STABLE", YELLOW if out["moved"] else GREEN),
    ]
    for i, (k, v, c) in enumerate(rows):
        text(d, f"{k}:", (8, 42 + i * 22), 0.5, GREY)
        text(d, v, (215, 42 + i * 22), 0.5, c, 1)
    if not p_ok or vision.baseline_lab is None or not out["baseline_ok"]:
        overall, oc = "SUSPENDED", RED
    elif not out["valid"]:
        overall, oc = "HOLDING LAST STATE", YELLOW
    elif lg["uncertain"]:
        overall, oc = f"PARTIAL ({lg['uncertain']} uncertain)", YELLOW
    else:
        overall, oc = "OK", GREEN
    text(d, "Overall:", (8, 130), 0.55, GREY)
    text(d, overall, (215, 130), 0.55, oc, 1)
    if out["reason"]:
        text(d, out["reason"][:70], (8, 152), 0.5, (0, 160, 255), 1)
    if out["baseline_ok"] and vision.baseline_lab is not None and p_ok:
        unc = f"  ({lg['uncertain']} uncertain not counted)" if lg["uncertain"] else ""
        text(d, f"OCCUPIED = {' + '.join(str(b) for b in lg['bits'])} = {lg['occupied']}", (8, 182), 0.5)
        text(d, f"AVAILABLE = {lg['total']} - {lg['occupied']} = {lg['available']}   confirmed free: {lg['free_confirmed']}{unc}", (8, 204), 0.5,
             GREEN if lg["free_confirmed"] else RED, 1)
        text(d, f"FULL = AND(all {lg['total']} OCCUPIED) = {lg['full']}", (8, 226), 0.5, RED if lg["full"] else WHITE)
        text(d, f"ENTRY detected = {lg['entry']}   GATE OPEN = ENTRY AND NOT FULL = {lg['gate_open']}", (8, 248), 0.5,
             GREEN if lg["gate_open"] else WHITE, 1)
        if not lg["valid"]:
            text(d, "gate forced CLOSED: frame not trusted", (8, 270), 0.5, RED)
    else:
        text(d, "Occupancy / gate logic suspended", (8, 204), 0.6, RED, 1)
    text(d, f"Source: {source_name}   {out.get('ms', 0):.0f} ms/frame", (8, 316), 0.42, GREY)
    text(d, "B baseline  L lock  F flip180  R rot90  E edit slots  S shot  Q quit", (8, 350), 0.42, GREY)
    panelD = fit(d, 640, 360, None)

    # Panel E: per-slot analysis
    e = np.full((267, 640, 3), 38, np.uint8)
    gain = out["gain"]
    gl = f"{gain:.2f} (field {out['gain_range'][0]:.2f}..{out['gain_range'][1]:.2f})" if (gain and "gain_range" in out) else "waiting"
    text(e, f"DETECTION ANALYSIS   lighting gain {gl}", (6, 15), 0.45, YELLOW)
    text(e, f"lane disturbance {out['disturb'] * 100:.1f}% (max {DISTURB_MAX * 100:.0f}%)   frame trusted: {'yes' if out['frame_ok'] else 'NO'}", (6, 32), 0.42,
         GREEN if out["frame_ok"] else (0, 160, 255))
    for i, name in enumerate(vision.slot_names):
        col, row = divmod(i, 5)
        x0, y = 6 + col * 318, 46 + row * 36
        st = status[name]
        r, c = ratios.get(name, 0.0), STATUS_COLOR[st]
        on, off = out["thr"][name]
        text(e, name, (x0, y + 11), 0.45)
        cv2.rectangle(e, (x0 + 26, y), (x0 + 126, y + 12), (70, 70, 70), -1)
        cv2.rectangle(e, (x0 + 26, y), (x0 + 26 + int(min(r, 1) * 100), y + 12), c, -1)
        for t in (on, off):
            cv2.line(e, (x0 + 26 + int(min(t, 1) * 100), y - 2), (x0 + 26 + int(min(t, 1) * 100), y + 14), YELLOW, 1)
        text(e, f"{r * 100:3.0f}%  {st}", (x0 + 132, y + 11), 0.42, c)
        text(e, f"conf {out['conf'][name]:.2f}  stable {out['stable'][name]}f  thr {on * 100:.0f}/{off * 100:.0f}%", (x0 + 26, y + 27), 0.36, GREY)
    text(e, f"ENTRY {status['ENTRY']}  {ratios.get('ENTRY', 0) * 100:.0f}%   (yellow ticks = ON/OFF thresholds, debounce {vision.frames}f)", (6, 236), 0.4,
         STATUS_COLOR[status["ENTRY"]])
    text(e, "ratio = largest foreground object / slot area (baseline diff + texture)", (6, 256), 0.38, GREY)
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
        vision.set_baseline(cv2.resize(cv2.imread(args.baseline), (SHEET_W, SHEET_H)), trusted=False)
        print(f"Loaded baseline {args.baseline}; it is used only if it still lines up with the sheet (press B to replace it)")
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
        dash = build_dashboard(vision, out, fps, args.source, src.status())
        if args.scale != 1.0:
            dash = cv2.resize(dash, None, fx=args.scale, fy=args.scale)
        cv2.imshow(win, dash)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("b"), ord("B")):
            if out["rect"] is not None and out["paper_state"] == "DETECTED":
                vision.set_baseline(out["rect"])
                cv2.imwrite(args.baseline, out["rect"])
                print("Baseline saved ->", args.baseline)
            else:
                print("Baseline NOT captured: the paper is not confidently detected right now")
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
