"""Orange-dot anchored parking detector: keeps working when the phone moves.

Why this exists
  * The 12 round orange dots (line ends next to the lanes) are fixed landmarks printed on the sheet.
    Every frame they are found and matched to a template, which gives an exact homography
    (sheet -> camera). Camera moves, zooms or tilts are followed frame by frame.
  * Because the sheet is always mapped back to the SAME reference view, the empty-sheet baseline
    never has to be re-captured. The photo you made the template from is the default baseline.
  * Slot polygons come from the dots, so slot margins are exact and identical every time.
  * Each changed blob (car) is given to ONE slot (the one it mostly covers), so a car sitting a bit
    over a line no longer lights up the neighbour.

Steps
  1. Template (once, from a photo of the EMPTY sheet, whole sheet in view):
         python parking_dots.py --make-template empty_sheet.jpg
  2. Run (phone stream / webcam index / photo):
         python parking_dots.py --source http://PHONE_IP:8080/video

Keys: B = use the current view as baseline (sheet empty)   R = back to the template baseline
      S = screenshot   Q = quit
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parking_vision import FrameSource, text, fit, Debounced, GREEN, RED, YELLOW, CYAN, WHITE, GREY  # noqa: E402

CAN_W = 640                       # width of the reference ("canonical") sheet view
DOT_H = (3, 24)                   # orange hue range (OpenCV hue is 0..179)
DOT_S, DOT_DS, DOT_V = 55, 35, 50  # min saturation, min saturation advantage over the surroundings, min brightness

L_THR, DARK_RATIO, AB_THR = 45, 0.62, 22
SPILL = 0.6                       # a neighbouring slot only counts if the blob covers it >= SPILL x the main slot
MIN_INLIERS_TRACK, MIN_INLIERS_INIT = 6, 8
DISTURB_MAX = 0.12                # more than this fraction of the lane area changed = hand / occlusion
ALIGN_MIN = 0.5                   # share of the sheet's fixed lines (lanes/divider) that must line up with the template
ALIGN_BAD_FRAMES = 6              # this many misaligned frames in a row = forget the pose and search again


# ----------------------------------------------------------------------------
# Orange dots
# ----------------------------------------------------------------------------
def detect_dots(frame, mode="track"):
    """Centres of orange dots as an (N, 5) array: x, y, area (frame px), aspect ratio, contrast score.
    1) white balance from the brightest pixels (paper), so a tinted camera does not matter;
    2) orange hue + saturation that stands out from its surroundings (works dim, tilted, tinted);
    3) fragments of a dot split by a pen line are merged.
    mode: "strict" = round dots only (template), "track" = allows the squashed look of a low
    camera angle, "relaxed" = also thin smears at the paper edge."""
    h, w = frame.shape[:2]
    s = min(1.0, 1280.0 / max(h, w))                       # keep full detail: dots are only a few pixels wide
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    q = small[::4, ::4].reshape(-1, 3).astype(np.float32)
    br = q.sum(axis=1)
    top = q[br >= np.percentile(br, 80)]
    gains = np.clip(top.mean() / np.maximum(top.mean(axis=0), 1.0), 0.6, 1.6)
    wb = np.clip(small.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    contrast = S - cv2.blur(S, (41, 41))
    m = ((hsv[..., 0] >= DOT_H[0]) & (hsv[..., 0] <= DOT_H[1]) & (S >= DOT_S) & (contrast >= DOT_DS) & (hsv[..., 2] >= DOT_V))
    m = m.astype(np.uint8) * 255
    L = float(max(small.shape[:2]))
    k = max(3, int(round(0.006 * L)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_aspect, min_fill = {"strict": (2.0, 0.5), "track": (3.2, 0.4), "relaxed": (6.0, 0.3)}[mode]
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        if a < 5 or not (0.003 * L <= max(bw, bh) <= 0.05 * L) or min(bw, bh) < 2:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > max_aspect or a / float(bw * bh) < min_fill:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 0:
            continue
        cm = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(cm, [c - [x, y]], -1, 255, -1)
        score = float(contrast[y:y + bh, x:x + bw][cm > 0].mean())
        out.append((M["m10"] / M["m00"] / s, M["m01"] / M["m00"] / s, a / (s * s), max(bw, bh) / float(max(1, min(bw, bh))), score))
    return np.float32(out).reshape(-1, 5)


def _kmeans_1d(x, k=4, iters=30):
    c = np.quantile(x, np.linspace(0.1, 0.9, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        for i in range(k):
            if np.any(lab == i):
                c[i] = x[lab == i].mean()
    order = np.argsort(c)
    remap = {int(o): i for i, o in enumerate(order)}
    return np.array([remap[int(v)] for v in np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)])


def _merge_close(pts, dist):
    pts = [np.float64(p) for p in sorted(pts, key=lambda p: p[1])]
    out = []
    for p in pts:
        if out and np.linalg.norm(out[-1][0] - p) < dist:
            out[-1].append(p)
            out[-1][0] = np.mean(out[-1][1:], axis=0)
        else:
            out.append([p, p])
    return [o[0] for o in out]


# ----------------------------------------------------------------------------
# Template from a photo of the empty sheet
# ----------------------------------------------------------------------------
def build_template(photo_path, json_path, png_path):
    img = cv2.imread(photo_path)
    if img is None:
        sys.exit(f"Could not read {photo_path}")
    h0, w0 = img.shape[:2]
    inner_all = detect_dots(img, "strict")
    relaxed = detect_dots(img, "relaxed")
    if len(inner_all) < 12:
        sys.exit(f"Found only {len(inner_all)} round orange dots (need 12). Use a sharper, brighter photo of the whole sheet.")
    cl = _kmeans_1d(relaxed[:, 0])
    cl_s = _kmeans_1d(inner_all[:, 0])
    cols = {i: inner_all[cl_s == i, :2] for i in range(4)}
    rows = {}
    for side, ci in (("L", 1), ("R", 2)):
        pts = _merge_close([tuple(p) for p in cols[ci]], 0.03 * h0)
        if len(pts) != 6:
            dbg = img.copy()
            for p in inner_all:
                cv2.circle(dbg, (int(p[0]), int(p[1])), 16, (0, 0, 255), 3)
            cv2.imwrite("template_debug.png", dbg)
            sys.exit(f"Expected 6 round dots next to the {'left' if side == 'L' else 'right'} lane, found {len(pts)}. See template_debug.png.")
        rows[side] = np.float32(sorted(pts, key=lambda p: p[1]))
    outer = {"L": relaxed[cl == 0, :2], "R": relaxed[cl == 3, :2]}
    out_pts = {}
    for side in "LR":
        found, offs = [], []
        for p in rows[side]:
            cand = outer[side][np.abs(outer[side][:, 1] - p[1]) < 0.04 * h0] if len(outer[side]) else []
            if len(cand):
                q = cand[np.argmin(np.abs(cand[:, 1] - p[1]))]
                found.append(q)
                offs.append(q - p)
            else:
                found.append(None)
        mean_off = np.mean(offs, axis=0) if offs else np.float32([-0.27 * w0 if side == "L" else 0.27 * w0, 0])
        out_pts[side] = np.float32([f if f is not None else p + mean_off for f, p in zip(found, rows[side])])

    s = CAN_W / float(w0)
    canon = cv2.resize(img, (CAN_W, int(round(h0 * s))), interpolation=cv2.INTER_AREA)
    CH = canon.shape[0]
    inset = 0.016 * CAN_W
    zones = []
    for side in "LR":
        inn, out = rows[side] * s, out_pts[side] * s
        for i in range(5):
            o0, o1 = out[i].copy(), out[i + 1].copy()
            for o, iv in ((o0, inn[i]), (o1, inn[i + 1])):                 # pull the paper-edge ends inwards a little
                d = iv - o
                o += d / (np.linalg.norm(d) + 1e-6) * inset
            poly = [o0, inn[i], inn[i + 1], o1] if side == "L" else [inn[i], o0, o1, inn[i + 1]]
            zones.append({"name": f"{side}{i + 1}", "poly": [[float(x), float(y)] for x, y in poly]})
    # locate the painted "P" in every slot: the biggest dark ink blob in the middle of the slot
    gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
    for z in zones:
        poly = np.int32(z["poly"])
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(m, [poly], 255)
        inner = cv2.erode(m, np.ones((31, 31), np.uint8)) > 0
        paper_lvl = float(np.percentile(gray[inner], 80)) if inner.any() else 200.0
        ink = ((gray < 0.62 * paper_lvl) & inner).astype(np.uint8) * 255
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            a = cv2.contourArea(c)
            if 120 <= a and 10 <= w <= 0.5 * CAN_W * 0.4 and 10 <= h <= 0.4 * CH and (best is None or a > best[0]):
                best = (a, [int(x), int(y), int(w), int(h)])
        z["p_box"] = None if best is None else best[1]
    anchors = np.vstack([rows["L"], rows["R"]]) * s
    allpts = np.vstack([relaxed[:, :2] * s])
    paper = cv2.convexHull(allpts.astype(np.float32)).reshape(-1, 2)
    markers = []                                   # ArUco corner stickers stuck on this sheet (see make_marker_stickers.py)
    if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        mc, mi, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        seen = set()
        for c, i in zip(mc, [] if mi is None else mi.ravel()):
            if 0 <= int(i) <= 3 and int(i) not in seen:
                seen.add(int(i))
                markers.append({"id": int(i), "corners": (c.reshape(4, 2) * s).tolist()})
    ids_found = sorted(m["id"] for m in markers)
    print(f"ArUco markers found in the photo: {ids_found if ids_found else 'none'}"
          + ("" if len(ids_found) >= 3 else "  (stick the 4 corner markers on the sheet and retake the photo for much more robust tracking)"))
    tpl = {"version": 1, "canon": [CAN_W, CH], "anchors": anchors.tolist(),
           "zones": zones, "paper": paper.tolist(), "markers": markers, "source_photo": os.path.basename(photo_path)}
    with open(json_path, "w") as f:
        json.dump(tpl, f, indent=1)
    cv2.imwrite(png_path, canon)
    prev = canon.copy()
    for z in zones:
        cv2.polylines(prev, [np.int32(z["poly"])], True, (0, 200, 0), 2)
        c = np.mean(z["poly"], axis=0)
        text(prev, z["name"], (int(c[0]) - 12, int(c[1]) + 5), 0.6)
    for p in anchors:
        cv2.circle(prev, (int(p[0]), int(p[1])), 9, (0, 0, 255), 2)
    cv2.imwrite("template_preview.png", prev)
    print(f"Template saved: {json_path}, {png_path}  (check template_preview.png)")
    return tpl


def build_template_from_layout(layout_path, photo_path, json_path, png_path, sheet_png=None):
    """Template for the PRINTED sheet. The geometry (dots, slots, P boxes) is exact, read from the layout file
    that make_printable_sheet.py wrote. With a photo of the empty sheet the photo is straightened and used as
    the reference picture (best match to your camera); with photo "-" the printed design itself is the reference."""
    with open(layout_path) as f:
        lay = json.load(f)
    pw, ph = lay["paper_mm"]
    s = CAN_W / float(pw)
    CH = int(round(ph * s))
    anchors = np.float32([[x * s, y * s] for x, y in lay["dots_left"]] + [[x * s, y * s] for x, y in lay["dots_right"]])
    src_name = "printed design"
    if photo_path and photo_path != "-":
        img = cv2.imread(photo_path)
        if img is None:
            sys.exit(f"Could not read {photo_path}")
        trk = SheetTracker(anchors)
        if not trk.update(detect_dots(img), img.shape, (CAN_W, CH)):
            cv2.imwrite("template_debug.png", img)
            sys.exit("Could not find all 12 orange dots in the photo. Use a sharp, evenly lit photo of the WHOLE empty sheet.")
        canon = cv2.warpPerspective(img, trk.H, (CAN_W, CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        src_name = os.path.basename(photo_path)
    else:
        sheet_png = sheet_png or layout_path.replace("_layout.json", ".png")
        sheet = cv2.imread(sheet_png)
        if sheet is None:
            sys.exit(f"Could not read the sheet image {sheet_png}")
        canon = cv2.resize(sheet, (CAN_W, CH), interpolation=cv2.INTER_AREA)
    inset = 0.016 * CAN_W
    zones = []
    for z in lay["slots"]:
        poly = np.float32(z["poly"]) * s
        outer, inner = ((0, 3), (1, 2)) if z["name"].startswith("L") else ((1, 2), (0, 3))
        for o, i in zip(outer, inner):                                   # pull the outer ends in a little
            dvec = poly[i] - poly[o]
            poly[o] += dvec / (np.linalg.norm(dvec) + 1e-6) * inset
        box = lay["p_boxes"].get(z["name"])
        zones.append({"name": z["name"], "poly": poly.tolist(), "p_box": None if box is None else [int(round(v * s)) for v in box]})
    allp = np.vstack([np.float32(z["poly"]) for z in zones])
    x0, y0, x1, y1 = allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    paper = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    markers = [{"id": int(m["id"]), "corners": (np.float32(m["corners"]) * s).tolist()} for m in lay.get("markers", [])]
    tpl = {"version": 1, "canon": [CAN_W, CH], "anchors": anchors.tolist(), "zones": zones, "paper": paper,
           "markers": markers, "source_photo": src_name}
    with open(json_path, "w") as f:
        json.dump(tpl, f, indent=1)
    cv2.imwrite(png_path, canon)
    prev = canon.copy()
    for z in zones:
        cv2.polylines(prev, [np.int32(z["poly"])], True, (0, 200, 0), 2)
        if z["p_box"]:
            bx, by, bw, bh = z["p_box"]
            cv2.rectangle(prev, (bx - 3, by - 3), (bx + bw + 3, by + bh + 3), (0, 0, 255), 1)
    for p in anchors:
        cv2.circle(prev, (int(p[0]), int(p[1])), 9, (255, 0, 0), 2)
    cv2.imwrite("template_preview.png", prev)
    print(f"Template saved: {json_path}, {png_path}  (reference: {src_name}; check template_preview.png)")
    return tpl


# ----------------------------------------------------------------------------
# Sheet tracker: dots -> homography (canonical sheet -> camera frame)
# ----------------------------------------------------------------------------
def _hull_quad(pts):
    hull = cv2.convexHull(np.ascontiguousarray(pts, dtype=np.float32)).reshape(-1, 1, 2)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _clockwise(q):
    c = q.mean(axis=0)
    return q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]


class SheetTracker:
    def __init__(self, anchors):
        self.anchor = np.float32(anchors)                     # canonical coordinates
        self.corner_idx = [0, 6, 11, 5]                       # L top, R top, R bottom, L bottom
        self.H = None                                         # canonical -> frame
        self.lost = 0
        self.inliers = 0
        self.rms = None
        self.matched = np.zeros(len(self.anchor), bool)
        self.mode = "SEARCHING"
        self.cooldown = 0
        self.did_init = False
        self.pose_scorer = None                        # set by the engine: (frame, H) -> (line match, structure correlation)
        self.frame_for_score = None

    @staticmethod
    def _match(proj, pts, r):
        pairs = {}
        for i, p in enumerate(proj):
            d = np.linalg.norm(pts - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < r and (j not in pairs or d[j] < pairs[j][1]):
                pairs[j] = (i, d[j])
        return [(i, j) for j, (i, _) in pairs.items()]

    def _refine(self, H, pts, radii, L):
        tol = max(3.0, 0.006 * L)
        for r in radii:
            proj = cv2.perspectiveTransform(self.anchor.reshape(-1, 1, 2), H).reshape(-1, 2)
            pairs = self._match(proj, pts, r)
            if len(pairs) < 5:
                return None
            A = self.anchor[[i for i, _ in pairs]]
            B = pts[[j for _, j in pairs]]
            Hn, _ = cv2.findHomography(A, B, cv2.RANSAC, max(2.5, 0.003 * L))
            if Hn is None:
                return None
            H = Hn
        proj = cv2.perspectiveTransform(self.anchor.reshape(-1, 1, 2), H).reshape(-1, 2)
        pairs = [(i, j) for i, j in self._match(proj, pts, tol)]
        if len(pairs) < 5:
            return None
        d = np.array([np.linalg.norm(proj[i] - pts[j]) for i, j in pairs])
        matched = np.zeros(len(self.anchor), bool)
        matched[[i for i, _ in pairs]] = True
        return H, len(pairs), float(np.sqrt(np.mean(d ** 2))), matched

    def _sane(self, H, shape, can):
        h, w = shape[:2]
        c = np.float32([[0, 0], [can[0], 0], [can[0], can[1]], [0, can[1]]]).reshape(-1, 1, 2)
        q = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        if not np.all(np.isfinite(q)) or not cv2.isContourConvex(q.reshape(-1, 1, 2)):
            return False
        a = cv2.contourArea(q) / float(h * w)
        return 0.01 < a < 6.0

    ROW_SETS = [(0, 5), (0, 4), (1, 5), (0, 3), (2, 5), (1, 4)]   # which template rows act as the 4 "corner" dots

    @staticmethod
    def _batch_homography(src, dst):
        """Homographies for M quads at once. src (4,2) shared, dst (M,4,2). Returns (M,3,3)."""
        M = dst.shape[0]
        A = np.zeros((M, 8, 8), np.float64)
        bb = np.zeros((M, 8), np.float64)
        for i in range(4):
            x, y = src[i]
            u, v = dst[:, i, 0], dst[:, i, 1]
            A[:, 2 * i, 0], A[:, 2 * i, 1], A[:, 2 * i, 2] = x, y, 1
            A[:, 2 * i, 6], A[:, 2 * i, 7] = -u * x, -u * y
            A[:, 2 * i + 1, 3], A[:, 2 * i + 1, 4], A[:, 2 * i + 1, 5] = x, y, 1
            A[:, 2 * i + 1, 6], A[:, 2 * i + 1, 7] = -v * x, -v * y
            bb[:, 2 * i], bb[:, 2 * i + 1] = u, v
        A += np.eye(8)[None] * 1e-9
        h = np.linalg.solve(A, bb[:, :, None])[:, :, 0]
        H = np.ones((M, 9), np.float64)
        H[:, :8] = h
        return H.reshape(M, 3, 3)

    def _init_global(self, pts, shape, can):
        """No previous pose. Treat every convex 4-dot subset as 4 template dots (a few row choices x
        4 rotations), score all hypotheses at once (numpy), refine the best few. Extra orange things are
        ignored; the hand-drawn asymmetry of the dots lets only the correct hypothesis fit within a few
        pixels. Needs about 8 of the 12 dots visible."""
        from itertools import combinations
        L = float(max(shape[:2]))
        n = len(pts)
        if n < MIN_INLIERS_INIT:
            return None
        combs = np.array(list(combinations(range(n), 4)), np.int32)
        Q = pts[combs].astype(np.float64)                                   # (K,4,2)
        cen = Q.mean(axis=1, keepdims=True)
        ang = np.arctan2(Q[..., 1] - cen[..., 1], Q[..., 0] - cen[..., 0])
        Q = np.take_along_axis(Q, np.argsort(ang, axis=1)[..., None], axis=1)   # clockwise
        e = np.roll(Q, -1, axis=1) - Q
        cr = e[:, :, 0] * np.roll(e, -1, axis=1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, axis=1)[:, :, 0]
        convex = np.all(cr > 0, axis=1) | np.all(cr < 0, axis=1)
        area = 0.5 * np.abs(np.sum(Q[:, :, 0] * np.roll(Q[:, :, 1], -1, axis=1) - np.roll(Q[:, :, 0], -1, axis=1) * Q[:, :, 1], axis=1))
        Q = Q[convex & (area > 0.01 * L * L)]
        if len(Q) == 0:
            return None
        rough = 0.015 * L
        cands = []
        for (ra, rb) in self.ROW_SETS:
            ctpl = self.anchor[[ra, 6 + ra, 6 + rb, rb]]                    # L_a, R_a, R_b, L_b  (clockwise)
            t_ratio = np.linalg.norm(ctpl[1] - ctpl[2]) / max(np.linalg.norm(ctpl[0] - ctpl[1]), 1e-6)
            for rot in range(4):
                Qr = np.roll(Q, -rot, axis=1)
                wide = np.linalg.norm(Qr[:, 0] - Qr[:, 1], axis=1) + np.linalg.norm(Qr[:, 2] - Qr[:, 3], axis=1)
                tall = np.linalg.norm(Qr[:, 1] - Qr[:, 2], axis=1) + np.linalg.norm(Qr[:, 3] - Qr[:, 0], axis=1)
                ok = (wide > 1e-3) & (tall / np.maximum(wide, 1e-3) > 0.3 * t_ratio) & (tall / np.maximum(wide, 1e-3) < 3.5 * t_ratio)
                if not np.any(ok):
                    continue
                Qs = Qr[ok]
                try:
                    Hs = self._batch_homography(ctpl, Qs)
                except np.linalg.LinAlgError:
                    continue
                P = np.einsum("mij,kj->mki", Hs, np.hstack([self.anchor, np.ones((len(self.anchor), 1))]))
                w = P[..., 2:3]
                good = np.all(np.abs(w) > 1e-9, axis=(1, 2))
                P = P[..., :2] / np.where(np.abs(w) > 1e-9, w, 1.0)
                Dall = np.sqrt(((P[:, :, None, :] - pts[None, None, :, :]) ** 2).sum(-1))       # (M,12,n)
                D, J = Dall.min(axis=2), Dall.argmin(axis=2)
                sc = (D < rough).sum(axis=1)
                mean_d = np.where(D < rough, D, 0).sum(axis=1) / np.maximum(sc, 1)
                for m in np.argsort(-sc)[:30]:
                    if not good[m] or sc[m] < 7:
                        continue
                    uniq = len(set(J[m][D[m] < rough].tolist()))           # one dot may explain only ONE template dot
                    if uniq >= 7:
                        cands.append((uniq, -float(mean_d[m]), Hs[m]))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        accepted = []
        for score, _, H0 in cands[:16]:
            res = self._refine(H0, pts, (0.04 * L, 0.02 * L, 0.01 * L), L)
            if res is None:
                continue
            H, inl, rms, matched = res
            tight = inl >= MIN_INLIERS_INIT and rms <= 0.0013 * L                  # few dots: must fit very exactly
            loose = inl >= 10 and rms <= 0.004 * L                                 # many dots: uneven paper may fit less exactly
            if (tight or loose) and self._sane(H, shape, can):
                accepted.append(((inl, -rms), res))
        if not accepted:
            return None
        accepted.sort(key=lambda a: a[0], reverse=True)
        scorer = getattr(self, "pose_scorer", None)
        frame = getattr(self, "frame_for_score", None)
        if scorer is not None and frame is not None:
            # dots alone can be fooled (a periodic layout fits shifted by a row, a turned sheet, cloth patterns):
            # judge every candidate by how much the warped picture looks like THIS sheet
            best, best_ncc = None, 0.5
            for _, res in accepted[:8]:
                _, ncc = scorer(frame, res[0])
                if ncc > best_ncc:
                    best, best_ncc = res, ncc
            return best
        return accepted[0][1]

    def update(self, dots, shape, can):
        self.did_init = False
        L = float(max(shape[:2]))
        pts = dots[:, :2] if len(dots) else np.zeros((0, 2), np.float32)
        res = None
        if self.H is not None and len(pts) >= MIN_INLIERS_TRACK:
            r0 = min(0.04 * L * (1 + 0.5 * self.lost), 0.15 * L)
            res = self._refine(self.H, pts, (r0, 0.015 * L), L)
            if res is not None and not (res[1] >= MIN_INLIERS_TRACK and res[2] <= 0.004 * L and self._sane(res[0], shape, can)):
                res = None
            self.mode = "TRACKING"
        if res is None and len(pts) >= MIN_INLIERS_INIT and self.cooldown <= 0:
            roundish = np.where(dots[:, 3] <= 2.4)[0]
            use = roundish if len(roundish) >= MIN_INLIERS_INIT else np.arange(len(dots))
            order = use[np.argsort(-dots[use, 4])[:18]]               # the 18 most clearly orange blobs at most
            res = self._init_global(np.ascontiguousarray(pts[order], np.float32), shape, can)
            if res is not None:
                self.did_init = True
                self.mode = "RE-ACQUIRED" if self.H is not None else "TRACKING"
            else:
                self.cooldown = 6                                    # do not search every single frame
        self.cooldown -= 1
        if res is None:
            self.lost += 1
            self.matched[:] = False
            self.inliers = 0
            self.mode = "SEARCHING" if self.H is None else ("HOLDING" if self.lost <= 15 else "LOST")
            return False
        self.H, self.inliers, self.rms, self.matched = res
        self.lost = 0
        return True


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
def _masked_field(val, mask, sigma, fallback):
    """Smooth field from values known only where mask is True (normalised blur, computed at 1/4 size)."""
    sm = cv2.resize(val * mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    mk = cv2.resize(mask.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(sm, (0, 0), sigma / 4.0)
    den = cv2.GaussianBlur(mk, (0, 0), sigma / 4.0)
    f = np.where(den > 0.02, num / np.maximum(den, 0.02), fallback).astype(np.float32)
    w = np.clip(den / 0.2, 0, 1)                                    # fade to the global value far from probe pixels
    f = w * f + (1 - w) * fallback
    return cv2.resize(f, (val.shape[1], val.shape[0]), interpolation=cv2.INTER_LINEAR)


class ParkingDots:
    def __init__(self, tpl_json, tpl_png, baseline_png=None, on=0.12, off=0.05, frames=8, spill=SPILL):
        with open(tpl_json) as f:
            self.tpl = json.load(f)
        self.CW, self.CH = self.tpl["canon"]
        self.canon_img = cv2.imread(tpl_png)
        if self.canon_img is None or self.canon_img.shape[1] != self.CW:
            sys.exit(f"Template image {tpl_png} is missing or does not match {tpl_json}")
        self.tracker = SheetTracker(self.tpl["anchors"])
        self.on, self.off, self.frames, self.spill = on, off, frames, spill
        self._build_zones()
        self.baseline_src = "template photo"
        self.set_baseline(self.canon_img)
        if baseline_png and os.path.exists(baseline_png):
            b = cv2.imread(baseline_png)
            if b is not None and b.shape[:2] == (self.CH, self.CW):
                self.set_baseline(b)
                self.baseline_src = baseline_png
        self.frame_idx = 0
        self.bad_align = 0

    # -- zones ------------------------------------------------------------------
    def _build_zones(self):
        CW, CH = self.CW, self.CH
        self.names = [z["name"] for z in self.tpl["zones"]]
        self.polys = {z["name"]: np.float32(z["poly"]) for z in self.tpl["zones"]}
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        union = np.zeros((CH, CW), np.uint8)
        regs = {}
        cover = np.zeros((CH, CW), np.uint8)
        for n in self.names:
            m = np.zeros((CH, CW), np.uint8)
            cv2.fillPoly(m, [np.int32(self.polys[n])], 255)
            regs[n] = m
            union |= m
            cover += (m > 0).astype(np.uint8)
        self.regions, self.areas = {}, {}
        for n in self.names:
            reg = cv2.erode(((regs[n] > 0) & (cover == 1)).astype(np.uint8) * 255, k) > 0
            self.regions[n] = reg
            self.areas[n] = max(int(np.count_nonzero(reg)), 1)
        paper = np.zeros((CH, CW), np.uint8)
        cv2.fillPoly(paper, [np.int32(self.tpl["paper"])], 255)
        paper = cv2.erode(paper, np.ones((25, 25), np.uint8))
        dots = np.zeros((CH, CW), np.uint8)
        for p in self.tpl["anchors"]:
            cv2.circle(dots, (int(p[0]), int(p[1])), 14, 255, -1)
        self.probe = (paper > 0) & (cv2.dilate(union, np.ones((21, 21), np.uint8)) == 0) & (dots == 0)
        self.med_area = float(np.median(list(self.areas.values())))
        self.states = {n: Debounced(self.on, self.off, frames=self.frames) for n in self.names}

    # -- baseline ---------------------------------------------------------------
    @staticmethod
    def _prep(img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
        edges = cv2.Canny(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), 60, 140)
        return lab, edges

    def set_baseline(self, img):
        self.baseline_img = img.copy()
        self.base_lab, edges = self._prep(img)
        self.base_edges = cv2.dilate(edges, np.ones((13, 13), np.uint8))
        self.base_probe_edges = (edges > 0) & self.probe
        k = np.ones((9, 9), np.uint8)
        self.base_lo, self.base_hi = cv2.erode(self.base_lab, k), cv2.dilate(self.base_lab, k)
        for s in self.states.values():
            s.reset()

    # -- difference ---------------------------------------------------------------
    def _diff(self, rect):
        lab, edges = self._prep(rect)
        ref = self.base_lab
        sel = self.probe & (ref[..., 0] > 120)
        out = {"gain": 1.0, "disturb": 0.0}
        if np.count_nonzero(sel) > 800:
            g = float(np.clip(np.median(ref[..., 0][sel]) / max(np.median(lab[..., 0][sel]), 1.0), 0.4, 2.5))
            ratio = np.clip(ref[..., 0] / np.maximum(lab[..., 0], 1.0), 0.3, 3.0)
            gain = _masked_field(ratio.astype(np.float32), sel, 90.0, g)
            lab = lab.copy()
            lab[..., 0] *= gain
            for c in (1, 2):                                        # white balance: remove the median colour shift
                lab[..., c] -= float(np.median(lab[..., c][sel] - ref[..., c][sel]))
            out["gain"] = g
        lo, hi = self.base_lo, self.base_hi
        excess = np.maximum(lo - lab, lab - hi)
        much_darker = lab[..., 0] < lo[..., 0] * DARK_RATIO - 6
        brighter = (lab[..., 0] - hi[..., 0]) > L_THR
        changed = much_darker | brighter | (np.maximum(excess[..., 1], excess[..., 2]) > AB_THR)
        new_edges = (edges > 0) & (self.base_edges == 0)
        new_edges = cv2.dilate(new_edges.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        new_edges = cv2.morphologyEx(new_edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        cnts, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:                                              # fill outlines of flat, light cars
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) > 400 and min(cv2.minAreaRect(hull)[1]) >= 18:
                cv2.drawContours(new_edges, [hull], -1, 255, cv2.FILLED)
        mask = changed.astype(np.uint8) * 255 | new_edges
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        out["disturb"] = float(np.count_nonzero(mask[self.probe]) / max(np.count_nonzero(self.probe), 1))
        n_ref = int(np.count_nonzero(self.base_probe_edges))
        out["align"] = None if n_ref < 150 else float(np.count_nonzero(self.base_probe_edges & (cv2.dilate(edges, np.ones((9, 9), np.uint8)) > 0)) / n_ref)
        out["mask"] = mask
        return out

    def _assign(self, mask):
        """Give every changed blob to the slot it mostly covers (neighbours only if it covers them nearly as much)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        ratios = {nm: 0.0 for nm in self.names}
        owner = {}
        if n <= 1:
            return ratios, owner
        min_area = 0.02 * self.med_area
        ov = np.zeros((n, len(self.names)), np.int64)
        for si, nm in enumerate(self.names):
            ov[:, si] = np.bincount(labels[self.regions[nm]], minlength=n)[:n]
        for b in range(1, n):
            if stats[b, cv2.CC_STAT_AREA] < min_area:
                continue
            main = int(np.argmax(ov[b]))
            if ov[b, main] == 0:
                continue
            for si, nm in enumerate(self.names):
                if ov[b, si] > 0 and (si == main or ov[b, si] >= self.spill * ov[b, main]):
                    ratios[nm] += ov[b, si] / self.areas[nm]
                    owner.setdefault(nm, []).append(b)
        return ratios, owner

    # -- one frame ------------------------------------------------------------------
    def process(self, frame):
        t0 = time.time()
        self.frame_idx += 1
        dots = detect_dots(frame)
        ok = self.tracker.update(dots, frame.shape, (self.CW, self.CH))
        out = {"frame": frame, "dots": dots, "tracking": ok, "rect": None, "mask": None, "ratios": {},
               "status": {}, "conf": {}, "gain": None, "disturb": 0.0, "reliable": False, "align": None}
        if self.tracker.H is None:
            out["ms"] = (time.time() - t0) * 1000
            return out
        rect = cv2.warpPerspective(frame, self.tracker.H, (self.CW, self.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        out["rect"] = rect
        d = self._diff(rect)
        ratios, owner = self._assign(d["mask"])
        align_ok = d["align"] is None or d["align"] >= ALIGN_MIN
        self.bad_align = 0 if align_ok else self.bad_align + 1
        if self.bad_align >= ALIGN_BAD_FRAMES:                    # the pose is wrong: forget it and search again
            self.tracker.H, self.tracker.lost, self.tracker.mode, self.bad_align = None, 0, "SEARCHING", 0
        reliable = ok and align_ok and d["disturb"] < DISTURB_MAX
        for nm in self.names:
            self.states[nm].update(ratios[nm], reliable)
            out["status"][nm] = self.states[nm].status
            out["conf"][nm] = self.states[nm].conf
        out.update(mask=d["mask"], ratios=ratios, gain=d["gain"], disturb=d["disturb"], reliable=reliable, align=d["align"])
        out["ms"] = (time.time() - t0) * 1000
        return out

    def logic(self, status):
        bits = [1 if status.get(n) == "OCCUPIED" else 0 for n in self.names]
        unk = sum(1 for n in self.names if status.get(n, "UNCERTAIN") == "UNCERTAIN")
        occ = sum(bits)
        return {"bits": bits, "occupied": occ, "total": len(bits), "available": len(bits) - occ,
                "full": int(all(bits)) if bits else 0, "uncertain": unk}


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
SC = {"EMPTY": GREEN, "OCCUPIED": RED, "UNCERTAIN": YELLOW}


def build_dashboard(pd, out, fps, src_name, cam_status):
    st, ratios = out["status"], out["ratios"]
    lg = pd.logic(st)
    trk = pd.tracker

    cam = out["frame"].copy()
    for x, y, *_ in out["dots"]:
        cv2.circle(cam, (int(x), int(y)), 12, YELLOW, 2, cv2.LINE_AA)
    if trk.H is not None:
        proj = cv2.perspectiveTransform(trk.anchor.reshape(-1, 1, 2), trk.H).reshape(-1, 2)
        for i, p in enumerate(proj):
            col = GREEN if trk.matched[i] else RED
            cv2.drawMarker(cam, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        over = cam.copy()
        polys = {}
        for n, poly in pd.polys.items():
            q = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), trk.H).reshape(-1, 2).astype(np.int32)
            polys[n] = q
            cv2.fillPoly(over, [q], SC[st.get(n, "UNCERTAIN")])
        cam = cv2.addWeighted(over, 0.28, cam, 0.72, 0)
        for n, q in polys.items():
            cv2.polylines(cam, [q], True, SC[st.get(n, "UNCERTAIN")], 2, cv2.LINE_AA)
            c = q.mean(axis=0).astype(int)
            text(cam, n, (c[0] - 12, c[1] + 5), 0.55, WHITE, 2)
    text(cam, f"camera: {cam_status}", (8, cam.shape[0] - 10), 0.8, GREEN if cam_status in ("CONNECTED", "IMAGE FILE") else RED, 2)
    pA = fit(cam, 640, 360, "1) CAMERA - dots found (yellow), matched (green +), missing (red +)")

    if out["rect"] is not None:
        rb = out["rect"].copy()
        over = rb.copy()
        for n, poly in pd.polys.items():
            cv2.fillPoly(over, [np.int32(poly)], SC[st.get(n, "UNCERTAIN")])
        rb = cv2.addWeighted(over, 0.28, rb, 0.72, 0)
        for n, poly in pd.polys.items():
            cv2.polylines(rb, [np.int32(poly)], True, SC[st.get(n, "UNCERTAIN")], 2, cv2.LINE_AA)
            c = poly.mean(axis=0).astype(int)
            text(rb, f"{n} {ratios.get(n, 0) * 100:.0f}%", (c[0] - 28, c[1] + 5), 0.5)
    else:
        rb = None
    pB = fit(rb, 320, 427, "2) SHEET, ALIGNED TO TEMPLATE")

    if out["mask"] is not None:
        heat = cv2.cvtColor(out["mask"], cv2.COLOR_GRAY2BGR)
        heat[out["mask"] > 0] = (60, 90, 255)
        for n, poly in pd.polys.items():
            cv2.polylines(heat, [np.int32(poly)], True, SC[st.get(n, "UNCERTAIN")], 2, cv2.LINE_AA)
        pC = fit(heat, 320, 427, "3) WHAT CHANGED vs EMPTY")
    else:
        pC = fit(None, 320, 427, "3) waiting for the sheet")

    d = np.full((360, 640, 3), 30, np.uint8)
    trk_col = GREEN if trk.mode in ("TRACKING", "RE-ACQUIRED") else (YELLOW if trk.mode == "HOLDING" else RED)
    rms = f"{trk.rms:.1f}px" if trk.rms is not None and trk.inliers else "-"
    text(d, f"Sheet: {trk.mode}   dots matched {trk.inliers}/{len(trk.anchor)}   error {rms}", (8, 20), 0.55, trk_col, 1)
    if trk.mode == "SEARCHING":
        text(d, "Show the WHOLE sheet once so all 12 round dots are visible", (8, 42), 0.5, YELLOW)
    al = out.get("align")
    text(d, f"Baseline: {pd.baseline_src}   line match {'-' if al is None else f'{al:.0%}'}   frame trusted: {'yes' if out['reliable'] else 'NO'}", (8, 62), 0.5, GREY)
    text(d, f"OCCUPIED = {' + '.join(str(b) for b in lg['bits'])} = {lg['occupied']}", (8, 92), 0.55)
    text(d, f"AVAILABLE = {lg['total']} - {lg['occupied']} = {lg['available']}", (8, 114), 0.55, GREEN if lg["available"] else RED, 1)
    text(d, f"FULL = AND(all slots) = {lg['full']}     uncertain slots: {lg['uncertain']}", (8, 136), 0.55, RED if lg["full"] else WHITE)
    half = (len(pd.names) + 1) // 2
    for i, n in enumerate(pd.names):
        col, row = divmod(i, half)
        x0, y = 8 + col * 318, 160 + row * 28
        r = ratios.get(n, 0)
        s = st.get(n, "UNCERTAIN")
        text(d, n, (x0, y + 12), 0.5)
        cv2.rectangle(d, (x0 + 34, y), (x0 + 164, y + 14), (70, 70, 70), -1)
        cv2.rectangle(d, (x0 + 34, y), (x0 + 34 + int(min(r, 1) * 130), y + 14), SC[s], -1)
        for t in (pd.on, pd.off):
            cv2.line(d, (x0 + 34 + int(t * 130), y - 2), (x0 + 34 + int(t * 130), y + 16), YELLOW, 1)
        text(d, f"{r * 100:3.0f}% {s}", (x0 + 170, y + 12), 0.45, SC[s])
    text(d, "B baseline=now   R baseline=template   S shot   Q quit", (8, 350), 0.42, GREY)

    e = np.full((293, 640, 3), 38, np.uint8)
    lines = [("PIPELINE", YELLOW),
             ("1 Find orange dots (hue/saturation), merge pen-split fragments", WHITE),
             (f"2 Match to the 12 template dots -> homography ({trk.inliers} matched, {rms})", WHITE),
             ("3 Warp the camera view onto the template sheet (any angle)", WHITE),
             (f"4 Lighting field + white balance: gain x{out['gain']:.2f}" if out["gain"] else "4 Lighting field + white balance", WHITE),
             ("5 Change vs baseline: much darker/brighter, colour, new edges", WHITE),
             (f"6 Each blob -> ONE slot (neighbour only if >= {pd.spill:.0%} of main overlap)", WHITE),
             (f"7 Debounce ON>{pd.on:.0%} OFF<{pd.off:.0%} for {pd.frames} frames", WHITE),
             (f"Lane disturbance {out['disturb']:.1%} (hand/occlusion if > {DISTURB_MAX:.0%});  {out.get('ms', 0):.0f} ms/frame; {fps:.0f} fps", GREY)]
    for i, (s, c) in enumerate(lines):
        text(e, s, (8, 22 + i * 28), 0.5 if i else 0.6, c)

    col1 = np.vstack([pA, d])
    col23 = np.vstack([np.hstack([pB, pC]), fit(e, 640, 293)])
    return np.hstack([col1, col23])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Orange-dot anchored smart parking vision")
    ap.add_argument("--make-template", metavar="PHOTO", help="build the sheet template from a photo of the EMPTY sheet (\"-\" = no photo, printed sheets only)")
    ap.add_argument("--layout", metavar="LAYOUT.json", help="layout file written by make_printable_sheet.py (printed sheet)")
    ap.add_argument("--template", default=os.path.join(HERE, "sheet_template.json"))
    ap.add_argument("--template-image", default=None)
    ap.add_argument("--source", default="0", help="webcam index, phone stream URL, or image file")
    ap.add_argument("--baseline", default="baseline.png")
    ap.add_argument("--on", type=float, default=0.12)
    ap.add_argument("--off", type=float, default=0.05)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--spill", type=float, default=SPILL, help="neighbour-slot share needed to count (0.6 = 60%% of the main overlap)")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    png = a.template_image or os.path.splitext(a.template)[0] + ".png"
    if a.make_template:
        if a.layout:
            build_template_from_layout(a.layout, a.make_template, a.template, png)
        else:
            build_template(a.make_template, a.template, png)
        return
    if not os.path.exists(a.template):
        sys.exit("No template yet. Run:  python parking_dots.py --make-template empty_sheet.jpg")
    pd = ParkingDots(a.template, png, a.baseline, a.on, a.off, a.frames, a.spill)
    print("Baseline:", pd.baseline_src)
    src = FrameSource(a.source)
    win = "Smart Parking (orange-dot tracking)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    fps, tp, shot = 0.0, time.time(), 0
    while True:
        frame = src.read()
        if frame is None:
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue
        out = pd.process(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - tp, 1e-3)
        tp = now
        dash = build_dashboard(pd, out, fps, a.source, src.status())
        if a.scale != 1.0:
            dash = cv2.resize(dash, None, fx=a.scale, fy=a.scale)
        cv2.imshow(win, dash)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("b"), ord("B")) and out["rect"] is not None:
            pd.set_baseline(out["rect"])
            pd.baseline_src = a.baseline
            cv2.imwrite(a.baseline, out["rect"])
            print("Baseline saved ->", a.baseline)
        elif k in (ord("r"), ord("R")):
            pd.set_baseline(pd.canon_img)
            pd.baseline_src = "template photo"
        elif k in (ord("s"), ord("S")):
            shot += 1
            cv2.imwrite(f"dashboard_{shot}.png", dash)
    src.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
