"""parking_live.py - real-time parking detector. No baseline to capture, no re-adjusting.

How it decides (all per frame, ~25-40 ms):
  1. Track the 12 orange dots with tiny search windows around where they were last frame
     (full-frame search only if they are lost). This gives the exact sheet -> camera mapping.
  2. Warp the sheet to a flat half-resolution view.
  3. For each slot:  is the painted "P" still visible?  (normalised correlation with the P from the
     sheet template - works in any lighting)  P visible = EMPTY.  P hidden = OCCUPIED.
     A cheap "something changed in this slot" cue backs it up for cars that do not cover the P.
  4. Decisions are time based (about 0.3 s to flag a car, 0.5 s to clear it). While the phone is
     moving / blurred / the sheet is lost, the last answer is HELD - slots never turn yellow.

The sheet template is part of the design (made once with parking_dots.py --make-template); it is not
a per-session baseline, so nothing has to be re-captured when the camera moves.

Run:  python parking_live.py --source http://PHONE_IP:8080/video
Keys: S screenshot, Q quit
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
from parking_dots import (detect_dots, SheetTracker, _masked_field, MIN_INLIERS_TRACK,  # noqa: E402
                          DOT_H, DOT_S, DOT_DS, DOT_V, ALIGN_MIN, ALIGN_BAD_FRAMES, DISTURB_MAX)
from parking_vision import FrameSource, text, fit, GREEN, RED, YELLOW, WHITE, GREY  # noqa: E402

SC = 0.5                         # working scale of the flat sheet view (0.5 = 320 px wide)
L_THR, DARK_RATIO, AB_THR = 45, 0.62, 22
P_LO, P_HI = 0.60, 0.80          # P correlation (local-normalised): <= P_LO hidden, >= P_HI visible
CAR_MIN_CHANGE = 0.07            # a hidden P only means 'car' if the slot ALSO changed at least this much
ALIGN_MIN_LIVE = 0.30            # looser than parking_dots: uneven / curled paper is allowed
EMA_ALPHA = 0.4                  # temporal noise reduction: weight of the newest aligned frame
SCORE_MEDIAN = 5                 # median of the last 5 scores per slot
WARMUP_FRAMES = 8                # trusted frames before the first answer
EXPO_JUMP = 0.30                 # a sudden sheet brightness jump (light switched) pauses decisions for 2 frames
TRUST_MIN_DOTS = 8               # fewer matched dots = shaky alignment: keep tracking but do not decide
MASS_FLIP = 0.4                  # share of EMPTY slots flagged in the same frame = lighting event, not cars
BLUR_RATIO = 0.45                # frame sharper than this x its usual sharpness is trusted
DARK = (30, 30, 30)
UNKNOWN_COL = (150, 150, 150)


def _odd(n):
    n = int(round(n))
    return max(3, n | 1)


# ----------------------------------------------------------------------------
# Dot tracking in small windows (fast path)
# ----------------------------------------------------------------------------
class FastTracker(SheetTracker):
    def __init__(self, anchors):
        super().__init__(anchors)
        self.gains = np.ones(3, np.float32)
        self.gain_age = 999
        self.speed = 0.0
        self.t_track = 0.0

    @staticmethod
    def _wb(frame):
        q = frame[::8, ::8].reshape(-1, 3).astype(np.float32)
        br = q.sum(axis=1)
        top = q[br >= np.percentile(br, 80)]
        return np.clip(top.mean() / np.maximum(top.mean(axis=0), 1.0), 0.6, 1.6).astype(np.float32)

    def _blob(self, frame, cx, cy, R, L):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = max(0, int(cx - R)), max(0, int(cy - R)), min(w, int(cx + R)), min(h, int(cy + R))
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        crop = np.clip(frame[y0:y1, x0:x1].astype(np.float32) * self.gains, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        S = hsv[..., 1].astype(np.float32)
        m = ((hsv[..., 0] >= DOT_H[0]) & (hsv[..., 0] <= DOT_H[1]) & (S >= DOT_S) &
             ((S - np.median(S)) >= DOT_DS) & (hsv[..., 2] >= DOT_V)).astype(np.uint8) * 255
        k = _odd(0.006 * L)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, bd = None, 1e9
        for c in cnts:
            a = cv2.contourArea(c)
            x, y, bw, bh = cv2.boundingRect(c)
            if a < 5 or min(bw, bh) < 2 or max(bw, bh) > 0.05 * L:
                continue
            if max(bw, bh) / float(min(bw, bh)) > 3.2 or a / float(bw * bh) < 0.4:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            px, py = x0 + M["m10"] / M["m00"], y0 + M["m01"] / M["m00"]
            d = np.hypot(px - cx, py - cy)
            if d < bd:
                best, bd = (px, py), d
        return best

    def _track_roi(self, frame, L, can):
        proj = cv2.perspectiveTransform(self.anchor.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        R = float(np.clip(0.03 * L + 3.0 * self.speed, 0.03 * L, 0.12 * L))
        idx, pts = [], []
        for i, (px, py) in enumerate(proj):
            p = self._blob(frame, px, py, R, L)
            if p is not None:
                idx.append(i)
                pts.append(p)
        if len(idx) < MIN_INLIERS_TRACK:
            return None
        A, B = self.anchor[idx], np.float32(pts)
        Hn, _ = cv2.findHomography(A, B, cv2.RANSAC, max(2.5, 0.003 * L))
        if Hn is None:
            return None
        d = np.linalg.norm(cv2.perspectiveTransform(A.reshape(-1, 1, 2), Hn).reshape(-1, 2) - B, axis=1)
        ok = d < max(3.0, 0.006 * L)
        if ok.sum() < MIN_INLIERS_TRACK:
            return None
        Hn, _ = cv2.findHomography(A[ok], B[ok], 0)                      # least squares on the good dots
        if Hn is None or not self._sane(Hn, frame.shape, can):
            return None
        d = np.linalg.norm(cv2.perspectiveTransform(A[ok].reshape(-1, 1, 2), Hn).reshape(-1, 2) - B[ok], axis=1)
        rms = float(np.sqrt(np.mean(d ** 2)))
        if rms > 0.004 * L:
            return None
        matched = np.zeros(len(self.anchor), bool)
        matched[np.array(idx)[ok]] = True
        new_proj = cv2.perspectiveTransform(self.anchor.reshape(-1, 1, 2), Hn).reshape(-1, 2)
        self.speed = 0.5 * self.speed + 0.5 * float(np.mean(np.linalg.norm(new_proj - proj, axis=1)))
        return Hn, int(ok.sum()), rms, matched

    def update_frame(self, frame, can):
        t0 = time.time()
        L = float(max(frame.shape[:2]))
        if self.gain_age >= 20:
            self.gains, self.gain_age = self._wb(frame), 0
        self.gain_age += 1
        if self.H is not None and self.lost <= 10:
            res = self._track_roi(frame, L, can)
            if res is not None:
                self.H, self.inliers, self.rms, self.matched = res
                self.lost, self.mode = 0, "TRACKING"
                self.t_track = (time.time() - t0) * 1000
                return True
        ok = self.update(detect_dots(frame), frame.shape, can)           # slower full search: lost / fast move
        self.t_track = (time.time() - t0) * 1000
        return ok


# ----------------------------------------------------------------------------
# Time-based decision per slot (no frame counting, never yellow once started)
# ----------------------------------------------------------------------------
class SlotDecider:
    HI, LO = 0.6, 0.4

    def __init__(self, t_occupy=0.30, t_clear=0.50):
        self.t_occupy, self.t_clear = t_occupy, t_clear
        self.state, self.since = None, None
        self.hist = []                               # recent scores (noise reduction: median filter)
        self.n = 0

    def smoothed(self):
        return float(np.median(self.hist[-SCORE_MEDIAN:])) if self.hist else 0.0

    def update(self, score, now, trusted):
        if not trusted:
            self.since = None                        # hold the last answer, collect nothing
            return
        self.hist.append(score)
        self.hist = self.hist[-SCORE_MEDIAN:]
        sc = self.smoothed()
        if self.state is None:                       # first answer only after a short, agreeing warm-up
            self.n += 1
            if self.n >= WARMUP_FRAMES:
                self.state = sc > 0.5
            return
        want = None
        if not self.state and sc > self.HI:
            want = True
        elif self.state and sc < self.LO:
            want = False
        if want is None:
            self.since = None
            return
        if self.since is None:
            self.since = now
        if now - self.since >= (self.t_occupy if want else self.t_clear):
            self.state, self.since = want, None

    @property
    def status(self):
        return "UNKNOWN" if self.state is None else ("OCCUPIED" if self.state else "EMPTY")


# ----------------------------------------------------------------------------
class LiveEngine:
    def __init__(self, tpl_json, tpl_png, on=0.12, off=0.05, spill=0.6, t_occupy=0.30, t_clear=0.50):
        with open(tpl_json) as f:
            self.tpl = json.load(f)
        self.cw0, self.ch0 = self.tpl["canon"]
        self.CW, self.CH = int(round(self.cw0 * SC)), int(round(self.ch0 * SC))
        big = cv2.imread(tpl_png)
        if big is None:
            sys.exit(f"Missing {tpl_png}. Run: python parking_dots.py --make-template empty_sheet.jpg")
        self.ref_img = cv2.resize(big, (self.CW, self.CH), interpolation=cv2.INTER_AREA)
        self.tracker = FastTracker(self.tpl["anchors"])
        self.on, self.off, self.spill = on, off, spill
        self.names = [z["name"] for z in self.tpl["zones"]]
        self.polys = {z["name"]: np.float32(z["poly"]) * SC for z in self.tpl["zones"]}
        self.pbox = {z["name"]: (None if not z.get("p_box") else [v * SC for v in z["p_box"]]) for z in self.tpl["zones"]}
        self.deciders = {n: SlotDecider(t_occupy, t_clear) for n in self.names}
        self._build_masks()
        self._set_reference(self.ref_img)
        self.sharp_med, self.bad_align, self.t_diff, self.t_total = None, 0, 0.0, 0.0
        self.avg, self.prev_level, self.settle = None, None, 0

    def _build_masks(self):
        CW, CH = self.CW, self.CH
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        regs, union, cover = {}, np.zeros((CH, CW), np.uint8), np.zeros((CH, CW), np.uint8)
        for n in self.names:
            m = np.zeros((CH, CW), np.uint8)
            cv2.fillPoly(m, [np.int32(self.polys[n])], 255)
            regs[n] = m
            union |= m
            cover += (m > 0).astype(np.uint8)
        self.regions, self.areas = {}, {}
        for n in self.names:
            reg = cv2.erode(((regs[n] > 0) & (cover == 1)).astype(np.uint8) * 255, k) > 0
            self.regions[n], self.areas[n] = reg, max(int(np.count_nonzero(reg)), 1)
        paper = np.zeros((CH, CW), np.uint8)
        cv2.fillPoly(paper, [np.int32(np.float32(self.tpl["paper"]) * SC)], 255)
        paper = cv2.erode(paper, np.ones((13, 13), np.uint8))
        dots = np.zeros((CH, CW), np.uint8)
        for p in self.tpl["anchors"]:
            cv2.circle(dots, (int(p[0] * SC), int(p[1] * SC)), 7, 255, -1)
        self.probe = (paper > 0) & (cv2.dilate(union, np.ones((11, 11), np.uint8)) == 0) & (dots == 0)
        self.med_area = float(np.median(list(self.areas.values())))

    @staticmethod
    def _prep(img):
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.Canny(gray, 60, 140), gray

    @staticmethod
    def _local_norm(gray):
        """Remove shading, glare gradients and exposure: subtract the local mean, divide by the local contrast."""
        g = gray.astype(np.float32)
        mu = cv2.GaussianBlur(g, (0, 0), 4)
        d = g - mu
        sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 4))
        return d / (sd + 8.0)

    def _set_reference(self, img):
        self.ref_lab, edges, self.ref_gray = self._prep(img)
        self.ref_ln = self._local_norm(cv2.GaussianBlur(self.ref_gray, (3, 3), 0))
        self.ref_level = float(np.median(self.ref_gray[self.probe])) if np.count_nonzero(self.probe) else 128.0
        self.ref_edges = cv2.dilate(edges, np.ones((7, 7), np.uint8))
        self.ref_probe_edges = (edges > 0) & self.probe
        k = np.ones((5, 5), np.uint8)
        self.ref_lo, self.ref_hi = cv2.erode(self.ref_lab, k), cv2.dilate(self.ref_lab, k)

    # -- cues ---------------------------------------------------------------------
    def _diff(self, rect):
        lab, edges, gray = self._prep(rect)
        ref = self.ref_lab
        sel = self.probe & (ref[..., 0] > 120)
        gain = 1.0
        if np.count_nonzero(sel) > 200:
            gain = float(np.clip(np.median(ref[..., 0][sel]) / max(np.median(lab[..., 0][sel]), 1.0), 0.4, 2.5))
            ratio = np.clip(ref[..., 0] / np.maximum(lab[..., 0], 1.0), 0.3, 3.0).astype(np.float32)
            lab = lab.copy()
            lab[..., 0] *= _masked_field(ratio, sel, 45.0, gain)
            for c in (1, 2):
                lab[..., c] -= float(np.median(lab[..., c][sel] - ref[..., c][sel]))
        lo, hi = self.ref_lo, self.ref_hi
        excess = np.maximum(lo - lab, lab - hi)
        changed = (lab[..., 0] < lo[..., 0] * DARK_RATIO - 6) | ((lab[..., 0] - hi[..., 0]) > L_THR) | \
                  (np.maximum(excess[..., 1], excess[..., 2]) > AB_THR)
        new_edges = (edges > 0) & (self.ref_edges == 0)
        new_edges = cv2.dilate(new_edges.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        new_edges = cv2.morphologyEx(new_edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        cnts, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            hull = cv2.convexHull(c)
            if cv2.contourArea(hull) > 100 and min(cv2.minAreaRect(hull)[1]) >= 9:
                cv2.drawContours(new_edges, [hull], -1, 255, cv2.FILLED)
        mask = changed.astype(np.uint8) * 255 | new_edges
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_ref = int(np.count_nonzero(self.ref_probe_edges))
        align = None if n_ref < 80 else float(np.count_nonzero(self.ref_probe_edges & (cv2.dilate(edges, np.ones((7, 7), np.uint8)) > 0)) / n_ref)
        disturb = float(np.count_nonzero(mask[self.probe]) / max(np.count_nonzero(self.probe), 1))
        return mask, gray, gain, align, disturb

    def _assign(self, mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        ratios = {nm: 0.0 for nm in self.names}
        if n <= 1:
            return ratios
        ov = np.zeros((n, len(self.names)), np.int64)
        for si, nm in enumerate(self.names):
            ov[:, si] = np.bincount(labels[self.regions[nm]], minlength=n)[:n]
        for b in range(1, n):
            if stats[b, cv2.CC_STAT_AREA] < 0.02 * self.med_area:
                continue
            main = int(np.argmax(ov[b]))
            if ov[b, main] == 0:
                continue
            for si, nm in enumerate(self.names):
                if ov[b, si] > 0 and (si == main or ov[b, si] >= self.spill * ov[b, main]):
                    ratios[nm] += ov[b, si] / self.areas[nm]
        return ratios

    def _p_scores(self, ln):
        out = {}
        pad, srch = 3, 9
        for n in self.names:
            box = self.pbox[n]
            if box is None:
                out[n] = None
                continue
            x, y, w, h = [int(round(v)) for v in box]
            tx0, ty0, tx1, ty1 = x - pad, y - pad, x + w + pad, y + h + pad
            sx0, sy0, sx1, sy1 = max(0, tx0 - srch), max(0, ty0 - srch), min(self.CW, tx1 + srch), min(self.CH, ty1 + srch)
            tpl = self.ref_ln[ty0:ty1, tx0:tx1]
            win = ln[sy0:sy1, sx0:sx1]
            if tpl.size == 0 or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
                out[n] = None
                continue
            out[n] = float(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED).max())
        return out

    def _score(self, ratio, ncc):
        """0 = surely empty, 1 = surely occupied. P visible = empty. A car needs the P hidden AND a real
        change in the slot, so noise that merely blurs the P, or light that changes the slot, cannot flag one."""
        ch = float(np.clip((ratio - self.off) / (self.on - self.off), 0, 1))
        if ncc is None:                                         # slot without a P: change cue only
            return ch
        if ncc >= P_HI:
            return 0.1                                          # P clearly visible
        if ncc <= P_LO:                                         # P hidden
            return 0.45 + 0.55 * float(np.clip(ratio / CAR_MIN_CHANGE, 0, 1)) if ratio < CAR_MIN_CHANGE else 1.0
        part = (P_HI - ncc) / (P_HI - P_LO)                     # P partly covered / uncertain: needs a clear change
        return 0.8 * float(np.clip((ratio - 0.10) / 0.20, 0, 1)) * (0.5 + 0.5 * part)

    # -- one frame ---------------------------------------------------------------------
    def process(self, frame, now=None):
        now = time.time() if now is None else now
        t0 = time.time()
        ok = self.tracker.update_frame(frame, (self.cw0, self.ch0))
        out = {"frame": frame, "tracking": ok, "rect": None, "mask": None, "ratios": {}, "ncc": {}, "scores": {},
               "status": {n: self.deciders[n].status for n in self.names}, "trusted": False, "holding": False,
               "align": None, "disturb": 0.0, "gain": None}
        if self.tracker.H is None:
            out["holding"] = True
            self.t_total = (time.time() - t0) * 1000
            return out
        Hh = self.tracker.H @ np.diag([1.0 / SC, 1.0 / SC, 1.0])
        raw = cv2.warpPerspective(frame, Hh, (self.CW, self.CH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        t1 = time.time()
        # exposure flicker / auto-exposure: bring every aligned frame to the template brightness first,
        # so averaging frames removes sensor noise instead of mixing different exposures
        lvl_raw = float(np.median(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)[self.probe]))
        expo_gain = float(np.clip(self.ref_level / max(lvl_raw, 1.0), 0.5, 2.0))
        raw = np.clip(raw.astype(np.float32) * expo_gain, 0, 255).astype(np.uint8)
        # temporal noise reduction: average frames that are already aligned on the flat sheet
        moving = self.tracker.speed > 0.004 * max(frame.shape[:2]) or not ok
        if self.avg is None or moving:
            self.avg = raw.astype(np.float32)
        else:
            cv2.accumulateWeighted(raw, self.avg, EMA_ALPHA)
        rect = self.avg.astype(np.uint8)
        mask, gray, gain, align, disturb = self._diff(rect)
        solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))      # thin lines / speckle vanish, cars stay
        ratios = self._assign(solid)
        ncc = self._p_scores(self._local_norm(cv2.GaussianBlur(gray, (3, 3), 0)))
        self.t_diff = (time.time() - t1) * 1000
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        self.sharp_med = sharp if self.sharp_med is None else (0.97 * self.sharp_med + 0.03 * sharp if sharp > BLUR_RATIO * self.sharp_med else self.sharp_med)
        blurred = sharp < BLUR_RATIO * self.sharp_med
        align_ok = align is None or align >= ALIGN_MIN_LIVE
        self.bad_align = 0 if align_ok else self.bad_align + 1
        if self.bad_align >= ALIGN_BAD_FRAMES:
            self.tracker.H, self.tracker.lost, self.tracker.mode, self.bad_align = None, 0, "SEARCHING", 0
        # only a really sudden brightness change (light switched, camera exposure jumps) pauses decisions
        level = lvl_raw
        jump = self.prev_level is not None and abs(level - self.prev_level) / max(self.prev_level, 1.0) > EXPO_JUMP
        self.prev_level = level
        self.settle = 2 if jump else max(0, self.settle - 1)
        scores = {n: self._score(ratios[n], ncc[n]) for n in self.names}
        # lighting event: many EMPTY slots look "changed" in the very same frame = glare / shadow / flicker, not cars
        empties = [n for n in self.names if self.deciders[n].state is False]
        mass = len(empties) >= 3 and sum(scores[n] > 0.5 for n in empties) >= max(3, MASS_FLIP * len(empties))
        trusted = (ok and self.tracker.inliers >= TRUST_MIN_DOTS and align_ok and disturb < DISTURB_MAX
                   and not blurred and self.settle == 0 and not mass)
        for n in self.names:
            self.deciders[n].update(scores[n], now, trusted)
        out.update(rect=rect, mask=mask, ratios=ratios, ncc=ncc, scores=scores, trusted=trusted, holding=not trusted,
                   status={n: self.deciders[n].status for n in self.names}, align=align, disturb=disturb, gain=gain)
        self.t_total = (time.time() - t0) * 1000
        return out

    def logic(self, status):
        bits = [1 if status.get(n) == "OCCUPIED" else 0 for n in self.names]
        occ = sum(bits)
        return {"bits": bits, "occupied": occ, "total": len(bits), "available": len(bits) - occ,
                "full": int(all(bits)) if bits else 0, "unknown": sum(1 for n in self.names if status.get(n) == "UNKNOWN")}


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
def _col(s):
    return {"EMPTY": GREEN, "OCCUPIED": RED}.get(s, UNKNOWN_COL)


def build_dashboard(eng, out, fps, src_name, cam_status):
    st = out["status"]
    lg = eng.logic(st)
    trk = eng.tracker
    cam = out["frame"].copy()
    if trk.H is not None:
        proj = cv2.perspectiveTransform(trk.anchor.reshape(-1, 1, 2), trk.H).reshape(-1, 2)
        for i, p in enumerate(proj):
            cv2.drawMarker(cam, (int(p[0]), int(p[1])), GREEN if trk.matched[i] else RED, cv2.MARKER_CROSS, 16, 2)
        over = cam.copy()
        polys = {}
        Hf = trk.H @ np.diag([1.0 / SC, 1.0 / SC, 1.0])
        for n, poly in eng.polys.items():
            q = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), Hf).reshape(-1, 2).astype(np.int32)
            polys[n] = q
            cv2.fillPoly(over, [q], _col(st[n]))
        cam = cv2.addWeighted(over, 0.28, cam, 0.72, 0)
        for n, q in polys.items():
            cv2.polylines(cam, [q], True, _col(st[n]), 2, cv2.LINE_AA)
            c = q.mean(axis=0).astype(int)
            text(cam, n, (c[0] - 12, c[1] + 5), 0.55, WHITE, 2)
    if out["holding"]:
        text(cam, "HOLDING last answer (moving / blurred / searching)", (10, 28), 0.7, YELLOW, 2)
    text(cam, f"camera: {cam_status}", (8, cam.shape[0] - 10), 0.8, GREEN if cam_status in ("CONNECTED", "IMAGE FILE") else RED, 2)
    pA = fit(cam, 640, 360, "1) CAMERA - dots matched (green +), missing (red +)")

    if out["rect"] is not None:
        rb = cv2.resize(out["rect"], (320, int(320 * eng.CH / eng.CW)))
        sc = 320.0 / eng.CW
        over = rb.copy()
        for n, poly in eng.polys.items():
            cv2.fillPoly(over, [np.int32(poly * sc)], _col(st[n]))
        rb = cv2.addWeighted(over, 0.28, rb, 0.72, 0)
        for n, poly in eng.polys.items():
            cv2.polylines(rb, [np.int32(poly * sc)], True, _col(st[n]), 2, cv2.LINE_AA)
            b = eng.pbox[n]
            if b:
                cv2.rectangle(rb, (int(b[0] * sc) - 2, int(b[1] * sc) - 2), (int((b[0] + b[2]) * sc) + 2, int((b[1] + b[3]) * sc) + 2),
                              GREEN if (out["ncc"].get(n) or 0) >= P_HI else (RED if (out["ncc"].get(n) or 0) <= P_LO else YELLOW), 1)
    else:
        rb = None
    pB = fit(rb, 320, 427, "2) FLAT SHEET  (box = the P being looked for)")
    if out["mask"] is not None:
        heat = cv2.cvtColor(out["mask"], cv2.COLOR_GRAY2BGR)
        heat[out["mask"] > 0] = (60, 90, 255)
        pC = fit(cv2.resize(heat, (320, int(320 * eng.CH / eng.CW))), 320, 427, "3) EXTRA CUE: what changed")
    else:
        pC = fit(None, 320, 427, "3) waiting for the sheet")

    d = np.full((360, 640, 3), 30, np.uint8)
    mode_col = GREEN if trk.mode in ("TRACKING", "RE-ACQUIRED") else (YELLOW if trk.mode == "HOLDING" else RED)
    text(d, f"Sheet: {trk.mode}   dots {trk.inliers}/{len(trk.anchor)}   {fps:.0f} fps   {eng.t_total:.0f} ms (dots {trk.t_track:.0f}, slots {eng.t_diff:.0f})", (8, 20), 0.5, mode_col)
    if trk.mode == "SEARCHING":
        text(d, "Show the whole sheet once so the dots can be found", (8, 42), 0.5, YELLOW)
    text(d, f"OCCUPIED = {' + '.join(str(b) for b in lg['bits'])} = {lg['occupied']}", (8, 72), 0.55)
    text(d, f"AVAILABLE = {lg['total']} - {lg['occupied']} = {lg['available']}      FULL = {lg['full']}", (8, 94), 0.55, GREEN if lg["available"] else RED)
    half = (len(eng.names) + 1) // 2
    for i, n in enumerate(eng.names):
        col, row = divmod(i, half)
        x0, y = 8 + col * 318, 118 + row * 40
        s = st[n]
        ncc = out["ncc"].get(n)
        text(d, n, (x0, y + 12), 0.55)
        text(d, s, (x0 + 34, y + 12), 0.5, _col(s))
        pv = "P ?" if ncc is None else ("P visible" if ncc >= P_HI else ("P hidden" if ncc <= P_LO else "P partly"))
        text(d, f"{pv} ({0 if ncc is None else ncc:.2f})   change {out['ratios'].get(n, 0) * 100:.0f}%", (x0 + 34, y + 30), 0.42, GREY)
    text(d, "S screenshot   Q quit   (no baseline needed)", (8, 350), 0.42, GREY)

    e = np.full((293, 640, 3), 38, np.uint8)
    lines = [("PIPELINE", YELLOW),
             ("1 Track the 12 orange dots in small windows (fast path)", WHITE),
             ("2 Warp the sheet flat at half resolution", WHITE),
             ("3 Per slot: is the P visible? (correlation with the template P)", WHITE),
             ("4 Backup cue: big change in the slot, each car -> one slot", WHITE),
             ("5 Time-based decision: 0.3 s to flag a car, 0.5 s to clear", WHITE),
             ("6 Moving / blurred / sheet lost -> HOLD last answer", WHITE),
             (f"line match {'-' if out['align'] is None else format(out['align'], '.0%')}   lane disturbance {out['disturb']:.1%}   trusted: {'yes' if out['trusted'] else 'no'}", GREY)]
    for i, (s, c) in enumerate(lines):
        text(e, s, (8, 24 + i * 30), 0.5 if i else 0.6, c)
    return np.hstack([np.vstack([pA, d]), np.vstack([np.hstack([pB, pC]), fit(e, 640, 293)])])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Real-time, baseline-free smart parking vision")
    ap.add_argument("--source", default="0", help="webcam index, phone stream URL, or image file")
    ap.add_argument("--template", default=os.path.join(HERE, "sheet_template.json"))
    ap.add_argument("--on", type=float, default=0.12)
    ap.add_argument("--off", type=float, default=0.05)
    ap.add_argument("--spill", type=float, default=0.6)
    ap.add_argument("--occupy-time", type=float, default=0.30, help="seconds of evidence to flag a car")
    ap.add_argument("--clear-time", type=float, default=0.50, help="seconds of evidence to clear a slot")
    ap.add_argument("--scale", type=float, default=1.0, help="dashboard window scale")
    ap.add_argument("--no-ui", action="store_true", help="no window, print the slot states (fastest)")
    a = ap.parse_args()
    if not os.path.exists(a.template):
        sys.exit("No template yet. Run:  python parking_dots.py --make-template empty_sheet.jpg")
    eng = LiveEngine(a.template, os.path.splitext(a.template)[0] + ".png", a.on, a.off, a.spill, a.occupy_time, a.clear_time)
    src = FrameSource(a.source)
    win = "Smart Parking LIVE"
    if not a.no_ui:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    fps, tp, shot, last_print = 0.0, time.time(), 0, ""
    while True:
        frame = src.read()
        if frame is None:
            if not a.no_ui and cv2.waitKey(20) & 0xFF == ord("q"):
                break
            time.sleep(0.01)
            continue
        out = eng.process(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - tp, 1e-3)
        tp = now
        if a.no_ui:
            line = " ".join(f"{n}:{'#' if out['status'][n] == 'OCCUPIED' else ('.' if out['status'][n] == 'EMPTY' else '?')}" for n in eng.names)
            if line != last_print:
                print(line, flush=True)
                last_print = line
            continue
        dash = build_dashboard(eng, out, fps, a.source, src.status())
        if a.scale != 1.0:
            dash = cv2.resize(dash, None, fx=a.scale, fy=a.scale)
        cv2.imshow(win, dash)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("s"), ord("S")):
            shot += 1
            cv2.imwrite(f"live_{shot}.png", dash)
    src.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
