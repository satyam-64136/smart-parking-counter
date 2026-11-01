"""Self-test for parking_dots.py. Uses YOUR empty-sheet photo as the sheet and simulates a phone
camera (pinhole model) looking at it from many angles, while moving, with toy cars (including one
straddling a line), lighting changes, hidden dots and a hand over the sheet.
No camera or display needed:   python selftest_dots.py"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parking_dots as pdm  # noqa: E402

PHOTO = os.path.join(HERE, "empty_sheet.jpg")
TPL_JSON = os.path.join(HERE, "sheet_template.json")
TPL_PNG = os.path.join(HERE, "sheet_template.png")
if not os.path.exists(TPL_JSON):
    pdm.build_template(PHOTO, TPL_JSON, TPL_PNG)

photo0 = cv2.imread(PHOTO)
H0, W0 = photo0.shape[:2]
S = pdm.CAN_W / float(W0)                       # photo px -> template px
rng = np.random.default_rng(3)


def zones_photo():
    import json
    t = json.load(open(TPL_JSON))
    return {z["name"]: np.float32(z["poly"]) / S for z in t["zones"]}


ZP = zones_photo()


def draw_car(img, cx, cy, w, h, ang, body, roof=None, outline=(25, 25, 25)):
    """Simple toy car seen from above: body, darker windows, outline, a little texture."""
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
    cv2.fillPoly(img, [box], body)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    def pt(u, v):
        return [int(cx + u * c - v * s), int(cy + u * s + v * c)]
    roofp = np.int32([pt(-0.22 * w, -0.32 * h), pt(0.18 * w, -0.32 * h), pt(0.18 * w, 0.32 * h), pt(-0.22 * w, 0.32 * h)])
    cv2.fillPoly(img, [roofp], roof if roof is not None else tuple(int(v * 0.55) for v in body))
    cv2.polylines(img, [box], True, outline, 3)
    cv2.polylines(img, [roofp], True, outline, 2)
    return img


CARS = {  # slot -> (body colour BGR, roof BGR)
    "L2": ((40, 40, 190), None),             # red
    "L4": ((70, 70, 70), (30, 30, 30)),      # dark grey
    "R1": ((190, 100, 40), None),            # blue
    "R3": ((235, 235, 235), (200, 200, 200)),  # white car on white paper
    "R5": ((30, 150, 190), None),            # yellow-ish
}


def scene_photo(cars, straddle=None):
    img = photo0.copy()
    for name, (body, roof) in cars.items():
        c = ZP[name].mean(axis=0)
        draw_car(img, c[0], c[1], 140, 78, 8 if name[0] == "L" else -6, body, roof)
    if straddle:                                  # (slot_a, slot_b, fraction in a)
        a, b, frac = straddle
        ca, cb = ZP[a].mean(axis=0), ZP[b].mean(axis=0)
        c = ca * frac + cb * (1 - frac)
        draw_car(img, c[0], c[1], 140, 78, 0, (40, 40, 190))
    return img


# background so that every view is filled (tile cloth from the left strip of the photo)
_cloth = photo0[120:900, 0:60]
_tile = np.vstack([_cloth, _cloth[::-1]])
_tile = np.hstack([_tile, _tile[:, ::-1]])
CANVAS_W, CANVAS_H = 3000, 3800
_bg = np.tile(_tile, (CANVAS_H // _tile.shape[0] + 1, CANVAS_W // _tile.shape[1] + 1, 1))[:CANVAS_H, :CANVAS_W]
OX, OY = (CANVAS_W - W0) // 2, (CANVAS_H - H0) // 2


def canvas_with(photo):
    cv = _bg.copy()
    cv[OY:OY + H0, OX:OX + W0] = photo
    return cv


def view_H(tilt=0, roll=0, yaw=0, dist=60.0, shift=(0, 0), f=950.0, size=(1280, 720)):
    """Canvas pixels -> frame pixels for a pinhole camera looking at the sheet plane (A4 ~ 38 px/cm in the photo)."""
    px_per_cm = 38.0
    cx, cy = OX + W0 / 2, OY + H0 / 2
    ref = np.float32([[cx - 500, cy - 650], [cx + 500, cy - 650], [cx + 500, cy + 650], [cx - 500, cy + 650]])
    obj = np.hstack([(ref - [cx, cy]) / px_per_cm, np.zeros((4, 1), np.float32)]).astype(np.float64)
    Rt, _ = cv2.Rodrigues(np.array([np.radians(tilt), 0, 0.]))
    Rr, _ = cv2.Rodrigues(np.array([0, 0, np.radians(roll)]))
    Ry, _ = cv2.Rodrigues(np.array([0, np.radians(yaw), 0.]))
    rv, _ = cv2.Rodrigues(Rr @ Rt @ Ry)
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.]])
    pts, _ = cv2.projectPoints(obj, rv, np.array([shift[0], shift[1], float(dist)]), K, None)
    return cv2.getPerspectiveTransform(ref, pts.reshape(4, 2).astype(np.float32))


def render(photo, Hv, bright=1.0, tint=(1, 1, 1), shade=0.0, seed=0, size=(1280, 720)):
    r = np.random.default_rng(seed)
    frame = cv2.warpPerspective(canvas_with(photo), Hv, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    frame = frame.astype(np.float32)
    if shade:                                      # lighting falling off from left to right
        grad = np.linspace(1 - shade, 1 + shade, size[0], dtype=np.float32)[None, :, None]
        frame *= grad
    frame *= bright * np.float32(tint)[None, None, :]
    frame = cv2.GaussianBlur(frame, (0, 0), 0.9) + r.normal(0, 3.0, frame.shape).astype(np.float32)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def cover_dots(frame, Hv, idxs, tpl_anchors):
    out = frame.copy()
    # template px -> photo px -> canvas px -> frame px
    T = np.array([[1 / S, 0, OX], [0, 1 / S, OY], [0, 0, 1]])
    for i in idxs:
        p = np.array([tpl_anchors[i][0], tpl_anchors[i][1], 1.0])
        q = Hv @ (T @ p)
        q = q[:2] / q[2]
        cv2.circle(out, (int(q[0]), int(q[1])), 26, (40, 40, 45), -1)
    return out


def hand_over(frame, x0, y0, x1, y1):
    out = frame.copy()
    cv2.rectangle(out, (x0, y0), (x1, y1), (110, 140, 190), -1)
    return out


VIEWS = {
    "overhead":            dict(tilt=0, roll=0, dist=45),
    "tilt 35":             dict(tilt=35, roll=0, dist=46),
    "tilt 55 low angle":   dict(tilt=55, roll=8, dist=44),
    "roll 30, tilt 25":    dict(tilt=25, roll=30, dist=50),
    "yaw 25, tilt 35":     dict(tilt=35, roll=-12, yaw=25, dist=48),
    "off-centre, far":     dict(tilt=30, roll=5, dist=58, shift=(4, -3)),
    "close-up (zoomed)":   dict(tilt=20, roll=-5, dist=42),
}
TRUTH = sorted(CARS)
fails = 0


def run(pd, frames, label, expect, extra=""):
    global fails
    out = None
    for fr in frames:
        out = pd.process(fr)
    occ = sorted(n for n in pd.names if out["status"].get(n) == "OCCUPIED")
    unc = [n for n in pd.names if out["status"].get(n) == "UNCERTAIN"]
    ok = occ == sorted(expect) and not unc
    fails += (not ok)
    al = out.get("align")
    print(f"{label:34s} occupied={occ}{' uncertain=' + str(unc) if unc else ''} dots={pd.tracker.inliers}/12 "
          f"err={pd.tracker.rms if pd.tracker.rms is None else round(pd.tracker.rms, 1)}px align={'-' if al is None else round(al, 2)} {extra} {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("      ratios:", {n: round(float(v), 2) for n, v in out["ratios"].items()})
    return out


def fresh():
    return pdm.ParkingDots(TPL_JSON, TPL_PNG, None, frames=4)


if __name__ == "__main__":
    ph_cars = scene_photo(CARS)
    print("--- 1. cold start from each view (baseline = template photo, no clicking) ---")
    first_dash = None
    for name, v in VIEWS.items():
        pd = fresh()
        Hv = view_H(**v)
        frames = [render(ph_cars, Hv, seed=i) for i in range(9)]
        out = run(pd, frames, name, TRUTH)
        if name == "tilt 55 low angle":
            first_dash = pdm.build_dashboard(pd, out, 25.0, "phone (simulated)", "CONNECTED")
    if first_dash is not None:
        cv2.imwrite("selftest_dots_dashboard.png", first_dash)

    print("--- 2. empty sheet must stay empty ---")
    for name in ("overhead", "tilt 55 low angle", "roll 30, tilt 25"):
        pd = fresh()
        Hv = view_H(**VIEWS[name])
        run(pd, [render(photo0, Hv, seed=i) for i in range(9)], f"empty, {name}", [])

    print("--- 3. lighting different from the template photo ---")
    for label, kw in [("darker x0.6", dict(bright=0.6)), ("brighter x1.25", dict(bright=1.25)),
                      ("warm tint", dict(tint=(0.85, 1.0, 1.12))), ("cool tint", dict(tint=(1.12, 1.0, 0.88))),
                      ("shading left-right 30%", dict(shade=0.3))]:
        for cars, exp, tag in ((ph_cars, TRUTH, "cars"), (photo0, [], "empty")):
            pd = fresh()
            Hv = view_H(**VIEWS["tilt 35"])
            run(pd, [render(cars, Hv, seed=i, **kw) for i in range(9)], f"{label} ({tag})", exp)

    print("--- 4. car sitting over a line: it must count for ONE slot ---")
    for frac, expect in ((0.8, ["L2"]), (0.7, ["L2"]), (0.65, ["L2"])):
        pd = fresh()
        Hv = view_H(**VIEWS["tilt 35"])
        ph = scene_photo({}, straddle=("L2", "L3", frac))
        run(pd, [render(ph, Hv, seed=i) for i in range(9)], f"car {int(frac * 100)}% in L2 / {int((1 - frac) * 100)}% in L3", expect)

    print("--- 5. camera moving while cars are parked (no re-baseline) ---")
    pd = fresh()
    changes, prev, outs = 0, None, []
    N = 60
    for i in range(N):
        t = i / (N - 1)
        Hv = view_H(tilt=25 + 25 * t, roll=-10 + 30 * t, yaw=15 * np.sin(3 * t), dist=47 + 6 * np.sin(2 * t * np.pi), shift=(3 * t, -2 * t))
        out = pd.process(render(ph_cars, Hv, seed=i))
        occ = tuple(sorted(n for n in pd.names if out["status"].get(n) == "OCCUPIED"))
        if i >= 12 and prev is not None and occ != prev:
            changes += 1
        if i >= 8:
            prev = occ
        outs.append(out)
    final = sorted(n for n in pd.names if outs[-1]["status"].get(n) == "OCCUPIED")
    ok = final == TRUTH and changes == 0
    fails += (not ok)
    print(f"moving camera, {N} frames: final={final} state changes after warm-up={changes} "
          f"tracked frames={sum(o['tracking'] for o in outs)}/{N} {'PASS' if ok else 'FAIL'}")

    print("--- 6. dots hidden / hand over the sheet ---")
    tpl = pd.tpl["anchors"]
    Hv = view_H(**VIEWS["tilt 35"])
    base = [render(ph_cars, Hv, seed=i) for i in range(9)]
    pd = fresh()
    run(pd, [cover_dots(f, Hv, [3, 8], tpl) for f in base], "cold start, 2 of 12 dots covered", TRUTH)
    pd = fresh()
    for f in base:
        pd.process(f)                                              # tracking established with all dots visible
    run(pd, [cover_dots(f, Hv, [0, 3, 8, 11], tpl) for f in base], "tracking, then 4 of 12 dots covered", TRUTH)
    pd = fresh()
    for f in base:
        pd.process(f)
    handf = [hand_over(f, 250, 150, 1000, 460) for f in base[:6]]
    out = None
    for f in handf:
        out = pd.process(f)
    occ = sorted(n for n in pd.names if out["status"].get(n) == "OCCUPIED")
    ok = occ == TRUTH
    fails += (not ok)
    print(f"{'big hand over the sheet (hold state)':34s} occupied={occ} frame trusted={out['reliable']} {'PASS' if ok else 'FAIL'}")

    import time
    pd = fresh()
    fr = [render(ph_cars, view_H(**VIEWS["tilt 35"]), seed=i) for i in range(12)]
    for f in fr[:3]:
        pd.process(f)
    t0 = time.time()
    for f in fr[3:]:
        pd.process(f)
    print(f"speed while tracking: {(time.time() - t0) / 9 * 1000:.0f} ms/frame (1280x720)")
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
