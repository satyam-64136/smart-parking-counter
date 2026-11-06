"""Robustness test for parking_live.py (no camera, no display), built from the failures seen in real use:
  A. cold starts from random angles / rotations / resolutions with leaves, branches and orange flowers around the sheet:
     never trust a WRONG pose; correct slots whenever it locks
  B. the paper is taken away: the boxes must disappear and the slots must turn UNKNOWN (no ghost boxes)
  C. the camera looks away at the clutter and comes back: it must find the sheet again, never a random box
    python selftest_robust.py"""
import os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selftest_dots as T
import parking_live as PL

DT = 1 / 15.0
Tm = np.array([[1 / T.S, 0, T.OX], [0, 1 / T.S, T.OY], [0, 0, 1]])
fails = 0
SIZES = [((1280, 720), 950.0), ((640, 480), 620.0), ((480, 360), 470.0)]


def report(label, ok, extra=""):
    global fails
    fails += (not ok)
    print(f"{label:52s} {extra} {'PASS' if ok else 'FAIL'}")


def clutter(canvas, seed, margin=900):
    r = np.random.default_rng(seed)
    cv = canvas.copy()
    x0, y0, x1, y1 = T.OX - margin, T.OY - margin, T.OX + T.W0 + margin, T.OY + T.H0 + margin
    on_sheet = lambda c, m=20: T.OX - m < c[0] < T.OX + T.W0 + m and T.OY - m < c[1] < T.OY + T.H0 + m
    for _ in range(900):
        c = (int(r.uniform(x0, x1)), int(r.uniform(y0, y1)))
        if on_sheet(c):
            continue
        col = tuple(int(v) for v in (r.integers(20, 80), r.integers(80, 190), r.integers(20, 90)))
        cv2.ellipse(cv, c, (int(r.uniform(15, 55)), int(r.uniform(6, 20))), r.uniform(0, 180), 0, 360, col, -1)
    for _ in range(120):
        p = (int(r.uniform(x0, x1)), int(r.uniform(y0, y1)))
        a, l = r.uniform(0, 6.28), r.uniform(80, 350)
        if on_sheet(p):
            continue
        cv2.line(cv, p, (int(p[0] + l * np.cos(a)), int(p[1] + l * np.sin(a))), (30, 50, 90), int(r.uniform(3, 11)))
    for _ in range(25):
        c = (int(r.uniform(x0, x1)), int(r.uniform(y0, y1)))
        if on_sheet(c):
            continue
        cv2.circle(cv, c, int(r.uniform(7, 16)), (30, 120, 255), -1)
    return cv


def frame_from(cv_, Hv, size, seed, noise=4.0):
    r = np.random.default_rng(seed)
    f = cv2.warpPerspective(cv_, Hv, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return np.clip(cv2.GaussianBlur(f, (0, 0), 0.9) + r.normal(0, noise, f.shape), 0, 255).astype(np.uint8)


def pose_error(e, Hv):
    c = np.vstack([np.float32(z["poly"]) for z in e.tpl["zones"]]).reshape(-1, 1, 2)
    return float(np.mean(np.linalg.norm(cv2.perspectiveTransform(c, e.tracker.H).reshape(-1, 2) - cv2.perspectiveTransform(c, Hv @ Tm).reshape(-1, 2), axis=1)))


def occ(e, out):
    return sorted(n for n in e.names if out["status"][n] == "OCCUPIED")


if __name__ == "__main__":
    truth = sorted(T.CARS)
    cv_sheet = clutter(T.canvas_with(T.scene_photo(T.CARS)), 1)
    cv_none = clutter(T._bg.copy(), 3)
    rng = np.random.default_rng(11)

    print("--- A. cold starts: random angle / rotation / resolution, clutter and flowers around the sheet ---")
    N, wrong_total, locked, right = 14, 0, 0, 0
    for k in range(N):
        size, f = SIZES[k % 3]
        v = dict(tilt=rng.uniform(15, 55), roll=rng.uniform(0, 360), yaw=rng.uniform(-20, 20), dist=rng.uniform(44, 56) * (size[0] / 1280 if size[0] < 1000 else 1.0) * 1.0, shift=(rng.uniform(-3, 3), rng.uniform(-3, 3)))
        v["dist"] = rng.uniform(44, 56)
        Hv = T.view_H(size=size, f=f, **v)
        e = PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)
        wrong, out = 0, None
        for i in range(35):
            out = e.process(frame_from(cv_sheet, Hv, size, k * 100 + i), i * DT)
            if out["pose_ok"] and out["trusted"] and e.tracker.H is not None and pose_error(e, Hv) > 6.0:
                wrong += 1
        ok_lock = out["pose_ok"] and e.tracker.H is not None and pose_error(e, Hv) < 6.0
        locked += ok_lock
        right += ok_lock and occ(e, out) == truth
        wrong_total += wrong
        print(f"   view {k:2d}: {size[0]}x{size[1]} tilt {v['tilt']:4.0f} roll {v['roll']:4.0f}  locked={ok_lock!s:5s} slots right={occ(e, out) == truth!s:5s} wrong-but-trusted frames={wrong}")
    report("A. never trusts a wrong pose", wrong_total == 0, f"wrong-but-trusted frames: {wrong_total}")
    report("A. locks and reads correctly", right >= int(0.7 * N), f"{locked}/{N} locked, {right}/{N} fully right")

    print("--- B. the paper is taken away ---")
    size, f = SIZES[0]
    Hv = T.view_H(size=size, f=f, tilt=35, roll=0, dist=46)
    e = PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)
    boxes_after, first_gone, occ_after, unknown_end = 0, None, 0, 0
    out = None
    for i in range(110):
        src = cv_sheet if i < 25 else cv_none
        out = e.process(frame_from(src, Hv, size, 5000 + i), i * DT)
        if i >= 25:
            if first_gone is None and not out["pose_ok"]:
                first_gone = (i - 25) * DT
            if i >= 25 + 22 and out["pose_ok"]:
                boxes_after += 1
        if i >= 25 + 52 and occ(e, out):
            occ_after += 1
    unknown_end = e.logic(out["status"])["unknown"]
    report("B. boxes disappear after the paper is removed", first_gone is not None and first_gone <= 1.0 and boxes_after == 0,
           f"boxes gone after {first_gone if first_gone is None else round(first_gone, 2)} s, ghost-box frames later: {boxes_after}")
    report("B. slots turn UNKNOWN, no ghost occupancy", occ_after == 0 and unknown_end == len(e.names), f"unknown slots at the end: {unknown_end}/{len(e.names)}, occupied frames after 3.5 s: {occ_after}")

    print("--- C. camera looks away at the clutter and comes back ---")
    e = PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)
    wrong, out = 0, None
    for i in range(110):
        far = 35 <= i < 65
        Hv = T.view_H(size=size, f=f, tilt=35, roll=3 * np.sin(i * 0.1), dist=46, shift=((70.0 if far else 0.0) + 2 * np.sin(i * 0.2), 0.0))
        out = e.process(frame_from(cv_sheet, Hv, size, 7000 + i), i * DT)
        if out["pose_ok"] and out["trusted"] and e.tracker.H is not None and pose_error(e, Hv) > 6.0:
            wrong += 1
    err = pose_error(e, Hv) if e.tracker.H is not None else None
    report("C. finds the sheet again, never a random box", wrong == 0 and out["pose_ok"] and err is not None and err < 4.0 and occ(e, out) == truth,
           f"wrong-but-trusted frames {wrong}, final pose error {err if err is None else round(err, 1)} px, final slots right={occ(e, out) == truth}")
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
