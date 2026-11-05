"""Sweep test for parking_live.py (no camera, no display): the phone is moved continuously left-right
at low resolution (640x480), with and without the dots visible. Checks the things that went wrong in real use:
the boxes must never sit on the wrong place while the program trusts them, empty slots must never turn
occupied during the motion, and a lost pose must end as UNKNOWN, not as old answers.
    python selftest_sweep.py        (SELFTEST_* variables switch to the printed sheet, see selftest_dots.py)"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selftest_dots as T
import parking_live as PL

DT = 1 / 15.0
SIZE = (640, 480)
Tm = np.array([[1 / T.S, 0, T.OX], [0, 1 / T.S, T.OY], [0, 0, 1]])
fails = 0


def report(label, ok, extra=""):
    global fails
    fails += (not ok)
    print(f"{label:58s} {extra} {'PASS' if ok else 'FAIL'}")


def pose_error(e, Hv):
    c = np.vstack([np.float32(z["poly"]) for z in e.tpl["zones"]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(c, e.tracker.H).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, Hv @ Tm).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def run(photo, truth, hide_dots_after=None, n=150, label="", period=4.0, amp_cm=6.0, noise=4.0):
    e = PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)
    anchors = e.tpl["anchors"]
    cv_ = T.canvas_with(photo)
    wrong_trusted, false_occ, errs, locked_frames, unknown_end = 0, 0, [], 0, 0
    rng = np.random.default_rng(3)
    out = None
    for i in range(n):
        t = i * DT
        x = amp_cm * np.sin(2 * np.pi * t / period)
        Hv = T.view_H(tilt=30, roll=3 * np.sin(1.3 * t), yaw=6 * np.sin(0.9 * t), dist=46, shift=(x, 0.0), f=620.0, size=SIZE)
        fr = cv2.warpPerspective(cv_, Hv, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        fr = np.clip(cv2.GaussianBlur(fr, (0, 0), 0.8) + rng.normal(0, noise, fr.shape), 0, 255).astype(np.uint8)
        if hide_dots_after is not None and i >= hide_dots_after:
            fr = T.cover_dots(fr, Hv, list(range(len(anchors))), anchors)
        out = e.process(fr, t)
        if e.tracker.H is not None:
            err = pose_error(e, Hv)
            if out["trusted"]:
                locked_frames += 1
                errs.append(err)
                if err > 6.0:
                    wrong_trusted += 1
        if i >= 30:
            for k in e.names:
                if k not in truth and out["status"][k] == "OCCUPIED":
                    false_occ += 1
    occ = sorted(k for k in e.names if out["status"][k] == "OCCUPIED")
    unknown = sum(1 for k in e.names if out["status"][k] == "UNKNOWN")
    return e, out, dict(wrong=wrong_trusted, false_occ=false_occ, err=(np.mean(errs) if errs else None), trusted=locked_frames, occ=occ, unknown=unknown)


if __name__ == "__main__":
    cars = T.scene_photo(T.CARS)
    truth = sorted(T.CARS)
    has_markers = bool(PL.LiveEngine(T.TPL_JSON, T.TPL_PNG).tracker.markers)
    print(f"sheet has ArUco markers: {has_markers}")

    e, out, r = run(cars, truth, label="sweep")
    report("left-right sweep, 640x480, dots visible", r["wrong"] == 0 and r["false_occ"] == 0 and r["occ"] == truth,
           f"pose error {r['err'] if r['err'] is None else round(r['err'], 1)} px, wrong-but-trusted {r['wrong']}, false occupied {r['false_occ']}, trusted {r['trusted']}/150, final {r['occ']}")

    e, out, r = run(T.photo0, [], label="empty sweep")
    report("EMPTY sheet, left-right sweep", r["wrong"] == 0 and r["false_occ"] == 0,
           f"wrong-but-trusted {r['wrong']}, false occupied {r['false_occ']}, trusted {r['trusted']}/150")

    def paint_out_dots(photo):
        """Dots too small / washed out to be detected: paint them over with paper white in the sheet photo itself."""
        hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 24) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)).astype(np.uint8)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        out = photo.copy()
        out[m > 0] = np.median(photo[m == 0].reshape(-1, 3)[::50], axis=0).astype(np.uint8)
        return out

    e, out, r = run(paint_out_dots(cars), truth)
    ok = r["wrong"] == 0 and r["false_occ"] == 0 and r["occ"] == truth
    report("sweep, dots NOT detectable (sheet's own drawing / markers)", ok,
           f"pose error {r['err'] if r['err'] is None else round(r['err'], 1)} px, wrong-but-trusted {r['wrong']}, false occupied {r['false_occ']}, trusted {r['trusted']}/150, final {r['occ']}")
    e, out, r = run(cars, truth, hide_dots_after=25)
    report("sweep, ALL dots covered after lock-on", r["wrong"] == 0 and r["false_occ"] == 0,
           f"wrong-but-trusted {r['wrong']}, false occupied {r['false_occ']}, trusted {r['trusted']}/150, final {r['occ']}, unknown {r['unknown']}")

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
