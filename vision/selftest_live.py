"""Self-test for parking_live.py (no camera, no display).
Simulates a phone at 15 fps with fast camera moves, motion blur, lighting changes, cars covering
the P, a car sitting over a line, and a car that does not cover the P.   python selftest_live.py"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selftest_dots as T          # reuses the camera simulator (render, view_H, cars...)
import parking_live as PL

DT = 1 / 15.0
fails = 0


def fresh():
    return PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)


def feed(eng, frames, t0=0.0):
    out = None
    for i, f in enumerate(frames):
        out = eng.process(f, t0 + i * DT)
    return out


def occ(eng, out):
    return sorted(n for n in eng.names if out["status"][n] == "OCCUPIED")


def report(label, ok, extra=""):
    global fails
    fails += (not ok)
    print(f"{label:46s} {extra} {'PASS' if ok else 'FAIL'}")


TRUTH = sorted(T.CARS)
ph_cars = T.scene_photo(T.CARS)

if __name__ == "__main__":
    print("--- 1. cold start, no baseline, answer within 1.3 s ---")
    for name, v in T.VIEWS.items():
        e = fresh()
        Hv = T.view_H(**v)
        out = feed(e, [T.render(ph_cars, Hv, seed=i) for i in range(20)])
        o = occ(e, out)
        report(name, o == TRUTH, f"occupied={o} dots={e.tracker.inliers}/12 unknown={e.logic(out['status'])['unknown']}")

    print("--- 2. empty sheet stays empty ---")
    for name in ("overhead", "tilt 35", "roll 30, tilt 25"):
        e = fresh()
        out = feed(e, [T.render(T.photo0, T.view_H(**T.VIEWS[name]), seed=i) for i in range(20)])
        report(f"empty, {name}", occ(e, out) == [], f"occupied={occ(e, out)}")

    print("--- 3. lighting different from the template photo ---")
    for label, kw in [("darker x0.6", dict(bright=0.6)), ("brighter x1.25", dict(bright=1.25)),
                      ("warm tint", dict(tint=(0.85, 1.0, 1.12))), ("cool tint", dict(tint=(1.12, 1.0, 0.88))),
                      ("shading left-right 30%", dict(shade=0.3))]:
        for cars, exp, tag in ((ph_cars, TRUTH, "cars"), (T.photo0, [], "empty")):
            e = fresh()
            Hv = T.view_H(**T.VIEWS["tilt 35"])
            out = feed(e, [T.render(cars, Hv, seed=i, **kw) for i in range(20)])
            report(f"{label} ({tag})", occ(e, out) == exp, f"occupied={occ(e, out)}")

    print("--- 4. car over a line counts for ONE slot ---")
    for frac in (0.8, 0.7, 0.65):
        e = fresh()
        ph = T.scene_photo({}, straddle=("L2", "L3", frac))
        out = feed(e, [T.render(ph, T.view_H(**T.VIEWS["tilt 35"]), seed=i) for i in range(20)])
        report(f"car {int(frac * 100)}% in L2 / {int((1 - frac) * 100)}% in L3", occ(e, out) == ["L2"], f"occupied={occ(e, out)}")

    print("--- 5. FAST camera movement, cars parked ---")
    e = fresh()
    N, changes, prev, tracked, blurred = 75, 0, None, 0, 0
    Hs = []
    for i in range(N):
        t = i / (N - 1)
        Hs.append(T.view_H(tilt=25 + 22 * np.sin(2.5 * t * np.pi) ** 2, roll=-15 + 40 * t + 8 * np.sin(6 * t),
                           yaw=14 * np.sin(5 * t), dist=46 + 6 * np.sin(4 * t * np.pi), shift=(4 * np.sin(7 * t), -3 * t)))
    step = [float(np.mean(np.abs(cv2.perspectiveTransform(np.float32([[[640, 360]]]), Hs[i + 1])
                                  - cv2.perspectiveTransform(np.float32([[[640, 360]]]), Hs[i])))) for i in range(N - 1)]
    for i in range(N):
        fr = T.render(ph_cars, Hs[i], seed=i)
        out = e.process(fr, i * DT)
        tracked += out["tracking"]
        o = tuple(occ(e, out))
        if i >= 20 and prev is not None and o != prev:
            changes += 1
        if i >= 15:
            prev = o
    report("fast moving camera (75 frames)", occ(e, out) == TRUTH and changes == 0,
           f"final={occ(e, out)} flickers={changes} tracked={tracked}/{N} max step={max(step):.0f}px/frame")

    print("--- 6. motion blur / hand: last answer is HELD, never yellow ---")
    e = fresh()
    Hv = T.view_H(**T.VIEWS["tilt 35"])
    base = [T.render(ph_cars, Hv, seed=i) for i in range(30)]
    feed(e, base[:15])
    held = []
    for i, f in enumerate(base[15:24]):
        g = cv2.GaussianBlur(f, (0, 0), 7) if 3 <= i <= 6 else f            # 4 very blurred frames
        out = e.process(g, (15 + i) * DT)
        held.append(occ(e, out))
    report("4 heavily blurred frames", all(h == TRUTH for h in held) and e.logic(out["status"])["unknown"] == 0,
           f"all frames kept the right answer={all(h == TRUTH for h in held)}")
    hand = T.hand_over(base[0], 250, 150, 1000, 460)
    e2 = fresh()
    feed(e2, base[:15])
    out = feed(e2, [T.hand_over(f, 250, 150, 1000, 460) for f in base[15:22]], 15 * DT)
    report("big hand over the sheet", occ(e2, out) == TRUTH, f"occupied={occ(e2, out)} trusted={out['trusted']}")

    print("--- 7. the P cue ---")
    e = fresh()
    ph = T.photo0.copy()
    c = T.ZP["L3"].mean(axis=0)
    T.draw_car(ph, c[0], c[1], 70, 42, 5, (60, 60, 200))                     # SMALL car, covers the P only
    out = feed(e, [T.render(ph, T.view_H(**T.VIEWS["tilt 35"]), seed=i) for i in range(20)])
    report("small car covering only the P", occ(e, out) == ["L3"], f"occupied={occ(e, out)} P score L3={out['ncc'].get('L3'):.2f}")
    e = fresh()
    ph = T.photo0.copy()
    z = T.ZP["L3"]
    T.draw_car(ph, z[:, 0].min() + 60, z[:, 1].min() + 45, 70, 40, 0, (60, 60, 200))   # in a corner, P stays visible
    out = feed(e, [T.render(ph, T.view_H(**T.VIEWS["tilt 35"]), seed=i) for i in range(20)])
    print(f"{'small car in a corner, P visible (limit)':46s} occupied={occ(e, out)} (documented limit: P visible = empty)")

    print("--- 8. speed and latency ---")
    e = fresh()
    fr = [T.render(ph_cars, T.view_H(**T.VIEWS["tilt 35"]), seed=i) for i in range(16)]
    feed(e, fr[:4])
    t0 = time.time()
    for i, f in enumerate(fr[4:]):
        e.process(f, (4 + i) * DT)
    ms = (time.time() - t0) / 12 * 1000
    report("processing speed (1280x720)", ms < 60, f"{ms:.0f} ms/frame (~{1000 / ms:.0f} fps)")
    e = fresh()
    first = None
    for i in range(40):
        out = e.process(T.render(ph_cars, T.view_H(**T.VIEWS["tilt 35"]), seed=i), i * DT)
        if first is None and occ(e, out) == TRUTH:
            first = (i + 1) * DT
    report("time to the correct answer from cold start", first is not None and first < 1.0, f"{first:.2f} s at 15 fps" if first else "never")

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
