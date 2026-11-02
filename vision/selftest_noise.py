"""Real-world noise test for parking_live.py (no camera, no display).
Simulates what a phone sees in a real room: curled / uneven paper, sensor noise, light flicker,
auto-exposure drift, drifting shadows, glare, camera shake and bursts of motion blur.
Counts how often an EMPTY sheet is wrongly reported as occupied, and checks that parked cars stay found.
    python selftest_noise.py"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selftest_dots as T
import parking_live as PL

DT = 1 / 15.0
W0, H0 = T.W0, T.H0


def curl_photo(photo, amp=6.0, seed=1):
    """Uneven paper: smooth displacement of the sheet content (a few px at photo scale)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H0, 0:W0].astype(np.float32)
    dx = np.zeros_like(xx)
    dy = np.zeros_like(yy)
    for _ in range(3):
        fx, fy = r.uniform(0.5, 2.2) * 2 * np.pi / W0, r.uniform(0.5, 2.2) * 2 * np.pi / H0
        px, py = r.uniform(0, 6.28, 2)
        a = amp * r.uniform(0.4, 1.0)
        dx += a * np.sin(fy * yy + px) * np.cos(fx * xx + py)
        dy += a * np.cos(fx * xx + py) * np.sin(fy * yy + px)
    return cv2.remap(photo, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_CANVAS = {}


def _canvas(photo):
    k = id(photo)
    if k not in _CANVAS:
        _CANVAS[k] = (photo, T.canvas_with(photo))
    return _CANVAS[k][1]


def render_real(photo, Hv, i, cfg, rng):
    t = i * DT
    jit = np.eye(3)
    if cfg.get("shake", 0):
        jit[0, 2], jit[1, 2] = rng.normal(0, cfg["shake"], 2)
        a = np.radians(rng.normal(0, 0.15 * cfg["shake"]))
        jit[0, 1], jit[1, 0] = -a, a
    frame = cv2.warpPerspective(_canvas(photo), jit @ Hv, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    f = frame.astype(np.float32)
    # uneven lighting + auto-exposure drift + flicker + colour drift
    grad = np.linspace(1 - cfg.get("shade", 0), 1 + cfg.get("shade", 0), 1280, dtype=np.float32)[None, :, None]
    expo = 1.0 + cfg.get("flicker", 0) * np.sin(2 * np.pi * 1.7 * t) + rng.normal(0, cfg.get("flicker", 0) * 0.4)
    wb = 1.0 + rng.normal(0, cfg.get("wb", 0), 3)
    f = f * grad * expo * wb[None, None, :]
    # moving soft shadow (a hand / person) and glare
    if cfg.get("shadow", 0):
        cx = 300 + 700 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t))
        yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
        s = np.exp(-(((xx - cx) / 170.0) ** 2 + ((yy - 360) / 120.0) ** 2))
        f *= (1 - cfg["shadow"] * s)[..., None]
    if cfg.get("glare", 0):
        gx = 450 + 300 * np.sin(2 * np.pi * 0.07 * t)
        yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
        s = np.exp(-(((xx - gx) / 60.0) ** 2 + ((yy - 300) / 45.0) ** 2))
        f += cfg["glare"] * s[..., None]
    f = np.clip(f, 0, 255)
    f = cv2.GaussianBlur(f, (0, 0), 0.8)
    f += rng.normal(0, cfg.get("noise", 3), f.shape).astype(np.float32)
    f = np.clip(f, 0, 255).astype(np.uint8)
    if cfg.get("blur_burst", 0) and (i % 40) in range(18, 18 + cfg["blur_burst"]):
        f = cv2.GaussianBlur(f, (0, 0), 5)                          # a short burst of motion blur
    return f


CFGS = {
    "mild room":        dict(noise=5, flicker=0.05, wb=0.02, shake=0.8, shade=0.15),
    "noisy + flicker":  dict(noise=10, flicker=0.12, wb=0.04, shake=1.5, shade=0.25),
    "shadows + glare":  dict(noise=7, flicker=0.06, wb=0.03, shake=1.0, shade=0.2, shadow=0.35, glare=90),
    "everything bad":   dict(noise=12, flicker=0.15, wb=0.05, shake=2.2, shade=0.3, shadow=0.4, glare=110, blur_burst=4),
}


def run_case(photo, Hv, cfg, n=120, curl_seed=None, engine=None, seed=0):
    rng = np.random.default_rng(seed)
    e = engine or PL.LiveEngine(T.TPL_JSON, T.TPL_PNG)
    false_frames, frames_any, counts = 0, 0, {}
    last = None
    for i in range(n):
        out = e.process(render_real(photo, Hv, i, cfg, rng), i * DT)
        if i >= 20:
            occ = [k for k in e.names if out["status"][k] == "OCCUPIED"]
            last = occ
            for k in occ:
                counts[k] = counts.get(k, 0) + 1
    return e, last, counts


if __name__ == "__main__":
    fails = 0
    cars = T.scene_photo(T.CARS)
    truth = sorted(T.CARS)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    curls = (float(sys.argv[2]),) if len(sys.argv) > 2 else (0.0, 6.0)
    for curl_amp in curls:
        empty = curl_photo(T.photo0, curl_amp) if curl_amp else T.photo0
        carp = curl_photo(cars, curl_amp) if curl_amp else cars
        for name, cfg in CFGS.items():
            if only and only != name:
                continue
            for view in ("tilt 35",):
                Hv = T.view_H(**T.VIEWS[view])
                e, last, counts = run_case(empty, Hv, cfg)
                bad = sum(counts.values())
                ok_e = bad == 0
                e2, last2, counts2 = run_case(carp, Hv, cfg, seed=1)
                missing = [k for k in truth if counts2.get(k, 0) < 0.9 * 100]
                extra = [k for k in counts2 if k not in truth and counts2[k] > 0]
                ok_c = not missing and not extra
                fails += (not ok_e) + (not ok_c)
                print(f"curl {curl_amp:3.0f} | {name:16s} | {view:16s} | EMPTY sheet false-occupied frames={bad:4d} {'PASS' if ok_e else 'FAIL'}"
                      f" | CARS missing={missing} extra={extra} {'PASS' if ok_c else 'FAIL'}")
    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
