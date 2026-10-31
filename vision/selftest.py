"""Self-test: simulates a phone camera (pinhole model) looking at the drawn sheet from many
angles, with toy cars, brightness changes and shadows, and checks the detector.
Needs no camera or display:   python selftest.py
Every line should say PASS (shadow rows just show the ratio). Saves selftest_dashboard.png."""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parking_vision as pv

W, H = pv.SHEET_W, pv.SHEET_H
ZONES = dict(pv.DEFAULT_ZONES)

def make_sheet():
    img = np.full((H, W, 3), 245, np.uint8)
    for name, poly in pv.DEFAULT_ZONES:
        if name == 'ENTRY': continue
        px = (np.float32(poly) * [W, H]).astype(np.int32)
        cv2.polylines(img, [px], True, (60, 60, 60), 3)
        c = px.mean(0).astype(int)
        cv2.putText(img, "P", tuple(c - [8, -10]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
    cv2.rectangle(img, (int(.5*W) - 40, int(.14*H)), (int(.5*W) + 40, int(.86*H)), (70, 70, 70), 3)
    cv2.arrowedLine(img, (int(.74*W), int(.02*H)), (int(.74*W), int(.16*H)), (60,60,60), 4)
    return img

def put_car(img, name, kind):
    poly = np.float32(ZONES[name]) * [W, H]
    c = poly.mean(axis=0)
    ang = np.degrees(np.arctan2(poly[1][1]-poly[0][1], poly[1][0]-poly[0][0]))
    box = cv2.boxPoints(((c[0], c[1]), (90, 40), ang)).astype(np.int32)
    col = {'red': (30,30,200), 'blue': (200,90,30), 'dark': (40,40,40)}.get(kind)
    if col: cv2.fillPoly(img, [box], col)
    else:
        cv2.fillPoly(img, [box], (238,238,238)); cv2.polylines(img, [box], True, (150,150,150), 2)
        cv2.rectangle(img, tuple(box.min(0)+[10,10]), tuple(box.max(0)-[10,10]), (190,190,190), 2)

def render(sheet_img, tilt, roll, dist, yaw=0, bright=1.0, seed=0, f=900):
    """pinhole camera looking at the A4 sheet (21 x 29.7 cm) lying on z=0"""
    r = np.random.default_rng(seed)
    obj = np.float32([[-10.5, -14.85, 0], [10.5, -14.85, 0], [10.5, 14.85, 0], [-10.5, 14.85, 0]])
    rv_tilt, _ = cv2.Rodrigues(np.array([np.radians(tilt), 0, 0], np.float64))
    rv_roll, _ = cv2.Rodrigues(np.array([0, 0, np.radians(roll)], np.float64))
    rv_yaw, _ = cv2.Rodrigues(np.array([0, np.radians(yaw), 0], np.float64))
    R = rv_roll @ rv_tilt @ rv_yaw
    rvec, _ = cv2.Rodrigues(R)
    K = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1]], np.float64)
    pts, _ = cv2.projectPoints(obj, rvec, np.array([0, 0, dist], np.float64), K, None)
    pts = pts.reshape(4, 2).astype(np.float32)
    bg = np.full((720, 1280, 3), (70, 110, 70), np.uint8)
    bg = np.clip(bg.astype(np.int16) + r.integers(-25, 25, bg.shape), 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 2)
    Hm = cv2.getPerspectiveTransform(pv.DST, pts)
    warped = cv2.warpPerspective(sheet_img, Hm, (1280, 720), flags=cv2.INTER_AREA)
    m = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hm, (1280, 720)) > 0
    img = bg.copy(); img[m] = warped[m]
    img = np.clip(img.astype(np.float32) * bright, 0, 255).astype(np.uint8)
    img = np.clip(img.astype(np.int16) + r.integers(-4, 4, img.shape), 0, 255).astype(np.uint8)
    return img, pts

sheet = make_sheet()
POSES = {
  "overhead":            dict(tilt=0,  roll=0,   dist=48),
  "tilt 35deg":          dict(tilt=35, roll=0,   dist=50),
  "tilt 55deg":          dict(tilt=55, roll=10,  dist=52),
  "phone landscape 90":  dict(tilt=40, roll=90,  dist=50),
  "roll 150, tilt 45":   dict(tilt=45, roll=150, dist=50),
  "roll 45, tilt 30":    dict(tilt=30, roll=45,  dist=48),
  "yaw 30, tilt 40":     dict(tilt=40, roll=-20, dist=50, yaw=30),
}
CARS = {'L2': 'red', 'L4': 'blue', 'R1': 'dark', 'R5': 'white', 'R3': 'red'}
truth = sorted(CARS)

# baseline from overhead pose, then evaluate every pose with its OWN fresh tracker + baseline from that same pose
for name, pose in POSES.items():
    v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
    empty, pts = render(sheet, seed=0, **pose)
    o = v.process(empty)
    est_aspect = v.aspect
    # rectified result should resemble the original sheet (either orientation up to flip)
    rect = o['rect']
    err0 = np.abs(rect.astype(int) - sheet.astype(int)).mean()
    err180 = np.abs(rect[::-1, ::-1].astype(int) - sheet.astype(int)).mean()
    flipped = err180 < err0
    if flipped: v.flip_180(); o = v.process(empty)
    v.set_baseline(o['rect'])
    s2 = sheet.copy()
    for n, k in CARS.items(): put_car(s2, n, k)
    res = None
    for i in range(6):
        img, _ = render(s2, seed=3 + i, bright=(0.8 if "tilt 35" in name else 1.0), **pose)
        res = v.process(img)
    occ = sorted(n for n in v.slot_names if res['occ'].get(n))
    ok = occ == truth
    print(f"{name:22s} aspect_est={est_aspect:.2f} auto_flipped_needed={flipped!s:5s} err={min(err0,err180):5.1f} -> {occ}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    ratios:", {n: round(float(res['ratios'][n]), 2) for n in res['ratios']})
    if name == "tilt 55deg":
        cv2.imwrite('selftest_dashboard.png', pv.build_dashboard(v, res, 24.0, 'phone (simulated)'))


print("---- cross-pose test: baseline taken at one pose, cars seen from another pose ----")
def fix_flip(v, frame):
    o = v.process(frame); rect = o['rect']
    e0 = np.abs(rect.astype(int) - sheet.astype(int)).mean(); e1 = np.abs(rect[::-1, ::-1].astype(int) - sheet.astype(int)).mean()
    if e1 < e0: v.flip_180(); o = v.process(frame)
    return o
cross = [("overhead","tilt 55deg"), ("tilt 35deg","phone landscape 90"), ("tilt 55deg","yaw 30, tilt 40"), ("overhead","roll 45, tilt 30")]
for pa, pb in cross:
    v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
    e, _ = render(sheet, seed=0, **POSES[pa]); o = fix_flip(v, e); v.set_baseline(o['rect'])
    # new camera pose: re-acquire tracking (phone moved)
    v.corners = None
    e2, _ = render(sheet, seed=5, bright=0.9, **POSES[pb]); fix_flip(v, e2)
    for s_ in v.states.values(): s_.reset()
    # empty scene from the new pose first -> must say all empty
    for i in range(6): r_empty = v.process(render(sheet, seed=7+i, bright=0.9, **POSES[pb])[0])
    occ_e = sorted(n for n in v.slot_names if r_empty['occ'].get(n))
    s2 = sheet.copy()
    for n, k in CARS.items(): put_car(s2, n, k)
    for i in range(6): r = v.process(render(s2, seed=20+i, bright=0.9, **POSES[pb])[0])
    occ = sorted(n for n in v.slot_names if r['occ'].get(n))
    print(f"baseline@{pa:11s} view@{pb:20s} empty->{occ_e}  cars->{occ}  {'PASS' if (occ_e==[] and occ==truth) else 'FAIL'}")
    if not (occ_e==[] and occ==truth):
        print("   empty ratios", {n: round(float(r_empty['ratios'][n]),2) for n in r_empty['ratios']})
        print("   cars  ratios", {n: round(float(r['ratios'][n]),2) for n in r['ratios']})

print("---- lighting stress on empty sheet (should stay all empty) ----")
for br in (0.55, 0.7, 1.0, 1.2, 1.4):
    v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
    e, _ = render(sheet, seed=0, **POSES["tilt 35deg"]); o = fix_flip(v, e); v.set_baseline(o['rect'])
    for i in range(6): r = v.process(render(sheet, seed=9+i, bright=br, **POSES["tilt 35deg"])[0])
    mx = max(r['ratios'][n] for n in v.slot_names)
    print(f"brightness x{br:.2f}: gain={r['gain']:.2f} max slot ratio={mx:.2f} occupied={[n for n in v.slot_names if r['occ'].get(n)]}")

print("---- soft shadow across an empty slot ----")
v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
e, _ = render(sheet, seed=0, **POSES["tilt 35deg"]); o = fix_flip(v, e); v.set_baseline(o['rect'])
sh = sheet.copy().astype(np.float32)
poly = (np.float32(ZONES['L3']) * [W, H]).astype(np.int32)
m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [poly], 1.0); m = cv2.GaussianBlur(m, (0,0), 18)
sh *= (1 - 0.25 * m[..., None]); sh = sh.astype(np.uint8)
for i in range(6): r = v.process(render(sh, seed=9+i, **POSES["tilt 35deg"])[0])
print("L3 ratio with 25% soft shadow:", round(float(r['ratios']['L3']), 2), "occupied:", r['occ'].get('L3'))

print("---- shadow strength sweep on empty slot L3 + a gray car ----")
for strength in (0.15, 0.25, 0.35, 0.45):
    v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
    e, _ = render(sheet, seed=0, **POSES["tilt 35deg"]); o = fix_flip(v, e); v.set_baseline(o['rect'])
    sh = sheet.copy().astype(np.float32)
    m = np.zeros((H, W), np.float32); cv2.fillPoly(m, [(np.float32(ZONES['L3']) * [W, H]).astype(np.int32)], 1.0); m = cv2.GaussianBlur(m, (0,0), 18)
    sh *= (1 - strength * m[..., None]); sh = sh.astype(np.uint8)
    for i in range(6): r = v.process(render(sh, seed=9+i, **POSES["tilt 35deg"])[0])
    print(f"shadow {int(strength*100)}% -> L3 ratio {r['ratios']['L3']:.2f} occupied={r['occ']['L3']}")
v = pv.ParkingVision(pv.DEFAULT_ZONES, frames=3)
e, _ = render(sheet, seed=0, **POSES["tilt 35deg"]); o = fix_flip(v, e); v.set_baseline(o['rect'])
s3 = sheet.copy(); poly = np.float32(ZONES['L3']) * [W, H]; c = poly.mean(0)
box = cv2.boxPoints(((c[0], c[1]), (90, 40), 25)).astype(np.int32); cv2.fillPoly(s3, [box], (150,150,150))
for i in range(6): r = v.process(render(s3, seed=9+i, **POSES["tilt 35deg"])[0])
print("light-gray car (150) in L3 ->", round(float(r['ratios']['L3']),2), r['occ']['L3'])
