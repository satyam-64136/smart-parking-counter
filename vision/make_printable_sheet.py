"""Makes the printable parking sheet (PDF + PNG, exact size) for the dot-tracking detector.

    python make_printable_sheet.py            # A4: slots 58 mm wide, 38-46 mm tall (small cars up to ~50 mm)
    python make_printable_sheet.py --a3       # A3: 1.41x bigger (1:64 cars such as Hot Wheels / Matchbox)

Print at 100% / "Actual size" (NOT "fit to page"), on matte paper, in colour. Check the 100 mm bar.
Then make the detector template (exact geometry comes from the layout file):
    python parking_dots.py --layout parking_sheet_A4_layout.json --make-template photo_of_empty_sheet.jpg
  (or without a photo: python parking_dots.py --layout parking_sheet_A4_layout.json --make-template -)

What the detector needs (all of it is on the sheet):
  * 12 solid orange dots, 10 mm wide, beside the lanes (these are tracked); nothing else on the sheet is orange
  * thick black slot lines between the dots, a bold "P" in every slot
  * a layout that is deliberately irregular: the right block is 8 mm off-centre and the slot heights differ
    (38/46/40/44/42 mm), so the detector can never lock one row off or upside down
"""
import argparse
import math

from PIL import Image, ImageDraw, ImageFont

DPI = 300
ORANGE = (255, 110, 0)
BLACK = (15, 15, 15)
GREY = (120, 120, 120)

# ---- layout in millimetres on A4 (scaled for A3) --------------------------------------------
X_LO, X_LI, X_RI, X_RO = 12.0, 70.0, 132.0, 190.0     # outer-left, inner-left, inner-right, outer-right dot columns
Y0 = 44.0                                             # first slot line
PITCHES = [38.0, 46.0, 40.0, 44.0, 42.0]              # slot heights, deliberately uneven so the sheet can never be
                                                       # mistaken for itself shifted by one row (or turned around)
LINES = len(PITCHES) + 1
DOT_D = 10.0                                          # dot diameter
LINE_W = 1.3                                          # slot line thickness
P_H, P_STROKE = 22.0, 4.0                          # height of the P and its stroke (big, so any parked car overlaps it)
DIV_X0, DIV_X1 = 95.5, 106.5                          # hatched divider between the lanes


def build(a3=False):
    k = 297.0 / 210.0 if a3 else 1.0
    W_MM, H_MM = (297.0, 420.0) if a3 else (210.0, 297.0)
    px = lambda mm: mm * k * DPI / 25.4
    ss = 2                                            # 2x supersampling for clean edges
    W, H = int(round(W_MM * DPI / 25.4)) * ss, int(round(H_MM * DPI / 25.4)) * ss
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    P = lambda mm: px(mm) * ss
    ys = [Y0]
    for p in PITCHES:
        ys.append(ys[-1] + p)

    def font(mm):
        for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, int(P(mm)))
            except OSError:
                continue
        return ImageFont.load_default()

    # slot lines, then dots on top
    for y in ys:
        for xa, xb in ((X_LO, X_LI), (X_RI, X_RO)):
            d.line([(P(xa), P(y)), (P(xb), P(y))], fill=BLACK, width=int(P(LINE_W)))   # runs under the dot / to the tick
    # P in every slot (drawn from shapes, so it needs no font)
    def draw_p(cx, cy):
        h, sw = P(P_H), P(P_STROKE)
        w = h * 0.70
        x0, y0 = cx - w / 2, cy - h / 2
        r = h * 0.30                                  # radius of the round part (bowl height = 2 r)
        # outer shape of the bowl, then cut the inside out, then the stem
        d.rectangle([x0, y0, x0 + w - r, y0 + 2 * r], fill=BLACK)
        d.ellipse([x0 + w - 2 * r, y0, x0 + w, y0 + 2 * r], fill=BLACK)
        ri = r - sw
        d.rectangle([x0 + sw, y0 + sw, x0 + w - r, y0 + 2 * r - sw], fill="white")
        d.ellipse([x0 + w - r - ri, y0 + sw, x0 + w - r + ri, y0 + 2 * r - sw], fill="white")
        d.rectangle([x0, y0, x0 + sw, y0 + h], fill=BLACK)
    for i in range(LINES - 1):
        cy = (ys[i] + ys[i + 1]) / 2
        draw_p(P((X_LO + X_LI) / 2), P(cy))
        draw_p(P((X_RI + X_RO) / 2), P(cy))
    # hatched divider
    d.rounded_rectangle([P(DIV_X0), P(ys[0]), P(DIV_X1), P(ys[-1])], radius=int(P(2)), outline=BLACK, width=int(P(1.3)), fill="white")
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([P(DIV_X0) + P(1.3), P(ys[0]) + P(1.3), P(DIV_X1) - P(1.3), P(ys[-1]) - P(1.3)], fill=255)
    hatch = Image.new("RGB", (W, H), "white")
    hd = ImageDraw.Draw(hatch)
    step = 4.2
    n = int((ys[-1] - ys[0] + 12) / step)
    for i in range(-3, n + 1):
        y = ys[0] + i * step
        hd.line([(P(DIV_X0), P(y + 9)), (P(DIV_X1), P(y))], fill=BLACK, width=int(P(0.8)))
    img.paste(hatch, (0, 0), mask)
    # lane arrows (light grey, decorative): left lane up, right lane down
    def arrow(cx, y_from, y_to):
        d.line([(P(cx), P(y_from)), (P(cx), P(y_to))], fill=GREY, width=int(P(1.0)))
        s = 1 if y_to > y_from else -1
        d.polygon([(P(cx), P(y_to + s * 4)), (P(cx - 4), P(y_to - s * 3)), (P(cx + 4), P(y_to - s * 3))], fill=GREY)
    arrow((X_LI + DIV_X0) / 2, 165, 130)
    arrow((DIV_X1 + X_RI) / 2, 130, 165)
    # black end ticks at the outer line ends (these are NOT orange on purpose: only the 12 dots beside the
    # lanes are orange, so the detector can never confuse the columns)
    for y in ys:
        for x in (X_LO, X_RO):
            d.line([(P(x), P(y - 3.0)), (P(x), P(y + 3.0))], fill=BLACK, width=int(P(LINE_W)))
    # orange dots last (solid, nothing on top of them)
    for y in ys:
        for x in (X_LI, X_RI):
            d.ellipse([P(x - DOT_D / 2), P(y - DOT_D / 2), P(x + DOT_D / 2), P(y + DOT_D / 2)], fill=ORANGE)
    # title, scale bar, notes (all outside the slot block)
    d.text((P(12), P(14)), "SMART PARKING SHEET  -  print at 100% (actual size), matte paper", fill=GREY, font=font(3.6))
    d.text((P(12), P(21)), "Keep flat. Keep the orange dots clean. Whole sheet in the camera view.", fill=GREY, font=font(3.2))
    yb = 276.0
    d.line([(P(12), P(yb)), (P(112), P(yb))], fill=BLACK, width=int(P(0.6)))
    for i in range(11):
        t = 3 if i % 5 == 0 else 1.8
        d.line([(P(12 + 10 * i), P(yb - t)), (P(12 + 10 * i), P(yb + t))], fill=BLACK, width=int(P(0.5)))
    d.text((P(12), P(yb + 5)), "this bar must measure exactly 100 mm", fill=GREY, font=font(3.2))
    img = img.resize((W // ss, H // ss), Image.LANCZOS)
    pw = P_H * k * 0.70 / k                                 # P width in mm (A4 units)
    layout = {
        "paper_mm": [W_MM, H_MM], "scale": k,
        "dots_left": [[X_LI * k, y * k] for y in ys], "dots_right": [[X_RI * k, y * k] for y in ys],
        "slots": [], "p_boxes": {},
    }
    for i in range(LINES - 1):
        a, b = ys[i] * k, ys[i + 1] * k
        layout["slots"].append({"name": f"L{i + 1}", "poly": [[X_LO * k, a], [X_LI * k, a], [X_LI * k, b], [X_LO * k, b]]})
        cy, cx = (a + b) / 2, (X_LO + X_LI) / 2 * k
        layout["p_boxes"][f"L{i + 1}"] = [cx - pw * k / 2, cy - P_H * k / 2, pw * k, P_H * k]
    for i in range(LINES - 1):
        a, b = ys[i] * k, ys[i + 1] * k
        layout["slots"].append({"name": f"R{i + 1}", "poly": [[X_RI * k, a], [X_RO * k, a], [X_RO * k, b], [X_RI * k, b]]})
        cy, cx = (a + b) / 2, (X_RI + X_RO) / 2 * k
        layout["p_boxes"][f"R{i + 1}"] = [cx - pw * k / 2, cy - P_H * k / 2, pw * k, P_H * k]
    return img, (W_MM, H_MM), k, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a3", action="store_true", help="A3 sheet (bigger slots for 1:64 cars)")
    ap.add_argument("--out", default="parking_sheet")
    a = ap.parse_args()
    img, (w, h), k, layout = build(a.a3)
    tag = "A3" if a.a3 else "A4"
    img.save(f"{a.out}_{tag}.pdf", "PDF", resolution=DPI)
    img.save(f"{a.out}_{tag}.png", dpi=(DPI, DPI))
    import json
    with open(f"{a.out}_{tag}_layout.json", "w") as f:
        json.dump(layout, f, indent=1)
    print(f"{a.out}_{tag}.pdf / .png / _layout.json  ({w:.0f} x {h:.0f} mm, {img.size[0]} x {img.size[1]} px @ {DPI} dpi). "
          f"Slots {(X_LI - X_LO) * k:.0f} mm wide, {min(PITCHES) * k:.0f}-{max(PITCHES) * k:.0f} mm tall. Print at 100%.")


if __name__ == "__main__":
    main()
