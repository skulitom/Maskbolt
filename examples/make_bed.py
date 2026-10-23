"""Synthetic laser bed: plywood offcuts (some with cut-out holes) on a honeycomb bed, plus each piece's true mask."""
import json, math, random
from PIL import Image, ImageDraw, ImageFilter

random.seed(7)
W, H = 1200, 800
bed = Image.new("RGB", (W, H), (38, 40, 44))
d = ImageDraw.Draw(bed)
r = 9
for j in range(-1, H // int(r * 1.5) + 2):
    for i in range(-1, W // int(r * 1.732) + 2):
        cx = i * r * 1.732 + (j % 2) * r * 0.866
        cy = j * r * 1.5
        pts = [(cx + r * math.cos(math.radians(60 * k + 30)), cy + r * math.sin(math.radians(60 * k + 30))) for k in range(6)]
        d.polygon(pts, outline=(70, 72, 78))
bed = bed.filter(ImageFilter.GaussianBlur(0.6))

# Offcuts: outer polygon in pixels, and holes (circles or rectangles already cut out).
pieces = [
    {"outer": [(80, 70), (470, 60), (480, 300), (300, 310), (290, 200), (90, 210)], "holes": [("circle", 390, 140, 45)]},
    {"outer": [(560, 80), (1110, 90), (1100, 260), (560, 250)], "holes": [("rect", 640, 120, 760, 210), ("rect", 820, 120, 940, 210)]},
    {"outer": [(90, 330), (380, 360), (350, 700), (70, 690)], "holes": [("circle", 220, 520, 70)]},
    {"outer": [(450, 380), (760, 360), (780, 560), (600, 720), (440, 640)], "holes": []},
    {"outer": [(830, 330), (1130, 320), (1140, 720), (980, 730), (970, 520), (840, 510)], "holes": [("circle", 1060, 420, 35), ("circle", 1060, 620, 35)]},
]
truth = []
for n, p in enumerate(pieces):
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.polygon(p["outer"], fill=255)
    for h in p["holes"]:
        if h[0] == "circle":
            _, cx, cy, rr = h
            md.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=0)
        else:
            md.rectangle(h[1:], fill=0)
    # Plywood: warm tan with grain stripes and a little noise.
    base = (196 + random.randint(-12, 12), 160 + random.randint(-10, 10), 112 + random.randint(-10, 10))
    wood = Image.new("RGB", (W, H), base)
    wd = ImageDraw.Draw(wood)
    ang = random.uniform(-0.3, 0.3)
    for k in range(-H, W + H, 7):
        shade = random.randint(-18, 10)
        col = tuple(max(0, min(255, c + shade)) for c in base)
        wd.line([(k, 0), (k + ang * H, H)], fill=col, width=random.randint(2, 5))
    wood = wood.filter(ImageFilter.GaussianBlur(1.2))
    bed.paste(wood, (0, 0), mask)
    # Scorched edge where the laser cut the holes.
    edge = mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
    bed.paste(Image.new("RGB", (W, H), (70, 45, 25)), (0, 0), edge.point(lambda v: 120 if v else 0))
    mask.save(f"images/bed-truth-{n + 1}.png")
    truth.append({"piece": n + 1, "outer": p["outer"], "holes": p["holes"], "area": sum(1 for v in mask.getdata() if v)})
bed.save("images/bed.png")
json.dump({"width": W, "height": H, "mm_per_px": 0.5, "pieces": truth}, open("images/bed-truth.json", "w"), indent=1)
print("images/bed.png", [t["area"] for t in truth])
