import sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=None
# usage: crop.py src out cx cy size label=file ...  (cx,cy as fraction of image)
src = Image.open(sys.argv[1]); out = sys.argv[2]; cx, cy, S = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
items = [a.split("=",1) for a in sys.argv[6:]]
panels = []
for label, f in items:
    im = Image.open(f)
    w, h = im.size
    x0, y0 = int(cx*w - S/2), int(cy*h - S/2)
    panels.append((label, im.crop((x0, y0, x0+S, y0+S))))
sw, sh = src.size
base = Image.open(items[0][1]).size
f = base[0]/sw
s = S/f
x0, y0 = cx*sw - s/2, cy*sh - s/2
panels.insert(0, ("Lanczos (no AI)", src.crop((int(x0), int(y0), int(x0+s), int(y0+s))).resize((S,S), Image.LANCZOS)))
c = Image.new("RGB", (len(panels)*(S+10), S+30), "white")
for i,(t,p) in enumerate(panels):
    c.paste(p, (i*(S+10), 30)); ImageDraw.Draw(c).text((i*(S+10)+5, 8), t, fill="black")
c.save(out)
