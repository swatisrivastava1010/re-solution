import sys, time, math, numpy as np, torch, spandrel
from PIL import Image, ImageDraw
dev = torch.device("mps")
src = Image.open(sys.argv[1]).convert("RGB"); out = sys.argv[2]; models = sys.argv[3:]
def run(p):
    m = spandrel.ModelLoader().load_from_file(p)
    dt = torch.float16 if m.supports_half else (torch.bfloat16 if m.supports_bfloat16 else torch.float32)
    net = m.model.eval().to(dev).to(dt)
    x = torch.from_numpy(np.array(src)).to(dev).permute(2,0,1)[None].to(dt)/255
    t = time.perf_counter()
    with torch.inference_mode(): y = net(x)
    y = (y.float().clamp(0,1)*255).round().byte()[0].permute(1,2,0).cpu().numpy()
    return Image.fromarray(y), time.perf_counter()-t
S = src.width*4
panels = [("Lanczos (no AI)", src.resize((S, src.height*4), Image.LANCZOS))]
for p in models:
    im, t = run(p); panels.append((p.split('/')[-1].rsplit('.',1)[0], im)); print(p, f"{t:.1f}s", flush=True)
cols = 4; rows = math.ceil(len(panels)/cols)
W, H = panels[0][1].size
g = Image.new("RGB", (cols*(W+8), rows*(H+28)), "white")
for i,(t,im) in enumerate(panels):
    x, y = (i%cols)*(W+8), (i//cols)*(H+28)
    g.paste(im, (x, y+24)); ImageDraw.Draw(g).text((x+4, y+6), t, fill="black")
g.save(out)
