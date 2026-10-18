import sys, time, math, resource, numpy as np, torch, spandrel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
dev = torch.device("mps")
def upscale(img, net, scale, tile=512, pad=16, dt=torch.float16):
    a = np.asarray(img.convert("RGB"))
    H, W, _ = a.shape
    out = np.empty((H*scale, W*scale, 3), np.uint8)
    x = torch.from_numpy(a).to(dev).permute(2,0,1).unsqueeze(0)
    x = x.to(dt)
    x = x / 255.0
    ny, nx = math.ceil(H/tile), math.ceil(W/tile)
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy*tile, ix*tile
            y1, x1 = min(y0+tile, H), min(x0+tile, W)
            py0, px0 = max(y0-pad, 0), max(x0-pad, 0)
            py1, px1 = min(y1+pad, H), min(x1+pad, W)
            with torch.inference_mode():
                o = net(x[:, :, py0:py1, px0:px1])
            o = o[:, :, (y0-py0)*scale:(y0-py0+y1-y0)*scale, (x0-px0)*scale:(x0-px0+x1-x0)*scale]
            out[y0*scale:y1*scale, x0*scale:x1*scale] = (o.float().clamp(0,1)*255).round().byte().squeeze(0).permute(1,2,0).cpu().numpy()
    return Image.fromarray(out)
m = spandrel.ModelLoader().load_from_file(sys.argv[1]); dt = torch.float16 if m.supports_half else (torch.bfloat16 if m.supports_bfloat16 else torch.float32); net = m.model.eval().to(dev).to(dt)
img = Image.open(sys.argv[2]); target = int(sys.argv[3])
t = time.perf_counter()
up = upscale(img, net, m.scale, dt=dt)
t1 = time.perf_counter()
f = target / max(img.size)
final = up.resize((round(img.width*f), round(img.height*f)), Image.LANCZOS) if up.size[0] != round(img.width*f) else up
t2 = time.perf_counter()
final.save(sys.argv[4], quality=95, subsampling=0)
print(f"model {t1-t:.1f}s, resize {t2-t1:.1f}s, save {time.perf_counter()-t2:.1f}s, out {final.size}, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB, MPS driver alloc {torch.mps.driver_allocated_memory()/1e9:.2f} GB")
