import sys, time, torch, spandrel
dev = torch.device("mps")
def bench(path, tile=512, half=False, n=3):
    m = spandrel.ModelLoader().load_from_file(path)
    net = m.model.eval().to(dev)
    if half: net = net.half()
    x = torch.rand(1, 3, tile, tile, device=dev, dtype=torch.float16 if half else torch.float32)
    with torch.inference_mode():
        y = net(x); torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(n): y = net(x)
        torch.mps.synchronize()
    dt = (time.perf_counter() - t) / n
    mp = tile * tile / 1e6
    ok = torch.isfinite(y.float()).all().item()
    print(f"{path.split('/')[-1]:34s} arch={m.architecture.name:10s} x{m.scale} tile={tile} {'fp16' if half else 'fp32'}: {dt:6.2f}s/tile  {dt/mp:6.1f} s per input MP  finite={ok}", flush=True)
    del net; torch.mps.empty_cache()
for p in sys.argv[1:]:
    for half in (False, True):
        try: bench(p, half=half)
        except Exception as e: print(p, half, "ERR", e)
