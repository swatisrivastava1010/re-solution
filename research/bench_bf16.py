import time, torch, spandrel, sys
dev = torch.device("mps")
for p in sys.argv[1:]:
    m = spandrel.ModelLoader().load_from_file(p)
    print(p.split('/')[-1], "supports_bf16:", m.supports_bfloat16, "supports_fp16:", m.supports_half)
    net = m.model.eval().to(dev).to(torch.bfloat16)
    x = torch.rand(1,3,512,512, device=dev, dtype=torch.bfloat16)
    with torch.inference_mode():
        y = net(x); torch.mps.synchronize(); t=time.perf_counter(); y = net(x); torch.mps.synchronize()
    print(f"  bf16 {time.perf_counter()-t:.2f}s/tile finite={torch.isfinite(y.float()).all().item()}")
