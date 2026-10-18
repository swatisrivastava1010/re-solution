# Research prototypes

Throwaway scripts used for the measurements in `docs/RESEARCH.md` (Apple M4, 16 GB). They are kept for reference only; the app does not import them.

- `bench.py` — speed per model at a 512 px tile in fp32 and fp16 on the Apple GPU (MPS).
- `bench_bf16.py` — bf16 check for the DAT2 models (fp16 gives broken output for UltraSharpV2).
- `tiled.py` — full tiled upscale to an exact long-edge size, with timing and peak memory.
- `grid.py`, `crop.py` — side-by-side comparison images of several models.
- `finish.py` — finishing-stage prototype: detail-strength blend, tone/colour lock, film grain.

SHA-256 of the model files used:

| File | SHA-256 |
|---|---|
| RealESRGAN_x4plus.pth | 4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1 |
| RealESRGAN_x2plus.pth | 49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb |
| realesr-general-x4v3.pth | 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292 |
| RealESRGAN_x4plus_anime_6B.pth | f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da |
| 4x-ClearRealityV1.safetensors | 83a9c8e279e3d07548e2b0d736dc2c73c1736a07072a748ed4fafbba99b44693 |
| 4xNomos2_hq_dat2.pth | 1f2be2b4786b5031776aed90e1818564f5576ab1e44da9355c7f3788ea25bdea |
| 4xNomos2_hq_mosr.pth | c60dbfc7e6f7d27e03517d1bec3f3cbd16e8cd4288eefd1358952f73f8497ddc |
| 4xBHI_dat2_real.safetensors | 1f651fe8e2b7ad978eee0db9c8de89b7bbfd9d2d671ffababb9baf6e2d5b38f6 |
| 4x-UltraSharpV2.safetensors | 2b5db674aa8f3864a696dccd1d07d7cc0500fbf220f35044cd30eee9a7f971a7 |
| 4x-UltraSharpV2_Lite.safetensors | 7a330ef6b7c632477d311ac03499c91c350e35c5986d4a6c295f0628cb871a09 |
