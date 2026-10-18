# Upscaling AI images to 6K–8K and videos to 4K: feasibility research

October 2026. Tested on an Apple M4 MacBook (10-core GPU, 16 GB unified memory, macOS 26.6). The speeds below come from this machine unless a section says otherwise.

## The short answer

- **Yes, it works locally, and it is fast enough.** The best in-house model takes about **3 minutes** to turn a 2K image into an 8K one on the M4. Fast models take **2–20 seconds**. Splitting the image into tiles keeps peak GPU memory under about 5 GB, so 16 GB is plenty.
- **Upscaling is not the same as "high quality".** In a 2K→8K upscale, 15 of every 16 output pixels are new. Classic AI upscalers (GAN and transformer models such as Real-ESRGAN and DAT) sharpen edges and add plausible texture. On their own they don't make a mediocre image look like a high-end photo. That is the "texture gets sharper, but the image still doesn't look premium" problem.
- **What actually lifts perceived quality:**
  1. A model that keeps real micro-detail instead of smoothing and oversharpening. The DAT2 models tested here do this best.
  2. A finishing stage that keeps the original's tones and colours, controls how strong the added detail is, and adds very fine grain at output size.
  3. For the last step up, a *generative* (diffusion) upscaler that redraws detail with an understanding of what the object is: fabric weave, wood grain, skin. Generative upscalers can change exact details such as faces, logos and furniture shapes, so they are an opt-in mode with a "how creative" control. They are not the default.
- **SeedVR2-3B runs on this 16 GB M4** as a slow "Photo-real" mode, about 15 minutes per 2K→8K image. On a JPEG-softened mountain it produced a convincing high-resolution photo where the best transformer model produced streaks (§4a). In the cloud, fal.ai runs the same model at full 8K for about $0.05 per image.
- **Video to 4K is practical for short clips.** A 10-second 1080p clip takes about **2–4 minutes** with the fast natural-looking model. Long home videos take hours: roughly 1–2 hours per 5 minutes of 1080p footage.
- **Licences:** everything in the default setup is free *and* allowed for commercial use (Apache-2.0, BSD-3, CC-BY-4.0 with credit). Some popular community models, including UltraSharp and UltraSharpV2, are **non-commercial only**. They are fine for family photos, but they must not be used for business or "team" imagery.

## 1. What 2K→8K upscaling can and can't do

| Approach | What it does | What 8K looks like | Exact details (faces, logos, furniture) |
|---|---|---|---|
| Classic resize (Lanczos) | Interpolates pixels | Soft and blurry at 100% zoom | Perfectly faithful, but no new detail |
| GAN / transformer SR (Real-ESRGAN, SPAN, DAT2) | Learned sharpening plus texture synthesis | Crisp edges and plausible texture. Weaker models look "painted" or waxy | Faithful in shape and layout. Fine texture is invented but consistent with the source |
| Diffusion / generative SR (SeedVR2, Topaz Redefine, Magnific) | Redraws detail with semantic knowledge | The most photographic look: pores, weave, grain | Can drift: faces change slightly, text gets garbled, small objects get "reinterpreted" |

Things that matter at 8K:
- **One upscale pass, then an exact resize.** Running a 4× model once and then resizing with Lanczos to the exact target beats stacking several 2× or 4× passes, because each pass adds its own artifacts. Inputs under about 2,000 px that need more than 4× get a second pass. Those results look more artificial, and the app warns about it.
- **Tiles.** No model can process a 33–67 MP image in one go. The image is cut into 512 px tiles, each with 16 px of extra context, and the pieces are stitched together. No seams were visible in the tests.
- **Already-compressed inputs.** JPEG/WebP blocks and noise in the source get "upscaled" too. Models trained for clean input amplify them, so the app offers a "clean up compression" model for those images.
- **Out-of-focus areas.** Weaker models also sharpen blurred backgrounds, which looks unnatural. The DAT2 models were noticeably better at keeping bokeh soft.

## 2. In-house models tested (M4, 16 GB)

| Model | Architecture | Licence | Speed (s per input megapixel) | 2K→8K time | Verdict |
|---|---|---|---|---|---|
| **4xNomos2_hq_dat2** (Phhofm) | DAT2 transformer | **CC-BY-4.0** (credit required, commercial OK) | 41 (bf16) | ~3 min | **Best faithful detail.** Kept the iris texture and feather strands that other models smoothed away |
| 4x-UltraSharpV2 (Kim2091) | DAT2 | CC-BY-NC-SA-4.0 (**non-commercial**) | 41 (bf16; fp16 gives broken output) | ~3 min | Very clean, but denoises fine texture (the owl's iris came out smoothed). Personal use only |
| 4x-UltraSharpV2_Lite | RealPLKSR | CC-BY-NC-SA-4.0 (**non-commercial**) | 5.6 | ~25 s | Very crisp. Personal use only |
| **4xNomos2_hq_mosr** (Phhofm) | MoSR | **CC-BY-4.0** | 3.8 | ~16 s | Good balanced option |
| **4x-ClearRealityV1** (Kim2091) | SPAN | **Apache-2.0** (relicensed Feb 2026) | 0.4 | ~2 s | Natural and gentle. Best fast option, and good for video |
| RealESRGAN_x4plus | RRDB (ESRGAN) | BSD-3 | 10.2 | ~45 s | Older baseline: waxy feathers, "painted" textures |
| realesr-general-x4v3 | Compact | BSD-3 | 0.6 | ~3 s | Crisp but blotchy |
| RealESRGAN_x4plus_anime_6B | RRDB | BSD-3 | 3.3 | ~15 s | For illustrations and anime only |
| 4xNomos2_realplksr_dysample | RealPLKSR + DySample | CC-BY-4.0 | 4.7 | – | **Broken on Apple GPUs** (blocky output). Excluded |

Notes:
- Timings are for the model step only. Saving an 8K JPEG takes under a second.
- **Peak memory** for 2K→8K: about 4.8 GB of GPU memory for the large models and about 1 GB for the fast ones. Process memory stayed under 1 GB.
- **Precision:** DAT2 models need bfloat16 on Apple GPUs. The others run in float16, which is about 25% faster than float32.
- **4K → 8K** is only 2×, but the models are 4×, so the app runs the model on the full 4K input and then resizes down. Downscaling the input first would throw away real detail. The best model takes about 11 minutes for a 4K input, and the balanced one about 1 minute.

## 2a. Results on the family's own ChatGPT images

Four ChatGPT-generated PNGs were tested: two breakfast table flat-lays at 941×1672 and two retail/interior scenes with furniture at 2048×1539. Each was cut into 256 px crops and upscaled 4×, the same scale as 2K→8K.

- **ChatGPT images carry a faint crosshatch "mesh" texture**, visible on egg whites, bread, flowers and fabric, and their fine detail is soft.
- **Every classic model amplified the mesh into a crunchy mosaic.** This covers Fast, Balanced, Faithful (Nomos2_hq_dat2) and x4plus, and it is exactly the "sharper but not higher quality" problem.
  - Nomos2_hq_dat2, which is trained for clean input, also *kept* the softness. On the wall panels it was barely sharper than Lanczos, and sofa edges came out fuzzy.
- **4xBHI_dat2_real** (trained for degraded "real-world" input) gave the best classic result: clean edges and believable fabric on the sofa, and crisper panel edges. It still showed some mesh on food. **It becomes the Faithful mode.**
- **SeedVR2 (Photo-real) was clearly best overall.** It removed the mesh completely and rendered convincing toast crumb, egg white with pepper, daisies, and jute-rug texture on the furniture panels.
  - The trade-off showed in the flowers: petals and flower positions were redrawn slightly. They were plausible, but not identical.
- **Speed:** with the model kept loaded, SeedVR2 took about 36 s per 256 px crop on the M4.
- **mflux's `--low-ram` flag deletes the model after the first image**, so batches and tiling crash. The app instead keeps the model loaded with a memory cache limit and VAE tiling.

## 3. The "sharper but not better" problem and what the app does about it

Why upscaled images often still look cheap:
1. **Uniform crunchy sharpness.** Every surface gets the same sharpening, with halos around edges and over-defined noise.
2. **Waxy surfaces.** GANs smooth skin, fabric and wood into plastic, then sharpen the edges.
3. **Colour and tone drift.** Models shift colours slightly and flatten gradients.
4. **No natural grain.** Real 8K photos have fine sensor grain. Perfectly clean 8K reads as "CG".
5. **Detail that doesn't match the object.** A GAN doesn't know a sofa is velvet; a diffusion model does.

What the app does:
- **Defaults to the most faithful, detail-keeping model** (Nomos2_hq_dat2) and avoids sharpening-heavy models.
- **Tone and colour lock:** low-frequency colour and tone always come from the original, and only fine detail comes from the AI. The image keeps its look; it just gains resolution.
- **Detail strength** (0–100%): blends the AI detail over a clean Lanczos base, which tames oversharpening.
- **Fine grain** (off / subtle / medium): adds photographic grain at output size, mostly in the midtones.
- **Generative modes** (optional): see sections 4 and 6.

## 4. Diffusion upscalers that run locally

Summary: no diffusion model is good enough to be the default on a 16 GB Mac. SeedVR2-3B is the only serious candidate (licence Apache-2.0 for code and weights).

| Model | Licence | Runs on a 16 GB Mac? | Notes |
|---|---|---|---|
| **SeedVR2-3B / 7B** (ByteDance) | Apache-2.0 | 3B: yes, through mflux (MLX, MIT) | One-step, no prompt. Its own model card warns it can "overgenerate" detail on clean AI images |
| SUPIR | Non-commercial (SupPixel) | No (needs SDXL plus LLaVA-13B, 28 GB+) | |
| StableSR, InvSR, Upscale-A-Video | S-Lab (non-commercial) | No / impractical | |
| HYPIR | Non-commercial (SupPixel) | CUDA only | |
| DiffBIR, OSEDiff, PiSA-SR, TSD-SR | Apache code, weight licences unclear | CUDA-only repos | |
| Flux-dev upscalers (Jasper ControlNet, LucidFlux, FluxSR) | Flux-dev non-commercial | No (28 GB+) | |
| Ultimate SD Upscale / Tiled Diffusion + Tile ControlNet | GPL / CC-BY-NC-SA, SDXL OpenRAIL++ | Technically yes | Estimated 20–40 min per 8K image, with seams and drift to manage |

### 4a. SeedVR2-3B tested on this M4 (mflux 0.23, MLX, 8-bit)

**First tests** (research phase, mflux's own command line):

| Input → output (4×) | Diffusion step | Total wall time | Peak memory |
|---|---|---|---|
| 256 → 1024 | 13 s | (first run included a 7 GB download) | 15.5 GB (normal mode) |
| 512 → 2048 | 47 s | 145 s (the other ~100 s was mostly the VAE, not model loading; see below) | 11.3 GB (`--low-ram`) |
| 1024 → 4096 | not completed | | 18.4 GB peak: too much for 16 GB |

What it looks like:
- On a distant, JPEG-softened mountain, the best transformer model (Nomos2_hq_dat2) amplified the compression into vertical streaks. SeedVR2 produced a convincing forested slope with snow patches: the image looks like a genuinely high-resolution photo.
- On owl feathers it gave more natural, less "etched" strands than the transformer model, but smoothed the iris texture slightly.
- This is the mode that solves "sharper but not better". The price is that it **invents** detail (individual trees, feather strands) that the source never contained.

**Measurements for the app's engine** (Task 3, October 2026). Setup: the 8-bit transformer stays loaded between tiles (`MemorySaver(keep_transformer=True, cache_limit_bytes=1 GB)`), `TilingConfig()` VAE tiling (512 px VAE tiles), seed 42, 4× per tile. Times are the average of two runs after a warm-up tile. Peak is `mx.get_peak_memory()` per tile. Metal's recommended GPU working set on this 16 GB M4 is **12.7 GB**; going above it makes macOS page GPU memory and swap.

| VAE setup | Input tile | VAE encode | Transformer | VAE decode | Per tile | Peak |
|---|---|---|---|---|---|---|
| mflux as shipped | 384 | 17–21 s | 23 s | 34–39 s | 81 s (swapping) | 15.5 GB |
| mflux as shipped | 512 | | | | 137 s (swapping) | 15.5 GB |
| 256 px VAE tiles | 384 | 25 s | 23 s | 49 s | 97 s | 10.1 GB |
| 256 px VAE tiles | 512 | 48–57 s | 41–50 s | 103–109 s | 200 s | 9.9 GB |
| **Single-frame VAE (what the app uses)** | **384** | 6.5 s | 23 s | 15 s | **46 s** | **9.1 GB** |
| **Single-frame VAE (what the app uses)** | **512** | | | | **85 s** | **9.8 GB** |
| Single-frame VAE, **2×** instead of 4× | 512 | | | | 21 s | 9.1 GB |

Findings:
- **The VAE, not the transformer, sets both the time and the peak.** For a 384 px tile the transformer peaks at 8.2 GB, VAE encode at 12.4 GB and VAE decode at 15.6 GB. The 15.5 GB "normal mode" peak in the first tests was the VAE decode.
- **Single-frame VAE (the app's fix).** SeedVR2's VAE is a video VAE. For one frame, every causal 3D convolution pads the frame with two copies of itself, so its time kernel only ever sees identical frames. Adding the kernel's time slices together gives an ordinary 2D convolution with the same result. The app switches the VAE's convolutions to this when the input is a single frame (`mlx_engine.use_single_frame_convs`), and real video still takes the 3D path. On a 384 px tile the output matched mflux's to 0.07 levels on average (max 1.5 of 255, fp16 rounding). The VAE ran **3× faster** (60 s → 21 s) and the tile's peak fell from 15.6 GB to **9.1 GB**.
- **Smaller VAE tiles** also get under the limit (10 GB), but they are slower, because the overlaps redo more work.
- **Tile size: 512 px maximum.** 384 px tiles were 4% faster per input pixel (311 vs 324 s per megapixel), which is inside the run-to-run spread (512 px: 81–89 s). 512 needs fewer tiles, so there are fewer seams, and it was faster on the portrait image below. Both peaks (9.1 / 9.8 GB) are far under the 12.5 GB budget.
- **Tile grid.** Each image gets one tile size: the fewest tiles of at most 512 px that overlap by at least 64 px, each made just big enough to cover the image. Full 512 px tiles would redo 1.33–2.0× of the input pixels; the fitted grid redoes 1.19–1.27×. Tiles are blended with linear ramps across the whole overlap.
- **Loading is not the problem.** Loading the fp16 weights from the Hugging Face cache and quantizing them to 8-bit took 2.4 s with the files in the disk cache. It peaked at 4.8 GB (MLX), with a 6.25 GB process footprint.
- **Pre-quantized copy: not used.** A saved 8-bit copy (`save_model`, 4.7 GB on disk) loads in 1.3 s with a 4.7 GB MLX peak and a 4.8 GB footprint. That only trims the start-up footprint, which is already below a tile's 9–10 GB, so it isn't worth another 4.7 GB of disk.
- **Seam check** on one of the family's breakfast-table images (a 760×480 crop, 2 tiles): no visible seam in the overlap. It took 133 s including start-up, with a 9.1 GB peak.

Expected time per image on the M4 (fitted grid, about 324 s per processed input megapixel):

| Input | Tiles | Photo-real 4× (to ~8K) |
|---|---|---|
| 2048×1539 (furniture/interiors) | 20 of 461×433 | about 22 min |
| 941×1672 (table flat-lays) | 8 of 503×466 | about 10 min |

Feasibility:
- **Photo-real is the default image mode** (the user's decision). Expect about 10–22 minutes per 2K→8K image on a 16 GB M4, plus a one-time 7.3 GB download. The UI's time hint should say this. The full-image timings in Task 9 will confirm it.
- A **2× pass costs about a quarter of a 4× pass** (21 s vs 85 s per 512 tile). The design's hybrid option (SeedVR2 2× to about 4K, then Faithful or Lanczos to 8K) would therefore take about 5–6 minutes of SeedVR2 for a 2048×1539 image. Its quality against the full 4× hasn't been compared yet.
- Face, logo and text fidelity are untested so far; that needs the family's real images.

**Windows/NVIDIA runner** (`torch_engine.py`):
- It uses numz's ComfyUI-SeedVR2_VideoUpscaler standalone CLI (Apache-2.0), version 2.5.24.
  - 2.5.24 was never tagged; the newest tag is v2.5.23. The app pins commit `4490bd1`, the main-branch commit that set the version to 2.5.24 (v2.5.23 plus an Apple-GPU memory-leak fix), and checks the archive's SHA-256.
- It lives in its own uv venv under `<data_dir>/seedvr2-torch/`, with torch from the cu130 index plus the CLI's `requirements.txt`. Nothing is installed into the app's environment.
- Each image is one CLI run on a folder of tiles, using the CLI's real flags:
  - `--model_dir` points at links to the weights already in the Hugging Face cache, with `--dit_model seedvr2_ema_3b_fp16.safetensors`.
  - `--resolution` is the short side × scale, with `--seed 42` and `--color_correction lab`.
  - `--cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu --vae_encode_tiled --vae_decode_tiled`.
- **Smoke test on this Mac** (MPS, PyPI torch instead of cu130, scratch folder):
  - The install worked (most packages came from uv's cache).
  - 128 → 512 px took 85 s (49 s inside the CLI, including the first-run check of the weights' hashes). The output was finite, with mean colour within 0.5 levels of Lanczos.
- **Not run on an NVIDIA card**, so it ships as "should work". Any failure falls back to Faithful with a note.

## 5. Video to 4K

Measured on the M4 (model plus resize to 3840×2160, per frame):

| Model | 720p → 4K | 1080p → 4K | 10 s clip at 30 fps (1080p) | 5 min video (1080p) |
|---|---|---|---|---|
| **4x-ClearRealityV1** (Apache) | 0.38 s | 0.81 s | ~4 min | ~2 h |
| realesr-general-x4v3 (BSD) | 0.67 s | 1.45 s | ~7 min | ~3.6 h |
| RealESRGAN_x2plus (BSD) | 2.4 s | 5.5 s | ~27 min | ~14 h |
| 4x-UltraSharpV2_Lite (NC) | 5.1 s | 11.8 s | ~1 h | ~30 h |

Findings:
- **Per-frame models flicker** because each frame invents its own fine texture. The gentle models (ClearRealityV1, general-x4v3) flicker least, and the detail-strength blend reduces it further.
- **True video models** with temporal consistency are mostly CUDA-only or non-commercial: BasicVSR++/RealBasicVSR (mmcv stack), Upscale-A-Video (non-commercial), STAR, VEnhancer and FlashVSR (A100-class GPUs), and SeedVR2 (an H100 for 720p).
- **Apple's built-in video super-resolution** (VideoToolbox `VTSuperResolutionScaler`, macOS 26) is temporally stable and licence-free. Its limits: 4× only, input up to 1920×1080 (portrait must be rotated), a one-time model download, and Swift-only access. It's worth adding later as an "Apple Enhance" engine on Macs running macOS 26.
- **Pipeline:**
  - Frames stream through memory and are never written to disk.
  - Audio is copied unchanged.
  - Colour is tagged BT.709.
  - Encoding uses Apple's hardware HEVC encoder (`hevc_videotoolbox`, 10-bit, constant quality) on Macs, and NVENC or x265 on Windows.
  - The `av` (PyAV) package bundles FFmpeg with all of these encoders, so nobody has to install FFmpeg.
- **Phone videos** are often variable frame rate, rotated, or HDR (Dolby Vision or HLG). The app handles VFR and rotation. HDR sources are converted to SDR, with a warning.

## 6. Paid / cloud upscalers (bring your own API key)

Checked against official API docs and pricing pages in October 2026. Prices assume a 3:2 image taken to 8K (8192×5461, about 45 MP) and to 6K. ⚠ marks items I couldn't fully verify.

### Options that reach 6K–8K in one call

| Provider / model | Faithful or creative | 8K in one call? | Approx. cost per 8K image (6K) | Personal key without a business contract? |
|---|---|---|---|---|
| **Topaz API**, Gigapixel High Fidelity V2/V3, Standard V2 | Faithful | Yes (up to 1,024 MP output) | ~$0.20 ($0.20) | Yes ⚠ |
| Topaz API, Wonder 3.5 / Redefine / Recover | Generative | Yes (Wonder up to 128 MP) | ~$1.20 ($0.70) | Yes |
| Topaz API, Bloom 2 | Creative | Yes | ~$2.30 | Yes |
| **fal.ai**, Topaz precision | Faithful | Yes (1–4×) | $0.16 | Yes (prepaid) |
| fal.ai, Topaz generative (Wonder) | Generative | Yes | $0.48 ($0.32) | Yes |
| fal.ai, **SeedVR2** | Mostly faithful diffusion | Yes ⚠ | ~$0.045 ($0.025) | Yes |
| fal.ai, Crystal (Clarity AI) | Portrait-tuned, creativity 0–10 | Yes (up to 200 MP) | ~$0.72 ($0.40) | Yes |
| **Replicate**, Topaz image-upscale (Standard V2, HF V2, Low-Res V2, CGI, Text Refine) | Faithful | Yes (up to 6×, 512 MP) | $0.10 | Yes |
| Runway API, Magnific Precision V2 | Faithful plus detail synthesis | Yes (above 4096 px) | $1.50 | Yes |
| Magnific API direct, Creative / Precision | Creative / faithful | Creative is capped at 25 MP: 6K fits, 8K doesn't | €0.50+ | Paid plans only: API calls spend plan credits. Update Oct 2026: Magnific's pricing page now lists API access on Premium and higher (earlier reports said Business only) ⚠ |

**Can't reach 6K–8K in one call:**

| Provider | Output cap |
|---|---|
| Recraft | 4096 px |
| Stability (Fast / Conservative / Creative) | 4K |
| Google Imagen upscale | 17 MP |
| Leonardo | 20 MP |
| Pixelcut | 6000 px |
| Ideogram | 2× |
| Adobe Firefly API | 6K, enterprise contract only |

### What matters for furniture and people

- **For exact details, use Topaz's faithful models** (High Fidelity V2/V3, Standard V2). Reviewers consistently call them the safe choice when you can't lose identity or shape.
- **Turn face enhancement off.** fal's Topaz endpoints turn it on by default (strength 0.8), and it can change identity. Our integration must default it to off, or set its creativity to 0.
- **Generative modes add the convincing "photographic" detail but can drift** (Topaz Wonder/Redefine, Magnific Creative, Clarity). Magnific's faces are reported to drift at high creativity.
- **No independent leaderboard exists.** The comparisons I found are by vendors or affiliated blogs. Test on your own images: the app's before/after view is meant for exactly this.

### Privacy and terms

| Service | What it keeps |
|---|---|
| Topaz | Inputs and outputs for 7 days. No copyright claim on your images |
| fal.ai | Request payloads for 90 days by default; the app turns this off with the `X-Fal-Store-IO: 0` header |
| Replicate | API inputs and outputs deleted after 1 hour: the best default |
| All of the above | None says explicitly whether it trains on uploads ⚠ |

Uploading confidential team images to any of these is a judgement call. The app always shows which service an image goes to before uploading.

### Subscriptions vs API

- Consumer subscriptions (Topaz Gigapixel desktop, Magnific web) **do not include API access**. API use is separate pay-as-you-go.
- Topaz's EULA forbids driving the desktop app from the command line under a personal licence, so the tool can't automate an existing Gigapixel desktop install.

### Recommendation for which to integrate

1. **fal.ai first.** One key covers Topaz precision (faithful 8K, $0.16), Topaz generative, **SeedVR2 at full 8K on cloud GPUs (~$0.05)** and Crystal for portraits.
2. **Replicate second.** Topaz faithful upscale at $0.10 with the strongest privacy default (deleted after 1 hour).
3. **Later, if wanted:**
   - Topaz direct, for 1,024 MP outputs and exact pixel sizes.
   - Runway, the only pay-as-you-go route to Magnific Precision V2.

**Paid video:** fal Topaz Proteus about $0.08 per second of 4K output, Starlight (generative) about $0.26/s, SeedVR2 about $0.25/s. A 10-second clip costs $0.80–2.60; a 5-minute video $25–80.

## 7. Licences

| Component | Licence | Free family use | Business / team use | Duties |
|---|---|---|---|---|
| PyTorch | BSD-3 | ✅ | ✅ | Keep notice |
| spandrel (model loader) | MIT | ✅ | ✅ | Keep notice |
| Gradio (web UI) | Apache-2.0 | ✅ | ✅ | Keep notice |
| PyAV + bundled FFmpeg (x264/x265) | BSD + GPL build | ✅ | ✅ (using it is fine; only redistributing modified binaries brings GPL duties) | – |
| Real-ESRGAN models | BSD-3 | ✅ | ✅ | Keep notice |
| 4x-ClearRealityV1 | Apache-2.0 | ✅ | ✅ | Keep notice |
| 4xNomos2_hq_dat2 / hq_mosr (Phhofm) | CC-BY-4.0 | ✅ | ✅ | **Credit the author** (shown in the app's About tab) |
| 4x-UltraSharpV2 / Lite (Kim2091) | CC-BY-NC-SA-4.0 | ✅ | ❌ | Non-commercial only, credit the author, share-alike |
| SeedVR2 (ByteDance) | Apache-2.0 | ✅ | ✅ | Keep notice |
| mflux (MLX runner) | MIT | ✅ | ✅ | Keep notice |

**Your own images:** none of these licences claim any rights over the images you upscale.

## 8. Existing free tools (prior art)

- **Upscayl** (AGPL-3.0, free on GitHub/Homebrew, $12.99 on the Mac App Store): images only, older Real-ESRGAN-style models, no video.
- **Final2x** (BSD-3, Mac arm64 build Sept 2026): images, custom models.
- **chaiNNer** (GPL-3.0): powerful node editor, too technical for family use.
- **Video2X**: Windows and Linux only.
- **Topaz Gigapixel / Video AI**: paid desktop apps; see section 6.

What this tool adds over these: commercially safe faithful defaults, exact 6K/8K targets, the finishing stage, video to 4K on a Mac, sharing over home Wi-Fi, and optional paid engines in the same app.

### 4b. Photo-real 4× vs faster 2× routes (on the family's ChatGPT images, M4)

Tested on 256 px crops at 4× (the same scale as 2K→8K), with the model kept loaded:

| Route | Time per crop | What it looks like at 100% |
|---|---|---|
| **SeedVR2 4×** | ~22 s | **Best.** The most natural, photographic micro-detail: fine bread crumb, true rug weave, delicate plant leaves |
| SeedVR2 2× + Lanczos 2× | ~5 s | Natural but softer; textures coarser (bigger "carved" holes in bread, chunkier rug pattern) |
| SeedVR2 2× + Fast AI | ~5 s + 1 s | Crisp, but with harsh, etched GAN edges: the "sharper but not better" look the user dislikes |

**Decision:**
- Photo-real stays **SeedVR2 4×** by default ("Best quality", about 10–22 min per 2K→8K image).
- A **"Quicker"** option uses SeedVR2 2× followed by a Lanczos resize: about 4× faster, softer at 100%, and still free of GAN harshness.

**16K second stage**, applied to SeedVR2 4× output: the Fast AI 2× pass adds crisp but natural micro-detail. On clean diffusion output it does not produce the harshness seen above. Lanczos looks soft, and a 50/50 blend sits in between. **Decision:** keep the Fast AI second stage for 16K, as approved, with the usual detail-strength control.

## 2b. Auto-suggest calibration

`upscaler/suggest.py` measures cheap signals on the processor and applies the DESIGN rules in order. Measured on this M4 (October 2026); the family's private images were only read, never copied.

**How each signal is measured:**
- **AI mesh:** the luminance spectrum of up to 64 tiles of 128 px (from the more detailed half of the picture), averaged, minus its own 9-bin median background. The score is the strongest peak at periods of 2–5 px, in units of the leftover's robust spread (median absolute deviation), so busy and plain pictures score alike. Taking the single highest peak instead was too noisy: one ChatGPT interior swung between 1.1 and 2.4 depending on which tiles were used.
  - It is measured on an area-averaged copy no longer than 2048 px. At full size, the real 3840 px Tahoe photo scored 26.8 and unresized 3000 px crops of it 26–28, so no large photo could ever reach rule 6 (see Findings).
  - A picture with no real texture scores 0: the tiles' sharpness (below) must be at least 20. The leftover's spread is floored at 0.1 (real pictures leave 0.14–0.29): a smooth 8-bit gradient leaves almost none, and its regular 8-bit steps scored 0.1–5.9 million. Smooth gradients have a sharpness of 0.1–10 (with σ=1 noise); the softest AI image's is 126.
- **Illustration:** the stronger of two tests at 768 px.
  - *Flat digital art:* 8 px blocks that are almost exactly two tones. The score rises from 0 at 41% of the busy blocks to 1 at 52%, so the cut-off is about 47%.
  - *Painted or printed art:* four colours explain at least about 90% of the colour variance (k-means in Lab), colour changes are abrupt rather than gradual, and edges cover at least 30–40% of the picture.
- **Text:** MSER letter shapes grouped into lines of 3 or more, kept only if the line is high-contrast and two-toned with calm space above or below. The score is the share of the picture covered by such lines.
- **Faces:** YuNet (OpenCV Zoo, MIT) at 640 and 256 px, confidence at least 0.85.
- **Sharpness:** the 90th-percentile Laplacian variance of the 128 px tiles, at full size.
- **Blockiness:** the step between neighbouring pixels on the strongest 8 px grid position, compared with the median position and with the position half a block away (1.0 = no blocks).

**Thresholds:**
- large faces: over 3% of the picture
- illustration: ≥ 0.5
- text: ≥ 0.004 (0.4% of the picture)
- mesh: ≥ 9
- small: < 1200 px; large: ≥ 3000 px
- sharp: ≥ 150
- blocky: ≥ 1.2

| Picture | Long edge | Mesh | Illustration | Text | Sharpness | Blocks | Suggestion |
|---|---|---|---|---|---|---|---|
| ChatGPT flat-lay 1 | 1672 | **19.3** | 0 | 0.0007 | 2159 | 1.01 | Photo-real Balanced (mesh) |
| ChatGPT flat-lay 2 | 1672 | **16.2** | 0 | 0 | 2450 | 1.01 | Photo-real Balanced (mesh) |
| ChatGPT interior 1 | 2048 | **11.4** | 0 | 0 | 130 | 1.04 | Photo-real Balanced (mesh) |
| ChatGPT interior 2 | 2048 | **14.5** | 0 | 0 | 126 | 1.03 | Photo-real Balanced (mesh) |
| 3 more images from the same folder | 1672–2172 | 17.8–21.0 | 0 | 0 | 690–1201 | ≤ 1.05 | Photo-real Balanced (mesh) |
| The 7 ChatGPT images, enlarged 1.6–2.5× | 2675–5430 | 10.5–24.7 | | | | | Photo-real Balanced (mesh) |
| 2 phone screenshots of text-heavy ads | 1596–2048 | 3.8–5.4 | 0 | **0.044–0.050** | ~5000 | ≤ 1.05 | Faithful (text) |
| Sample: interior | 1024 | 3.5 | 0 | 0 | 878 | 1.17 | Photo-real Balanced (small) |
| Sample: food | 1024 | 3.2 | 0 | 0 | 1702 | 1.03 | Photo-real Balanced (small) |
| Sample: Hokusai print | 1024 | 3.5 | **1.00** | 0 | 6652 | 1.04 | Illustration |
| Sample: old photo | 1024 | 3.2 | 0 | 0 | 1784 | 1.07 | Photo-real Balanced (small) |
| Sample: product | 1024 | 3.3 | 0 | 0 | 285 | **1.24** | Photo-real Balanced (small, blocks) |
| Tahoe photo, full 3840 | 3840 | 5.1 | 0 | 0 | 219 | 1.06 | **Faithful for sharp photos** |
| Tahoe, unresized 3000×2000 crops (12) | 3000 | 6.0–8.5 | 0 | 0 | 208–243 | ≤ 1.10 | Faithful for sharp photos |
| Tahoe, unresized 3000×1688 crops (12) | 3000 | 5.1–9.7 | 0 | 0 | 205–296 | ≤ 1.19 | 9 Faithful for sharp photos, 3 Photo-real (mesh) |
| Tahoe, resized to 3000 or 3200 | 3000–3200 | 3.7–5.9 | 0 | 0 | 242–448 | ≤ 1.02 | Faithful for sharp photos |
| Tahoe, resized to 2048 | 2048 | 5.1 | 0 | 0 | 577–1169 | ≤ 1.07 | Photo-real Balanced (default) |
| Tahoe at 1280, bicubic back to 3840 | 3840 | 4.3 | 0 | 0 | 25 | 1.01 | Photo-real Balanced (soft) |

Findings:
- **Mesh:** the threshold sits at 9.
  - *AI images:* the lowest ChatGPT image scores 11.4 at its own size, and 10.5 when enlarged 2× with Lanczos. The flat-lays peak at the checkerboard frequency (a diagonal period of 2 px). The interiors peak at a horizontal and vertical period of about 2.8 px and a diagonal one of about 4 px.
  - *Pictures that aren't AI-made:* 3.2–5.4 at 2048 px or less, and 5.1 for the full Tahoe photo.
  - *Unresized 3000 px crops:* 3000×2000 crops of the Tahoe photo score 6.0–8.5. Wider 3000×1688 crops score 5.1–9.7, so 3 of 12 still get the mesh reason and Photo-real: a 2048 px copy keeps part of the photo's own pixel pattern at that size. 3500×2160 crops score 4.5–5.7, and phone and camera photos are 4000 px and more.
  - *Why it is measured at 2048:* at full size, the Tahoe photo carries a period-2–3 px pattern, probably from the camera's or Apple's processing, and scored 26.8. That sent every large photo to Photo-real with the reason "faint grid-like texture". The earlier note here that 5.4 was the highest real photo "at its own size" was wrong: 5.4 was for the 2048 px copy.
  - *The trade-off:* an AI picture made at more than 2048 px loses some of its mesh in the copy. A synthetic period-2 crosshatch scores 50 at 2048 px, 20 at 2752 px and 4.2 at 3200 px; periods of 3–5 px still score 24–30 at 3200 px.
  - *JPEG weakens the mesh:* the ChatGPT images saved at JPEG quality 90 score 4.8–15.1, and at quality 75 they score 3.5–6.8. Under 3000 px these get Photo-real Balanced anyway, by rule 7.
  - *The reason text* says "common in AI-made and resized pictures", because a real photo cropped at its own pixels can show the pattern too.
- **Sharpness** only matters for rule 6 (3000 px or more, no mesh, no blocks). The real 3840 px Tahoe photo scores 219, its 3000–3200 px crops and copies 205–448, and its soft re-enlargement 25, so the threshold sits at 150. Synthetic sharp and soft 3200 px pictures score about 1550 and 18.
- **Text:** only the two screenshots pass, by a factor of 10. A synthetic poster scores 0.013, a single logo word 0.008, and a small caption on a busy photo 0.
- **Faces:** no face was found in any calibration picture. The old photo's face is in profile and looking down: YuNet gave it 0.49, under the 0.85 cut-off. This set has no frontal portrait, so the face rule is tested with a stand-in detector; YuNet's own accuracy is on its model card.
- **Illustration** is the weakest signal.
  - *What it catches:* flat cartoons, graphics and the Hokusai print. The print's score stays 0.87–1.0 after JPEG re-saving, resizing or cropping.
  - *macOS account pictures (43, at 512 px):* with the first cut-off (a two-tone share of about 33%), it flagged the yin-yang, the target, the 8-ball, the soccer ball and the snowflake. These are not all graphics: the soccer ball and the snowflake are photos, and the 8-ball looks like one. Product photos cut out on a plain background are everyday work for the design team.
  - *The raised cut-off (about 47%):* the soccer ball (0.38) and the 8-ball (0.42) are no longer flagged. The yin-yang (0.63) and the target (0.62) still are.
  - *The snowflake* is a macro photo and is still flagged, by the painted-art test. That test can't tell it from the Hokusai print: inks 0.920 against 0.917, abrupt/gradual 0.34 against 0.36, edges 0.51 against 0.42.
  - *The cost:* the test cartoons at 1200 px score shares of 0.51–0.66 as PNG or JPEG at quality 60–85, and stay flagged. The same cartoons saved at 600–800 px as JPEG score only 0.35–0.47, like the cut-out photos, so most of them now get Photo-real.
  - A sunflower close-up scores 0.43. No photo or AI image from the table comes close.
- **Synthetic checks** (in the tests):
  - A 3 grey-level crosshatch at a 2–5 px period scores 19–50 at 2048 px. Enlarged 2× to 3200 or 4096 px, it scores 19–52, so a large, sharp, enlarged AI picture still gets Photo-real.
  - The same photo saved as JPEG at quality 40–90 stays at 3.2–3.8, with blockiness 1.45 at quality 40, 1.18 at 75 and 1.01 at 90.
  - Smooth gradients score 0, with or without σ=1 noise.
- **Speed:** `analyze()` takes 0.07–0.17 s for these pictures on the M4's processor, including YuNet (about 13 ms). The full Tahoe photo takes 0.16 s and an 8K picture 0.21 s (target: under 1.5 s for 2K).
  - A 2048×1536 page of dense type (12–20 px, about 10–50 thousand letter candidates) takes 0.3–0.4 s. Letters are only paired with letters in nearby row bands, and nested letter boxes are found through a coarse grid. Comparing every letter with every other took 1–3 s on such pages.
- **The face model is 232,589 bytes** (SHA-256 `8f2383e4…2552fa4`), downloaded on first use from opencv_zoo, pinned to commit `f12e127`.
  - If it can't be downloaded, suggestions work without faces.
  - A picture waits at most 3 s for that first download, and a failed download is retried after 10 minutes.
  - While it downloads, models already on the computer still load at once: `ensure_model` only waits for the download lock when a model is missing.

## 9. End-to-end acceptance (the real app, Apple M4 16 GB, 2026-10-17)

Run through the command line on the family's own ChatGPT images. The images are kept private, and none of them or their outputs are in this repo.

| Job | Output | Time | Estimate shown by the app |
|---|---|---|---|
| Fast → 8K (2048×1539) | 8192×6156 JPEG, 300 DPI | 5 s | under a minute |
| **Photo-real Balanced → 8K** (2048×1539, the default) | 8192×6156 JPEG | **5 min 12 s** | ~5.4 min |
| Faithful → 8K (2048×1539) | 8192×6156 JPEG | 2 min 54 s | ~3 min |
| Photo-real Balanced → 6K (941×1672 portrait) | 3458×6144 JPEG | 2 min 34 s | – |
| Photo-real Balanced → **16K TIFF, 300 DPI** (2048×1539) | 16384×12312 TIFF (deflate), 276 MB | 5 min 28 s | – |
| Photo-real Rich → 8K (941×1672 portrait) | 4610×8192 JPEG | 9 min 32 s | ~10 min |
| Video Fast: 5 s 1080p → 4K | 3840×2160 HEVC Main10 `hvc1`, BT.709, AAC audio | 2 min 46 s | – |
| Video Fast: 2 s 720p phone clip with 90° rotation | 2160×3840, upright | 34 s | – |

- **Visual check at 100%:** no tile seams. Balanced gave crisp, natural detail (panel edges, plant, wood grain, marble texture). Faithful kept clean fabric edges.
- **Launcher:** `Start Re-Solution.command` started the server, printed "Re:Solution is running." and served the UI. `/api/status` reported Apple M4, Photo-real supported, weights present.

**Final checks, 2026-10-18** (Apple M4):
- Test suites: 1,441 default tests, 9 GPU tests with the real models including SeedVR2, 7 browser end-to-end stress tests, and 4 slow memory tests. All pass.
- Smoke run: Photo-real Balanced turned the 1024 px interior sample into a 4K image in 1 min 17 s. A 4 s 720p clip from a real photograph became 4K video in 54 s.
- The coloured ringing the Fast video model shows on the synthetic `testsrc2` colour bars does not appear on natural footage. Tree against sky and rocks against water were clean at 100%.
