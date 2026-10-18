# Upscaler design (v1)

Status: approved and built. This page records the intent and the decisions in the order they were made: the v1 design first, then the v1.1 additions at the end. Where the build changed a v1 decision, the line says so ("Changed during the build"). The [README](../README.md) describes the app as it ships. Background and measurements are in [RESEARCH.md](RESEARCH.md).

## Intent

Family members and a small team generate AI images at roughly 2K–4K, mostly of **furniture/interiors and people**. They want 6K and 8K versions that look truly high quality, not just sharper, and they need to **keep the exact details**: faces and identity, furniture shapes, wood grain, fabric, logos. They also want to upscale videos (short AI clips and longer home videos) to 4K.

The tool must be free, run locally, and be usable by non-technical people on Apple Silicon Macs and on Windows PCs with NVIDIA GPUs. Optionally, people can plug in their own paid cloud upscaler account for maximum quality.

## Goals

1. Upscale images to exact 6K or 8K sizes, plus 4K and custom scales, with the most faithful high-detail local model available.
2. Make the result *look* high quality, not just sharper. A finishing stage locks the original's colour and tone, controls detail strength and optionally adds fine grain.
3. Upscale videos to 4K with sound kept. Show progress and an ETA, allow cancelling, and offer a "preview 3 seconds first" option for long videos.
4. Let each person optionally use their own paid upscaler subscription or API key for the best quality. Keys stay on that person's computer.
5. One double-click to start, on Mac and on Windows. The app opens in the browser. It can also be shared over home Wi-Fi so others just open a link.
6. Default models must be allowed for commercial use (team imagery).

## Non-goals (v1)

- Hosting on the internet or user accounts.
- Face "restoration" models (GFPGAN, CodeFormer). They change identity and some are non-commercial.
- A signed native macOS app or Windows installer.
- Editing features (crop, colour grading, background removal).

## Anti-goals (would count as failure even if it technically works)

- Output that changes faces, furniture shapes, logos or text without the user choosing a "creative" mode.
- "Crunchy" oversharpened output with halos.
- Colour or tone shifts from the original.
- Setup that needs Terminal knowledge for the normal path.
- Silently sending images to a cloud service, or using someone's API key without their say.
- Shipping non-commercial models as defaults.

## Constraints

- Main machine: Apple M4, 16 GB unified memory, macOS 26. Others: Apple Silicon Macs and Windows with NVIDIA. CPU-only works, but slowly.
- Everything free. Licences must allow the use; see RESEARCH.md §7.
- Windows can't be tested on this machine, so that path ships as "should work, untested". *Changed during the build:* GitHub Actions now runs the automated tests on Windows, an Apple Silicon Mac and Linux on every change (`.github/workflows/ci.yml`), including a dry run of the Windows installer's checks and a test of its copy step in PowerShell 7 and 5.1, and a start-up check. A full install on Windows hasn't been run yet, and those computers have no NVIDIA card, so Photo-real on NVIDIA is still untested.

## How it works

### Stack (my call)

- **Python 3.12**, managed by **uv**. The launcher installs uv and Python automatically on first run.
- **PyTorch** for inference: Apple GPU (MPS) on Macs, CUDA on NVIDIA, CPU fallback. **spandrel** loads the models.
- **Custom web UI**, changed from Gradio because the user asked for a "nice, beautiful and guided" experience:
  - Built on **FastAPI + Uvicorn** with a hand-built single-page frontend (HTML/CSS/vanilla JS, nothing loaded from the internet), so LAN guests and offline use work.
  - Server-sent events drive live progress.
  - It runs locally and opens in the browser. Turning on "share on Wi-Fi" binds it to the local network.
- **imageio-ffmpeg**'s bundled `ffmpeg` program decodes, encodes and joins video. It uses the hardware HEVC encoder on Macs (VideoToolbox), NVENC on NVIDIA and x265 otherwise. **PyAV** only reads each video's details (frame rate, rotation, colour, HDR). *Changed during the build: v1 planned PyAV for decoding and encoding as well.*
- **Pillow** for image I/O, pinned below 13 (13.0.0 fails to import, as found during research).
- A small command-line tool alongside the UI (`python -m upscaler image photo.jpg --size 8k`, run with the Python in the app folder's `.venv`) for power users and testing. *Changed during the build: v1 named it `upscale`, but the app isn't installed as a Python package, so there's no separate command.*

### Image pipeline

1. Load the image and apply its EXIF rotation. Keep the ICC colour profile.
2. Work out the target size from the preset: 4K = 3840 px long edge, 6K = 6144, 8K = 8192, "8K TV" = fit within 7680×4320, or a custom 2×/3×/4×.
3. Run the model in 512 px tiles with 16 px of context, choosing the precision per model (fp16, or bf16 for DAT2). A 4× model runs once. If more than 4× is needed, it runs twice, with an exact resize in between, and the app warns about it.
4. Resize to the exact target with Lanczos.
5. Finishing:
   - Tone and colour lock (always on).
   - Detail strength. Default 100%; a per-mode default is allowed.
   - Grain: off, subtle or medium. Default off.
6. Save as high-quality JPEG (q95, 4:4:4) by default, or PNG. Keep the ICC profile. Save to the download button *and* to `Pictures/Upscaled` on the machine running the app.
7. Batch: several images at once, downloadable as a zip.
8. Show a before/after comparison at 100% zoom on a crop, because a whole 8K image shrunk to screen size hides the difference.

### Local quality modes (images)

| Mode | Model | Licence | Approx. 2K→8K on M4 |
|---|---|---|---|
| Faithful | **4xBHI_dat2_real** (chosen after testing on the family's ChatGPT images: cleanest edges and real fabric texture, and it doesn't keep the source's softness) | CC-BY-4.0 | ~3 min |
| Faithful – already-sharp photos (under "More modes") | 4xNomos2_hq_dat2 (best on clean, sharp photos; amplifies the ChatGPT mesh texture) | CC-BY-4.0 | ~3 min |
| Fast | 4x-ClearRealityV1 | Apache-2.0 | ~2 s |
| Illustration / anime | RealESRGAN_x4plus_anime_6B | BSD-3 | ~15 s |
| **Photo-real (default)** | SeedVR2-3B: mflux (MLX) on Apple Silicon; PyTorch runner on NVIDIA | Apache-2.0 | ~3–6 min for Balanced, the default level; ~10–22 min for Rich (see the v1.1 detail levels) |

**Photo-real mode (SeedVR2) is the default for images**, as the user decided. It passed testing on the M4 (RESEARCH §4a). It is the mode that fixes "sharper but not better": on a JPEG-damaged mountain it produced a convincing photo where the transformer model made streaks. Setup and behaviour:
- **First use:**
  - Installs an add-on of about 0.5 GB of packages and downloads the 7 GB model once.
  - A friendly prompt explains this first: "Photo-real needs a one-time 7 GB download (about 10 minutes on fast Wi-Fi). Start download / Use Faithful instead".
  - Download progress is shown, and the model is stored once per computer.
- **Processing:**
  - Overlapping 512 px input tiles (4× each), wide feathered blending, a fixed seed, then the tone/colour lock.
  - Low-RAM mode peaks at about 11 GB.
  - 1024 px tiles were measured at 18.4 GB, so they are not used on 16 GB machines.
- **Automatic fallback to Faithful,** with a one-line explanation in the UI, when:
  - the computer has under 16 GB of memory (Apple Silicon) or under 12 GB of VRAM (NVIDIA)
  - the computer has no supported GPU
  - the add-on failed to install
- **Hybrid option (builder decides from measurements):** a 2× SeedVR2 pass to about 4K, then the Balanced model or Lanczos to 8K, if that is much faster at similar quality.
- **Warning in the UI:** "Photo-real invents fine detail. Check faces, logos and text, or switch to Faithful."
- **Windows/NVIDIA:** SeedVR2 runs through a PyTorch runner, the standalone CLI from numz's Apache-2.0 SeedVR2 project or an equivalent; the builder verifies which one works headless.
  - Untested on this Mac, so it ships as "should work".
  - Cloud SeedVR2 (fal.ai, about $0.05 per image) is the backup.

The video default stays the fast ClearRealityV1 model. Local SeedVR2 video on a Mac would take hours per second of footage.

Models download on first use from their official sources and are checked against SHA-256 hashes. They're kept in the `models` folder inside the app's data folder (`~/Library/Application Support/Upscaler` on a Mac, `%LOCALAPPDATA%\Upscaler` on Windows), and Photo-real's 7 GB model in the shared Hugging Face cache, so updating or re-installing the app keeps them. *Changed during the build: v1 kept them in the app folder.* The About tab credits every model and links its licence; CC-BY requires this.

The Balanced model (4xNomos2_hq_mosr) was dropped after testing: it amplified the ChatGPT mesh texture without improving on Fast.

Non-commercial models (UltraSharp, UltraSharpV2) are not shipped in v1 (my call). They looked no better than the CC-BY model on detail fidelity, and they would cause licence trouble for team use.

### Video pipeline

1. Probe the video: fps, rotation, colour, HDR. Decode frames in memory.
2. Upscale each frame (default model ClearRealityV1, or general-x4v3), then resize to 4K (3840 long edge, keeping the aspect ratio) and apply finishing (colour lock and detail strength).
3. Encode as HEVC 10-bit with BT.709 tags in an MP4 (with the `hvc1` tag, so it plays in QuickTime), copying the original audio. Windows' own Photos and Media Player apps need Microsoft's HEVC Video Extensions to play it, so the README says so and suggests VLC as a free alternative.
4. Progress bar with ETA (calibrated on the first frames), cancel button, and "Preview first 3 s".
5. HDR iPhone video is converted to SDR, with a warning.
6. **Long jobs survive interruptions** (my call; needed because a 5-minute 1080p video takes about 2 hours):
   - Jobs run in a background worker inside the app server, not tied to the browser tab. Closing the tab doesn't stop a job, and reopening the page shows its progress.
   - Video is encoded in segments of about 10 seconds into a job folder. After a crash or restart, the job resumes from the last finished segment. At the end, the segments are joined and the audio is added.
   - The computer is kept awake while a job runs (`caffeinate` on Mac, `SetThreadExecutionState` on Windows).
   - *Added during the build:* a video that runs out of disk space keeps its finished segments, and **Retry** carries on from them once there's room (`POST /api/jobs/{id}/retry`).
7. **Old camcorder or tape transfers:**
   - Interlaced video is detected and deinterlaced (bwdif).
   - Non-square pixels are converted to square.
   - Sources that need more than 4× (480p to 4K) get one 4× model pass and then a Lanczos resize. There is no second model pass for video, because it would multiply both time and flicker.

Cloud engines for video are out of scope for v1 (my call). A 5-minute video would cost $25–80, and long remote jobs need a different interface. RESEARCH §6 lists the options for later.

### Cloud engines (bring your own key)

- A provider plug-in interface: `upscale(image, target_size, mode) -> image`, with a cost estimate shown before running.
- Providers and what each unlocks (RESEARCH §6):
  - **fal.ai.** One key unlocks:
    - Topaz faithful precision (~$0.16 per 8K image)
    - Topaz generative Wonder (~$0.48)
    - SeedVR2 at full 8K (~$0.05)
    - Crystal for portraits
  - **Replicate.** Topaz faithful upscale at ~$0.10, and uploads are deleted after 1 hour.
- Face enhancement defaults to **off** (and creativity to 0) on every provider, so identity is preserved.
- Results come back at the provider's size. The app then does the exact resize to 6K/8K and applies the same tone/colour lock as local results.
- Live calls are only tested once someone adds their own key. Until then the integration is covered by offline tests against recorded API shapes.
- Providers: **fal.ai** and **Replicate** (user's choice), plus **Magnific** because someone already has a Magnific/Freepik subscription.
  - The Magnific guide explains that "unlimited" web allowances don't cover API use: every API call spends plan credits. As of October 2026, Magnific's pricing page lists API access on Premium and higher plans, with Business aimed at API use; this was verified by the cloud task and Magnific has changed it several times. Runway resells Magnific Precision pay-as-you-go.
- **Guided setup.** The user asked for a "nice, beautiful and guided" flow, because nobody has an API key yet. A friendly wizard covers, per provider:
  1. What it is, what it costs per 8K image, and what happens to uploaded images.
  2. Step-by-step sign-up, with direct links to the provider's sign-up, billing and API-key pages.
  3. Pasting the key and testing the connection with an account check that doesn't spend money.
  4. A success screen.
  The wizard is reachable from Settings and from an "Unlock cloud quality" card on the main screen. People who never want cloud can ignore it entirely.
- Keys:
  - Entered in the wizard or the app's Settings tab.
  - Stored in the operating system's secure keychain (macOS Keychain / Windows Credential Manager, via the `keyring` package). Never written to the project folder.
- **Sharing over Wi-Fi:** guests can't use the host's keys. The host can tick "let guests use my cloud account", but only after setting an access password (below), so nobody else on the network can spend the host's money. Guests can paste their own key, which is kept for that browser session only.
- Before any upload, the app shows a clear notice of which service the image goes to, plus that service's retention policy.
- *Added during the build:* a cloud job that was running when the app closed, crashed or restarted is ended, not sent again, because the service may already have charged for it. The pictures it finished stay listed, and the person starts it again by hand. A guest's own key lives only in memory, so their waiting cloud jobs are stopped after a restart.

### Packaging and launch

- `Start Re-Solution.command` (Mac) and `Start Re-Solution.bat` (Windows):
  - Double-click.
  - Installs uv if it's missing, then runs the app; the first run downloads about 400 MB, or about 3 GB on a PC with an NVIDIA card.
  - Opens the browser at `http://127.0.0.1:7860`.
- `Start Re-Solution (Share on Wi-Fi)` variants of both: they start with sharing on and print the address to open on other devices (for example `http://192.168.1.20:7861`: sharing uses the app's port + 1, see "Header controls" below). *Changed during the build: v1 named the launchers `Start Upscaler` and shared on the app's own port.*
- README with step-by-step setup:
  - The Mac Gatekeeper "unidentified developer" workaround.
  - The firewall prompt.
  - How to share the folder (zip via AirDrop/Drive).

### Sharing and the job queue (my calls)

- **Access password:** optional, off by default on home Wi-Fi. Required before guests can use the host's cloud account.
- **Queue:** one GPU worker, jobs run one at a time in order, and the queue is visible to everyone. The host (on the machine itself) can cancel any job; guests can cancel their own.
- **Guest results:** kept in the app's temporary folder and deleted after 24 hours. They are not saved into the host's `Pictures/Upscaled`, which avoids privacy and disk-space problems. The host's own results are saved there.

### Research prototypes and reference images

- The benchmark and prototype scripts used in the research (tiling, precision choice, colour lock and grain, comparison grids) are copied into `research/` for reference.
- Real furniture and people images from the family or team go in `samples/`. The person who asked for the tool judges quality on those, using the app's before/after view.

## Left to the builder

UI layout details, file and module structure, the exact grain and colour-lock maths (prototype in research), the exact ETA method, and test structure.

## v1.1 additions (user requests relayed 2026-10-17)

### Product identity
- **Name: "Re:Solution"**, a pun on resolution plus solving a problem. File and repo names use `Re-Solution` / `re-solution`, because colons are invalid in Windows filenames and GitHub repo names. The internal Python package stays `upscaler`.
- **Credit**, shown in the app footer and About page and in the README: "Designed by Swati Srivastava".
  - Colourful icon links to the portfolio (https://swatisrivastava.in) and LinkedIn (https://www.linkedin.com/in/swatisrivastava1010).
  - LinkedIn uses its blue "in" mark for the profile link; the portfolio uses a gradient globe. *Changed during the build:* the portfolio link uses Swati's own "S" monogram from her site (`upscaler/web/brand/swati-mark.png`, used as a CSS mask so it takes the brand colour in both themes), instead of the gradient globe.

### 16K output (user's choice: hybrid)
- **Presets:**
  - `16k`: 16384 px on the long side.
  - `16k_tv`: fits within 15360×8640, orientation-aware like `8k_tv`.
- **Pass planning when more than 5× is needed:**
  - **Pass 1:** the chosen mode at 4× (Photo-real, Faithful, cloud, and so on).
  - **Pass 2:** always the **Fast** model, going straight from the pass-1 result to the exact target. Each tile is upscaled 4×, then resized to the needed scale before being written into the target canvas, so no 32K intermediate image ever exists.
  - This also replaces the earlier "two passes of the same model" rule for small sources.
- **Honesty in the UI:**
  - Before a 16K job starts, show the time estimate (e.g. Photo-real Balanced 2K→16K on an M4 takes about 3–6 min, Rich about 11–22 min) and the estimated file size (JPEG ~40–100 MB; PNG/TIFF ~300–600 MB).
  - Note that 16K mainly helps large prints: on screens it looks the same as 8K.
  - Check free disk space before starting.
- **Limits:**
  - JPEG supports up to 65,535 px, so it's fine at 16K.
  - WebP isn't offered, because it can't exceed 16,383 px.
  - Finishing runs in strips, so memory stays bounded. The 16K canvas is about 0.6 GB of uint8.

### Designer features
- **Output formats:** JPEG (q95, 4:4:4), PNG, and **TIFF** (8-bit, deflate compression) for print workflows.
- **DPI written into every file,** default 300; choices 72 / 150 / 300 / 600.
- A live readout per image before starting, e.g. `2048×1539 → 16384×12312 · 54.6 × 41.0 in (138.7 × 104.2 cm) at 300 DPI · ≈ 45 MB JPEG · ≈ 18 min`.
- **Use-case presets** (one click; each sets mode, size, format and grain, and every choice stays editable):

  | Preset | Mode | Size | Notes |
  |---|---|---|---|
  | Interiors & furniture | Photo-real | 8K | |
  | Products & e-commerce | Faithful | 8K | Keeps logos and text exact |
  | People & portraits | Faithful | 8K | Tip: "try Photo-real for more skin detail" |
  | AI art | Photo-real | 8K | |
  | Print & posters | Photo-real | 16K | TIFF, 300 DPI |
  | Illustration & anime | Illustration | 8K | |
  | Old photos | Photo-real | 4K | |

- Batch processing with a zip download is already in v1. Also add drag-and-drop anywhere on the page and paste from the clipboard (⌘V / Ctrl+V).

### UX direction: keep the current design, add features (user correction)
The user said not to copy Pixelcut's design: "i liked the features they had. Our current tool is also good." So **keep Re:Solution's existing look, layout and visual language** from v1 Task 7. Use Pixelcut's upscaler only as a checklist of *features* to add, applied within our current design.

**The user also allowed UX improvements:** "You can improve the UI/UX as well if you think we can make it even better for the user." So the UI pass:
1. Starts with a UX critique of the current interface, using the impeccable critique/audit approach.
2. Improves anything that clearly helps non-technical users and designers: hierarchy, wording, flow, feedback, empty and error states, accessibility, phone layout.
3. Keeps Re:Solution's own visual identity, never Pixelcut's.

**Features to add (feature parity, our own design):**
- **Drop anywhere** on the page, and **paste from the clipboard** (⌘V / Ctrl+V).
- **"Try a sample"** row: a few small built-in sample images to try the app in one click. Hidden if no samples are bundled; see the open question about sample images.
- **Use-case presets** that set mode, size, format and grain (table in "Designer features"). They appear as chips in the existing options area.
- **Creative controls for Photo-real,** two plain-English sliders shown when Photo-real is selected:
  - **"Creativity"** (0–100%, default 0) maps to SeedVR2's `softness` 0–0.5. Higher values let the model invent more new detail.
  - **"Stay close to the original"** (0–100%, default 0%) blends the Photo-real result back toward a plain Lanczos enlargement of the original: 0% keeps Photo-real's full result and 100% gives the plain enlargement. It sets Photo-real's detail strength (100% minus the slider), with the tone lock always on. *Changed during the build: first planned as a detail strength starting at 100%; the slider was turned round so that 0% means no blending.*
- **Before/After compare labelled with sizes** ("Before 2048×1539" / "After 8192×6156"), plus **zoom to 100% and pan** on results, added to the existing result cards and slider.
- **Up to 16K,** with the size, time and file-size readout before starting.
- **Reassurance pills** in the current header or hero area: "Runs on your computer", "No watermark", "No sign-up", "Up to 16K".
- **"Sharpens & cleans up noise and compression"** stated plainly in the mode descriptions.
- **Bulk:** the existing batch and zip, plus a clear "Download all" and a per-batch progress summary.
- **Video in the same app** (already there).
- **For power users:** a short "Use it from the command line" note in About and the README. This is our version of a developer API.

**Polish requested by the user, applied to the existing design:**
- **Clear colour coding, used consistently everywhere:**

  | Mode | Colour | Icon |
  |---|---|---|
  | Photo-real | violet | sparkles |
  | Faithful | teal | shield-check |
  | Fast | amber | lightning bolt |
  | Illustration | pink | brush |
  | Cloud | sky blue | cloud |
  | Video | indigo | film |

  Job states use grey (queued), blue (working), green (done) and red (problem). Colour always comes with an icon and a text label.
- **Motion:** gentle and purposeful. The drop zone lifts on drag, cards fade and slide in, progress has a smooth shimmer, results reveal with a before→after wipe, and a small success burst plays on completion. All of it respects `prefers-reduced-motion`.
- **Self-explanatory icons:** inline SVGs paired with text labels. Icon-only buttons only where the meaning is universal, and those carry tooltips and aria-labels.

### One-step install from GitHub (user's choice: public repo, MIT licence)
- **Repo:** `github.com/swatisrivastava1010/re-solution`. It's created and pushed only after the app passes its tests (user-approved).
- **One-line installs:**
  - Mac (Terminal): `curl -fsSL https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.sh | bash`
  - Windows (PowerShell): `irm https://raw.githubusercontent.com/swatisrivastava1010/re-solution/main/install.ps1 | iex`
- **The installer:**
  1. Detects the OS, CPU architecture and GPU (Apple Silicon; NVIDIA via `nvidia-smi`; otherwise CPU).
  2. Installs uv without touching shell profiles.
  3. Downloads the app with `git clone` if git exists, otherwise the GitHub zip of `main`.
  4. Installs to `~/Applications/Re-Solution` (Mac) or `%LOCALAPPDATA%\Re-Solution` (Windows).
  5. Installs the matching PyTorch build with `uv sync --frozen --extra <cpu|cu130>`:
     - Macs use the `cpu` extra; Apple GPU support (MPS) is included in those wheels.
     - NVIDIA PCs use `cu130`.
     - Others use `cpu`.
  6. Creates shortcuts. Mac: a `Re-Solution` app in `~/Applications` plus a Desktop alias. Windows: Start Menu and Desktop shortcuts.
  7. Starts the app.
- **Updating:** close Re:Solution, then run the same command again. Your settings, keys and models are kept. The installer refuses to update a copy that is still running.
- **The double-click launchers** (for zip-shared copies) use the same detection.

### All image formats (user request: "all kinds of image formats including heic etc.")
- **Inputs:**
  - JPEG, PNG, WebP, HEIC/HEIF (iPhone), AVIF.
  - TIFF: 8/16-bit and multi-page; multi-page uses the first page, with a note.
  - BMP; GIF and animated WebP (first frame, with a note that animated files aren't upscaled frame by frame yet); JPEG 2000, ICO (largest size), TGA, PPM.
  - Photoshop PSD: flattened composite. If that fails, a friendly "save as PNG/TIFF from Photoshop" message.
  - **Camera RAW** (DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2) via `rawpy`, developed with camera white balance to sRGB.
  - **JPEG XL** only if a permissively licensed Pillow plugin exists; otherwise a friendly message.
- **Vectors (SVG, PDF, EPS)** don't need AI upscaling. SVG is rendered directly at the target size if a permissive renderer is available; otherwise the app explains "this is a vector file; export it as PNG at the size you need".
- **Colour:** CMYK, grey and Lab inputs are converted to sRGB through their embedded ICC profile (Pillow ImageCms) before upscaling. RGB inputs (sRGB, Display P3, Adobe RGB) keep their own profile, which is embedded in the result, so the most vivid P3 and Adobe RGB colours aren't clipped to sRGB.
- **Browser previews:** the server inspects every file on drop. It returns the size, format, warnings and a JPEG thumbnail, because many browsers can't show HEIC, RAW or TIFF. The UI uses these instead of decoding files in the browser.
- **Outputs:** JPEG, PNG, TIFF, plus **WebP** for sizes up to 16,383 px, and **"Same as original"** where the format can be written: JPEG, PNG, TIFF, WebP, HEIC, AVIF. Otherwise it falls back to PNG, with a note.
- **Videos:** anything FFmpeg can read (MP4, MOV, M4V, MKV, AVI, WebM, 3GP, MTS/M2TS, WMV). The file picker lists all of them.

### Photo-real detail levels, auto-suggest, and "Preview styles" (user feedback, 2026-10-17)
On the bread crop the user preferred **SeedVR2 2× + Fast AI** over SeedVR2 4×: "seedvr2 added a lot of texture which does not look natural". RESEARCH §4b showed 4× winning on rug and plants, so the best route depends on the image. The user asked for the app to "suggest users to use what model when i.e auto suggest/compare".

- **Photo-real gets three detail levels.** All use SeedVR2.

  | Level | Route | Speed (2K→8K, M4) | Notes |
  |---|---|---|---|
  | **Balanced (default)** | SeedVR2 2×, then the Fast model to the target | ~3–6 min | Natural with crisp edges |
  | Rich | SeedVR2 4× | ~10–22 min | Most new detail; can over-texture food and organic surfaces |
  | Gentle | SeedVR2 2×, then Lanczos | ~3–5 min | Softest and most literal |

  - For more than 4×, including 16K: Balanced and Gentle add one Fast pass (via `upscale_tiled_to`) to the target, and Rich adds a Fast second stage as before.
  - **First-pass rules** (`sizing.photoreal_route`; the time estimates follow them):
    - **Small pictures:** more than 10× (2× plus the Fast model's 4× can't reach the target) → SeedVR2 runs 4× at every level, so the end isn't a blurry stretch. For example, 1024 px to 16K. On a picture that small, 4× takes minutes.
    - **Rich:** uses 2× when a plain resize can cover the rest: up to 2.5× normally, and up to 2.6× for a large photo that 4× would otherwise shrink. For example, 3200 px to 8K is 2× and then 1.28×.
    - **Large photos:** SeedVR2 always gets every real pixel, as long as its result is no longer than 8K or the target, whichever is bigger. For example, 3000 px to 4K becomes 6000 px and is then resized down, and a 4032 px phone photo keeps every pixel at both 6K and 8K.
    - **Beyond that cap only:** SeedVR2 gets a copy shrunk just enough to land on the cap, which bounds time and memory on a 16 GB Mac. For example, Rich 5000 px to 16K runs 4× on 4096 px instead of making 20000 px, and 5000 px to 6K runs 2× on 4096 px.
    - **The trade-off:** this re-invents some detail on very large photos. Faithful (sharp photos), which auto-suggest picks for sharp photos of 3000 px and up, keeps them literal.
- **Auto-suggest.** Every inspected image gets a recommendation with plain-English reasons, built from cheap local signals:
  - **Faces:** OpenCV's YuNet face detector (MIT model, Apache-2.0 OpenCV).
  - **AI "mesh" texture:** a periodic peak score in the luminance FFT of a few detailed crops.
  - **Illustration look:** flat-region share, palette size, edge hardness.
  - **Text or logo likelihood:** MSER text-like regions.
  - **Size, sharpness** (variance of Laplacian) and **JPEG blockiness.**
  
  Rules, in order:
  1. vector → draw directly
  2. illustration → Illustration
  3. large faces (>3% of the area) → Faithful, with the tip "preview Photo-real Balanced for more skin detail"
  4. likely text or logo → Faithful
  5. AI mesh or a small source (long edge <1200 px) → Photo-real Balanced
  6. sharp, large photo (long edge ≥3000 px) → Faithful (sharp photos)
  7. otherwise → Photo-real Balanced

  The suggestion is preselected for that image (the user's own choice always wins) and shown as a chip: "Suggested: … · Why?".
- **"Preview styles" comparison.**
  - **What runs:** Photo-real Balanced, Rich and Gentle, plus Faithful and Fast, on one ~256 px source crop (4× → 1024 px). The crop is the most detailed region by default, or wherever the user clicks on the image.
  - **What the user sees:** cards at 100% with the style colour and icon, the full-image time estimate and "Use this"; an A/B slider between any two styles; and "Pick for all images in this batch".
  - **Speed:** about 35–40 s on the M4. Preview jobs jump ahead of queued (not running) jobs.

### Header controls: Share on Wi-Fi and theme (user requests)
- **"Share on Wi-Fi" button** (host only): "Share on Wifi button should be there if the user wants to share this over wifi".
  - The app normally listens only on 127.0.0.1, so the operating system's firewall never asks.
  - Switching sharing on starts a second listener for the same app on `0.0.0.0` at port+1, without a restart. Switching it off stops that listener.
  - Requests through the LAN listener are always guests.
  - The panel shows the address or addresses, a copy button, a **QR code** for phones (segno, BSD-3), the macOS "click Allow" note, and a warning plus a Settings link when no password is set.
  - The button shows a clear "Sharing on" state.
  - The Share launchers start with sharing on. The state is not remembered across restarts.
- **Theme button:** a sun/moon/auto icon that cycles Light → Dark → Auto (follows the system). The choice is remembered per browser and applied before the first paint, so there's no flash. All colours are CSS variables for both themes.

### Batch-friendly layout (user request, 2026-10-18)
"if users upload a lot of images, the UI should not feel like very bloated and very long page, ability to hide/show should be there. Think about the UI/UX". This keeps the current visual identity and restructures for scale:
1. **Collapsible steps.** Each step has a show/hide chevron. A completed step folds into a one-line summary (e.g. "Size: 8K · Quality: Photo-real Balanced · Saved as JPEG, 300 DPI") with a "Change" link. Collapsed state is remembered per browser. *Changed during the build:* after the first batch is sent, the settings steps (size, quality, how it's saved) that nobody folded or opened by hand fold into their summaries.
2. **Compact picture tray** instead of large stacked cards:
   - A summary line: count, how many need attention, total time, total size.
   - A **Grid / List** toggle (small thumbnails, or dense rows: thumbnail, name, size → target, mode chip, time, status).
   - **Show all / Show less** (planned as "Show all / Hide"), which collapses the tray to a single horizontal filmstrip.
   - Lazy-loaded thumbnails and incremental rendering ("Show 48 more", or IntersectionObserver), so 200+ pictures stay fast.
   - *Changed during the build:* on phones, grid tiles show each picture's quality as its colour and icon only (the words are in the list view, the filters and the side panel, and screen readers hear them). An SVG shows "Vector" whatever quality is picked, because it's drawn at the new size without AI.
3. **Per-picture side panel** (drawer): suggestion with its "why", Preview styles, a per-picture mode override (badge on the item), remove. Nothing expands inline.
4. **Selection and bulk actions:**
   - Checkboxes with shift-click range select.
   - "Use suggestions for all" and "Remove selected".
   - Filter chips: All, Needs attention, Photo-real, Faithful, Has people.
5. **Sticky bottom action bar:** count, total time, total size, the Upscale button. It respects safe areas on phones.
6. **Results grouped by batch:**
   - Each batch is a card with overall progress ("7 of 24 done · about 38 min left"), a mini thumbnail grid, "Download all (zip)" and "Clear finished".
   - Finished batches auto-collapse. *Changed during the build:* except the newest one, and one that made nothing because it failed or a restart stopped it, so its reason isn't folded away.
   - *Changed during the build:* "Clear finished" takes a batch off the list in this browser only. The host's pictures stay in Pictures › Upscaled; a guest is asked first, because the list is their only way to download theirs. "Clear all finished" in the results header clears every finished batch and video at once, and "Clear from list" clears a single finished video or stopped job.
   - *Changed during the build:* a batch that needs several qualities is sent as one job per quality (and per 100 pictures), and each job downloads as its own zip, so "Download all (zip)" saves one zip per job.
   - A result opens a full-screen compare viewer with Before/After, zoom and pan, arrow keys for next and previous, and Esc to close, instead of an inline slider per result.
7. **Calm uploads:** "Checking 37 of 120…" with a progress bar. Inspect requests run a few at a time (concurrency 3), and failures appear as "Needs attention" items, not alerts.

### Visual identity: "Earth & Light" palette and fonts (user request, 2026-10-18)
**Chosen by the user: Option A, "Earth & Light".** A second option, "Cocoa & Powder" (Truffle #4B2F24, Pearl #EFEADC, Cocoa #7F5B3B, Powder #8EA7C1), was also prepared and contrast-checked, but the user preferred Earth & Light.
The user supplied a reference palette (Mother Earth #513229, Lucky Dice #F4F1E2, Something Blue #D8EBF9, The Bay #FCE6B7, Walking Vinnie #D7D4B1) and asked for a palette with defined roles, plus two free Google Fonts for headlines and body copy. Every pair below was contrast-checked (WCAG AA; see the check in the session log).

| Token | Light | Dark | Use |
|---|---|---|---|
| `--primary` / `--on-primary` | #513229 / #F4F1E2 | #F4F1E2 / #513229 | Main buttons, brand mark, headings |
| `--primary-hover` | #3D251E | #FFFFFF | Hover state of main buttons |
| `--page` | #F4F1E2 | #1C1512 | Page background |
| `--surface` | #FBF9F1 | #261D19 | Cards, drawer, dialogs |
| `--surface-raised` | #FFFDF8 | #30251F | Popovers |
| `--selected` | #D7D4B1 | #4A3C33 | Selected chips, hover fills |
| `--border` / `--border-strong` | #CFC8B0 / #B9B096 | #4A3C33 / #5E4E43 | Hairlines and dividers |
| `--text` | #513229 | #F4F1E2 | Body text |
| `--text-2` | #6B5548 | #CDBFAD | Secondary text |
| `--text-muted` | #76635A | #A9998A | Muted text, ≥4.5:1 on page and cards |
| `--accent` / `--accent-tint` | #A86F1A / #FCE6B7 | #F2C66D / #3D2E14 | Focus ring, progress, "Recommended", highlights |
| `--info` / tint | #2F5F86 / #D8EBF9 | #A3C9EC / #1D3245 | Info and tips |
| `--success` / tint | #4E6B3A / #E3EBD3 | #A9C98F / #24331C | Done states |
| `--warning` / tint | #8A5A12 / #FCE6B7 | #F2C66D / #3D2E14 | Warnings |
| `--danger` / tint | #A23B2A / #F6DDD6 | #EE9B8A / #45201A | Problems and errors |
| Mode Photo-real (Plum Wine) | #6B4C7A / #ECE3EF | #D2B3DF / #3A2C40 | |
| Mode Faithful (Pine) | #2F6B5E / #DDEBE5 | #93CDBC / #1E3A33 | |
| Mode Fast (Honey Jar) | #8A5A12 / #FCE6B7 | #F2C66D / #3D2E14 | |
| Mode Illustration (Rose Clay) | #A3455E / #F5E0E5 | #EBA3B6 / #41232B | |
| Mode Cloud (Lake) | #2F5F86 / #D8EBF9 | #A3C9EC / #1D3245 | |
| Mode Video (Dusk) | #4A5687 / #E2E5F2 | #B0B9E4 / #272C46 | |
| Status queued (Stone) | #6E6A55 / #ECEADD | #C3BEA6 / #34302A | |

- **Headline font: Noto Serif** (Google Fonts, SIL Open Font License 1.1), chosen by the user. It replaces the first proposal, Fraunces. Variable weight; use 500–600 for headlines.
- **Body font: Inter** (Google Fonts, SIL OFL 1.1), chosen by the user. It replaces the first proposal, Figtree. Use weights 400–600, with `tabular-nums` for times and sizes.
- **Self-hosted:** the latin WOFF2 subsets of both fonts are bundled in `upscaler/web/fonts/` with their OFL licence files, so the app makes no requests to Google and works offline and on the LAN. Fallbacks: `Georgia, serif` and the system UI sans.

### Deep-zoom compare viewer (user bug report, 2026-10-18)
"when I am using Fit, it does not fit/show the complete image … and accordingly for 100% and 200% as well, make it really nice and interactive and user friendly". The old viewer only had 640 px crops. The new one:
- **Tile pyramid per result:** 512 px JPEG tiles, written at the end of each job while the picture is in memory, plus a `viewer.json`. The before side is generated on demand: the original's region, enlarged with Lanczos.
- **Fit** shows the whole picture. **100%** means actual pixels (devicePixelRatio-aware). **200%** is also available, along with smooth free zoom around the cursor.
- **Ways to move around:** pinch, drag to pan, double-click to toggle Fit ↔ 100%, keys `+ − 0 1 2`, arrows, `[ ]` and Esc. The arrows pan while zoomed in and otherwise move to the previous or next picture; `[ ]` (and Shift with an arrow) always move.
- **Compare modes:** a slider (draggable, click to move, keyboard-accessible) or synchronised side by side.
- **Also:** a minimap with a draggable viewport, a blurred preview while tiles load, and honest zoom labels.

### Side-by-side video compare (user request, 2026-10-18)
"for videos, let the user see the comparison of old vs new upscaled video side by side but do not use slider for videos, let those two be side by side in their own frame but matched to timelines (playback)".
- **Layout:** two players, each in its own labelled frame (stacked on phones), with no slider.
- **One control bar** drives both: play/pause, a shared timeline, frame steps, speed, loop, and mute. Audio plays from the upscaled side only.
- **Zoom:** synchronised Fit / 100% / 200%, with a linked pan.
- **Sync:** the upscaled player is the master. Drift of more than half a frame is corrected (gently by speed when it's small, by one jump when it's over half a second) until it's within a quarter of a frame, and both pause while either buffers. *Changed during the build: the first plan allowed up to a whole frame.*
- **Comparison copies:** browser-safe H.264 copies of the original and the result are made on demand (hardware encoder where available), cached, and shown with progress.
- **Originals:** the job's original input is kept until its results expire, for both the image before-side and the video compare: 7 days for the host's jobs and 24 hours for guests', in the job's folder in the data folder, with the tile pyramid and the comparison copies. A very large video original is swapped for its smaller comparison copy when the job finishes.
- *Added during the build:* a finished video plays in its own card (a still from it first, then the browser-safe copy when played).
