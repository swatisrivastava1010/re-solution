# Third-party licences

Re:Solution is built on other people's free work. This page lists every AI model and library it uses and every sample picture it includes, who made them, and their licences.

**What's in this repository:** the app's own code (MIT, see [LICENSE](LICENSE)), the five sample pictures in `upscaler/web/samples/` (public domain or CC0, [listed below](#sample-pictures)), the two fonts in `upscaler/web/fonts/` (SIL Open Font License 1.1, [listed below](#fonts)), and the screenshots in `docs/images/`, which show only those samples.

**What isn't:** no model or library files. The installer (or the Start Re-Solution launcher) downloads the libraries the first time: PyTorch from download.pytorch.org and everything else from the Python Package Index (PyPI). The app downloads each model from the source listed below the first time it's needed, and checks every model file, and the NVIDIA Photo-real runner, against known SHA-256 hashes.

Every model Re:Solution offers allows commercial use. None of these licences claims any rights over the pictures and videos you upscale.

## AI models

| Used for | Model (file) | Author | Licence | Source |
|---|---|---|---|---|
| Photo-real (images, default) | SeedVR2-3B | ByteDance Seed team (Jianyi Wang, Shanchuan Lin, Zhijie Lin, Yuxi Ren, Meng Wei, Zongsheng Yue, Shangchen Zhou, Hao Chen, Yang Zhao, Ceyuan Yang, Xuefeng Xiao, Chen Change Loy, Lu Jiang) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [github.com/ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR), [huggingface.co/ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) |
| Photo-real model files | SeedVR2 3B fp16 weights and VAE (`seedvr2_ema_3b_fp16.safetensors`, `ema_vae_fp16.safetensors`), repackaged from ByteDance's release | NumZ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [huggingface.co/numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) |
| Faithful (images), Best (video) | 4xBHI_dat2_real (`4xBHI_dat2_real.safetensors`) | Philip Hofmann (Phhofm) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | [github.com/Phhofm/models, release 4xBHI_dat2_real](https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real) |
| Faithful for sharp photos (images) | 4xNomos2_hq_dat2 (`4xNomos2_hq_dat2.pth`) | Philip Hofmann (Phhofm) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (the release page reads "CC-BY-0.4", a typo; [OpenModelDB](https://openmodeldb.info/models/4x-Nomos2-hq-dat2) lists CC-BY-4.0) | [github.com/Phhofm/models, release 4xNomos2_hq_dat2](https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_dat2) |
| Fast (images and video), and the second pass for 16K and Photo-real Balanced | 4x-ClearRealityV1 (`4x-ClearRealityV1.safetensors`) | Kim2091 | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [huggingface.co/Kim2091/ClearRealityV1](https://huggingface.co/Kim2091/ClearRealityV1) |
| Illustration (images) | RealESRGAN_x4plus_anime_6B (`RealESRGAN_x4plus_anime_6B.pth`) | Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| Balanced (video) | realesr-general-x4v3 (`realesr-general-x4v3.pth`) | Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan (Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| Spotting faces, so auto-suggest can pick a style (not an upscaler) | YuNet (`face_detection_yunet_2023mar.onnx`, pinned to opencv_zoo commit `f12e127`) | Shiqi Yu (YuNet), OpenCV Model Zoo | [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) | [github.com/opencv/opencv_zoo, face_detection_yunet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |

### Credit required by CC-BY-4.0

CC-BY-4.0 lets anyone use a model, including commercially, as long as they credit the author. Re:Solution gives this credit here, in the README and on the app's About page:

> **4xBHI_dat2_real** and **4xNomos2_hq_dat2** by Philip Hofmann ([github.com/Phhofm/models](https://github.com/Phhofm/models)), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Re:Solution uses the model files unchanged.

If you share pictures made with Faithful mode, you don't have to credit anyone: the licence covers the model, not your images.

### Model code

[spandrel](https://github.com/chaiNNer-org/spandrel) supplies the code that runs the upscaling models other than Photo-real (the DAT, SPAN, RRDB and SRVGGNetCompact network designs). Its main package only includes architectures under permissive or public-domain licences (MIT, Apache-2.0, public domain), and each architecture keeps its original licence. YuNet runs through OpenCV (below).

## Photo-real engines

| Engine | Used on | Author | Licence | Source |
|---|---|---|---|---|
| mflux 0.23 | Apple Silicon Macs | Filip Strand and the mflux community | [MIT](https://github.com/mflux-community/mflux/blob/main/LICENSE) | [github.com/mflux-community/mflux](https://github.com/mflux-community/mflux) |
| MLX and mlx-metal | Apple Silicon Macs | Apple (ml-explore) | [MIT](https://github.com/ml-explore/mlx/blob/main/LICENSE) | [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) |
| ComfyUI-SeedVR2_VideoUpscaler (its standalone command-line tool, downloaded on first use with its own packages into a separate folder) | NVIDIA PCs | NumZ and AInVFX (Adrien Toupet) | [Apache-2.0](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/blob/main/LICENSE) | [github.com/numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) |

## Main libraries

| Library | What it does in Re:Solution | Author | Licence | Source |
|---|---|---|---|---|
| uv | Installs Python and the libraries below (the installer and launchers install uv itself) | Astral | MIT or Apache-2.0 | [github.com/astral-sh/uv](https://github.com/astral-sh/uv) |
| Python 3.12 | Runs the app | Python Software Foundation | [PSF-2.0](https://docs.python.org/3/license.html) | [python.org](https://www.python.org) |
| PyTorch (`torch`) | Runs the AI models on the Apple or NVIDIA graphics chip | The PyTorch Foundation and contributors | BSD-3-Clause, plus bundled parts under Apache-2.0, MIT, BSD-2-Clause and BSL-1.0 | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) |
| torchvision | Image helpers for PyTorch | The PyTorch Foundation and contributors | BSD-3-Clause | [github.com/pytorch/vision](https://github.com/pytorch/vision) |
| NVIDIA CUDA libraries (inside the Windows PyTorch build for NVIDIA cards) | Let PyTorch use NVIDIA graphics cards | NVIDIA | [NVIDIA CUDA licence](https://docs.nvidia.com/cuda/eula/index.html) | [pytorch.org](https://pytorch.org) |
| spandrel | Loads the upscaling models | chaiNNer team | MIT | [github.com/chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) |
| Hugging Face Hub (`huggingface_hub`), safetensors, transformers, tokenizers, hf-transfer, hf-xet | Download and read the Photo-real model | Hugging Face | Apache-2.0 | [github.com/huggingface](https://github.com/huggingface) |
| NumPy | Pixel maths | NumPy developers | BSD-3-Clause, plus bundled parts under 0BSD, MIT, Zlib and CC0-1.0 | [numpy.org](https://numpy.org) |
| Pillow | Opens and saves most picture formats | Jeffrey A. Clark and contributors | MIT-CMU | [github.com/python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| pillow-heif | Opens and saves iPhone HEIC photos | Alexander Piskun | BSD-3-Clause for its own code; its download bundles libheif and libde265 (LGPL-3.0) and x265 (GPL-2.0), so the bundle as a whole is GPL-2.0 | [github.com/bigcat88/pillow_heif](https://github.com/bigcat88/pillow_heif) |
| rawpy | Opens camera RAW photos | Maik Riechert | [MIT](https://github.com/letmaik/rawpy/blob/main/LICENSE) for its own code; it bundles LibRaw (below), and on a Mac also Little CMS (MIT), libjpeg (IJG and BSD-style licences) and JasPer (JasPer License 2.0, MIT-style) | [github.com/letmaik/rawpy](https://github.com/letmaik/rawpy) |
| LibRaw | Develops the RAW data (inside rawpy) | LibRaw LLC | LGPL-2.1 or CDDL-1.0, at the user's choice | [libraw.org](https://www.libraw.org) |
| resvg-py | Draws SVG pictures straight at the full size | baseplate-admin | MIT | [github.com/baseplate-admin/resvg-py](https://github.com/baseplate-admin/resvg-py) |
| resvg (inside resvg-py, version 0.48) | The SVG renderer itself | Yevhenii Reizner and the Linebender contributors | Apache-2.0 or MIT (releases before 0.45, from February 2025, were MPL-2.0); its parts include tiny-skia (BSD-3-Clause) and Rust libraries under MIT, Apache-2.0, Zlib, BSD-2-Clause, Unlicense and Unicode-3.0, as listed in the package's bill of materials | [github.com/linebender/resvg](https://github.com/linebender/resvg) |
| OpenCV (`opencv-python`) | Looks at each picture for auto-suggest (faces with YuNet, text, sharpness) and finds the most detailed spot for Preview styles; mflux uses it too | OpenCV team | Apache-2.0 for OpenCV; the download bundles FFmpeg, Qt 5 and other libraries under LGPL-2.1 and permissive licences (listed in its `LICENSE-3RD-PARTY.txt`), and the macOS download also contains x264 and x265 (GPL-2.0) | [github.com/opencv/opencv-python](https://github.com/opencv/opencv-python) |
| segno | Draws the QR code in the Share on Wi-Fi panel | Lars Heuer | BSD-3-Clause | [github.com/heuer/segno](https://github.com/heuer/segno) |
| PyAV (`av`) | Reads video details | PyAV contributors (WyattBlue, Jeremy Lainé and others) | BSD-3-Clause for its own code; it bundles FFmpeg's libraries, built as LGPL-3.0-or-later (`--enable-version3`), together with the x264 and x265 libraries (GPL-2.0-or-later), so treat the bundle as GPL | [github.com/PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) |
| imageio-ffmpeg | Provides the `ffmpeg` program that decodes and encodes video, and reads EXR, HDR, DDS and QOI pictures | imageio contributors | BSD-2-Clause for its own code; the bundled `ffmpeg` program is a GPL build: GPL-2.0-or-later on a Mac (`--enable-gpl`), GPL-3.0-or-later on Windows (a gyan.dev build with `--enable-gpl --enable-version3`) | [github.com/imageio/imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| FFmpeg | Video decoding, encoding and audio copying (inside PyAV, imageio-ffmpeg and OpenCV) | The FFmpeg developers | LGPL-2.1-or-later, but a build that includes FFmpeg's optional GPL parts (such as x264 and x265) is GPL as a whole, as [FFmpeg's legal page](https://ffmpeg.org/legal.html) explains. As built here: imageio-ffmpeg's `ffmpeg` program is GPL-2.0-or-later on a Mac and GPL-3.0-or-later on Windows; PyAV's libraries are LGPL-3.0-or-later but come with the GPL x264 and x265; OpenCV's are LGPL-2.1 | [ffmpeg.org/legal.html](https://ffmpeg.org/legal.html) |
| x264, x265 | Video codecs compiled into the FFmpeg builds above | VideoLAN; MulticoreWare | GPL-2.0 | [videolan.org/developers/x264.html](https://www.videolan.org/developers/x264.html), [github.com/Multicorewareinc/x265](https://github.com/Multicorewareinc/x265) |
| FastAPI | The app's web server | Sebastián Ramírez | MIT | [github.com/fastapi/fastapi](https://github.com/fastapi/fastapi) |
| Starlette | Web toolkit under FastAPI | Encode and contributors | BSD-3-Clause | [github.com/Kludex/starlette](https://github.com/Kludex/starlette) |
| Uvicorn | Serves the app to your browser | Encode and contributors | BSD-3-Clause | [github.com/Kludex/uvicorn](https://github.com/Kludex/uvicorn) |
| Pydantic | Checks the data the browser sends | Samuel Colvin and contributors | MIT | [github.com/pydantic/pydantic](https://github.com/pydantic/pydantic) |
| python-multipart | Receives uploaded files | Andrew Dunham and contributors | Apache-2.0 | [github.com/Kludex/python-multipart](https://github.com/Kludex/python-multipart) |
| HTTPX | Downloads models and talks to cloud services | Encode and contributors | BSD-3-Clause | [github.com/encode/httpx](https://github.com/encode/httpx) |
| keyring | Keeps cloud keys in the macOS Keychain or Windows Credential Manager | Jason R. Coombs and contributors | MIT | [github.com/jaraco/keyring](https://github.com/jaraco/keyring) |
| platformdirs | Finds the right folder for settings and models on each system | tox-dev contributors | MIT | [github.com/tox-dev/platformdirs](https://github.com/tox-dev/platformdirs) |
| Matplotlib | Pulled in by mflux (Macs only) | Matplotlib development team | [Matplotlib licence](https://matplotlib.org/stable/project/license.html) (PSF-based) | [matplotlib.org](https://matplotlib.org) |

### Smaller supporting packages

These come along with the libraries above. All of them allow commercial use.

- **MIT:** annotated-doc, annotated-types, anyio, charset-normalizer, einops, filelock, fonttools, h11, httptools, jaraco.classes, jaraco.context, jaraco.functools, markdown-it-py, mdurl, more-itertools, piexif, pydantic-core, pyparsing, pyyaml, rich, setuptools, six, toml, typer, typing-inspection, urllib3, watchfiles
- **BSD:** click, colorama (Windows), contourpy, cycler, fsspec, httpcore, idna, jinja2, kiwisolver, markupsafe, mpmath, networkx, protobuf, pygments, python-dotenv, pywin32-ctypes (Windows), sympy, websockets
- **Apache-2.0:** opentelemetry-api, requests, sentencepiece
- **Dual or mixed permissive:** packaging (Apache-2.0 or BSD-2-Clause), python-dateutil (Apache-2.0 and BSD-3-Clause), regex (Apache-2.0 and CNRI-Python), uvloop (MIT and Apache-2.0, Macs only), typing-extensions (PSF-2.0)
- **ISC:** shellingham (Macs only)
- **MPL-2.0:** certifi, and tqdm (MPL-2.0 and MIT). Re:Solution uses both unchanged.
- **Linux only** (the Mac installer also runs on Linux, untested): cffi (MIT-0), cryptography (Apache-2.0 or BSD-3-Clause), jeepney (MIT), pycparser (BSD-3-Clause), SecretStorage (BSD-3-Clause) for the Linux keychain; and for NVIDIA cards, triton (MIT), cuda-bindings and cuda-pathfinder (Apache-2.0), and NVIDIA's CUDA packages (cuda-toolkit and the `nvidia-*` packages, under NVIDIA's own licences).

The test tools pytest (MIT), its helpers iniconfig and pluggy (MIT), and respx (BSD-3-Clause) are only for development. The installer and launchers don't install them.

## Fonts

The page's two fonts come with Re:Solution in `upscaler/web/fonts/`, so it never fetches them from Google or anyone else. Each is the Latin, variable-weight WOFF2 file from Google Fonts (as packaged by [Fontsource](https://fontsource.org), version 5.3.0), unchanged, next to its licence text (`noto-serif-OFL.txt`, `inter-OFL.txt`, copied from the [google/fonts](https://github.com/google/fonts) repository). The SIL Open Font License lets anyone use, share and bundle the fonts, including commercially, as long as the licence travels with them and they aren't sold on their own.

| Font (file) | Used for | Author | Licence | Source |
|---|---|---|---|---|
| Noto Serif (`noto-serif-latin-wght-normal.woff2`) | Headings | Google, the Noto project (The Noto Project Authors) | [OFL-1.1](https://openfontlicense.org) | [fonts.google.com/specimen/Noto+Serif](https://fonts.google.com/specimen/Noto+Serif), [github.com/notofonts/latin-greek-cyrillic](https://github.com/notofonts/latin-greek-cyrillic) |
| Inter (`inter-latin-wght-normal.woff2`) | Everything else | Rasmus Andersson (The Inter Project Authors) | [OFL-1.1](https://openfontlicense.org) | [fonts.google.com/specimen/Inter](https://fonts.google.com/specimen/Inter), [github.com/rsms/inter](https://github.com/rsms/inter) |

## Sample pictures

The "Try a sample" pictures in `upscaler/web/samples/` are public domain or CC0, so anyone may share them, including commercially, without permission. Credit isn't legally required; Re:Solution gives it as a courtesy, here, in the README and on the app's About page. Each was downscaled to 1,024 pixels on the long side and saved as JPEG, with a 240-pixel `-thumb.jpg` copy. [CREDITS.md](upscaler/web/samples/CREDITS.md) has the full titles and the licence notes for each.

| File | Picture | Author / institution | Licence | Source |
|---|---|---|---|---|
| `interior.jpg` | Interior of the Farnsworth House (Ludwig Mies van der Rohe, 1951), Plano, Illinois | Carol M. Highsmith; Library of Congress, Carol M. Highsmith Archive | Public domain | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Interior_views_of_the_1951_Farnsworth,_a_modernist_classic_%22glass_house%22_designed_by_architect_Mies_van_der_Rohe,_Plano,_Illinois_LCCN2011633480.tif) |
| `product.jpg` | Teapot, ca. 1879 | Christopher Dresser (designer), James Dixon & Sons (maker); The Metropolitan Museum of Art | CC0 (Met Open Access) | [The Met collection](https://www.metmuseum.org/art/collection/search/823191) |
| `old-photo.jpg` | Power house mechanic working on steam pump, 1920 | Lewis Wickes Hine; U.S. National Archives | Public domain | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lewis_Hine_Power_house_mechanic_working_on_steam_pump.jpg) |
| `food.jpg` | Assortment of fruits, vegetables, nuts and grains (USDA ARS image K3839-3) | Keith Weller, USDA Agricultural Research Service | Public domain (U.S. federal government work) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Foods_(cropped).jpg) |
| `illustration.jpg` | Under the Wave off Kanagawa (The Great Wave), ca. 1830–32 | Katsushika Hokusai; The Metropolitan Museum of Art, H. O. Havemeyer Collection | CC0 (Met Open Access) | [The Met collection](https://www.metmuseum.org/art/collection/search/45434) |

## About the GPL parts

Some downloaded libraries contain code under the GPL: the `ffmpeg` program inside imageio-ffmpeg (GPL-2.0-or-later on a Mac, GPL-3.0-or-later on Windows, because those builds include GPL parts; see [ffmpeg.org/legal.html](https://ffmpeg.org/legal.html)), the x264 and x265 libraries that come with PyAV's FFmpeg, the x265 library inside pillow-heif, and the x264 and x265 libraries in OpenCV's macOS download. Others are under the LGPL: PyAV's FFmpeg libraries (LGPL-3.0-or-later), LibRaw (inside rawpy; also available under CDDL-1.0), libheif and libde265 (inside pillow-heif), and OpenCV's FFmpeg and Qt libraries.

Running them on your own computer is fine for any purpose, including business use. These licences only add duties for someone who passes those program files on to others: they must also offer the matching source code. This repository doesn't contain these files. Each person's copy of uv downloads them straight from PyPI, and the source links above point to their code.

## Cloud services

If you connect a cloud account (fal.ai, Replicate or Magnific), that service's own terms and prices apply to the pictures you send it. Those services aren't open-source software, so they have no entry here. The in-app guide explains what each one costs and how long it keeps your pictures.

## Full licence texts

Each library keeps its full licence text in its installed folder. After the first start, look in `.venv/lib/python3.12/site-packages/<library>-<version>.dist-info/` on a Mac, or `.venv\Lib\site-packages\<library>-<version>.dist-info\` on Windows.
