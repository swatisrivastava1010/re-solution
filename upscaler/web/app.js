/* Re:Solution web UI.
   A plain browser script (no build step, no modules, so it works however the
   server labels .js files). It talks only to this app's own /api endpoints.
   The pure helpers at the top are also exported for tests run under Node.
   js/compare.js (before/after with zoom) and js/styles.js (Preview styles)
   load first and are handed this file's DOM helpers. */
(function () {
  'use strict';

  /* ---------- Choices and limits ---------- */

  const DEFAULTS = Object.freeze({
    imageMode: 'photoreal', // Photo-real (SeedVR2) is the default for images.
    photorealLevel: 'balanced', // SeedVR2 2x, then the Fast model to the target.
    size: '8k',
    videoMode: 'fast',
    strength: 100,
    creativity: 0,
    stayClose: 0, // 0% = Photo-real's full result (resemblance 1.0)
    grain: 'off',
    format: 'jpeg',
    dpi: 300,
  });

  const MAX_IMAGE_BYTES = 300 * 1024 * 1024;
  const PHOTO_TYPES_HINT = 'JPEG, PNG, HEIC, TIFF, camera RAW and more, up to 300 MB each.';
  const PREVIEW_SECONDS = 3;
  const IMAGE_UPLOAD_FIELD = 'files';
  const VIDEO_UPLOAD_FIELD = 'file';
  const INSPECT_FIELD = 'file';
  const INSPECT_AT_ONCE = 3; // pictures checked at the same time (DESIGN "Calm uploads")
  const ESTIMATES_AT_ONCE = 4;
  const TRAY_PAGE = 48; // pictures drawn at a time in the tray; "Show 48 more" adds the next ones
  const TRAY_FILTERS = ['all', 'attention', 'photoreal', 'faithful', 'people'];
  const FILTER_LABELS = { all: 'All', attention: 'Needs attention', photoreal: 'Photo-real', faithful: 'Faithful', people: 'Has people' };
  const PEOPLE_WORDS = /\b(faces?|people|person|portraits?)\b/i;
  const STEPS_KEY = 'resolution.steps'; // which steps this browser folded
  const TRAY_KEY = 'resolution.tray'; // grid or list, and whether the tray is folded to a strip
  const CLEARED_KEY = 'resolution.cleared'; // finished jobs this browser cleared from its results
  const SETTINGS_STEPS = ['photo-step-size', 'photo-step-mode', 'photo-step-file'];
  const EIGHT_K_PIXELS = 8192 * 6144; // cloud prices are quoted for an 8K picture (4:3)
  const MIN_PASSWORD = 6;
  const WINDOWS_HEVC_NOTE = "If Windows can't play it, install the free HEVC Video Extensions from the Microsoft Store.";
  const MAX_PICTURES_PER_JOB = 100; // the server takes at most this many pictures in one job
  // The server keeps checked pictures for 24 hours; older ones are sent again as files.
  const TOKEN_MAX_AGE_MS = 23 * 3600 * 1000;
  const STAGED_GONE = /isn.t here any more/i;
  const OWN_KEYS_COOKIE = 'resolution_own_keys'; // cloud services this browser's guest added a key for
  const JOBS_SHOWN = 12;
  const MY_JOBS_KEY = 'upscaler.myJobs';
  const LONG_EDGE = { '4k': 3840, '6k': 6144, '8k': 8192, '16k': 16384 };
  const TV_BOX = { '8k_tv': [7680, 4320], '16k_tv': [15360, 8640] }; // landscape; portrait photos turn it
  const SIXTEEN_K = new Set(['16k', '16k_tv']);
  const VIDEO_LONG_EDGE = 3840;
  const MORE_MODES = new Set(['faithful_sharp']);
  const LEVEL_IDS = ['balanced', 'rich', 'gentle'];
  const GRAINS = ['off', 'subtle', 'medium'];
  const DPI_CHOICES = [72, 150, 300, 600];
  const NO_CHANGE_LABEL = 'No change needed';
  const VECTOR = 'vector'; // an SVG drawn straight at the new size (pipeline_image), so no AI pass
  const VECTOR_LABEL = 'Drawn directly (vector)';
  const VECTOR_NOTE = "It's a vector drawing, so it's drawn sharp at the new size. No AI is needed.";
  const DEFAULT_PREVIEW_ROUTES = ['photoreal:balanced', 'photoreal:rich', 'photoreal:gentle', 'faithful', 'fast'];
  const PREVIEW_EXTRA_MODES = ['faithful_sharp', 'illustration']; // also previewable when a picture is suggested or set to one

  const DEFAULT_FORMATS = [
    { id: 'jpeg', max_long_edge: 65535 }, { id: 'png' }, { id: 'tiff' }, { id: 'webp', max_long_edge: 16383 },
    { id: 'original' },
  ];
  // When the chosen file type can't hold the size: the first of these that can.
  const FALLBACK_FORMATS = ['jpeg', 'png', 'tiff'];
  const FORMAT_LABELS = {
    jpeg: 'JPEG', png: 'PNG', tiff: 'TIFF', webp: 'WebP', heic: 'HEIC', avif: 'AVIF', original: 'Same as original',
  };
  // How a result is really saved, mirroring imageio_utils.resolve_format and pipeline_image._output_format:
  // the source types "Same as original" can write back, which types keep see-through areas, and their limits.
  const ORIGINAL_FORMATS = {
    jpeg: 'jpeg', jpg: 'jpeg', mpo: 'jpeg', png: 'png', tiff: 'tiff', tif: 'tiff',
    webp: 'webp', heic: 'heic', heif: 'heic', avif: 'avif',
  };
  const FORMATS_WITH_ALPHA = ['png', 'tiff', 'webp', 'heic', 'avif'];
  const FORMAT_MAX_SIDE = { jpeg: 65535, webp: 16383, avif: 65536 };
  const FORMAT_MAX_PIXELS = { heic: 142606336 };
  const SIZED_LIKE_JPEG = ['heic', 'avif']; // /api/estimate takes jpeg/png/tiff/webp; these come out about JPEG-sized
  const FORMAT_HINTS = {
    jpeg: 'Small files that open anywhere. Right for sharing and most printing.',
    png: 'Keeps every pixel exactly, and see-through areas. Bigger files.',
    tiff: 'For print shops and design apps. The biggest files.',
    webp: 'Small files for websites, up to 16,383 pixels on the long side.',
    original: "The same type of file you added. If that type can't be saved, you get a PNG.",
  };
  // Typical file size per output pixel (sizing.py; WebP is a rough guess for the fallback readout).
  const BYTES_PER_PIXEL = { jpeg: 0.2, png: 1.6, tiff: 1.8, webp: 0.15 };
  const ALPHA_SIZE_FACTOR = 4 / 3;
  const VERY_LARGE_PIXELS = 100000000;

  const FALLBACK_ACCEPT = {
    images: ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif', '.tif', '.tiff', '.bmp', '.gif',
      '.jp2', '.ico', '.tga', '.ppm', '.psd', '.svg', '.exr', '.hdr', '.dds', '.qoi', '.jxl',
      '.dng', '.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.pef', '.srw'],
    videos: ['.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm', '.3gp', '.mts', '.m2ts', '.wmv', '.mpg', '.mpeg'],
  };

  const PHOTOREAL_LEVELS = [
    {
      id: 'balanced', label: 'Balanced',
      description: 'Natural new detail with crisp edges. Right for most pictures.',
      eta_hint: 'About 3–6 minutes per picture',
    },
    {
      id: 'rich', label: 'Rich',
      description: 'The most new detail. Can over-texture food, plants and skin.',
      eta_hint: 'About 10–22 minutes per picture',
    },
    {
      id: 'gentle', label: 'Gentle',
      description: 'The softest and most literal. Adds the least new detail.',
      eta_hint: 'About 3–5 minutes per picture',
    },
  ];

  const enc = encodeURIComponent;
  const API = {
    status: '/api/status',
    login: '/api/login',
    images: '/api/images',
    imagesSample: '/api/images/sample',
    inspect: '/api/inspect',
    thumb: (token) => `/api/staging/${enc(token)}/thumb.jpg`,
    estimate: '/api/estimate',
    previewStyles: '/api/preview',
    videos: '/api/videos',
    jobs: '/api/jobs',
    job: (id) => `/api/jobs/${enc(id)}`,
    cancel: (id) => `/api/jobs/${enc(id)}/cancel`,
    retry: (id) => `/api/jobs/${enc(id)}/retry`,
    file: (id, n) => `/api/jobs/${enc(id)}/results/${n}/file`,
    preview: (id, n) => `/api/jobs/${enc(id)}/results/${n}/preview.jpg`,
    before: (id, n) => `/api/jobs/${enc(id)}/results/${n}/before.jpg`,
    after: (id, n) => `/api/jobs/${enc(id)}/results/${n}/after.jpg`,
    view: (id, n) => `/api/jobs/${enc(id)}/results/${n}/view`,
    // The compare viewer: the result's tile pyramid, its tiles, and the original's (made on request).
    viewerInfo: (id, n) => `/api/jobs/${enc(id)}/results/${n}/viewer.json`,
    tile: (id, n, level, col, row) => `/api/jobs/${enc(id)}/results/${n}/tiles/${level}/${col}_${row}.jpg`,
    beforeTile: (id, n, level, col, row) => `/api/jobs/${enc(id)}/results/${n}/before/${level}/${col}_${row}.jpg`,
    // Side-by-side video comparison: copies any browser plays, made the first time it opens.
    compare: (id, n) => `/api/jobs/${enc(id)}/results/${n}/compare`,
    compareStatus: (id, n) => `/api/jobs/${enc(id)}/results/${n}/compare/status`,
    compareCancel: (id, n) => `/api/jobs/${enc(id)}/results/${n}/compare/cancel`,
    // A still from a finished video, for its card (the video itself plays from the comparison's copy).
    poster: (id, n) => `/api/jobs/${enc(id)}/results/${n}/poster.jpg`,
    zip: (id) => `/api/jobs/${enc(id)}/zip`,
    events: '/api/events',
    photorealDownload: '/api/photoreal/download',
    guide: (id) => `/api/providers/${enc(id)}/guide`,
    providerKey: (id) => `/api/providers/${enc(id)}/key`,
    sessionKey: (id) => `/api/providers/${enc(id)}/session-key`,
    settings: '/api/settings',
    openFolder: '/api/open-folder',
    share: '/api/share',
    shareQr: (url) => `/api/share/qr.svg?url=${enc(url)}`,
  };

  // The header's theme button cycles Light -> Dark -> Auto. The head script in index.html reads the same key.
  const THEME_KEY = 'resolution.theme';
  const THEME_ORDER = ['light', 'dark', 'auto'];
  const THEME_NAMES = { light: 'Light', dark: 'Dark', auto: 'Auto (follows your computer)' };
  const THEME_ICONS = { light: 'sun', dark: 'moon', auto: 'monitor' };
  const THEME_TOASTS = { light: 'Light theme on.', dark: 'Dark theme on.', auto: 'The theme now follows your computer.' };

  const ICONS = {
    photoreal: 'sparkle',
    faithful: 'shield-check',
    faithful_sharp: 'shield-check',
    fast: 'bolt',
    illustration: 'brush',
    general: 'gauge',
    cloud: 'cloud',
    video: 'film',
  };

  // Server preset icon names and preset ids -> icons in icons.svg.
  const PRESET_ICONS = {
    interiors: 'sofa',
    sofa: 'sofa',
    products: 'bag',
    bag: 'bag',
    tag: 'bag',
    portraits: 'user',
    user: 'user',
    ai_art: 'wand',
    wand: 'wand',
    sparkles: 'sparkle',
    sparkle: 'sparkle',
    print: 'printer',
    printer: 'printer',
    illustration: 'brush',
    brush: 'brush',
    old_photos: 'history',
    history: 'history',
    clock: 'history',
    image: 'image',
    photo: 'image',
  };

  const VIDEO_LABELS = { fast: 'Fast', general: 'Balanced', faithful: 'Best' };

  const SIZE_HINTS = {
    '4k': 'The long side becomes 3,840 pixels. Good for screens and 4K TVs.',
    '6k': 'The long side becomes 6,144 pixels. Good for large prints.',
    '8k': 'The long side becomes 8,192 pixels. Good for posters and wall prints.',
    '8k_tv': 'Fits an 8K TV screen (7,680 by 4,320) without cropping.',
    '16k': 'The long side becomes 16,384 pixels, for huge prints. On a screen it looks the same as 8K, and files are very large.',
    '16k_tv': 'Fits a 16K screen (15,360 by 8,640) without cropping. Files are very large.',
    '2x': 'Twice as wide and twice as tall.',
    '3x': 'Three times as wide and three times as tall.',
    '4x': 'Four times as wide and four times as tall.',
  };

  const COPY = {
    photorealWarning: 'Photo-real invents fine detail. Check faces, logos and text, or switch to Faithful.',
    photorealFirstRun: 'The first time, it needs a one-time 7 GB download.',
    videoBestWarning: 'Very slow. Even a short clip can take many hours.',
    cantReach: "Can't reach Re:Solution. Check that it's still running on the main computer, then try again.",
    offline: "Can't reach Re:Solution right now. Check that it's still running on the main computer.",
    generic: "That didn't work. Try again in a moment.",
    hostOnly: 'Only the person on the main computer can change this.',
    serverTrouble: 'Something went wrong on the main computer. Try again in a moment.',
    shareNoNetwork: "Sharing is on, but this computer isn't connected to a network, so there's no address to give out yet. Connect to Wi-Fi and it shows up here.",
    shareRestart: 'To let others at home use it from their phone or computer, quit Re:Solution and start it again by double-clicking "Start Re-Solution (Share on Wi-Fi)" in the app\'s folder.',
  };

  /* ---------- Pure helpers ---------- */

  function roundHalfUp(x) {
    return Math.floor(x + 0.5);
  }

  const round1 = (x) => Math.round(x * 10) / 10;
  const round2 = (x) => Math.round(x * 100) / 100;
  const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  /** Output size for a photo, mirroring upscaler/sizing.py. Never shrinks. */
  function targetSize(w, h, preset) {
    const scale = /^(\d+)x$/.exec(preset);
    let factor;
    if (scale) {
      factor = Number(scale[1]);
    } else if (has(TV_BOX, preset)) {
      const [boxW, boxH] = TV_BOX[preset];
      const [bw, bh] = w >= h ? [boxW, boxH] : [boxH, boxW];
      factor = Math.min(bw / w, bh / h);
    } else if (has(LONG_EDGE, preset)) {
      factor = LONG_EDGE[preset] / Math.max(w, h);
    } else {
      return [w, h];
    }
    if (factor <= 1) return [w, h];
    return [roundHalfUp(w * factor), roundHalfUp(h * factor)];
  }

  /** Approximate 4K video size: 3840 on the long side, even numbers, never smaller. */
  function videoOutputSize(w, h) {
    const even = (x) => Math.max(2, 2 * Math.round(x / 2));
    const long = Math.max(w, h);
    if (long >= VIDEO_LONG_EDGE) return [even(w), even(h)];
    const factor = VIDEO_LONG_EDGE / long;
    return [even(w * factor), even(h * factor)];
  }

  /** "About 4 minutes", "About 1 hour 20 minutes", "Less than a minute". */
  function formatApproxTime(seconds) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '';
    if (seconds < 45) return 'Less than a minute';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return minutes === 1 ? 'About 1 minute' : `About ${minutes} minutes`;
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    const hourText = hours === 1 ? '1 hour' : `${hours} hours`;
    if (rest === 0) return `About ${hourText}`;
    return `About ${hourText} ${rest} minute${rest === 1 ? '' : 's'}`;
  }

  function formatEta(seconds) {
    const text = formatApproxTime(seconds);
    return text ? `${text} left` : '';
  }

  function formatDuration(seconds) {
    if (seconds == null || !Number.isFinite(seconds)) return '';
    const s = Math.max(0, Math.round(seconds));
    if (s < 60) return `${s} s`;
    if (s < 3600) {
      const rest = s % 60;
      return rest ? `${Math.floor(s / 60)} min ${rest} s` : `${s / 60} min`;
    }
    // Round to whole minutes first, so 1 h 59 min 50 s reads "2 h", not "1 h 60 min".
    const totalMinutes = Math.round(s / 60);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return minutes ? `${hours} h ${minutes} min` : `${hours} h`;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    if (bytes < 1024 ** 3) return `${Math.round(bytes / 1024 ** 2)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }

  /** An estimated size in megabytes (decimal, like Finder and sizing.py): "about 40 MB", "about 1.2 GB". */
  function formatFileSize(mb) {
    if (!Number.isFinite(mb) || mb < 0) return '';
    if (mb >= 1000) return `about ${(mb / 1000).toFixed(1)} GB`;
    return `about ${Math.max(1, Math.round(mb))} MB`;
  }

  /** "8K (8192 px)" -> ["8K", "8192 px"]; "2x" -> ["2x", ""]. */
  function splitSizeLabel(label) {
    const match = /^(.*?)\s*\((.+)\)\s*$/.exec(String(label));
    return match ? [match[1], match[2]] : [String(label), ''];
  }

  /** How big a w×h picture prints at `dpi`, like sizing.print_size (to 0.1). */
  function printSize(w, h, dpi) {
    const inches = [w / dpi, h / dpi];
    return { inches: inches.map(round1), cm: inches.map((side) => round1(side * 2.54)), dpi };
  }

  function formatPrint(print) {
    if (!print || !print.inches) return '';
    const [wi, hi] = print.inches;
    const [wc, hc] = print.cm;
    return `${wi.toFixed(1)} × ${hi.toFixed(1)} in (${wc.toFixed(1)} × ${hc.toFixed(1)} cm) at ${print.dpi} DPI`;
  }

  function describeMegabytes(bytes) {
    return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
  }

  /** The readout worked out in the browser, for when /api/estimate isn't available. No time: that needs the server. */
  function localEstimate({ width, height, size, format, dpi = DEFAULTS.dpi, alpha = false }) {
    const out = targetSize(width, height, size);
    const pixels = out[0] * out[1];
    const perPixel = has(BYTES_PER_PIXEL, format) ? BYTES_PER_PIXEL[format] : BYTES_PER_PIXEL.png;
    const bytes = pixels * perPixel * (alpha ? ALPHA_SIZE_FACTOR : 1);
    const warnings = [];
    if (pixels > VERY_LARGE_PIXELS) {
      const what = SIXTEEN_K.has(size) ? '16K files are' : 'Pictures this big are';
      warnings.push(`${what} very large: about ${describeMegabytes(bytes)} as ${FORMAT_LABELS[format] || 'PNG'}. `
        + 'They mainly help big prints; on a screen they look the same as 8K.');
    }
    return {
      out, megapixels: round1(pixels / 1e6), seconds: null, file_mb: round1(bytes / 1e6),
      print: printSize(out[0], out[1], dpi), passes: null, warnings,
    };
  }

  function formatHolds(format, out) {
    if (!out) return true;
    const [w, h] = out;
    const side = FORMAT_MAX_SIDE[format];
    const pixels = FORMAT_MAX_PIXELS[format];
    return (!side || Math.max(w, h) <= side) && (!pixels || w * h <= pixels);
  }

  /** The file type a result is really saved as, like imageio_utils.resolve_format plus the pipeline's
      switch to PNG for see-through pictures. "Same as original" keeps the source's type when Re:Solution
      can write it and it holds a picture of `out` (width, height); a WebP over 16,383 px then becomes JPEG
      (PNG when it has see-through areas), as does a HEIC that's too big. */
  function savedFormat(format, sourceFormat, out, alpha = false) {
    let saved = format;
    if (format === 'original') {
      saved = ORIGINAL_FORMATS[String(sourceFormat || '').trim().toLowerCase()];
      if (!saved) return 'png';
      if (!formatHolds(saved, out)) saved = !alpha && saved !== 'jpeg' && formatHolds('jpeg', out) ? 'jpeg' : 'png';
    }
    if (alpha && !FORMATS_WITH_ALPHA.includes(saved)) return 'png';
    return saved;
  }

  /** The format to ask /api/estimate about: what's really saved, with HEIC and AVIF sized like JPEG. */
  function estimateFormat(format, sourceFormat, out, alpha = false) {
    const saved = savedFormat(format, sourceFormat, out, alpha);
    return SIZED_LIKE_JPEG.includes(saved) ? 'jpeg' : saved;
  }

  /** The server's estimate for a picture with see-through areas. /api/estimate doesn't know about them,
      so its file size is scaled the way sizing.estimated_file_bytes(alpha=True) does. */
  function withAlpha(est, alpha) {
    if (!alpha || !est || !Number.isFinite(est.file_mb)) return est;
    return { ...est, file_mb: round1(est.file_mb * ALPHA_SIZE_FACTOR) };
  }

  // "…very large: about 42 MB as JPEG…": the part of the size warning that names one picture's file size.
  const SIZE_CLAUSE = /about [\d.,]+ ?[KMG]B as [A-Za-z]+/;
  const LONG_JOB = /^This will take about [\d,]+ minutes\.$/;

  /** A warning with its numbers taken out, so the same warning about different pictures matches. */
  function warningKind(text) {
    return String(text).replace(SIZE_CLAUSE, '#').replace(/\d[\d,.]*\s*(?:[KMG]B)?/g, '#');
  }

  /** The warnings for the pictures in `items` ({warnings, file_mb, seconds, format_label}), one per kind:
      a batch of 20 pictures at 16K gets one "very large" note with the total, not 20 of them. The size
      named in that note is this readout's own (so it counts see-through areas), and a long-job note
      for a batch gives the time for all of them. Other kinds (disk space, file type) are kept as they are. */
  function mergeWarnings(items) {
    const list = items || [];
    const groups = new Map();
    for (const item of list) {
      for (const text of item.warnings || []) {
        const kind = warningKind(text);
        if (!groups.has(kind)) groups.set(kind, text);
      }
    }
    const batch = list.length > 1;
    return Array.from(groups.values()).map((text) => {
      if (SIZE_CLAUSE.test(text)) {
        if (!batch) {
          const [only] = list;
          return Number.isFinite(only.file_mb) && only.format_label
            ? text.replace(SIZE_CLAUSE, `about ${describeMegabytes(only.file_mb * 1e6)} as ${only.format_label}`)
            : text;
        }
        const total = list.reduce((sum, item) => sum + (Number.isFinite(item.file_mb) ? item.file_mb : 0), 0);
        return text.replace(SIZE_CLAUSE, `about ${describeMegabytes(total * 1e6)} in total`);
      }
      if (batch && LONG_JOB.test(text) && list.every((item) => Number.isFinite(item.seconds))) {
        const total = list.reduce((sum, item) => sum + item.seconds, 0);
        return `This will take ${formatApproxTime(total).replace(/^About/, 'about')} for all of them.`;
      }
      return text;
    });
  }

  /** The longest side a file type can hold, in pixels, or null when it has no limit (WebP: 16,383, JPEG: 65,535). */
  function formatLimit(formatId, formats) {
    const listed = (formats || DEFAULT_FORMATS).find((f) => f.id === formatId);
    const known = DEFAULT_FORMATS.find((f) => f.id === formatId);
    return (listed && listed.max_long_edge) || (known && known.max_long_edge) || null;
  }

  /** Whether `formatId` can hold every output size in `outs` (WebP stops at 16,383 px, JPEG at 65,535). */
  function formatAllowed(formatId, outs, formats) {
    const limit = formatLimit(formatId, formats);
    if (!limit) return true;
    return (outs || []).every(([w, h]) => Math.max(w, h) <= limit);
  }

  /** The file type to save as at these output sizes. A pick that's too small for them gives way to the
      first of JPEG, PNG and TIFF that can hold them, and is kept as `blocked`, so it comes back by itself
      once the size allows it again. */
  function fitFormat(chosen, blocked, outs, formats) {
    const listed = (id) => (formats || DEFAULT_FORMATS).some((f) => f.id === id);
    if (blocked && listed(blocked) && formatAllowed(blocked, outs, formats)) return { format: blocked, blocked: null };
    if (!formatAllowed(chosen, outs, formats)) {
      const fallback = FALLBACK_FORMATS.find((id) => id !== chosen && listed(id) && formatAllowed(id, outs, formats));
      return { format: fallback || 'png', blocked: chosen };
    }
    return { format: chosen, blocked: blocked && listed(blocked) ? blocked : null };
  }

  const isCloudId = (id) => typeof id === 'string' && id.startsWith('cloud:');
  const KEY_CHECK_PATH = /^\/api\/providers\/[^/]+\/(key|session-key)$/;

  /** The colour family for a kind of work, used with [data-theme]. */
  function modeTheme(id, kind) {
    if (kind === 'video') return 'video';
    if (typeof id !== 'string') return 'neutral';
    if (isCloudId(id) || id.startsWith('cloud-')) return 'cloud';
    if (id === 'photoreal' || id.startsWith('photoreal:')) return 'photoreal';
    if (id === 'faithful' || id === 'faithful_sharp') return 'faithful';
    if (id === 'fast' || id === 'illustration') return id;
    return 'neutral';
  }

  function modeIcon(id) {
    const theme = modeTheme(id);
    if (theme === 'cloud') return ICONS.cloud;
    if (theme === 'neutral') return 'image';
    return ICONS[id] || ICONS[theme];
  }

  /** Photo-real's detail levels: from the server, or the agreed defaults. */
  function photorealLevels(status) {
    const photoreal = ((status && status.image_modes) || []).find((m) => m.id === 'photoreal');
    if (photoreal && Array.isArray(photoreal.levels) && photoreal.levels.length) return photoreal.levels;
    return PHOTOREAL_LEVELS;
  }

  /** A mode's name for people, e.g. "Photo-real · Rich"; "No change needed" when a picture was already big enough,
      and "Drawn directly (vector)" for an SVG, which is drawn at the new size with no AI. */
  function modeLabelFor(id, kind, status, level) {
    if (!id) return '';
    if (id === 'original') return NO_CHANGE_LABEL;
    if (id === VECTOR) return VECTOR_LABEL;
    const s = status || {};
    if (kind === 'video') {
      const mode = (s.video_modes || []).find((m) => m.id === id);
      return VIDEO_LABELS[id] || (mode && mode.label) || id;
    }
    const providers = s.providers || [];
    const providerName = (pid) => (providers.find((p) => p.id === pid) || { name: pid }).name;
    const mode = (s.image_modes || []).find((m) => m.id === id);
    let label;
    if (mode) label = mode.label;
    else if (id.startsWith('cloud-')) label = `Cloud (${providerName(id.slice(6))})`;
    else if (isCloudId(id)) {
      const [, providerId, modelId] = id.split(':');
      const provider = providers.find((p) => p.id === providerId);
      const model = provider && (provider.models || []).find((m) => m.id === modelId);
      label = model ? `${provider.name} ${model.label}` : `Cloud (${providerName(providerId)})`;
    } else if (id === 'photoreal') label = 'Photo-real';
    else label = id;
    if (id === 'photoreal' && level) {
      const found = photorealLevels(s).find((l) => l.id === level);
      if (found) label = `${label} · ${found.label}`;
    }
    return label;
  }

  /** The quality a picture will get. Its own choice wins, then the batch's explicit choice, then the suggestion. */
  function choiceFor(item, batch, selectable) {
    const canUse = (id) => !selectable || selectable.includes(id);
    const suggestion = item && item.suggestion;
    let mode;
    let level = null;
    let source;
    if (item && item.choice && item.choice.mode) {
      mode = item.choice.mode;
      level = item.choice.level || null;
      source = 'chosen';
    } else if (batch.explicit) {
      mode = batch.mode;
      source = 'chosen';
    } else if (suggestion && suggestion.mode && canUse(suggestion.mode)) {
      mode = suggestion.mode;
      level = suggestion.photoreal_level || null;
      source = 'suggested';
    } else {
      mode = batch.mode;
      source = 'default';
    }
    if (mode !== 'photoreal') level = null;
    else if (!(source === 'chosen' && level) && (batch.levelExplicit || !level)) level = batch.level;
    return { mode, level, source };
  }

  /** "photoreal:rich" -> {mode: "photoreal", level: "rich"}; the original crop isn't a choice. */
  function routeChoice(route) {
    if (!route || route === 'original') return null;
    const [mode, level] = String(route).split(':');
    return { mode, level: mode === 'photoreal' ? (level || DEFAULTS.photorealLevel) : null };
  }

  function routeLabel(route, status) {
    if (!route || route === 'original') return 'Original (enlarged)';
    const choice = routeChoice(route);
    return modeLabelFor(choice.mode, 'image', status, choice.level) || route;
  }

  /** The styles Preview styles makes: the usual five (without Photo-real when this computer can't run it),
      then a picture's suggested or chosen style when it isn't one of them, such as Illustration. */
  function previewRoutes(photorealOk, extraModes, selectable) {
    const routes = DEFAULT_PREVIEW_ROUTES.filter((route) => photorealOk || !route.startsWith('photoreal'));
    for (const mode of extraModes || []) {
      if (!PREVIEW_EXTRA_MODES.includes(mode) || routes.includes(mode)) continue;
      if (selectable && !selectable.includes(mode)) continue;
      routes.push(mode);
    }
    return routes;
  }

  /** A finished preview job's crops, in order: the plain enlargement first, then each style. A style
      that couldn't be made has no picture, only the server's friendly `error`. */
  function previewResults(job, routes, filePath, status) {
    return ((job && job.results) || []).map((result, n) => {
      const route = result.route || (n === 0 ? 'original' : (routes || [])[n - 1] || `style-${n}`);
      const error = result.error ? String(result.error) : null;
      return {
        route,
        label: result.label || routeLabel(route, status),
        url: result.url || (error ? null : filePath(job.id, n)),
        seconds: result.seconds,
        estFull: result.est_full_seconds,
        error,
      };
    });
  }

  /** True when a request failed only because this copy of the server doesn't have that endpoint yet. */
  function missingEndpoint(status, detail) {
    return status === 405 || (status === 404 && detail === 'Not Found');
  }

  /** The settings a use-case preset changes, keeping only values this app offers. */
  function presetChoices(preset, status) {
    const s = status || {};
    const out = {};
    if (!preset) return out;
    if ((s.sizes || []).some((size) => size.id === preset.size)) out.size = preset.size;
    const modes = (s.image_modes || []).filter((m) => m.available !== false).map((m) => m.id);
    if (modes.includes(preset.mode)) {
      out.mode = preset.mode;
      out.level = preset.mode === 'photoreal' && LEVEL_IDS.includes(preset.photoreal_level) ? preset.photoreal_level : null;
    }
    if ((s.formats || DEFAULT_FORMATS).some((f) => f.id === preset.output_format)) out.format = preset.output_format;
    if ((s.dpi_choices || DPI_CHOICES).includes(preset.dpi)) out.dpi = preset.dpi;
    if (GRAINS.includes(preset.grain)) out.grain = preset.grain;
    return out;
  }

  /** Pictures that need different qualities go in separate jobs, one per quality, in the order added. */
  function groupBySettings(choices) {
    const groups = [];
    for (const choice of choices) {
      const level = choice.mode === 'photoreal' ? (choice.level || null) : null;
      let group = groups.find((g) => g.mode === choice.mode && g.level === level);
      if (!group) {
        group = { mode: choice.mode, level, ids: [] };
        groups.push(group);
      }
      group.ids.push(choice.id);
    }
    return groups;
  }

  /** `list` in pieces of at most `size`, in order. */
  function inChunks(list, size) {
    const pieces = [];
    for (let i = 0; i < list.length; i += size) pieces.push(list.slice(i, i + size));
    return pieces;
  }

  /** How one group's pictures are sent, one job per part: checked pictures by their token and the rest
      as files, at most MAX_PICTURES_PER_JOB in a job (the server's limit), and samples one by one.
      A picture checked more than about a day ago is sent as its file, because the server has let go of it. */
  function sendingPlan(items, now = Date.now(), perJob = MAX_PICTURES_PER_JOB) {
    const stale = (p) => Number.isFinite(p.checkedAt) && now - p.checkedAt > TOKEN_MAX_AGE_MS;
    const byToken = (p) => Boolean(p.token) && !(stale(p) && p.file);
    const pictures = items.filter((p) => !p.sample);
    const staged = pictures.filter(byToken);
    const files = pictures.filter((p) => !byToken(p) && p.file);
    return [
      ...inChunks(staged, perJob).map((part) => ({ by: 'tokens', items: part })),
      ...inChunks(files, perJob).map((part) => ({ by: 'files', items: part })),
      ...items.filter((p) => p.sample).map((p) => ({ by: 'sample', items: [p] })),
    ];
  }

  /** True when the server no longer has a checked picture (it keeps them for 24 hours). */
  function stagedGone(httpStatus, detail) {
    return (httpStatus === 400 || httpStatus === 404) && STAGED_GONE.test(String(detail || ''));
  }

  /** A click at (px, py) on a viewW×viewH picture -> a size×size spot in source pixels, kept inside the picture. */
  function regionAt(px, py, viewW, viewH, srcW, srcH, size = 256) {
    const w = Math.min(size, srcW);
    const h = Math.min(size, srcH);
    const cx = (px / viewW) * srcW;
    const cy = (py / viewH) * srcH;
    const clampTo = (value, max) => Math.min(Math.max(value, 0), max);
    return [clampTo(Math.round(cx - w / 2), srcW - w), clampTo(Math.round(cy - h / 2), srcH - h), w, h];
  }

  /** The options sent for one group of pictures. Photo-real uses its own creative controls. */
  function imageOptionsFor(batch, choice) {
    const photoreal = choice.mode === 'photoreal';
    // "Stay close to the original" blends back toward the plain enlargement: resemblance 1.0 is
    // Photo-real's full result, 0 the enlargement (DESIGN: the blend between the two).
    const resemblance = photoreal ? round2(1 - batch.stayClose / 100) : 1;
    return {
      mode: choice.mode,
      photoreal_level: choice.level || batch.level,
      size: batch.size,
      strength: photoreal ? resemblance : round2(batch.strength / 100),
      creativity: photoreal ? round2(batch.creativity / 100) : 0,
      resemblance,
      grain: batch.grain,
      output_format: batch.format,
      dpi: batch.dpi,
    };
  }

  /** Progress over a batch, counted in pictures (a job can hold several). Pictures a stopped job
      didn't get to are "stopped", not failed, and the ones a running job has already finished count
      as done. */
  function batchSummary(jobs) {
    let photos = 0;
    let done = 0;
    let failed = 0;
    let stopped = 0;
    let working = 0;
    let progress = 0;
    let active = 0;
    for (const job of jobs || []) {
      const count = Math.max(1, (job.inputs || []).length);
      const made = Math.min(count, (job.results || []).length);
      photos += count;
      if (job.status === 'done' || job.status === 'cancelled') {
        done += made;
        if (job.status === 'done') failed += count - made;
        else stopped += count - made;
      } else if (job.status === 'failed') {
        failed += count;
      } else {
        active += 1;
        done += made;
        working += count - made;
        const fraction = Number.isFinite(job.progress) ? Math.min(1, Math.max(0, job.progress)) : 0;
        progress += Math.max(0, fraction * count - made);
      }
    }
    return {
      photos, done, failed, stopped, working,
      fraction: photos ? round2(Math.min(photos, done + failed + stopped + progress) / photos) : 0,
      finished: active === 0,
    };
  }

  /** The file picker's accept list from what the server can read. */
  function acceptAttribute(accept, kind) {
    const list = accept && Array.isArray(accept[kind]) && accept[kind].length ? accept[kind] : FALLBACK_ACCEPT[kind];
    return [kind === 'videos' ? 'video/*' : 'image/*', ...list].join(',');
  }

  /** "image", "video" or null, from the file name's extension, then its type. */
  function fileKind(file, accept) {
    const name = String((file && file.name) || '').toLowerCase();
    const dot = name.lastIndexOf('.');
    const ext = dot >= 0 ? name.slice(dot) : '';
    const list = (kind) => (accept && Array.isArray(accept[kind]) && accept[kind].length ? accept[kind] : FALLBACK_ACCEPT[kind])
      .map((e) => String(e).toLowerCase());
    if (ext && list('images').includes(ext)) return 'image';
    if (ext && list('videos').includes(ext)) return 'video';
    const type = String((file && file.type) || '');
    if (type.startsWith('image/')) return 'image';
    if (type.startsWith('video/')) return 'video';
    return null;
  }

  /** A 401 means "log in again" only for a guest, and never when a cloud service turned a key down. */
  function asksForPassword(httpStatus, path, viewerIsHost) {
    return httpStatus === 401 && !viewerIsHost && path !== API.login && !KEY_CHECK_PATH.test(path);
  }

  /** Keep the "mine" flag this viewer already has when an update arrives without one. (Server-sent
      job events are built for each viewer, so theirs is right and always wins.) */
  function mergeJob(previous, incoming) {
    if (typeof incoming.mine === 'boolean' || !previous || typeof previous.mine !== 'boolean') return incoming;
    return { ...incoming, mine: previous.mine };
  }

  /** The connected services a guest can forget their own key for: the server says so (own_key), or,
      from servers that don't, this browser remembers adding one. A key the host shares isn't theirs. */
  function ownKeyProviders(providers, remembered) {
    const added = new Set(remembered || []);
    return (providers || []).filter((p) => p.connected && (typeof p.own_key === 'boolean' ? p.own_key : added.has(p.id)));
  }

  /** Which picture result n of a job came from. A batch leaves out the pictures that failed (each is
      named in a note: "NAME wasn't upscaled: …"), and a stopped one the pictures it didn't get to, so
      result n is the n-th picture that didn't fail. null when that can't be told for sure. */
  function resultInputIndex(job, n) {
    const inputs = (job && job.inputs) || [];
    const results = (job && job.results) || [];
    if (results.length === inputs.length) return n < inputs.length ? n : null;
    const notes = results.flatMap((result) => result.notes || []);
    const failed = inputs.map((name) => notes.some((note) => String(note).startsWith(`${name} wasn't upscaled:`)));
    if (failed.some(Boolean) && new Set(inputs).size !== inputs.length) return null; // can't tell which one
    const kept = inputs.map((_, i) => i).filter((i) => !failed[i]);
    const complete = kept.length === results.length || (job.status === 'cancelled' && kept.length > results.length);
    return complete && n < results.length ? kept[n] : null;
  }

  /** The cloud quality to start with for a service: its SeedVR2 model when it has one (the app's default look). */
  function preferredCloudMode(modes, providerId) {
    const offered = modes.filter((m) => m.providerId === providerId);
    return offered.find((m) => m.id.endsWith(':seedvr2')) || offered[0] || null;
  }

  /** The quality chosen when the page opens: the host's default, else Photo-real, else Faithful. */
  function pickDefaultMode(status) {
    const modes = (status && status.image_modes) || [];
    const usable = (id) => modes.some((m) => m.id === id && m.available !== false);
    const preferred = status && status.settings && status.settings.default_image_mode;
    for (const id of [preferred, DEFAULTS.imageMode, 'faithful']) {
      if (id && usable(id)) return id;
    }
    const first = modes.find((m) => m.available !== false && !m.needs_key);
    return first ? first.id : DEFAULTS.imageMode;
  }

  /** The time-left line under a running job. Empty when the job's own message already says how long
      (the video engine adds "…, less than a minute left" before the queue has a time of its own). */
  function jobEtaText(job) {
    if (!job || job.status !== 'running') return '';
    if (/\bleft\.?\s*$/i.test(String(job.message || ''))) return '';
    const eta = formatEta(job.eta_seconds);
    if (eta) return eta;
    return job.kind === 'video' ? 'Working out how long this takes…' : '';
  }

  /** The drop zone's words. Phones and tablets can't drag files or paste with a shortcut, so they aren't told to. */
  function dropzoneCopy({ touch = false, count = 0, pasteKeys = 'Ctrl+V' } = {}) {
    const button = count ? 'Add more' : 'Choose pictures';
    if (touch) return { title: count ? 'Add more pictures' : 'Pick from your photos', button, hint: PHOTO_TYPES_HINT };
    return {
      title: count ? 'Drop more pictures here' : 'Drag pictures here',
      button,
      hint: `Or paste one (${pasteKeys}). ${PHOTO_TYPES_HINT}`,
    };
  }

  /** "Preview first 3 seconds" only helps when the clip is longer than that. An unknown length still offers it. */
  function offerVideoPreview(duration) {
    return !(Number.isFinite(duration) && duration > 0 && duration <= PREVIEW_SECONDS + 0.5);
  }

  /** The file types to offer, with their names for people: from the server, or the agreed list. */
  function formatChoices(status) {
    const listed = (status && Array.isArray(status.formats) && status.formats.length) ? status.formats : DEFAULT_FORMATS;
    return listed.filter((f) => f && f.id)
      .map((f) => ({ ...f, label: FORMAT_LABELS[f.id] || f.label || String(f.id).toUpperCase() }));
  }

  /** After a batch is sent, a quality picked for it no longer outranks the suggestions of pictures added next. */
  function choicesAfterSending(photo, remaining) {
    return remaining > 0 ? { ...photo } : { ...photo, explicit: false, levelExplicit: false };
  }

  /** Settings' "default quality" list: the local modes this computer can run, showing the one new photos really get. */
  function defaultModeChoices(status, settings) {
    const options = ((status && status.image_modes) || []).filter((m) => !isCloudId(m.id) && m.available !== false);
    let value = pickDefaultMode({ ...status, settings });
    if (!options.some((m) => m.id === value)) value = options.length ? options[0].id : '';
    return { options, value };
  }

  /** "light", "dark" or "auto" from a stored value; anything else is Auto. */
  function themeFrom(value) {
    return value === 'light' || value === 'dark' ? value : 'auto';
  }

  /** The theme the button moves to next: Light -> Dark -> Auto -> Light. */
  function nextTheme(theme) {
    return THEME_ORDER[(THEME_ORDER.indexOf(themeFrom(theme)) + 1) % THEME_ORDER.length];
  }

  /** The theme button's name and tooltip: where it is now, and what a click does. */
  function themeLabel(theme) {
    const now = themeFrom(theme);
    return `Theme: ${THEME_NAMES[now]}. Switch to ${THEME_NAMES[nextTheme(now)].replace(/ \(.*\)$/, '')}`;
  }

  /** What to do when the computer asks about letting others connect, on the host's own computer
      (it's the host who sees the share panel). */
  function firewallNote(platform) {
    const name = String(platform || '');
    if (/Mac|iPhone|iPad/i.test(name)) return 'If your Mac asks whether to allow incoming connections, click Allow.';
    if (/Win/i.test(name)) return 'If Windows asks about the firewall, click Allow access.';
    return '';
  }

  /** The firewall note for the share panel: the server's own (it knows the system it runs on, even when
      that's null for none), or the browser's guess from an older server that doesn't say. */
  function shareNote(share, platform) {
    return share && has(share, 'note') ? share.note || '' : firewallNote(platform);
  }

  /** Where the share panel's QR code comes from: the server's drawing (status.share.qr_svg) shown as a
      picture, so nothing in it can run; older servers draw it at /api/share/qr.svg instead. */
  function shareQrSource(share, urls) {
    if (share && typeof share.qr_svg === 'string' && /^\s*<svg[\s>]/.test(share.qr_svg)) {
      return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(share.qr_svg)}`;
    }
    return urls && urls.length ? API.shareQr(urls[0]) : null;
  }

  /* ---------- Batches of many pictures (DESIGN "Batch-friendly layout") ---------- */

  /** Runs at most `limit` tasks (functions returning a promise) at a time, in the order given.
      run(task) returns a promise for that task's result. */
  function limiter(limit) {
    let active = 0;
    const waiting = [];
    const pump = () => {
      while (active < limit && waiting.length) {
        const { task, resolve, reject } = waiting.shift();
        active += 1;
        Promise.resolve().then(task).then(resolve, reject).finally(() => {
          active -= 1;
          pump();
        });
      }
    };
    const run = (task) => new Promise((resolve, reject) => {
      waiting.push({ task, resolve, reject });
      pump();
    });
    run.active = () => active;
    run.waiting = () => waiting.length;
    return run;
  }

  /** "Checking 37 of 120…": the picture being checked now, counting the ones already done. */
  function checkingText(done, total) {
    if (!total) return '';
    const now = Math.min(total, done + 1);
    return total === 1 ? 'Checking your picture…' : `Checking ${num0(now)} of ${num0(total)}…`;
  }

  function num0(n) {
    return Number(n).toLocaleString('en-US');
  }

  /** A picture that needs a look: it can't be used, it came with a warning, or it couldn't be measured. */
  function needsAttention(item) {
    if (!item) return false;
    if (item.status === 'problem') return true;
    if ((item.warnings || []).length) return true;
    return item.status === 'ready' && Boolean(item.previewBroken) && !item.width;
  }

  /** Whether a picture shows people: the server's face count when it sends one, else its suggestion's
      reasons (a suggestion made because of faces says so). */
  function hasPeople(item) {
    const suggestion = item && item.suggestion;
    if (!suggestion) return false;
    const faces = Number.isFinite(suggestion.faces) ? suggestion.faces
      : suggestion.signals && Number.isFinite(suggestion.signals.faces) ? suggestion.signals.faces : null;
    if (faces != null) return faces > 0;
    return (suggestion.reasons || []).some((reason) => PEOPLE_WORDS.test(String(reason)));
  }

  /** Whether a picture belongs under a filter chip. `choice` is the quality it will get. */
  function matchesFilter(item, filter, choice) {
    const usable = item.status === 'ready';
    switch (filter) {
      case 'attention': return needsAttention(item);
      case 'photoreal': return usable && Boolean(choice) && choice.mode === 'photoreal';
      case 'faithful': return usable && Boolean(choice) && (choice.mode === 'faithful' || choice.mode === 'faithful_sharp');
      case 'people': return hasPeople(item);
      default: return true;
    }
  }

  function filterItems(items, filter, choiceOf) {
    return (items || []).filter((item) => matchesFilter(item, filter, choiceOf(item)));
  }

  /** How many pictures each filter chip would show. */
  function filterCounts(items, choiceOf) {
    const counts = Object.fromEntries(TRAY_FILTERS.map((f) => [f, 0]));
    for (const item of items || []) {
      const choice = choiceOf(item);
      for (const filter of TRAY_FILTERS) if (matchesFilter(item, filter, choice)) counts[filter] += 1;
    }
    return counts;
  }

  /** The part of a long list that's drawn: at most `shown` (a multiple of TRAY_PAGE), plus what's left. */
  function trayPage(list, shown = TRAY_PAGE) {
    const limit = Math.max(TRAY_PAGE, shown || 0);
    const items = (list || []).slice(0, limit);
    const rest = Math.max(0, (list || []).length - items.length);
    return { items, rest, more: Math.min(TRAY_PAGE, rest) };
  }

  /** The "Show more" button's words for `rest` pictures not drawn yet. */
  function moreLabel(rest) {
    if (rest <= 0) return '';
    if (rest > TRAY_PAGE) return `Show ${TRAY_PAGE} more (${num0(rest)} not shown)`;
    return rest === 1 ? 'Show the last one' : `Show the other ${rest}`;
  }

  /** Selection after clicking the checkbox of `target`. With Shift held after an earlier click (`anchor`),
      every picture between the two (in the order shown) follows the clicked one. Returned in that order;
      selected pictures a filter hides stay selected. */
  function rangeSelect(selected, order, anchor, target, on) {
    const result = new Set(selected || []);
    const from = anchor == null ? -1 : order.indexOf(anchor);
    const to = order.indexOf(target);
    if (from < 0 || to < 0) {
      if (on) result.add(target);
      else result.delete(target);
    } else {
      const [a, b] = from <= to ? [from, to] : [to, from];
      for (let i = a; i <= b; i += 1) {
        if (on) result.add(order[i]);
        else result.delete(order[i]);
      }
    }
    const shown = order.filter((id) => result.has(id));
    const hidden = Array.from(result).filter((id) => !order.includes(id));
    return [...shown, ...hidden];
  }

  /** Keeps Tab inside an open panel: the index of the control to move to, or -1 to let the browser
      move on by itself. `index` is where focus is now among `count` controls (-1 when it's outside). */
  function trapFocus(count, index, backwards) {
    if (!count) return -1;
    if (index < 0) return backwards ? count - 1 : 0;
    if (backwards && index === 0) return count - 1;
    if (!backwards && index === count - 1) return 0;
    return -1;
  }

  /** The compare viewer's labels: "Before · 1,024 × 795, simply enlarged" and "After · 3,840 × 2,981". */
  function compareLabels(before, after) {
    const size = (pair) => (pair && pair[0] && pair[1] ? `${num0(pair[0])} × ${num0(pair[1])}` : '');
    const b = size(before);
    const a = size(after);
    return {
      before: b ? `Before · ${b}, simply enlarged` : 'Before · simply enlarged',
      after: a ? `After · ${a}` : 'After',
      single: a ? `Upscaled · ${a}` : 'Upscaled',
    };
  }

  /** The viewer's keys: the picture to show next, or null when the key isn't one of its own. ([ and ]
      always move; the arrows do when the zoom viewer doesn't use them to look around.) */
  function viewerTarget(index, count, key) {
    if (!count) return null;
    if (key === 'ArrowLeft' || key === '[') return Math.max(0, index - 1);
    if (key === 'ArrowRight' || key === ']') return Math.min(count - 1, index + 1);
    if (key === 'Home') return 0;
    if (key === 'End') return count - 1;
    return null;
  }

  /** "about 55 minutes": formatApproxTime in the middle of a sentence. */
  function approxLower(seconds) {
    const text = formatApproxTime(seconds);
    return text ? text.charAt(0).toLowerCase() + text.slice(1) : '';
  }

  /** Dollars for people: "$0.05", "$1.20", "$0.008". */
  function formatUsd(usd) {
    if (!Number.isFinite(usd) || usd < 0) return '';
    if (usd > 0 && usd < 0.1) return `$${usd.toFixed(3).replace(/0$/, '')}`;
    return `$${usd.toFixed(2)}`;
  }

  /** What one picture costs on a cloud service: the server's own figure (cost_usd), or the service's
      price for an 8K picture scaled by how many pixels this one will have. null when it isn't known. */
  function cloudCost(est, out, cost8k) {
    if (est && Number.isFinite(est.cost_usd)) return est.cost_usd;
    if (!Number.isFinite(cost8k) || cost8k <= 0 || !out) return null;
    return Math.round(((cost8k * out[0] * out[1]) / EIGHT_K_PIXELS) * 1000) / 1000;
  }

  /** Whether a cloud picture is paid from the person's plan credits instead of a published price: the
      server says so with cost_usd null (Magnific), or, before it has answered, the service's 8K price is 0. */
  function cloudUsesCredits(est, cost8k) {
    if (est && Object.prototype.hasOwnProperty.call(est, 'cost_usd')) return est.cost_usd === null;
    return cost8k === 0;
  }

  /** The words for one cloud picture's cost. */
  function cloudCostText(cost, credits) {
    if (credits) return "Uses your plan's credits";
    if (cost === 0) return 'Nothing to pay';
    return Number.isFinite(cost) ? `about ${formatUsd(cost)}` : 'Price not published';
  }

  /** Totals for the tray's summary and the sticky bar. Each row: {status, attention, seconds, file_mb,
      cost, cloud}. Pictures that can't be used aren't counted, and a time is only given when every
      picture has one. */
  function trayTotals(rows) {
    const list = rows || [];
    const usable = list.filter((row) => row.status !== 'problem');
    const measured = usable.filter((row) => row.status === 'ready');
    const timed = measured.every((row) => Number.isFinite(row.seconds));
    const cloud = usable.filter((row) => row.cloud);
    const priced = cloud.every((row) => Number.isFinite(row.cost));
    return {
      count: usable.length,
      problems: list.length - usable.length,
      attention: list.filter((row) => row.attention).length,
      checking: usable.length - measured.length,
      seconds: measured.length && timed ? measured.reduce((sum, row) => sum + row.seconds, 0) : null,
      mb: measured.reduce((sum, row) => sum + (Number.isFinite(row.file_mb) ? row.file_mb : 0), 0),
      cloud: cloud.length,
      cost: cloud.length && priced ? Math.round(cloud.reduce((sum, row) => sum + row.cost, 0) * 1000) / 1000 : null,
    };
  }

  /** "19 pictures · 1 needs attention · about 55 minutes · about 291 MB". */
  function traySummaryText(totals) {
    const parts = [plural0(totals.count + totals.problems, 'picture')];
    if (totals.attention) parts.push(`${num0(totals.attention)} need${totals.attention === 1 ? 's' : ''} attention`);
    if (totals.seconds != null && !totals.checking) parts.push(approxLower(totals.seconds));
    if (totals.mb > 0) parts.push(formatFileSize(totals.mb));
    if (totals.cost != null) parts.push(`about ${formatUsd(totals.cost)}`);
    return parts.join(' · ');
  }

  function plural0(n, one, many) {
    return n === 1 ? `1 ${one}` : `${num0(n)} ${many || `${one}s`}`;
  }

  /** The sticky bar's second line: time, size and (for cloud pictures) cost. */
  function actionBarTotals(totals) {
    const parts = [];
    // While pictures are still being checked the size only covers the ones done: "so far".
    if (totals.checking) parts.push(`still checking ${num0(totals.checking)}`);
    if (totals.seconds != null && !totals.checking) parts.push(approxLower(totals.seconds));
    if (totals.mb > 0) parts.push(`${formatFileSize(totals.mb)}${totals.checking ? ' so far' : ''}`);
    if (totals.cost != null) parts.push(`about ${formatUsd(totals.cost)} for ${plural0(totals.cloud, 'cloud picture')}`);
    let text = parts.join(' · ');
    if (totals.problems) text += `${text ? '. ' : ''}${totals.problems === 1 ? "1 picture can't be used and is left out" : `${num0(totals.problems)} pictures can't be used and are left out`}.`;
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  /** A folded step's one-line summary of qualities: "Photo-real · Balanced", or the most used ones with
      how many pictures each: "Photo-real · Balanced (12), Faithful (5), +1 more". */
  function qualitySummary(labels) {
    const counts = new Map();
    for (const label of labels || []) counts.set(label, (counts.get(label) || 0) + 1);
    if (counts.size <= 1) return counts.size ? counts.keys().next().value : '';
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    const parts = sorted.slice(0, 2).map(([label, n]) => `${label} (${n})`);
    const extra = sorted.length - 2;
    return `${parts.join(', ')}${extra > 0 ? `, +${extra} more` : ''}`;
  }

  /** "JPEG, 300 DPI", plus grain and detail strength when they aren't the usual. */
  function savedSummary({ format, dpi, grain = 'off', strength = 100, showStrength = false }) {
    const parts = [`${format}, ${dpi} DPI`];
    if (grain && grain !== 'off') parts.push(`${grain} grain`);
    if (showStrength && strength !== 100) parts.push(`${strength}% detail`);
    return parts.join(', ');
  }

  /** Folded steps as this browser saved them ({stepId: true}); anything unreadable is "none folded". */
  function parseSteps(raw) {
    try {
      const value = JSON.parse(raw || '{}');
      if (!value || typeof value !== 'object' || Array.isArray(value)) return {};
      return Object.fromEntries(Object.entries(value).filter(([, folded]) => typeof folded === 'boolean'));
    } catch {
      return {};
    }
  }

  /** After the first batch is sent, the settings steps nobody folded or opened by hand fold into their
      summaries ("a completed step folds"); a step the person opened or closed keeps their choice. */
  function foldAfterSending(stored, ids = SETTINGS_STEPS) {
    const next = { ...(stored || {}) };
    for (const id of ids) if (typeof next[id] !== 'boolean') next[id] = true;
    return next;
  }

  /** The tray's view ("grid" or "list") and fold, as saved; anything else is the grid, shown in full. */
  function parseTray(raw) {
    try {
      const value = JSON.parse(raw || '{}') || {};
      return { view: value.view === 'list' ? 'list' : 'grid', folded: value.folded === true };
    } catch {
      return { view: 'grid', folded: false };
    }
  }

  /** Results in the order shown: each batch this browser sent is one group (its jobs in the order sent),
      every other job a group of its own; newest first. */
  function groupJobs(jobs, batches) {
    const byId = new Map((jobs || []).map((job) => [job.id, job]));
    const used = new Set();
    const groups = [];
    for (const batch of batches || []) {
      const members = (batch.jobIds || []).map((id) => byId.get(id)).filter((job) => job && !used.has(job.id));
      if (!members.length) continue;
      for (const job of members) used.add(job.id);
      groups.push({ id: `batch-${batch.id || members[0].id}`, jobs: members, created: Math.min(...members.map((j) => j.created || 0)) });
    }
    for (const job of jobs || []) {
      if (!used.has(job.id)) groups.push({ id: `job-${job.id}`, jobs: [job], created: job.created || 0 });
    }
    return groups.sort((a, b) => b.created - a.created);
  }

  /** Time left for a batch: the running job's own estimate (or its share of the estimate made before
      sending), plus the estimates of its jobs still waiting. null when one of them isn't known. */
  function batchSecondsLeft(jobs, estimates) {
    const est = estimates || {};
    let left = 0;
    for (const job of jobs || []) {
      if (job.status === 'running') {
        if (Number.isFinite(job.eta_seconds)) left += job.eta_seconds;
        else if (Number.isFinite(est[job.id])) left += est[job.id] * (1 - Math.min(1, Math.max(0, job.progress || 0)));
        else return null;
      } else if (job.status === 'queued') {
        if (!Number.isFinite(est[job.id])) return null;
        left += est[job.id];
      }
    }
    return left;
  }

  /** "7 of 24 done · about 38 minutes left", or how a finished batch ended. */
  function batchProgressText(summary, secondsLeft) {
    if (summary.finished) {
      if (!summary.failed && !summary.stopped) {
        return summary.photos === 1 ? 'Your picture is done.' : `All ${num0(summary.photos)} pictures are done.`;
      }
      const parts = [`${num0(summary.done)} of ${plural0(summary.photos, 'picture')} done`];
      if (summary.failed) parts.push(`${num0(summary.failed)} didn't work`);
      if (summary.stopped) parts.push(`${num0(summary.stopped)} stopped`);
      return `Finished: ${parts.join(', ')}.`;
    }
    let text = `${num0(summary.done)} of ${plural0(summary.photos, 'picture')} done`;
    if (secondsLeft != null && secondsLeft >= 0) text += ` · ${approxLower(secondsLeft)} left`;
    return text;
  }

  /** Jobs to forget after a fresh list from the server: the ones it no longer lists, unless this page
      heard about them after it asked (a job made while the list was on its way). */
  function jobsToDrop(known, listed, askedAt) {
    const present = new Set(listed || []);
    return (known || []).filter((job) => !present.has(job.id) && !(job.seenAt > askedAt)).map((job) => job.id);
  }

  /** The cloud services a batch uses, in the order its pictures name them, with how many go to each. */
  function cloudProviderCounts(modeIds) {
    const counts = new Map();
    for (const id of modeIds || []) {
      if (typeof id !== 'string' || !id.startsWith('cloud:')) continue;
      const provider = id.split(':')[1];
      counts.set(provider, (counts.get(provider) || 0) + 1);
    }
    return Array.from(counts, ([id, count]) => ({ id, count }));
  }

  /** "fal.ai, Replicate and Magnific". */
  function andList(words) {
    return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`;
  }

  /** The notice shown before anything is uploaded: every service the pictures go to (each picture can
      have its own quality), and what each one keeps, from its guide (`notes`, once loaded). */
  function cloudNoticeLines(entries, notes) {
    const known = notes || {};
    if (entries.length === 1) {
      const { id, name } = entries[0];
      const cost = `${name} charges for each picture.`;
      return [known[id] ? `${known[id]} ${cost}` : `Your pictures will be sent to ${name} to be upscaled. ${cost}`];
    }
    const where = andList(entries.map((e, i) => (i === 0
      ? `${plural0(e.count, 'picture')} will be sent to ${e.name}` : `${num0(e.count)} to ${e.name}`)));
    return [`${where} to be upscaled. Each service charges for each picture.`, ...entries.map((e) => known[e.id]).filter(Boolean)];
  }

  /** "Download all (zip)" for a batch sent as several jobs: a job of one picture saves that picture. */
  function downloadAllTitle(jobs) {
    const zips = jobs.filter((job) => (job.results || []).length > 1).length;
    const single = jobs.filter((job) => (job.results || []).length === 1).length;
    const parts = [];
    if (zips) parts.push(plural0(zips, '.zip file'));
    if (single) parts.push(plural0(single, 'picture'));
    return `Saves ${parts.join(' and ')}`;
  }

  /** "Select all"'s name starts with the words it shows ("3 selected"), so voice control finds it. */
  function selectAllLabel(selected, filter, shown) {
    const what = filter === 'all' ? 'Select all pictures' : `Select all ${num0(shown)} pictures shown`;
    return selected ? `${num0(selected)} selected. ${what}` : what;
  }

  /** A tray picture's name for screen readers: its parts, without a doubled full stop. */
  function openLabel(parts) {
    const said = parts.map((part) => String(part || '').replace(/[.\s]+$/, '')).filter(Boolean).join(', ');
    return `${said}. Show details`;
  }

  /** Arrow keys in a list of pictures that is one Tab stop. `count` controls, `kinds` per picture (a
      tick box and the picture), `columns` pictures to a row. Left and Right go one control at a time,
      Up and Down a row (to the last picture when there's none right below), Home and End to the first
      and last picture. Returns the control to move to (the same one at an edge), or null for other keys. */
  function rovingTarget(count, columns, index, key, kinds = 1) {
    if (count <= 0 || index < 0 || index >= count) return null;
    const pictures = Math.ceil(count / kinds);
    const at = Math.floor(index / kinds);
    const kind = index % kinds;
    const across = Math.max(1, columns || 1);
    let to;
    if (key === 'ArrowRight') to = index + 1;
    else if (key === 'ArrowLeft') to = index - 1;
    else if (key === 'ArrowDown') {
      const lastRow = Math.floor((pictures - 1) / across) * across;
      if (at >= lastRow) return index;
      to = Math.min(at + across, pictures - 1) * kinds + kind;
    } else if (key === 'ArrowUp') {
      if (at < across) return index;
      to = (at - across) * kinds + kind;
    } else if (key === 'Home') to = kind;
    else if (key === 'End') to = (pictures - 1) * kinds + kind;
    else return null;
    return Math.max(0, Math.min(count - 1, to));
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      limiter, checkingText, needsAttention, hasPeople, matchesFilter, filterItems, filterCounts, trayPage,
      moreLabel, rangeSelect, trapFocus, viewerTarget, formatUsd, cloudCost, cloudUsesCredits, cloudCostText, trayTotals, traySummaryText,
      actionBarTotals, qualitySummary, savedSummary, parseSteps, foldAfterSending, parseTray, groupJobs,
      batchSecondsLeft, batchProgressText, jobsToDrop, TRAY_PAGE, INSPECT_AT_ONCE, MIN_PASSWORD,
      WINDOWS_HEVC_NOTE,
      DEFAULTS, DEFAULT_PREVIEW_ROUTES, targetSize, videoOutputSize, formatEta, formatApproxTime, formatDuration,
      formatBytes, formatFileSize, splitSizeLabel, printSize, formatPrint, localEstimate, estimateFormat,
      savedFormat, withAlpha, mergeWarnings, formatLimit, formatAllowed, fitFormat, modeTheme, modeIcon,
      photorealLevels, modeLabelFor, choiceFor, routeChoice, routeLabel, previewRoutes, previewResults,
      missingEndpoint, presetChoices, groupBySettings, regionAt, imageOptionsFor, batchSummary, acceptAttribute,
      fileKind, pickDefaultMode, defaultModeChoices, asksForPassword, mergeJob, preferredCloudMode, jobEtaText,
      dropzoneCopy, offerVideoPreview, formatChoices, choicesAfterSending, themeFrom, nextTheme, themeLabel,
      firewallNote, shareNote, shareQrSource, MAX_PICTURES_PER_JOB, sendingPlan, stagedGone, ownKeyProviders,
      resultInputIndex, compareLabels, cloudProviderCounts, cloudNoticeLines, downloadAllTitle, selectAllLabel,
      openLabel, rovingTarget,
    };
  }
  if (typeof document === 'undefined') return;

  /* ====================================================================== */
  /*                              Browser app                               */
  /* ====================================================================== */

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clamp01 = (x) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);
  const num = (n) => Number(n).toLocaleString();
  const dims = (w, h) => `${num(w)} × ${num(h)}`;
  const isActive = (job) => job.status === 'queued' || job.status === 'running';
  const baseName = (path) => String(path || '').split(/[\\/]/).pop();
  const lowerFirst = (text) => (text ? text.charAt(0).toLowerCase() + text.slice(1) : text);
  const plural = (n, one, many) => (n === 1 ? `1 ${one}` : `${n} ${many || `${one}s`}`);
  const pasteKeys = () => (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? '⌘V' : 'Ctrl+V');
  // A phone or tablet: no mouse to drag files with, and no keyboard shortcut to paste.
  const touchOnly = () => window.matchMedia('(hover: none) and (pointer: coarse)').matches;

  const state = {
    status: null,
    settings: null,
    view: 'loading',
    jobs: new Map(),
    cards: new Map(),
    newJobs: new Set(),
    celebrate: new Set(),
    jobThumbs: new Map(),
    showAllJobs: false,
    mine: loadMine(),
    ownKeys: loadOwnKeys(), // services this browser's guest added their own key for
    photos: [],
    photo: {
      size: DEFAULTS.size,
      mode: DEFAULTS.imageMode,
      level: DEFAULTS.photorealLevel,
      explicit: false, // the person picked a quality (or a preset) for the whole batch
      levelExplicit: false,
      preset: null,
      strength: DEFAULTS.strength,
      creativity: DEFAULTS.creativity,
      stayClose: DEFAULTS.stayClose,
      grain: DEFAULTS.grain,
      format: DEFAULTS.format,
      dpi: DEFAULTS.dpi,
      formatNote: '',
      blockedFormat: null, // the person's file type while the size is too big for it; it comes back after
    },
    endpoints: { inspect: true, estimate: true, sample: true, preview: true, share: true },
    check: { total: 0, paintTimer: null }, // the pictures being checked now ("Checking 37 of 120…")
    tray: { ...loadTray(), filter: 'all', shown: TRAY_PAGE, selected: new Set(), anchor: null, order: [] },
    trayNodes: new Map(), // picture id -> {node, sig}: drawn once, redrawn only when something on it changes
    steps: loadSteps(), // the steps this browser folded
    drawerId: null, // the picture whose details are open
    cleared: loadCleared(), // finished jobs cleared from this browser's results
    batchOpen: new Map(), // batch cards opened or closed by hand
    batchShown: new Map(), // how many finished pictures each batch card draws
    seen: new Map(), // job id -> when this page last heard about it
    estimates: new Map(),
    estimateTimer: null,
    estimateRun: 0,
    video: null,
    videoMode: DEFAULTS.videoMode,
    uploading: null,
    autostartPhotoreal: false,
    downloadJobId: null,
    guides: new Map(),
    signatures: new Map(),
    events: null,
    statusRefreshQueued: false,
    pollTimer: null,
    doneWhileHidden: false,
    toastTimer: null,
    sharing: false, // a Share on Wi-Fi switch is on its way to the server
    connectionLost: false, // live updates dropped; the status is fetched again when they're back
    confirmClear: false, // "Remove all" is asking first
    stylesFor: null, // the picture Preview styles was opened for, to give focus back to it
  };

  const el = {};
  let wizard = null;
  let styles = null;
  let viewer = null;
  let videoCompare = null;
  const inspectPool = limiter(INSPECT_AT_ONCE);
  const estimatePool = limiter(ESTIMATES_AT_ONCE);

  /* ---------- DOM helpers ---------- */

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value == null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'dataset') Object.assign(node.dataset, value);
        else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
        else node.setAttribute(key, value === true ? '' : String(value));
      }
    }
    for (const child of children.flat(Infinity)) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function icon(name, className = 'icon') {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', className);
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    const use = document.createElementNS(SVG_NS, 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.append(use);
    return svg;
  }

  /** A coloured mode label: icon + words, never colour alone. */
  function modeTag(id, kind, level, { prefix = '', quiet = false, tagName = 'span', ...props } = {}) {
    const theme = modeTheme(id, kind);
    const name = modeLabel(id, kind, level);
    return h(tagName, { class: `tag${quiet ? ' tag-quiet' : ''}`, 'data-theme': theme, ...props },
      icon(kind === 'video' ? ICONS.video : modeIcon(id)), `${prefix}${name}`);
  }

  function showError(node, message) {
    node.textContent = message;
    node.hidden = !message;
  }

  function setButton(button, { label, busy = false, disabled = false } = {}) {
    if (label != null) {
      const span = button.querySelector('.btn-label');
      if (span) span.textContent = label;
      else button.textContent = label;
    }
    button.classList.toggle('is-busy', busy);
    button.disabled = disabled;
  }

  function setFeedback(node, message, isError = false) {
    node.textContent = message || '';
    node.classList.toggle('is-error', Boolean(isError));
  }

  function announce(text) {
    el.announcer.textContent = '';
    window.setTimeout(() => { el.announcer.textContent = text; }, 60);
  }

  /** A short visible note at the bottom of the screen (also read out). */
  function toast(text) {
    el.toast.textContent = text;
    el.toast.hidden = false;
    window.clearTimeout(state.toastTimer);
    state.toastTimer = window.setTimeout(() => { el.toast.hidden = true; }, 3200);
  }

  /** Re-render a radio group without losing keyboard focus on the chosen option. */
  function keepingFocus(container, render) {
    const active = document.activeElement;
    const value = active && container.contains(active) && active.name ? active.value : null;
    render();
    if (value != null) {
      const again = $$('input', container).find((input) => input.value === value);
      if (again) again.focus();
    }
  }

  function changed(key, value) {
    const signature = JSON.stringify(value);
    if (state.signatures.get(key) === signature) return false;
    state.signatures.set(key, signature);
    return true;
  }

  function setProgress(bar, fraction) {
    const known = Number.isFinite(fraction) && fraction > 0;
    const pct = known ? Math.min(100, fraction * 100) : 0;
    bar.classList.toggle('is-indeterminate', !known);
    bar.style.setProperty('--value', `${pct}%`);
    if (known) bar.setAttribute('aria-valuenow', String(Math.round(pct)));
    else bar.removeAttribute('aria-valuenow');
  }

  /* ---------- Talking to the server ---------- */

  class ApiError extends Error {
    constructor(message, status, detail) {
      super(message);
      this.status = status;
      this.detail = detail; // the server's own words, to tell a missing endpoint from a real error
    }
  }

  function rawDetail(data) {
    const detail = data && (data.detail ?? data.message ?? data.error);
    if (typeof detail === 'string') return detail;
    return detail && typeof detail.message === 'string' ? detail.message : '';
  }

  function errorMessage(data, status, path = '') {
    const detail = rawDetail(data);
    if (detail.trim() && !missingEndpoint(status, detail)) return detail;
    if (status === 401 && KEY_CHECK_PATH.test(path)) return "That key didn't work. Check that you copied all of it, then try again.";
    if (status === 401) return 'Re:Solution needs the password again.';
    if (status === 403) return COPY.hostOnly;
    if (status === 413) return 'That file is too big to send.';
    if (status === 507) return 'The main computer is out of space. Free some up and try again.';
    if (status >= 500) return COPY.serverTrouble;
    return COPY.generic;
  }

  /** `signal` (an AbortController's) stops the request; it then rejects with an error named AbortError. */
  async function api(path, { method = 'GET', body, signal } = {}) {
    const init = { method, credentials: 'same-origin', headers: {} };
    if (signal) init.signal = signal;
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (err) {
      if (err && err.name === 'AbortError') throw err; // stopped on purpose (a viewer closed): not "offline"
      throw new ApiError(COPY.offline, 0, '');
    }
    const data = await response.json().catch(() => null);
    if (asksForPassword(response.status, path, isHost())) showLogin();
    if (!response.ok) throw new ApiError(errorMessage(data, response.status, path), response.status, rawDetail(data));
    return data;
  }

  /** Multipart upload with progress (fetch cannot report upload progress). `onStart` gets the request, to abort it. */
  function upload(path, form, onProgress, onStart) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', path);
      xhr.withCredentials = true;
      xhr.responseType = 'json';
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      });
      xhr.addEventListener('load', () => {
        const data = xhr.response;
        if (asksForPassword(xhr.status, path, isHost())) showLogin();
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new ApiError(errorMessage(data, xhr.status, path), xhr.status, rawDetail(data)));
      });
      xhr.addEventListener('error', () => reject(new ApiError(COPY.offline, 0, '')));
      xhr.addEventListener('abort', () => reject(new ApiError('Stopped.', 0, 'aborted')));
      if (onStart) onStart(xhr);
      xhr.send(form);
    });
  }

  function getGuide(providerId) {
    if (!state.guides.has(providerId)) {
      const request = api(API.guide(providerId)).catch((err) => {
        state.guides.delete(providerId);
        throw err;
      });
      state.guides.set(providerId, request);
    }
    return state.guides.get(providerId);
  }

  /* ---------- Jobs that belong to this browser ---------- */

  /** The batches this browser sent (newest first), each {id, jobIds, created, est: {jobId: seconds}}. */
  function batchList(saved) {
    const list = Array.isArray(saved.batches) ? saved.batches : (saved.batch ? [saved.batch] : []);
    return list.filter((b) => b && Array.isArray(b.jobIds) && b.jobIds.length)
      .map((b) => ({ id: b.id || b.jobIds[0], jobIds: b.jobIds, created: b.created || 0, est: b.est || {} }));
  }

  function loadMine() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(MY_JOBS_KEY) || '{}');
      return {
        ids: new Set(saved.ids || []),
        owner: saved.owner || null,
        dims: new Map(Object.entries(saved.dims || {})),
        batches: batchList(saved),
      };
    } catch {
      return { ids: new Set(), owner: null, dims: new Map(), batches: [] };
    }
  }

  function saveMine() {
    try {
      const ids = Array.from(state.mine.ids).slice(-200);
      const keep = new Set(ids);
      const dimsEntries = Array.from(state.mine.dims.entries()).filter(([id]) => keep.has(id)).slice(-200);
      const batches = state.mine.batches.filter((b) => b.jobIds.length).slice(0, 40);
      window.localStorage.setItem(MY_JOBS_KEY, JSON.stringify({
        ids, owner: state.mine.owner, dims: Object.fromEntries(dimsEntries), batches,
      }));
    } catch {
      /* Private windows can refuse storage; the page still works. */
    }
  }

  /* A guest's own cloud keys live in the server's memory for one browser session. This browser
     remembers which services it added one for, so it can offer to forget them (the server doesn't say
     whose key a connected service uses). It's a session cookie, so it goes when the browser closes,
     like the server's copy of the key. */

  function loadOwnKeys() {
    try {
      const entry = document.cookie.split('; ').find((part) => part.startsWith(`${OWN_KEYS_COOKIE}=`));
      const value = entry ? decodeURIComponent(entry.slice(OWN_KEYS_COOKIE.length + 1)) : '';
      return new Set(value.split(',').filter((id) => /^[\w-]+$/.test(id)));
    } catch {
      return new Set();
    }
  }

  function saveOwnKeys() {
    const value = Array.from(state.ownKeys).join(',');
    document.cookie = value
      ? `${OWN_KEYS_COOKIE}=${encodeURIComponent(value)}; path=/; SameSite=Lax`
      : `${OWN_KEYS_COOKIE}=; path=/; max-age=0; SameSite=Lax`;
  }

  function rememberMine(job, inputDims) {
    state.mine.ids.add(job.id);
    if (typeof job.owner === 'string' && job.owner.startsWith('guest:')) state.mine.owner = job.owner;
    if (inputDims) state.mine.dims.set(job.id, inputDims);
    saveMine();
  }

  function isHost() {
    return Boolean(state.status && state.status.is_host);
  }

  function isMine(job) {
    if (typeof job.mine === 'boolean') return job.mine;
    if (isHost()) return !job.owner || job.owner === 'host';
    return state.mine.ids.has(job.id) || Boolean(state.mine.owner && job.owner === state.mine.owner);
  }

  /* ---------- Start-up ---------- */

  /** The header and the photo action bar stay on screen as the page scrolls. Their heights (which change
      with the screen width and the bar's text) go into --topbar-h and --actionbar-h, which app.css uses
      as scroll padding, so a picture reached with Tab is never left half under either bar. */
  function bindStickyRoom() {
    if (!('ResizeObserver' in window)) return;
    const root = document.documentElement;
    const topbar = document.querySelector('.topbar');
    const watch = (node, name) => {
      if (!node) return;
      new ResizeObserver(() => root.style.setProperty(name, `${Math.round(node.getBoundingClientRect().height)}px`)).observe(node);
    };
    watch(topbar, '--topbar-h');
    watch(el.actionBar, '--actionbar-h');
  }

  async function boot() {
    collectElements();
    injectSprite();
    bindTheme();
    bindNavigation();
    bindPicker();
    bindSteps();
    bindPhotos();
    bindTray();
    bindDrawer();
    bindResults();
    bindVideos();
    bindPhotorealDialog();
    bindSettings();
    bindShare();
    bindDialogs();
    bindPageDrop();
    bindPaste();
    bindStickyRoom();
    el.loginForm.addEventListener('submit', submitLogin);
    el.bootRetry.addEventListener('click', () => start({ moveFocus: true }));
    document.addEventListener('visibilitychange', updateTitle);
    if (touchOnly()) el.videoDrop.querySelector('.dz-title').textContent = 'Pick from your videos';
    if (window.UpscalerWizard) {
      wizard = window.UpscalerWizard.create({
        dialog: el.wizardDialog,
        api,
        h,
        icon,
        getGuide,
        isHost,
        providers: () => (state.status && state.status.providers) || [],
        onConnected: (providerId) => {
          if (!isHost() && providerId) {
            state.ownKeys.add(providerId);
            saveOwnKeys();
          }
          return refreshStatus().catch(() => {});
        },
        onTryIt: tryCloudProvider,
      });
    }
    if (window.ReStyles && window.ReCompare) {
      styles = createStyles();
      // Choosing a style redraws the picture's details, so the button that opened the window has been
      // replaced by the time it closes: give focus to the new one instead of losing it.
      el.stylesDialog.addEventListener('close', () => {
        if (el.stylesDialog.open) return; // opened again before this late event arrived
        const active = document.activeElement;
        const kept = active && active !== document.body && document.contains(active) && !el.stylesDialog.contains(active);
        if (kept) return;
        const again = el.drawer.open ? ($('[data-focus="styles"]', el.drawerBody) || el.drawerTitle) : trayOpener(state.stylesFor);
        (again || el.photoDrop.querySelector('.dz-button')).focus({ preventScroll: true });
      });
    }
    if (window.ReViewer) {
      viewer = window.ReViewer.create({
        dialog: el.viewerDialog, h, icon, deepzoom: window.ReDeepZoom, compare: window.ReCompare, target: viewerTarget, trap: trapFocus,
      });
    }
    if (window.ReVideoCompare && window.ReLife) {
      videoCompare = window.ReVideoCompare.createDialog({ dialog: el.videoCompareDialog, h, icon, api, life: window.ReLife, trap: trapFocus });
    }
    // For checks in the browser (and the automated stress tests): what each viewer shows right now.
    window.__resolutionDebug = Object.assign(window.__resolutionDebug || {}, {
      viewer: () => (viewer && viewer.isOpen() ? viewer.debug() : null),
      videoCompare: () => (videoCompare && videoCompare.isOpen() ? videoCompare.debug() : null),
      cardVideo: () => (cardVideo ? { playing: Boolean(cardVideo.video && !cardVideo.video.paused) } : null),
    });
    // Mobile Safari only shows a button's pressed (:active) look when the page listens for touches.
    document.addEventListener('touchstart', () => {}, { passive: true });
    applySteps();
    applyTrayPrefs();
    await start();
  }

  function collectElements() {
    const ids = {
      tabs: '.tabs', shareButton: '#share-button', shareButtonText: '#share-button-text', themeButton: '#theme-button',
      banner: '#connection-banner', bootText: '#boot-text',
      bootRetry: '#boot-retry', loginForm: '#login-form', loginPassword: '#login-password', loginError: '#login-error',
      cpuNote: '#cpu-note', cpuNoteText: '#cpu-note-text',
      photoForm: '#photo-form', photoStepFiles: '#photo-step-files', photoStepSize: '#photo-step-size',
      photoStepMode: '#photo-step-mode', photoStepFile: '#photo-step-file', photoDrop: '#photo-drop',
      photoInput: '#photo-input', photoHint: '#photo-hint', photoList: '#photo-list', photoError: '#photo-error',
      photoCount: '#photo-count', photoClear: '#photo-clear', photoClearRow: '#photo-clear-row',
      tray: '#photo-tray', traySummary: '#tray-summary', trayFold: '#tray-fold', trayFoldText: '#tray-fold-text',
      trayView: '#tray-view', trayChecking: '#tray-checking', trayCheckingText: '#tray-checking-text',
      trayBody: '#tray-body', trayFilters: '#tray-filters', trayActions: '#tray-actions', traySelectAll: '#tray-select-all',
      traySelectText: '#tray-select-text', traySuggest: '#tray-suggest', traySuggestText: '#tray-suggest-text',
      trayRemove: '#tray-remove', trayEmpty: '#tray-empty', trayMore: '#tray-more', trayStrip: '#tray-strip',
      drawer: '#photo-drawer', drawerTitle: '#drawer-title', drawerPosition: '#drawer-position', drawerBody: '#drawer-body',
      drawerPrev: '#drawer-prev', drawerNext: '#drawer-next', viewerDialog: '#viewer-dialog', videoCompareDialog: '#video-compare-dialog', resultsClear: '#results-clear',
      resultsTitle: '#results-title',
      actionBar: '#photo-action-bar',
      ownKeys: '#own-keys',
      sampleRow: '#sample-row', sampleList: '#sample-list',
      presetRow: '#preset-row', presetChips: '#preset-chips', presetHint: '#preset-hint',
      sizeChips: '#size-chips', sizeHint: '#size-hint', photoModes: '#photo-modes',
      prPanel: '#photoreal-panel', levelOptions: '#level-options', levelMixed: '#level-mixed', modeMixed: '#mode-mixed',
      creativity: '#creativity', creativityValue: '#creativity-value', stayClose: '#stay-close', stayCloseValue: '#stay-close-value',
      moreModes: '#more-modes', photoModesMore: '#photo-modes-more', cloudWrap: '#cloud-modes-wrap',
      photoModesCloud: '#photo-modes-cloud', unlock: '#unlock-cloud', unlockTitle: '#unlock-title',
      unlockBody: '#unlock-body', unlockButton: '#unlock-button', strength: '#strength', strengthValue: '#strength-value',
      strengthOption: '#strength-option', formatChips: '#format-chips', formatHint: '#format-hint', dpiChips: '#dpi-chips',
      estimate: '#estimate', estimateWarnings: '#estimate-warnings',
      photoCloudNotice: '#photo-cloud-notice', photoCloudNoticeText: '#photo-cloud-notice-text', photoStart: '#photo-start',
      photoModeNotice: '#photo-mode-notice', photoModeNoticeText: '#photo-mode-notice-text',
      photoEta: '#photo-eta', photoWhere: '#photo-where', photoUpload: '#photo-upload', photoSubmitError: '#photo-submit-error',
      videoForm: '#video-form', videoStepFile: '#video-step-file', videoStepMode: '#video-step-mode', videoDrop: '#video-drop',
      videoInput: '#video-input', videoCard: '#video-card', videoError: '#video-error', videoModes: '#video-modes',
      videoPreview: '#video-preview', videoStart: '#video-start', videoUpload: '#video-upload', videoStartHelp: '#video-start-help',
      videoSubmitError: '#video-submit-error', results: '#results', resultsNote: '#results-note',
      resultsEmpty: '#results-empty', jobList: '#job-list', jobsMore: '#jobs-more', shareBody: '#share-body',
      passwordStatus: '#password-status', passwordForm: '#password-form', newPassword: '#new-password',
      newPasswordLabel: '#new-password-label', passwordSave: '#password-save', passwordClear: '#password-clear',
      passwordFeedback: '#password-feedback', allowGuestsCloud: '#allow-guests-cloud', allowGuestsHelp: '#allow-guests-help',
      serviceList: '#service-list', servicesFeedback: '#services-feedback', defaultMode: '#default-mode',
      defaultFormat: '#default-format-options',
      defaultsFeedback: '#defaults-feedback', aboutMachine: '#about-machine', aboutVersion: '#about-version',
      aboutSamples: '#about-samples', sampleCredits: '#sample-credits',
      prDialog: '#photoreal-dialog', prDownload: '#pr-download', prProgress: '#pr-progress', prMessage: '#pr-message',
      prEta: '#pr-eta', prCancel: '#pr-cancel', prRetry: '#pr-retry', prErrorText: '#pr-error-text',
      wizardDialog: '#wizard-dialog', stylesDialog: '#styles-dialog', dropOverlay: '#drop-overlay',
      shareDialog: '#share-dialog', shareSwitch: '#share-switch', shareSwitchRow: '#share-switch-row',
      shareSwitchHelp: '#share-switch-help', shareFeedback: '#share-feedback', shareDetails: '#share-details',
      dropOverlayText: '#drop-overlay-text', toast: '#toast', announcer: '#announcer',
    };
    for (const [name, selector] of Object.entries(ids)) el[name] = $(selector);
  }

  async function injectSprite() {
    try {
      const response = await fetch('icons.svg');
      const text = await response.text();
      const svg = new DOMParser().parseFromString(text, 'image/svg+xml').documentElement;
      if (svg && svg.nodeName === 'svg') document.body.prepend(document.importNode(svg, true));
    } catch {
      /* Icons are decoration; the page works without them. */
    }
  }

  /** Opens the app. `moveFocus` puts keyboard focus on the page's heading, for when the button that
      was pressed (Try again, or Open Re:Solution on the password screen) goes away. */
  async function start({ moveFocus = false } = {}) {
    showView('loading');
    el.bootText.textContent = 'Opening Re:Solution…';
    el.bootRetry.hidden = true;
    try {
      const status = await api(API.status);
      if (status.needs_login) {
        showLogin();
        return;
      }
      applyStatus(status, true);
      route(moveFocus);
      connectEvents();
      refreshJobs().catch(() => {});
    } catch (err) {
      if (err.status === 401) return;
      el.bootText.textContent = COPY.cantReach;
      el.bootRetry.hidden = false;
    }
  }

  async function refreshStatus() {
    const status = await api(API.status);
    if (status.needs_login) {
      showLogin();
      return;
    }
    applyStatus(status);
  }

  /** Photo-real is selectable whenever this computer can run it; the download happens on first use. */
  function normaliseStatus(status) {
    const photoreal = status.photoreal;
    if (!photoreal || !Array.isArray(status.image_modes)) return status;
    const imageModes = status.image_modes.map((m) => (m.id === 'photoreal' ? { ...m, available: photoreal.supported !== false } : m));
    return { ...status, image_modes: imageModes };
  }

  function applyStatus(raw, first = false) {
    const status = normaliseStatus(raw);
    state.status = status;
    if (status.settings) state.settings = { ...(state.settings || {}), ...status.settings };
    if (first) initChoices(status);
    if (!modeIsSelectable(state.photo.mode)) state.photo.mode = pickDefaultMode(status);
    const videoModes = status.video_modes || [];
    if (videoModes.length && !videoModes.some((m) => m.id === state.videoMode && m.available !== false)) {
      const fallback = videoModes.find((m) => m.id === DEFAULTS.videoMode && m.available !== false) || videoModes.find((m) => m.available !== false);
      if (fallback) state.videoMode = fallback.id;
    }
    el.photoInput.accept = acceptAttribute(status.accept, 'images');
    el.videoInput.accept = acceptAttribute(status.accept, 'videos');
    renderHostBits();
    renderSamples();
    renderPresets();
    renderSizes();
    renderPhotoModes();
    renderVideoModes();
    renderUnlock();
    renderOwnKeys();
    renderPhotoSummary();
    renderVideoSummary();
    renderSettings();
    renderShareDialog();
    renderAbout();
    renderJobs();
  }

  function initChoices(status) {
    const settings = status.settings || {};
    const sizes = status.sizes || [];
    if (sizes.length && !sizes.some((s) => s.id === state.photo.size)) state.photo.size = sizes[0].id;
    state.photo.mode = pickDefaultMode(status);
    if (formatList().some((f) => f.id === settings.output_format)) state.photo.format = settings.output_format;
    if (GRAINS.includes(settings.grain)) state.photo.grain = settings.grain;
    const dpis = status.dpi_choices || DPI_CHOICES;
    if (!dpis.includes(state.photo.dpi)) state.photo.dpi = dpis.includes(DEFAULTS.dpi) ? DEFAULTS.dpi : dpis[0];
    for (const input of $$('input[name="grain"]')) input.checked = input.value === state.photo.grain;
  }

  /* ---------- Views and navigation ---------- */

  const ROUTES = ['photos', 'videos', 'settings', 'about'];

  function showView(name) {
    // A picture's details and the viewer belong to the page they were opened on.
    if (name !== 'photos' && el.drawer && el.drawer.open) el.drawer.close();
    if (name !== state.view && viewer && viewer.isOpen()) viewer.close();
    if (name !== state.view && videoCompare && videoCompare.isOpen()) videoCompare.close();
    if (name !== state.view) stopCardVideo();
    state.view = name;
    for (const view of $$('[data-view]')) view.hidden = view.dataset.view !== name;
    const inApp = ROUTES.includes(name);
    el.tabs.hidden = !inApp;
    el.results.hidden = !(name === 'photos' || name === 'videos');
    for (const tab of $$('.tab')) {
      if (tab.dataset.route === name) tab.setAttribute('aria-current', 'page');
      else tab.removeAttribute('aria-current');
    }
  }

  function route(moveFocus) {
    if (!state.status) return;
    let name = location.hash.replace(/^#\/?/, '');
    if (!ROUTES.includes(name)) name = 'photos';
    if (name === 'settings' && !isHost()) name = 'photos';
    showView(name);
    if (name === 'settings') loadSettings();
    if (moveFocus) {
      window.scrollTo(0, 0);
      const heading = $(`[data-view="${name}"] h1`);
      if (heading) heading.focus({ preventScroll: true });
    }
  }

  function bindNavigation() {
    window.addEventListener('hashchange', () => route(true));
  }

  function goTo(view) {
    if (state.view === view) return;
    location.hash = `#/${view}`;
    route(false);
  }

  function renderHostBits() {
    const host = isHost();
    for (const node of $$('[data-host-only]')) node.hidden = !host;
    renderShareButton();
    el.resultsNote.textContent = host
      ? 'Your results are also saved in Pictures › Upscaled.'
      : 'Results are kept for 24 hours, so download what you want to keep.';
    const hardware = state.status.hardware || {};
    const cpuOnly = Boolean(hardware.cpu_only) || hardware.device === 'cpu';
    el.cpuNote.hidden = !cpuOnly;
    if (cpuOnly) {
      el.cpuNoteText.textContent = hardware.note || 'CPU only: slow';
      el.cpuNote.title = 'This computer has no supported graphics chip, so everything runs on the processor and takes much longer.';
    }
  }

  function showLogin() {
    closeEvents();
    state.status = null;
    for (const dialog of $$('dialog[open]')) dialog.close();
    showView('login');
    showError(el.loginError, '');
    el.loginPassword.focus();
  }

  async function submitLogin(event) {
    event.preventDefault();
    const password = el.loginPassword.value;
    if (!password) {
      showError(el.loginError, 'Type the password first.');
      el.loginPassword.focus();
      return;
    }
    const button = el.loginForm.querySelector('button[type="submit"]');
    button.disabled = true;
    try {
      await api(API.login, { method: 'POST', body: { password } });
      el.loginPassword.value = '';
      await start({ moveFocus: true });
    } catch (err) {
      const wrong = err.status === 401 || err.status === 403;
      showError(el.loginError, wrong ? "That password didn't work. Check with the person who runs Re:Solution." : err.message);
      el.loginPassword.select();
    } finally {
      button.disabled = false;
    }
  }

  /* ---------- File pickers, drag and drop, paste ---------- */

  function bindPicker() {
    document.addEventListener('click', (event) => {
      const trigger = event.target.closest('[data-pick]');
      if (!trigger) return;
      if (event.target.closest('.tray, .video-card')) return;
      const input = document.getElementById(trigger.dataset.pick);
      if (input) input.click();
    });
  }

  function hasFiles(event) {
    return Boolean(event.dataTransfer && Array.from(event.dataTransfer.types || []).includes('Files'));
  }

  function canTakeFiles() {
    return Boolean(state.status) && ROUTES.includes(state.view) && !$('dialog[open]');
  }

  /** Files can be dropped anywhere: pictures go to Photos and videos to Videos, from any page. */
  function bindPageDrop() {
    let depth = 0;
    const show = (on) => {
      el.dropOverlay.hidden = !on;
      el.photoDrop.classList.toggle('is-over', on && state.view === 'photos');
      el.videoDrop.classList.toggle('is-over', on && state.view === 'videos');
    };
    document.addEventListener('dragenter', (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth += 1;
      if (!canTakeFiles()) return;
      el.dropOverlayText.textContent = state.view === 'videos' ? 'Drop to add your video' : 'Drop to add your pictures';
      show(true);
    });
    document.addEventListener('dragover', (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = canTakeFiles() ? 'copy' : 'none';
    });
    document.addEventListener('dragleave', (event) => {
      if (!hasFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) show(false);
    });
    document.addEventListener('drop', (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      depth = 0;
      show(false);
      if (!canTakeFiles()) return;
      takeFiles(Array.from(event.dataTransfer.files || []));
    });
  }

  /** Sends dropped or pasted files where they belong. */
  function takeFiles(files, { pasted = false } = {}) {
    if (!files.length) return;
    const kinds = files.map((file) => fileKind(file, state.status.accept));
    const wantsVideo = state.view === 'videos' ? kinds.includes('video') || !kinds.includes('image') : !kinds.includes('image') && kinds.includes('video');
    if (wantsVideo) {
      goTo('videos');
      addVideo(files);
      return;
    }
    goTo('photos');
    const added = addPhotos(files);
    if (pasted && added) toast(added === 1 ? 'Pasted 1 picture.' : `Pasted ${added} pictures.`);
  }

  function bindPaste() {
    document.addEventListener('paste', (event) => {
      if (!canTakeFiles()) return;
      const target = event.target;
      if (target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]')) return;
      const data = event.clipboardData;
      if (!data) return;
      let files = Array.from(data.files || []);
      if (!files.length) {
        files = Array.from(data.items || []).filter((item) => item.kind === 'file').map((item) => item.getAsFile()).filter(Boolean);
      }
      if (!files.length) return;
      event.preventDefault();
      const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' }).replace(/:/g, '.');
      const named = files.map((file, i) => {
        if (file.name && !/^image\.(png|jpe?g|gif|webp|tiff?)$/i.test(file.name)) return file;
        const ext = (/\/(\w+)/.exec(file.type || '') || [null, 'png'])[1].replace('jpeg', 'jpg');
        return new File([file], `Pasted picture ${stamp}${files.length > 1 ? `-${i + 1}` : ''}.${ext}`, { type: file.type, lastModified: Date.now() });
      });
      takeFiles(named, { pasted: true });
    });
  }

  /* ---------- Photos ---------- */

  function bindPhotos() {
    el.photoInput.addEventListener('change', () => {
      addPhotos(el.photoInput.files);
      el.photoInput.value = '';
    });
    el.sizeChips.addEventListener('change', (event) => {
      if (event.target.name !== 'photo-size') return;
      state.photo.size = event.target.value;
      state.photo.preset = null;
      renderSizes(); // keeps the chips' redraw signature in step, so a preset can move the choice back later
      renderPresets();
      renderPhotoSummary();
    });
    for (const group of [el.photoModes, el.photoModesMore, el.photoModesCloud]) {
      group.addEventListener('change', (event) => {
        if (event.target.name !== 'photo-mode') return;
        chooseModeForAll(event.target.value);
      });
    }
    el.levelOptions.addEventListener('change', (event) => {
      if (event.target.name !== 'photoreal-level') return;
      state.photo.level = event.target.value;
      state.photo.levelExplicit = true;
      state.photo.preset = null;
      for (const photo of state.photos) {
        if (photo.choice && photo.choice.mode === 'photoreal') photo.choice = { ...photo.choice, level: event.target.value };
      }
      renderPresets();
      renderPhotoModes();
      renderPhotoSummary();
    });
    const slider = (input, output, key) => {
      input.addEventListener('input', () => {
        state.photo[key] = Number(input.value);
        output.textContent = `${state.photo[key]}%`;
        renderStepSummaries();
      });
    };
    slider(el.strength, el.strengthValue, 'strength');
    slider(el.creativity, el.creativityValue, 'creativity');
    slider(el.stayClose, el.stayCloseValue, 'stayClose');
    el.photoForm.addEventListener('change', (event) => {
      if (event.target.name === 'grain') {
        state.photo.grain = event.target.value;
        state.photo.preset = null;
        renderPresets();
        renderStepSummaries();
      }
      if (event.target.name === 'format') {
        state.photo.format = event.target.value;
        state.photo.formatNote = '';
        state.photo.blockedFormat = null;
        state.photo.preset = null;
        renderPresets();
        renderPhotoSummary();
      }
      if (event.target.name === 'dpi') {
        state.photo.dpi = Number(event.target.value);
        state.photo.preset = null;
        renderPresets();
        renderPhotoSummary();
      }
    });
    el.photoForm.addEventListener('submit', (event) => {
      event.preventDefault();
      startPhotos();
    });
    el.unlockButton.addEventListener('click', () => wizard && wizard.open());
    el.photoClear.addEventListener('click', () => {
      state.confirmClear = true;
      renderClearAll(); // asks first, and moves focus to its "Remove all"
    });
  }

  /** The person picked a quality card: it applies to every picture in the batch. */
  function chooseModeForAll(mode) {
    state.photo.mode = mode;
    state.photo.explicit = true;
    state.photo.preset = null;
    for (const photo of state.photos) photo.choice = null;
    renderPresets();
    renderPhotoModes();
    renderPhotoSummary();
  }

  function newId() {
    return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
  }

  /** A picture in the tray. `status` is "checking" (the server is looking at it), "ready", or "problem"
      (Re:Solution can't use the file; `problem` says why, and it isn't sent). */
  function newItem(file, extra = {}) {
    return {
      id: newId(), name: file.name, file, url: null, sample: null,
      width: 0, height: 0, format: (/\.([a-z0-9]+)$/i.exec(file.name) || [null, ''])[1].toUpperCase(),
      hasAlpha: false, isVector: false, warnings: [], token: null, thumbUrl: null, status: 'checking', progress: 0,
      previewBroken: false, suggestion: null, choice: null, xhr: null, problem: '', ...extra,
    };
  }

  /** The pictures that will be sent: everything except files Re:Solution can't use. */
  function sendablePhotos() {
    return state.photos.filter((p) => p.status !== 'problem');
  }

  /** Adds picked, dropped or pasted files. Returns how many were added. A file Re:Solution can't use
      joins the tray as a "Needs attention" picture that says why, instead of an alert. */
  function addPhotos(fileList) {
    const notes = [];
    let added = 0;
    for (const file of Array.from(fileList || [])) {
      const kind = fileKind(file, state.status && state.status.accept);
      if (kind === 'video') {
        notes.push(`${file.name} is a video. Add videos on the Videos page.`);
        continue;
      }
      const duplicate = state.photos.some((p) => p.file && p.file.name === file.name && p.file.size === file.size && p.file.lastModified === file.lastModified);
      if (duplicate) continue;
      if (kind !== 'image') {
        state.photos.push(newItem(file, { status: 'problem', problem: "This isn't a picture Re:Solution can read." }));
      } else if (file.size > MAX_IMAGE_BYTES) {
        state.photos.push(newItem(file, { status: 'problem', problem: "It's bigger than 300 MB. Try a smaller copy." }));
      } else {
        const item = newItem(file, { url: URL.createObjectURL(file) });
        state.photos.push(item);
        queueInspect(item);
      }
      added += 1;
    }
    showError(el.photoError, notes.join(' '));
    showError(el.photoSubmitError, '');
    if (added) openStep(el.photoStepFiles);
    renderPhotoSummary();
    return added;
  }

  function addSample(sample) {
    if (state.photos.some((p) => p.sample && p.sample.id === sample.id)) return;
    const firstOne = state.photos.length === 0;
    state.photos.push({
      id: newId(), name: sample.title, file: null, url: null, sample,
      width: sample.width || 0, height: sample.height || 0, format: 'JPEG', hasAlpha: false, isVector: false, warnings: [],
      token: null, thumbUrl: sample.thumb_url, status: 'ready', progress: 1, previewBroken: false,
      suggestion: null, choice: null, xhr: null, problem: '',
    });
    // A sample shows off its use case: start from that preset, as if its chip were pressed. It isn't
    // the person's own choice, though, so pictures they add later still get their suggestions.
    const preset = firstOne && (state.status.presets || []).find((p) => p.id === sample.preset);
    if (preset) applyPreset(preset, { quiet: true, byPerson: false });
    showError(el.photoError, '');
    renderPhotoSummary();
    announce(`Added the sample picture: ${sample.title}.`);
  }

  /* Pictures are checked by the server as soon as they're added, three at a time: size, type, a
     thumbnail (browsers can't show HEIC, RAW or TIFF) and a suggested quality. One calm line says how
     far it has got ("Checking 37 of 120…") instead of a progress bar on every picture. */

  function queueInspect(item) {
    if (!state.photos.some((p) => p !== item && p.status === 'checking')) state.check.total = 0;
    state.check.total += 1;
    inspectPool(() => (state.photos.includes(item) && item.status === 'checking' ? inspectPhoto(item) : null))
      .catch(() => {})
      .finally(renderChecking);
  }

  /** The tray's "Checking 37 of 120…" line and its bar (pictures still being sent count in part). */
  function renderChecking() {
    const pending = state.photos.filter((p) => p.status === 'checking');
    if (!pending.length) {
      if (state.check.total > 1 && !el.trayChecking.hidden) {
        const attention = state.photos.filter(needsAttention).length;
        announce(`All ${state.check.total} pictures are checked.${attention ? ` ${plural(attention, 'picture needs', 'pictures need')} attention.` : ''}`);
      }
      state.check.total = 0;
      el.trayChecking.hidden = true;
      return;
    }
    const total = Math.max(state.check.total, pending.length);
    const done = total - pending.length;
    el.trayChecking.hidden = false;
    el.trayCheckingText.textContent = checkingText(done, total);
    const sending = pending.reduce((sum, p) => sum + (p.progress > 0 && p.progress < 1 ? p.progress : 0), 0);
    setProgress($('.progress', el.trayChecking), Math.max(0.02, (done + sending * 0.9) / total));
  }

  /** Repaints the checking line at most a few times a second while files are being sent. */
  function paintChecking() {
    if (state.check.paintTimer) return;
    state.check.paintTimer = window.setTimeout(() => {
      state.check.paintTimer = null;
      renderChecking();
    }, 150);
  }

  async function inspectPhoto(item) {
    if (!state.endpoints.inspect) {
      measureInBrowser(item);
      return;
    }
    const form = new FormData();
    form.append(INSPECT_FIELD, item.file, item.file.name);
    try {
      const data = await upload(API.inspect, form, (fraction) => {
        item.progress = fraction;
        paintChecking();
      }, (xhr) => { item.xhr = xhr; });
      item.xhr = null;
      if (!state.photos.includes(item)) return;
      Object.assign(item, {
        token: data.token,
        name: data.name || item.name,
        width: data.width,
        height: data.height,
        format: data.format || item.format,
        hasAlpha: Boolean(data.has_alpha),
        isVector: Boolean(data.is_vector),
        frames: data.frames,
        warnings: Array.isArray(data.warnings) ? data.warnings : [],
        thumbUrl: data.thumb_url || (data.token ? API.thumb(data.token) : null),
        suggestion: data.suggestion && data.suggestion.mode ? data.suggestion : null,
        checkedAt: Date.now(), // the server keeps it for 24 hours
        status: 'ready',
        progress: 1,
      });
      renderPhotoSummary();
    } catch (err) {
      item.xhr = null;
      if (!state.photos.includes(item)) return;
      if (missingEndpoint(err.status, err.detail)) {
        state.endpoints.inspect = false; // this copy of the server can't inspect: decode in the browser instead
        measureInBrowser(item);
        return;
      }
      if (err.status === 400 || err.status === 413 || err.status === 415 || err.status === 422) {
        // A file Re:Solution can't open stays in the tray as "Needs attention", saying why.
        if (item.url) URL.revokeObjectURL(item.url);
        // The name is shown right above the reason, so "photo.jpg couldn't be opened…" reads "It couldn't…".
        const reason = err.message.startsWith(`${item.name} `) ? `It ${err.message.slice(item.name.length + 1)}` : err.message;
        Object.assign(item, { status: 'problem', problem: reason, url: null, progress: 1 });
        renderPhotoSummary();
        return;
      }
      // Couldn't check it (network trouble, say). It can still be sent the old way when starting.
      item.warnings = [`Couldn't check this picture yet (${lowerFirst(err.message).replace(/\.$/, '')}). It will still be sent.`];
      measureInBrowser(item);
    }
  }

  function measureInBrowser(item) {
    if (!item.url) {
      item.status = 'ready';
      renderPhotoSummary();
      return;
    }
    const img = new Image();
    img.addEventListener('load', () => {
      item.width = img.naturalWidth;
      item.height = img.naturalHeight;
      item.status = 'ready';
      renderPhotoSummary();
    });
    img.addEventListener('error', () => {
      item.previewBroken = true;
      item.status = 'ready';
      renderPhotoSummary();
    });
    img.src = item.url;
  }

  /** Takes a picture out of the list (and out of the selection), stopping its check if it's under way. */
  function dropPhoto(item) {
    const index = state.photos.indexOf(item);
    if (index < 0) return -1;
    if (item.xhr) item.xhr.abort();
    if (item.url) URL.revokeObjectURL(item.url);
    state.photos.splice(index, 1);
    state.tray.selected.delete(item.id);
    if (state.tray.anchor === item.id) state.tray.anchor = null;
    return index;
  }

  function removePhoto(id) {
    const item = state.photos.find((p) => p.id === id);
    if (!item) return;
    dropPhoto(item);
    renderPhotoSummary();
    announce(`Removed ${item.name}.`);
  }

  /** "Remove selected": the pictures ticked in the tray, in one go. */
  function removeSelected() {
    if (state.uploading === 'photos') return;
    const chosen = state.photos.filter((p) => state.tray.selected.has(p.id));
    if (!chosen.length) return;
    const first = state.tray.order.indexOf(chosen[0].id);
    for (const item of chosen) dropPhoto(item);
    state.tray.selected.clear();
    state.tray.anchor = null;
    renderPhotoSummary();
    const text = chosen.length === 1 ? `Removed ${chosen[0].name}.` : `Removed ${chosen.length} pictures.`;
    announce(text);
    toast(text);
    const nextId = state.tray.order[Math.min(Math.max(0, first), state.tray.order.length - 1)];
    (trayOpener(nextId) || (el.tray.hidden ? el.photoDrop.querySelector('.dz-button') : el.traySelectAll)).focus({ preventScroll: true });
  }

  /** "Remove all" in the tray, for a batch too big to clear by hand. It asks first. */
  function renderClearAll() {
    const count = state.photos.length;
    const hadFocus = el.photoClearRow.contains(document.activeElement);
    if (count < 2 || state.tray.selected.size) state.confirmClear = false;
    el.photoClearRow.hidden = count < 2 || state.tray.selected.size > 0;
    el.photoClear.hidden = state.confirmClear;
    el.photoClear.disabled = state.uploading === 'photos';
    let confirm = $('.confirm', el.photoClearRow);
    if (!state.confirmClear) {
      if (confirm) confirm.remove();
    } else {
      const text = `Remove all ${plural(count, 'picture')}?`;
      if (!confirm) {
        confirm = h('span', { class: 'confirm' },
          h('span', { class: 'confirm-text' }),
          h('button', { type: 'button', class: 'btn btn-danger btn-sm photo-clear-yes', onclick: clearPhotos }, 'Remove all'),
          h('button', {
            type: 'button', class: 'btn btn-secondary btn-sm',
            onclick: () => {
              state.confirmClear = false;
              renderClearAll();
              el.photoClear.focus();
            },
          }, 'Keep them'));
        el.photoClearRow.append(confirm);
      }
      $('.confirm-text', confirm).textContent = text;
    }
    if (hadFocus && !el.photoClearRow.contains(document.activeElement)) {
      if (el.photoClearRow.hidden) (el.tray.hidden ? el.photoDrop.querySelector('.dz-button') : el.traySelectAll).focus();
      else (state.confirmClear ? $('.photo-clear-yes', el.photoClearRow) : el.photoClear).focus();
    }
  }

  function clearPhotos() {
    if (state.uploading === 'photos') return;
    const count = state.photos.length;
    for (const item of state.photos.slice()) dropPhoto(item);
    state.confirmClear = false;
    state.tray.shown = TRAY_PAGE;
    state.tray.filter = 'all';
    showError(el.photoError, '');
    showError(el.photoSubmitError, '');
    renderPhotoSummary();
    el.photoDrop.querySelector('.dz-button').focus();
    announce(count === 1 ? 'Removed your picture.' : `Removed all ${count} pictures.`);
  }

  /* ---------- Quality per picture ---------- */

  function modeLabel(id, kind, level) {
    return modeLabelFor(id, kind, state.status, level);
  }

  function batchChoices() {
    const p = state.photo;
    return { mode: p.mode, level: p.level, explicit: p.explicit, levelExplicit: p.levelExplicit };
  }

  function selectableModeIds() {
    if (!state.status) return [];
    const local = (state.status.image_modes || []).filter((m) => !isCloudId(m.id) && m.available !== false).map((m) => m.id);
    return [...local, ...cloudModes().map((m) => m.id)];
  }

  function itemChoice(item) {
    return choiceFor(item, batchChoices(), selectableModeIds());
  }

  /** The quality card to show as chosen: the one every picture uses, or none when they differ. */
  function checkedMode() {
    const photos = sendablePhotos();
    if (!photos.length) return state.photo.mode;
    const modes = new Set(photos.map((p) => itemChoice(p).mode));
    return modes.size === 1 ? modes.values().next().value : null;
  }

  function photorealInUse() {
    const photos = sendablePhotos();
    if (!photos.length) return state.photo.mode === 'photoreal';
    return photos.some((p) => itemChoice(p).mode === 'photoreal');
  }

  function checkedLevel() {
    const levels = new Set(sendablePhotos().map(itemChoice).filter((c) => c.mode === 'photoreal').map((c) => c.level));
    if (!levels.size) return state.photo.level;
    return levels.size === 1 ? levels.values().next().value : null;
  }

  /** "Use this" in Preview styles: for one picture, or for every picture in the batch. */
  function useStyle(itemId, choice, forAll) {
    if (!choice) return;
    if (forAll) {
      state.photo.mode = choice.mode;
      state.photo.explicit = true;
      if (choice.level) {
        state.photo.level = choice.level;
        state.photo.levelExplicit = true;
      }
      for (const photo of state.photos) photo.choice = null;
      announce(`${modeLabel(choice.mode, 'image', choice.level)} will be used for all your pictures.`);
    } else {
      const item = state.photos.find((p) => p.id === itemId);
      if (!item) return;
      item.choice = { mode: choice.mode, level: choice.level };
      announce(`${modeLabel(choice.mode, 'image', choice.level)} will be used for ${item.name}.`);
    }
    state.photo.preset = null;
    renderPresets();
    renderPhotoModes();
    renderPhotoSummary();
  }

  /** "Use the suggestion" in a picture's details: that picture gets its suggestion; the others keep theirs.
      Focus moves to the quality it now has. */
  function useSuggestion(photo) {
    const suggestion = photo.suggestion;
    if (!suggestion) return;
    const level = suggestion.mode === 'photoreal' ? (suggestion.photoreal_level || state.photo.level) : null;
    useStyle(photo.id, { mode: suggestion.mode, level }, false);
    const chosen = el.drawer.open && $(`[data-focus="mode-${CSS.escape(suggestion.mode)}"]`, el.drawerBody);
    if (chosen) chosen.focus();
  }

  /** Whether a picture uses something other than its own suggestion (which it could still get). */
  function differsFromSuggestion(photo) {
    const suggestion = photo.suggestion;
    if (!suggestion || !suggestion.mode || photo.status !== 'ready' || !selectableModeIds().includes(suggestion.mode)) return false;
    const choice = itemChoice(photo);
    const level = suggestion.mode === 'photoreal' ? suggestion.photoreal_level : null;
    return choice.mode !== suggestion.mode || Boolean(level && level !== choice.level);
  }

  /** "Use suggestions for all", or for the ticked pictures: each gets its own suggested quality. */
  function useSuggestionsInScope() {
    const picked = state.photos.filter((p) => state.tray.selected.has(p.id));
    if (picked.length) {
      let changedCount = 0;
      for (const photo of picked) {
        if (!differsFromSuggestion(photo)) continue;
        const s = photo.suggestion;
        photo.choice = { mode: s.mode, level: s.mode === 'photoreal' ? (s.photoreal_level || state.photo.level) : null };
        changedCount += 1;
      }
      announce(changedCount === 1 ? '1 picture now uses its suggestion.' : `${changedCount} pictures now use their suggestions.`);
    } else {
      state.photo.explicit = false;
      state.photo.levelExplicit = false;
      for (const photo of state.photos) photo.choice = null;
      announce('Every picture now uses its suggested quality.');
    }
    state.photo.preset = null;
    renderPresets();
    renderPhotoModes();
    renderPhotoSummary();
    if (el.traySuggest.hidden) el.traySelectAll.focus();
  }

  function photorealUsable() {
    const info = state.status && state.status.photoreal;
    return Boolean(info && info.supported !== false && info.downloaded !== false);
  }

  function createStyles() {
    return window.ReStyles.create({
      dialog: el.stylesDialog,
      api,
      h,
      icon,
      endpoint: API.previewStyles,
      cancelPath: API.cancel,
      helpers: {
        regionAt, routeChoice, modeTheme, modeIcon, setProgress, formatEta, formatApproxTime, missingEndpoint,
        routeLabel: (route) => routeLabel(route, state.status),
      },
      routes: (item) => previewRoutes(photorealUsable(), item.extraModes, selectableModeIds()),
      // The whole-picture times are for the chosen size, and Photo-real's crops follow the Creativity slider.
      previewOptions: () => ({ size: state.photo.size, creativity: round2(state.photo.creativity / 100) }),
      results: (job, routes) => previewResults(job, routes, API.file, state.status),
      photorealUsable,
      getJob: (id) => state.jobs.get(id) || null,
      onJobCreated: (job) => {
        rememberMine(job);
        upsertJob(job);
      },
      onUse: useStyle,
      onUnavailable: () => {
        state.endpoints.preview = false;
        renderDrawer({ force: true });
      },
      batchSize: () => sendablePhotos().length,
    });
  }

  function openStyles(item) {
    if (!styles) return;
    state.stylesFor = item.id;
    const current = itemChoice(item);
    styles.open({
      id: item.id,
      name: item.name,
      token: item.token,
      sampleId: item.sample ? item.sample.id : null,
      width: item.width,
      height: item.height,
      pickUrl: item.sample ? (item.sample.url || item.sample.thumb_url) : (item.thumbUrl || item.url),
      current,
      extraModes: [item.suggestion && item.suggestion.mode, current.mode],
    });
  }

  /* ---------- The picture tray (step 1) ---------- */

  /** The picture to show for an item: the server's thumbnail, or (servers that can't check pictures)
      the file itself. Nothing while it's being checked, so 100 big files aren't decoded at once. */
  function thumbSource(item) {
    if (item.thumbUrl) return item.thumbUrl;
    if (item.status === 'checking') return null;
    if (item.url && !item.previewBroken) return item.url;
    return null;
  }

  /** Where keyboard focus is inside `root`: the item's data-id and the control's data-focus name. */
  function focusedControl(root) {
    const active = document.activeElement;
    if (!active || !root.contains(active)) return null;
    const card = active.closest('[data-id]');
    const control = active.closest('[data-focus]');
    return card ? { id: card.dataset.id, control: control ? control.dataset.focus : null } : null;
  }

  /** Focus the same control in the redrawn item, or the item's main button when that control is gone. */
  function restoreFocus(root, spot) {
    if (!spot || root.contains(document.activeElement)) return;
    const card = $(`[data-id="${CSS.escape(spot.id)}"]`, root);
    if (!card) return;
    const again = (spot.control && $(`[data-focus="${CSS.escape(spot.control)}"]`, card)) || $('[data-focus="open"]', card);
    if (again) again.focus({ preventScroll: true });
  }

  /** The button that opens a picture's details, in the tray or its folded strip. */
  function trayOpener(id) {
    if (!id) return null;
    const entry = state.trayNodes.get(id);
    if (entry && document.contains(entry.node)) return $('.item-open', entry.node);
    return $(`[data-id="${CSS.escape(id)}"] button`, el.trayStrip);
  }

  /* A list of pictures is one Tab stop (the grid or list, and the folded strip): the arrow keys move
     around it (rovingTarget) and Tab moves on, so 48 pictures aren't 96 presses of Tab. Whatever gets
     focus there, by key, click or a redraw, becomes the stop. */
  const TRAY_CONTROLS = '.item-check, .item-open'; // each picture's tick box, then the picture
  const rovingStops = new WeakMap(); // list -> which control is its Tab stop

  function rovingKey(control) {
    const item = control.closest('[data-id]');
    return `${item ? item.dataset.id : ''}|${control.dataset.focus || control.className}`;
  }

  function setRovingStop(list, selector, stop) {
    rovingStops.set(list, rovingKey(stop));
    for (const control of $$(selector, list)) {
      const want = control === stop ? '0' : '-1';
      if (control.getAttribute('tabindex') !== want) control.setAttribute('tabindex', want);
    }
  }

  /** After a redraw: the stop stays on the focused control, else the one that had it, else `first`. */
  function refreshRoving(list, selector, first) {
    const controls = $$(selector, list);
    if (!controls.length) return;
    const key = rovingStops.get(list);
    const stop = (controls.includes(document.activeElement) && document.activeElement)
      || (key && controls.find((control) => rovingKey(control) === key)) || $(first, list) || controls[0];
    setRovingStop(list, selector, stop);
  }

  /** `kinds` controls per picture; `columns()` says how many pictures fit in a row right now. */
  function bindRoving(list, selector, kinds, columns) {
    list.addEventListener('focusin', (event) => {
      const control = event.target.closest(selector);
      if (control && list.contains(control)) setRovingStop(list, selector, control);
    });
    list.addEventListener('keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const from = event.target.closest(selector);
      if (!from || !list.contains(from)) return;
      const controls = $$(selector, list);
      const to = rovingTarget(controls.length, columns(), controls.indexOf(from), event.key, kinds);
      if (to == null) return;
      event.preventDefault(); // at an edge too, so the page doesn't scroll instead
      if (controls[to] !== from) controls[to].focus();
    });
  }

  /** Pictures to a row in the tray: one in the list, as many as fit across in the grid. */
  function trayColumns() {
    if (state.tray.view === 'list') return 1;
    const items = el.photoList.children;
    let across = 0;
    while (across < items.length && items[across].offsetTop === items[0].offsetTop) across += 1;
    return Math.max(1, across);
  }

  function loadTray() {
    try {
      return parseTray(window.localStorage.getItem(TRAY_KEY));
    } catch {
      return parseTray(null);
    }
  }

  function saveTray() {
    try {
      window.localStorage.setItem(TRAY_KEY, JSON.stringify({ view: state.tray.view, folded: state.tray.folded }));
    } catch {
      /* Private windows can refuse storage; the choice still holds for this visit. */
    }
  }

  function applyTrayPrefs() {
    for (const input of $$('input[name="tray-view"]', el.trayView)) input.checked = input.value === state.tray.view;
  }

  function bindTray() {
    el.photoList.addEventListener('click', (event) => {
      const card = event.target.closest('[data-id]');
      if (!card) return;
      const id = card.dataset.id;
      const box = event.target.closest('.item-check');
      if (box) {
        // Shift-click ticks (or unticks) every picture between this one and the last one clicked.
        const anchor = event.shiftKey ? state.tray.anchor : null;
        state.tray.selected = new Set(rangeSelect(Array.from(state.tray.selected), state.tray.order, anchor, id, box.checked));
        state.tray.anchor = id;
        renderTray();
        renderClearAll();
        return;
      }
      if (event.target.closest('.item-open')) openDrawer(id);
    });
    el.trayStrip.addEventListener('click', (event) => {
      const card = event.target.closest('[data-id]');
      if (card && event.target.closest('button')) openDrawer(card.dataset.id);
    });
    bindRoving(el.photoList, TRAY_CONTROLS, 2, trayColumns);
    bindRoving(el.trayStrip, 'button', 1, () => el.trayStrip.children.length); // one row
    el.trayView.addEventListener('change', (event) => {
      if (event.target.name !== 'tray-view') return;
      state.tray.view = event.target.value;
      saveTray();
      renderTray();
    });
    el.trayFold.addEventListener('click', () => setTrayFolded(!state.tray.folded));
    el.traySelectAll.addEventListener('change', () => {
      const on = el.traySelectAll.checked;
      for (const id of state.tray.order) {
        if (on) state.tray.selected.add(id);
        else state.tray.selected.delete(id);
      }
      state.tray.anchor = null;
      renderTray();
      renderClearAll();
      announce(on ? `${plural(state.tray.selected.size, 'picture')} selected.` : 'Nothing selected.');
    });
    el.traySuggest.addEventListener('click', useSuggestionsInScope);
    el.trayRemove.addEventListener('click', removeSelected);
    el.trayMore.addEventListener('click', () => {
      const firstNew = state.tray.order[state.tray.shown];
      state.tray.shown += TRAY_PAGE;
      renderTray();
      const next = trayOpener(firstNew);
      if (next) next.focus({ preventScroll: true });
    });
    el.trayBody.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !state.tray.selected.size) return;
      event.preventDefault();
      state.tray.selected.clear();
      state.tray.anchor = null;
      renderTray();
      renderClearAll();
      announce('Nothing selected.');
    });
  }

  /** "Show less" folds the tray to one strip of small pictures; "Show all" brings the grid or list back.
      (The step's own Hide, just above, folds the whole step.) */
  function setTrayFolded(folded) {
    state.tray.folded = folded;
    // The strip shows every picture (its filter chips are hidden), so the drawer steps through all of them too.
    if (folded) state.tray.filter = 'all';
    saveTray();
    renderTray();
    el.trayFold.focus({ preventScroll: true });
  }

  function setTrayFilter(filter) {
    state.tray.filter = filter;
    state.tray.shown = TRAY_PAGE;
    renderTray();
  }

  /** What the tray's summary and the sticky bar add up: one row per picture. */
  function trayRows() {
    return state.photos.map((photo) => {
      const choice = itemChoice(photo);
      const est = photo.status === 'ready' && photo.width ? estimateFor(photo) : null;
      const cloud = isCloudId(choice.mode);
      const mode = cloud ? cloudModes().find((m) => m.id === choice.mode) : null;
      return {
        status: photo.status,
        attention: needsAttention(photo),
        seconds: est ? est.seconds : null,
        file_mb: est ? est.file_mb : null,
        cloud: cloud && photo.status !== 'problem',
        cost: cloud && est ? cloudCost(est, est.out, mode && mode.cost) : null,
      };
    });
  }

  function renderTray() {
    if (!state.status) return;
    const count = state.photos.length;
    el.tray.hidden = count === 0;
    if (!count) {
      state.tray.selected.clear();
      state.tray.anchor = null;
      state.tray.order = [];
      state.tray.shown = TRAY_PAGE;
      state.trayNodes.clear();
      el.photoList.replaceChildren();
      el.trayStrip.replaceChildren();
      state.signatures.delete('strip');
      return;
    }
    const counts = filterCounts(state.photos, itemChoice);
    if (state.tray.filter !== 'all' && !counts[state.tray.filter]) state.tray.filter = 'all';
    const list = filterItems(state.photos, state.tray.filter, itemChoice);
    state.tray.order = list.map((p) => p.id);
    const totals = trayTotals(trayRows());
    const [first, ...rest] = traySummaryText(totals).split(' · ');
    // (replaceChildren would write a null out as the word "null", so the rest is only added when there is one.)
    el.traySummary.replaceChildren(h('strong', null, first), ...(rest.length ? [` · ${rest.join(' · ')}`] : []));
    el.tray.dataset.view = state.tray.view;
    el.tray.dataset.folded = String(state.tray.folded);
    el.trayFold.setAttribute('aria-expanded', String(!state.tray.folded));
    el.trayFold.setAttribute('aria-controls', state.tray.folded ? 'tray-strip' : 'tray-body');
    el.trayFoldText.textContent = state.tray.folded ? 'Show all' : 'Show less';
    el.trayView.hidden = state.tray.folded;
    el.trayBody.hidden = state.tray.folded;
    el.trayStrip.hidden = !state.tray.folded;
    if (state.tray.folded) {
      renderStrip();
      return;
    }
    renderTrayFilters(counts);
    renderTrayActions(list);
    renderTrayItems(list);
  }

  function renderTrayFilters(counts) {
    const shown = TRAY_FILTERS.filter((f) => f === 'all' || counts[f] > 0);
    el.trayFilters.hidden = shown.length < 2;
    if (!changed('trayFilters', [shown.map((f) => [f, counts[f]]), state.tray.filter])) return;
    const focused = el.trayFilters.contains(document.activeElement) ? document.activeElement.dataset.filter : null;
    const icons = { attention: 'alert', photoreal: 'sparkle', faithful: 'shield-check', people: 'user' };
    el.trayFilters.replaceChildren(...shown.map((filter) => h('button', {
      type: 'button', class: 'filter', 'data-filter': filter, 'aria-pressed': String(state.tray.filter === filter),
      'data-theme': filter === 'photoreal' || filter === 'faithful' ? filter : null,
      onclick: () => setTrayFilter(filter),
    }, icons[filter] ? icon(icons[filter]) : null, FILTER_LABELS[filter],
    h('span', { class: 'filter-count' }, num0(counts[filter])))));
    if (focused) {
      const again = $(`[data-filter="${CSS.escape(focused)}"]`, el.trayFilters) || $('[data-filter="all"]', el.trayFilters);
      if (again) again.focus();
    }
  }

  function renderTrayActions(list) {
    const selected = state.tray.selected.size;
    const ticked = list.filter((p) => state.tray.selected.has(p.id)).length;
    el.traySelectAll.checked = list.length > 0 && ticked === list.length;
    el.traySelectAll.indeterminate = ticked > 0 && ticked < list.length;
    el.traySelectAll.setAttribute('aria-label', selectAllLabel(selected, state.tray.filter, list.length));
    el.traySelectText.textContent = selected ? `${num0(selected)} selected` : 'Select all';
    const scope = selected ? state.photos.filter((p) => state.tray.selected.has(p.id)) : state.photos;
    const hadFocus = document.activeElement === el.traySuggest || document.activeElement === el.trayRemove;
    el.traySuggest.hidden = !scope.some(differsFromSuggestion);
    el.traySuggestText.textContent = !selected ? 'Use suggestions for all'
      : selected === 1 ? 'Use its suggestion' : `Use suggestions for these ${num0(selected)}`;
    el.trayRemove.hidden = !selected;
    if (hadFocus && (document.activeElement.hidden || !el.trayActions.contains(document.activeElement))) el.traySelectAll.focus();
  }

  /** Draws the first 48 pictures of the list (more with "Show more"). Each picture is drawn once and
      only redrawn when something it shows changes, so a big batch stays quick. */
  function renderTrayItems(list) {
    const page = trayPage(list, state.tray.shown);
    const spot = focusedControl(el.photoList);
    const keep = new Set();
    let previous = null;
    for (const photo of page.items) {
      const sig = itemSignature(photo);
      let entry = state.trayNodes.get(photo.id);
      if (!entry || entry.sig !== sig) {
        const node = trayItem(photo);
        if (entry && entry.node.parentNode) entry.node.replaceWith(node);
        else if (!entry) node.classList.add('is-new');
        entry = { node, sig };
        state.trayNodes.set(photo.id, entry);
      }
      const ticked = state.tray.selected.has(photo.id);
      entry.node.classList.toggle('is-selected', ticked);
      const box = $('.item-check', entry.node);
      if (box.checked !== ticked) box.checked = ticked;
      keep.add(photo.id);
      const expected = previous ? previous.nextSibling : el.photoList.firstChild;
      if (entry.node !== expected) el.photoList.insertBefore(entry.node, expected);
      previous = entry.node;
    }
    for (const [id, entry] of state.trayNodes) {
      if (!keep.has(id)) {
        entry.node.remove();
        state.trayNodes.delete(id);
      }
    }
    restoreFocus(el.photoList, spot);
    refreshRoving(el.photoList, TRAY_CONTROLS, '.item-open');
    el.photoList.classList.toggle('is-selecting', state.tray.selected.size > 0);
    el.trayMore.hidden = page.rest === 0;
    el.trayMore.textContent = moreLabel(page.rest);
    el.trayEmpty.hidden = list.length > 0;
    if (!list.length) {
      el.trayEmpty.replaceChildren(`No pictures under "${FILTER_LABELS[state.tray.filter]}". `,
        h('button', { type: 'button', class: 'link-button', onclick: () => setTrayFilter('all') }, 'Show all pictures'));
    }
  }

  /** A picture's level for people ("Balanced"), or null. */
  function levelName(choice) {
    if (choice.mode !== 'photoreal' || !choice.level) return null;
    const level = photorealLevels(state.status).find((l) => l.id === choice.level);
    return level ? level.label : null;
  }

  /** The size line ("1,024 × 768 → 8,192 × 6,144") and the time for one picture. */
  function itemFacts(photo) {
    if (photo.status === 'problem') return { dims: photo.problem || "Re:Solution can't use this file.", time: '' };
    if (photo.status === 'checking') return { dims: 'Checking…', time: '' };
    if (!photo.width) return { dims: 'No preview, but it can still be upscaled', time: '' };
    const [w, hgt] = targetSize(photo.width, photo.height, state.photo.size);
    const same = w === photo.width && hgt === photo.height;
    const est = estimateFor(photo);
    return {
      dims: same ? `${dims(photo.width, photo.height)}, already this big` : `${dims(photo.width, photo.height)} → ${dims(w, hgt)}`,
      time: est && Number.isFinite(est.seconds) ? approxLower(est.seconds) : '',
    };
  }

  function itemSignature(photo) {
    const choice = itemChoice(photo);
    return JSON.stringify([photo.status, thumbSource(photo), photo.name, itemFacts(photo), choice.mode, choice.level,
      Boolean(photo.choice && photo.choice.mode), needsAttention(photo), photo.problem]);
  }

  /** One picture in the tray. The same markup is a tile in the grid and a row in the list (app.css
      lays it out): a tick box, then one button that opens the picture's details. */
  function trayItem(photo) {
    const choice = itemChoice(photo);
    const checking = photo.status === 'checking';
    const problem = photo.status === 'problem';
    const attention = needsAttention(photo);
    const vector = photo.isVector && !problem && !checking;
    const own = Boolean(photo.choice && photo.choice.mode) && !problem && !vector;
    const src = thumbSource(photo);
    const thumb = h('span', { class: `item-thumb${checking ? ' is-loading' : ''}`, 'aria-hidden': 'true' });
    if (src) {
      const img = h('img', { src, alt: '', loading: 'lazy', decoding: 'async' });
      img.addEventListener('error', () => {
        if (img.src.startsWith('blob:')) photo.previewBroken = true;
        else photo.thumbUrl = null;
        img.replaceWith(icon('image'));
      });
      thumb.append(img);
    } else if (!checking) {
      thumb.append(icon(problem ? 'alert' : 'image'));
    }
    const facts = itemFacts(photo);
    const level = levelName(choice);
    let mode;
    if (problem) {
      mode = h('span', { class: 'item-mode tag tag-problem' }, icon('alert'), h('span', { class: 'tag-name' }, "Can't use"));
    } else if (checking) {
      mode = h('span', { class: 'item-mode tag tag-quiet' }, h('span', { class: 'tag-name' }, 'Checking…'));
    } else if (vector) {
      // Whatever quality is picked, an SVG is drawn straight at the new size, with no AI.
      mode = h('span', { class: 'item-mode tag', 'data-theme': modeTheme(VECTOR), title: VECTOR_NOTE }, icon(modeIcon(VECTOR)),
        h('span', { class: 'tag-name' }, 'Vector'));
    } else {
      mode = h('span', { class: 'item-mode tag', 'data-theme': modeTheme(choice.mode) }, icon(modeIcon(choice.mode)),
        h('span', { class: 'tag-name' }, modeLabel(choice.mode, 'image')),
        level ? h('span', { class: 'tag-level' }, ` · ${level}`) : null,
        own ? h('span', { class: 'item-own', title: 'Your pick for this picture' }, icon('pencil'), h('span', { class: 'item-own-text' }, 'Your pick')) : null);
    }
    const stateName = problem ? 'problem' : checking ? 'checking' : attention ? 'attention' : 'ready';
    const statusText = { problem: "Can't use", checking: 'Checking…', attention: 'Needs a look', ready: 'Ready' }[stateName];
    const said = [photo.name, problem ? facts.dims : checking ? 'being checked' : vector ? VECTOR_LABEL : modeLabel(choice.mode, 'image', choice.level),
      own ? 'your pick' : null, attention && !problem ? 'needs a look' : null];
    return h('li', { class: 'item', 'data-id': photo.id, 'data-status': stateName },
      h('input', { type: 'checkbox', class: 'item-check', 'aria-label': `Select ${photo.name}`, 'data-focus': 'check' }),
      h('button', {
        type: 'button', class: 'item-open', 'data-focus': 'open', 'aria-haspopup': 'dialog', title: photo.name,
        'aria-label': openLabel(said),
      },
      thumb,
      attention ? h('span', { class: 'item-flag', 'aria-hidden': 'true' }, icon('alert')) : null,
      h('span', { class: 'item-name' }, photo.name),
      h('span', { class: 'item-dims' }, facts.dims),
      mode,
      h('span', { class: 'item-time' }, facts.time),
      h('span', { class: 'item-status', 'data-state': stateName }, problem || attention ? icon('alert') : null, statusText)));
  }

  /** The folded tray: one strip of small pictures (the first 48, then "+72"). */
  function renderStrip() {
    const page = trayPage(state.photos, TRAY_PAGE);
    const sig = [page.items.map((p) => [p.id, thumbSource(p), p.status, needsAttention(p), p.name]), page.rest];
    if (!changed('strip', sig)) return;
    const focused = el.trayStrip.contains(document.activeElement) ? (document.activeElement.closest('[data-id]') || {}).dataset : null;
    const items = page.items.map((photo) => {
      const src = thumbSource(photo);
      const shot = h('span', { class: `strip-thumb${photo.status === 'checking' ? ' is-loading' : ''}`, 'aria-hidden': 'true' },
        src ? h('img', { src, alt: '', loading: 'lazy', decoding: 'async' }) : (photo.status === 'checking' ? null : icon(photo.status === 'problem' ? 'alert' : 'image')));
      return h('li', { 'data-id': photo.id },
        h('button', { type: 'button', class: 'strip-item', 'data-status': photo.status, title: photo.name, 'aria-label': `${photo.name}. Show details` },
          shot, needsAttention(photo) ? h('span', { class: 'item-flag', 'aria-hidden': 'true' }, icon('alert')) : null));
    });
    if (page.rest) {
      items.push(h('li', null, h('button', {
        type: 'button', class: 'strip-more', onclick: () => setTrayFolded(false),
      }, `+${num0(page.rest)}`, h('span', { class: 'sr-only' }, ' more. Show all pictures'))));
    }
    el.trayStrip.replaceChildren(...items);
    if (focused && focused.id) {
      const again = $(`[data-id="${CSS.escape(focused.id)}"] button`, el.trayStrip);
      if (again) again.focus({ preventScroll: true });
    }
    refreshRoving(el.trayStrip, 'button', 'button');
  }

  /* ---------- One picture's details (side drawer) ---------- */

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function bindDrawer() {
    el.drawerPrev.addEventListener('click', () => stepDrawer(-1));
    el.drawerNext.addEventListener('click', () => stepDrawer(1));
    // The drawer is a modal dialog, so the page behind it can't be reached; Tab also wraps around
    // inside it instead of leaving for the browser's own buttons.
    el.drawer.addEventListener('keydown', (event) => {
      if (event.key !== 'Tab') return;
      const controls = $$(FOCUSABLE, el.drawer).filter((node) => node.getClientRects().length > 0
        && !(node.type === 'radio' && !node.checked && $$(`input[name="${CSS.escape(node.name)}"]`, el.drawer).some((r) => r.checked)));
      const to = trapFocus(controls.length, controls.indexOf(document.activeElement), event.shiftKey);
      if (to < 0) return;
      event.preventDefault();
      controls[to].focus();
    });
    // A press that starts and ends on the dimmed page beside the panel closes it, as people expect. (A drag
    // out of the panel, selecting text say, also ends on the dialog element, so where it started decides.)
    let pressedOutside = false;
    el.drawer.addEventListener('pointerdown', (event) => {
      pressedOutside = event.target === el.drawer && outsideBox(el.drawer, event);
    });
    el.drawer.addEventListener('click', (event) => {
      const close = pressedOutside && event.target === el.drawer && outsideBox(el.drawer, event);
      pressedOutside = false;
      if (close) el.drawer.close();
    });
    el.drawer.addEventListener('close', () => {
      // The event comes a frame later: if the drawer was opened again meanwhile, it belongs to that visit.
      if (el.drawer.open) return;
      const id = state.drawerId;
      state.drawerId = null;
      state.signatures.delete('drawer');
      // After the browser's own focus return, so focus lands on the picture last shown, not the first.
      window.setTimeout(() => {
        const again = trayOpener(id) || (el.tray.hidden ? el.photoDrop.querySelector('.dz-button') : el.trayFold);
        if (again) again.focus({ preventScroll: false });
      }, 0);
    });
  }

  /** Whether a pointer event happened outside an element's box (on a modal dialog's backdrop). */
  function outsideBox(node, event) {
    const box = node.getBoundingClientRect();
    return event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom;
  }

  /** Pictures in the order the drawer steps through: the tray's (filtered) order when it holds this one. */
  function drawerOrder() {
    return state.tray.order.includes(state.drawerId) ? state.tray.order : state.photos.map((p) => p.id);
  }

  function openDrawer(id) {
    if (!state.photos.some((p) => p.id === id)) return;
    state.drawerId = id;
    renderDrawer({ force: true });
    if (!el.drawer.open) el.drawer.showModal();
    el.drawerTitle.focus({ preventScroll: true });
  }

  function stepDrawer(delta) {
    const order = drawerOrder();
    const next = order[order.indexOf(state.drawerId) + delta];
    if (!next) return;
    state.drawerId = next;
    renderDrawer({ force: true });
  }

  /** "Remove this picture": the drawer moves on to the next picture (or closes after the last one). */
  function removeFromDrawer(id) {
    const order = drawerOrder();
    const at = order.indexOf(id);
    const next = order[at + 1] || order[at - 1] || null;
    if (!next) {
      el.drawer.close();
      removePhoto(id);
      return;
    }
    state.drawerId = next; // before the redraw, so the drawer shows the next picture instead of closing
    removePhoto(id);
    renderDrawer({ force: true });
    el.drawerTitle.focus({ preventScroll: true });
  }

  function drawerSignature(photo) {
    const choice = itemChoice(photo);
    const est = photo.width ? estimateFor(photo) : null;
    return [photo.id, photo.status, thumbSource(photo), photo.name, photo.width, photo.height, photo.format, state.photo.size,
      choice, photo.choice, photo.suggestion, photo.warnings, photo.problem, est && [est.seconds, est.file_mb, est.print, est.cost_usd],
      state.photo.format, state.photo.dpi, state.endpoints.preview, selectableModeIds()];
  }

  /** Redraws the open drawer when its picture changes (checked, estimated, a new quality), keeping focus. */
  function renderDrawer({ force = false } = {}) {
    if (!state.drawerId) return;
    const photo = state.photos.find((p) => p.id === state.drawerId);
    if (!photo) {
      if (el.drawer.open) el.drawer.close();
      return;
    }
    const order = drawerOrder();
    const at = order.indexOf(photo.id);
    const filtered = order === state.tray.order && state.tray.filter !== 'all';
    el.drawerTitle.textContent = photo.name;
    el.drawerPosition.textContent = order.length > 1
      ? `${num0(at + 1)} of ${num0(order.length)}${filtered ? ` in "${FILTER_LABELS[state.tray.filter]}"` : ''}` : '';
    const active = document.activeElement;
    el.drawerPrev.disabled = at <= 0;
    el.drawerNext.disabled = at < 0 || at >= order.length - 1;
    if ((active === el.drawerPrev && el.drawerPrev.disabled) || (active === el.drawerNext && el.drawerNext.disabled)) {
      (el.drawerPrev.disabled ? (el.drawerNext.disabled ? el.drawerTitle : el.drawerNext) : el.drawerPrev).focus();
    }
    const sig = drawerSignature(photo);
    if (force) state.signatures.set('drawer', JSON.stringify(sig));
    else if (!changed('drawer', sig)) return;
    const spot = active && el.drawerBody.contains(active) ? (active.closest('[data-focus]') || {}).dataset : null;
    el.drawerBody.replaceChildren(...drawerContent(photo));
    if (spot) {
      const again = spot.focus && $(`[data-focus="${CSS.escape(spot.focus)}"]`, el.drawerBody);
      (again || el.drawerTitle).focus({ preventScroll: true });
    }
  }

  function drawerFact(iconName, label, value, detail) {
    return h('div', null, h('dt', null, icon(iconName), label), h('dd', null, value, detail ? h('small', null, detail) : null));
  }

  function drawerContent(photo) {
    const choice = itemChoice(photo);
    const checking = photo.status === 'checking';
    const problem = photo.status === 'problem';
    const parts = [];
    const src = thumbSource(photo);
    const preview = h('div', { class: `drawer-preview${checking ? ' is-loading' : ''}` });
    if (src) {
      const img = h('img', { src, alt: `${photo.name}, as it is now`, decoding: 'async' });
      img.addEventListener('error', () => img.replaceWith(icon('image')));
      preview.append(img);
    } else if (!checking) {
      preview.append(icon(problem ? 'alert' : 'image'));
    }
    if (photo.format && photo.format.length <= 5 && !problem) preview.append(h('span', { class: 'thumb-badge' }, photo.format));
    parts.push(preview);

    if (problem) {
      parts.push(h('div', { class: 'alert alert-danger' }, icon('alert'), h('p', null, photo.problem || "Re:Solution can't use this file.")),
        h('p', { class: 'muted drawer-note' }, "It won't be sent. Remove it, or add a copy saved as JPEG, PNG or TIFF instead."));
    } else if (checking) {
      parts.push(h('p', { class: 'muted drawer-note' }, 'Checking this picture. Its size, a suggested quality and the time show up here in a moment.'));
    } else if (photo.width) {
      const est = estimateFor(photo);
      const out = est ? est.out : targetSize(photo.width, photo.height, state.photo.size);
      const same = out[0] === photo.width && out[1] === photo.height;
      const facts = [drawerFact('maximize', 'Size', `${dims(out[0], out[1])} px`, same ? 'Already this big, so it keeps its size' : `Now ${dims(photo.width, photo.height)}`)];
      if (est && est.print) {
        facts.push(drawerFact('printer', 'Print size', `${est.print.inches[0].toFixed(1)} × ${est.print.inches[1].toFixed(1)} in`,
          `${est.print.cm[0].toFixed(1)} × ${est.print.cm[1].toFixed(1)} cm at ${est.print.dpi} DPI`));
      }
      if (est) facts.push(drawerFact('file', 'File', formatFileSize(est.file_mb), savedFormatName(photo, out)));
      facts.push(drawerFact('clock', 'Time', est && Number.isFinite(est.seconds) ? formatApproxTime(est.seconds) : (etaHintFor(choice) || 'Not known yet'),
        photo.isVector ? VECTOR_LABEL : modeLabel(choice.mode, 'image', choice.level)));
      if (isCloudId(choice.mode)) {
        const mode = cloudModes().find((m) => m.id === choice.mode);
        const cost = cloudCost(est, out, mode && mode.cost);
        const text = cloudCostText(cost, cloudUsesCredits(est, mode && mode.cost));
        facts.push(drawerFact('cloud', 'Cloud cost', text.charAt(0).toUpperCase() + text.slice(1), providerName(mode ? mode.providerId : '')));
      }
      parts.push(h('dl', { class: 'estimate-facts drawer-facts' }, facts));
    } else {
      parts.push(h('p', { class: 'muted drawer-note' }, "This browser can't show it, but Re:Solution can still upscale it."));
    }
    if ((photo.warnings || []).length) {
      parts.push(h('ul', { class: 'thumb-warnings drawer-warnings' }, photo.warnings.map((w) => h('li', null, icon('info'), w))));
    }
    if (!problem && !checking && photo.isVector) {
      parts.push(vectorBlock(photo));
    } else if (!problem && !checking) {
      parts.push(suggestionBlock(photo), qualityBlock(photo, choice));
      // Preview styles needs the server to have the picture: checked (token) or bundled (sample).
      const previewable = photo.token || (photo.sample && state.endpoints.inspect);
      if (styles && state.endpoints.preview && previewable && photo.width) {
        parts.push(h('div', { class: 'drawer-section' },
          h('button', { type: 'button', class: 'btn btn-secondary btn-block', 'data-focus': 'styles', onclick: () => openStyles(photo) }, icon('layers'), 'Preview styles'),
          h('p', { class: 'hint' }, 'See one spot of this picture in each style, at full size, before you choose.')));
      }
    }
    parts.push(h('div', { class: 'drawer-section drawer-remove' },
      h('button', { type: 'button', class: 'btn btn-quiet btn-danger-text', 'data-focus': 'remove', onclick: () => removeFromDrawer(photo.id) },
        icon('trash'), 'Remove this picture')));
    return parts;
  }

  /** An SVG has no quality to pick: it's drawn straight at the new size (pipeline_image), so the drawer
      says that instead of offering AI styles it would never use. */
  function vectorBlock(photo) {
    const reasons = ((photo.suggestion && photo.suggestion.reasons) || []).filter(Boolean);
    return h('div', { class: 'drawer-section drawer-suggestion', 'data-theme': modeTheme(VECTOR) },
      h('h3', { class: 'drawer-subhead' }, 'How it\'s made'),
      h('div', { class: 'thumb-tags' }, modeTag(VECTOR, 'image')),
      h('ul', { class: 'why', 'aria-label': 'Why' }, (reasons.length ? reasons : [VECTOR_NOTE]).map((r) => h('li', null, r))),
      h('p', { class: 'hint' }, 'So there\'s no quality to pick for this one.'));
  }

  /** The suggestion with its reasons ("why"), and a one-click way back to it after picking another. */
  function suggestionBlock(photo) {
    const suggestion = photo.suggestion;
    const head = h('h3', { class: 'drawer-subhead' }, 'Suggested');
    if (!suggestion) {
      return h('div', { class: 'drawer-section' }, head,
        h('p', { class: 'muted' }, 'No suggestion for this one, so it gets the quality chosen for all your pictures.'));
    }
    const level = suggestion.mode === 'photoreal' ? suggestion.photoreal_level : null;
    const section = h('div', { class: 'drawer-section drawer-suggestion', 'data-theme': modeTheme(suggestion.mode) },
      head, h('div', { class: 'thumb-tags' }, modeTag(suggestion.mode, 'image', level)),
      (suggestion.reasons || []).length ? h('ul', { class: 'why', 'aria-label': 'Why' }, suggestion.reasons.map((r) => h('li', null, r))) : null);
    if (differsFromSuggestion(photo)) {
      // Another quality was picked: the suggestion stays one click away.
      const use = modeTag(suggestion.mode, 'image', level, {
        prefix: 'Use the suggestion: ', quiet: true, tagName: 'button', type: 'button', 'data-focus': 'suggestion',
        title: 'Use the suggestion for this picture', onclick: () => useSuggestion(photo),
      });
      use.classList.add('suggestion-use');
      section.append(use);
    }
    return section;
  }

  /** The qualities a picture can have, for its own choice. */
  function qualityOptions() {
    const local = (state.status.image_modes || []).filter((m) => !isCloudId(m.id) && m.available !== false);
    local.sort((a, b) => (b.id === 'photoreal') - (a.id === 'photoreal'));
    return [...local, ...cloudModes()];
  }

  /** "Quality for this picture": its own choice wins over the one for all pictures (a badge in the tray
      says so); "Same as the other pictures" takes it back. */
  function qualityBlock(photo, choice) {
    const own = Boolean(photo.choice && photo.choice.mode);
    const group = h('div', { class: 'choice-list', role: 'radiogroup', 'aria-labelledby': 'drawer-quality-title' },
      qualityOptions().map((mode) => h('label', { class: 'choice', 'data-theme': modeTheme(mode.id) },
        h('input', { type: 'radio', name: 'drawer-mode', value: mode.id, checked: choice.mode === mode.id, 'data-focus': `mode-${mode.id}` }),
        h('span', { class: 'choice-icon', 'aria-hidden': 'true' }, icon(isCloudId(mode.id) ? 'cloud' : modeIcon(mode.id))),
        h('span', { class: 'choice-text' },
          h('span', { class: 'choice-name' }, mode.label || mode.id),
          mode.id === 'photoreal' ? null : (mode.eta_hint || mode.cost ? h('span', { class: 'choice-hint' }, mode.eta_hint || formatCost(mode.cost)) : null)))));
    group.addEventListener('change', (event) => {
      if (event.target.name === 'drawer-mode') setPhotoChoice(photo, event.target.value, null);
    });
    const section = h('div', { class: 'drawer-section' },
      h('h3', { class: 'drawer-subhead', id: 'drawer-quality-title' }, 'Quality for this picture'), group);
    if (choice.mode === 'photoreal') {
      const levels = photorealLevels(state.status);
      const levelGroup = h('div', { class: 'segmented drawer-levels', role: 'radiogroup', 'aria-labelledby': 'drawer-level-label' },
        levels.map((level) => h('label', null,
          h('input', { type: 'radio', name: 'drawer-level', value: level.id, checked: level.id === choice.level, 'data-focus': `level-${level.id}` }),
          h('span', null, level.label))));
      levelGroup.addEventListener('change', (event) => {
        if (event.target.name === 'drawer-level') setPhotoChoice(photo, 'photoreal', event.target.value);
      });
      const level = levels.find((l) => l.id === choice.level);
      section.append(h('p', { class: 'field-label drawer-level-label', id: 'drawer-level-label' }, 'How much new detail Photo-real adds'), levelGroup,
        // Only what the level does: the Time above already says how long this picture takes (or, before
        // it's measured, the level's usual time), so a second, general time would only confuse.
        level && level.description ? h('p', { class: 'hint' }, level.description) : null,
        h('p', { class: 'hint drawer-warning' }, icon('alert'), COPY.photorealWarning));
    }
    if (own) {
      section.append(h('button', { type: 'button', class: 'btn btn-quiet btn-sm', 'data-focus': 'same', onclick: () => setPhotoChoice(photo, null) },
        'Same as the other pictures'));
    } else {
      section.append(h('p', { class: 'hint' }, choice.source === 'suggested'
        ? 'It uses its suggestion. Pick another quality to change just this picture.'
        : 'It uses the quality chosen for all your pictures. Pick another to change just this picture.'));
    }
    return section;
  }

  /** A picture's own quality (null: the same as the others). */
  function setPhotoChoice(photo, mode, level) {
    if (!mode) {
      photo.choice = null;
    } else {
      const current = itemChoice(photo);
      const suggested = photo.suggestion && photo.suggestion.mode === 'photoreal' ? photo.suggestion.photoreal_level : null;
      photo.choice = {
        mode,
        level: mode === 'photoreal' ? (level || (current.mode === 'photoreal' ? current.level : null) || suggested || state.photo.level) : null,
      };
    }
    renderPresets();
    renderPhotoModes();
    renderPhotoSummary();
    const now = itemChoice(photo);
    announce(`${photo.name}: ${modeLabel(now.mode, 'image', now.level)}${mode ? '' : ', the same as the other pictures'}.`);
    if (!mode && el.drawer.open) {
      const chosen = $(`[data-focus="mode-${CSS.escape(now.mode)}"]`, el.drawerBody);
      if (chosen) chosen.focus();
    }
  }

  /* ---------- Steps that fold into one line ---------- */

  function loadSteps() {
    try {
      return parseSteps(window.localStorage.getItem(STEPS_KEY));
    } catch {
      return {};
    }
  }

  function saveSteps() {
    try {
      window.localStorage.setItem(STEPS_KEY, JSON.stringify(state.steps));
    } catch {
      /* Private windows can refuse storage; the steps still fold for this visit. */
    }
  }

  /** Each step's Hide / Change button folds it into its one-line summary or opens it again. */
  function bindSteps() {
    document.addEventListener('click', (event) => {
      const button = event.target.closest('[data-step-toggle]');
      if (!button) return;
      const step = button.closest('.step');
      const fold = !step.classList.contains('is-collapsed');
      setStepFolded(step, fold);
      state.steps[step.id] = fold;
      saveSteps();
      if (!fold && !reducedMotion()) {
        // A short fade as it opens (only when opened by hand, never as the page loads).
        const body = $('.step-body', step);
        body.classList.add('is-opening');
        body.addEventListener('animationend', () => body.classList.remove('is-opening'), { once: true });
      }
    });
  }

  function setStepFolded(step, folded) {
    step.classList.toggle('is-collapsed', folded);
    $('.step-body', step).hidden = folded;
    const button = $('[data-step-toggle]', step);
    button.setAttribute('aria-expanded', String(!folded));
    $('.step-toggle-text', button).textContent = folded ? 'Change' : 'Hide';
    const summary = $('.step-summary', step);
    summary.hidden = !folded || !summary.textContent;
  }

  function applySteps() {
    for (const step of $$('.step')) setStepFolded(step, state.steps[step.id] === true);
  }

  /** Opens a folded step (new pictures were just added to it, say), and remembers that. */
  function openStep(step) {
    if (!step || !step.classList.contains('is-collapsed')) return;
    setStepFolded(step, false);
    state.steps[step.id] = false;
    saveSteps();
  }

  function setSummary(step, text) {
    const summary = $('.step-summary', step);
    summary.textContent = text || '';
    summary.hidden = !step.classList.contains('is-collapsed') || !text;
  }

  /** The step's toggle only shows when there's something to fold (there's no point hiding the drop zone
      before a picture is in). */
  function allowFold(step, allowed) {
    const button = $('[data-step-toggle]', step);
    button.hidden = !allowed;
    if (!allowed && step.classList.contains('is-collapsed')) setStepFolded(step, false); // not remembered
  }

  function renderStepSummaries() {
    if (!state.status) return;
    const count = state.photos.length;
    const attention = state.photos.filter(needsAttention).length;
    allowFold(el.photoStepFiles, count > 0);
    setSummary(el.photoStepFiles, count
      ? [plural(count, 'picture'), attention ? `${attention} need${attention === 1 ? 's' : ''} attention` : ''].filter(Boolean).join(' · ') : '');
    const size = (state.status.sizes || []).find((s) => s.id === state.photo.size);
    const [main, sub] = splitSizeLabel(size ? size.label : state.photo.size);
    const preset = (state.status.presets || []).find((p) => p.id === state.photo.preset);
    setSummary(el.photoStepSize, `${main.replace(/^(\d)x$/i, '$1×')}${sub ? ` (${sub})` : ''}${preset ? ` · ${preset.label}` : ''}`);
    const photos = sendablePhotos().filter((p) => p.status === 'ready');
    let quality = qualitySummary(photos.map((p) => {
      const choice = itemChoice(p);
      return modeLabel(choice.mode, 'image', choice.level);
    })) || modeLabel(state.photo.mode, 'image', state.photo.mode === 'photoreal' ? state.photo.level : null);
    if (photorealInUse() && (state.photo.creativity || state.photo.stayClose)) {
      const extras = [];
      if (state.photo.creativity) extras.push(`creativity ${state.photo.creativity}%`);
      if (state.photo.stayClose) extras.push(`stay close ${state.photo.stayClose}%`);
      quality += `; ${extras.join(', ')}`;
    }
    setSummary(el.photoStepMode, quality);
    setSummary(el.photoStepFile, savedSummary({
      format: FORMAT_LABELS[state.photo.format] || state.photo.format,
      dpi: state.photo.dpi,
      grain: state.photo.grain,
      strength: state.photo.strength,
      showStrength: checkedMode() !== 'photoreal',
    }));
  }

  function setStepState(step, value) {
    step.dataset.state = value;
  }

  function renderPhotoSummary() {
    if (!state.status) return;
    const count = state.photos.length;
    const usable = sendablePhotos().length;
    setStepState(el.photoStepFiles, usable ? 'done' : 'todo');
    for (const step of [el.photoStepSize, el.photoStepMode, el.photoStepFile]) setStepState(step, usable ? 'done' : 'ready');
    el.photoDrop.classList.toggle('is-compact', count > 0);
    const words = dropzoneCopy({ touch: touchOnly(), count, pasteKeys: pasteKeys() });
    el.photoDrop.querySelector('.dz-title').textContent = words.title;
    el.photoDrop.querySelector('.dz-button-text').textContent = words.button;
    el.photoHint.textContent = words.hint;
    el.sampleRow.hidden = count > 0 || !(state.status.samples || []).length;
    renderSizeHint();
    renderFormats();
    renderPhotoModes();
    renderTray();
    renderChecking();
    renderDrawer();
    renderClearAll();
    renderStepSummaries();
    updatePhotoStart();
    scheduleEstimates();
  }

  function renderSizeHint() {
    const size = state.photo.size;
    let text = SIZE_HINTS[size] || '';
    const measured = state.photos.filter((p) => p.width);
    if (state.photos.length === 1 && measured.length === 1) {
      const photo = measured[0];
      const [w, hgt] = targetSize(photo.width, photo.height, size);
      text += w === photo.width && hgt === photo.height
        ? ' This picture is already that big, so it is saved at its own size.'
        : ` Your picture becomes ${dims(w, hgt)}.`;
    } else if (state.photos.length > 1) {
      text += ' Each picture keeps its own shape.';
    }
    el.sizeHint.textContent = text.trim();
  }

  function renderSizes() {
    const sizes = state.status.sizes || [];
    if (!changed('sizes', [sizes, state.photo.size])) return;
    const chips = [];
    let sawScale = false;
    for (const size of sizes) {
      const isScale = /x$/.test(size.id);
      if (isScale && !sawScale && chips.length) chips.push(h('span', { class: 'chip-gap', 'aria-hidden': 'true' }));
      sawScale = sawScale || isScale;
      const [main, sub] = splitSizeLabel(size.label || size.id);
      chips.push(h('label', { class: 'chip' },
        h('input', { type: 'radio', name: 'photo-size', value: size.id, checked: size.id === state.photo.size }),
        h('span', { class: 'chip-main' }, main.replace(/^(\d)x$/i, '$1×')),
        sub ? h('span', { class: 'chip-sub' }, sub) : null));
    }
    keepingFocus(el.sizeChips, () => el.sizeChips.replaceChildren(...chips));
  }

  /* ---------- Try a sample ---------- */

  function sampleUrl(sample, key, fileKey) {
    if (sample[key]) return sample[key];
    return sample[fileKey] ? `/samples/${enc(sample[fileKey])}` : null;
  }

  function samplesList() {
    return (state.status.samples || []).map((s) => ({ ...s, thumb_url: sampleUrl(s, 'thumb_url', 'thumb'), url: sampleUrl(s, 'url', 'file') }))
      .filter((s) => s.id && s.thumb_url);
  }

  function renderSamples() {
    const samples = samplesList();
    state.status.samples = samples;
    if (!changed('samples', samples)) return;
    el.sampleList.replaceChildren(...samples.map((sample) => {
      const credit = [sample.caption, sample.credit && `By ${sample.credit}`, sample.license].filter(Boolean).join('. ');
      return h('li', null, h('button', {
        type: 'button', class: 'sample', title: credit, 'aria-label': `Try the sample: ${sample.title}. ${credit}`,
        onclick: () => addSample(sample),
      },
      h('img', { src: sample.thumb_url, alt: '', loading: 'lazy', decoding: 'async' }),
      h('span', null, h('span', { class: 'sample-title' }, sample.title), h('span', { class: 'sample-credit' }, sample.license || 'Free to use'))));
    }));
    el.sampleRow.hidden = state.photos.length > 0 || !samples.length;
    renderSampleCredits(samples);
  }

  function renderSampleCredits(samples) {
    el.aboutSamples.hidden = !samples.length;
    el.sampleCredits.replaceChildren(...samples.map((s) => h('li', null,
      h('strong', null, s.title), ': ', s.caption || '', s.credit ? `. ${s.credit}` : '', '. ',
      s.source_url && /^https:\/\//i.test(s.source_url)
        ? h('a', { href: s.source_url, target: '_blank', rel: 'noopener noreferrer' }, s.license || 'Source')
        : h('span', { class: 'muted' }, s.license || ''))));
  }

  /* ---------- Use-case presets ---------- */

  function renderPresets() {
    const presets = (state.status && state.status.presets) || [];
    el.presetRow.hidden = presets.length === 0;
    if (!presets.length) return;
    const chips = presets.map((preset) => h('button', {
      type: 'button', class: 'preset', 'data-theme': modeTheme(preset.mode), 'aria-pressed': String(state.photo.preset === preset.id),
      'data-preset': preset.id, title: preset.tip || null, onclick: () => applyPreset(preset),
    }, h('span', { class: 'preset-icon', 'aria-hidden': 'true' }, icon(PRESET_ICONS[preset.id] || PRESET_ICONS[preset.icon] || 'sparkle')), preset.label));
    const focused = document.activeElement && el.presetChips.contains(document.activeElement) ? document.activeElement.dataset.preset : null;
    el.presetChips.replaceChildren(...chips);
    if (focused) {
      const again = $(`[data-preset="${CSS.escape(focused)}"]`, el.presetChips);
      if (again) again.focus();
    }
    if (!state.photo.preset) el.presetHint.textContent = '';
  }

  function applyPreset(preset, { quiet = false, byPerson = true } = {}) {
    const c = presetChoices(preset, state.status);
    if (c.size) state.photo.size = c.size;
    if (c.mode) {
      state.photo.mode = c.mode;
      if (byPerson) {
        state.photo.explicit = true;
        for (const photo of state.photos) photo.choice = null;
      }
    }
    if (c.level) {
      state.photo.level = c.level;
      if (byPerson) state.photo.levelExplicit = true;
    }
    if (c.format) {
      state.photo.format = c.format;
      state.photo.formatNote = '';
      state.photo.blockedFormat = null;
    }
    if (c.dpi) state.photo.dpi = c.dpi;
    if (c.grain) {
      state.photo.grain = c.grain;
      for (const input of $$('input[name="grain"]')) input.checked = input.value === c.grain;
    }
    state.photo.preset = preset.id;
    const parts = [];
    if (c.size) parts.push(sizeLabel(c.size));
    if (c.mode) parts.push(modeLabel(c.mode, 'image', c.level));
    if (c.format) parts.push(FORMAT_LABELS[c.format] || c.format);
    if (c.dpi) parts.push(`${c.dpi} DPI`);
    let text = parts.length ? `Set to ${parts.join(', ')}. You can still change any of these.` : '';
    if (preset.tip) text += ` Tip: ${preset.tip}`;
    el.presetHint.textContent = text.trim();
    renderPresets();
    renderSizes();
    renderPhotoModes();
    renderPhotoSummary();
    if (!quiet) announce(text);
  }

  /* ---------- Quality cards ---------- */

  function modeIsSelectable(id) {
    if (!state.status) return false;
    if (isCloudId(id)) return cloudModes().some((m) => m.id === id);
    return (state.status.image_modes || []).some((m) => m.id === id && m.available !== false);
  }

  function providerById(id) {
    return ((state.status && state.status.providers) || []).find((p) => p.id === id) || null;
  }

  function providerName(id) {
    const provider = providerById(id);
    return provider ? provider.name : id;
  }

  /** Cloud qualities the user can pick now: from image_modes and from connected providers. */
  function cloudModes() {
    const found = new Map();
    for (const mode of state.status.image_modes || []) {
      if (isCloudId(mode.id) && mode.available !== false) found.set(mode.id, { ...mode });
    }
    for (const provider of state.status.providers || []) {
      if (!provider.connected) continue;
      for (const model of provider.models || []) {
        const id = `cloud:${provider.id}:${model.id}`;
        const known = found.get(id) || {};
        found.set(id, {
          id,
          label: known.label || model.label,
          description: known.description || model.description,
          eta_hint: known.eta_hint,
          cost: model.est_cost_8k_usd,
          kind: model.kind,
        });
      }
    }
    return Array.from(found.values()).map((mode) => ({ ...mode, providerId: mode.id.split(':')[1] }));
  }

  function formatCost(usd) {
    if (usd === 0) return "Uses your plan's credits"; // a service paid from plan credits (Magnific) lists no price
    if (!Number.isFinite(usd) || usd < 0) return 'Price not published';
    return `About $${usd < 0.1 ? usd.toFixed(3).replace(/0$/, '') : usd.toFixed(2)} per 8K photo`;
  }

  function modeCard(group, mode, options = {}) {
    const selected = group === 'photo-mode' ? checkedMode() : state.videoMode;
    const kind = group === 'photo-mode' ? 'image' : 'video';
    const meta = [];
    if (options.eta || mode.eta_hint) meta.push(h('span', null, icon('clock'), options.eta || mode.eta_hint));
    if (options.cost) meta.push(h('span', null, icon('cloud'), options.cost));
    if (options.kindNote) meta.push(h('span', null, icon(options.kindIcon), options.kindNote));
    return h('label', { class: `mode${options.featured ? ' mode-featured' : ''}`, dataset: { mode: mode.id }, 'data-theme': modeTheme(mode.id, kind) },
      h('input', { type: 'radio', name: group, value: mode.id, checked: selected === mode.id, disabled: mode.available === false }),
      h('span', { class: 'mode-icon', 'aria-hidden': 'true' }, icon(options.icon || ICONS[mode.id] || 'sparkle')),
      h('span', { class: 'mode-name' }, options.label || mode.label || mode.id,
        options.badge ? h('span', { class: 'badge' }, options.badge) : null,
        options.softBadge ? h('span', { class: 'badge badge-soft' }, options.softBadge) : null),
      mode.description ? h('span', { class: 'mode-desc' }, mode.description) : null,
      meta.length ? h('span', { class: 'mode-meta' }, meta) : null,
      options.extra || null,
      h('span', { class: 'mode-check', 'aria-hidden': 'true' }, icon('check')));
  }

  function photorealExtras() {
    const info = state.status.photoreal || {};
    if (info.supported === false) {
      const reason = (info.reason || "This computer can't run Photo-real").replace(/[.\s]+$/, '');
      return [h('span', { class: 'mode-note' }, icon('info'), `${reason}, so Faithful is used instead.`)];
    }
    const extras = [h('span', { class: 'mode-warning' }, icon('alert'), COPY.photorealWarning)];
    if (info.downloaded === false) {
      extras.push(h('span', { class: 'mode-note', 'data-photoreal-note': '' }, icon('download'), h('span', { class: 'pr-inline' }, COPY.photorealFirstRun)));
    }
    return extras;
  }

  /** A "Suggested" badge on the quality cards that a picture's suggestion points to. */
  function suggestionBadge(modeId) {
    const count = state.photos.filter((p) => p.suggestion && p.suggestion.mode === modeId).length;
    return count ? 'Suggested' : null;
  }

  function renderPhotoModes() {
    const status = state.status;
    if (!status) return;
    const signature = [status.image_modes, status.providers, status.photoreal && { ...status.photoreal }, checkedMode(),
      checkedLevel(), photorealInUse(), state.photos.map((p) => p.suggestion && p.suggestion.mode)];
    if (!changed('photoModes', signature)) return;

    const local = (status.image_modes || []).filter((m) => !isCloudId(m.id));
    local.sort((a, b) => (b.id === 'photoreal') - (a.id === 'photoreal'));
    const levels = photorealLevels(status);
    const level = levels.find((l) => l.id === (checkedLevel() || state.photo.level));
    let featured = null;
    const main = [];
    const more = [];
    for (const mode of local) {
      const isPhotoreal = mode.id === 'photoreal';
      const card = modeCard('photo-mode', mode, {
        featured: isPhotoreal,
        badge: isPhotoreal && mode.available !== false ? 'Recommended' : null,
        softBadge: suggestionBadge(mode.id),
        eta: isPhotoreal && level && mode.available !== false ? `${level.eta_hint} (${level.label})` : null,
        extra: isPhotoreal ? photorealExtras() : null,
      });
      if (isPhotoreal) featured = card;
      else (MORE_MODES.has(mode.id) ? more : main).push(card);
    }

    const cloudList = cloudModes();
    const cloud = cloudList.map((mode) => modeCard('photo-mode', mode, {
      icon: 'cloud',
      softBadge: String(mode.label).includes(providerName(mode.providerId)) ? null : providerName(mode.providerId),
      cost: formatCost(mode.cost),
      kindNote: mode.kind === 'generative' ? 'Adds new detail' : mode.kind === 'faithful' ? 'Keeps the details' : null,
      kindIcon: mode.kind === 'generative' ? 'sparkle' : 'shield-check',
    }));

    // The Photo-real panel sits right under the Photo-real card and keeps its own state between renders.
    keepingFocus(el.photoModes, () => {
      for (const child of Array.from(el.photoModes.children)) if (child !== el.prPanel) child.remove();
      if (el.prPanel.parentNode !== el.photoModes) el.photoModes.append(el.prPanel);
      if (featured) el.photoModes.insertBefore(featured, el.prPanel);
      el.photoModes.append(...main);
    });
    keepingFocus(el.photoModesMore, () => el.photoModesMore.replaceChildren(...more));
    keepingFocus(el.photoModesCloud, () => el.photoModesCloud.replaceChildren(...cloud));
    el.moreModes.hidden = more.length === 0;
    if (more.some((m) => m.dataset.mode === checkedMode())) el.moreModes.open = true;
    el.cloudWrap.hidden = cloud.length === 0;
    el.modeMixed.hidden = checkedMode() !== null;
    const missing = el.photoModeNotice.dataset.provider;
    if (missing && cloudList.some((m) => m.providerId === missing)) hideMissingCloudModes();
    renderPhotorealPanel(levels);
    renderPhotorealInline();
  }

  function renderPhotorealPanel(levels) {
    const usable = (state.status.photoreal || {}).supported !== false && modeIsSelectable('photoreal');
    const show = usable && photorealInUse();
    el.prPanel.hidden = !show;
    el.strengthOption.hidden = checkedMode() === 'photoreal';
    if (!show) return;
    const current = checkedLevel();
    el.levelMixed.hidden = current !== null;
    const cards = levels.map((level) => h('label', { class: 'level' },
      h('input', { type: 'radio', name: 'photoreal-level', value: level.id, checked: level.id === current }),
      h('span', { class: 'level-name' }, level.label, level.id === DEFAULTS.photorealLevel ? h('span', { class: 'badge' }, 'Recommended') : null),
      level.description ? h('span', { class: 'level-desc' }, level.description) : null,
      level.eta_hint ? h('span', { class: 'level-eta' }, icon('clock'), level.eta_hint) : null,
      h('span', { class: 'mode-check', 'aria-hidden': 'true' }, icon('check'))));
    keepingFocus(el.levelOptions, () => el.levelOptions.replaceChildren(...cards));
  }

  function renderUnlock() {
    const providers = state.status.providers || [];
    const connected = providers.filter((p) => p.connected);
    el.unlock.hidden = providers.length === 0 || connected.length === providers.length;
    if (connected.length) {
      el.unlockTitle.textContent = 'Connect another cloud service';
      el.unlockBody.textContent = `You're connected to ${connected.map((p) => p.name).join(' and ')}. You can add another one too.`;
      el.unlockButton.textContent = 'Connect a service';
    } else {
      el.unlockTitle.textContent = 'Want the very best?';
      el.unlockBody.textContent = "Connect a paid cloud service. It's optional and costs a few cents per photo.";
      el.unlockButton.textContent = 'Unlock cloud quality';
    }
  }

  /** A guest's own keys, each with "Forget my key": the key stays in the server's memory until then (or
      until Re:Solution restarts), so on a shared phone or computer it can be forgotten when they're done. */
  function renderOwnKeys() {
    const providers = state.status.providers || [];
    if (isHost()) {
      el.ownKeys.hidden = true;
      el.ownKeys.replaceChildren();
      return;
    }
    // A key that isn't connected any more went when Re:Solution restarted.
    let pruned = false;
    for (const id of Array.from(state.ownKeys)) {
      const provider = providers.find((p) => p.id === id);
      if (provider && (!provider.connected || provider.own_key === false)) {
        state.ownKeys.delete(id);
        pruned = true;
      }
    }
    if (pruned) saveOwnKeys();
    const own = ownKeyProviders(providers, state.ownKeys);
    if (!changed('ownKeys', own.map((p) => [p.id, p.name]))) return;
    const hadFocus = el.ownKeys.contains(document.activeElement);
    el.ownKeys.hidden = own.length === 0;
    el.ownKeys.replaceChildren(...own.map((provider) => {
      const forget = h('button', {
        type: 'button', class: 'btn btn-secondary btn-sm', 'aria-label': `Forget my ${provider.name} key`,
      }, 'Forget my key');
      const item = h('li', { class: 'own-key' },
        h('p', { class: 'own-key-text' }, h('strong', null, `Your own ${provider.name} key is connected.`), ' ',
          "It's forgotten when you close your browser, and after a few hours without use. On a shared phone or computer, forget it as soon as you're done."),
        forget);
      forget.addEventListener('click', () => forgetOwnKey(provider, item, forget));
      return item;
    }));
    if (hadFocus && !el.ownKeys.contains(document.activeElement) && !el.unlock.hidden) el.unlockButton.focus();
  }

  async function forgetOwnKey(provider, item, button) {
    if (button.getAttribute('aria-disabled') === 'true') return;
    button.setAttribute('aria-disabled', 'true');
    button.classList.add('is-busy');
    const old = $('.field-error', item);
    if (old) old.remove();
    try {
      await api(API.sessionKey(provider.id), { method: 'DELETE' });
      state.ownKeys.delete(provider.id);
      saveOwnKeys();
      announce(`Your ${provider.name} key is forgotten.`);
      toast(`Your ${provider.name} key is forgotten.`);
      await refreshStatus().catch(() => {});
    } catch (err) {
      button.removeAttribute('aria-disabled');
      button.classList.remove('is-busy');
      const message = missingEndpoint(err.status, err.detail)
        ? `This copy of Re:Solution can't forget keys yet. Your key goes when Re:Solution is restarted on the main computer.`
        : err.message;
      item.append(h('p', { class: 'field-error', role: 'alert' }, message));
    }
  }

  /* ---------- File type and print resolution ---------- */

  function formatList() {
    return formatChoices(state.status);
  }

  function outputSizes() {
    return state.photos.filter((p) => p.width).map((p) => targetSize(p.width, p.height, state.photo.size));
  }

  function renderFormats() {
    const formats = formatList();
    const outs = outputSizes();
    const fit = fitFormat(state.photo.format, state.photo.blockedFormat, outs, formats);
    state.photo.format = fit.format;
    state.photo.blockedFormat = fit.blocked;
    const blockedName = fit.blocked && (FORMAT_LABELS[fit.blocked] || fit.blocked);
    const usedName = FORMAT_LABELS[fit.format] || fit.format;
    state.photo.formatNote = fit.blocked
      ? `${blockedName} can't go above ${num(formatLimit(fit.blocked, formats))} pixels, so ${usedName} is used at this size.`
      : '';
    if (!formats.some((f) => f.id === state.photo.format) && formats.length) state.photo.format = formats[0].id;
    const chips = formats.map((format) => {
      const allowed = formatAllowed(format.id, outs, formats);
      return h('label', { class: 'chip', title: allowed ? null : `Not available above ${num(formatLimit(format.id, formats))} pixels` },
        h('input', { type: 'radio', name: 'format', value: format.id, checked: format.id === state.photo.format, disabled: !allowed }),
        h('span', { class: 'chip-main' }, format.label));
    });
    if (changed('formats', [formats, state.photo.format, formats.map((f) => formatAllowed(f.id, outs, formats))])) {
      keepingFocus(el.formatChips, () => el.formatChips.replaceChildren(...chips));
    }
    const blocked = formats.filter((f) => !formatAllowed(f.id, outs, formats)).map((f) => FORMAT_LABELS[f.id] || f.id);
    const why = blocked.length ? ` ${blocked.join(' and ')} can't be this big, so ${blocked.length === 1 ? "it's" : "they're"} not offered.` : '';
    el.formatHint.textContent = `${state.photo.formatNote || FORMAT_HINTS[state.photo.format] || ''}${state.photo.formatNote ? '' : why}`.trim();

    const dpis = (state.status && state.status.dpi_choices) || DPI_CHOICES;
    if (changed('dpi', [dpis, state.photo.dpi])) {
      keepingFocus(el.dpiChips, () => el.dpiChips.replaceChildren(...dpis.map((dpi) => h('label', null,
        h('input', { type: 'radio', name: 'dpi', value: String(dpi), checked: dpi === state.photo.dpi }),
        h('span', null, String(dpi))))));
    }
  }

  /* ---------- Live estimate ---------- */

  function estimateRequest(photo) {
    const choice = itemChoice(photo);
    const out = targetSize(photo.width, photo.height, state.photo.size);
    return {
      width: photo.width,
      height: photo.height,
      size: state.photo.size,
      mode: choice.mode,
      photoreal_level: choice.level || state.photo.level,
      // What's really saved: a see-through picture becomes PNG, and "Same as original" falls back
      // the way the pipeline does when the source's type can't hold this size.
      output_format: estimateFormat(state.photo.format, photo.format, out, photo.hasAlpha),
      dpi: state.photo.dpi,
      // An SVG is drawn straight at the new size, which takes seconds whatever the quality.
      is_vector: Boolean(photo.isVector),
    };
  }

  /** The type the picture is saved as, for the readout ("File: about 40 MB · JPEG"). */
  function savedFormatName(photo, out) {
    const saved = savedFormat(state.photo.format, photo.format, out, photo.hasAlpha);
    return FORMAT_LABELS[saved] || String(saved).toUpperCase();
  }

  function estimateKey(request) {
    return JSON.stringify(request);
  }

  function estimateFor(photo) {
    if (!photo.width) return null;
    const request = estimateRequest(photo);
    const fromServer = state.estimates.get(estimateKey(request));
    if (fromServer) return withAlpha(fromServer, photo.hasAlpha);
    return localEstimate({
      width: request.width, height: request.height, size: request.size, format: request.output_format,
      dpi: request.dpi, alpha: photo.hasAlpha,
    });
  }

  function scheduleEstimates() {
    window.clearTimeout(state.estimateTimer);
    renderEstimate(true);
    state.estimateTimer = window.setTimeout(refreshEstimates, 220);
  }

  /** Asks the server about every picture size and quality it hasn't been asked about yet, a few at a
      time (pictures of the same size share one answer). */
  async function refreshEstimates() {
    const run = ++state.estimateRun;
    const pending = [];
    const asked = new Set();
    for (const photo of sendablePhotos()) {
      if (!photo.width) continue;
      const request = estimateRequest(photo);
      const key = estimateKey(request);
      if (state.estimates.has(key) || asked.has(key) || !state.endpoints.estimate) continue;
      asked.add(key);
      pending.push(estimatePool(() => (state.endpoints.estimate ? api(API.estimate, { method: 'POST', body: request }) : null))
        .then((data) => { if (data && Array.isArray(data.out)) state.estimates.set(key, data); })
        .catch((err) => { if (missingEndpoint(err.status, err.detail)) state.endpoints.estimate = false; }));
    }
    if (pending.length) await Promise.all(pending);
    if (run !== state.estimateRun) return;
    renderEstimate(false);
    if (pending.length) {
      renderTray();
      renderDrawer();
      updatePhotoStart();
    }
  }

  function renderEstimate(updating) {
    const sendable = sendablePhotos();
    const photos = sendable.filter((p) => p.width);
    el.estimate.hidden = sendable.length === 0;
    if (!sendable.length) {
      el.estimateWarnings.hidden = true;
      return;
    }
    el.estimate.classList.toggle('is-updating', Boolean(updating) && photos.length > 0);
    if (!photos.length) {
      el.estimate.replaceChildren(h('p', { class: 'muted' }, 'Checking your pictures. The size, print size and time show up here.'));
      el.estimateWarnings.hidden = true;
      return;
    }
    const rows = photos.map((photo) => ({ photo, est: estimateFor(photo), choice: itemChoice(photo) }));
    const fact = (iconName, label, value, detail) => h('div', null,
      h('dt', null, icon(iconName), label), h('dd', null, value, detail ? h('small', null, detail) : null));
    const formatName = (row) => savedFormatName(row.photo, row.est.out);
    const timeOf = (row) => {
      if (Number.isFinite(row.est.seconds)) return row.est.seconds;
      return null;
    };
    const facts = [];
    let title;
    if (rows.length === 1) {
      const { photo, est, choice } = rows[0];
      title = h('p', { class: 'estimate-title' }, 'Your picture will be', h('span', { class: 'muted' }, `${dims(photo.width, photo.height)} now`));
      facts.push(fact('maximize', 'Size', `${dims(est.out[0], est.out[1])} px`, `${est.megapixels} megapixels`));
      if (est.print) {
        facts.push(fact('printer', 'Print size', `${est.print.inches[0].toFixed(1)} × ${est.print.inches[1].toFixed(1)} in`,
          `${est.print.cm[0].toFixed(1)} × ${est.print.cm[1].toFixed(1)} cm at ${est.print.dpi} DPI`));
      }
      facts.push(fact('file', 'File', formatFileSize(est.file_mb), formatName(rows[0])));
      const seconds = timeOf(rows[0]);
      facts.push(fact('clock', 'Time', seconds != null ? formatApproxTime(seconds) : etaHintFor(choice), modeLabel(choice.mode, 'image', choice.level)));
    } else {
      title = h('p', { class: 'estimate-title' }, plural(sendable.length, 'picture'), h('span', { class: 'muted' }, 'Each keeps its own shape'));
      const biggest = rows.reduce((a, b) => (a.est.out[0] * a.est.out[1] >= b.est.out[0] * b.est.out[1] ? a : b));
      facts.push(fact('maximize', 'Biggest', `${dims(biggest.est.out[0], biggest.est.out[1])} px`, `${biggest.est.megapixels} megapixels`));
      if (biggest.est.print) {
        facts.push(fact('printer', 'Print size', `up to ${biggest.est.print.inches[0].toFixed(1)} × ${biggest.est.print.inches[1].toFixed(1)} in`,
          `${biggest.est.print.cm[0].toFixed(1)} × ${biggest.est.print.cm[1].toFixed(1)} cm at ${biggest.est.print.dpi} DPI`));
      }
      const totalMb = rows.reduce((sum, row) => sum + (row.est.file_mb || 0), 0);
      facts.push(fact('file', 'Files', formatFileSize(totalMb), 'in total'));
      const times = rows.map(timeOf);
      const total = times.every((t) => t != null) ? times.reduce((a, b) => a + b, 0) : null;
      facts.push(fact('clock', 'Time', total != null ? formatApproxTime(total) : 'Depends on the quality', total != null ? 'for all of them, one after another' : 'see the time on each quality'));
    }
    const cost = cloudCostFact(rows);
    if (cost) facts.push(fact('cloud', 'Cloud cost', cost.value, cost.detail));
    el.estimate.replaceChildren(title, h('dl', { class: 'estimate-facts' }, facts));

    const warnings = mergeWarnings(rows.map((row) => ({
      warnings: row.est.warnings, file_mb: row.est.file_mb, seconds: timeOf(row), format_label: formatName(row),
    })));
    if (state.photo.formatNote && !warnings.includes(state.photo.formatNote)) warnings.push(state.photo.formatNote);
    el.estimateWarnings.hidden = warnings.length === 0;
    el.estimateWarnings.replaceChildren(...warnings.map((w) => h('li', { class: 'alert alert-warn' }, icon('alert'), h('p', null, w))));
  }

  /** What the cloud pictures in the batch cost: each picture's price (from /api/estimate, or the
      service's 8K price scaled to its size) times how many there are. */
  function cloudCostFact(rows) {
    const cloud = rows.filter((row) => isCloudId(row.choice.mode));
    if (!cloud.length) return null;
    const entries = cloud.map((row) => {
      const mode = cloudModes().find((m) => m.id === row.choice.mode);
      return { cost: cloudCost(row.est, row.est.out, mode && mode.cost), credits: cloudUsesCredits(row.est, mode && mode.cost) };
    });
    const sent = `for ${plural(cloud.length, 'picture')} sent to the cloud`;
    if (entries.every((e) => e.credits)) return { value: "Uses your plan's credits", detail: sent };
    if (!entries.every((e) => e.credits || Number.isFinite(e.cost))) return { value: 'Price not published', detail: sent };
    const costs = entries.filter((e) => !e.credits).map((e) => e.cost);
    const total = costs.reduce((a, b) => a + b, 0);
    if (entries.some((e) => e.credits)) return { value: `about ${formatUsd(total)}`, detail: "plus your plan's credits for the rest" };
    if (total === 0) return { value: 'Nothing to pay', detail: 'nothing needs the cloud service' };
    const same = costs.every((c) => Math.abs(c - costs[0]) < 0.0005);
    let detail;
    if (cloud.length === 1) detail = 'for this picture';
    else if (same) detail = `${cloud.length} pictures × about ${formatUsd(costs[0])} each`;
    else detail = `for ${cloud.length} cloud pictures`;
    return { value: `about ${formatUsd(total)}`, detail };
  }

  function etaHintFor(choice) {
    if (choice.mode === 'photoreal') {
      const level = photorealLevels(state.status).find((l) => l.id === choice.level);
      if (level && level.eta_hint) return level.eta_hint;
    }
    const mode = isCloudId(choice.mode) ? cloudModes().find((m) => m.id === choice.mode) : (state.status.image_modes || []).find((m) => m.id === choice.mode);
    return (mode && mode.eta_hint) || '';
  }

  /* ---------- Starting ---------- */

  function selectedPhotoMode() {
    const id = checkedMode() || state.photo.mode;
    if (isCloudId(id)) return cloudModes().find((m) => m.id === id) || null;
    return (state.status.image_modes || []).find((m) => m.id === id) || null;
  }

  /** The sticky bar: how many pictures, their total time and size (and cloud cost), and the button. */
  function updatePhotoStart() {
    if (!state.status) return;
    const usable = sendablePhotos();
    const count = usable.length;
    const checking = usable.some((p) => p.status === 'checking');
    const download = activeDownloadJob();
    const needsPhotoreal = count ? photorealInUse() : state.photo.mode === 'photoreal';
    const waiting = needsPhotoreal && Boolean(download) && state.autostartPhotoreal;
    if (state.uploading !== 'photos') {
      let label = count === 1 ? 'Upscale 1 picture' : count > 1 ? `Upscale ${count} pictures` : 'Upscale pictures';
      if (checking) label = 'Checking your pictures…';
      if (waiting) label = 'Starts after the download';
      setButton(el.photoStart, { label, busy: waiting || checking, disabled: waiting || checking || Boolean(state.uploading) });
    }
    const mode = selectedPhotoMode();
    el.photoEta.textContent = '';
    el.photoCount.textContent = count ? plural0(count, 'picture') : 'No pictures yet';
    el.actionBar.classList.toggle('has-pictures', count > 0);
    if (state.photos.length) {
      el.photoEta.textContent = actionBarTotals(trayTotals(trayRows()));
      if (!count) el.photoCount.textContent = 'Nothing to send yet';
    } else if (mode) {
      const hint = etaHintFor({ mode: mode.id, level: state.photo.level });
      el.photoEta.textContent = hint ? `${modeLabel(mode.id, 'image', mode.id === 'photoreal' ? state.photo.level : null)}: ${lowerFirst(hint)}.` : '';
    }
    el.photoWhere.textContent = isHost()
      ? 'Saved to the Pictures › Upscaled folder on this computer.'
      : "Download your pictures here when they're done. They're kept for 24 hours.";
    // Each picture can have its own quality, so the notice names every service the pictures go to.
    renderCloudNotice(cloudProviderCounts(count ? usable.map((p) => itemChoice(p).mode) : [mode && mode.id]));
  }

  /** Before anything is uploaded: which services the pictures go to, and what each one keeps. */
  function renderCloudNotice(counts) {
    if (!counts.length) {
      el.photoCloudNotice.hidden = true;
      delete el.photoCloudNotice.dataset.providers;
      delete el.photoCloudNotice.dataset.complete;
      return;
    }
    const entries = counts.map((c) => ({ ...c, name: providerName(c.id) }));
    const key = JSON.stringify(entries);
    const notice = el.photoCloudNotice;
    if (!notice.hidden && notice.dataset.providers === key && notice.dataset.complete === 'true') return;
    const show = (notes) => {
      if (notice.dataset.providers !== key || notice.hidden) return; // the pictures changed meanwhile
      const [lead, ...kept] = cloudNoticeLines(entries, notes);
      el.photoCloudNoticeText.replaceChildren(lead, ...kept.map((note) => h('span', { class: 'cloud-notice-note' }, note)));
      notice.dataset.complete = String(entries.every((e) => notes[e.id]));
    };
    notice.dataset.providers = key;
    notice.hidden = false;
    show({});
    Promise.all(entries.map((e) => getGuide(e.id).then((guide) => [e.id, guide && guide.privacy_note], () => [e.id, null])))
      .then((found) => show(Object.fromEntries(found.filter(([, note]) => note))));
  }

  function photorealReady() {
    const info = state.status.photoreal;
    return !info || info.downloaded !== false;
  }

  async function startPhotos() {
    showError(el.photoSubmitError, '');
    const usable = sendablePhotos();
    if (!usable.length) {
      showError(el.photoError, state.photos.length
        ? "Re:Solution can't use the pictures in the list. Remove them and add others, or try a sample."
        : 'Add at least one picture first, or try a sample.');
      el.photoDrop.querySelector('.dz-button').focus();
      return;
    }
    if (usable.some((p) => p.status === 'checking')) return;
    if (photorealInUse() && !photorealReady()) {
      openPhotorealDialog();
      return;
    }
    await submitPhotos();
  }

  /** Sends the batch: one job per quality, and per 100 pictures (the most the server takes in one job).
      `instead` swaps Photo-real for another mode ("Use Faithful this time"). */
  async function submitPhotos(instead) {
    if (state.uploading || !sendablePhotos().length) return;
    const photos = sendablePhotos();
    const choices = photos.map((photo) => {
      const choice = itemChoice(photo);
      if (instead && choice.mode === 'photoreal') return { id: photo.id, mode: instead, level: null };
      return { id: photo.id, mode: choice.mode, level: choice.level };
    });
    const groups = groupBySettings(choices);
    const batch = { ...state.photo };
    const jobs = [];
    // The batch's card in the results gathers its jobs as they're made (one per quality, and per 100).
    const record = { id: null, jobIds: [], created: Date.now(), est: {} };
    state.mine.batches = [record, ...state.mine.batches];

    state.uploading = 'photos';
    setButton(el.photoStart, { label: 'Sending…', busy: true, disabled: true });
    renderClearAll();
    const bar = $('.progress', el.photoUpload);
    try {
      for (const group of groups) {
        const items = group.ids.map((id) => photos.find((p) => p.id === id));
        const options = imageOptionsFor(batch, group);
        for (const part of sendingPlan(items)) {
          let data;
          if (part.by === 'tokens') data = await sendStaged(part.items, options, bar);
          else if (part.by === 'files') data = await uploadFiles(part.items, options, bar);
          else data = await submitSample(part.items[0], options, bar);
          const seconds = part.items.map((p) => (estimateFor(p) || {}).seconds);
          const job = afterSubmit(data, part.items);
          record.id = record.id || job.id;
          record.jobIds.push(job.id);
          if (seconds.every(Number.isFinite)) record.est[job.id] = seconds.reduce((a, b) => a + b, 0);
          jobs.push(job);
        }
      }
    } catch (err) {
      showError(el.photoSubmitError, jobs.length
        ? `${err.message} The pictures that were sent are in the queue; the rest are still here.`
        : err.message);
    } finally {
      state.uploading = null;
      el.photoUpload.hidden = true;
    }
    if (!jobs.length) state.mine.batches = state.mine.batches.filter((b) => b !== record);
    if (jobs.length) {
      state.photo = choicesAfterSending(state.photo, sendablePhotos().length);
      saveMine();
      // The settings steps were used: from now on they fold into one line each (unless opened by hand).
      state.steps = foldAfterSending(state.steps);
      saveSteps();
      applySteps();
      for (const job of jobs) upsertJob(job, { isNew: true });
      const sent = jobs.reduce((n, job) => n + Math.max(1, (job.inputs || []).length), 0);
      announce(sent === 1 ? 'Your picture is in the queue.' : `Your ${sent} pictures are in the queue.`);
      revealJob(jobs[0].id);
    }
    renderPhotoSummary();
  }

  /** Remembers a new job, frees its pictures from the list, and keeps their sizes for the result labels. */
  function afterSubmit(data, items) {
    const job = data && (data.job || data);
    if (!job || !job.id) throw new ApiError(COPY.generic, 0, '');
    rememberMine(job, items.map((p) => [p.width, p.height]));
    const cover = items.find((p) => thumbSource(p));
    const coverThumb = cover ? thumbSource(cover) : null;
    if (coverThumb) state.jobThumbs.set(job.id, coverThumb);
    for (const item of items) {
      const index = state.photos.indexOf(item);
      if (index >= 0) state.photos.splice(index, 1);
      state.tray.selected.delete(item.id);
      // Free each picture's in-memory copy unless the job's card still shows it (no server thumbnail).
      if (item.url && item.url !== coverThumb) URL.revokeObjectURL(item.url);
    }
    if (state.drawerId && items.some((item) => item.id === state.drawerId) && el.drawer.open) el.drawer.close();
    return job;
  }

  /** Pictures the server already checked go by their token. If it has let go of them (it keeps them
      for 24 hours), they're sent again as files, which this page still has. */
  async function sendStaged(items, options, bar) {
    try {
      return await api(API.images, { method: 'POST', body: { tokens: items.map((p) => p.token), options } });
    } catch (err) {
      if (!stagedGone(err.status, err.detail) || !items.every((p) => p.file)) throw err;
      for (const item of items) {
        item.token = null;
        item.thumbUrl = null; // its server thumbnail went with it
      }
      return uploadFiles(items, options, bar);
    }
  }

  async function uploadFiles(files, options, bar) {
    const form = new FormData();
    for (const photo of files) form.append(IMAGE_UPLOAD_FIELD, photo.file, photo.file.name);
    form.append('options', JSON.stringify(options));
    el.photoUpload.hidden = false;
    setProgress(bar, 0.01);
    return upload(API.images, form, (fraction) => {
      setProgress(bar, fraction);
      setButton(el.photoStart, { label: fraction < 1 ? `Sending… ${Math.round(fraction * 100)}%` : 'Almost there…', busy: true, disabled: true });
    });
  }

  /** A bundled sample is copied by the server; older servers get the file sent like any other. */
  async function submitSample(item, options, bar) {
    if (state.endpoints.sample) {
      try {
        return await api(API.imagesSample, { method: 'POST', body: { sample_id: item.sample.id, options } });
      } catch (err) {
        if (!missingEndpoint(err.status, err.detail)) throw err;
        state.endpoints.sample = false;
      }
    }
    const response = await fetch(item.sample.url, { credentials: 'same-origin' }).catch(() => null);
    if (!response || !response.ok) throw new ApiError("Couldn't open the sample picture. Try again.", 0, '');
    const blob = await response.blob();
    const name = baseName(item.sample.url) || `${item.sample.id}.jpg`;
    return uploadFiles([{ file: new File([blob], name, { type: blob.type || 'image/jpeg' }) }], options, bar);
  }

  function tryCloudProvider(providerId) {
    location.hash = '#/photos';
    refreshStatus().catch(() => {}).then(() => {
      if (!state.status) return;
      const mode = preferredCloudMode(cloudModes(), providerId);
      if (!mode) {
        showMissingCloudModes(providerId);
        return;
      }
      hideMissingCloudModes();
      chooseModeForAll(mode.id);
      state.signatures.delete('photoModes');
      renderPhotoModes();
      const input = $(`input[name="photo-mode"][value="${CSS.escape(mode.id)}"]`);
      if (input) {
        input.closest('.mode').scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
        input.focus({ preventScroll: true });
      }
    });
  }

  /** Shown when a service was just connected but the server lists none of its qualities for this viewer. */
  function showMissingCloudModes(providerId) {
    const name = providerName(providerId);
    el.photoModeNoticeText.textContent = isHost()
      ? `${name} is connected, but its qualities didn't show up here. Reload the page to try again.`
      : `${name} is connected, but its qualities didn't show up here. Reload the page, and if they're still missing, ask the person who runs Re:Solution.`;
    el.photoModeNotice.dataset.provider = providerId;
    el.photoModeNotice.hidden = false;
    el.photoModeNotice.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
  }

  function hideMissingCloudModes() {
    el.photoModeNotice.hidden = true;
    delete el.photoModeNotice.dataset.provider;
  }

  /* ---------- First-run Photo-real download ---------- */

  function bindPhotorealDialog() {
    el.prDownload.addEventListener('click', startPhotorealDownload);
    el.prRetry.addEventListener('click', startPhotorealDownload);
    el.prCancel.addEventListener('click', stopPhotorealDownload);
    for (const button of $$('[data-use-faithful]', el.prDialog)) {
      button.addEventListener('click', () => {
        el.prDialog.close();
        submitPhotos('faithful');
      });
    }
  }

  function activeDownloadJob() {
    for (const job of state.jobs.values()) if (job.kind === 'download' && isActive(job)) return job;
    return null;
  }

  /** Shows one part of the Photo-real window. When the button that had focus goes away with its part
      ("Download now", say), focus moves to the first button of the part now shown. */
  function setPhotorealPart(part) {
    const before = $$('.pr-part', el.prDialog).find((node) => !node.hidden);
    for (const node of $$('.pr-part', el.prDialog)) node.hidden = node.dataset.part !== part;
    const active = document.activeElement;
    const lost = !active || active === document.body || (el.prDialog.contains(active) && Boolean(active.closest('[hidden]')));
    if (!el.prDialog.open || !lost || (before && before.dataset.part === part)) return;
    const shown = $(`.pr-part[data-part="${part}"]`, el.prDialog);
    const next = shown && $('button:not([disabled])', shown);
    if (next) next.focus();
  }

  function openPhotorealDialog() {
    const download = activeDownloadJob();
    if (download) {
      state.downloadJobId = download.id;
      setPhotorealPart('progress');
      renderPhotorealProgress(download);
    } else {
      setPhotorealPart('ask');
    }
    if (!el.prDialog.open) el.prDialog.showModal();
  }

  async function startPhotorealDownload() {
    state.autostartPhotoreal = true;
    const existing = activeDownloadJob();
    if (existing) {
      state.downloadJobId = existing.id;
      setPhotorealPart('progress');
      renderPhotorealProgress(existing);
      updatePhotoStart();
      return;
    }
    setButton(el.prDownload, { busy: true, disabled: true });
    try {
      const data = await api(API.photorealDownload, { method: 'POST' });
      const job = data && (data.job || data);
      setPhotorealPart('progress');
      if (job && job.id) {
        state.downloadJobId = job.id;
        rememberMine(job);
        upsertJob(job);
        renderPhotorealProgress(job);
      }
    } catch (err) {
      state.autostartPhotoreal = false;
      el.prErrorText.textContent = err.message;
      setPhotorealPart('error');
    } finally {
      setButton(el.prDownload, { busy: false, disabled: false });
      updatePhotoStart();
    }
  }

  async function stopPhotorealDownload() {
    const job = activeDownloadJob();
    state.autostartPhotoreal = false;
    if (!job) {
      setPhotorealPart('ask');
      return;
    }
    el.prCancel.disabled = true;
    try {
      await api(API.cancel(job.id), { method: 'POST' });
      setPhotorealPart('ask');
    } catch (err) {
      el.prMessage.textContent = err.message;
    } finally {
      el.prCancel.disabled = false;
      updatePhotoStart();
    }
  }

  function renderPhotorealProgress(job) {
    setProgress(el.prProgress, job.progress);
    el.prMessage.textContent = job.status === 'queued' ? 'Waiting for the current job to finish first.' : (job.message || 'Starting the download…');
    el.prEta.textContent = formatEta(job.eta_seconds);
  }

  function renderPhotorealInline() {
    const note = $('.pr-inline', el.photoModes);
    if (!note) return;
    const job = activeDownloadJob();
    note.textContent = job
      ? `Downloading Photo-real: ${Math.floor(clamp01(job.progress) * 100)}% done.`
      : COPY.photorealFirstRun;
  }

  function onDownloadJob(job, previous) {
    if (isActive(job)) state.downloadJobId = job.id;
    if (job.id !== state.downloadJobId) return;
    if (el.prDialog.open) {
      if (isActive(job)) {
        setPhotorealPart('progress');
        renderPhotorealProgress(job);
      } else if (job.status === 'failed') {
        el.prErrorText.textContent = job.error || 'The download stopped.';
        setPhotorealPart('error');
      } else if (job.status === 'cancelled') {
        setPhotorealPart('ask');
      }
    }
    renderPhotorealInline();
    const finishedNow = job.status === 'done' && ((previous && previous.status !== 'done') || (!previous && state.autostartPhotoreal));
    if (finishedNow) photorealDownloaded();
    if (job.status === 'failed' || job.status === 'cancelled') state.autostartPhotoreal = false;
    updatePhotoStart();
  }

  async function photorealDownloaded() {
    if (state.status && state.status.photoreal) state.status.photoreal.downloaded = true;
    await refreshStatus().catch(() => {});
    if (el.prDialog.open) el.prDialog.close();
    if (!state.autostartPhotoreal) return;
    state.autostartPhotoreal = false;
    if (state.photos.length && photorealInUse()) {
      announce('Photo-real is ready. Starting your photos.');
      submitPhotos();
    }
  }

  /* ---------- Videos ---------- */

  function bindVideos() {
    el.videoInput.addEventListener('change', () => {
      addVideo(el.videoInput.files);
      el.videoInput.value = '';
    });
    el.videoModes.addEventListener('change', (event) => {
      if (event.target.name !== 'video-mode') return;
      state.videoMode = event.target.value;
      renderVideoSummary();
    });
    el.videoForm.addEventListener('submit', (event) => {
      event.preventDefault();
      submitVideo(null);
    });
    el.videoPreview.addEventListener('click', () => submitVideo(PREVIEW_SECONDS));
  }

  function addVideo(fileList) {
    const files = Array.from(fileList || []);
    const accept = state.status && state.status.accept;
    const file = files.find((f) => fileKind(f, accept) === 'video');
    showError(el.videoSubmitError, '');
    if (!file) {
      const first = files[0];
      showError(el.videoError, first && fileKind(first, accept) === 'image'
        ? `${first.name} is a photo. Add photos on the Photos page.`
        : "That file doesn't look like a video.");
      return;
    }
    if (state.video) URL.revokeObjectURL(state.video.url);
    state.video = { file, url: URL.createObjectURL(file), width: 0, height: 0, duration: 0, broken: false };
    openStep(el.videoStepFile);
    showError(el.videoError, files.length > 1 ? 'One video at a time, so only the first one was added.' : '');
    renderVideoCard();
    renderVideoSummary();
  }

  function removeVideo() {
    if (state.video) URL.revokeObjectURL(state.video.url);
    state.video = null;
    renderVideoCard();
    renderVideoSummary();
    el.videoDrop.querySelector('.dz-button').focus();
  }

  function renderVideoCard() {
    const video = state.video;
    el.videoCard.hidden = !video;
    el.videoDrop.hidden = Boolean(video);
    if (!video) {
      el.videoCard.replaceChildren();
      return;
    }
    const player = h('video', { muted: true, playsinline: true, preload: 'metadata', src: video.url, 'aria-hidden': 'true' });
    player.muted = true;
    const frame = h('div', { class: 'video-frame' }, player);
    player.addEventListener('loadedmetadata', () => {
      video.width = player.videoWidth;
      video.height = player.videoHeight;
      video.duration = player.duration;
      try { player.currentTime = Math.min(0.1, (player.duration || 0) / 2); } catch { /* Some formats can't seek here. */ }
      renderVideoInfo();
    });
    player.addEventListener('error', () => {
      video.broken = true;
      frame.replaceChildren(icon('video'));
      renderVideoInfo();
    });
    el.videoCard.replaceChildren(frame, h('div', { class: 'video-info', id: 'video-info' }));
    renderVideoInfo();
  }

  function renderVideoInfo() {
    const video = state.video;
    const info = $('#video-info');
    if (!video || !info) return;
    const facts = [formatBytes(video.file.size)];
    if (video.duration && Number.isFinite(video.duration)) facts.unshift(formatDuration(video.duration));
    let size = null;
    if (video.width) {
      const [w, hgt] = videoOutputSize(video.width, video.height);
      size = h('p', { class: 'thumb-dims' }, `${dims(video.width, video.height)} becomes `, h('strong', null, `${dims(w, hgt)} (4K)`));
    } else if (video.broken) {
      size = h('p', { class: 'muted' }, "This browser can't show a preview, but Re:Solution can still work on it.");
    }
    info.replaceChildren(
      h('p', { class: 'result-name', title: video.file.name }, video.file.name),
      h('p', { class: 'muted' }, facts.join(', ')),
      size,
      h('button', { type: 'button', class: 'btn btn-secondary btn-sm', onclick: removeVideo }, icon('x'), 'Choose a different video'));
    renderVideoSummary(); // the folded step's summary shows the clip's length once it's known
  }

  /** A clip of 3 seconds or less is upscaled in one go, so the preview button steps aside. */
  function renderVideoPreviewOffer() {
    const offer = offerVideoPreview(state.video ? state.video.duration : NaN);
    el.videoPreview.hidden = !offer;
    el.videoStartHelp.textContent = offer
      ? 'Try the preview first to check the look before the long run.'
      : "This video is only a few seconds long, so it's quickest to upscale all of it.";
  }

  function renderVideoModes() {
    const modes = state.status.video_modes || [];
    if (!changed('videoModes', [modes, state.videoMode])) return;
    const cards = modes.map((mode) => modeCard('video-mode', mode, {
      label: VIDEO_LABELS[mode.id] || mode.label,
      badge: mode.id === DEFAULTS.videoMode ? 'Recommended' : null,
      extra: mode.id === 'faithful' ? [h('span', { class: 'mode-warning' }, icon('alert'), COPY.videoBestWarning)] : null,
    }));
    keepingFocus(el.videoModes, () => el.videoModes.replaceChildren(...cards));
  }

  function renderVideoSummary() {
    if (!state.status) return;
    const has = Boolean(state.video);
    setStepState(el.videoStepFile, has ? 'done' : 'todo');
    setStepState(el.videoStepMode, has ? 'done' : 'ready');
    allowFold(el.videoStepFile, has);
    const length = has && Number.isFinite(state.video.duration) && state.video.duration > 0 ? ` · ${formatDuration(state.video.duration)}` : '';
    setSummary(el.videoStepFile, has ? `${state.video.file.name}${length}` : '');
    setSummary(el.videoStepMode, VIDEO_LABELS[state.videoMode] || modeLabel(state.videoMode, 'video'));
    if (state.uploading !== 'video') {
      setButton(el.videoPreview, { busy: false, disabled: Boolean(state.uploading) });
      setButton(el.videoStart, { busy: false, disabled: Boolean(state.uploading) });
    }
    renderVideoPreviewOffer();
  }

  async function submitVideo(previewSeconds) {
    showError(el.videoSubmitError, '');
    if (!state.video) {
      showError(el.videoError, 'Add a video first.');
      el.videoDrop.querySelector('.dz-button').focus();
      return;
    }
    if (state.uploading) return;
    const video = state.video;
    const form = new FormData();
    form.append(VIDEO_UPLOAD_FIELD, video.file, video.file.name);
    form.append('options', JSON.stringify({ mode: state.videoMode, preview_seconds: previewSeconds, strength: 1 }));

    const button = previewSeconds ? el.videoPreview : el.videoStart;
    const other = previewSeconds ? el.videoStart : el.videoPreview;
    state.uploading = 'video';
    setButton(button, { busy: true, disabled: true });
    setButton(other, { disabled: true });
    el.videoUpload.hidden = false;
    const bar = $('.progress', el.videoUpload);
    setProgress(bar, 0.01);
    try {
      const data = await upload(API.videos, form, (fraction) => setProgress(bar, fraction));
      const job = data && (data.job || data);
      if (!job || !job.id) throw new ApiError(COPY.generic, 0);
      rememberMine(job);
      upsertJob(job, { isNew: true });
      announce(previewSeconds ? 'Your 3-second preview is in the queue.' : 'Your video is in the queue.');
      // Keep the video after a preview so the whole run is one click away.
      if (!previewSeconds) {
        URL.revokeObjectURL(video.url);
        state.video = null;
        renderVideoCard();
      }
      revealJob(job.id);
    } catch (err) {
      showError(el.videoSubmitError, err.message);
    } finally {
      state.uploading = null;
      el.videoUpload.hidden = true;
      renderVideoSummary();
    }
  }

  /* ---------- Live updates ---------- */

  /** The whole job list again. Jobs the server no longer has (a guest's, tidied away after a day, or all
      of them after a restart) are dropped, unless this page heard of them after it asked. */
  async function refreshJobs() {
    const askedAt = Date.now();
    const data = await api(API.jobs);
    const list = Array.isArray(data) ? data : (data && data.jobs) || [];
    const known = Array.from(state.jobs.keys()).map((id) => ({ id, seenAt: state.seen.get(id) || 0 }));
    for (const id of jobsToDrop(known, list.map((job) => job.id), askedAt)) {
      state.jobs.delete(id);
      state.seen.delete(id);
    }
    for (const job of list) upsertJob(job, { render: false });
    renderJobs();
    setConnected(true);
  }

  function connectEvents() {
    closeEvents();
    if (!('EventSource' in window)) {
      startPolling();
      return;
    }
    const source = new EventSource(API.events);
    state.events = source;
    const parse = (event) => {
      try { return JSON.parse(event.data); } catch { return null; }
    };
    source.addEventListener('job', (event) => {
      const job = parse(event);
      if (job) upsertJob(job.job || job);
    });
    // A status event is only a nudge. Its body may be someone else's view (the host's
    // settings, say), so this viewer's status is always fetched fresh instead.
    source.addEventListener('status', queueStatusRefresh);
    source.addEventListener('message', (event) => {
      const data = parse(event);
      if (data && data.type === 'job' && data.job) upsertJob(data.job);
    });
    source.addEventListener('open', () => {
      stopPolling();
      refreshJobs().catch(() => {});
    });
    source.addEventListener('error', () => {
      state.connectionLost = true; // the app may have restarted: sharing and guests' own keys don't survive that
      startPolling();
      if (source.readyState === EventSource.CLOSED) {
        window.setTimeout(() => { if (state.events === source && state.status) connectEvents(); }, 5000);
      }
    });
  }

  function queueStatusRefresh() {
    if (state.statusRefreshQueued) return;
    state.statusRefreshQueued = true;
    window.setTimeout(() => {
      state.statusRefreshQueued = false;
      if (state.status) refreshStatus().catch(() => {});
    }, 250);
  }

  function closeEvents() {
    if (state.events) state.events.close();
    state.events = null;
    stopPolling();
  }

  function startPolling() {
    if (state.pollTimer) return;
    state.pollTimer = window.setInterval(() => {
      refreshJobs().catch(() => setConnected(false));
    }, 3000);
  }

  function stopPolling() {
    if (state.pollTimer) window.clearInterval(state.pollTimer);
    state.pollTimer = null;
    setConnected(true);
  }

  /** Shows or hides the "can't reach" banner. When the connection comes back after dropping, the status
      is fetched again: Re:Solution may have restarted, which turns sharing off and forgets guests' own keys. */
  function setConnected(ok) {
    el.banner.hidden = ok;
    if (!ok) {
      state.connectionLost = true;
    } else if (state.connectionLost) {
      state.connectionLost = false;
      queueStatusRefresh();
    }
  }

  /* ---------- Jobs and results ---------- */

  function upsertJob(incoming, { isNew = false, render = true } = {}) {
    if (!incoming || !incoming.id) return;
    const previous = state.jobs.get(incoming.id);
    const job = mergeJob(previous, incoming);
    state.jobs.set(job.id, job);
    state.seen.set(job.id, Date.now());
    if (isNew) state.newJobs.add(job.id);
    if (job.kind === 'download') onDownloadJob(job, previous);
    if (job.kind === 'preview' && styles) styles.onJob(job);
    if (previous && previous.status !== job.status) onStatusChange(job);
    if (render) renderJobs();
  }

  function onStatusChange(job) {
    if (!isMine(job) || job.kind === 'download' || job.kind === 'preview') return;
    const title = jobTitle(job);
    if (job.status === 'done') {
      announce(`${title} is done.`);
      state.celebrate.add(job.id);
      if (document.hidden) state.doneWhileHidden = true;
    } else if (job.status === 'failed') {
      announce(`${title} didn't finish.`);
    }
  }

  /** Preview-style jobs live in their own window, finished downloads aren't results, and finished jobs
      this browser cleared stay out of its list. */
  function visibleJobs() {
    return Array.from(state.jobs.values())
      .filter((job) => job.kind !== 'preview' && (job.kind !== 'download' || isActive(job)))
      .filter((job) => !(state.cleared.has(job.id) && !isActive(job)))
      .sort((a, b) => (b.created || 0) - (a.created || 0));
  }

  /** A batch card for pictures this viewer may see in full; the plain job card for videos, the
      Photo-real download and other people's jobs. */
  function groupKind(group) {
    return group.jobs.every((job) => job.kind === 'image' && jobView(job) === 'full') ? 'batch' : 'job';
  }

  function bindResults() {
    el.jobsMore.addEventListener('click', () => {
      state.showAllJobs = true;
      renderJobs();
    });
    el.resultsClear.addEventListener('click', () => {
      // Every finished card: batches of pictures, and videos too (each one can also be cleared on its own).
      const groups = groupJobs(visibleJobs(), state.mine.batches).filter((g) => !g.jobs.some(isActive));
      clearGroups(groups, el.resultsClear);
    });
  }

  function renderJobs() {
    if (!state.status) return;
    const groups = groupJobs(visibleJobs(), state.mine.batches);
    const shown = state.showAllJobs ? groups : groups.slice(0, JOBS_SHOWN);
    el.resultsEmpty.hidden = groups.length > 0;
    el.jobsMore.hidden = shown.length >= groups.length;
    el.jobsMore.textContent = `Show ${groups.length - shown.length} older`;
    const finished = groups.filter((g) => !g.jobs.some(isActive));
    if (document.activeElement === el.resultsClear && finished.length < 2) el.resultsTitle.focus();
    el.resultsClear.hidden = finished.length < 2;
    state.newestGroup = shown.length ? shown[0].id : null;
    state.jobGroup = new Map();
    for (const group of groups) for (const job of group.jobs) state.jobGroup.set(job.id, group.id);

    const keep = new Set(shown.map((group) => group.id));
    for (const [id, card] of state.cards) {
      if (!keep.has(id)) {
        releaseCardVideo(card);
        card.remove();
        state.cards.delete(id);
      }
    }
    let previous = null;
    for (const group of shown) {
      const kind = groupKind(group);
      let card = state.cards.get(group.id);
      if (card && card.dataset.kind !== kind) {
        releaseCardVideo(card);
        card.remove();
        card = null;
      }
      if (!card) {
        card = h('li', { class: 'job', 'data-kind': kind });
        if (group.jobs.map((job) => state.newJobs.delete(job.id)).some(Boolean)) card.classList.add('is-new');
        state.cards.set(group.id, card);
      }
      if (kind === 'batch') updateBatchCard(card, group);
      else updateJobCard(card, group.jobs[0]);
      const expected = previous ? previous.nextSibling : el.jobList.firstChild;
      if (card !== expected) el.jobList.insertBefore(card, expected);
      previous = card;
    }
    updateTitle();
  }

  function revealJob(id) {
    const card = state.cards.get(state.jobGroup && state.jobGroup.get(id));
    if (!card) return;
    const box = card.getBoundingClientRect();
    if (box.top < 0 || box.bottom > window.innerHeight) {
      card.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'center' });
    }
  }

  /* A batch: every picture sent together, whatever qualities they needed (one job each). Overall
     progress, a grid of small finished pictures that open the full-screen viewer, Download all,
     Clear finished. A finished batch folds to its header, except the newest one. */

  function batchRecord(group) {
    return state.mine.batches.find((b) => `batch-${b.id}` === group.id) || null;
  }

  function batchIsOpen(group, summary) {
    if (state.batchOpen.has(group.id)) return state.batchOpen.get(group.id);
    // Open while it works, the newest one, and one that made nothing but has a reason to read (it
    // failed, or a restart stopped a cloud picture), so the reason isn't folded away.
    return !summary.finished || group.id === state.newestGroup || (!summary.done && group.jobs.some((job) => Boolean(job.error)));
  }

  function batchStatus(jobs, summary) {
    if (jobs.some((job) => job.status === 'running')) return 'running';
    if (jobs.some((job) => job.status === 'queued')) return 'queued';
    if (!summary.done && summary.failed && !summary.stopped) return 'failed';
    // Stopped part-way: a grey "Stopped", not a green "Done" (the pictures finished before stay listed).
    if (summary.stopped) return 'cancelled';
    return 'done';
  }

  function updateBatchCard(card, group) {
    const summary = batchSummary(group.jobs);
    const open = batchIsOpen(group, summary);
    const shownCount = state.batchShown.get(group.id) || TRAY_PAGE;
    const phase = JSON.stringify([group.jobs.map((job) => [job.id, job.status, (job.results || []).length, job.error || '']),
      open, shownCount, isHost()]);
    if (card.dataset.phase !== phase) {
      card.dataset.phase = phase;
      buildBatchCard(card, group, summary, open, shownCount);
      if (viewer && viewer.isOpen() && state.viewerGroup === group.id) viewer.update(viewerItems(group), (item) => item.key);
    }
    updateBatchLive(card, group, summary);
  }

  function buildBatchCard(card, group, summary, open, shownCount) {
    const active = document.activeElement;
    const hadFocus = card.contains(active);
    const focusName = hadFocus ? (active.closest('[data-focus]') || {}).dataset : null;
    const jobs = group.jobs;
    const items = viewerItems(group);
    const live = jobs.some(isActive);
    const status = batchStatus(jobs, summary);
    card.classList.add('batch-card');
    card.dataset.status = status;
    const chip = statusChipFor(status, status === 'cancelled' ? 'Stopped' : null);
    if (status === 'done' && jobs.map((job) => state.celebrate.delete(job.id)).some(Boolean) && !reducedMotion()) {
      chip.classList.add('is-celebrating');
      chip.addEventListener('animationend', () => chip.classList.remove('is-celebrating'), { once: true });
    }
    const bodyId = `${group.id}-body`;
    const title = batchTitle(group);
    const toggle = h('button', {
      type: 'button', class: 'icon-button batch-toggle', 'data-focus': 'toggle', 'aria-expanded': String(open), 'aria-controls': bodyId,
      'aria-label': `${open ? 'Hide' : 'Show'} the pictures in ${title}`, title: open ? 'Hide these pictures' : 'Show these pictures',
    }, icon('chevron-down'));
    toggle.addEventListener('click', () => {
      state.batchOpen.set(group.id, !open);
      renderJobs();
    });
    const head = h('div', { class: 'job-head batch-head' },
      batchMosaic(group, items),
      h('div', { class: 'job-title' },
        h('p', { class: 'job-name', tabindex: '-1' }, title),
        h('p', { class: 'job-meta' }, batchMeta(group))),
      chip, toggle);
    const progress = h('div', { class: 'batch-progress' },
      live ? h('div', { class: 'progress is-live', role: 'progressbar', 'aria-label': `Progress for ${title}`, 'aria-valuemin': '0', 'aria-valuemax': '100' }, h('span')) : null,
      h('p', { class: 'batch-progress-text', 'data-live': 'batch-text' }));
    const parts = [head, progress];
    if (open) parts.push(batchBody(group, items, shownCount, bodyId));
    parts.push(batchActions(group, summary));
    card.replaceChildren(...parts);
    if (hadFocus && !card.contains(document.activeElement)) {
      const again = (focusName && focusName.focus && $(`[data-focus="${CSS.escape(focusName.focus)}"]`, card)) || $('.job-name', card);
      again.focus({ preventScroll: true });
    }
  }

  function batchTitle(group) {
    const names = group.jobs.flatMap((job) => (job.inputs || []).map(baseName)).filter(Boolean);
    if (!names.length) return 'Your pictures';
    return names.length === 1 ? names[0] : `${num0(names.length)} pictures`;
  }

  /** The qualities as coloured tags, then the size, file type and when it started. */
  function batchMeta(group) {
    const tags = [];
    const seen = new Set();
    for (const job of group.jobs) {
      const options = job.options || {};
      const used = (job.results || []).map((result) => result.mode_used);
      const mode = used.length && used.every((m) => m === VECTOR) ? VECTOR : options.mode;
      const level = mode === 'photoreal' ? options.photoreal_level : null;
      const key = `${mode}:${level}`;
      if (!mode || seen.has(key)) continue;
      seen.add(key);
      tags.push(modeTag(mode, 'image', level), ' ');
    }
    const options = group.jobs[0].options || {};
    const parts = [];
    if (options.size) parts.push(sizeLabel(options.size));
    if (options.output_format && options.output_format !== 'jpeg') parts.push(FORMAT_LABELS[options.output_format] || options.output_format);
    const when = startedAt(group.created);
    if (when) parts.push(when);
    const job = group.jobs[0];
    if (isHost() && job.owner && job.owner !== 'host' && !isMine(job)) parts.push('from a guest');
    return [...tags, parts.join(', ')];
  }

  /** Up to four finished pictures (or the first picture sent, while it works) as the batch's face. */
  function batchMosaic(group, items) {
    const sources = items.slice(0, 4).map((item) => item.preview);
    if (!sources.length) {
      const local = group.jobs.map((job) => state.jobThumbs.get(job.id)).find(Boolean);
      if (local) sources.push(local);
    }
    const box = h('span', { class: `batch-mosaic${sources.length > 1 ? ' is-grid' : ''}`, 'aria-hidden': 'true' });
    if (!sources.length) box.append(icon('image'));
    for (const src of sources) {
      const img = h('img', { src, alt: '', loading: 'lazy', decoding: 'async' });
      img.addEventListener('error', () => img.remove());
      box.append(img);
    }
    return box;
  }

  function batchBody(group, items, shownCount, bodyId) {
    const body = h('div', { class: 'batch-body', id: bodyId });
    const running = group.jobs.find((job) => job.status === 'running');
    const queued = group.jobs.find((job) => job.status === 'queued');
    if (running) {
      // The time left is in the batch's progress line above, for all of its pictures.
      body.append(h('div', { class: 'batch-now' },
        h('p', { class: 'job-message' }, h('span', { class: 'job-percent', 'data-live': 'percent' }), ' ', h('span', { 'data-live': 'message' }))));
    } else if (queued) {
      body.append(h('p', { class: 'batch-now muted', 'data-live': 'queue' }));
    }
    for (const job of group.jobs) {
      if (job.status === 'failed') {
        body.append(h('div', { class: 'alert alert-danger' }, icon('alert'),
          h('p', null, job.error || 'Something went wrong while working on these pictures.')));
        if (canRetry(job)) body.append(retryControls(job));
      } else if (job.status === 'cancelled' && (job.error || !(job.results || []).length)) {
        // The progress line above already says how many were done and stopped; this adds the server's
        // own reason (a cloud picture not sent again after a restart), or where to look when none are listed.
        body.append(h('p', { class: 'muted batch-note' }, stoppedText(job)));
      }
    }
    if (items.length) {
      const page = trayPage(items, shownCount);
      body.append(h('ul', { class: 'result-grid', 'aria-label': `Finished pictures in ${batchTitle(group)}` },
        page.items.map((item, i) => h('li', null, resultTile(group, item, i)))));
      if (page.rest) {
        body.append(h('button', {
          type: 'button', class: 'btn btn-secondary btn-sm batch-more', 'data-focus': 'more',
          onclick: () => {
            state.batchShown.set(group.id, (state.batchShown.get(group.id) || TRAY_PAGE) + TRAY_PAGE);
            renderJobs();
          },
        }, moreLabel(page.rest)));
      }
    }
    const missed = failureNotes(group);
    if (missed.length) body.append(h('ul', { class: 'notes' }, missed.map((note) => h('li', null, icon('info'), note))));
    if (items.length) body.append(h('p', { class: 'muted batch-where' }, whereText(group.jobs[0])));
    return body;
  }

  function resultTile(group, item, index) {
    const img = h('img', { src: item.preview, alt: '', loading: 'lazy', decoding: 'async' });
    const shot = h('span', { class: 'result-tile-img' }, img);
    img.addEventListener('error', () => shot.replaceChildren(icon('image')));
    const button = h('button', {
      type: 'button', class: 'result-tile', 'data-focus': `r-${item.key}`, 'aria-haspopup': 'dialog',
      'aria-label': `${item.title}${item.sizeText ? `, ${item.sizeText}` : ''}. ${item.compare ? 'Compare before and after' : 'Open it'}`,
    }, shot, h('span', { class: 'result-tile-name' }, item.title), item.sizeText ? h('span', { class: 'result-tile-dims' }, item.sizeText) : null);
    button.addEventListener('click', () => openViewer(group, index));
    return button;
  }

  function openViewer(group, index) {
    if (!viewer) return;
    stopCardVideo();
    state.viewerGroup = group.id;
    viewer.open(viewerItems(group), index, () => {
      state.viewerGroup = null;
      const title = el.viewerDialog.querySelector('#viewer-title').textContent;
      // Back to the picture that was last in view, even if the card was redrawn meanwhile. After the
      // browser's own focus return (Esc closes a dialog that way), so this one wins.
      window.setTimeout(() => {
        const card = state.cards.get(group.id);
        const last = viewerItems(group).find((item) => item.title === title);
        const again = card && ((last && $(`[data-focus="r-${CSS.escape(last.key)}"]`, card)) || $('.job-name', card));
        if (again) again.focus({ preventScroll: false });
      }, 0);
    });
  }

  function batchActions(group, summary) {
    const actions = h('div', { class: 'batch-actions' });
    const ready = group.jobs.filter((job) => (job.status === 'done' || job.status === 'cancelled') && (job.results || []).length);
    if (ready.length) actions.append(downloadAllControl(ready));
    const job = group.jobs[0];
    if (ready.length && isHost() && (!job.owner || job.owner === 'host' || isMine(job))) actions.append(openFolderButton(ready[0]));
    if (summary.finished) {
      const clear = h('button', { type: 'button', class: 'btn btn-quiet btn-sm', 'data-focus': 'clear' }, icon('check'), 'Clear finished');
      clear.addEventListener('click', () => clearGroups([group], clear));
      actions.append(clear);
    }
    const active = group.jobs.filter(isActive);
    if (active.length && active.every(canCancel)) actions.append(batchStopControls(active));
    return actions;
  }

  /** The live parts of a batch card: overall progress, and what the running job is doing now. */
  function updateBatchLive(card, group, summary) {
    const bar = $('.batch-progress .progress', card);
    if (bar) setProgress(bar, summary.fraction);
    const record = batchRecord(group);
    const text = $('[data-live="batch-text"]', card);
    if (text) text.textContent = batchProgressText(summary, batchSecondsLeft(group.jobs, record && record.est));
    const running = group.jobs.find((job) => job.status === 'running');
    if (running) {
      const message = $('[data-live="message"]', card);
      if (message) message.textContent = running.message || 'Working…';
      const percent = $('[data-live="percent"]', card);
      if (percent) percent.textContent = Number.isFinite(running.progress) && running.progress > 0 ? `${Math.floor(clamp01(running.progress) * 100)}%` : '';
    }
    const queue = $('[data-live="queue"]', card);
    const queued = group.jobs.find((job) => job.status === 'queued');
    if (queue && queued) queue.textContent = queueText(queued);
  }

  /** "X wasn't upscaled: …" notes, once each for the whole batch. */
  function failureNotes(group) {
    const notes = new Set();
    for (const job of group.jobs) {
      for (const result of job.results || []) for (const note of result.notes || []) if (WASNT_UPSCALED.test(note)) notes.add(note);
    }
    return Array.from(notes);
  }

  /** Clears finished batches from this browser's results. The host's pictures stay in Pictures ›
      Upscaled; a guest is asked first, because the list is their only way to download them. */
  function clearGroups(groups, button) {
    const ids = groups.flatMap((group) => group.jobs.filter((job) => !isActive(job)).map((job) => job.id));
    if (!ids.length) return;
    let ask = null;
    let done = false;
    const go = () => {
      if (done) return; // "Clear them" works once
      done = true;
      // The question goes, and the button comes back for renderJobs to show or hide (a batch's own
      // button goes with its card).
      if (ask && ask.isConnected) ask.replaceWith(button);
      for (const id of ids) state.cleared.add(id);
      saveCleared();
      for (const group of groups) state.batchOpen.delete(group.id);
      renderJobs();
      toast(isHost() ? 'Cleared from the list. Your pictures are still in Pictures › Upscaled.' : 'Cleared from the list.');
      const first = el.jobList.querySelector('.job-name');
      (first || el.resultsTitle).focus({ preventScroll: true });
    };
    if (isHost()) {
      go();
      return;
    }
    const wrap = button.parentNode;
    const yes = h('button', { type: 'button', class: 'btn btn-danger btn-sm' }, 'Clear them');
    const no = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, 'Keep them');
    ask = h('span', { class: 'confirm' }, h('span', { class: 'confirm-text' }, "Clear these? You won't be able to download them here any more."), yes, no);
    button.replaceWith(ask);
    yes.focus();
    yes.addEventListener('click', go);
    no.addEventListener('click', () => {
      if (ask.isConnected) ask.replaceWith(button);
      button.focus();
    });
    if (!wrap) go();
  }

  function loadCleared() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(CLEARED_KEY) || '[]');
      return new Set(Array.isArray(saved) ? saved.filter((id) => typeof id === 'string') : []);
    } catch {
      return new Set();
    }
  }

  function saveCleared() {
    try {
      window.localStorage.setItem(CLEARED_KEY, JSON.stringify(Array.from(state.cleared).slice(-500)));
    } catch {
      /* Private windows can refuse storage; the list is still cleared for this visit. */
    }
  }

  /** "Stop": stops every job of the batch that hasn't finished, after asking. */
  function batchStopControls(jobs) {
    const wrap = h('span', { class: 'job-actions batch-stop' });
    const stopButton = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': 'cancel' }, icon('x'), 'Stop');
    stopButton.addEventListener('click', () => {
      const several = jobs.length > 1 || (jobs[0].inputs || []).length > 1;
      const stop = h('button', { type: 'button', class: 'btn btn-danger btn-sm' }, several ? 'Stop them' : 'Stop it');
      const keep = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, 'Keep going');
      keep.addEventListener('click', () => {
        wrap.replaceChildren(stopButton);
        stopButton.focus();
      });
      stop.addEventListener('click', () => cancelJobs(jobs, wrap, stop));
      wrap.replaceChildren(h('span', { class: 'confirm-text' }, several ? 'Stop the pictures not done yet? Finished ones are kept.' : 'Stop this picture?'), stop, keep);
      stop.focus();
    });
    wrap.append(stopButton);
    return wrap;
  }

  /** The pictures finished so far are kept; queued jobs are stopped first, so the next doesn't start. */
  async function cancelJobs(jobs, wrap, button) {
    if (button.getAttribute('aria-disabled') === 'true') return;
    button.setAttribute('aria-disabled', 'true');
    button.classList.add('is-busy');
    const order = [...jobs.filter((job) => job.status === 'queued'), ...jobs.filter((job) => job.status !== 'queued')];
    try {
      for (const job of order) await api(API.cancel(job.id), { method: 'POST' });
      for (const job of order) {
        const fresh = await api(API.job(job.id)).catch(() => null);
        if (fresh) upsertJob(fresh.job || fresh);
      }
    } catch (err) {
      const card = wrap.closest('.job');
      const hadFocus = wrap.contains(document.activeElement);
      wrap.replaceChildren(h('p', { class: 'field-error', role: 'alert' }, err.message));
      if (hadFocus && card) $('.job-name', card).focus({ preventScroll: true });
    }
  }

  function downloadAllControl(jobs) {
    const files = jobs.reduce((n, job) => n + job.results.length, 0);
    if (jobs.length === 1) {
      const job = jobs[0];
      if (files === 1) {
        return h('a', { class: 'btn btn-primary btn-sm', href: API.file(job.id, 0), download: resultName(job.results[0], job, 0), 'data-focus': 'download' }, icon('download'), 'Download');
      }
      return h('a', { class: 'btn btn-primary btn-sm', href: API.zip(job.id), download: '', title: 'Saves one .zip file', 'data-focus': 'download' }, icon('archive'), 'Download all (zip)');
    }
    // Several jobs (different qualities): one zip each (a job of one picture saves just that picture),
    // started one after another.
    const button = h('button', { type: 'button', class: 'btn btn-primary btn-sm', title: downloadAllTitle(jobs), 'data-focus': 'download' }, icon('archive'), 'Download all (zip)');
    button.addEventListener('click', () => {
      jobs.forEach((job, i) => {
        window.setTimeout(() => {
          const link = job.results.length === 1
            ? h('a', { href: API.file(job.id, 0), download: resultName(job.results[0], job, 0) })
            : h('a', { href: API.zip(job.id), download: '' });
          document.body.append(link);
          link.click();
          link.remove();
        }, i * 700);
      });
      toast(`Downloading ${plural(files, 'picture')}.`);
    });
    return button;
  }

  /* ---------- The full-screen viewer's pictures ---------- */

  const WASNT_UPSCALED = / wasn't upscaled: /;

  /** "Made with Photo-real · Balanced in 3 min 40 s." */
  function madeWithText(job, result, modeUsed) {
    if (modeUsed === 'original') return `${NO_CHANGE_LABEL}: it was already this big, so it was saved as it is.`;
    if (modeUsed === VECTOR) return "Drawn at full size: it's a vector picture, so no AI was needed.";
    const options = job.options || {};
    const level = modeUsed === 'photoreal' ? (result.photoreal_level || options.photoreal_level) : null;
    const parts = [modeUsed ? `Made with ${modeLabel(modeUsed, job.kind, level)}` : 'Done'];
    if (Number.isFinite(result.seconds)) parts.push(`in ${formatDuration(result.seconds)}`);
    return `${parts.join(' ')}.`;
  }

  /** A result's notes, without the server's "Vector image: drawn directly…" (it repeats the line above)
      and the batch's "NAME wasn't upscaled" notes (listed once on the batch card). */
  function resultNotes(result, modeUsed) {
    return (result.notes || []).filter((note) => !(modeUsed === VECTOR && /vector/i.test(note)) && !WASNT_UPSCALED.test(note));
  }

  function resultFacts(result) {
    const facts = [];
    if (result.print && result.print.inches) facts.push(formatPrint(result.print));
    else if (result.dpi) facts.push(`${result.dpi} DPI`);
    if (Number.isFinite(result.file_bytes) && result.file_bytes > 0) facts.push(formatBytes(result.file_bytes));
    return facts;
  }

  /** Every finished picture of a batch, in order, as the viewer shows them. */
  function viewerItems(group) {
    const items = [];
    for (const job of group.jobs) {
      (job.results || []).forEach((result, n) => {
        const name = resultName(result, job, n);
        const modeUsed = result.mode_used || (job.options || {}).mode;
        const before = sourceSize(job, result, n);
        const after = result.width && result.height ? [result.width, result.height] : null;
        // Nothing to compare when the picture was kept as it was, or drawn straight from its vector shapes.
        const compare = modeUsed !== 'original' && modeUsed !== VECTOR;
        const labels = compareLabels(before, after);
        items.push({
          key: `${job.id}-${n}`,
          title: name,
          sizeText: after ? dims(after[0], after[1]) : '',
          compare,
          zoom: job.kind === 'image' ? {
            manifest: API.viewerInfo(job.id, n),
            tile: (side, level, col, row) => (side === 'before' ? API.beforeTile : API.tile)(job.id, n, level, col, row),
          } : null,
          preview: API.preview(job.id, n),
          previewAlt: `${name}, upscaled`,
          before: { src: API.before(job.id, n), alt: `Before: a close-up of ${name}, simply enlarged`, label: 'Before', detail: before ? dims(before[0], before[1]) : null, text: labels.before },
          after: { src: API.after(job.id, n), alt: 'After: the same close-up, upscaled', label: 'After', detail: after ? dims(after[0], after[1]) : null, text: labels.after },
          single: { text: labels.single },
          download: { href: API.file(job.id, n), name },
          open: { href: API.view(job.id, n), label: 'Open in new tab' },
          details: () => {
            const notes = resultNotes(result, modeUsed);
            const facts = resultFacts(result);
            return [
              h('p', { class: 'viewer-dims' },
                before ? h('span', null, 'Before ', h('strong', null, dims(before[0], before[1]))) : null,
                before && after ? h('span', { 'aria-hidden': 'true' }, ' → ') : null,
                after ? h('span', null, 'After ', h('strong', null, dims(after[0], after[1]))) : null),
              h('p', { class: 'muted' }, madeWithText(job, result, modeUsed)),
              facts.length ? h('p', { class: 'result-facts' }, facts.join(' · ')) : null,
              notes.length ? h('ul', { class: 'notes' }, notes.map((note) => h('li', null, icon('info'), note))) : null,
            ];
          },
        });
      });
    }
    return items;
  }

  /* ---------- Job cards (videos, the Photo-real download, other people's jobs) ---------- */

  function jobView(job) {
    return isHost() || isMine(job) ? 'full' : 'compact';
  }

  function updateJobCard(card, job) {
    const view = jobView(job);
    const phase = `${view}:${job.status}:${(job.results || []).length}`;
    if (card.dataset.phase !== phase) {
      card.dataset.phase = phase;
      buildJobCard(card, job, view);
    }
    if (!isActive(job)) return;
    if (view === 'full') updateLiveParts(card, job);
    else $('.job-meta', card).textContent = queueText(job);
  }

  function buildJobCard(card, job, view) {
    // A card is rebuilt when its job moves on (it starts, finishes, or is stopped): keep keyboard focus in it.
    const active = document.activeElement;
    const hadFocus = card.contains(active);
    const focusName = hadFocus && active.dataset ? active.dataset.focus : null;
    card.classList.toggle('job-compact', view === 'compact');
    card.dataset.status = job.status;
    const chip = statusChip(job);
    if (state.celebrate.delete(job.id) && !reducedMotion()) {
      chip.classList.add('is-celebrating');
      chip.addEventListener('animationend', () => chip.classList.remove('is-celebrating'), { once: true });
    }
    releaseCardVideo(card); // a video playing in the old card stops and lets go of its file
    const head = h('div', { class: 'job-head' },
      jobThumb(job, view),
      h('div', { class: 'job-title' },
        h('p', { class: 'job-name', tabindex: '-1' }, view === 'compact' ? othersTitle(job) : jobTitle(job)),
        h('p', { class: 'job-meta' }, view === 'compact' ? queueText(job) : jobMeta(job))),
      chip);
    card.replaceChildren(head, ...(view === 'full' ? jobBody(job) : []));
    if (hadFocus && !card.contains(document.activeElement)) {
      const again = (focusName && $(`[data-focus="${CSS.escape(focusName)}"]`, card)) || $('.job-name', card);
      again.focus({ preventScroll: true });
    }
  }

  function othersTitle(job) {
    return job.kind === 'video' ? "Someone else's video" : job.kind === 'download' ? 'Photo-real download' : "Someone else's pictures";
  }

  function jobTitle(job) {
    if (job.kind === 'download') return 'Photo-real download';
    const names = (job.inputs || []).map(baseName).filter(Boolean);
    if (!names.length) return job.kind === 'video' ? 'Your video' : 'Your pictures';
    return names.length === 1 ? names[0] : `${names[0]} and ${names.length - 1} more`;
  }

  function sizeLabel(id) {
    const size = (state.status.sizes || []).find((s) => s.id === id);
    return size ? splitSizeLabel(size.label)[0] : id;
  }

  function startedAt(created) {
    if (!created) return '';
    const date = new Date(created > 1e12 ? created : created * 1000);
    const today = new Date().toDateString() === date.toDateString();
    const time = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    return today ? `started ${time}` : `started ${date.toLocaleDateString([], { day: 'numeric', month: 'short' })}, ${time}`;
  }

  /** "[Fast] Whole video, started 18:15": the mode as a coloured tag, then plain words. */
  function jobMeta(job) {
    const options = job.options || {};
    const parts = [];
    let tag = null;
    if (job.kind === 'image') {
      if (options.mode) tag = modeTag(options.mode, 'image', options.mode === 'photoreal' ? options.photoreal_level : null);
      // SVGs are drawn, not upscaled, whatever quality was picked: say so once they're done.
      const used = (job.results || []).map((result) => result.mode_used);
      if (used.length && used.every((mode) => mode === VECTOR)) tag = modeTag(VECTOR, 'image');
      if (options.size) parts.push(sizeLabel(options.size));
    } else if (job.kind === 'video') {
      if (options.mode) tag = modeTag(options.mode, 'video');
      parts.push(options.preview_seconds ? `First ${options.preview_seconds} seconds` : 'Whole video');
    } else if (job.kind === 'download') {
      parts.push('One-time, about 7 GB');
    }
    const when = startedAt(job.created);
    if (when) parts.push(when);
    if (isHost() && job.owner && job.owner !== 'host' && !isMine(job)) parts.push('from a guest');
    let text = parts.join(', ');
    if (!tag) text = text.charAt(0).toUpperCase() + text.slice(1);
    return tag ? [tag, ' ', text] : text;
  }

  function jobThumb(job, view) {
    const local = view === 'full' && state.jobThumbs.get(job.id);
    const fallback = icon(job.kind === 'video' ? 'film' : job.kind === 'download' ? 'download' : 'image');
    const theme = job.kind === 'video' ? 'video' : null;
    if (!local) return h('span', { class: 'job-thumb', 'aria-hidden': 'true', 'data-theme': theme }, fallback);
    const img = h('img', { src: local, alt: '', decoding: 'async' });
    img.addEventListener('error', () => img.replaceWith(fallback));
    return h('span', { class: 'job-thumb', 'aria-hidden': 'true' }, img);
  }

  const STATUS_TEXT = { queued: 'Waiting', running: 'Working', done: 'Done', failed: "Didn't finish", cancelled: 'Cancelled' };
  const STATUS_ICON = { queued: 'clock', done: 'check', failed: 'alert', cancelled: 'x' };

  function statusChipFor(status, label = null) {
    return h('span', { class: 'status', 'data-status': status },
      STATUS_ICON[status] ? icon(STATUS_ICON[status]) : h('span', { class: 'status-dot', 'aria-hidden': 'true' }),
      label || STATUS_TEXT[status] || status);
  }

  function statusChip(job) {
    return statusChipFor(job.status);
  }

  function queueText(job) {
    if (job.status === 'running') return 'Working on it now.';
    if (job.status !== 'queued') return STATUS_TEXT[job.status] || '';
    const ahead = Array.from(state.jobs.values()).filter((other) => other.id !== job.id && isActive(other) && (other.created || 0) < (job.created || 0)).length;
    if (ahead === 0) return 'Starting in a moment.';
    return ahead === 1 ? 'Waiting for 1 job ahead of this one.' : `Waiting for ${ahead} jobs ahead of this one.`;
  }

  function canCancel(job) {
    return isHost() || isMine(job);
  }

  function jobBody(job) {
    if (job.status === 'queued' || job.status === 'running') {
      const body = h('div', { class: 'job-body' });
      if (job.status === 'running') {
        body.append(
          h('div', { class: 'progress is-live', role: 'progressbar', 'aria-label': `Progress for ${jobTitle(job)}`, 'aria-valuemin': '0', 'aria-valuemax': '100' }, h('span')),
          h('div', { class: 'progress-row' },
            h('p', { class: 'job-message' }, h('span', { class: 'job-percent', 'data-live': 'percent' }), ' ', h('span', { 'data-live': 'message' })),
            h('p', { class: 'muted', 'data-live': 'eta' })));
      } else {
        body.append(h('p', { class: 'job-message muted', 'data-live': 'message' }));
      }
      if (canCancel(job)) body.append(cancelControls(job));
      return [body];
    }
    if (job.status === 'done') return resultsBlock(job);
    if (job.status === 'failed') {
      return [h('div', { class: 'job-body' },
        h('div', { class: 'alert alert-danger' }, icon('alert'), h('p', null, job.error || 'Something went wrong while working on this.')),
        canRetry(job) ? retryControls(job) : null,
        h('div', { class: 'job-actions job-clear' }, clearJobButton(job)))];
    }
    return stoppedBlock(job);
  }

  /** "Clear from list" on a finished video (or a stopped job): like a batch's "Clear finished", for this
      browser only, so the list doesn't keep growing. */
  function clearJobButton(job) {
    const button = h('button', { type: 'button', class: 'btn btn-quiet btn-sm', 'data-focus': 'clear' }, icon('check'), 'Clear from list');
    button.addEventListener('click', () => clearGroups([{ id: `job-${job.id}`, jobs: [job] }], button));
    return button;
  }

  /** A failed job that kept its work (a video that ran out of space) can carry on from where it stopped. */
  function canRetry(job) {
    return job.retryable === true && job.status === 'failed' && canCancel(job);
  }

  function retryControls(job) {
    const wrap = h('div', { class: 'job-actions job-retry' });
    const button = h('button', { type: 'button', class: 'btn btn-primary btn-sm', 'data-focus': 'retry' },
      h('span', { class: 'spinner', 'aria-hidden': 'true' }), icon('refresh'), h('span', { class: 'btn-label' }, 'Retry'));
    button.addEventListener('click', async () => {
      // aria-disabled rather than disabled, so the button keeps keyboard focus until the card is rebuilt.
      if (button.getAttribute('aria-disabled') === 'true') return;
      button.setAttribute('aria-disabled', 'true');
      button.classList.add('is-busy');
      try {
        const data = await api(API.retry(job.id), { method: 'POST' });
        upsertJob((data && data.job) || { ...job, status: 'queued', error: null, retryable: false });
        announce(`${jobTitle(job)} carries on from where it stopped.`);
      } catch (err) {
        // 403 and 409 come with the server's own words (not yours to retry, nothing to carry on from).
        const card = wrap.closest('.job');
        const hadFocus = wrap.contains(document.activeElement);
        wrap.replaceChildren(h('p', { class: 'field-error', role: 'alert' }, err.message));
        if (hadFocus && card) $('.job-name', card).focus({ preventScroll: true });
      }
    });
    wrap.append(button, h('span', { class: 'muted job-retry-note' }, 'It carries on from where it stopped.'));
    return wrap;
  }

  /** Why a stopped job has the results it has: the pictures finished before the stop were saved (and
      are listed when the server sends them), so it doesn't say nothing was. */
  function stoppedText(job) {
    // The server's own reason when it has one: a cloud job stopped by a restart (not sent again, so
    // nothing was charged twice), or a guest's own key that was forgotten.
    if (job.error) return job.error;
    const results = job.results || [];
    const batch = job.kind === 'image' && (job.inputs || []).length > 1;
    if (results.length) return `Stopped. ${results.length === 1 ? 'This picture was' : `These ${results.length} pictures were`} finished before you stopped.`;
    if (batch && isHost() && (!job.owner || job.owner === 'host' || isMine(job))) {
      return 'Stopped. Any pictures finished before you stopped are in Pictures › Upscaled.';
    }
    if (batch) return 'Stopped before all the pictures were done.';
    return 'Cancelled. Nothing was saved.';
  }

  function stoppedBlock(job) {
    const note = h('div', { class: 'job-body' }, h('p', { class: 'muted' }, stoppedText(job)));
    if ((job.results || []).length) return [note, ...resultsBlock(job)];
    note.append(h('div', { class: 'job-actions job-clear' }, clearJobButton(job)));
    return [note];
  }

  function updateLiveParts(card, job) {
    const bar = $('.progress', card);
    if (bar) setProgress(bar, job.progress);
    const message = $('[data-live="message"]', card);
    if (message) message.textContent = job.status === 'queued' ? queueText(job) : (job.message || 'Working…');
    const percent = $('[data-live="percent"]', card);
    if (percent) percent.textContent = Number.isFinite(job.progress) && job.progress > 0 ? `${Math.floor(clamp01(job.progress) * 100)}%` : '';
    const eta = $('[data-live="eta"]', card);
    if (eta) eta.textContent = jobEtaText(job);
  }

  function cancelControls(job) {
    const wrap = h('div', { class: 'job-actions' });
    const cancelButton = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': 'cancel' }, icon('x'), 'Cancel');
    const askFirst = () => {
      const stop = h('button', { type: 'button', class: 'btn btn-danger btn-sm' }, 'Stop it');
      const keep = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, 'Keep going');
      keep.addEventListener('click', () => {
        wrap.replaceChildren(cancelButton);
        cancelButton.focus();
      });
      stop.addEventListener('click', () => cancelJob(job, wrap, stop));
      wrap.replaceChildren(h('span', { class: 'confirm-text' }, 'Stop this job?'), stop, keep);
      stop.focus();
    };
    cancelButton.addEventListener('click', askFirst);
    wrap.append(cancelButton);
    return wrap;
  }

  async function cancelJob(job, wrap, button) {
    // aria-disabled rather than disabled, so "Stop it" keeps keyboard focus until the card is rebuilt.
    if (button.getAttribute('aria-disabled') === 'true') return;
    button.setAttribute('aria-disabled', 'true');
    button.classList.add('is-busy');
    try {
      await api(API.cancel(job.id), { method: 'POST' });
      const fresh = await api(API.job(job.id)).catch(() => null);
      if (fresh) upsertJob(fresh.job || fresh);
    } catch (err) {
      const card = wrap.closest('.job');
      const hadFocus = wrap.contains(document.activeElement);
      wrap.replaceChildren(h('p', { class: 'field-error', role: 'alert' }, err.message));
      if (hadFocus && card) $('.job-name', card).focus({ preventScroll: true });
    }
  }

  function resultName(result, job, n) {
    const fromResult = result.filename || result.name || baseName(result.output_path || result.path);
    if (fromResult) return fromResult;
    const input = (job.inputs || [])[n];
    return input ? baseName(input) : `Result ${n + 1}`;
  }

  function whereText(job) {
    const guestJob = job.owner && job.owner !== 'host' && !(isHost() && isMine(job));
    if (isHost() && !guestJob) return 'Saved in Pictures › Upscaled on this computer.';
    if (isHost()) return "A guest's results are kept for 24 hours.";
    return 'Kept here for 24 hours, so download what you want to keep.';
  }

  /** A finished video (pictures are shown in their batch card instead). */
  function resultsBlock(job) {
    const results = job.results || [];
    if (!results.length) {
      return [h('div', { class: 'job-body' }, h('p', { class: 'muted' }, 'Finished, but there was nothing to save.'))];
    }
    const list = h('ol', { class: 'result-list' }, results.map((result, n) => resultItem(job, result, n)));
    const foot = h('div', { class: 'results-foot' }, h('p', { class: 'muted' }, whereText(job)));
    if (results.length > 1) {
      foot.append(h('a', { class: 'btn btn-secondary btn-sm', href: API.zip(job.id), download: '', title: 'Saves one .zip file' }, icon('archive'), 'Download all (zip)'));
    }
    if (isHost() && (!job.owner || job.owner === 'host' || isMine(job))) foot.append(openFolderButton(job));
    foot.append(h('span', { class: 'job-clear' }, clearJobButton(job)));
    return [list, foot];
  }

  function openFolderButton(job) {
    const button = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': 'folder' }, icon('folder'), 'Open folder');
    button.addEventListener('click', async () => {
      button.disabled = true;
      try {
        await api(API.openFolder, { method: 'POST', body: { job_id: job.id } });
      } catch (err) {
        button.replaceChildren(icon('alert'), "Couldn't open the folder");
        window.setTimeout(() => button.replaceChildren(icon('folder'), 'Open folder'), 4000);
      } finally {
        button.disabled = false;
      }
    });
    return button;
  }

  /** The picture's size before upscaling, for the "Before" label: from the server, or remembered when it
      was sent (in the order sent, so a batch where a picture failed is matched up by name first). */
  function sourceSize(job, result, n) {
    const w = result.source_width || result.src_width || (result.source && result.source[0]);
    const hgt = result.source_height || result.src_height || (result.source && result.source[1]);
    if (w && hgt) return [w, hgt];
    const remembered = state.mine.dims.get(job.id);
    const index = resultInputIndex(job, n);
    const size = remembered && index != null ? remembered[index] : null;
    return size && size[0] ? size : null;
  }

  function resultItem(job, result, n) {
    const name = resultName(result, job, n);
    const isVideo = job.kind === 'video';
    const modeUsed = result.mode_used || (job.options || {}).mode;
    const notes = resultNotes(result, modeUsed);
    const facts = resultFacts(result);
    const info = h('div', { class: 'result-info' },
      h('p', { class: 'result-name', title: name }, name),
      result.width && result.height ? h('p', { class: 'result-dims' }, dims(result.width, result.height), h('small', null, ' pixels')) : null,
      facts.length ? h('p', { class: 'result-facts' }, facts.join(' · ')) : null,
      h('p', { class: 'result-mode' }, madeWithText(job, result, modeUsed)),
      notes.length ? h('ul', { class: 'notes' }, notes.map((note) => h('li', null, icon('info'), note))) : null,
      // HEVC plays in QuickTime and on phones; Windows' own apps need Microsoft's free extension.
      isVideo ? h('p', { class: 'result-hint' }, icon('info'), WINDOWS_HEVC_NOTE) : null,
      h('div', { class: 'result-actions' },
        h('a', { class: 'btn btn-primary btn-sm', href: API.file(job.id, n), download: name }, icon('download'), isVideo ? 'Download video' : 'Download'),
        isVideo && videoCompare ? videoCompareButton(job, result, n, name) : null));
    return h('li', { class: 'result' }, isVideo ? videoPreview(job, n, name) : photoPreview(job, n, name), info);
  }

  /** "Compare with the original": the two videos side by side, played together (js/videocompare.js). */
  function videoCompareButton(job, result, n, name) {
    const button = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': `compare-${n}`, 'aria-haspopup': 'dialog' },
      icon('columns'), 'Compare with the original');
    button.addEventListener('click', () => openVideoCompare(job, result, n, name));
    return button;
  }

  function openVideoCompare(job, result, n, name) {
    if (!videoCompare) return;
    stopCardVideo(); // never more than the comparison's two videos at once
    const options = job.options || {};
    const parts = [modeLabel(result.mode_used || options.mode, 'video')];
    if (options.preview_seconds) parts.push(`first ${options.preview_seconds} seconds`);
    videoCompare.open({
      title: name,
      subtitle: `Original and upscaled, played together · ${parts.filter(Boolean).join(', ')}`,
      startUrl: API.compare(job.id, n),
      statusUrl: API.compareStatus(job.id, n),
      cancelUrl: API.compareCancel(job.id, n),
      download: { href: API.file(job.id, n), name },
    }, () => {
      // Back to the button that opened it, after the browser's own focus return.
      window.setTimeout(() => {
        const card = state.cards.get(state.jobGroup && state.jobGroup.get(job.id));
        const again = card && $(`[data-focus="compare-${n}"]`, card);
        if (again) again.focus({ preventScroll: true });
      }, 0);
    });
  }

  function photoPreview(job, n, name) {
    const img = h('img', { src: API.preview(job.id, n), alt: '', loading: 'lazy', decoding: 'async' });
    const link = h('a', { class: 'result-preview', href: API.preview(job.id, n), target: '_blank', rel: 'noopener', 'aria-label': `Open a bigger preview of ${name}` }, img);
    img.addEventListener('error', () => link.replaceChildren(icon('image')));
    return link;
  }

  /* A finished video in its card: a still from it and a Play button. It plays from the comparison's
     browser-friendly H.264 copy (made the first time, with progress, and shared with "Compare with
     the original"), never from the 10-bit HEVC file itself, which many browsers can't play and which
     is heavy to decode. One card plays at a time, and opening a viewer stops it. */

  let cardVideo = null; // {box, video, stop} for the one card video playing (or getting ready)

  function stopCardVideo() {
    if (!cardVideo) return;
    const playing = cardVideo;
    cardVideo = null;
    playing.stop();
  }

  /** A card that is about to be rebuilt or removed: its video stops and lets go of its file. */
  function releaseCardVideo(card) {
    if (cardVideo && card && card.contains(cardVideo.box)) stopCardVideo();
  }

  function videoPreview(job, n, name) {
    const box = h('div', { class: 'result-preview result-video' });
    function still() {
      const img = h('img', { src: API.poster(job.id, n), alt: '', loading: 'lazy', decoding: 'async' });
      img.addEventListener('error', () => img.replaceWith(h('span', { class: 'result-video-blank', 'aria-hidden': 'true' }, icon('film'))));
      const play = h('button', { type: 'button', class: 'result-video-play', 'data-focus': `play-${n}`, 'aria-label': `Play ${name}`, title: 'Play the upscaled video here' },
        h('span', { class: 'result-video-button', 'aria-hidden': 'true' }, icon('play')));
      play.addEventListener('click', () => playCardVideo(box, job, n, name, still));
      box.replaceChildren(img, play);
      return play;
    }
    still();
    return box;
  }

  function playCardVideo(box, job, n, name, still) {
    stopCardVideo();
    const life = window.ReLife;
    const scope = life.createScope('card-video');
    const hadFocus = box.contains(document.activeElement);
    const playing = {
      box,
      video: null,
      stop() {
        scope.dispose(); // pauses and unloads the video, stops the requests
        if (!box.isConnected) return;
        const button = still();
        if (box.contains(document.activeElement) || document.activeElement === document.body) button.focus({ preventScroll: true });
      },
    };
    cardVideo = playing;
    const text = h('span', null, 'Getting it ready to play here…');
    const status = h('div', { class: 'result-video-status', role: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), text);
    const stop = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': `play-${n}` }, icon('x'), 'Stop');
    stop.addEventListener('click', () => { if (cardVideo === playing) stopCardVideo(); });
    box.replaceChildren(status, stop);
    if (hadFocus) stop.focus({ preventScroll: true });

    const request = (url, init = {}) => {
      const ctrl = scope.controller();
      return api(url, { ...init, signal: ctrl.signal }).finally(() => scope.done(ctrl));
    };
    const problem = (message) => {
      if (scope.disposed) return;
      const again = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', 'data-focus': `play-${n}` }, icon('refresh'), 'Try again');
      again.addEventListener('click', () => playCardVideo(box, job, n, name, still));
      const focus = box.contains(document.activeElement);
      box.replaceChildren(h('div', { class: 'result-video-status is-problem', role: 'alert' }, icon('alert'), h('span', null, message)), again);
      if (focus) again.focus({ preventScroll: true });
    };
    const onError = (err) => {
      if (scope.disposed || life.isAbort(err)) return;
      problem((err && err.message) || "Couldn't get the video ready to play here. Use Download instead.");
    };
    const attach = (url) => {
      const video = h('video', { controls: true, playsinline: true, preload: 'metadata', 'aria-label': `${name}, upscaled` });
      video.src = url;
      scope.video(video);
      scope.listen(video, 'error', () => problem("This video couldn't be played here. Use Download instead."));
      playing.video = video;
      const focus = box.contains(document.activeElement);
      box.replaceChildren(video);
      if (focus) video.focus({ preventScroll: true });
      const started = video.play();
      if (started && typeof started.catch === 'function') started.catch(() => {}); // the controls are there to press
    };
    const onState = (st) => {
      if (scope.disposed || !st) return;
      if (st.status === 'ready' && st.upscaled_url) {
        attach(st.upscaled_url);
        return;
      }
      if (st.status === 'failed' || st.status === 'cancelled') {
        problem(st.error || "Couldn't get the video ready to play here. Use Download instead.");
        return;
      }
      const pct = Math.floor(Math.max(0, Math.min(1, Number(st.progress) || 0)) * 100);
      text.textContent = st.status === 'waiting' ? 'Waiting for another video to get ready…' : `Getting it ready to play here… ${pct}%`;
      scope.timeout(() => request(API.compareStatus(job.id, n)).then(onState, onError), 900);
    };
    request(API.compare(job.id, n), { method: 'POST' }).then(onState, onError);
  }

  function updateTitle() {
    if (!document.hidden) state.doneWhileHidden = false;
    const running = Array.from(state.jobs.values()).find((job) => job.status === 'running' && job.kind !== 'preview' && isMine(job));
    let title = 'Re:Solution';
    if (running && Number.isFinite(running.progress) && running.progress > 0) title = `(${Math.floor(clamp01(running.progress) * 100)}%) Re:Solution`;
    else if (running) title = '(Working) Re:Solution';
    else if (state.doneWhileHidden) title = '(Done) Re:Solution';
    document.title = title;
  }

  /* ---------- Theme ---------- */

  function currentTheme() {
    return themeFrom(document.documentElement.dataset.colorScheme);
  }

  function storedTheme() {
    try {
      return themeFrom(window.localStorage.getItem(THEME_KEY));
    } catch {
      return 'auto';
    }
  }

  function applyTheme(theme, { turned = false } = {}) {
    const root = document.documentElement;
    if (theme === 'auto') delete root.dataset.colorScheme;
    else root.dataset.colorScheme = theme;
    const label = themeLabel(theme);
    el.themeButton.setAttribute('aria-label', label);
    el.themeButton.title = label;
    el.themeButton.replaceChildren(icon(THEME_ICONS[theme], turned ? 'icon is-new' : 'icon'));
  }

  function bindTheme() {
    applyTheme(currentTheme());
    el.themeButton.addEventListener('click', () => {
      const next = nextTheme(currentTheme());
      try {
        if (next === 'auto') window.localStorage.removeItem(THEME_KEY);
        else window.localStorage.setItem(THEME_KEY, next);
      } catch {
        /* Private windows can refuse storage; the theme still changes for this visit. */
      }
      applyTheme(next, { turned: true });
      toast(THEME_TOASTS[next]);
    });
    // Another tab of Re:Solution changed it.
    window.addEventListener('storage', (event) => {
      if (event.key === THEME_KEY) applyTheme(storedTheme());
    });
  }

  /* ---------- Share on Wi-Fi (host only) ---------- */

  function shareState() {
    return (state.status && state.status.share) || { enabled: false, urls: [] };
  }

  /** Whether guests need a password: the share status says so directly; older servers only in the settings. */
  function sharePasswordSet(share) {
    return typeof share.password_set === 'boolean' ? share.password_set : hasPassword(currentSettings());
  }


  function renderShareButton() {
    const on = Boolean(shareState().enabled);
    // The words say the state ("Sharing on"); it isn't a toggle button, because a click opens the panel.
    el.shareButton.dataset.sharing = on ? 'on' : 'off';
    el.shareButtonText.textContent = on ? 'Sharing on' : 'Share on Wi-Fi';
    el.shareButton.title = on
      ? 'Others on your Wi-Fi can use Re:Solution. Open for the address and QR code.'
      : 'Let phones and computers on your Wi-Fi use Re:Solution';
  }

  function urlList(urls) {
    return h('ul', { class: 'url-list' }, urls.map((url) => {
      const code = h('code', null, url);
      return h('li', { class: 'url-item' }, code, copyButton(url, code));
    }));
  }

  function bindShare() {
    el.shareButton.addEventListener('click', openShare);
    el.shareSwitch.addEventListener('change', () => setSharing(el.shareSwitch.checked));
  }

  function openShare() {
    if (!isHost()) return;
    setFeedback(el.shareFeedback, '');
    renderShareDialog(true);
    if (!el.shareDialog.open) el.shareDialog.showModal();
    (state.endpoints.share ? el.shareSwitch : el.shareDialog.querySelector('.dialog-close')).focus();
  }

  /** The dialog's switch and details. The details are redrawn only when something in them changes, so a
      status refresh doesn't move keyboard focus off a Copy button. */
  function renderShareDialog(force = false) {
    if (!state.status || !isHost()) {
      if (el.shareDialog.open) el.shareDialog.close();
      return;
    }
    const share = shareState();
    const on = Boolean(share.enabled);
    const urls = share.urls || [];
    const locked = sharePasswordSet(share);
    const canSwitch = Boolean(state.endpoints.share);
    const qrSource = on && urls.length ? shareQrSource(share, urls) : null;
    if (!state.sharing) el.shareSwitch.checked = on; // not while a switch is on its way
    el.shareSwitchRow.hidden = !canSwitch;
    el.shareSwitchHelp.textContent = on
      ? 'Phones and computers on your Wi-Fi can use Re:Solution. Sharing stops when you quit it.'
      : 'Only this computer can use Re:Solution right now.';
    el.shareDialog.classList.toggle('is-on', on);
    if (!changed('share-dialog', [on, urls, locked, canSwitch, qrSource]) && !force) return;

    const parts = [];
    if (!canSwitch && !on) parts.push(h('p', null, COPY.shareRestart));
    if (on && urls.length) {
      parts.push(h('p', null, 'On a phone or computer on the same ', h('span', { class: 'nowrap' }, 'Wi-Fi'), ', open:'), urlList(urls));
      const qr = h('figure', { class: 'share-qr' },
        h('img', { src: qrSource, alt: `QR code for ${urls[0]}`, width: '148', height: '148', decoding: 'async' }),
        h('figcaption', null, 'Or point a phone\'s camera at this code to open it.'));
      qr.querySelector('img').addEventListener('error', () => { qr.hidden = true; });
      parts.push(qr);
    } else if (on) {
      parts.push(h('p', null, COPY.shareNoNetwork));
    }
    // What this computer's firewall asks the first time sharing starts.
    const note = on ? shareNote(share, navigator.platform || navigator.userAgent) : '';
    if (note) parts.push(h('p', { class: 'share-note' }, icon('info'), h('span', null, note)));
    if (!locked) {
      const toSettings = h('a', { href: '#/settings' }, 'Set a password in Settings');
      toSettings.addEventListener('click', (event) => {
        event.preventDefault();
        el.shareDialog.close();
        if (state.view === 'settings') {
          el.newPassword.focus();
          return;
        }
        // After the page change, which would otherwise put focus on the Settings heading.
        window.addEventListener('hashchange', () => el.newPassword.focus(), { once: true });
        goTo('settings');
      });
      parts.push(h('div', { class: 'alert alert-warn share-warn' }, icon('alert'),
        h('p', null, on ? 'Anyone on your Wi-Fi can use Re:Solution. ' : 'There\'s no password, so once sharing is on anyone on your Wi-Fi can use Re:Solution. ',
          toSettings, '.')));
    }
    el.shareDetails.replaceChildren(...parts);
  }

  /** Turns sharing on or off without a restart (POST /api/share starts or stops the Wi-Fi listener). */
  async function setSharing(on) {
    if (!isHost() || state.sharing) return;
    state.sharing = true;
    const hadFocus = document.activeElement === el.shareSwitch;
    el.shareSwitch.disabled = true;
    setFeedback(el.shareFeedback, on ? 'Turning sharing on…' : 'Turning sharing off…');
    try {
      const data = await api(API.share, { method: 'POST', body: { enabled: on } });
      if (data && typeof data.enabled === 'boolean') {
        // The whole share status (addresses, QR code, note), so the panel is right even if the refresh fails.
        state.status = { ...state.status, share: { ...data, urls: Array.isArray(data.urls) ? data.urls : [] } };
      }
      await refreshStatus().catch(() => {});
      const nowOn = Boolean(shareState().enabled);
      setFeedback(el.shareFeedback, nowOn ? 'Sharing is on.' : 'Sharing is off. Only this computer can use Re:Solution now.');
    } catch (err) {
      if (missingEndpoint(err.status, err.detail)) {
        state.endpoints.share = false; // this copy of the server can only share when started that way
        setFeedback(el.shareFeedback, '');
      } else {
        setFeedback(el.shareFeedback, err.message, true);
      }
    } finally {
      state.sharing = false;
      el.shareSwitch.disabled = false;
      renderShareButton();
      renderSettings();
      renderShareDialog(true);
      if (!state.endpoints.share) el.shareDialog.querySelector('.dialog-close').focus();
      else if (hadFocus && el.shareDialog.open) el.shareSwitch.focus(); // it lost focus while it was switched off
    }
  }

  /* ---------- Settings (host only) ---------- */

  function bindSettings() {
    el.passwordForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const password = el.newPassword.value;
      if (password.length < MIN_PASSWORD) {
        setFeedback(el.passwordFeedback, `Use at least ${MIN_PASSWORD} characters.`, true);
        el.newPassword.focus();
        return;
      }
      el.passwordSave.disabled = true;
      const ok = await saveSettings({ access_password: password }, el.passwordFeedback, 'Password saved. Guests now need it to open Re:Solution.');
      el.passwordSave.disabled = false;
      if (ok) el.newPassword.value = '';
    });
    el.passwordClear.addEventListener('click', async () => {
      el.passwordClear.disabled = true;
      await saveSettings({ access_password: '', allow_guests_cloud: false }, el.passwordFeedback, 'Password removed.');
      el.passwordClear.disabled = false;
    });
    el.allowGuestsCloud.addEventListener('change', () => {
      const on = el.allowGuestsCloud.checked;
      if (on === Boolean(currentSettings().allow_guests_cloud)) return;
      saveSettings({ allow_guests_cloud: on }, el.passwordFeedback, on ? 'Guests can now use your cloud services.' : 'Guests can no longer use your cloud services.');
    });
    el.defaultMode.addEventListener('change', () => {
      if (el.defaultMode.value === defaultModeChoices(state.status, currentSettings()).value) return;
      saveSettings({ default_image_mode: el.defaultMode.value }, el.defaultsFeedback);
    });
    el.defaultFormat.addEventListener('change', (event) => {
      const input = event.target;
      if (input.name !== 'default-format' || !input.checked) return;
      if (input.value === (currentSettings().output_format || DEFAULTS.format)) return;
      saveSettings({ output_format: input.value }, el.defaultsFeedback);
    });
  }

  function currentSettings() {
    return state.settings || (state.status && state.status.settings) || {};
  }

  function hasPassword(settings) {
    return Boolean(settings.has_password ?? settings.password_set ?? settings.access_password_hash);
  }

  async function loadSettings() {
    try {
      const data = await api(API.settings);
      state.settings = (data && data.settings) || data || {};
      renderSettings();
    } catch (err) {
      setFeedback(el.defaultsFeedback, err.message, true);
    }
  }

  async function saveSettings(patch, feedback, okText = 'Saved.') {
    setFeedback(feedback, 'Saving…');
    try {
      const data = await api(API.settings, { method: 'POST', body: patch });
      const fresh = data && typeof data === 'object' ? (data.settings || data) : null;
      state.settings = fresh && Object.keys(fresh).length ? fresh : { ...currentSettings(), ...patch };
      setFeedback(feedback, okText);
      await refreshStatus().catch(() => {});
      renderSettings();
      return true;
    } catch (err) {
      setFeedback(feedback, err.message, true);
      renderSettings();
      return false;
    }
  }

  function renderSettings() {
    if (!isHost()) return;
    const settings = currentSettings();
    const share = state.status.share || {};

    const urls = share.urls || [];
    const openButton = (label) => h('button', { type: 'button', class: 'btn btn-secondary btn-sm share-open', 'aria-haspopup': 'dialog', onclick: openShare }, icon('wifi'), label);
    if (share.enabled) {
      el.shareBody.replaceChildren(
        urls.length ? h('p', null, 'Sharing is on. Others on your home Wi-Fi can open Re:Solution at:') : h('p', null, COPY.shareNoNetwork),
        urls.length ? urlList(urls) : null,
        openButton(urls.length ? 'QR code and sharing options' : 'Sharing options'));
    } else if (!state.endpoints.share) {
      el.shareBody.replaceChildren(
        h('p', null, 'Sharing is off, so only this computer can use Re:Solution.'),
        h('p', { class: 'hint' }, COPY.shareRestart));
    } else {
      el.shareBody.replaceChildren(
        h('p', null, 'Sharing is off, so only this computer can use Re:Solution.'),
        openButton('Share on Wi-Fi'));
    }

    const locked = hasPassword(settings);
    el.passwordStatus.textContent = locked
      ? 'Guests need a password to use Re:Solution.'
      : 'No password. Anyone on your Wi-Fi can use Re:Solution while sharing is on.';
    el.newPasswordLabel.textContent = locked ? 'Change the password' : 'Choose a password';
    el.passwordSave.textContent = locked ? 'Change password' : 'Set password';
    el.passwordClear.hidden = !locked;

    el.allowGuestsCloud.checked = Boolean(settings.allow_guests_cloud) && locked;
    el.allowGuestsCloud.disabled = !locked;
    el.allowGuestsHelp.textContent = locked
      ? 'Guests can then pick cloud quality, and you pay for it.'
      : 'Set a password first, so nobody else on your Wi-Fi can spend your money.';

    renderServices();

    const { options, value } = defaultModeChoices(state.status, settings);
    el.defaultMode.replaceChildren(...options.map((m) => h('option', { value: m.id, selected: m.id === value }, m.label)));
    el.defaultMode.value = value;
    const format = settings.output_format || DEFAULTS.format;
    keepingFocus(el.defaultFormat, () => el.defaultFormat.replaceChildren(...formatChoices(state.status).map((f) => h('label', { class: 'chip' },
      h('input', { type: 'radio', name: 'default-format', value: f.id, checked: f.id === format }),
      h('span', { class: 'chip-main' }, f.label)))));
  }

  /** A Copy button for `text`. If the browser won't copy (some block it on a page that isn't https),
      the address in `shown` is selected instead, so a keyboard shortcut copies it. */
  function copyButton(text, shown) {
    const button = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, icon('copy'), 'Copy');
    button.addEventListener('click', async () => {
      let ok = false;
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch {
        const area = h('textarea', { 'aria-hidden': 'true', style: 'position:fixed;opacity:0' });
        area.value = text;
        // Inside an open dialog the rest of the page can't be selected, so the helper goes next to the button.
        (button.closest('dialog') || document.body).append(area);
        area.select();
        try { ok = document.execCommand('copy'); } catch { ok = false; }
        area.remove();
      }
      let label = ok ? 'Copied' : 'Select and copy it';
      if (!ok && shown) {
        const range = document.createRange();
        range.selectNodeContents(shown);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        label = `Press ${/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? '⌘C' : 'Ctrl+C'} to copy`;
      }
      button.replaceChildren(icon(ok ? 'check' : 'alert'), label);
      window.setTimeout(() => button.replaceChildren(icon('copy'), 'Copy'), ok ? 2500 : 5000);
    });
    return button;
  }

  function renderServices() {
    const providers = state.status.providers || [];
    if (!providers.length) {
      el.serviceList.replaceChildren(h('li', { class: 'muted' }, 'No cloud services in this version.'));
      return;
    }
    el.serviceList.replaceChildren(...providers.map((provider) => {
      const item = h('li', { class: 'service' },
        h('span', { class: 'service-name' }, provider.name),
        h('span', { class: `service-state${provider.connected ? ' is-on' : ''}` },
          provider.connected ? icon('check') : null, provider.connected ? 'Connected' : 'Not connected'));
      if (provider.connected) {
        const remove = h('button', { type: 'button', class: 'btn btn-quiet btn-sm btn-danger-text' }, 'Remove');
        remove.addEventListener('click', () => confirmRemove(item, provider, remove));
        item.append(remove);
      } else {
        item.append(h('button', {
          type: 'button', class: 'btn btn-secondary btn-sm',
          onclick: () => wizard && wizard.open({ providerId: provider.id }),
        }, 'Connect'));
      }
      return item;
    }));
  }

  function confirmRemove(item, provider, removeButton) {
    const yes = h('button', { type: 'button', class: 'btn btn-danger btn-sm' }, 'Remove key');
    const no = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, 'Keep it');
    const prompt = h('span', { class: 'confirm-text' }, `Remove your ${provider.name} key from this computer?`);
    const controls = [prompt, yes, no];
    removeButton.replaceWith(...controls);
    yes.focus();
    no.addEventListener('click', () => renderServices());
    yes.addEventListener('click', async () => {
      yes.disabled = true;
      try {
        await api(API.providerKey(provider.id), { method: 'DELETE' });
        setFeedback(el.servicesFeedback, `${provider.name} removed.`);
        await refreshStatus();
      } catch (err) {
        setFeedback(el.servicesFeedback, err.message, true);
        renderServices();
      }
    });
  }

  /* ---------- About ---------- */

  function renderAbout() {
    const hardware = state.status.hardware || {};
    const memory = Number.isFinite(hardware.memory_gb) ? ` with ${Math.round(hardware.memory_gb)} GB of memory` : '';
    const cpuOnly = Boolean(hardware.cpu_only) || hardware.device === 'cpu';
    const machine = hardware.device_name ? `Running on ${hardware.device_name}${memory}.` : '';
    el.aboutMachine.textContent = cpuOnly ? `${machine} ${hardware.note || 'CPU only: slow'}.`.trim().replace(/\.\.$/, '.') : machine;
    el.aboutVersion.textContent = state.status.version ? `Re:Solution ${state.status.version}` : '';
  }

  /* ---------- Dialogs ---------- */

  function bindDialogs() {
    for (const dialog of $$('dialog')) {
      dialog.addEventListener('click', (event) => {
        if (event.target.closest('[data-close]')) dialog.close();
      });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
