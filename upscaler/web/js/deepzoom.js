/* The compare viewer for finished pictures: deep zoom over the result's tile pyramid.

   Fit shows the whole picture. 100% shows one picture pixel per screen pixel (it allows for the
   screen's devicePixelRatio, so it is honest on Retina screens too) and 200% two. Free zoom with the
   wheel, a trackpad pinch or two fingers keeps the point under the cursor or fingers where it is.
   Only the tiles in view are loaded, from the level that keeps them sharp, with the smallest level
   underneath (soft) while they arrive; tiles that leave the view before they arrive are cancelled.
   Before and After are drawn on the same grid, pixel for pixel: as a slider (drag the divider, click
   or tap to move it, or use its keys) or side by side with one shared view. When zoomed in, a small
   overview map shows where you are and can be dragged.

   Everything a viewer starts (animation frames, timers, tile and manifest requests, listeners on the
   window, the resize observer) belongs to one lifecycle scope (js/lifecycle.js), so destroy() ends
   all of it in one go, however many times it is called.

   Plain browser script; js/viewer.js makes one per picture shown. The geometry helpers at the top are
   also exported for tests run under Node. */
(function () {
  'use strict';

  const MODE_KEY = 'resolution.compareMode'; // Slider or Side by side, remembered per browser
  const MODES = ['slider', 'side'];
  const MAX_SCALE = 8; // 800%: screen pixels per picture pixel
  const KEY_ZOOM = 1.5;
  const WHEEL_ZOOM = 0.0018; // per wheel pixel; a mouse notch (about 100 px) is about 17%
  const PINCH_ZOOM = 0.012; // trackpad pinches arrive as small ctrl+wheel steps
  const WHEEL_EASE = 0.32; // share of the way to the wanted zoom per frame (smooth, but quick)
  const ANIMATE_MS = 240;
  const PAN_KEY_SHARE = 0.18; // of the view, per arrow key
  const HANDLE_REACH = 24; // CSS px either side of the divider that grab it
  const TAP_SLOP = 6; // CSS px a press may move and still be a click or tap
  const DOUBLE_TAP_MS = 320;
  const IN_FLIGHT = 6; // tiles requested at once, per side
  const RETRY_MS = 4000;
  const LOADING_DELAY_MS = 250; // "Loading detail" only shows when it takes a moment
  const MAP_MAX = 168; // the overview map's longer side, CSS px
  let counter = 0;

  /* ---------- Geometry (pure, also run under Node) ---------- */

  /** The scale (screen pixels per picture pixel) that shows the whole picture in a pane. */
  function fitScale(width, height, paneW, paneH) {
    if (!(width > 0 && height > 0 && paneW > 0 && paneH > 0)) return 1;
    return Math.min(paneW / width, paneH / height);
  }

  /** The pyramid level to draw at `scale`: the smallest one that still gives at least one of its
      pixels per screen pixel (level n is about 2^n times smaller than the picture). */
  function levelFor(scale, levelCount) {
    if (!(levelCount > 0)) return 0;
    if (!(scale > 0)) return levelCount - 1;
    const level = Math.floor(Math.log2(1 / scale) + 1e-9);
    return Math.max(0, Math.min(levelCount - 1, level));
  }

  /** A view {scale, cx, cy} (cx, cy: the picture point in the middle of the pane) kept on the
      picture: centred along a side the picture doesn't fill, edge to edge along one it does. */
  function clampView(view, width, height, paneW, paneH) {
    const s = view.scale;
    const halfW = paneW / (2 * s);
    const halfH = paneH / (2 * s);
    const cx = width <= 2 * halfW ? width / 2 : Math.min(width - halfW, Math.max(halfW, view.cx));
    const cy = height <= 2 * halfH ? height / 2 : Math.min(height - halfH, Math.max(halfH, view.cy));
    return { scale: s, cx, cy };
  }

  /** `view` zoomed to `scale`, with the picture point under (px, py) (pane pixels) left where it is. */
  function zoomAround(view, scale, px, py, paneW, paneH) {
    const ix = view.cx + (px - paneW / 2) / view.scale;
    const iy = view.cy + (py - paneH / 2) / view.scale;
    return { scale, cx: ix - (px - paneW / 2) / scale, cy: iy - (py - paneH / 2) / scale };
  }

  /** The picture point (x, y) under pane pixel (px, py). */
  function pictureAt(view, px, py, paneW, paneH) {
    return { x: view.cx + (px - paneW / 2) / view.scale, y: view.cy + (py - paneH / 2) / view.scale };
  }

  /** Tiles (col, row) of `level` that a pane shows, nearest its middle first. */
  function visibleTiles(view, level, width, height, tileSize, paneW, paneH) {
    if (!level || !(view.scale > 0)) return [];
    const fx = width / level.width;
    const fy = height / level.height;
    const left = (view.cx - paneW / (2 * view.scale)) / fx;
    const right = (view.cx + paneW / (2 * view.scale)) / fx;
    const top = (view.cy - paneH / (2 * view.scale)) / fy;
    const bottom = (view.cy + paneH / (2 * view.scale)) / fy;
    const c0 = Math.max(0, Math.floor(left / tileSize));
    const c1 = Math.min(level.cols - 1, Math.floor((right - 1e-6) / tileSize));
    const r0 = Math.max(0, Math.floor(top / tileSize));
    const r1 = Math.min(level.rows - 1, Math.floor((bottom - 1e-6) / tileSize));
    const midX = view.cx / fx / tileSize - 0.5;
    const midY = view.cy / fy / tileSize - 0.5;
    const tiles = [];
    for (let row = r0; row <= r1; row += 1) {
      for (let col = c0; col <= c1; col += 1) tiles.push({ col, row, d: (col - midX) ** 2 + (row - midY) ** 2 });
    }
    return tiles.sort((a, b) => a.d - b.d).map(({ col, row }) => ({ col, row }));
  }

  /** What the zoom readout says: "Fit · 23%", "100% · actual pixels", "200%", "37%". */
  function zoomLabel(scale, fitted) {
    const pct = Math.round(scale * 100);
    if (fitted) return `Fit · ${pct}%`;
    if (pct === 100) return '100% · actual pixels';
    return `${pct}%`;
  }

  /** Side by side as two columns or two rows: whichever shows the picture bigger. */
  function sideLayout(width, height, stageW, stageH, gap = 0) {
    const columns = fitScale(width, height, (stageW - gap) / 2, stageH);
    const rows = fitScale(width, height, stageW, (stageH - gap) / 2);
    return rows > columns ? 'rows' : 'columns';
  }

  function readMode(storage) {
    try {
      const saved = storage && storage.getItem(MODE_KEY);
      return MODES.includes(saved) ? saved : 'slider';
    } catch {
      return 'slider';
    }
  }

  function saveMode(storage, mode) {
    try {
      if (storage) storage.setItem(MODE_KEY, mode);
    } catch {
      /* Private windows can refuse storage; the choice still holds for this visit. */
    }
  }

  const helpers = { fitScale, levelFor, clampView, zoomAround, pictureAt, visibleTiles, zoomLabel, sideLayout, readMode, saveMode, MODE_KEY, MAX_SCALE };
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (typeof document === 'undefined') return;

  /* ---------- The viewer ---------- */

  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const touchFirst = () => window.matchMedia('(hover: none) and (pointer: coarse)').matches;
  const now = () => performance.now();

  function storage() {
    try {
      return window.localStorage;
    } catch {
      return null;
    }
  }

  function context2d(canvas) {
    // Wide-gamut screens get a wide-gamut canvas, so Display P3 pictures keep their most vivid colours.
    const wide = window.matchMedia('(color-gamut: p3)').matches;
    try {
      if (wide) {
        const ctx = canvas.getContext('2d', { colorSpace: 'display-p3' });
        if (ctx) return ctx;
      }
    } catch {
      /* Older browsers: the ordinary canvas below. */
    }
    return canvas.getContext('2d');
  }

  function decode(blob) {
    if (typeof createImageBitmap === 'function') return createImageBitmap(blob);
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('decode')); };
      img.src = url;
    });
  }

  /**
   * deps: {h, icon}
   * options: {
   *   manifest: URL of viewer.json, tile: (side 'before'|'after', level, col, row) -> URL,
   *   name: what the picture is called, compare: show Before as well (false for nothing to compare),
   *   before: {text}, after: {text}, single: {text} (the label when there's only one side),
   *   wipe: play the before-to-after reveal once, onUnavailable: called if there's no pyramid to show,
   *   onReady: (manifest) called once the pyramid is known
   * }
   */
  function create(deps, options) {
    const { h, icon } = deps;
    const life = deps.life || window.ReLife;
    const scope = life.createScope('deep-zoom');
    const id = `zv-${++counter}`;
    const tiles = { before: new Map(), after: new Map() };
    const inFlight = { before: 0, after: 0 };
    const state = {
      manifest: null, W: 0, H: 0, tileSize: 512, levels: [],
      view: { scale: 1, cx: 0, cy: 0 }, fitted: true,
      mode: readMode(storage()), compare: options.compare !== false, pos: 0.5,
      dpr: window.devicePixelRatio || 1, paneW: 1, paneH: 1,
      anim: null, wheel: null, frame: 0, scheduled: false, destroyed: false,
      loadingSince: 0, loadingTimer: 0, cacheLimit: touchFirst() ? 96 : 200, wiped: !options.wipe,
    };
    let panes = [];
    let gesture = null;
    let lastTap = null;
    const pointers = new Map();

    /* ---- Building blocks ---- */

    const modeGroup = h('div', { class: 'segmented zv-modes', role: 'radiogroup', 'aria-label': 'How to compare' },
      [['slider', 'arrows-h', 'Slider'], ['side', 'columns', 'Side by side']].map(([value, glyph, label]) => h('label', { title: label },
        h('input', { type: 'radio', name: `${id}-mode`, value, checked: value === state.mode }),
        h('span', null, icon(glyph), h('span', { class: 'zv-mode-word' }, label)))));
    const zoomButton = (value, label, extra = {}) => h('button', { type: 'button', class: 'zv-zoom-btn', 'data-zoom': value, 'aria-pressed': 'false', ...extra }, label);
    const fitButton = zoomButton('fit', 'Fit', { title: 'Show the whole picture (0)' });
    const oneButton = zoomButton('1', '100%', { title: 'Actual pixels: one picture pixel per screen pixel (1)' });
    const twoButton = zoomButton('2', '200%', { title: 'Twice actual size (2)' });
    const zoomOut = h('button', { type: 'button', class: 'icon-button zv-step', 'aria-label': 'Zoom out', title: 'Zoom out (−)' }, icon('minus'));
    const zoomIn = h('button', { type: 'button', class: 'icon-button zv-step', 'aria-label': 'Zoom in', title: 'Zoom in (+)' }, icon('plus'));
    const readout = h('span', { class: 'zv-readout', id: `${id}-zoom` }, '');
    const zoomGroup = h('div', { class: 'zv-zoom', role: 'group', 'aria-label': 'Zoom' },
      h('div', { class: 'zv-presets' }, fitButton, oneButton, twoButton),
      h('div', { class: 'zv-stepper' }, zoomOut, readout, zoomIn));
    const toolbar = h('div', { class: 'zv-toolbar' }, modeGroup, zoomGroup);

    const handle = h('div', {
      class: 'zv-handle', role: 'slider', tabindex: '0', 'aria-label': 'Divider between before and after',
      'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '50', 'aria-orientation': 'horizontal',
    }, h('span', { class: 'zv-knob' }, icon('arrows-h')));
    const mapCanvas = h('canvas', { class: 'zv-map-canvas' });
    const mapView = h('span', { class: 'zv-map-view' });
    const minimap = h('div', { class: 'zv-minimap', 'aria-hidden': 'true', title: 'Drag to move around the picture' }, mapCanvas, mapView);
    const loading = h('div', { class: 'zv-loading', 'aria-hidden': 'true' }, h('span', { class: 'zv-loading-dot' }), 'Loading detail');
    const message = h('div', { class: 'zv-message', hidden: true });
    const hintText = touchFirst()
      ? 'Pinch to zoom, drag to look around, double-tap for 100%.'
      : 'Scroll or pinch to zoom, drag to look around, double-click for 100%. Keys: + − 0 1 2 and the arrows.';
    const hint = h('p', { class: 'zv-hint', id: `${id}-hint` }, hintText);
    const live = h('span', { class: 'sr-only', 'aria-live': 'polite' });
    const stage = h('div', {
      class: 'zv-stage is-loading', tabindex: '0', role: 'group', 'aria-roledescription': 'zoomable picture',
      'aria-label': `${options.name || 'Picture'}: before and after`, 'aria-describedby': `${id}-hint`,
    }, handle, minimap, loading, message);
    const root = h('div', { class: 'zv', 'data-mode': state.mode }, toolbar, stage, hint, live);

    /* ---- Panes ---- */

    function sides() {
      if (!state.compare) return ['after'];
      return state.mode === 'side' ? ['before', 'after'] : ['both'];
    }

    function tag(side) {
      const text = !state.compare ? (options.single || options.after || {}).text
        : side === 'before' ? (options.before || {}).text : (options.after || {}).text;
      if (!text) return null;
      return h('span', { class: `zv-tag zv-tag-${side}`, 'data-tag': side }, text);
    }

    function buildPanes() {
      for (const pane of panes) pane.el.remove();
      panes = sides().map((side) => {
        const canvas = h('canvas', { class: 'zv-canvas', 'aria-hidden': 'true' });
        const labels = side === 'both' ? [tag('before'), tag('after')] : [tag(side)];
        const el = h('div', { class: 'zv-pane', 'data-side': side }, canvas, ...labels);
        return { el, canvas, ctx: context2d(canvas), side, w: 1, h: 1 };
      });
      stage.prepend(...panes.map((pane) => pane.el));
      const mode = state.compare ? state.mode : 'single';
      root.dataset.mode = mode;
      stage.dataset.mode = mode;
      handle.hidden = mode !== 'slider';
      modeGroup.hidden = !state.compare;
      measure();
    }

    function measure() {
      if (state.destroyed) return;
      state.dpr = window.devicePixelRatio || 1;
      // Layout sizes, not getBoundingClientRect: while the viewer opens it is scaled (0.985) for a moment,
      // and a canvas sized from that on-screen box stayed a little smaller than its pane, so Fit showed
      // the picture off-centre with a gap at the right and bottom.
      const box = { width: stage.clientWidth, height: stage.clientHeight };
      if (state.compare && state.mode === 'side' && state.W) {
        stage.dataset.layout = sideLayout(state.W, state.H, box.width, box.height, 8);
      }
      for (const pane of panes) {
        pane.w = Math.max(1, Math.round(pane.el.clientWidth * state.dpr));
        pane.h = Math.max(1, Math.round(pane.el.clientHeight * state.dpr));
        if (pane.canvas.width !== pane.w) pane.canvas.width = pane.w;
        if (pane.canvas.height !== pane.h) pane.canvas.height = pane.h;
        // The canvas is exactly as many screen pixels as it has, so 100% stays sharp.
        pane.canvas.style.width = `${pane.w / state.dpr}px`;
        pane.canvas.style.height = `${pane.h / state.dpr}px`;
      }
      state.paneW = panes.length ? panes[0].w : 1;
      state.paneH = panes.length ? panes[0].h : 1;
      if (!state.manifest) return;
      if (state.fitted) state.view = fitView();
      else state.view = clamp(state.view);
      render();
    }

    /* ---- The view ---- */

    const fit = () => fitScale(state.W, state.H, state.paneW, state.paneH);
    const minScale = () => Math.min(fit(), 1);
    const maxScale = () => Math.max(MAX_SCALE, fit() * 2);
    const clampScale = (s) => Math.min(maxScale(), Math.max(minScale(), s));
    const clamp = (view) => clampView({ ...view, scale: clampScale(view.scale) }, state.W, state.H, state.paneW, state.paneH);
    const fitView = () => ({ scale: fit(), cx: state.W / 2, cy: state.H / 2 });

    function pannable() {
      const s = state.view.scale;
      return state.W * s > state.paneW + 0.5 || state.H * s > state.paneH + 0.5;
    }

    function setView(next, { animate = false, fitted = false } = {}) {
      const target = fitted ? fitView() : clamp(next);
      state.fitted = fitted;
      state.wheel = null;
      if (animate && !reducedMotion()) state.anim = { from: { ...state.view }, to: target, start: now() };
      else {
        state.anim = null;
        state.view = target;
      }
      render();
    }

    /** Where the view is heading: the end of a zoom that is still moving, or where it is now. */
    function heading() {
      if (state.anim) return state.anim.to;
      if (state.wheel) return clamp(zoomAround(state.view, state.wheel.target, state.wheel.px, state.wheel.py, state.paneW, state.paneH));
      return state.view;
    }

    function zoomTo(scale, { animate = true, at = null } = {}) {
      const px = at ? at.x : state.paneW / 2;
      const py = at ? at.y : state.paneH / 2;
      const from = at ? state.view : heading();
      setView(zoomAround(from, clampScale(scale), px, py, state.paneW, state.paneH), { animate });
      say(zoomLabel(clampScale(scale), false));
    }

    function zoomStep(factor) {
      const next = heading().scale * factor;
      if (factor < 1 && next <= fit() * 1.001) fitAll();
      else zoomTo(next);
    }

    function fitAll(animate = true) {
      setView(null, { animate, fitted: true });
      say(zoomLabel(fit(), true));
    }

    function toggleAt(at) {
      if (state.fitted || state.view.scale < 0.99) zoomTo(1, { at });
      else fitAll();
    }

    function stepAnimations(time) {
      if (state.anim) {
        const { from, to, start } = state.anim;
        const t = Math.min(1, (time - start) / ANIMATE_MS);
        const e = 1 - (1 - t) ** 3; // ease-out cubic
        const scale = Math.exp(Math.log(from.scale) + (Math.log(to.scale) - Math.log(from.scale)) * e);
        state.view = { scale, cx: from.cx + (to.cx - from.cx) * e, cy: from.cy + (to.cy - from.cy) * e };
        if (t >= 1) {
          state.view = to;
          state.anim = null;
        }
      }
      if (state.wheel) {
        const { target, px, py } = state.wheel;
        const current = state.view.scale;
        const next = Math.exp(Math.log(current) + (Math.log(target) - Math.log(current)) * WHEEL_EASE);
        const done = Math.abs(Math.log(target / next)) < 0.002;
        state.view = clamp(zoomAround(state.view, done ? target : next, px, py, state.paneW, state.paneH));
        if (done) state.wheel = null;
      }
      return Boolean(state.anim || state.wheel);
    }

    /* ---- Drawing ---- */

    function render() {
      if (state.scheduled || state.destroyed) return;
      state.scheduled = true;
      scope.frame(frame);
    }

    function frame(time) {
      state.scheduled = false;
      if (state.destroyed || !state.manifest) return;
      const moving = stepAnimations(time);
      state.frame += 1;
      for (const pane of panes) draw(pane);
      drawMap();
      updateChrome();
      schedule();
      if (moving) render();
    }

    function offset(pane) {
      const s = state.view.scale;
      let ox = pane.w / 2 - state.view.cx * s;
      let oy = pane.h / 2 - state.view.cy * s;
      if (Math.abs(s - Math.round(s)) < 1e-6) { // 100%, 200%…: picture pixels land on whole screen pixels
        ox = Math.round(ox);
        oy = Math.round(oy);
      }
      return { ox, oy };
    }

    function draw(pane) {
      const { ctx } = pane;
      if (!ctx) return;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, pane.w, pane.h);
      const { ox, oy } = offset(pane);
      const target = levelFor(state.view.scale, state.levels.length);
      if (pane.side !== 'both') {
        drawSide(ctx, pane.side, ox, oy, target, pane);
        return;
      }
      drawSide(ctx, 'after', ox, oy, target, pane);
      const split = Math.round(state.pos * pane.w);
      if (split <= 0) return;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, split, pane.h);
      ctx.clip();
      drawSide(ctx, 'before', ox, oy, target, pane);
      ctx.restore();
    }

    function drawSide(ctx, side, ox, oy, target, pane) {
      const cache = tiles[side];
      const s = state.view.scale;
      const size = state.tileSize;
      // The smallest level first (always there, soft when enlarged), then sharper ones over it.
      for (let lv = state.levels.length - 1; lv >= target; lv -= 1) {
        const level = state.levels[lv];
        const fx = state.W / level.width;
        const fy = state.H / level.height;
        const smooth = lv > target || s * fx < 2; // at 200% and beyond, show the real pixels
        ctx.imageSmoothingEnabled = smooth;
        if (smooth) ctx.imageSmoothingQuality = 'high';
        for (const { col, row } of visibleTiles(state.view, level, state.W, state.H, size, pane.w, pane.h)) {
          const entry = cache.get(`${lv}/${col}_${row}`);
          if (!entry || !entry.image) continue;
          entry.used = state.frame;
          const x0 = col * size;
          const y0 = row * size;
          const x1 = Math.min(x0 + size, level.width);
          const y1 = Math.min(y0 + size, level.height);
          // Edges are rounded the same way for neighbouring tiles, so they meet without a seam.
          const dx0 = Math.round(ox + x0 * fx * s);
          const dy0 = Math.round(oy + y0 * fy * s);
          const dx1 = Math.round(ox + x1 * fx * s);
          const dy1 = Math.round(oy + y1 * fy * s);
          if (dx1 > dx0 && dy1 > dy0) ctx.drawImage(entry.image, dx0, dy0, dx1 - dx0, dy1 - dy0);
        }
      }
    }

    function drawMap() {
      const show = state.manifest && pannable() && !state.fitted;
      minimap.classList.toggle('is-shown', Boolean(show));
      if (!show) return;
      const k = MAP_MAX / Math.max(state.W, state.H) * (window.innerWidth < 600 ? 0.66 : 1);
      const w = Math.max(24, Math.round(state.W * k));
      const hgt = Math.max(24, Math.round(state.H * k));
      const dpr = state.dpr;
      if (mapCanvas.width !== Math.round(w * dpr)) mapCanvas.width = Math.round(w * dpr);
      if (mapCanvas.height !== Math.round(hgt * dpr)) mapCanvas.height = Math.round(hgt * dpr);
      mapCanvas.style.width = `${w}px`;
      mapCanvas.style.height = `${hgt}px`;
      const ctx = mapCanvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      const lv = state.levels.length - 1;
      const level = state.levels[lv];
      const sx = mapCanvas.width / level.width;
      const sy = mapCanvas.height / level.height;
      for (let row = 0; row < level.rows; row += 1) {
        for (let col = 0; col < level.cols; col += 1) {
          const entry = tiles.after.get(`${lv}/${col}_${row}`);
          if (!entry || !entry.image) continue;
          const x0 = col * state.tileSize;
          const y0 = row * state.tileSize;
          const x1 = Math.min(x0 + state.tileSize, level.width);
          const y1 = Math.min(y0 + state.tileSize, level.height);
          ctx.drawImage(entry.image, x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
        }
      }
      const s = state.view.scale;
      const left = Math.max(0, (state.view.cx - state.paneW / (2 * s)) / state.W);
      const top = Math.max(0, (state.view.cy - state.paneH / (2 * s)) / state.H);
      const right = Math.min(1, (state.view.cx + state.paneW / (2 * s)) / state.W);
      const bottom = Math.min(1, (state.view.cy + state.paneH / (2 * s)) / state.H);
      mapView.style.left = `${left * w}px`;
      mapView.style.top = `${top * hgt}px`;
      mapView.style.width = `${Math.max(4, (right - left) * w)}px`;
      mapView.style.height = `${Math.max(4, (bottom - top) * hgt)}px`;
    }

    function updateChrome() {
      const s = state.view.scale;
      readout.textContent = zoomLabel(s, state.fitted);
      const at = (value) => !state.fitted && !state.anim && Math.abs(s - value) < 0.005;
      fitButton.setAttribute('aria-pressed', String(state.fitted));
      oneButton.setAttribute('aria-pressed', String(at(1)));
      twoButton.setAttribute('aria-pressed', String(at(2)));
      zoomOut.disabled = s <= minScale() * 1.001;
      zoomIn.disabled = s >= maxScale() * 0.999;
      const canPan = pannable();
      stage.classList.toggle('is-pannable', canPan);
      root.dataset.zoomed = String(!state.fitted);
      if (state.compare && state.mode === 'slider' && panes[0]) {
        const pct = Math.round(state.pos * 100);
        handle.style.left = `${state.pos * 100}%`;
        handle.setAttribute('aria-valuenow', String(pct));
        handle.setAttribute('aria-valuetext', `${pct}% before, ${100 - pct}% after`);
        const width = panes[0].w / state.dpr;
        const split = state.pos * width;
        for (const label of panes[0].el.querySelectorAll('.zv-tag')) {
          const covered = label.dataset.tag === 'before' ? split < label.offsetWidth + 20 : width - split < label.offsetWidth + 20;
          label.classList.toggle('is-covered', covered);
        }
      }
      const waiting = targetPending();
      if (waiting && !state.loadingSince) state.loadingSince = now();
      if (!waiting) state.loadingSince = 0;
      const slow = Boolean(state.loadingSince) && now() - state.loadingSince > LOADING_DELAY_MS;
      loading.classList.toggle('is-shown', slow);
      if (waiting && !slow && !state.loadingTimer) {
        state.loadingTimer = scope.timeout(() => {
          state.loadingTimer = 0;
          render();
        }, LOADING_DELAY_MS);
      }
      stage.setAttribute('aria-busy', String(waiting));
    }

    function shownSides() {
      return state.compare ? ['before', 'after'] : ['after'];
    }

    function targetPending() {
      const target = levelFor(state.view.scale, state.levels.length);
      const level = state.levels[target];
      if (!level) return false;
      for (const side of shownSides()) {
        for (const { col, row } of visibleTiles(state.view, level, state.W, state.H, state.tileSize, state.paneW, state.paneH)) {
          const entry = tiles[side].get(`${target}/${col}_${row}`);
          if (!entry || entry.state === 'loading') return true;
        }
      }
      return false;
    }

    /* ---- Tiles ---- */

    function schedule() {
      const base = state.levels.length - 1;
      const target = levelFor(state.view.scale, state.levels.length);
      for (const side of shownSides()) {
        const wanted = [];
        const baseLevel = state.levels[base];
        for (let row = 0; row < baseLevel.rows; row += 1) for (let col = 0; col < baseLevel.cols; col += 1) wanted.push([base, col, row]);
        if (target !== base) {
          for (const { col, row } of visibleTiles(state.view, state.levels[target], state.W, state.H, state.tileSize, state.paneW, state.paneH)) {
            wanted.push([target, col, row]);
          }
        }
        const keys = new Set(wanted.map(([lv, col, row]) => `${lv}/${col}_${row}`));
        // Tiles that left the view before they arrived aren't wanted any more: stop them.
        for (const [key, entry] of tiles[side]) {
          if (entry.state === 'loading' && !keys.has(key)) {
            if (entry.ctrl) entry.ctrl.abort();
            tiles[side].delete(key);
          }
        }
        for (const [lv, col, row] of wanted) {
          if (inFlight[side] >= IN_FLIGHT) break;
          const key = `${lv}/${col}_${row}`;
          const entry = tiles[side].get(key);
          if (entry && (entry.state !== 'failed' || now() - entry.failedAt < RETRY_MS)) continue;
          load(side, lv, col, row, key);
        }
      }
    }

    function load(side, lv, col, row, key) {
      const ctrl = scope.controller(); // new AbortController(), which closing the viewer aborts too
      const entry = { state: 'loading', image: null, ctrl, used: state.frame, lv, failedAt: 0 };
      tiles[side].set(key, entry);
      inFlight[side] += 1;
      fetch(options.tile(side, lv, col, row), { credentials: 'same-origin', signal: ctrl.signal })
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.blob();
        })
        .then(decode)
        .then((image) => {
          if (state.destroyed || tiles[side].get(key) !== entry) {
            if (image.close) image.close();
            return;
          }
          entry.image = image;
          entry.state = 'ready';
          evict(side);
        })
        .catch((err) => {
          if (err && err.name === 'AbortError') return;
          entry.state = 'failed';
          entry.failedAt = now();
        })
        .finally(() => {
          scope.done(ctrl);
          inFlight[side] -= 1;
          if (state.destroyed) return;
          stage.classList.toggle('is-loading', !baseReady());
          maybeWipe();
          render();
        });
    }

    function baseReady() {
      const base = state.levels.length - 1;
      return shownSides().every((side) => {
        for (let row = 0; row < state.levels[base].rows; row += 1) {
          for (let col = 0; col < state.levels[base].cols; col += 1) {
            const entry = tiles[side].get(`${base}/${col}_${row}`);
            if (!entry || entry.state !== 'ready') return false;
          }
        }
        return true;
      });
    }

    function evict(side) {
      const cache = tiles[side];
      if (cache.size <= state.cacheLimit) return;
      const base = state.levels.length - 1;
      const old = [...cache.entries()]
        .filter(([, entry]) => entry.state !== 'loading' && entry.lv !== base && entry.used < state.frame)
        .sort((a, b) => a[1].used - b[1].used);
      for (const [key, entry] of old) {
        if (cache.size <= state.cacheLimit) break;
        if (entry.image && entry.image.close) entry.image.close();
        cache.delete(key);
      }
    }

    /* ---- The divider ---- */

    function setPos(value) {
      state.pos = Math.min(1, Math.max(0, value));
      render();
    }

    function posFromClient(clientX) {
      const pane = panes[0];
      if (!pane) return state.pos;
      const rect = pane.el.getBoundingClientRect();
      return (clientX - rect.left) / Math.max(1, rect.width);
    }

    function nearHandle(clientX) {
      if (!state.compare || state.mode !== 'slider' || !panes[0]) return false;
      const rect = panes[0].el.getBoundingClientRect();
      return Math.abs(clientX - (rect.left + state.pos * rect.width)) <= HANDLE_REACH;
    }

    function maybeWipe() {
      if (state.wiped || !baseReady() || !state.compare || state.mode !== 'slider') return;
      state.wiped = true;
      if (reducedMotion()) return;
      const started = now();
      const step = () => {
        if (state.destroyed || gesture) return;
        const t = Math.min(1, (now() - started) / 760);
        state.pos = 0.92 - 0.42 * (1 - (1 - t) ** 4); // ease-out quart, from mostly Before to the middle
        render();
        if (t < 1) scope.frame(step);
      };
      scope.frame(step);
    }

    handle.addEventListener('keydown', (event) => {
      const fine = event.shiftKey ? 0.01 : 0.05;
      const moves = { ArrowLeft: -fine, ArrowDown: -fine, ArrowRight: fine, ArrowUp: fine, PageDown: -0.1, PageUp: 0.1 };
      let next = null;
      if (event.key in moves) next = state.pos + moves[event.key];
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = 1;
      if (next == null) return;
      event.preventDefault();
      event.stopPropagation();
      state.wiped = true;
      setPos(next);
    });

    /* ---- Pointer, wheel and touch ---- */

    function local(event) {
      const pane = panes[0];
      const rect = (pane ? pane.el : stage).getBoundingClientRect();
      // In side by side, the same spot in either pane is the same picture point.
      const box = paneUnder(event) || rect;
      return { x: (event.clientX - box.left) * state.dpr, y: (event.clientY - box.top) * state.dpr };
    }

    function paneUnder(event) {
      for (const pane of panes) {
        const rect = pane.el.getBoundingClientRect();
        if (event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom) return rect;
      }
      return null;
    }

    function stopMotion() {
      state.anim = null;
      state.wheel = null;
      state.wiped = true;
    }

    stage.addEventListener('pointerdown', (event) => {
      if (!state.manifest || (event.pointerType === 'mouse' && event.button !== 0)) return;
      if (event.target.closest('.zv-minimap')) return;
      // A mouse has one pointer: anything left over is from a press whose release never arrived
      // (let go outside the window, say). Starting afresh keeps it from turning into a phantom pinch.
      if (event.pointerType === 'mouse' || event.isPrimary) {
        pointers.clear();
        gesture = null;
      }
      try {
        stage.setPointerCapture(event.pointerId);
      } catch {
        /* The press still works while the pointer stays over the picture. */
      }
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      stopMotion();
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const mid = local({ clientX: (a.x + b.x) / 2, clientY: (a.y + b.y) / 2 });
        gesture = { kind: 'pinch', dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, mid, view: { ...state.view } };
        stage.classList.remove('is-dragging', 'is-panning');
        return;
      }
      if (pointers.size > 2) return;
      const divider = nearHandle(event.clientX);
      gesture = { kind: divider ? 'divider' : 'press', x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, moved: false };
      if (divider) {
        stage.classList.add('is-dragging');
        setPos(posFromClient(event.clientX));
      }
    });

    stage.addEventListener('pointermove', (event) => {
      if (!pointers.has(event.pointerId)) {
        if (event.pointerType === 'mouse' && state.manifest) stage.classList.toggle('is-near-handle', nearHandle(event.clientX));
        return;
      }
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (!gesture) return;
      if (gesture.kind === 'pinch') {
        if (pointers.size < 2) return;
        const [a, b] = [...pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y) || 1;
        const mid = local({ clientX: (a.x + b.x) / 2, clientY: (a.y + b.y) / 2 });
        const scale = clampScale(gesture.view.scale * (dist / gesture.dist));
        // The picture point under the fingers when they landed stays under them (that pans as well).
        const spot = pictureAt(gesture.view, gesture.mid.x, gesture.mid.y, state.paneW, state.paneH);
        setView({ scale, cx: spot.x - (mid.x - state.paneW / 2) / scale, cy: spot.y - (mid.y - state.paneH / 2) / scale });
        return;
      }
      if (gesture.kind === 'divider') {
        setPos(posFromClient(event.clientX));
        return;
      }
      const dx = event.clientX - gesture.lastX;
      const dy = event.clientY - gesture.lastY;
      if (!gesture.moved && Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) < TAP_SLOP) return;
      gesture.moved = true;
      gesture.lastX = event.clientX;
      gesture.lastY = event.clientY;
      if (pannable()) {
        gesture.kind = 'press';
        stage.classList.add('is-panning');
        const s = state.view.scale;
        setView({ scale: s, cx: state.view.cx - (dx * state.dpr) / s, cy: state.view.cy - (dy * state.dpr) / s });
      } else if (state.compare && state.mode === 'slider') {
        gesture.kind = 'divider'; // nothing to pan at Fit: a drag moves the divider instead
        stage.classList.add('is-dragging');
        setPos(posFromClient(event.clientX));
      }
    });

    function endPointer(event) {
      if (!pointers.has(event.pointerId)) return;
      pointers.delete(event.pointerId);
      if (!gesture) return;
      if (gesture.kind === 'pinch') {
        if (pointers.size === 1) { // one finger lifted: the other carries on panning
          const [rest] = [...pointers.values()];
          gesture = { kind: 'press', x: rest.x, y: rest.y, lastX: rest.x, lastY: rest.y, moved: true };
        } else if (!pointers.size) gesture = null;
        return;
      }
      if (pointers.size) return;
      const tap = gesture.kind === 'press' && !gesture.moved && event.type === 'pointerup';
      gesture = null;
      stage.classList.remove('is-dragging', 'is-panning');
      if (!tap) return;
      if (state.compare && state.mode === 'slider') setPos(posFromClient(event.clientX)); // click or tap moves the divider
      if (event.pointerType !== 'mouse') {
        const at = local(event);
        if (lastTap && now() - lastTap.time < DOUBLE_TAP_MS && Math.hypot(event.clientX - lastTap.x, event.clientY - lastTap.y) < 30) {
          lastTap = null;
          toggleAt(at);
        } else {
          lastTap = { time: now(), x: event.clientX, y: event.clientY };
        }
      }
    }
    stage.addEventListener('pointerup', endPointer);
    stage.addEventListener('pointercancel', endPointer);
    stage.addEventListener('lostpointercapture', endPointer); // released somewhere the stage never heard about

    stage.addEventListener('dblclick', (event) => {
      if (!state.manifest || event.target.closest('.zv-minimap, .zv-handle')) return;
      event.preventDefault();
      toggleAt(local(event));
    });

    stage.addEventListener('wheel', (event) => {
      if (!state.manifest) return;
      event.preventDefault();
      state.anim = null;
      state.wiped = true;
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
      const delta = event.deltaY * unit;
      const factor = Math.exp(-delta * (event.ctrlKey ? PINCH_ZOOM : WHEEL_ZOOM));
      const at = local(event);
      const from = state.wheel ? state.wheel.target : state.view.scale;
      const target = clampScale(from * factor);
      state.fitted = false;
      if (reducedMotion() || event.ctrlKey || Math.abs(delta) < 40) {
        // Trackpads send many small steps: follow them directly, they're already smooth.
        state.wheel = null;
        state.view = clamp(zoomAround(state.view, target, at.x, at.y, state.paneW, state.paneH));
      } else {
        state.wheel = { target, px: at.x, py: at.y };
      }
      if (Math.abs(target - fit()) / fit() < 0.01 && target <= fit()) state.fitted = true;
      render();
    }, { passive: false });

    // Safari's own trackpad pinch (it sends gesture events instead of ctrl+wheel).
    let gestureStart = null;
    stage.addEventListener('gesturestart', (event) => {
      if (pointers.size > 1) return;
      event.preventDefault();
      gestureStart = { scale: state.view.scale };
    });
    stage.addEventListener('gesturechange', (event) => {
      if (!gestureStart || pointers.size > 1) return;
      event.preventDefault();
      state.fitted = false;
      const at = local(event);
      state.view = clamp(zoomAround(state.view, clampScale(gestureStart.scale * event.scale), at.x, at.y, state.paneW, state.paneH));
      render();
    });
    stage.addEventListener('gestureend', () => { gestureStart = null; });

    /* ---- The overview map ---- */

    let mapDrag = false;
    function moveToMap(event) {
      const rect = mapCanvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * state.W;
      const y = ((event.clientY - rect.top) / Math.max(1, rect.height)) * state.H;
      setView({ scale: state.view.scale, cx: x, cy: y });
    }
    minimap.addEventListener('pointerdown', (event) => {
      event.stopPropagation();
      event.preventDefault();
      stopMotion();
      mapDrag = true;
      try {
        minimap.setPointerCapture(event.pointerId);
      } catch {
        /* Dragging still works while the pointer stays over the map. */
      }
      minimap.classList.add('is-dragging');
      moveToMap(event);
    });
    minimap.addEventListener('pointermove', (event) => { if (mapDrag) moveToMap(event); });
    const endMap = () => {
      mapDrag = false;
      minimap.classList.remove('is-dragging');
    };
    minimap.addEventListener('pointerup', endMap);
    minimap.addEventListener('pointercancel', endMap);
    minimap.addEventListener('lostpointercapture', endMap);

    /* ---- Toolbar ---- */

    fitButton.addEventListener('click', () => fitAll());
    oneButton.addEventListener('click', () => zoomTo(1));
    twoButton.addEventListener('click', () => zoomTo(2));
    zoomIn.addEventListener('click', () => zoomStep(KEY_ZOOM));
    zoomOut.addEventListener('click', () => zoomStep(1 / KEY_ZOOM));
    modeGroup.addEventListener('change', (event) => {
      if (event.target.name !== `${id}-mode` || !MODES.includes(event.target.value)) return;
      state.mode = event.target.value;
      saveMode(storage(), state.mode);
      buildPanes();
      say(state.mode === 'side' ? 'Side by side' : 'Slider');
    });

    /* ---- Keys (js/viewer.js passes them on) ---- */

    function handleKey(event) {
      if (!state.manifest || event.altKey || event.ctrlKey || event.metaKey) return false;
      const key = event.key;
      if (key === '+' || key === '=') zoomStep(KEY_ZOOM);
      else if (key === '-' || key === '_' || key === '−') zoomStep(1 / KEY_ZOOM);
      else if (key === '0') fitAll();
      else if (key === '1') zoomTo(1);
      else if (key === '2') zoomTo(2);
      else if (key.startsWith('Arrow') && !event.shiftKey && !state.fitted && pannable()) {
        const step = PAN_KEY_SHARE / state.view.scale;
        const moves = { ArrowLeft: [-state.paneW * step, 0], ArrowRight: [state.paneW * step, 0], ArrowUp: [0, -state.paneH * step], ArrowDown: [0, state.paneH * step] };
        const move = moves[key];
        if (!move) return false;
        // Held down, the key repeats: no animation then, so it doesn't lag behind.
        setView({ scale: state.view.scale, cx: state.view.cx + move[0], cy: state.view.cy + move[1] }, { animate: !event.repeat });
      } else return false;
      return true;
    }

    let sayTimer = 0;
    function say(text) {
      scope.clearTimeout(sayTimer);
      sayTimer = scope.timeout(() => { live.textContent = text; }, 350);
    }

    /* ---- Start ---- */

    function showMessage(text) {
      message.hidden = !text;
      message.textContent = text || '';
    }

    if ('ResizeObserver' in window) scope.observe(new ResizeObserver(() => measure())).observe(stage);
    scope.listen(window, 'resize', () => measure());

    buildPanes();
    scope.fetch(options.manifest, { credentials: 'same-origin' })
      .then((response) => (response.ok ? response.json() : null))
      .then((manifest) => {
        if (state.destroyed) return;
        if (!manifest || manifest.unavailable || !Array.isArray(manifest.levels) || !manifest.levels.length) {
          if (options.onUnavailable) options.onUnavailable();
          else showMessage("This picture can't be shown here right now. Download it to see it at full size.");
          return;
        }
        state.manifest = manifest;
        state.W = manifest.width;
        state.H = manifest.height;
        state.tileSize = manifest.tile_size || 512;
        state.levels = manifest.levels;
        if (state.compare && manifest.before === false) {
          state.compare = false;
          hint.textContent = `${hintText} The original of this result isn't kept any more, so only the upscaled picture is shown.`;
          buildPanes();
        }
        state.fitted = true;
        measure();
        if (options.onReady) options.onReady(manifest);
      })
      .catch((err) => {
        if (state.destroyed || life.isAbort(err)) return;
        if (options.onUnavailable) options.onUnavailable();
        else showMessage("This picture can't be shown here right now.");
      });

    /** Ends everything this viewer started. Safe to call more than once. */
    function destroy() {
      if (state.destroyed) return;
      state.destroyed = true;
      scope.dispose(); // frames, timers, requests (tiles too), the observer and the window listener
      pointers.clear();
      gesture = null;
      for (const side of Object.keys(tiles)) {
        for (const entry of tiles[side].values()) {
          if (entry.ctrl && entry.state === 'loading') entry.ctrl.abort();
          if (entry.image && entry.image.close) entry.image.close();
        }
        tiles[side].clear();
      }
    }

    return {
      root,
      destroy,
      handleKey,
      focus: () => stage.focus({ preventScroll: true }),
      isZoomed: () => Boolean(state.manifest) && !state.fitted && pannable(),
      // For checks in the browser: what is on screen right now.
      debug: () => ({
        scale: state.view.scale, cx: state.view.cx, cy: state.view.cy, fitted: state.fitted, mode: state.mode,
        compare: state.compare, pos: state.pos, dpr: state.dpr, paneW: state.paneW, paneH: state.paneH,
        level: levelFor(state.view.scale, state.levels.length), fit: fit(),
        loaded: { before: tiles.before.size, after: tiles.after.size }, inFlight: { ...inFlight },
      }),
    };
  }

  window.ReDeepZoom = { create, ...helpers };
})();
