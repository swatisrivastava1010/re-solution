/* Before/after comparison: a divider to drag, zoom to 100% or 200%, and panning.
   Used for the close-ups on finished pictures and for the A/B view in Preview styles.
   Its animation frames and resize observer belong to a lifecycle scope (js/lifecycle.js) that
   destroy() ends. Plain browser script; app.js passes in its small DOM helpers. */
(function () {
  'use strict';

  const ZOOMS = [
    { id: 'fit', label: 'Fit' },
    { id: '1', label: '100%' },
    { id: '2', label: '200%' },
  ];
  const PAN_STEP = 48; // px per arrow-key press while zoomed in
  const HANDLE_REACH = 22; // px either side of the divider that grab the divider, not the picture
  const WIPE_MS = 750;
  let counter = 0;

  function reducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /**
   * options: {
   *   before: {src, alt, label, detail}, after: {src, alt, label, detail},
   *   name: what is being compared (for the slider's label),
   *   caption: text under the picture, wipe: play the before-to-after reveal once,
   *   onError: called once if a picture can't be loaded
   * }
   */
  function create(deps, options) {
    const { h, icon } = deps;
    const life = deps.life || window.ReLife;
    const scope = life ? life.createScope('close-ups') : null;
    const frame = (fn) => (scope ? scope.frame(fn) : window.requestAnimationFrame(fn));
    const cancelFrame = (handle) => (scope ? scope.cancelFrame(handle) : window.cancelAnimationFrame(handle));
    const id = `compare-${++counter}`;
    const state = { pos: 50, zoom: 'fit', tx: 0, ty: 0, nw: 0, nh: 0, failed: false, wipe: null };

    const beforeImg = h('img', { alt: '', draggable: 'false', decoding: 'async' });
    const afterImg = h('img', { alt: '', draggable: 'false', decoding: 'async' });
    const beforeTag = h('span', { class: 'compare-tag compare-tag-before', 'aria-hidden': 'true' });
    const afterTag = h('span', { class: 'compare-tag compare-tag-after', 'aria-hidden': 'true' });
    const range = h('input', {
      class: 'compare-range sr-only', type: 'range', min: '0', max: '100', step: '1', value: '50',
      'aria-label': `Compare before and after for ${options.name || 'this picture'}`,
    });
    const handle = h('div', { class: 'compare-handle', 'aria-hidden': 'true' },
      h('span', { class: 'compare-knob' }, icon('arrows-h')));
    const box = h('div', { class: 'compare' },
      h('div', { class: 'compare-layer compare-layer-before' }, beforeImg),
      h('div', { class: 'compare-layer compare-layer-after' }, afterImg),
      handle, beforeTag, afterTag, range);

    const panHint = h('p', { class: 'compare-pan-hint', hidden: true }, 'Drag the picture to look around.');
    const zoomGroup = h('div', { class: 'segmented', role: 'radiogroup', 'aria-labelledby': `${id}-zoom` },
      ZOOMS.map((zoom) => h('label', null,
        h('input', { type: 'radio', name: `${id}-zoom-level`, value: zoom.id, checked: zoom.id === 'fit' }),
        h('span', null, zoom.label))));
    const toolbar = h('div', { class: 'compare-toolbar' },
      h('span', { class: 'compare-zoom-label', id: `${id}-zoom` }, icon('zoom-in'), 'Zoom'),
      zoomGroup, panHint);
    const figure = h('figure', { class: 'compare-figure' }, toolbar, box,
      options.caption ? h('figcaption', { class: 'compare-caption' }, options.caption) : null);

    /* ---- Pictures and labels ---- */

    function tagContent(side, fallback) {
      const label = (side && side.label) || fallback;
      return side && side.detail ? [label, ' ', h('small', null, side.detail)] : [label];
    }

    function setImages(before, after) {
      state.nw = 0;
      state.nh = 0;
      beforeImg.src = before.src;
      afterImg.src = after.src;
      beforeImg.alt = before.alt || '';
      afterImg.alt = after.alt || '';
      beforeTag.replaceChildren(...tagContent(before, 'Before'));
      afterTag.replaceChildren(...tagContent(after, 'After'));
    }

    function onLoad(img) {
      if (!img.naturalWidth) return;
      // Both pictures are the same spot at the same size; take the larger in case one is still loading.
      state.nw = Math.max(state.nw, img.naturalWidth);
      state.nh = Math.max(state.nh, img.naturalHeight);
      applyZoom();
    }
    for (const img of [beforeImg, afterImg]) {
      img.addEventListener('load', () => onLoad(img));
      img.addEventListener('error', () => {
        if (state.failed) return;
        state.failed = true;
        if (options.onError) options.onError();
      });
    }

    /* ---- Divider ---- */

    function setPosition(pct) {
      const value = Math.min(100, Math.max(0, pct));
      state.pos = value;
      box.style.setProperty('--pos', `${value}%`);
      range.value = String(Math.round(value));
      range.setAttribute('aria-valuetext', `${Math.round(value)}% before, ${100 - Math.round(value)}% after`);
    }

    function positionFromPointer(event) {
      const rect = box.getBoundingClientRect();
      setPosition(((event.clientX - rect.left) / rect.width) * 100);
    }

    function nearDivider(event) {
      const rect = box.getBoundingClientRect();
      const x = rect.left + (state.pos / 100) * rect.width;
      return Math.abs(event.clientX - x) <= HANDLE_REACH;
    }

    /* ---- Zoom and pan ---- */

    function zoomFactor() {
      return state.zoom === 'fit' ? 0 : Number(state.zoom);
    }

    /** Where the picture's pixel (ix, iy) is drawn now, as a scale and an offset in the box. */
    function currentView() {
      const rect = box.getBoundingClientRect();
      const z = zoomFactor();
      if (z) return { scale: z, ox: state.tx, oy: state.ty, rect };
      // Fit: like object-fit: cover.
      const scale = state.nw ? Math.max(rect.width / state.nw, rect.height / state.nh) : 1;
      return { scale, ox: (rect.width - state.nw * scale) / 2, oy: (rect.height - state.nh * scale) / 2, rect };
    }

    function clampOffsets(rect) {
      const z = zoomFactor();
      const w = state.nw * z;
      const hgt = state.nh * z;
      state.tx = w <= rect.width ? (rect.width - w) / 2 : Math.min(0, Math.max(rect.width - w, state.tx));
      state.ty = hgt <= rect.height ? (rect.height - hgt) / 2 : Math.min(0, Math.max(rect.height - hgt, state.ty));
    }

    function applyZoom() {
      const z = zoomFactor();
      const zoomed = Boolean(z && state.nw);
      box.classList.toggle('is-zoomed', zoomed);
      panHint.hidden = !zoomed;
      if (!zoomed) {
        box.removeAttribute('tabindex');
        box.removeAttribute('aria-label');
        box.removeAttribute('role');
        return;
      }
      const rect = box.getBoundingClientRect();
      clampOffsets(rect);
      box.style.setProperty('--img-w', `${state.nw * z}px`);
      box.style.setProperty('--img-h', `${state.nh * z}px`);
      box.style.setProperty('--tx', `${state.tx}px`);
      box.style.setProperty('--ty', `${state.ty}px`);
      box.setAttribute('tabindex', '0');
      box.setAttribute('role', 'group');
      box.setAttribute('aria-label', `Zoomed to ${z * 100}%. Use the arrow keys to look around.`);
    }

    /** Zoom so the picture point under (cx, cy) in the box stays where it is (default: the centre). */
    function setZoom(zoom, focus) {
      const before = currentView();
      const cx = focus ? focus.x : before.rect.width / 2;
      const cy = focus ? focus.y : before.rect.height / 2;
      const ix = (cx - before.ox) / before.scale;
      const iy = (cy - before.oy) / before.scale;
      state.zoom = zoom;
      const z = zoomFactor();
      if (z) {
        state.tx = cx - ix * z;
        state.ty = cy - iy * z;
      }
      for (const input of zoomGroup.querySelectorAll('input')) input.checked = input.value === zoom;
      applyZoom();
    }

    zoomGroup.addEventListener('change', (event) => {
      if (event.target.name === `${id}-zoom-level`) setZoom(event.target.value);
    });

    let drag = null; // 'divider' | 'pan'
    let last = null;
    box.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      stopWipe();
      const zoomed = box.classList.contains('is-zoomed');
      drag = zoomed && !nearDivider(event) ? 'pan' : 'divider';
      last = { x: event.clientX, y: event.clientY };
      box.classList.add(drag === 'pan' ? 'is-panning' : 'is-dragging');
      box.setPointerCapture(event.pointerId);
      if (drag === 'divider') positionFromPointer(event);
    });
    box.addEventListener('pointermove', (event) => {
      if (!drag) return;
      if (drag === 'divider') {
        positionFromPointer(event);
        return;
      }
      state.tx += event.clientX - last.x;
      state.ty += event.clientY - last.y;
      last = { x: event.clientX, y: event.clientY };
      applyZoom();
    });
    const endDrag = () => {
      drag = null;
      box.classList.remove('is-dragging', 'is-panning');
    };
    box.addEventListener('pointerup', endDrag);
    box.addEventListener('pointercancel', endDrag);
    box.addEventListener('dblclick', (event) => {
      const rect = box.getBoundingClientRect();
      setZoom(state.zoom === 'fit' ? '1' : 'fit', { x: event.clientX - rect.left, y: event.clientY - rect.top });
    });
    box.addEventListener('keydown', (event) => {
      if (event.target !== box || !box.classList.contains('is-zoomed')) return;
      const moves = { ArrowLeft: [PAN_STEP, 0], ArrowRight: [-PAN_STEP, 0], ArrowUp: [0, PAN_STEP], ArrowDown: [0, -PAN_STEP] };
      const move = moves[event.key];
      if (!move) return;
      event.preventDefault();
      state.tx += move[0];
      state.ty += move[1];
      applyZoom();
    });
    range.addEventListener('input', () => {
      stopWipe();
      setPosition(Number(range.value));
    });

    if ('ResizeObserver' in window) {
      const observer = new ResizeObserver(() => { if (zoomFactor()) applyZoom(); });
      observer.observe(box);
      if (scope) scope.observe(observer);
    }

    /* ---- The before-to-after reveal ---- */

    function stopWipe() {
      if (state.wipe) cancelFrame(state.wipe);
      state.wipe = null;
    }

    function wipe() {
      if (reducedMotion()) {
        setPosition(50);
        return;
      }
      setPosition(100);
      const startAt = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - startAt) / WIPE_MS);
        const eased = 1 - (1 - t) ** 4; // ease-out quart
        setPosition(100 - 50 * eased);
        state.wipe = t < 1 ? frame(step) : null;
      };
      // Wait for the after picture so the reveal shows something.
      const begin = () => { state.wipe = frame(step); };
      if (afterImg.complete && afterImg.naturalWidth) begin();
      else afterImg.addEventListener('load', begin, { once: true });
    }

    setImages(options.before, options.after);
    setPosition(50);
    if (options.wipe) wipe();

    /** Ends the reveal and the resize observer. Safe to call more than once. */
    function destroy() {
      stopWipe();
      if (scope) scope.dispose();
    }

    return { root: figure, setImages, setZoom, wipe, destroy };
  }

  window.ReCompare = { create };
})();
