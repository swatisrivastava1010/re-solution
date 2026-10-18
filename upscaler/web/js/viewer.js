/* The full-screen viewer for finished pictures. Each picture opens in the deep-zoom compare viewer
   (js/deepzoom.js): the whole picture at Fit, every pixel at 100% and 200%, Before and After as a
   slider or side by side. A result whose zoomable tiles can't be had falls back to the close-ups
   (js/compare.js), or the whole picture when there's nothing to compare.

   Keys: + − 0 1 2 zoom, the arrows look around a zoomed-in picture, [ and ] (or Shift with the left
   and right arrows) go to the previous and next picture, as do the plain arrows at Fit, Home and End
   go to the first and last, and Esc closes. Tab stays inside the viewer.

   Close, Esc and a click outside close it straight away, every time, whatever it is doing (tiles
   loading, zoomed in, side by side, after an error): the picture is torn down first (one idempotent
   call that ends its timers, requests and listeners), then the dialog closes and focus goes back to
   whatever opened it.

   Plain browser script; app.js hands in its DOM helpers, the zoom and compare modules and the pure
   key helpers (viewerTarget, trapFocus). */
(function () {
  'use strict';

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  /**
   * deps: {dialog, h, icon, deepzoom (window.ReDeepZoom), compare (window.ReCompare),
   *        target (index, count, key) -> index | null, trap (count, index, backwards) -> index | -1}
   * Each item: {title, compare, zoom: {manifest, tile: (side, level, col, row) -> URL} | null,
   *             before: {src, alt, label, detail, text}, after: {…}, single: {text}, preview, previewAlt,
   *             details: () => Node[], download: {href, name}, open: {href, label}}
   */
  function create(deps) {
    const { dialog, h, icon, compare, deepzoom, target } = deps;
    const title = dialog.querySelector('#viewer-title');
    const position = dialog.querySelector('#viewer-position');
    const actions = dialog.querySelector('#viewer-actions');
    const stage = dialog.querySelector('#viewer-stage');
    const info = dialog.querySelector('#viewer-info');
    const prev = dialog.querySelector('#viewer-prev');
    const next = dialog.querySelector('#viewer-next');
    let items = [];
    let index = 0;
    let onClose = null;
    let current = null; // the deep-zoom viewer on show, if any
    let closeUps = null; // or the close-ups (js/compare.js)
    let opened = false; // the before-to-after reveal plays once, when the viewer opens
    let opener = null;

    /** Stops whatever picture is on show and empties the stage. Safe to call any number of times. */
    function clearStage() {
      for (const shown of [current, closeUps]) {
        if (!shown || typeof shown.destroy !== 'function') continue;
        try {
          shown.destroy();
        } catch (err) {
          window.console.warn('Re:Solution: closing the picture', err);
        }
      }
      current = null;
      closeUps = null;
      stage.replaceChildren();
    }

    function show(at) {
      index = Math.max(0, Math.min(items.length - 1, at));
      const item = items[index];
      if (!item) return;
      // A button that's about to be switched off would drop keyboard focus on the page behind.
      const active = document.activeElement;
      title.textContent = item.title;
      position.textContent = items.length > 1 ? `${index + 1} of ${items.length}` : '';
      prev.disabled = index === 0;
      next.disabled = index === items.length - 1;
      prev.hidden = next.hidden = items.length < 2;
      if ((active === prev && prev.disabled) || (active === next && next.disabled)) (prev.disabled ? next : prev).focus();

      const stageHadFocus = stage.contains(active);
      clearStage();
      stage.replaceChildren(item.zoom && deepzoom ? zoomView(item) : fallbackView(item));
      // Moving on with [ ] or the arrows replaced the picture that had keyboard focus: the new one gets it.
      if (dialog.open && (stageHadFocus || document.activeElement === document.body)) {
        if (current) current.focus();
        else title.focus({ preventScroll: true });
      }
      // Details are made when shown, so a batch of 100 doesn't build 100 sets up front.
      info.replaceChildren(...(typeof item.details === 'function' ? item.details() : item.details || []).filter(Boolean));
      const buttons = [];
      if (item.download) {
        buttons.push(h('a', { class: 'btn btn-primary btn-sm', href: item.download.href, download: item.download.name },
          icon('download'), 'Download'));
      }
      if (item.open) {
        buttons.push(h('a', { class: 'btn btn-secondary btn-sm', href: item.open.href, target: '_blank', rel: 'noopener' },
          icon('external'), item.open.label, h('span', { class: 'sr-only' }, ' (opens in a new tab)')));
      }
      actions.replaceChildren(...buttons);
    }

    function zoomView(item) {
      const shownAt = index;
      current = deepzoom.create({ h, icon }, {
        manifest: item.zoom.manifest,
        tile: item.zoom.tile,
        name: item.title,
        compare: item.compare,
        before: item.before,
        after: item.after,
        single: item.single,
        wipe: !opened,
        onUnavailable: () => {
          if (index !== shownAt || !dialog.open) return;
          clearStage();
          stage.replaceChildren(fallbackView(item)); // e.g. a guest's result whose picture was tidied away
        },
      });
      opened = true;
      return current.root;
    }

    function fallbackView(item) {
      if (item.compare && compare) {
        const shownAt = index;
        closeUps = compare.create({ h, icon }, {
          before: item.before,
          after: item.after,
          name: item.title,
          caption: 'A close-up at full size. Left: your picture simply enlarged. Right: upscaled. Drag the handle, or zoom in and drag to look around.',
          onError: () => {
            if (index !== shownAt || !dialog.open) return;
            clearStage();
            stage.replaceChildren(wholePicture(item));
          },
        });
        return closeUps.root;
      }
      return wholePicture(item);
    }

    function wholePicture(item) {
      const img = h('img', { src: item.preview, alt: item.previewAlt || '', decoding: 'async' });
      const box = h('div', { class: 'viewer-whole' }, img);
      img.addEventListener('error', () => box.replaceChildren(icon('image')));
      return box;
    }

    /** Opens on item `at` of `list`; `closed` runs when the viewer closes (to give focus back). */
    function open(list, at = 0, closed = null) {
      items = list || [];
      if (!items.length) return;
      const active = document.activeElement;
      if (!dialog.open) opener = active && active !== document.body && !dialog.contains(active) ? active : null;
      onClose = closed;
      opened = false;
      show(at);
      if (!dialog.open) dialog.showModal();
      title.focus({ preventScroll: true });
    }

    /** Closes at once, from any state. */
    function close() {
      clearStage();
      if (dialog.open) dialog.close();
    }

    /** New results arrived while it's open (a batch still running): keep the same picture in view. */
    function update(list, keyOf) {
      if (!dialog.open || !list || !list.length) return;
      const shown = items[index];
      items = list;
      const again = shown ? list.findIndex((item) => keyOf(item) === keyOf(shown)) : -1;
      if (again >= 0) {
        index = again;
        position.textContent = items.length > 1 ? `${index + 1} of ${items.length}` : '';
        prev.disabled = index === 0;
        next.disabled = index === items.length - 1;
        prev.hidden = next.hidden = items.length < 2;
      } else {
        show(Math.min(index, items.length - 1));
      }
    }

    function trapTab(event) {
      const controls = Array.from(dialog.querySelectorAll(FOCUSABLE)).filter((node) => node.getClientRects().length > 0
        && !(node.type === 'radio' && !node.checked
          && Array.from(dialog.querySelectorAll(`input[name="${CSS.escape(node.name)}"]`)).some((r) => r.checked)));
      const trap = deps.trap || (() => -1);
      const to = trap(controls.length, controls.indexOf(document.activeElement), event.shiftKey);
      if (to < 0) return;
      event.preventDefault();
      controls[to].focus();
    }

    prev.addEventListener('click', () => show(index - 1));
    next.addEventListener('click', () => show(index + 1));
    for (const button of dialog.querySelectorAll('[data-close]')) button.addEventListener('click', close);
    // A click on the backdrop lands on the dialog element itself, outside its box.
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const box = dialog.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close();
    });
    // Esc through the browser's own route: stop everything before the close event even arrives.
    dialog.addEventListener('cancel', clearStage);
    dialog.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
        return;
      }
      if (event.key === 'Tab') {
        trapTab(event);
        return;
      }
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const from = event.target;
      if (!from.closest) return;
      // Typing and the controls that use these keys themselves keep them.
      if (from.closest('textarea, select, input:not([type="radio"]):not([type="range"])')) return;
      const navigation = /^(Arrow|Home$|End$|Page)/.test(event.key);
      if (navigation && from.closest('input[type="radio"], input[type="range"], [role="slider"], .compare-range, .compare.is-zoomed')) return;
      // The zoom viewer first: + − 0 1 2, and the arrows while there's something to look around
      // (with Shift, the arrows always go to the previous or next picture).
      if (current && current.handleKey(event)) {
        event.preventDefault();
        return;
      }
      const to = target(index, items.length, event.key);
      if (to == null) return;
      event.preventDefault();
      if (to !== index) show(to); // no animation: it's a keyboard action people repeat quickly
    });
    // However it closed (Close, Esc, a click outside, another page), the 'close' event follows.
    dialog.addEventListener('close', () => {
      if (dialog.open) return; // it arrives a frame later: opened again meanwhile, so it isn't this visit's
      clearStage();
      const done = onClose;
      const back = opener;
      onClose = null;
      opener = null;
      if (done) done();
      else if (back && back.isConnected) back.focus({ preventScroll: true });
    });

    return {
      open,
      update,
      isOpen: () => dialog.open,
      close,
      debug: () => (current && current.debug ? current.debug() : null),
    };
  }

  window.ReViewer = { create };
})();
