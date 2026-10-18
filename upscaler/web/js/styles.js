/* "Preview styles": the same small spot of one picture made in each style, shown at full size,
   so people can pick the look they like before the long run.
   1 Pick a spot (or let Re:Solution choose the most detailed one), 2 wait about a minute,
   3 compare the crops side by side or with an A/B slider, then "Use this" for the picture
   or for every picture in the batch. Plain browser script; app.js passes in its helpers. */
(function () {
  'use strict';

  const SPOT = 256; // source pixels; each style makes it 4x bigger
  const KEY_STEP = 48; // source pixels per arrow-key press

  function create(deps) {
    const { dialog, api, h, icon, helpers } = deps;
    const body = dialog.querySelector('#styles-body');
    const foot = dialog.querySelector('#styles-foot');
    const sub = dialog.querySelector('#styles-sub');
    const state = {
      item: null, // {id, name, token, sampleId, width, height, pickUrl, current, extraModes}
      region: null, // [x, y, w, h] in source pixels, or null for "most detailed spot"
      phase: 'pick',
      jobId: null,
      results: [],
      chosen: null, // route in use for this picture
      routes: [], // the routes asked for, in order, in case results don't name theirs
      compare: null,
      starting: false,
    };
    const jobsByItem = new Map(); // picture id -> {id, routes} of its latest preview job, so reopening shows it again
    const openable = new Map(); // crop url -> a blob: url of the same picture, for "open in a new tab"

    /** The A/B slider stops its reveal and resize observer when it's replaced or the window closes. */
    function dropCompare() {
      if (state.compare && typeof state.compare.destroy === 'function') state.compare.destroy();
      state.compare = null;
    }

    dialog.addEventListener('close', () => {
      if (dialog.open) return; // it arrives a frame later: opened again meanwhile, so it isn't this visit's
      dropCompare();
      for (const url of openable.values()) URL.revokeObjectURL(url);
      openable.clear();
    });

    /* ---- Opening ---- */

    function open(item) {
      const sameItem = state.item && state.item.id === item.id;
      state.item = item;
      if (!sameItem) state.region = null;
      state.chosen = routeFor(item.current);
      sub.textContent = item.name;
      const last = jobsByItem.get(item.id);
      const job = last ? deps.getJob(last.id) : null;
      state.jobId = job ? job.id : null;
      state.routes = job ? last.routes : [];
      if (job && (job.status === 'queued' || job.status === 'running')) showRunning(job);
      else if (job && job.status === 'done' && (job.results || []).length) showResults(job);
      else showPick();
      if (!dialog.open) {
        dialog.showModal();
        // showModal moves focus to the first button (Close); start on the heading instead, as each step does.
        focusHeading();
      }
    }

    function routeFor(choice) {
      if (!choice || !choice.mode) return null;
      return choice.mode === 'photoreal' ? `photoreal:${choice.level || 'balanced'}` : choice.mode;
    }

    function heading(text) {
      return h('h3', { tabindex: '-1' }, text);
    }

    function focusHeading() {
      const title = body.querySelector('h3');
      if (title) title.focus({ preventScroll: true });
      dialog.scrollTop = 0;
    }

    /* ---- 1 Pick a spot ---- */

    function showPick(errorText) {
      state.phase = 'pick';
      const item = state.item;
      const img = h('img', { src: item.pickUrl, alt: `${item.name}. Click to choose the spot to compare.`, draggable: 'false' });
      const box = h('span', { class: 'region-box', hidden: true });
      const picker = h('div', {
        class: 'region', tabindex: '0', role: 'group',
        'aria-label': 'Spot to compare. Click the picture, or use the arrow keys to move the box.',
      }, img, box);
      const help = h('p', { class: 'region-help' });

      const drawBox = () => {
        const region = state.region;
        box.hidden = !region || !img.clientWidth;
        if (box.hidden) return;
        const sx = img.clientWidth / item.width;
        const sy = img.clientHeight / item.height;
        box.style.left = `${region[0] * sx}px`;
        box.style.top = `${region[1] * sy}px`;
        box.style.width = `${Math.max(12, region[2] * sx)}px`;
        box.style.height = `${Math.max(12, region[3] * sy)}px`;
      };
      const renderHelp = () => {
        const parts = [icon('crosshair')];
        if (state.region) {
          parts.push(h('span', null, 'The box shows the spot that will be compared.'));
          parts.push(h('button', {
            type: 'button', class: 'btn btn-quiet btn-sm',
            onclick: () => { state.region = null; drawBox(); renderHelp(); },
          }, 'Let Re:Solution choose'));
        } else {
          parts.push(h('span', null, 'Re:Solution picks the most detailed spot. Click the picture to choose a different one.'));
        }
        help.replaceChildren(...parts);
      };
      const choose = (x, y) => {
        state.region = helpers.regionAt(x, y, img.clientWidth, img.clientHeight, item.width, item.height, SPOT);
        drawBox();
        renderHelp();
      };

      img.addEventListener('load', drawBox);
      picker.addEventListener('click', (event) => {
        const rect = img.getBoundingClientRect();
        if (!rect.width) return;
        choose(event.clientX - rect.left, event.clientY - rect.top);
      });
      picker.addEventListener('keydown', (event) => {
        const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (event.key === 'Enter') {
          event.preventDefault();
          startPreview();
          return;
        }
        const move = moves[event.key];
        if (!move || !img.clientWidth) return;
        event.preventDefault();
        const current = state.region || helpers.regionAt(img.clientWidth / 2, img.clientHeight / 2,
          img.clientWidth, img.clientHeight, item.width, item.height, SPOT);
        const cx = current[0] + current[2] / 2 + move[0] * KEY_STEP;
        const cy = current[1] + current[3] / 2 + move[1] * KEY_STEP;
        choose((cx / item.width) * img.clientWidth, (cy / item.height) * img.clientHeight);
      });
      if ('ResizeObserver' in window) new ResizeObserver(drawBox).observe(img);
      renderHelp();

      const parts = [
        heading('Which look do you like best?'),
        h('p', { class: 'wiz-lede' }, 'Re:Solution makes one small spot of your picture in each style and shows them at full size, side by side. It takes about a minute.'),
      ];
      if (!deps.photorealUsable()) {
        parts.push(h('div', { class: 'alert alert-warn' }, icon('info'),
          h('p', null, "Photo-real isn't ready on this computer, so the preview leaves it out.")));
      }
      if (errorText) parts.push(h('div', { class: 'alert alert-danger', role: 'alert' }, icon('alert'), h('p', null, errorText)));
      parts.push(picker, help);
      body.replaceChildren(...parts);

      const start = h('button', { type: 'button', class: 'btn btn-primary', onclick: startPreview },
        h('span', { class: 'spinner', 'aria-hidden': 'true' }), icon('layers'), h('span', { class: 'btn-label' }, 'Make previews'));
      foot.replaceChildren(
        h('button', { type: 'button', class: 'btn btn-secondary', onclick: () => dialog.close() }, 'Cancel'),
        start);
      focusHeading();
    }

    async function startPreview() {
      if (state.starting) return;
      const item = state.item;
      const startButton = foot.querySelector('.btn-primary');
      state.starting = true;
      if (startButton) {
        startButton.disabled = true;
        startButton.classList.add('is-busy');
      }
      const routes = deps.routes(item);
      const request = { ...deps.previewOptions(), routes };
      if (item.token) request.token = item.token;
      else request.sample_id = item.sampleId;
      if (state.region) request.region = state.region;
      try {
        const data = await api(deps.endpoint, { method: 'POST', body: request });
        const job = data && (data.job || data);
        if (!job || !job.id) throw new Error("The preview didn't start. Try again in a moment.");
        jobsByItem.set(item.id, { id: job.id, routes });
        state.jobId = job.id;
        state.routes = routes;
        deps.onJobCreated(job);
        // Already finished (it can be that quick, and its last update may have come before this answer):
        // show the styles now instead of waiting for an update that won't come.
        if (job.status === 'done') showResults(job);
        else showRunning(job);
      } catch (err) {
        if (helpers.missingEndpoint(err.status, err.detail)) {
          deps.onUnavailable();
          showPick("Preview styles isn't available in this copy of Re:Solution yet. You can still pick a quality on the page.");
          foot.querySelector('.btn-primary').remove();
          return;
        }
        showPick(err.message);
      } finally {
        state.starting = false;
      }
    }

    /* ---- 2 Making the previews ---- */

    function showRunning(job) {
      state.phase = 'running';
      const bar = h('div', {
        class: 'progress progress-lg is-live', role: 'progressbar', 'aria-label': 'Making the previews',
        'aria-valuemin': '0', 'aria-valuemax': '100',
      }, h('span'));
      body.replaceChildren(
        heading('Making the previews…'),
        h('p', { class: 'wiz-lede' }, 'Each style works on the same spot, so you can compare them fairly.'),
        bar,
        h('div', { class: 'progress-row' }, h('p', { 'data-live': 'message' }), h('p', { class: 'muted tabular', 'data-live': 'eta' })));
      const stop = h('button', { type: 'button', class: 'btn btn-quiet btn-danger-text' }, 'Stop the preview');
      stop.addEventListener('click', async () => {
        stop.disabled = true;
        try {
          await api(deps.cancelPath(job.id), { method: 'POST' });
        } catch (err) {
          stop.disabled = false;
          body.append(h('p', { class: 'field-error', role: 'alert' }, err.message));
        }
      });
      foot.replaceChildren(
        h('p', { class: 'styles-foot-note' }, 'You can close this; it keeps going.'),
        stop,
        h('button', { type: 'button', class: 'btn btn-secondary', onclick: () => dialog.close() }, 'Hide'));
      updateRunning(job);
      focusHeading();
    }

    function updateRunning(job) {
      const bar = body.querySelector('.progress');
      if (bar) helpers.setProgress(bar, job.status === 'running' ? job.progress : 0);
      const message = body.querySelector('[data-live="message"]');
      if (message) {
        message.textContent = job.status === 'queued'
          ? 'Waiting for the job that is running now. Previews go next.'
          : (job.message || 'Working…');
      }
      const eta = body.querySelector('[data-live="eta"]');
      if (eta) eta.textContent = job.status === 'running' ? helpers.formatEta(job.eta_seconds) : '';
    }

    /** Called by app.js for every update to a preview job. */
    function onJob(job) {
      if (!state.item || job.id !== state.jobId) return;
      if (!dialog.open) return;
      if (job.status === 'queued' || job.status === 'running') {
        if (state.phase !== 'running') showRunning(job);
        else updateRunning(job);
      } else if (job.status === 'done') {
        if (state.phase !== 'results') showResults(job);
      } else if (job.status === 'failed') {
        showPick(job.error || "The preview didn't finish. Try again, or pick a different spot.");
      } else if (job.status === 'cancelled') {
        showPick();
      }
    }

    /* ---- 3 Compare and choose ---- */

    /** A style that was made: it has a picture to show and can be compared. */
    function usable(result) {
      return Boolean(result && result.url && !result.error);
    }

    function showResults(job) {
      state.phase = 'results';
      state.results = deps.results(job, state.routes);
      const list = h('ul', { class: 'style-grid', 'aria-label': 'Styles at full size' },
        state.results.map((result) => styleCard(result)));
      followPointer(list);
      const look = window.matchMedia('(hover: none)').matches
        ? 'Tap one to open it on its own.'
        : 'Move the pointer over one to look around; they all move together.';
      const parts = [
        heading('Pick the look you like'),
        h('p', { class: 'wiz-lede' }, `The same spot in each style, shown at full size. ${look} Times are for your whole picture.`),
        list,
      ];
      const ab = compareSection();
      if (ab) parts.push(ab);
      body.replaceChildren(...parts);
      renderFoot();
      focusHeading();
    }

    /** The cards show their pictures at full size, so each shows only part of the spot. Pointing at one
        moves every card to the same part, like one magnifying glass over all the styles. */
    function followPointer(list) {
      list.addEventListener('pointermove', (event) => {
        if (event.pointerType === 'touch') return;
        const shot = event.target.closest('.style-shot');
        if (!shot) return;
        const rect = shot.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const along = (offset, size) => `${(Math.min(1, Math.max(0, offset / size)) * 100).toFixed(1)}%`;
        list.style.setProperty('--shot-x', along(event.clientX - rect.left, rect.width));
        list.style.setProperty('--shot-y', along(event.clientY - rect.top, rect.height));
      });
    }

    /** The A/B slider between any two styles that were made, or null when fewer than two were. */
    function compareSection() {
      const good = state.results.filter(usable);
      dropCompare();
      if (good.length < 2) return null;
      const chosen = good.find((r) => r.route === state.chosen && r.route !== 'original');
      const left = good.find((r) => r.route === 'original') || good.find((r) => r !== chosen);
      const right = (chosen !== left && chosen)
        || good.find((r) => r !== left && r.route !== 'original') || good.find((r) => r !== left);
      const pickA = picker('ab-left', 'Left side', left.route, good);
      const pickB = picker('ab-right', 'Right side', right.route, good);
      const sideOf = (route) => {
        const result = good.find((r) => r.route === route) || good[0];
        return { src: result.url, alt: `${result.label}, at full size`, label: result.label };
      };
      state.compare = window.ReCompare.create({ h, icon }, {
        before: sideOf(left.route),
        after: sideOf(right.route),
        name: 'the two styles',
        caption: 'Drag the handle to compare two styles. Zoom to 100% to see every pixel.',
      });
      const refresh = () => state.compare.setImages(sideOf(pickA.select.value), sideOf(pickB.select.value));
      pickA.select.addEventListener('change', refresh);
      pickB.select.addEventListener('change', refresh);
      return h('section', { class: 'ab', 'aria-label': 'Compare two styles' },
        h('h3', null, 'Compare two styles'),
        h('div', { class: 'ab-pickers' }, pickA.field, pickB.field),
        state.compare.root);
    }

    function picker(idName, label, value, results) {
      const select = h('select', { class: 'input select', id: idName },
        results.map((r) => h('option', { value: r.route, selected: r.route === value }, r.label)));
      select.value = value;
      return { select, field: h('div', { class: 'field' }, h('label', { class: 'field-label', for: idName }, label), select) };
    }

    function styleCard(result) {
      const choice = helpers.routeChoice(result.route);
      const theme = choice ? helpers.modeTheme(choice.mode) : 'neutral';
      const tag = h('span', { class: 'tag' }, icon(choice ? helpers.modeIcon(choice.mode) : 'image'), result.label);
      if (!usable(result)) {
        // The server says why in plain words (first use while offline, say); nothing here to pick.
        return h('li', { class: 'style-card is-failed', 'data-theme': theme, 'data-route': result.route },
          tag,
          h('div', { class: 'alert alert-warn style-failed' }, icon('alert'),
            h('p', null, result.error || "This style couldn't be made here.")));
      }
      const chosen = result.route === state.chosen;
      const card = h('li', { class: `style-card${chosen ? ' is-chosen' : ''}`, 'data-theme': theme, 'data-route': result.route },
        tag,
        openInTab(h('a', { class: 'style-shot', href: result.url, target: '_blank', rel: 'noopener', 'aria-label': `Open ${result.label} at full size in a new tab` },
          h('img', { src: result.url, alt: '', loading: 'lazy', decoding: 'async' })), result.url),
        h('p', { class: 'style-meta' }, metaText(result)));
      if (choice) {
        if (chosen) {
          card.append(h('p', { class: 'style-chosen' }, icon('check'), 'Using this'));
        } else {
          card.append(h('button', {
            type: 'button', class: 'btn btn-secondary btn-sm', 'aria-label': `Use ${result.label} for this picture`,
            onclick: () => choose(result),
          }, 'Use this'));
        }
      }
      return card;
    }

    /** The server sends crops as downloads, so a link straight to one would save it rather than show it.
        The link gets the browser's own copy of the picture instead, which opens in a new tab like any picture. */
    function openInTab(link, url) {
      const ready = openable.get(url);
      if (ready) {
        link.href = ready;
        return link;
      }
      fetch(url, { credentials: 'same-origin' })
        .then((response) => (response.ok ? response.blob() : null))
        .then((blob) => {
          if (!blob || !dialog.open) return;
          if (!openable.has(url)) openable.set(url, URL.createObjectURL(blob));
          link.href = openable.get(url);
        })
        .catch(() => { /* the link still saves the crop */ });
      return link;
    }

    function metaText(result) {
      if (result.route === 'original') return 'Your picture, simply enlarged.';
      const full = helpers.formatApproxTime(result.estFull);
      if (full) return `Whole picture: ${full.charAt(0).toLowerCase()}${full.slice(1)}.`;
      return Number.isFinite(result.seconds) ? `This spot took ${Math.round(result.seconds)} s.` : '';
    }

    function choose(result) {
      const choice = helpers.routeChoice(result.route);
      if (!choice) return;
      state.chosen = result.route;
      deps.onUse(state.item.id, choice, false);
      for (const card of body.querySelectorAll('.style-card')) {
        const fresh = styleCard(state.results.find((r) => r.route === card.dataset.route));
        card.replaceWith(fresh);
      }
      const again = body.querySelector('.style-card.is-chosen');
      if (again) again.querySelector('.style-chosen').setAttribute('tabindex', '-1');
      if (again) again.querySelector('.style-chosen').focus();
      renderFoot();
    }

    function renderFoot() {
      const count = deps.batchSize();
      const result = state.results.find((r) => r.route === state.chosen && usable(r));
      const parts = [];
      if (count > 1 && result) {
        parts.push(h('button', {
          type: 'button', class: 'btn btn-secondary',
          onclick: () => {
            deps.onUse(state.item.id, helpers.routeChoice(result.route), true);
            dialog.close();
          },
        }, `Use for all ${count} pictures in this batch`));
      } else {
        const anyMade = state.results.some((r) => r.route !== 'original' && usable(r));
        let note = 'Pick a style with "Use this".';
        if (result) note = `${result.label} will be used for this picture.`;
        else if (!anyMade) note = "None of the styles could be made this time. Each one says why.";
        parts.push(h('p', { class: 'styles-foot-note' }, note));
      }
      parts.push(h('button', { type: 'button', class: 'btn btn-primary', onclick: () => dialog.close() }, 'Done'));
      foot.replaceChildren(...parts);
    }

    return { open, onJob };
  }

  window.ReStyles = { create };
})();
