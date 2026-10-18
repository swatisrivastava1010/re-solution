/* Side-by-side video comparison: the original and the upscaled video, each in its own frame, played
   together. No slider: the two sit next to each other (one above the other on narrow screens), and
   one control bar drives both: play and pause, a shared timeline, a frame back or forward, speed,
   loop and sound (from the upscaled side only), and Fit / 100% / 200% with one shared drag to look
   around.

   Keeping the two in step, without freezing the page:
   - The upscaled player is the one clock. The original follows it: at most every 250 ms, and never
     while either is seeking, a drift of more than half a frame is corrected (gently by speed when it's
     small, by one jump when it's over half a second) until it's back within a quarter of a frame, so
     the two show the same frame.
   - Loop restarts both together (the browsers' own looping would restart each on its own).
   - Seeks are coalesced: while a video is still seeking, only the newest wanted time is kept, so
     dragging along the timeline never queues up dozens of seeks.
   - When either video runs out of data, both wait with a clear "Loading…", and a friendly
     "Try again" appears if it takes much longer than usual.
   - The page is redrawn once per new video frame (requestVideoFrameCallback where the browser has
     it), and only the parts that changed.
   - Everything the player starts belongs to one lifecycle scope (js/lifecycle.js): closing stops the
     videos and lets go of their files, and ends every timer, frame callback, request and listener.

   The server first makes copies of both that any browser can play (POST …/compare); a calm
   "Preparing the comparison… 42%" shows meanwhile. Plain browser script; app.js opens it in its own
   full-screen dialog. The timing helpers at the top are also exported for tests run under Node. */
(function () {
  'use strict';

  const SPEEDS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];
  const SYNC_MS = 250; // the drift check runs at most this often
  const POLL_MS = 700;
  const JUMP_SECONDS = 1;
  const NUDGE = 3; // how hard the follower's speed leans against a small drift (3 at 250 ms: most of it gone by the next check)
  const MAX_NUDGE = 0.25;
  const HARD_SEEK_SECONDS = 0.5; // further apart than this, the follower jumps instead of catching up
  const LOADING_SLOW_MS = 12000; // "This is taking longer than usual" after this long
  const SEEK_WATCHDOG_MS = 1500; // a seek that never says it's done is retried after this long
  const ZOOMS = [{ id: 'fit', label: 'Fit' }, { id: '1', label: '100%' }, { id: '2', label: '200%' }];
  let counter = 0;

  /* ---------- Timing (pure, also run under Node) ---------- */

  /** "30000/1001" or 25 -> frames per second. */
  function parseRate(rate) {
    if (typeof rate === 'number') return rate > 0 ? rate : 0;
    const [num, den] = String(rate || '').split('/').map(Number);
    if (!(num > 0)) return 0;
    return den > 0 ? num / den : num;
  }

  /** The frame shown at `time` (frame n covers [n / fps, (n + 1) / fps)). */
  function frameAt(time, fps) {
    return Math.max(0, Math.floor(time * fps + 1e-6));
  }

  /** A time inside frame n: its middle, so a seek lands on it whatever the rounding. */
  function timeOfFrame(frame, fps) {
    return (Math.max(0, frame) + 0.5) / fps;
  }

  /** "0:03.2", "1:05.0", "1:02:03.5". */
  function formatClock(seconds) {
    const t = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    const tenths = Math.floor(t * 10 + 1e-6) % 10;
    const whole = Math.floor(t + 1e-6);
    const s = whole % 60;
    const m = Math.floor(whole / 60) % 60;
    const hrs = Math.floor(whole / 3600);
    const ss = String(s).padStart(2, '0');
    return hrs ? `${hrs}:${String(m).padStart(2, '0')}:${ss}.${tenths}` : `${m}:${ss}.${tenths}`;
  }

  /** What to do about the follower being `drift` seconds ahead (negative: behind) of the leader:
      nothing within `within` (one frame), catch up gently by speed within half a second, otherwise
      jump. Once correcting, the player passes a quarter of a frame as `within`, so it ends close. */
  function driftAction(drift, fps, rate, within = 1 / (fps || 30)) {
    if (!Number.isFinite(drift) || Math.abs(drift) <= within) return { seek: false, rate };
    if (Math.abs(drift) > HARD_SEEK_SECONDS) return { seek: true, rate };
    const lean = Math.max(-MAX_NUDGE, Math.min(MAX_NUDGE, drift * NUDGE));
    return { seek: false, rate: rate * (1 - lean) };
  }

  /** The drift once a loop has wrapped one player round before the other. */
  function loopDrift(drift, duration, looping) {
    if (!looping || !(duration > 0) || Math.abs(drift) <= duration / 2) return drift;
    return drift - Math.sign(drift) * duration;
  }

  /** Why play() was refused: 'interrupted' (our own pause or a new source got there first: not a
      problem), 'blocked' (the browser wants a tap first) or 'failed'. */
  function playFailure(err) {
    const name = err && err.name;
    if (name === 'AbortError') return 'interrupted';
    if (name === 'NotAllowedError') return 'blocked';
    return 'failed';
  }

  /** Two frames as columns or rows: whichever shows the video bigger. */
  function videoLayout(width, height, stageW, stageH, gap = 12) {
    const fit = (w, hgt) => Math.min(w / width, hgt / height);
    return fit(stageW, (stageH - gap) / 2) > fit((stageW - gap) / 2, stageH) ? 'rows' : 'columns';
  }

  /** The displayed size of the videos at a zoom: Fit inside the frame, or 1 or 2 video pixels of the
      upscaled video per screen pixel. Both videos are shown at this size, so they show the same spot. */
  function displaySize(zoom, video, frame, dpr) {
    if (zoom === 'fit') {
      const s = Math.min(frame.width / video.width, frame.height / video.height);
      return { width: video.width * s, height: video.height * s };
    }
    const z = Number(zoom) || 1;
    return { width: (video.width * z) / dpr, height: (video.height * z) / dpr };
  }

  const helpers = {
    parseRate, frameAt, timeOfFrame, formatClock, driftAction, loopDrift, playFailure, videoLayout, displaySize,
    SPEEDS, SYNC_MS, LOADING_SLOW_MS,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = helpers;
  if (typeof document === 'undefined') return;

  /* ---------- The comparison ---------- */

  const numberFormat = new Intl.NumberFormat(); // made once: toLocaleString() builds a new one every call
  const num = (n) => numberFormat.format(Number(n));
  const PREPARE_NOTE = 'Making copies of both videos that play in any browser. This happens once: next time, the comparison opens straight away.';
  const FAILED = "Couldn't get the comparison ready. Please try again.";
  const LOADING = 'Loading…';
  const SLOW = 'This is taking longer than usual.';
  const BROKEN = "This video couldn't be played here. Try again, or use Download.";

  /**
   * deps: {h, icon, api (path, {method, signal}) -> Promise, life (window.ReLife)}
   * options: {startUrl, statusUrl, cancelUrl, name}
   */
  function createComparison(deps, options) {
    const { h, icon, api } = deps;
    const life = deps.life || window.ReLife;
    const scope = life.createScope('video-compare');
    const id = `vc-${++counter}`;
    const root = h('div', { class: 'vc' });
    let pollTimer = 0;
    let player = null;
    let generation = 0; // answers to an older request are ignored

    function request(url, init = {}) {
      const ctrl = scope.controller();
      return Promise.resolve()
        .then(() => api(url, { ...init, signal: ctrl.signal }))
        .finally(() => scope.done(ctrl));
    }

    function clearPoll() {
      scope.clearTimeout(pollTimer);
      pollTimer = 0;
    }

    function dropPlayer() {
      if (!player) return;
      const old = player;
      player = null;
      old.destroy();
    }

    /** Focus may move into the comparison only if it was there already (or nowhere), never away from Close. */
    function mayTakeFocus() {
      const active = document.activeElement;
      return !active || active === document.body || root.contains(active) || (active.matches && active.matches('h2[tabindex="-1"]'));
    }

    /* ---- Getting ready ---- */

    const percent = h('span', { class: 'vc-prep-pct' }, '');
    const bar = h('div', {
      class: 'progress progress-lg is-live', role: 'progressbar', 'aria-label': 'Preparing the comparison',
      'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0',
    }, h('span'));
    const stepText = h('p', { class: 'vc-prep-step muted' }, '');
    const stop = h('button', { type: 'button', class: 'btn btn-secondary btn-sm' }, icon('x'), 'Stop');
    const prepare = h('div', { class: 'vc-prep', role: 'status' },
      h('span', { class: 'vc-prep-art', 'aria-hidden': 'true' }, icon('film')),
      h('p', { class: 'vc-prep-title' }, 'Preparing the comparison… ', percent),
      bar, stepText, h('p', { class: 'vc-prep-note muted' }, PREPARE_NOTE), stop);

    function showPreparing(state) {
      if (prepare.parentNode !== root) {
        dropPlayer();
        root.replaceChildren(prepare);
      }
      const pct = Math.floor(Math.max(0, Math.min(1, Number(state.progress) || 0)) * 100);
      percent.textContent = `${pct}%`;
      bar.style.setProperty('--value', `${pct}%`);
      bar.setAttribute('aria-valuenow', String(pct));
      bar.classList.toggle('is-indeterminate', state.status === 'waiting' || pct === 0);
      stepText.textContent = state.status === 'waiting' ? 'Waiting for another comparison to finish…' : (state.message && state.message !== 'Preparing the comparison' ? `${state.message}…`.replace(/……$/, '…') : '');
    }

    function showProblem(text, retry = true) {
      clearPoll();
      const focus = mayTakeFocus();
      dropPlayer();
      const again = h('button', { type: 'button', class: 'btn btn-primary btn-sm' }, icon('refresh'), 'Try again');
      again.addEventListener('click', begin);
      root.replaceChildren(h('div', { class: 'vc-prep vc-problem' },
        h('div', { class: 'alert alert-danger', role: 'alert' }, icon('alert'), h('p', null, text)),
        retry ? again : null));
      if (retry && focus) again.focus({ preventScroll: true });
    }

    /** Stopped on purpose (the Stop button): a calm note with Try again, not a red alert. */
    function showStopped(text) {
      clearPoll();
      const focus = mayTakeFocus();
      dropPlayer();
      const again = h('button', { type: 'button', class: 'btn btn-primary btn-sm' }, icon('refresh'), 'Try again');
      again.addEventListener('click', begin);
      root.replaceChildren(h('div', { class: 'vc-prep vc-stopped', role: 'status' },
        h('span', { class: 'vc-prep-art', 'aria-hidden': 'true' }, icon('film')),
        h('p', { class: 'vc-stopped-text' }, text), again));
      if (focus) again.focus({ preventScroll: true });
    }

    function onState(state, mine) {
      if (scope.disposed || !state || mine !== generation) return;
      if (state.status === 'ready') {
        clearPoll();
        showPlayer(state);
        return;
      }
      if (state.status === 'failed') {
        showProblem(state.error || FAILED);
        return;
      }
      if (state.status === 'cancelled') {
        showStopped(state.message || 'Stopped.');
        return;
      }
      showPreparing(state);
      clearPoll();
      pollTimer = scope.timeout(() => {
        pollTimer = 0;
        request(options.statusUrl).then((next) => onState(next, mine), (err) => onError(err, mine));
      }, POLL_MS);
    }

    function onError(err, mine) {
      if (scope.disposed || life.isAbort(err) || mine !== generation) return;
      showProblem((err && err.message) || FAILED);
    }

    function begin() {
      if (scope.disposed) return;
      generation += 1;
      const mine = generation;
      clearPoll();
      showPreparing({ status: 'preparing', progress: 0 });
      request(options.startUrl, { method: 'POST' }).then((state) => onState(state, mine), (err) => onError(err, mine));
    }

    stop.addEventListener('click', () => {
      if (stop.disabled) return;
      stop.disabled = true;
      const mine = generation;
      request(options.cancelUrl, { method: 'POST' })
        .then((state) => onState(state, mine), (err) => onError(err, mine))
        .finally(() => { stop.disabled = false; });
    });

    /* ---- The players ---- */

    function showPlayer(info) {
      const focus = mayTakeFocus();
      dropPlayer();
      player = createPlayer(info, focus);
    }

    function createPlayer(info, takeFocus) {
      const ps = life.createScope('video-player');
      const fps = parseRate(info.fps) || 30;
      const frameTime = 1 / fps;
      const frames = Math.max(1, Number(info.frames) || Math.round((Number(info.duration) || 1) * fps));
      const duration = Number(info.duration) || frames / fps;
      const sizes = { original: info.original || {}, upscaled: info.upscaled || {} };
      const video = (src, label) => {
        // "metadata" until the first Play, so opening the comparison doesn't start two big downloads.
        const node = h('video', {
          class: 'vc-video', playsinline: true, preload: 'metadata', 'aria-label': label,
          disablepictureinpicture: true, disableremoteplayback: true,
        });
        node.muted = true;
        node.src = src;
        return ps.video(node);
      };
      const original = video(info.original_url, 'Original video');
      const upscaled = video(info.upscaled_url, 'Upscaled video');
      const leader = upscaled;
      const follower = original;
      const frame = (node, text, theme) => {
        const view = h('div', { class: 'vc-view' }, node, h('span', { class: 'vc-buffering', 'aria-hidden': 'true' }));
        return { view, el: h('figure', { class: 'vc-frame', 'data-side': theme }, view, h('figcaption', { class: 'vc-tag' }, text)) };
      };
      const label = (word, size) => (size.width && size.height ? `${word} · ${num(size.width)} × ${num(size.height)}` : word);
      const left = frame(original, label('Original', sizes.original), 'original');
      const right = frame(upscaled, label('Upscaled', sizes.upscaled), 'upscaled');
      const noticeText = h('p', { class: 'vc-notice-text' }, '');
      const noticeRetry = h('button', { type: 'button', class: 'btn btn-secondary btn-sm', hidden: true }, icon('refresh'), 'Try again');
      const notice = h('div', { class: 'vc-notice', role: 'status', 'aria-live': 'polite', hidden: true },
        h('span', { class: 'vc-notice-spin', 'aria-hidden': 'true' }), noticeText, noticeRetry);
      const stage = h('div', { class: 'vc-stage' }, left.el, right.el, notice);

      // The play button keeps the same icon element and only changes what it shows, so a click that
      // lands on the icon is never lost to the icon being swapped mid-click.
      const playIcon = icon('play');
      const playButton = h('button', { type: 'button', class: 'vc-play', 'aria-label': 'Play (Space)', title: 'Play (Space)' }, playIcon);
      const back = h('button', { type: 'button', class: 'icon-button vc-btn', 'aria-label': 'Previous frame (,)', title: 'Previous frame (,)' }, icon('step-back'));
      const forward = h('button', { type: 'button', class: 'icon-button vc-btn', 'aria-label': 'Next frame (.)', title: 'Next frame (.)' }, icon('step-forward'));
      const clock = h('span', { class: 'vc-clock' }, '');
      const frameText = h('span', { class: 'vc-frame-no' }, '');
      const timeline = h('input', {
        class: 'vc-timeline', type: 'range', min: '0', max: String(frames - 1), step: '1', value: '0', 'aria-label': 'Timeline',
      });
      const speed = h('select', { class: 'vc-speed', 'aria-label': 'Playback speed', title: 'Playback speed' },
        SPEEDS.map((value) => h('option', { value: String(value), selected: value === 1 }, `${value}×`)));
      const loopButton = h('button', { type: 'button', class: 'icon-button vc-btn vc-loop', 'aria-pressed': 'false', 'aria-label': 'Loop', title: 'Loop' }, icon('repeat'));
      const hasSound = Boolean(info.has_audio);
      const muteIcon = icon(hasSound ? 'volume' : 'volume-off');
      const mute = h('button', {
        type: 'button', class: 'icon-button vc-btn vc-mute', 'aria-pressed': 'false', disabled: !hasSound,
        'aria-label': hasSound ? 'Mute (M)' : 'This video has no sound', title: hasSound ? 'Sound from the upscaled video (M)' : 'This video has no sound',
      }, muteIcon);
      const zoomGroup = h('div', { class: 'segmented vc-zoom', role: 'radiogroup', 'aria-label': 'Zoom' },
        ZOOMS.map((zoom) => h('label', null,
          h('input', { type: 'radio', name: `${id}-zoom`, value: zoom.id, checked: zoom.id === 'fit', 'aria-label': `Zoom ${zoom.label}` }),
          h('span', null, zoom.label))));
      const controls = h('div', { class: 'vc-bar', role: 'group', 'aria-label': 'Playback for both videos' },
        h('div', { class: 'vc-row vc-row-time' }, clock, timeline, frameText),
        h('div', { class: 'vc-row vc-row-buttons' },
          h('div', { class: 'vc-group' }, back, playButton, forward),
          h('div', { class: 'vc-group' }, speed, loopButton, mute),
          zoomGroup));
      // The keys as keys, so ", and ." doesn't read as stray punctuation.
      const key = (name, label) => h('kbd', label ? { title: label } : null, name);
      const hint = h('p', { class: 'vc-hint' }, key('Space'), ' plays and pauses, ', key(',', 'Comma'), ' and ', key('.', 'Full stop'),
        ' step a frame, ', key('J'), ' and ', key('L'), ' jump a second. At 100% or 200%, drag either video to look around.');
      root.replaceChildren(h('div', { class: 'vc-player' }, stage, controls, hint));

      let wantPlay = false; // what the person asked for; the videos follow it
      let loading = false; // both wait while either is still loading
      let held = false; // both stopped for a moment by the page itself (a seek while playing)
      let loadingTimer = 0;
      let broken = false;
      let looping = false;
      let scrub = null; // {pointerId} while the timeline is being dragged
      let rate = 1;
      let zoom = 'fit';
      const pan = { x: 0.5, y: 0.5 };
      let loop = 0; // the one running redraw loop
      let seekTarget = null; // the newest wanted time while a seek is still under way
      let seekWatch = 0;
      let seekStarted = 0;
      let seekCost = 0.08; // seconds the original takes to seek, learned as it plays
      let lastFrame = -1;
      let correcting = false; // the original is being brought back in step
      let startWait = 0; // waiting for the upscaled video's clock to start before starting the original
      let startBy = 0;
      // The upscaled video's own 'play' and 'pause' events still to come from the page's own calls. Events
      // arrive a moment later (after more presses, maybe), so anything beyond these came from outside.
      let ownPlays = 0;
      let ownPauses = 0;
      let shownPlaying = null;
      let upgraded = false;
      const durationText = formatClock(duration);
      const framesText = num(frames);

      /* Playing together */

      const seeking = () => leader.seeking || follower.seeking;
      const timeNow = () => (seekTarget != null ? seekTarget : leader.currentTime || 0);

      function start(node) {
        if (node === leader && leader.paused) ownPlays += 1;
        let promise;
        try {
          promise = node.play();
        } catch (err) {
          promise = Promise.reject(err);
        }
        if (!promise || typeof promise.catch !== 'function') return;
        promise.catch((err) => {
          if (ps.disposed || playFailure(err) === 'interrupted') return; // our own pause got there first
          if (node === leader && wantPlay) {
            // The browser wouldn't start (no tap yet on a phone, say): both stay paused, in step.
            wantPlay = false;
            stopBoth();
            reflect();
          }
        });
      }

      function stopBoth() {
        if (!leader.paused) ownPauses += 1;
        leader.pause();
        follower.pause();
      }

      function align() {
        if (seeking() || seekTarget != null) return;
        if (Math.abs(follower.currentTime - leader.currentTime) > frameTime / 2) follower.currentTime = leader.currentTime;
      }

      function play() {
        if (ps.disposed || broken) return;
        wantPlay = true;
        if (!upgraded) { // now it's wanted: load ahead properly
          upgraded = true;
          leader.preload = 'auto';
          follower.preload = 'auto';
        }
        if (timeNow() >= duration - frameTime) seek(0);
        reflect();
        resume();
      }

      /** Starts both when nothing is loading, seeking or being dragged (each of those calls it again when done). */
      function resume() {
        if (!wantPlay || loading || scrub || broken || ps.disposed) return;
        if (seekTarget != null || seeking()) return;
        if (!canGoOn(leader) || !canGoOn(follower)) { // start together once both have enough, not one ahead
          startLoading();
          return;
        }
        held = false;
        leader.playbackRate = rate;
        follower.playbackRate = rate;
        align();
        if (leader.paused) startTogether();
        else if (follower.paused && !startWait) start(follower);
        startLoop();
      }

      /** Starts the upscaled video, and the original the moment the upscaled one's clock really moves
          (the first start can take a tenth of a second while the sound gets going), so they set off
          together instead of the original running ahead. */
      function startTogether() {
        const from = leader.currentTime;
        start(leader);
        if (!follower.paused || startWait) return;
        startBy = performance.now() + 500;
        const check = () => {
          startWait = 0;
          if (!wantPlay || held || loading || scrub || broken) return;
          if (leader.currentTime !== from || leader.ended || performance.now() > startBy) {
            start(follower);
            return;
          }
          startWait = ps.frame(check);
        };
        startWait = ps.frame(check);
      }

      function pause() {
        wantPlay = false;
        held = false;
        endLoading();
        stopBoth();
        follower.playbackRate = rate;
        align();
        reflect();
        draw(true);
      }

      function toggle() {
        if (wantPlay) pause();
        else play();
      }

      function seek(seconds) {
        const t = Math.max(0, Math.min(seconds, Math.max(0, duration - frameTime / 2)));
        if (wantPlay && !loading && !scrub) { // both hold still until both have arrived, then go on together
          held = true;
          stopBoth();
        }
        seekTarget = t;
        draw(true, t);
        pumpSeek();
      }

      /** Sends the newest wanted time to both videos, once neither is still busy with the last one. */
      function pumpSeek() {
        if (seekTarget == null || seeking()) return;
        const t = seekTarget;
        seekTarget = null;
        leader.currentTime = t;
        follower.currentTime = t;
        ps.clearTimeout(seekWatch);
        seekWatch = ps.timeout(() => {
          seekWatch = 0;
          if (seekTarget != null) { // a seek that never finished: send the newest time anyway
            const latest = seekTarget;
            seekTarget = null;
            leader.currentTime = latest;
            follower.currentTime = latest;
          }
        }, SEEK_WATCHDOG_MS);
      }

      function stepFrame(delta) {
        if (wantPlay) pause();
        const current = Math.min(frames - 1, frameAt(timeNow(), fps));
        seek(timeOfFrame(Math.max(0, Math.min(frames - 1, current + delta)), fps));
      }

      /* Waiting for data */

      // Ready to go on: enough loaded ahead, or the last frames (where there is nothing ahead to load).
      const canGoOn = (node) => node.readyState >= 3 || (node.readyState >= 2 && node.currentTime >= duration - 2 * frameTime);

      function startLoading() {
        if (loading) return;
        loading = true;
        stopBoth();
        stage.classList.add('is-buffering');
        showNotice(LOADING, false);
        ps.clearTimeout(loadingTimer);
        loadingTimer = ps.timeout(() => {
          loadingTimer = 0;
          if (loading) showNotice(SLOW, true);
        }, LOADING_SLOW_MS);
      }

      function endLoading() {
        if (!loading) return;
        loading = false;
        ps.clearTimeout(loadingTimer);
        loadingTimer = 0;
        stage.classList.remove('is-buffering');
        if (!broken) hideNotice();
      }

      function tryResume() {
        if (!loading || seekTarget != null || seeking()) return;
        if (!canGoOn(leader) || !canGoOn(follower)) return;
        endLoading();
        resume();
      }

      function showNotice(text, retry) {
        noticeText.textContent = text;
        noticeRetry.hidden = !retry;
        notice.hidden = false;
        notice.classList.toggle('has-action', Boolean(retry));
      }

      function hideNotice() {
        notice.hidden = true;
        notice.classList.remove('has-action');
      }

      function fail() {
        if (broken || ps.disposed) return;
        broken = true;
        wantPlay = false;
        loading = false;
        ps.clearTimeout(loadingTimer);
        stopBoth();
        stage.classList.remove('is-buffering');
        reflect();
        showNotice(BROKEN, true);
      }

      // Try again: ask the server afresh (the copies may have been tidied away) and build new players.
      ps.listen(noticeRetry, 'click', () => begin());

      for (const node of [leader, follower]) {
        ps.listen(node, 'waiting', () => {
          // A seek isn't a stall: it settles by itself (and 'seeked' starts both again).
          if (!wantPlay || loading || scrub || node.seeking || seekTarget != null) return;
          startLoading();
        });
        ps.listen(node, 'stalled', () => {
          if (wantPlay && !loading && !scrub && node.readyState < 3 && !node.seeking) startLoading();
        });
        for (const type of ['canplay', 'canplaythrough', 'playing', 'loadeddata']) ps.listen(node, type, tryResume);
        ps.listen(node, 'seeking', () => { if (node === follower) seekStarted = performance.now(); });
        ps.listen(node, 'seeked', () => {
          if (node === follower && seekStarted) {
            const took = (performance.now() - seekStarted) / 1000;
            seekCost = Math.min(0.3, seekCost * 0.7 + took * 0.3);
            seekStarted = 0;
          }
          pumpSeek();
          if (seekTarget != null || seeking()) return;
          ps.clearTimeout(seekWatch);
          draw(true);
          if (loading) tryResume();
          else resume();
        });
        ps.listen(node, 'error', fail);
        ps.listen(node, 'loadedmetadata', layout);
      }
      // The upscaled player is the clock: what it does by itself (the end, a media key), both do.
      ps.listen(leader, 'ended', () => {
        if (looping && wantPlay) { // both go back to the start together, then carry on together
          seek(0);
          return;
        }
        wantPlay = false;
        follower.pause();
        reflect();
        draw(true);
      });
      // Paused or started from outside the page (a headset unplugged, a media key): both follow.
      ps.listen(leader, 'pause', () => {
        if (ownPauses > 0) {
          ownPauses -= 1;
          return;
        }
        if (wantPlay && leader.paused && !broken && !leader.ended) pause();
      });
      ps.listen(leader, 'play', () => {
        if (ownPlays > 0) {
          ownPlays -= 1;
          return;
        }
        if (!wantPlay && !leader.paused && !broken) play();
      });
      ps.listen(leader, 'timeupdate', () => draw(false));

      /** The drift check: at most every SYNC_MS, never while seeking, loading or dragging. */
      function sync() {
        if (loading) { // in case the video that was waiting got its data without saying so
          tryResume();
          return;
        }
        if (!wantPlay || held || scrub || broken || startWait || seekTarget != null || seeking()) return;
        if (leader.ended) return;
        if (leader.paused) { // stopped by itself (a browser saving power, say): start it again
          start(leader);
          return;
        }
        if (follower.ended) return; // the original can be a frame shorter: it waits on its last frame
        if (follower.paused) {
          align();
          start(follower);
          return;
        }
        const drift = loopDrift(follower.currentTime - leader.currentTime, duration, looping);
        // Past half a frame the two would often show different frames (55 next to 56), so that's where
        // the catching up starts; once it has, it goes on to a quarter of a frame.
        const act = driftAction(drift, fps, rate, correcting ? frameTime / 4 : frameTime / 2);
        correcting = act.seek || act.rate !== rate;
        if (act.seek) {
          // One jump, a little ahead, so it lands where the upscaled side will be once the seek is done.
          follower.currentTime = Math.min(duration - frameTime / 2, leader.currentTime + seekCost * rate);
          return;
        }
        if (Math.abs(follower.playbackRate - act.rate) > 1e-3) follower.playbackRate = act.rate;
      }
      ps.interval(sync, SYNC_MS);

      /* Drawing: once per new frame, and only what changed */

      function startLoop() {
        if (loop || ps.disposed) return;
        loop = ps.videoFrame(leader, onFrame);
      }

      function onFrame(now, metadata) {
        loop = 0;
        draw(false, metadata && Number.isFinite(metadata.mediaTime) ? metadata.mediaTime : undefined);
        if (wantPlay && !ps.disposed) loop = ps.videoFrame(leader, onFrame);
      }

      function draw(force, at) {
        const t = at != null ? at : timeNow();
        const n = Math.min(frames - 1, frameAt(t, fps));
        if (n === lastFrame && !force) return;
        lastFrame = n;
        clock.textContent = `${formatClock(t)} / ${durationText}`;
        frameText.textContent = `Frame ${num(n + 1)} of ${framesText}`;
        if (!scrub) timeline.value = String(n);
        timeline.style.setProperty('--fill', `${frames > 1 ? (n / (frames - 1)) * 100 : 0}%`);
        // Screen readers hear the position when it's still, not thirty times a second.
        if (!wantPlay || force) timeline.setAttribute('aria-valuetext', `${formatClock(t)} of ${durationText}, frame ${n + 1} of ${frames}`);
      }

      function reflect() {
        const playing = wantPlay;
        if (shownPlaying === playing) return;
        shownPlaying = playing;
        const word = playing ? 'Pause (Space)' : 'Play (Space)';
        playIcon.firstChild.setAttribute('href', `#i-${playing ? 'pause' : 'play'}`);
        playButton.setAttribute('aria-label', word);
        playButton.title = word;
        playButton.classList.toggle('is-playing', playing);
      }

      /* Controls */

      ps.listen(playButton, 'click', toggle);
      ps.listen(back, 'click', () => stepFrame(-1));
      ps.listen(forward, 'click', () => stepFrame(1));

      // Dragging along the timeline: both pause while it's held and carry on after if they were playing.
      ps.listen(timeline, 'pointerdown', (event) => {
        if (event.pointerType === 'mouse' && event.button !== 0) return;
        scrub = { pointerId: event.pointerId };
        stopBoth();
        try {
          timeline.setPointerCapture(event.pointerId);
        } catch {
          /* The range keeps the pointer by itself in most browsers. */
        }
      });
      const endScrub = (event) => {
        if (!scrub || (event && event.pointerId != null && scrub.pointerId != null && event.pointerId !== scrub.pointerId)) return;
        scrub = null;
        draw(true);
        resume();
      };
      for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) ps.listen(timeline, type, endScrub);
      ps.listen(timeline, 'input', () => seek(timeOfFrame(Number(timeline.value), fps)));
      ps.listen(timeline, 'change', () => { if (scrub) endScrub(); });

      ps.listen(speed, 'change', () => {
        rate = Number(speed.value) || 1;
        leader.playbackRate = rate;
        follower.playbackRate = rate;
      });
      ps.listen(loopButton, 'click', () => {
        looping = !looping; // handled here, not with the videos' own loop, so both restart together
        loopButton.setAttribute('aria-pressed', String(looping));
      });
      function setMuted(muted) {
        leader.muted = muted;
        follower.muted = true; // the original never plays sound: one soundtrack, from the upscaled side
        mute.setAttribute('aria-pressed', String(muted));
        muteIcon.firstChild.setAttribute('href', `#i-${muted ? 'volume-off' : 'volume'}`);
        mute.setAttribute('aria-label', muted ? 'Unmute (M)' : 'Mute (M)');
      }
      ps.listen(mute, 'click', () => setMuted(!leader.muted));
      if (hasSound) setMuted(false);

      /* Zoom and the shared drag */

      function layout() {
        if (ps.disposed) return;
        // Layout sizes (the dialog is scaled for a moment as it opens, which shrinks on-screen boxes).
        const stageBox = { width: stage.clientWidth, height: stage.clientHeight };
        const ref = sizes.upscaled.width ? sizes.upscaled : { width: upscaled.videoWidth || 16, height: upscaled.videoHeight || 9 };
        const arrangement = videoLayout(ref.width, ref.height, stageBox.width, stageBox.height);
        if (stage.dataset.layout !== arrangement) stage.dataset.layout = arrangement;
        const box = { width: left.view.clientWidth, height: left.view.clientHeight };
        if (!box.width || !box.height) return;
        const dpr = window.devicePixelRatio || 1;
        const shown = displaySize(zoom, ref, box, dpr);
        let x = box.width / 2 - pan.x * shown.width;
        let y = box.height / 2 - pan.y * shown.height;
        x = shown.width <= box.width ? (box.width - shown.width) / 2 : Math.min(0, Math.max(box.width - shown.width, x));
        y = shown.height <= box.height ? (box.height - shown.height) / 2 : Math.min(0, Math.max(box.height - shown.height, y));
        pan.x = (box.width / 2 - x) / shown.width;
        pan.y = (box.height / 2 - y) / shown.height;
        if (zoom !== 'fit') { // on whole screen pixels, so 100% stays sharp
          x = Math.round(x * dpr) / dpr;
          y = Math.round(y * dpr) / dpr;
        }
        for (const node of [original, upscaled]) {
          node.style.width = `${shown.width}px`;
          node.style.height = `${shown.height}px`;
          node.style.transform = `translate(${x}px, ${y}px)`;
        }
        const zoomed = shown.width > box.width + 0.5 || shown.height > box.height + 0.5;
        stage.classList.toggle('is-zoomed', zoomed);
      }

      // The zoom changes at once: animating the size of two 4K videos costs a layout every frame, and
      // the presses have to feel instant.
      function setZoom(next, at) {
        if (!ZOOMS.some((z) => z.id === next)) return;
        if (at) { // the spot that was double-clicked comes to the middle
          pan.x = at.x;
          pan.y = at.y;
        }
        zoom = next;
        for (const input of zoomGroup.querySelectorAll('input')) input.checked = input.value === next;
        layout();
      }
      ps.listen(zoomGroup, 'change', (event) => { if (event.target.name === `${id}-zoom`) setZoom(event.target.value); });

      let drag = null;
      for (const side of [left, right]) {
        ps.listen(side.view, 'pointerdown', (event) => {
          if (!stage.classList.contains('is-zoomed') || (event.pointerType === 'mouse' && event.button !== 0)) return;
          drag = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
          try {
            side.view.setPointerCapture(event.pointerId);
          } catch {
            /* Dragging still works while the pointer stays over the video. */
          }
          stage.classList.add('is-panning');
        });
        ps.listen(side.view, 'pointermove', (event) => {
          if (!drag || event.pointerId !== drag.pointerId) return;
          const shown = upscaled.getBoundingClientRect();
          pan.x -= (event.clientX - drag.x) / Math.max(1, shown.width);
          pan.y -= (event.clientY - drag.y) / Math.max(1, shown.height);
          drag.x = event.clientX;
          drag.y = event.clientY;
          layout();
        });
        const end = () => {
          drag = null;
          stage.classList.remove('is-panning');
        };
        for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) ps.listen(side.view, type, end);
        ps.listen(side.view, 'dblclick', (event) => {
          const node = side === left ? original : upscaled;
          const shown = node.getBoundingClientRect();
          const at = { x: (event.clientX - shown.left) / shown.width, y: (event.clientY - shown.top) / shown.height };
          setZoom(zoom === 'fit' ? '1' : 'fit', zoom === 'fit' ? at : null);
        });
      }

      if ('ResizeObserver' in window) ps.observe(new ResizeObserver(() => layout())).observe(stage);
      ps.listen(window, 'resize', layout);
      layout();
      reflect();
      draw(true);

      function handleKey(event) {
        if (event.altKey || event.ctrlKey || event.metaKey) return false;
        const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
        if (key === ' ' || key === 'k') toggle();
        else if (key === ',') stepFrame(-1);
        else if (key === '.') stepFrame(1);
        else if (key === 'j' || key === 'ArrowLeft') seek(timeNow() - JUMP_SECONDS);
        else if (key === 'l' || key === 'ArrowRight') seek(timeNow() + JUMP_SECONDS);
        else if (key === 'm' && hasSound) setMuted(!leader.muted);
        else if (key === '0') setZoom('fit');
        else if (key === '1') setZoom('1');
        else if (key === '2') setZoom('2');
        else return false;
        return true;
      }

      if (takeFocus) playButton.focus({ preventScroll: true });

      return {
        handleKey,
        destroy: () => ps.dispose(), // pauses and unloads both videos, ends every timer and listener
        debug: () => {
          const quality = (node) => {
            const q = typeof node.getVideoPlaybackQuality === 'function' ? node.getVideoPlaybackQuality() : null;
            return q ? { total: q.totalVideoFrames, dropped: q.droppedVideoFrames } : null;
          };
          return {
            leader: leader.currentTime, follower: follower.currentTime, drift: follower.currentTime - leader.currentTime,
            paused: [leader.paused, follower.paused], rate: [leader.playbackRate, follower.playbackRate], wantPlay,
            buffering: loading, loading, broken, seeking: seeking() || seekTarget != null, scrubbing: Boolean(scrub),
            readyState: [leader.readyState, follower.readyState], frame: frameAt(leader.currentTime, fps), frames, fps,
            zoom, muted: [leader.muted, follower.muted], looping, notice: notice.hidden ? '' : noticeText.textContent,
            quality: { upscaled: quality(leader), original: quality(follower) },
          };
        },
      };
    }

    begin();

    return {
      root,
      handleKey: (event) => (player ? player.handleKey(event) : false),
      debug: () => (player ? player.debug() : null),
      /** Stops everything this comparison started. Safe to call more than once. */
      destroy() {
        if (scope.disposed) return;
        generation += 1;
        dropPlayer();
        scope.dispose();
      },
    };
  }

  /* ---------- The dialog ---------- */

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  /**
   * deps: {dialog, h, icon, api, life, trap (count, index, backwards) -> index | -1}
   * open(item, closed): item {title, subtitle, startUrl, statusUrl, cancelUrl, download: {href, name}}
   *
   * Close, Esc and a click outside close it straight away, every time, whatever it is doing: the
   * comparison is torn down first (one idempotent call), then the dialog closes and focus goes back
   * to whatever opened it.
   */
  function createDialog(deps) {
    const { dialog, h, icon } = deps;
    const title = dialog.querySelector('[data-vc="title"]');
    const subtitle = dialog.querySelector('[data-vc="subtitle"]');
    const actions = dialog.querySelector('[data-vc="actions"]');
    const stage = dialog.querySelector('[data-vc="stage"]');
    let current = null;
    let onClose = null;
    let opener = null;

    /** Stops the comparison and empties the dialog. Safe to call any number of times. */
    function teardown() {
      if (current) {
        const old = current;
        current = null;
        try {
          old.destroy();
        } catch (err) {
          window.console.warn('Re:Solution: closing the comparison', err);
        }
      }
      stage.replaceChildren();
    }

    function open(item, closed = null) {
      teardown();
      const active = document.activeElement;
      if (!dialog.open) opener = active && active !== document.body && !dialog.contains(active) ? active : null;
      onClose = closed;
      title.textContent = item.title || 'Compare';
      subtitle.textContent = item.subtitle || '';
      actions.replaceChildren(...(item.download ? [h('a', { class: 'btn btn-primary btn-sm', href: item.download.href, download: item.download.name }, icon('download'), 'Download')] : []));
      current = createComparison(deps, item);
      stage.replaceChildren(current.root);
      if (!dialog.open) dialog.showModal();
      title.focus({ preventScroll: true });
    }

    /** Closes at once, from any state. */
    function close() {
      teardown();
      if (dialog.open) dialog.close();
    }

    // However it closed (Close, Esc, a click outside, another page), the 'close' event follows.
    dialog.addEventListener('close', () => {
      if (dialog.open) return; // it arrives a frame later: opened again meanwhile, so it isn't this visit's
      teardown();
      const done = onClose;
      const back = opener;
      onClose = null;
      opener = null;
      if (done) done();
      else if (back && back.isConnected) back.focus({ preventScroll: true });
    });
    // Esc through the browser's own route: stop everything before the close event even arrives.
    dialog.addEventListener('cancel', teardown);
    for (const button of dialog.querySelectorAll('[data-close]')) button.addEventListener('click', close);
    // A click on the backdrop lands on the dialog element itself, outside its box. (A drag that ends
    // somewhere else inside also reaches the dialog element, so the position decides.)
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const box = dialog.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) close();
    });

    dialog.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
        return;
      }
      if (event.key === 'Tab') {
        const controls = Array.from(dialog.querySelectorAll(FOCUSABLE)).filter((node) => node.getClientRects().length > 0
          && !(node.type === 'radio' && !node.checked && Array.from(dialog.querySelectorAll(`input[name="${CSS.escape(node.name)}"]`)).some((r) => r.checked)));
        const trap = deps.trap || (() => -1);
        const to = trap(controls.length, controls.indexOf(document.activeElement), event.shiftKey);
        if (to >= 0) {
          event.preventDefault();
          controls[to].focus();
        }
        return;
      }
      if (!current) return;
      const from = event.target;
      if (!from || !from.closest) return;
      // Controls keep their own keys: Space presses a button, arrows move the timeline or the zoom choice.
      // The timeline keeps only those, so after dragging it Space, , . J L still work.
      if (from.closest('select, textarea, input:not([type="radio"]):not([type="range"])')) return;
      if ((event.key === ' ' || event.key === 'Enter') && from.closest('button, a')) return;
      if (/^(Arrow|Home$|End$|Page)/.test(event.key) && from.closest('input[type="radio"], input[type="range"]')) return;
      if (current.handleKey(event)) event.preventDefault();
    });

    return { open, close, isOpen: () => dialog.open, debug: () => (current ? current.debug() : null) };
  }

  window.ReVideoCompare = { createDialog, createComparison, ...helpers };
})();
