/* Everything a viewer starts, it also stops: one "scope" per viewer holds its timers, animation
   frames, video-frame callbacks, event listeners, observers, network requests and videos, and
   dispose() ends all of them at once. Closing a viewer is then a single call that can't miss
   anything and can safely run twice (a click on Close and the dialog's own close event, say).

   The counts of what is still running are kept for checks in the browser:
   window.__resolutionDebug.counts() (the automated stress tests read it after every close).

   Plain browser script, loaded before the viewers. Also runs under Node for tests. */
(function () {
  'use strict';

  const root = typeof window !== 'undefined' ? window : globalThis;
  const live = { scopes: 0, timeouts: 0, intervals: 0, frames: 0, videoFrames: 0, listeners: 0, observers: 0, requests: 0, videos: 0 };

  function nextFrame(fn) {
    if (typeof root.requestAnimationFrame === 'function') return { kind: 'raf', id: root.requestAnimationFrame(fn) };
    return { kind: 'timeout', id: root.setTimeout(() => fn(Date.now()), 16) };
  }

  function cancelFrameHandle(handle) {
    if (handle.kind === 'raf') root.cancelAnimationFrame(handle.id);
    else root.clearTimeout(handle.id);
  }

  /** A video lets go of its file (and its network connection) only when its source is removed and it
      is told to load nothing. Safe to call on a video that was never started. */
  function unloadVideo(video) {
    try {
      video.pause();
    } catch {
      /* A video that is already gone. */
    }
    try {
      video.removeAttribute('src');
      for (const source of Array.from(video.querySelectorAll ? video.querySelectorAll('source') : [])) source.remove();
      video.load();
    } catch {
      /* Nothing more to let go of. */
    }
  }

  /** A scope: `name` only helps when reading the counts. */
  function createScope(name = 'scope') {
    const timeouts = new Set();
    const intervals = new Set();
    const frames = new Map(); // our id -> handle
    const videoFrames = new Map(); // our id -> {video, id}
    const listeners = new Set();
    const observers = new Set();
    const controllers = new Set();
    const videos = new Set();
    const cleanups = [];
    let disposed = false;
    let seq = 0;
    live.scopes += 1;

    const scope = {
      name,
      get disposed() {
        return disposed;
      },

      timeout(fn, ms) {
        if (disposed) return 0;
        const id = root.setTimeout(() => {
          if (!timeouts.delete(id)) return;
          live.timeouts -= 1;
          fn();
        }, ms);
        timeouts.add(id);
        live.timeouts += 1;
        return id;
      },
      clearTimeout(id) {
        if (!timeouts.delete(id)) return;
        root.clearTimeout(id);
        live.timeouts -= 1;
      },

      interval(fn, ms) {
        if (disposed) return 0;
        const id = root.setInterval(fn, ms);
        intervals.add(id);
        live.intervals += 1;
        return id;
      },
      clearInterval(id) {
        if (!intervals.delete(id)) return;
        root.clearInterval(id);
        live.intervals -= 1;
      },

      /** requestAnimationFrame that disposing cancels. Returns an id for cancelFrame. */
      frame(fn) {
        if (disposed) return 0;
        seq += 1;
        const mine = seq;
        const handle = nextFrame((time) => {
          if (!frames.delete(mine)) return;
          live.frames -= 1;
          fn(time);
        });
        frames.set(mine, handle);
        live.frames += 1;
        return mine;
      },
      cancelFrame(mine) {
        const handle = frames.get(mine);
        if (!handle) return;
        frames.delete(mine);
        cancelFrameHandle(handle);
        live.frames -= 1;
      },

      /** requestVideoFrameCallback where the browser has it (called when the video shows a new
          frame), otherwise the next animation frame. fn(now, metadata | null). */
      videoFrame(video, fn) {
        if (disposed) return 0;
        if (!video || typeof video.requestVideoFrameCallback !== 'function') return scope.frame((now) => fn(now, null));
        seq += 1;
        const mine = seq;
        const id = video.requestVideoFrameCallback((now, metadata) => {
          if (!videoFrames.delete(mine)) return;
          live.videoFrames -= 1;
          fn(now, metadata);
        });
        videoFrames.set(mine, { video, id });
        live.videoFrames += 1;
        return mine;
      },
      cancelVideoFrame(mine) {
        const entry = videoFrames.get(mine);
        if (entry) {
          videoFrames.delete(mine);
          try {
            entry.video.cancelVideoFrameCallback(entry.id);
          } catch {
            /* The video is already unloaded. */
          }
          live.videoFrames -= 1;
          return;
        }
        scope.cancelFrame(mine);
      },

      /** addEventListener that disposing removes. Returns a function that removes it sooner. */
      listen(target, type, fn, options) {
        if (disposed || !target) return () => {};
        target.addEventListener(type, fn, options);
        const entry = { target, type, fn, options };
        listeners.add(entry);
        live.listeners += 1;
        return () => {
          if (!listeners.delete(entry)) return;
          target.removeEventListener(type, fn, options);
          live.listeners -= 1;
        };
      },

      /** A ResizeObserver, IntersectionObserver or MutationObserver to disconnect when disposing. */
      observe(observer) {
        if (disposed) {
          observer.disconnect();
          return observer;
        }
        observers.add(observer);
        live.observers += 1;
        return observer;
      },

      /** An AbortController for a request; disposing aborts it. Call done(controller) when it settles. */
      controller() {
        const ctrl = typeof AbortController === 'function' ? new AbortController() : { signal: undefined, abort() {} };
        if (disposed) {
          ctrl.abort();
          return ctrl;
        }
        controllers.add(ctrl);
        live.requests += 1;
        return ctrl;
      },
      done(ctrl) {
        if (!controllers.delete(ctrl)) return;
        live.requests -= 1;
      },

      /** fetch() that disposing aborts. */
      fetch(url, init = {}) {
        const ctrl = scope.controller();
        if (disposed) return Promise.reject(Object.assign(new Error('Closed'), { name: 'AbortError' }));
        return root.fetch(url, { ...init, signal: ctrl.signal }).finally(() => scope.done(ctrl));
      },

      /** A video to pause and unload when disposing. */
      video(node) {
        if (disposed) {
          unloadVideo(node);
          return node;
        }
        if (!videos.has(node)) {
          videos.add(node);
          live.videos += 1;
        }
        return node;
      },
      /** Unloads one of the scope's videos now. */
      unload(node) {
        if (videos.delete(node)) live.videos -= 1;
        unloadVideo(node);
      },

      /** Anything else to undo when disposing. */
      add(fn) {
        if (disposed) {
          fn();
          return;
        }
        cleanups.push(fn);
      },

      /** Stops everything. Safe to call more than once. */
      dispose() {
        if (disposed) return;
        disposed = true;
        live.scopes -= 1;
        for (const id of timeouts) root.clearTimeout(id);
        live.timeouts -= timeouts.size;
        timeouts.clear();
        for (const id of intervals) root.clearInterval(id);
        live.intervals -= intervals.size;
        intervals.clear();
        for (const handle of frames.values()) cancelFrameHandle(handle);
        live.frames -= frames.size;
        frames.clear();
        for (const { video, id } of videoFrames.values()) {
          try {
            video.cancelVideoFrameCallback(id);
          } catch {
            /* Already unloaded. */
          }
        }
        live.videoFrames -= videoFrames.size;
        videoFrames.clear();
        for (const { target, type, fn, options } of listeners) target.removeEventListener(type, fn, options);
        live.listeners -= listeners.size;
        listeners.clear();
        for (const observer of observers) observer.disconnect();
        live.observers -= observers.size;
        observers.clear();
        for (const ctrl of controllers) ctrl.abort();
        live.requests -= controllers.size;
        controllers.clear();
        for (const node of videos) unloadVideo(node);
        live.videos -= videos.size;
        videos.clear();
        for (const fn of cleanups.splice(0)) {
          try {
            fn();
          } catch (err) {
            // One clean-up that fails mustn't stop the rest (or the dialog closing).
            if (root.console) root.console.warn('Re:Solution: a clean-up step failed', err);
          }
        }
      },
    };
    return scope;
  }

  /** What is still running, across every scope. */
  function counts() {
    return { ...live };
  }

  /** A request that the page stopped on purpose (closing a viewer), not a real failure. */
  function isAbort(err) {
    return Boolean(err) && (err.name === 'AbortError' || err.aborted === true);
  }

  const api = { createScope, counts, isAbort, unloadVideo };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window === 'undefined') return;
  window.ReLife = api;
  // For checks in the browser (and the automated stress tests): what is still running. app.js adds
  // the viewers' own debug views to it.
  window.__resolutionDebug = Object.assign(window.__resolutionDebug || {}, { counts });
})();
