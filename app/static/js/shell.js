/*
 * shell.js -- the things every page needs: theme, toasts, the offline
 * indicator, the idle-session countdown, and the offline write queue.
 *
 * Kept separate from app.js (which is about summarizing) because these
 * are application chrome. A bug in the canvas shouldn't break the theme
 * toggle, and vice versa.
 */
(function () {
  "use strict";

  // -------------------------------------------------------------------
  // Theme
  //
  // Stored in localStorage AND mirrored to a cookie. localStorage is
  // what the inline script in <head> reads to avoid a flash of the wrong
  // theme; the cookie is what the server reads so the very first HTML it
  // sends already has the right attribute. Either alone leaves a gap.
  // -------------------------------------------------------------------
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("se-theme", theme);
    } catch (e) {
      /* private mode -- the cookie below still carries it */
    }
    document.cookie = "se-theme=" + theme + "; path=/; max-age=31536000; samesite=lax";
  }

  document.addEventListener("click", function (event) {
    var toggle = event.target.closest("#theme-toggle");
    if (!toggle) return;
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  });

  // -------------------------------------------------------------------
  // Toasts -- small, non-blocking confirmations. Exposed globally so any
  // page script can report success or failure the same way.
  // -------------------------------------------------------------------
  function toast(message, kind) {
    var host = document.getElementById("toasts");
    if (!host) return;
    var el = document.createElement("div");
    el.className = "se-toast se-toast-" + (kind || "info");
    el.setAttribute("role", "status");
    el.textContent = message;
    host.appendChild(el);
    // Force a frame so the entry transition runs instead of being
    // collapsed into the initial paint.
    requestAnimationFrame(function () {
      el.classList.add("se-toast-in");
    });
    setTimeout(function () {
      el.classList.remove("se-toast-in");
      setTimeout(function () {
        el.remove();
      }, 250);
    }, kind === "error" ? 6000 : 3200);
  }
  window.seToast = toast;

  // -------------------------------------------------------------------
  // Offline awareness + the write queue
  //
  // The queue is the client half of the CRDT sync in app/core/crdt.py.
  // Anything that changes data while offline is appended here with a
  // client-generated id and a Lamport counter, then flushed on
  // reconnect. Because operations carry their own identity, a flush
  // that fails halfway can simply be retried whole -- the server drops
  // the duplicates.
  // -------------------------------------------------------------------
  var QUEUE_KEY = "se-sync-queue";
  var REPLICA_KEY = "se-replica-id";
  var CLOCK_KEY = "se-lamport";

  function safeRead(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function safeWrite(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      // Quota exceeded or storage blocked. Losing the queue is bad, so
      // say so rather than failing silently.
      return false;
    }
  }

  function replicaId() {
    var id = safeRead(REPLICA_KEY, null);
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, "");
      safeWrite(REPLICA_KEY, id);
    }
    return id;
  }

  function nextLamport() {
    var value = safeRead(CLOCK_KEY, 0) + 1;
    safeWrite(CLOCK_KEY, value);
    return value;
  }

  function queueOperation(entity, entityId, field, action, value) {
    var queue = safeRead(QUEUE_KEY, []);
    queue.push({
      op_id: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, ""),
      replica_id: replicaId(),
      entity: entity,
      entity_id: String(entityId),
      field: field || "",
      action: action,
      value: value,
      lamport: nextLamport(),
      wall_clock: Date.now() / 1000
    });
    if (!safeWrite(QUEUE_KEY, queue)) {
      toast("Couldn't save that change locally — storage is full.", "error");
      return false;
    }
    updateOfflineBanner();
    if (navigator.onLine) flushQueue();
    return true;
  }
  window.seQueueOperation = queueOperation;

  var flushing = false;
  function flushQueue() {
    if (flushing || !navigator.onLine) return;
    var queue = safeRead(QUEUE_KEY, []);
    if (!queue.length) return;

    flushing = true;
    fetch("/api/sync/push", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ operations: queue })
    })
      .then(function (response) {
        if (!response.ok) throw new Error("sync rejected");
        return response.json();
      })
      .then(function (result) {
        // Only clear what we actually sent: anything queued during the
        // request must survive.
        var current = safeRead(QUEUE_KEY, []);
        safeWrite(QUEUE_KEY, current.slice(queue.length));
        if (result.accepted) toast("Synced " + result.accepted + " offline change" + (result.accepted === 1 ? "" : "s"), "success");
        updateOfflineBanner();
      })
      .catch(function () {
        // Leave the queue intact and try again on the next online event.
      })
      .finally(function () {
        flushing = false;
      });
  }

  function updateOfflineBanner() {
    var banner = document.getElementById("offline-banner");
    if (!banner) return;
    var pending = safeRead(QUEUE_KEY, []).length;
    if (!navigator.onLine) {
      banner.hidden = false;
      banner.textContent = pending
        ? "You're offline. " + pending + " change" + (pending === 1 ? "" : "s") + " saved here, waiting to sync."
        : "You're offline. You can keep reading and taking notes — changes save here and sync when you reconnect.";
    } else {
      banner.hidden = true;
    }
  }

  window.addEventListener("online", function () {
    updateOfflineBanner();
    flushQueue();
  });
  window.addEventListener("offline", updateOfflineBanner);
  document.addEventListener("DOMContentLoaded", function () {
    updateOfflineBanner();
    flushQueue();
  });

  // -------------------------------------------------------------------
  // Idle session countdown
  //
  // Losing half-typed work to a silent redirect is a miserable way to
  // discover a session expired. This polls the remaining time and warns
  // with enough notice to do something about it.
  // -------------------------------------------------------------------
  var warning = document.getElementById("session-warning");
  if (warning) {
    var countdownEl = document.getElementById("session-countdown");
    var stayButton = document.getElementById("session-stay");
    var remaining = null;
    var ticker = null;

    function renderCountdown() {
      if (remaining === null) return;
      if (remaining <= 0) {
        countdownEl.textContent = "a moment";
        return;
      }
      var minutes = Math.floor(remaining / 60);
      var seconds = remaining % 60;
      countdownEl.textContent = minutes > 0
        ? minutes + " min " + seconds + " sec"
        : seconds + " seconds";
    }

    function checkSession() {
      fetch("/api/session/heartbeat", { credentials: "same-origin" })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
          if (!data) return;
          remaining = data.seconds_remaining;
          if (remaining === null) {
            warning.hidden = true;
            return;
          }
          // Warn with two minutes to spare.
          warning.hidden = remaining > 120;
          renderCountdown();
        })
        .catch(function () { /* offline; the banner covers it */ });
    }

    stayButton.addEventListener("click", function () {
      // Any authenticated request slides the idle window forward. This
      // one is cheap and has no side effects.
      fetch("/api/sync/status", { credentials: "same-origin" })
        .then(function () {
          warning.hidden = true;
          remaining = null;
          toast("You're still signed in.", "success");
        })
        .catch(function () {
          toast("Couldn't reach the server.", "error");
        });
    });

    ticker = setInterval(function () {
      if (remaining !== null && remaining > 0) {
        remaining -= 1;
        renderCountdown();
      }
    }, 1000);

    checkSession();
    setInterval(checkSession, 45000);
    window.addEventListener("beforeunload", function () { clearInterval(ticker); });
  }

  // -------------------------------------------------------------------
  // Work in progress
  //
  // Jobs run on the server and are rows in the database, so they survive
  // navigation, a closed tab, and a restart. What was missing was any
  // way for a page to learn about work it did not start itself, which is
  // why leaving the upload page used to look like the upload had died.
  //
  // This polls for the account's running jobs from every page and shows
  // one persistent bar. Progress is reported as steps completed rather
  // than a percentage, because the server genuinely does not know how
  // long a transcription will take and a fake percentage that sticks at
  // 90% is worse than an honest step count.
  // -------------------------------------------------------------------
  (function workTracker() {
    var bar = document.getElementById("workbar");
    if (!bar) return;

    var titleEl = document.getElementById("workbar-title");
    var stepEl = document.getElementById("workbar-step");
    var fillEl = document.getElementById("workbar-fill");
    var hideButton = document.getElementById("workbar-hide");
    var hiddenIds = {};
    var lastActiveIds = [];
    var pollTimer = null;

    var KIND_LABELS = {
      text: "Summarising text",
      pdf: "Reading a document",
      article: "Fetching a web page",
      youtube: "Processing a video",
      audio: "Processing a recording",
      image: "Reading an image",
      video: "Processing a video"
    };

    hideButton.addEventListener("click", function () {
      lastActiveIds.forEach(function (id) { hiddenIds[id] = true; });
      bar.hidden = true;
    });

    function render(active) {
      var visible = active.filter(function (job) { return !hiddenIds[job.id]; });
      if (!visible.length) {
        bar.hidden = true;
        return;
      }

      var job = visible[0];
      var label = KIND_LABELS[job.kind] || "Working";
      titleEl.textContent = visible.length > 1
        ? label + " (and " + (visible.length - 1) + " more)"
        : label;
      stepEl.textContent = job.latest || "";

      // Most pipelines report five or six steps. Cap the visual fill
      // short of full so it never looks finished while it is running.
      var fraction = Math.min(0.92, (job.step_count || 0) / 6);
      fillEl.style.width = Math.round(fraction * 100) + "%";
      bar.hidden = false;
    }

    function poll() {
      fetch("/api/jobs/active", { credentials: "same-origin" })
        .then(function (response) { return response.ok ? response.json() : null; })
        .then(function (data) {
          if (!data) return;

          var active = data.active || [];
          var activeIds = active.map(function (job) { return job.id; });

          // Anything that was running last time and is not now has
          // finished while the user was elsewhere. Say so, once.
          lastActiveIds.forEach(function (id) {
            if (activeIds.indexOf(id) === -1 && !hiddenIds[id]) {
              var done = (data.recent || []).filter(function (r) { return r.id === id; })[0];
              if (done && done.doc_id) {
                toast("Finished. It's in your records.", "success");
              }
            }
          });

          lastActiveIds = activeIds;
          render(active);

          // Poll briskly while something is running, and back off to an
          // idle heartbeat when nothing is, so a quiet tab is not making
          // a request every two seconds all day.
          schedule(active.length ? 2500 : 15000);
        })
        .catch(function () { schedule(20000); });
    }

    function schedule(delay) {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(poll, delay);
    }

    document.addEventListener("DOMContentLoaded", poll);
    // Check immediately on return to a backgrounded tab rather than
    // waiting out the remaining interval.
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden) schedule(200);
    });
  })();

  // -------------------------------------------------------------------
  // A confirm() replacement that matches the rest of the interface.
  // Returns a promise so callers read linearly.
  // -------------------------------------------------------------------
  window.seConfirm = function (message, confirmLabel) {
    return new Promise(function (resolve) {
      var overlay = document.createElement("div");
      overlay.className = "se-confirm-overlay";
      overlay.innerHTML =
        '<div class="se-confirm" role="dialog" aria-modal="true">' +
        '<p class="se-confirm-message"></p>' +
        '<div class="se-confirm-actions">' +
        '<button type="button" class="btn btn-sm btn-outline-secondary" data-act="cancel">Cancel</button>' +
        '<button type="button" class="btn btn-sm btn-danger" data-act="ok"></button>' +
        "</div></div>";
      overlay.querySelector(".se-confirm-message").textContent = message;
      overlay.querySelector('[data-act="ok"]').textContent = confirmLabel || "Confirm";
      document.body.appendChild(overlay);

      var okButton = overlay.querySelector('[data-act="ok"]');
      okButton.focus();

      function close(result) {
        overlay.remove();
        document.removeEventListener("keydown", onKey);
        resolve(result);
      }
      function onKey(event) {
        if (event.key === "Escape") close(false);
      }
      overlay.addEventListener("click", function (event) {
        if (event.target === overlay) return close(false);
        var act = event.target.getAttribute("data-act");
        if (act === "ok") close(true);
        if (act === "cancel") close(false);
      });
      document.addEventListener("keydown", onKey);
    });
  };
})();
