/*
 * reader.js -- SummarEase RSVP speed reader.
 *
 * RSVP works by removing saccades: the words come to the eye instead of
 * the eye going to the words. That only pays off if the fixation point
 * truly never moves, so the central mechanic here is that the ORP
 * character sits at one fixed x position and the rest of the word slides
 * around it. Everything else -- the pacing, the sentence rewind, the
 * countdown -- exists to protect that.
 *
 * No build step, no framework, no dependencies. Bootstrap 5 is only used
 * for layout classes.
 */
(function () {
  "use strict";

  var ROOT_ID = "rsvp-root";
  var root = document.getElementById(ROOT_ID);
  if (!root) {
    return;
  }

  var REDUCED_MOTION =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var WPM_MIN = 100;
  var WPM_MAX = 1000;
  var WPM_STEP = 25;
  var MIN_FRAME_MS = 34; // two frames at 60Hz; mirrors reading.py
  var STORAGE_KEY = "se.reader.wpm";

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------

  var state = {
    doc: null, // {id, title, text, words, questions?}
    words: [],
    schedule: [], // per-word milliseconds
    index: 0,
    playing: false,
    finished: false,
    wpm: 300,
    timer: null,
    countdownTimer: null,
    nextAt: 0, // performance.now() target for the next advance
    elapsedMs: 0, // accumulated playing time only
    playStartedAt: 0,
    maxIndexSeen: 0,
    sessionPosted: false,
    questions: null,
    answers: [],
    documents: []
  };

  // -------------------------------------------------------------------
  // Small helpers
  // -------------------------------------------------------------------

  function el(id) {
    return document.getElementById(id);
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function text(node, value) {
    if (node) {
      node.textContent = value;
    }
  }

  function show(node, visible) {
    if (node) {
      node.hidden = !visible;
    }
  }

  function formatClock(ms) {
    var total = Math.max(0, Math.round(ms / 1000));
    var minutes = Math.floor(total / 60);
    var seconds = total % 60;
    return minutes + ":" + (seconds < 10 ? "0" : "") + seconds;
  }

  function setStatus(message, tone) {
    var node = el("rsvp-status");
    if (!node) {
      return;
    }
    node.textContent = message || "";
    node.className = "rsvp-status" + (tone ? " rsvp-status-" + tone : "");
    show(node, Boolean(message));
  }

  function announce(message) {
    var node = el("rsvp-live");
    if (node) {
      node.textContent = message;
    }
  }

  function api(path, options) {
    var settings = options || {};
    settings.credentials = "same-origin";
    settings.headers = settings.headers || {};
    settings.headers.Accept = "application/json";
    if (settings.body && !settings.headers["Content-Type"]) {
      settings.headers["Content-Type"] = "application/json";
    }
    return fetch(path, settings).then(function (response) {
      if (!response.ok) {
        var error = new Error("Request failed with status " + response.status);
        error.status = response.status;
        throw error;
      }
      return response.json();
    });
  }

  // -------------------------------------------------------------------
  // Pacing -- mirrors app/core/reading.py:pacing_plan
  //
  // The multipliers are normalised to a mean of 1.0 before scaling, so
  // the requested wpm is the wpm you actually get: time is redistributed
  // across the stream rather than added to it. The server computes the
  // same schedule; it is duplicated here because the speed changes live
  // and a round trip per keypress would be absurd.
  // -------------------------------------------------------------------

  function buildSchedule(words, wpm) {
    var schedule = new Array(words.length);
    if (!words.length) {
      return schedule;
    }
    var baseMs = 60000 / clamp(wpm, 50, 1500);

    var sum = 0;
    var i;
    for (i = 0; i < words.length; i++) {
      var m = Number(words[i].delay_multiplier);
      sum += isFinite(m) && m > 0 ? m : 1;
    }
    var mean = sum / words.length || 1;

    var carry = 0;
    for (i = 0; i < words.length; i++) {
      var multiplier = Number(words[i].delay_multiplier);
      if (!isFinite(multiplier) || multiplier <= 0) {
        multiplier = 1;
      }
      var exact = baseMs * (multiplier / mean) + carry;
      var slot = Math.round(exact);
      if (slot < MIN_FRAME_MS) {
        slot = MIN_FRAME_MS;
      }
      carry = exact - slot;
      schedule[i] = slot;
    }
    return schedule;
  }

  function remainingMs(fromIndex) {
    var total = 0;
    for (var i = fromIndex; i < state.schedule.length; i++) {
      total += state.schedule[i];
    }
    return total;
  }

  // -------------------------------------------------------------------
  // The display
  // -------------------------------------------------------------------

  function renderWord(entry) {
    var left = el("rsvp-left");
    var pivot = el("rsvp-pivot");
    var right = el("rsvp-right");
    if (!entry) {
      text(left, "");
      text(pivot, "");
      text(right, "");
      return;
    }
    var word = String(entry.word || "");
    var orp = clamp(Number(entry.orp) || 0, 0, Math.max(0, word.length - 1));
    // Three spans in a 1fr / auto / 1fr grid. The pivot column is the
    // only one sized to content, so its centre sits on the container's
    // centre line no matter how the flanks change length. That is the
    // fixed fixation point, done in CSS rather than measured in JS so it
    // cannot drift by a subpixel between frames.
    text(left, word.slice(0, orp));
    text(pivot, word.charAt(orp));
    text(right, word.slice(orp + 1));
  }

  function updateStats() {
    var total = state.words.length;
    var done = Math.min(state.index, total);
    var left = Math.max(0, total - done);

    text(el("rsvp-stat-wpm"), String(state.wpm));
    text(el("rsvp-stat-elapsed"), formatClock(currentElapsed()));
    text(el("rsvp-stat-left"), String(left));
    text(el("rsvp-stat-finish"), left ? "~" + formatClock(remainingMs(done)) : "done");

    var percent = total ? (done / total) * 100 : 0;
    var bar = el("rsvp-progress-bar");
    if (bar) {
      bar.style.width = percent.toFixed(2) + "%";
    }
    var scrub = el("rsvp-scrub");
    if (scrub && document.activeElement !== scrub) {
      scrub.value = String(Math.round(percent));
    }
    var scrubLabel = el("rsvp-scrub-label");
    if (scrubLabel) {
      scrubLabel.textContent = Math.round(percent) + "%";
    }
    if (scrub) {
      scrub.setAttribute("aria-valuetext", Math.round(percent) + " percent read");
    }
  }

  function currentElapsed() {
    if (state.playing && state.playStartedAt) {
      return state.elapsedMs + (performance.now() - state.playStartedAt);
    }
    return state.elapsedMs;
  }

  function updatePlayButton() {
    var button = el("rsvp-play");
    if (!button) {
      return;
    }
    var label = state.playing ? "Pause" : state.finished ? "Replay" : "Play";
    button.textContent = label;
    button.setAttribute("aria-label", label + " (space)");
    button.setAttribute("aria-pressed", state.playing ? "true" : "false");
  }

  // -------------------------------------------------------------------
  // Playback
  // -------------------------------------------------------------------

  function clearTimers() {
    if (state.timer) {
      clearTimeout(state.timer);
      state.timer = null;
    }
    if (state.countdownTimer) {
      clearTimeout(state.countdownTimer);
      state.countdownTimer = null;
    }
  }

  function step() {
    if (!state.playing) {
      return;
    }
    if (state.index >= state.words.length) {
      finish();
      return;
    }

    var entry = state.words[state.index];
    renderWord(entry);
    if (state.index > state.maxIndexSeen) {
      state.maxIndexSeen = state.index;
    }
    updateStats();

    var slot = state.schedule[state.index] || 200;
    state.nextAt += slot;
    // Schedule against an absolute target rather than a fixed delay, so
    // the stream does not slowly fall behind when the main thread is busy.
    var delay = Math.max(0, state.nextAt - performance.now());
    state.timer = setTimeout(function () {
      state.index += 1;
      step();
    }, delay);
  }

  function startPlaying() {
    state.playing = true;
    state.playStartedAt = performance.now();
    state.nextAt = performance.now();
    updatePlayButton();
    setStatus("");
    step();
  }

  function countdownThenPlay() {
    // A pre-flight countdown is not decoration. Starting cold means the
    // first several words go by while the eye is still finding the
    // fixation point, and those are usually the words that set up the
    // whole passage.
    clearTimers();
    var overlay = el("rsvp-countdown");
    if (!overlay) {
      startPlaying();
      return;
    }

    var counter = 3;
    show(overlay, true);
    overlay.classList.toggle("rsvp-countdown-animated", !REDUCED_MOTION);
    renderWord(null);

    function tickDown() {
      if (counter <= 0) {
        show(overlay, false);
        startPlaying();
        return;
      }
      overlay.textContent = String(counter);
      announce(String(counter));
      counter -= 1;
      state.countdownTimer = setTimeout(tickDown, 600);
    }
    tickDown();
  }

  function play() {
    if (!state.words.length) {
      return;
    }
    if (state.finished || state.index >= state.words.length) {
      state.index = 0;
      state.finished = false;
      state.sessionPosted = false;
    }
    if (state.playing) {
      return;
    }
    countdownThenPlay();
  }

  function pause() {
    if (!state.playing) {
      return;
    }
    clearTimers();
    state.elapsedMs += performance.now() - state.playStartedAt;
    state.playStartedAt = 0;
    state.playing = false;
    updatePlayButton();
    updateStats();
    announce("Paused at word " + (state.index + 1));
  }

  function togglePlay() {
    if (state.playing) {
      pause();
    } else {
      play();
    }
  }

  function seek(index, options) {
    var settings = options || {};
    var target = clamp(Math.round(index), 0, Math.max(0, state.words.length - 1));
    var wasPlaying = state.playing;
    if (wasPlaying) {
      pause();
    }
    state.index = target;
    state.finished = false;
    renderWord(state.words[target]);
    updateStats();
    updatePlayButton();
    if (wasPlaying && settings.resume !== false) {
      // Resume without a countdown: the eye is already parked.
      startPlaying();
    }
  }

  function finish() {
    clearTimers();
    if (state.playing) {
      state.elapsedMs += performance.now() - state.playStartedAt;
    }
    state.playing = false;
    state.playStartedAt = 0;
    state.finished = true;
    state.index = state.words.length;
    renderWord(null);
    updateStats();
    updatePlayButton();
    announce("Finished. Comprehension check ready.");
    openQuiz();
  }

  // -------------------------------------------------------------------
  // Sentence navigation
  //
  // The "wait, what?" control. Every RSVP tool ships without it and every
  // one of them is unusable for it: the moment you lose a sentence there
  // is no way back, so you either restart or give up, and most people
  // give up. A single key that drops you at the start of the sentence you
  // just missed is the difference between a demo and a tool.
  // -------------------------------------------------------------------

  function sentenceStartAt(index) {
    var j = Math.min(index, state.words.length - 1) - 1;
    while (j >= 0 && !state.words[j].is_sentence_end) {
      j -= 1;
    }
    return j + 1;
  }

  function backOneSentence() {
    if (!state.words.length) {
      return;
    }
    var start = sentenceStartAt(state.index);
    // Within the first couple of words of a sentence, "back" means the
    // sentence before this one -- otherwise the button feels dead.
    if (state.index - start <= 2 && start > 0) {
      start = sentenceStartAt(start - 1);
    }
    seek(start);
    announce("Back to the start of the sentence.");
  }

  function forwardOneSentence() {
    if (!state.words.length) {
      return;
    }
    var j = state.index;
    while (j < state.words.length && !state.words[j].is_sentence_end) {
      j += 1;
    }
    seek(Math.min(j + 1, state.words.length - 1));
    announce("Forward one sentence.");
  }

  function restart() {
    clearTimers();
    state.playing = false;
    state.playStartedAt = 0;
    state.elapsedMs = 0;
    state.index = 0;
    state.maxIndexSeen = 0;
    state.finished = false;
    state.sessionPosted = false;
    state.answers = [];
    closeQuiz();
    renderWord(state.words[0]);
    updateStats();
    updatePlayButton();
    setStatus("");
    announce("Restarted.");
  }

  // -------------------------------------------------------------------
  // Speed
  // -------------------------------------------------------------------

  function setWpm(value, options) {
    var settings = options || {};
    var next = clamp(Math.round(value / WPM_STEP) * WPM_STEP, WPM_MIN, WPM_MAX);
    if (next === state.wpm && !settings.force) {
      return;
    }
    state.wpm = next;
    state.schedule = buildSchedule(state.words, state.wpm);

    var slider = el("rsvp-speed");
    if (slider && document.activeElement !== slider) {
      slider.value = String(next);
    }
    text(el("rsvp-speed-value"), next + " wpm");
    try {
      window.localStorage.setItem(STORAGE_KEY, String(next));
    } catch (err) {
      /* private mode; the speed just will not persist */
    }
    updateStats();
    if (!settings.silent) {
      announce(next + " words per minute");
    }
  }

  function nudgeWpm(direction) {
    setWpm(state.wpm + direction * WPM_STEP);
  }

  // -------------------------------------------------------------------
  // Comprehension quiz
  // -------------------------------------------------------------------

  function loadQuestions() {
    if (state.questions) {
      return Promise.resolve(state.questions);
    }
    if (state.doc && Array.isArray(state.doc.questions) && state.doc.questions.length) {
      state.questions = state.doc.questions;
      return Promise.resolve(state.questions);
    }
    if (!state.doc) {
      return Promise.resolve([]);
    }
    return api("/api/reader/questions/" + encodeURIComponent(state.doc.id))
      .then(function (data) {
        state.questions = (data && data.questions) || [];
        return state.questions;
      })
      .catch(function () {
        state.questions = [];
        return state.questions;
      });
  }

  function openQuiz() {
    var panel = el("rsvp-quiz");
    if (!panel) {
      return;
    }
    show(panel, true);
    var body = el("rsvp-quiz-body");
    text(el("rsvp-quiz-result"), "");
    show(el("rsvp-quiz-result"), false);
    body.textContent = "Preparing the check.";

    loadQuestions().then(function (questions) {
      if (!questions.length) {
        renderSelfReport(body);
        return;
      }
      renderQuestions(body, questions);
    });
  }

  function closeQuiz() {
    show(el("rsvp-quiz"), false);
  }

  function renderQuestions(body, questions) {
    body.textContent = "";
    state.answers = new Array(questions.length).fill(-1);

    questions.forEach(function (question, qIndex) {
      var block = document.createElement("fieldset");
      block.className = "rsvp-q";

      var legend = document.createElement("legend");
      legend.className = "rsvp-q-prompt";
      legend.textContent = qIndex + 1 + ". " + question.question;
      block.appendChild(legend);

      var kind = document.createElement("span");
      kind.className = "rsvp-q-kind";
      kind.textContent = question.kind || "recall";
      block.appendChild(kind);

      (question.options || []).forEach(function (option, oIndex) {
        var id = "rsvp-q" + qIndex + "-o" + oIndex;
        var wrap = document.createElement("label");
        wrap.className = "rsvp-option";
        wrap.setAttribute("for", id);

        var input = document.createElement("input");
        input.type = "radio";
        input.name = "rsvp-q" + qIndex;
        input.id = id;
        input.value = String(oIndex);
        input.addEventListener("change", function () {
          state.answers[qIndex] = oIndex;
          var submit = el("rsvp-quiz-submit");
          if (submit) {
            submit.disabled = state.answers.indexOf(-1) !== -1;
          }
        });

        var span = document.createElement("span");
        span.textContent = option;

        wrap.appendChild(input);
        wrap.appendChild(span);
        block.appendChild(wrap);
      });

      body.appendChild(block);
    });

    var submit = el("rsvp-quiz-submit");
    if (submit) {
      submit.disabled = true;
      submit.hidden = false;
      submit.textContent = "Check my recall";
      submit.onclick = function () {
        gradeQuiz(questions);
      };
    }
  }

  function renderSelfReport(body) {
    // Fallback when the server has no generated questions for this
    // document. Self-report is a much weaker measure and it is labelled
    // as such rather than quietly folded in with the real scores.
    body.textContent = "";
    var note = document.createElement("p");
    note.className = "rsvp-muted";
    note.textContent =
      "No generated questions for this document, so this is your own estimate. " +
      "It moves the speed, but it is a weaker signal than a scored check.";
    body.appendChild(note);

    var choices = [
      ["Almost none of it", 0.2],
      ["About half", 0.5],
      ["Most of it", 0.75],
      ["All of it", 0.95]
    ];
    var group = document.createElement("div");
    group.className = "rsvp-selfreport";
    choices.forEach(function (choice) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-outline-secondary btn-sm";
      button.textContent = choice[0];
      button.addEventListener("click", function () {
        submitSession(choice[1], true);
      });
      group.appendChild(button);
    });
    body.appendChild(group);

    var submit = el("rsvp-quiz-submit");
    if (submit) {
      submit.hidden = true;
    }
  }

  function gradeQuiz(questions) {
    var correct = 0;
    questions.forEach(function (question, qIndex) {
      var chosen = state.answers[qIndex];
      var right = Number(question.answer_index);
      var block = document.querySelectorAll(".rsvp-q")[qIndex];
      if (chosen === right) {
        correct += 1;
      }
      if (!block) {
        return;
      }
      var options = block.querySelectorAll(".rsvp-option");
      options.forEach(function (option, oIndex) {
        option.classList.remove("rsvp-option-right", "rsvp-option-wrong");
        var input = option.querySelector("input");
        if (input) {
          input.disabled = true;
        }
        if (oIndex === right) {
          option.classList.add("rsvp-option-right");
        } else if (oIndex === chosen) {
          option.classList.add("rsvp-option-wrong");
        }
      });
      if (question.evidence) {
        var evidence = document.createElement("p");
        evidence.className = "rsvp-evidence";
        evidence.textContent = "From the text: " + question.evidence;
        block.appendChild(evidence);
      }
    });

    var score = questions.length ? correct / questions.length : 0;
    var submit = el("rsvp-quiz-submit");
    if (submit) {
      submit.disabled = true;
      submit.textContent = correct + " of " + questions.length + " correct";
    }
    submitSession(score, false);
  }

  function submitSession(comprehension, selfReported) {
    if (state.sessionPosted) {
      return;
    }
    var result = el("rsvp-quiz-result");
    show(result, true);
    result.textContent = "Saving.";

    if (!state.doc) {
      result.textContent = "No document loaded.";
      return;
    }
    state.sessionPosted = true;

    var wordsRead = Math.max(state.maxIndexSeen + 1, 0);
    var payload = {
      entry_id: state.doc.id,
      wpm: state.wpm,
      words_read: wordsRead,
      duration_ms: Math.round(currentElapsed()),
      comprehension: comprehension,
      completed: state.finished
    };

    api("/api/reader/session", {
      method: "POST",
      body: JSON.stringify(payload)
    })
      .then(function (data) {
        var percent = Math.round(comprehension * 100);
        var lines = [
          "You scored " +
            percent +
            "%" +
            (selfReported ? " by your own estimate" : "") +
            " on " +
            wordsRead +
            " words in " +
            formatClock(payload.duration_ms) +
            "."
        ];
        if (data && data.reason) {
          lines.push(data.reason);
        }
        if (data && data.next_wpm) {
          setWpm(data.next_wpm, { silent: true, force: true });
        }
        result.textContent = lines.join(" ");
        loadHistory();
      })
      .catch(function () {
        state.sessionPosted = false;
        result.textContent =
          "The session could not be saved. Your reading still counted, the record did not.";
      });
  }

  // -------------------------------------------------------------------
  // Documents
  // -------------------------------------------------------------------

  function loadDocuments() {
    var list = el("rsvp-doc-list");
    if (!list) {
      return;
    }
    list.textContent = "Loading your saved reading.";
    api("/api/reader/documents")
      .then(function (data) {
        state.documents = (data && data.items) || [];
        renderDocuments(state.documents);
      })
      .catch(function () {
        list.textContent = "Could not load your documents. Reload the page to try again.";
      });
  }

  function renderDocuments(items) {
    var list = el("rsvp-doc-list");
    list.textContent = "";
    if (!items.length) {
      var empty = document.createElement("p");
      empty.className = "rsvp-muted mb-0";
      empty.textContent =
        "Nothing saved to read yet. Summarize something and it shows up here.";
      list.appendChild(empty);
      return;
    }

    items.forEach(function (item) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "rsvp-doc";
      button.dataset.entryId = String(item.id);

      var title = document.createElement("span");
      title.className = "rsvp-doc-title";
      title.textContent = item.title || "Untitled";

      var meta = document.createElement("span");
      meta.className = "rsvp-doc-meta";
      var words = Number(item.word_count) || 0;
      var minutes = words ? Math.max(1, Math.round(words / state.wpm)) : 0;
      meta.textContent =
        (item.source_type || "text") +
        (words ? " · " + words.toLocaleString() + " words" : "") +
        (minutes ? " · about " + minutes + " min at " + state.wpm + " wpm" : "");

      button.appendChild(title);
      button.appendChild(meta);
      button.addEventListener("click", function () {
        loadDocument(item.id);
      });
      list.appendChild(button);
    });
  }

  function loadDocument(entryId) {
    setStatus("Loading document.", "info");
    clearTimers();
    api("/api/reader/document/" + encodeURIComponent(entryId))
      .then(function (data) {
        state.doc = data;
        state.words = (data && data.words) || [];
        state.questions = Array.isArray(data && data.questions) ? data.questions : null;
        state.schedule = buildSchedule(state.words, state.wpm);
        state.index = 0;
        state.maxIndexSeen = 0;
        state.elapsedMs = 0;
        state.finished = false;
        state.playing = false;
        state.sessionPosted = false;
        state.answers = [];

        text(el("rsvp-title"), data.title || "Untitled");
        text(
          el("rsvp-subtitle"),
          state.words.length.toLocaleString() +
            " words · about " +
            Math.max(1, Math.round(state.words.length / state.wpm)) +
            " min at " +
            state.wpm +
            " wpm"
        );
        show(el("rsvp-stage"), true);
        closeQuiz();
        renderWord(state.words[0]);
        updateStats();
        updatePlayButton();
        setStatus("");

        var list = el("rsvp-doc-list");
        if (list) {
          Array.prototype.forEach.call(list.children, function (child) {
            if (child.dataset) {
              child.classList.toggle(
                "rsvp-doc-active",
                child.dataset.entryId === String(entryId)
              );
            }
          });
        }
        var stage = el("rsvp-stage");
        if (stage && stage.scrollIntoView) {
          stage.scrollIntoView({
            behavior: REDUCED_MOTION ? "auto" : "smooth",
            block: "start"
          });
        }
      })
      .catch(function () {
        setStatus("That document could not be loaded.", "warn");
      });
  }

  // -------------------------------------------------------------------
  // History and the chart
  //
  // Drawn as inline SVG by hand. A chart library for one line of a few
  // dozen points would be more bytes than the entire reader.
  // -------------------------------------------------------------------

  function loadHistory() {
    var panel = el("rsvp-history-body");
    if (!panel) {
      return;
    }
    api("/api/reader/stats")
      .then(function (data) {
        renderHistory(data || {});
      })
      .catch(function () {
        panel.textContent = "History is unavailable right now.";
      });
  }

  function renderHistory(data) {
    var panel = el("rsvp-history-body");
    panel.textContent = "";

    var fitness = data.fitness || {};
    var series = fitness.series || [];

    var tiles = [
      ["Effective wpm", fitness.effective_wpm || 0],
      ["Best sustained", fitness.best_sustained_wpm || 0],
      ["Words read", (fitness.total_words || 0).toLocaleString()],
      ["Day streak", fitness.streak_days || 0]
    ];
    var row = document.createElement("div");
    row.className = "rsvp-tiles";
    tiles.forEach(function (tile) {
      var box = document.createElement("div");
      box.className = "rsvp-tile";
      var value = document.createElement("span");
      value.className = "rsvp-tile-value";
      value.textContent = String(tile[1]);
      var label = document.createElement("span");
      label.className = "rsvp-tile-label";
      label.textContent = tile[0];
      box.appendChild(value);
      box.appendChild(label);
      row.appendChild(box);
    });
    panel.appendChild(row);

    if (series.length >= 2) {
      panel.appendChild(buildChart(series));
    } else {
      var hint = document.createElement("p");
      hint.className = "rsvp-muted";
      hint.textContent =
        "The chart needs at least two scored sessions before it says anything.";
      panel.appendChild(hint);
    }

    if (fitness.assessment) {
      var assessment = document.createElement("p");
      assessment.className = "rsvp-assessment";
      assessment.textContent = fitness.assessment;
      panel.appendChild(assessment);
    }
  }

  function svg(name, attributes) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, String(attributes[key]));
    });
    return node;
  }

  function buildChart(series) {
    var width = 640;
    var height = 220;
    var padLeft = 46;
    var padRight = 14;
    var padTop = 14;
    var padBottom = 30;

    var values = series.map(function (point) {
      return Number(point.effective_wpm) || 0;
    });
    var rawValues = series.map(function (point) {
      return Number(point.wpm) || 0;
    });
    var maxValue = Math.max.apply(null, values.concat(rawValues));
    var top = Math.max(50, Math.ceil((maxValue * 1.15) / 50) * 50);

    var plotWidth = width - padLeft - padRight;
    var plotHeight = height - padTop - padBottom;

    function x(i) {
      return series.length === 1
        ? padLeft + plotWidth / 2
        : padLeft + (i / (series.length - 1)) * plotWidth;
    }
    function y(value) {
      return padTop + plotHeight - (value / top) * plotHeight;
    }

    var figure = document.createElement("figure");
    figure.className = "rsvp-chart";

    var caption = document.createElement("figcaption");
    caption.textContent =
      "Effective words per minute (speed multiplied by measured recall) across " +
      series.length +
      " sessions. The fainter line is raw speed.";
    figure.appendChild(caption);

    var chart = svg("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": caption.textContent,
      preserveAspectRatio: "xMidYMid meet"
    });

    // Gridlines and y axis labels.
    var steps = 4;
    for (var g = 0; g <= steps; g++) {
      var value = (top / steps) * g;
      var gy = y(value);
      chart.appendChild(
        svg("line", {
          x1: padLeft,
          y1: gy,
          x2: width - padRight,
          y2: gy,
          class: "rsvp-grid"
        })
      );
      var label = svg("text", {
        x: padLeft - 8,
        y: gy + 4,
        class: "rsvp-axis",
        "text-anchor": "end"
      });
      label.textContent = String(Math.round(value));
      chart.appendChild(label);
    }

    function pathFor(list) {
      return list
        .map(function (value, i) {
          return (i === 0 ? "M" : "L") + x(i).toFixed(1) + " " + y(value).toFixed(1);
        })
        .join(" ");
    }

    chart.appendChild(
      svg("path", { d: pathFor(rawValues), class: "rsvp-line rsvp-line-raw", fill: "none" })
    );
    chart.appendChild(
      svg("path", { d: pathFor(values), class: "rsvp-line rsvp-line-effective", fill: "none" })
    );

    series.forEach(function (point, i) {
      var dot = svg("circle", {
        cx: x(i).toFixed(1),
        cy: y(values[i]).toFixed(1),
        r: 3.5,
        class: "rsvp-dot"
      });
      var title = svg("title", {});
      title.textContent =
        (point.date || "session " + (i + 1)) +
        ": " +
        values[i] +
        " effective wpm (" +
        point.wpm +
        " wpm at " +
        Math.round((Number(point.comprehension) || 0) * 100) +
        "%)";
      dot.appendChild(title);
      chart.appendChild(dot);
    });

    // X axis: first and last labels only. More would overlap.
    var firstLabel = svg("text", {
      x: padLeft,
      y: height - 8,
      class: "rsvp-axis",
      "text-anchor": "start"
    });
    firstLabel.textContent = series[0].date || "first";
    chart.appendChild(firstLabel);

    var lastLabel = svg("text", {
      x: width - padRight,
      y: height - 8,
      class: "rsvp-axis",
      "text-anchor": "end"
    });
    lastLabel.textContent = series[series.length - 1].date || "latest";
    chart.appendChild(lastLabel);

    figure.appendChild(chart);
    return figure;
  }

  // -------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------

  function isTypingTarget(node) {
    if (!node) {
      return false;
    }
    var tag = node.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      node.isContentEditable === true
    );
  }

  function onKeyDown(event) {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (isTypingTarget(event.target)) {
      return;
    }
    if (!state.words.length) {
      return;
    }

    var handled = true;
    switch (event.key) {
      case " ":
      case "Spacebar":
        togglePlay();
        break;
      case "ArrowUp":
        nudgeWpm(1);
        break;
      case "ArrowDown":
        nudgeWpm(-1);
        break;
      case "ArrowLeft":
        backOneSentence();
        break;
      case "ArrowRight":
        forwardOneSentence();
        break;
      case "Home":
        seek(0, { resume: false });
        break;
      case "End":
        seek(state.words.length - 1, { resume: false });
        break;
      case "Escape":
        pause();
        break;
      case "r":
      case "R":
        restart();
        break;
      case "q":
      case "Q":
        openQuiz();
        break;
      default:
        handled = false;
    }
    if (handled) {
      event.preventDefault();
    }
  }

  function bind() {
    var play = el("rsvp-play");
    if (play) {
      play.addEventListener("click", togglePlay);
    }
    var back = el("rsvp-back");
    if (back) {
      back.addEventListener("click", backOneSentence);
    }
    var forward = el("rsvp-forward");
    if (forward) {
      forward.addEventListener("click", forwardOneSentence);
    }
    var again = el("rsvp-restart");
    if (again) {
      again.addEventListener("click", restart);
    }
    var slower = el("rsvp-slower");
    if (slower) {
      slower.addEventListener("click", function () {
        nudgeWpm(-1);
      });
    }
    var faster = el("rsvp-faster");
    if (faster) {
      faster.addEventListener("click", function () {
        nudgeWpm(1);
      });
    }
    var quiz = el("rsvp-quiz-open");
    if (quiz) {
      quiz.addEventListener("click", function () {
        pause();
        openQuiz();
      });
    }
    var closeButton = el("rsvp-quiz-close");
    if (closeButton) {
      closeButton.addEventListener("click", closeQuiz);
    }

    var speed = el("rsvp-speed");
    if (speed) {
      speed.min = String(WPM_MIN);
      speed.max = String(WPM_MAX);
      speed.step = String(WPM_STEP);
      speed.addEventListener("input", function () {
        setWpm(Number(speed.value), { silent: true });
      });
      speed.addEventListener("change", function () {
        setWpm(Number(speed.value));
      });
    }

    var scrub = el("rsvp-scrub");
    if (scrub) {
      scrub.addEventListener("input", function () {
        if (!state.words.length) {
          return;
        }
        var target = (Number(scrub.value) / 100) * (state.words.length - 1);
        seek(target, { resume: false });
      });
    }

    document.addEventListener("keydown", onKeyDown);

    window.addEventListener("beforeunload", clearTimers);
    document.addEventListener("visibilitychange", function () {
      // Words flashing at a tab nobody is looking at are words nobody
      // read, and they would inflate the session record.
      if (document.hidden && state.playing) {
        pause();
        setStatus("Paused when you switched away.", "info");
      }
    });
  }

  // -------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------

  function boot() {
    var stored = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      stored = null;
    }
    var initial = Number(root.dataset.initialWpm) || 300;
    if (stored && Number(stored)) {
      initial = Number(stored);
    }
    setWpm(initial, { silent: true, force: true });

    bind();
    loadDocuments();
    loadHistory();
    updatePlayButton();

    // Keep the clock honest while playing without re-rendering words.
    window.setInterval(function () {
      if (state.playing) {
        updateStats();
      }
    }, 500);

    var params = new URLSearchParams(window.location.search);
    var entry = params.get("entry");
    if (entry) {
      loadDocument(entry);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
