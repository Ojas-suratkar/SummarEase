// Collaborative reading rooms -- the browser half.
//
// TRANSPORT: Server-Sent Events, not WebSockets. Almost everything here
// is one-directional (someone highlights a sentence; twelve screens need
// to learn about it), and the small amount going the other way is happy
// as an ordinary POST. EventSource is built into the browser, works over
// plain HTTP through proxies that mangle the WebSocket upgrade, and
// needs nothing added to the server but a generator. The cost is one
// held connection per viewer, which is why the server caps stream
// lifetime and why everything below is written to survive a stream
// ending at any moment without the user noticing.
//
// RELIABILITY MODEL: every event carries a monotonic id. We remember the
// highest id we have processed. Every (re)connect sends it back as
// `last_event_id`, and the server replays exactly what we missed before
// resuming live delivery. So a dropped connection is not a lost
// highlight -- it is a pause. Reconnects use exponential backoff with
// jitter so a server restart does not get a thundering herd of every
// participant retrying on the same tick. If SSE fails repeatedly (a
// proxy that buffers event streams into uselessness is a real thing), we
// degrade to polling /state and say so in the status pill rather than
// pretending to be live.
(function () {
  "use strict";

  var root = document.getElementById("se-room-root");
  if (!root) return;

  // ---------------------------------------------------------------- config

  var ROOM_ID = root.dataset.roomId;
  var API = "/api/rooms/" + encodeURIComponent(ROOM_ID);
  var DISPLAY_NAME = root.dataset.displayName || "Guest";

  var CURSOR_INTERVAL_MS = 50;      // ~20 updates/sec, the spec'd budget
  var CURSOR_STALE_MS = 15000;      // hide a pointer we stopped hearing about
  var BACKOFF_BASE_MS = 750;
  var BACKOFF_MAX_MS = 20000;
  var SSE_FAILURES_BEFORE_POLLING = 5;
  var POLL_INTERVAL_MS = 4000;
  var SSE_RETRY_AFTER_POLLING_MS = 45000;

  var EVENT_TYPES = [
    "presence.join",
    "presence.leave",
    "cursor.move",
    "highlight.add",
    "highlight.remove",
    "comment.add",
    "comment.remove",
    "reaction.add",
    "state.sync"
  ];

  // ---------------------------------------------------------------- state

  var participantId = null;
  var lastEventId = 0;
  var source = null;
  var failures = 0;
  var reconnectTimer = null;
  var pollTimer = null;
  var retrySseTimer = null;
  var mode = "idle"; // idle | live | reconnecting | polling
  var closing = false;

  var people = {};      // participant_id -> participant
  var highlights = {};  // highlight_id  -> highlight
  var comments = {};    // comment_id    -> comment
  var cursors = {};     // participant_id -> {x,y,tx,ty,name,hue,seen,el}
  var selectedHighlight = null;
  var pendingSelection = null;
  var pristineHTML = "";

  // ---------------------------------------------------------------- elements

  var docPane = document.getElementById("se-room-doc");
  var overlay = document.getElementById("se-room-overlay");
  var statusPill = document.getElementById("se-room-status");
  var peopleList = document.getElementById("se-room-people");
  var threadList = document.getElementById("se-room-thread");
  var threadTitle = document.getElementById("se-room-thread-title");
  var commentForm = document.getElementById("se-room-comment-form");
  var commentInput = document.getElementById("se-room-comment-input");
  var commentContext = document.getElementById("se-room-comment-context");
  var clearAnchorBtn = document.getElementById("se-room-clear-anchor");
  var selectionBar = document.getElementById("se-room-selection-bar");
  var highlightBtn = document.getElementById("se-room-highlight-btn");
  var copyBtn = document.getElementById("se-room-copy-code");
  var noticeBox = document.getElementById("se-room-notice");
  var reactionBar = document.getElementById("se-room-reactions");

  // ---------------------------------------------------------------- helpers

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clamp(n, lo, hi) {
    return Math.max(lo, Math.min(hi, n));
  }

  // Same FNV-1a the server uses, so a person is the same colour on every
  // screen -- including before the presence list has arrived.
  function hueFor(id) {
    var h = 0x811c9dc5;
    for (var i = 0; i < id.length; i++) {
      h ^= id.charCodeAt(i) & 0xff;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h % 360;
  }

  function initials(name) {
    var parts = String(name || "").split(/\s+/).filter(Boolean);
    if (!parts.length) return "?";
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }

  function notice(message) {
    if (!noticeBox) return;
    noticeBox.textContent = message || "";
    noticeBox.classList.toggle("d-none", !message);
  }

  function postJSON(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body || {})
    }).then(function (res) {
      if (!res.ok) throw new Error("Request failed (" + res.status + ")");
      return res.json();
    });
  }

  function send(type, payload) {
    if (!participantId) return Promise.resolve(null);
    return postJSON(API + "/event", {
      participant_id: participantId,
      type: type,
      payload: payload || {}
    }).catch(function (err) {
      // A failed POST is worth surfacing: the user just did something and
      // nobody else saw it. Silently swallowing it is how "my highlight
      // didn't show up" becomes an unreproducible bug report.
      notice("Couldn't send that just now. " + err.message);
      return null;
    });
  }

  // ---------------------------------------------------------------- status

  function setMode(next, detail) {
    mode = next;
    if (!statusPill) return;
    var label = {
      idle: "Connecting",
      live: "Connected",
      reconnecting: "Reconnecting",
      polling: "Offline - refreshing every few seconds"
    }[next] || next;
    statusPill.textContent = detail ? label + " - " + detail : label;
    statusPill.dataset.mode = next;
  }

  // ---------------------------------------------------------------- document text index
  //
  // HIGHLIGHT SERIALISATION
  // A highlight travels as two integers: character offsets into the
  // concatenated text of the document container. Not XPath, not element
  // ids, not a CSS selector path -- those describe the *shape* of one
  // person's DOM, and that shape differs between screens the moment
  // anything touches it, including our own highlight spans. Character
  // offsets describe the *text*, which is identical for everyone because
  // it came from the same server-rendered markup, and stays identical no
  // matter how many spans we wrap around it (wrapping never changes
  // textContent). So offsets computed on one screen land on exactly the
  // same words on every other.

  function buildIndex(container) {
    var nodes = [];
    var total = 0;
    var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
    var node;
    while ((node = walker.nextNode())) {
      var len = node.nodeValue.length;
      if (!len) continue;
      nodes.push({ node: node, start: total, end: total + len });
      total += len;
    }
    return { nodes: nodes, length: total };
  }

  function offsetOf(index, node, offset) {
    // Text node: direct lookup. Element node: the offset is a child
    // index, so resolve it to the text position where that child begins.
    if (node.nodeType === Node.TEXT_NODE) {
      for (var i = 0; i < index.nodes.length; i++) {
        if (index.nodes[i].node === node) return index.nodes[i].start + offset;
      }
      return null;
    }
    var child = node.childNodes[offset] || null;
    if (!child) {
      // Past the last child: the end of this element's text.
      var last = null;
      for (var j = 0; j < index.nodes.length; j++) {
        if (node.contains(index.nodes[j].node)) last = index.nodes[j];
      }
      return last ? last.end : null;
    }
    for (var k = 0; k < index.nodes.length; k++) {
      if (child === index.nodes[k].node || child.contains(index.nodes[k].node)) {
        return index.nodes[k].start;
      }
    }
    return null;
  }

  function serializeSelection() {
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
    var range = sel.getRangeAt(0);
    if (!docPane.contains(range.commonAncestorContainer)) return null;
    var index = buildIndex(docPane);
    var start = offsetOf(index, range.startContainer, range.startOffset);
    var end = offsetOf(index, range.endContainer, range.endOffset);
    if (start === null || end === null || start === end) return null;
    if (start > end) { var t = start; start = end; end = t; }
    var text = range.toString().replace(/\s+/g, " ").trim();
    if (!text) return null;
    return { start: start, end: end, text: text };
  }

  // ---------------------------------------------------------------- highlight rendering
  //
  // Overlaps are handled by *segmentation*, not by nesting. We take every
  // highlight boundary, cut the document into runs between consecutive
  // boundaries, and give each run exactly one span that knows which
  // highlights cover it. Nesting spans inside spans is what shreds the
  // DOM when two people highlight crossing ranges; one flat span per run
  // cannot. And because we rebuild from the pristine markup each time, a
  // removed highlight leaves nothing behind.

  function renderHighlights() {
    if (!docPane) return;
    var scroll = docPane.scrollTop;
    docPane.innerHTML = pristineHTML;
    var index = buildIndex(docPane);
    var list = Object.keys(highlights).map(function (k) { return highlights[k]; })
      .filter(function (h) {
        return typeof h.start === "number" && typeof h.end === "number" &&
          h.end > h.start && h.start < index.length;
      });

    if (list.length) {
      var segments = computeSegments(list, index.length);
      // Back to front: wrapping splits text nodes, and splitting only
      // ever affects offsets *after* the split point. Walking backwards
      // therefore keeps every not-yet-applied offset valid without
      // rebuilding the index between segments.
      for (var si = segments.length - 1; si >= 0; si--) {
        wrapSegment(index, segments[si]);
      }
    }

    docPane.scrollTop = scroll;
    bindHighlightClicks();
  }

  // Cut [0, length) at every highlight boundary and keep the runs that at
  // least one highlight covers. Each run is disjoint from every other, so
  // overlapping highlights produce adjacent flat spans instead of nested
  // ones -- which is the whole reason two people can highlight crossing
  // ranges without the document turning to soup.
  function computeSegments(list, length) {
    var points = {};
    list.forEach(function (h) {
      points[clamp(h.start, 0, length)] = true;
      points[clamp(h.end, 0, length)] = true;
    });
    var bounds = Object.keys(points).map(Number).sort(function (a, b) { return a - b; });
    var segments = [];
    for (var i = 0; i < bounds.length - 1; i++) {
      var s = bounds[i], e = bounds[i + 1];
      if (e <= s) continue;
      var covering = list.filter(function (h) { return h.start <= s && h.end >= e; });
      if (covering.length) segments.push({ start: s, end: e, covering: covering });
    }
    return segments;
  }

  function wrapSegment(index, seg) {
    for (var i = index.nodes.length - 1; i >= 0; i--) {
      var entry = index.nodes[i];
      if (entry.end <= seg.start || entry.start >= seg.end) continue;
      var node = entry.node;
      if (!node.parentNode) continue;
      var localStart = Math.max(0, seg.start - entry.start);
      var localEnd = Math.min(node.nodeValue.length, seg.end - entry.start);
      if (localEnd <= localStart) continue;

      if (localEnd < node.nodeValue.length) node.splitText(localEnd);
      var middle = localStart > 0 ? node.splitText(localStart) : node;

      var span = el("span", "se-room-hl");
      var top = seg.covering[seg.covering.length - 1];
      var hue = typeof top.hue === "number" ? top.hue : hueFor(top.participant_id || top.highlight_id);
      span.style.background = "hsla(" + hue + ", 90%, 62%, 0.28)";
      span.style.borderBottom = "2px solid hsla(" + hue + ", 75%, 45%, 0.75)";
      if (seg.covering.length > 1) {
        span.classList.add("is-overlap");
        var second = seg.covering[seg.covering.length - 2];
        var hue2 = typeof second.hue === "number" ? second.hue : hueFor(second.participant_id || "x");
        span.style.boxShadow = "inset 0 -6px 0 hsla(" + hue2 + ", 90%, 62%, 0.22)";
      }
      span.dataset.highlightIds = seg.covering.map(function (h) { return h.highlight_id; }).join(",");
      span.title = seg.covering.map(function (h) { return h.display_name || "Someone"; }).join(", ");

      middle.parentNode.insertBefore(span, middle);
      span.appendChild(middle);
    }
  }

  function bindHighlightClicks() {
    docPane.querySelectorAll(".se-room-hl").forEach(function (span) {
      span.addEventListener("click", function (ev) {
        ev.stopPropagation();
        var ids = (span.dataset.highlightIds || "").split(",").filter(Boolean);
        if (!ids.length) return;
        selectHighlight(ids[ids.length - 1]);
      });
    });
  }

  function selectHighlight(id) {
    selectedHighlight = highlights[id] ? id : null;
    renderThread();
    docPane.querySelectorAll(".se-room-hl").forEach(function (span) {
      var ids = (span.dataset.highlightIds || "").split(",");
      span.classList.toggle("is-active", !!selectedHighlight && ids.indexOf(selectedHighlight) !== -1);
    });
  }

  // ---------------------------------------------------------------- presence

  function renderPeople() {
    if (!peopleList) return;
    peopleList.innerHTML = "";
    var list = Object.keys(people).map(function (k) { return people[k]; });
    list.sort(function (a, b) { return (a.joined_at || 0) - (b.joined_at || 0); });
    if (!list.length) {
      peopleList.appendChild(el("p", "se-room-empty", "Nobody here yet."));
      return;
    }
    list.forEach(function (p) {
      var hue = typeof p.hue === "number" ? p.hue : hueFor(p.participant_id);
      var row = el("div", "se-room-person");
      var dot = el("span", "se-room-avatar", p.initials || initials(p.display_name));
      dot.style.background = "hsl(" + hue + ", 68%, 52%)";
      row.appendChild(dot);
      var name = el("span", "se-room-person-name", p.display_name || "Guest");
      if (p.participant_id === participantId) {
        name.textContent = (p.display_name || "Guest") + " (you)";
      }
      row.appendChild(name);
      peopleList.appendChild(row);
    });
  }

  // ---------------------------------------------------------------- live cursors
  //
  // Positions travel as fractions of the document pane, not pixels, so a
  // laptop and a 4K monitor point at the same sentence. We render towards
  // a target and ease into it on every frame: at 20 updates/sec raw
  // positions read as a twitch, and interpolation costs one lerp.

  function cursorFor(pid) {
    if (cursors[pid]) return cursors[pid];
    var person = people[pid] || {};
    var hue = typeof person.hue === "number" ? person.hue : hueFor(pid);
    var node = el("div", "se-room-cursor");
    node.style.setProperty("--se-cursor-hue", String(hue));
    var arrow = el("span", "se-room-cursor-arrow");
    arrow.textContent = "➤";
    node.appendChild(arrow);
    node.appendChild(el("span", "se-room-cursor-name", person.display_name || "Someone"));
    if (overlay) overlay.appendChild(node);
    cursors[pid] = { x: 0, y: 0, tx: 0, ty: 0, el: node, seen: 0, placed: false };
    return cursors[pid];
  }

  function moveCursor(pid, x, y) {
    if (pid === participantId) return;
    if (typeof x !== "number" || typeof y !== "number") return;
    var c = cursorFor(pid);
    c.tx = clamp(x, 0, 1);
    c.ty = clamp(y, 0, 1);
    c.seen = Date.now();
    if (!c.placed) { c.x = c.tx; c.y = c.ty; c.placed = true; }
    c.el.classList.remove("is-gone");
    var person = people[pid];
    if (person) {
      var label = c.el.querySelector(".se-room-cursor-name");
      if (label) label.textContent = person.display_name || "Someone";
    }
  }

  function dropCursor(pid) {
    var c = cursors[pid];
    if (!c) return;
    if (c.el && c.el.parentNode) c.el.parentNode.removeChild(c.el);
    delete cursors[pid];
  }

  function animate() {
    var now = Date.now();
    var width = overlay ? overlay.clientWidth : 0;
    var height = overlay ? overlay.clientHeight : 0;
    Object.keys(cursors).forEach(function (pid) {
      var c = cursors[pid];
      if (now - c.seen > CURSOR_STALE_MS) {
        c.el.classList.add("is-gone");
        return;
      }
      c.x += (c.tx - c.x) * 0.22;
      c.y += (c.ty - c.y) * 0.22;
      c.el.style.transform = "translate(" + (c.x * width).toFixed(1) + "px," + (c.y * height).toFixed(1) + "px)";
    });
    window.requestAnimationFrame(animate);
  }

  // ---------------------------------------------------------------- comments

  function renderThread() {
    if (!threadList) return;
    threadList.innerHTML = "";
    var all = Object.keys(comments).map(function (k) { return comments[k]; });
    var scoped = selectedHighlight
      ? all.filter(function (c) { return c.highlight_id === selectedHighlight; })
      : all;

    if (threadTitle) {
      threadTitle.textContent = selectedHighlight && highlights[selectedHighlight]
        ? "On: " + truncate(highlights[selectedHighlight].text, 70)
        : "All comments";
    }
    if (commentContext) {
      commentContext.textContent = selectedHighlight && highlights[selectedHighlight]
        ? "Replying on the highlighted passage."
        : "Not anchored to a highlight. Select a highlight to attach this.";
    }
    if (clearAnchorBtn) clearAnchorBtn.classList.toggle("d-none", !selectedHighlight);

    if (!scoped.length) {
      threadList.appendChild(el("p", "se-room-empty", "No comments yet."));
      return;
    }

    scoped.sort(function (a, b) { return (a.created_at || 0) - (b.created_at || 0); });
    var byParent = {};
    scoped.forEach(function (c) {
      var key = c.parent_id || "__root__";
      (byParent[key] = byParent[key] || []).push(c);
    });

    function renderLevel(parentKey, depth) {
      (byParent[parentKey] || []).forEach(function (c) {
        threadList.appendChild(commentNode(c, depth));
        if (depth < 4) renderLevel(c.comment_id, depth + 1);
      });
    }
    // Replies whose parent is filtered out still need a home, so anything
    // whose parent is not in view is treated as a root here.
    var visible = {};
    scoped.forEach(function (c) { visible[c.comment_id] = true; });
    Object.keys(byParent).forEach(function (key) {
      if (key !== "__root__" && !visible[key]) {
        byParent.__root__ = (byParent.__root__ || []).concat(byParent[key]);
        delete byParent[key];
      }
    });
    renderLevel("__root__", 0);
  }

  function commentNode(c, depth) {
    var hue = typeof c.hue === "number" ? c.hue : hueFor(c.participant_id || c.comment_id);
    var wrap = el("div", "se-room-comment");
    wrap.style.marginLeft = (depth * 14) + "px";
    var head = el("div", "se-room-comment-head");
    var dot = el("span", "se-room-avatar se-room-avatar-sm", initials(c.display_name));
    dot.style.background = "hsl(" + hue + ", 68%, 52%)";
    head.appendChild(dot);
    head.appendChild(el("span", "se-room-comment-author", c.display_name || "Someone"));
    head.appendChild(el("span", "se-room-comment-time", timeAgo(c.created_at)));
    wrap.appendChild(head);
    wrap.appendChild(el("div", "se-room-comment-body", c.body || ""));

    var actions = el("div", "se-room-comment-actions");
    var reply = el("button", "se-room-link-btn", "Reply");
    reply.type = "button";
    reply.addEventListener("click", function () {
      commentInput.dataset.parentId = c.comment_id;
      commentInput.placeholder = "Reply to " + (c.display_name || "them");
      commentInput.focus();
    });
    actions.appendChild(reply);
    if (c.participant_id === participantId) {
      var remove = el("button", "se-room-link-btn", "Delete");
      remove.type = "button";
      remove.addEventListener("click", function () {
        send("comment.remove", { comment_id: c.comment_id });
      });
      actions.appendChild(remove);
    }
    wrap.appendChild(actions);
    return wrap;
  }

  function truncate(text, n) {
    text = String(text || "");
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  }

  function timeAgo(ts) {
    if (!ts) return "";
    var secs = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (secs < 60) return "just now";
    if (secs < 3600) return Math.floor(secs / 60) + "m ago";
    return Math.floor(secs / 3600) + "h ago";
  }

  // ---------------------------------------------------------------- reactions

  function floatReaction(emoji, x, y, hue) {
    if (!overlay) return;
    var node = el("div", "se-room-reaction", emoji || "👍");
    node.style.left = (clamp(x || 0.5, 0, 1) * 100) + "%";
    node.style.top = (clamp(y || 0.5, 0, 1) * 100) + "%";
    node.style.setProperty("--se-reaction-hue", String(typeof hue === "number" ? hue : 250));
    overlay.appendChild(node);
    window.setTimeout(function () {
      if (node.parentNode) node.parentNode.removeChild(node);
    }, 2400);
  }

  // ---------------------------------------------------------------- event application

  function applyEvent(ev) {
    if (!ev || typeof ev.id !== "number") return;
    if (ev.id <= lastEventId && ev.type !== "state.sync") return;
    if (ev.id > lastEventId) lastEventId = ev.id;

    var p = ev.payload || {};
    switch (ev.type) {
      case "state.sync":
        applyState(p);
        return;
      case "presence.join":
        people[ev.participant_id] = {
          participant_id: ev.participant_id,
          display_name: ev.display_name || p.display_name || "Guest",
          initials: initials(ev.display_name || p.display_name),
          hue: typeof ev.hue === "number" ? ev.hue : hueFor(ev.participant_id),
          joined_at: ev.ts
        };
        renderPeople();
        return;
      case "presence.leave":
        delete people[ev.participant_id];
        dropCursor(ev.participant_id);
        renderPeople();
        return;
      case "cursor.move":
        moveCursor(ev.participant_id, p.x, p.y);
        return;
      case "highlight.add":
        highlights[p.highlight_id] = {
          highlight_id: p.highlight_id,
          start: p.start,
          end: p.end,
          text: p.text,
          participant_id: ev.participant_id,
          display_name: ev.display_name || "Someone",
          hue: typeof ev.hue === "number" ? ev.hue : hueFor(ev.participant_id || ""),
          created_at: ev.ts
        };
        renderHighlights();
        selectHighlight(selectedHighlight);
        return;
      case "highlight.remove":
        delete highlights[p.highlight_id];
        if (selectedHighlight === p.highlight_id) selectedHighlight = null;
        renderHighlights();
        renderThread();
        return;
      case "comment.add":
        comments[p.comment_id] = {
          comment_id: p.comment_id,
          highlight_id: p.highlight_id || null,
          parent_id: p.parent_id || null,
          body: p.body || "",
          participant_id: ev.participant_id,
          display_name: ev.display_name || "Someone",
          hue: typeof ev.hue === "number" ? ev.hue : hueFor(ev.participant_id || ""),
          created_at: ev.ts
        };
        renderThread();
        return;
      case "comment.remove":
        delete comments[p.comment_id];
        renderThread();
        return;
      case "reaction.add":
        floatReaction(p.emoji, p.x, p.y, typeof ev.hue === "number" ? ev.hue : undefined);
        return;
      default:
        return;
    }
  }

  function applyState(state) {
    if (!state) return;
    people = {};
    (state.participants || []).forEach(function (p) { people[p.participant_id] = p; });
    highlights = {};
    (state.highlights || []).forEach(function (h) { highlights[h.highlight_id] = h; });
    comments = {};
    (state.comments || []).forEach(function (c) { comments[c.comment_id] = c; });
    if (typeof state.last_event_id === "number" && state.last_event_id > lastEventId) {
      lastEventId = state.last_event_id;
    }
    Object.keys(cursors).forEach(function (pid) {
      if (!people[pid]) dropCursor(pid);
    });
    (state.participants || []).forEach(function (p) {
      if (p.cursor && typeof p.cursor.x === "number") moveCursor(p.participant_id, p.cursor.x, p.cursor.y);
    });
    renderPeople();
    renderHighlights();
    if (selectedHighlight && !highlights[selectedHighlight]) selectedHighlight = null;
    selectHighlight(selectedHighlight);
    renderThread();
  }

  // ---------------------------------------------------------------- transport

  function connect() {
    if (closing || !participantId) return;
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
    stopPolling();
    if (source) { try { source.close(); } catch (e) { /* already dead */ } }

    // last_event_id rides in the query string rather than relying on the
    // browser's Last-Event-ID header: we close and recreate the
    // EventSource ourselves (the built-in retry has no backoff and no
    // jitter), and a fresh EventSource does not resend that header.
    var url = API + "/stream?participant_id=" + encodeURIComponent(participantId) +
      "&last_event_id=" + encodeURIComponent(String(lastEventId));

    try {
      source = new EventSource(url, { withCredentials: true });
    } catch (err) {
      scheduleReconnect();
      return;
    }

    source.onopen = function () {
      failures = 0;
      // Cancel the "try SSE again" alarm left over from a polling spell,
      // otherwise it fires later and tears down a stream that is working.
      window.clearTimeout(retrySseTimer);
      retrySseTimer = null;
      setMode("live");
      notice("");
    };

    EVENT_TYPES.forEach(function (type) {
      source.addEventListener(type, function (msg) {
        try {
          applyEvent(JSON.parse(msg.data));
        } catch (err) { /* a truncated frame is not worth tearing down the room for */ }
      });
    });

    source.onmessage = function (msg) {
      try {
        applyEvent(JSON.parse(msg.data));
      } catch (err) { /* ignore */ }
    };

    source.onerror = function () {
      if (closing) return;
      try { source.close(); } catch (e) { /* ignore */ }
      source = null;
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (closing) return;
    failures += 1;
    if (failures >= SSE_FAILURES_BEFORE_POLLING) {
      startPolling();
      return;
    }
    // Exponential backoff with full jitter. Without the jitter, every
    // participant in every room retries on the same tick after a restart
    // and knocks the server over again the moment it comes back.
    var ceiling = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * Math.pow(2, failures - 1));
    var delay = Math.round(ceiling * (0.5 + Math.random() * 0.5));
    setMode("reconnecting", "retrying in " + Math.round(delay / 1000) + "s");
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, delay);
  }

  // Graceful degradation. Some proxies buffer text/event-stream until the
  // response completes, which makes SSE look connected and deliver
  // nothing. Polling is worse in every way except that it works, and the
  // status pill says plainly that this is what is happening.
  function startPolling() {
    if (pollTimer) return;
    setMode("polling");
    pollState();
    pollTimer = window.setInterval(pollState, POLL_INTERVAL_MS);
    window.clearTimeout(retrySseTimer);
    retrySseTimer = window.setTimeout(function () {
      failures = 0;
      connect();
    }, SSE_RETRY_AFTER_POLLING_MS);
  }

  function stopPolling() {
    if (pollTimer) { window.clearInterval(pollTimer); pollTimer = null; }
  }

  function pollState() {
    fetch(API + "/state", { credentials: "same-origin" })
      .then(function (res) {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then(function (state) {
        // Polling cannot see transient events (cursors, reactions) -- they
        // are gone by the time we ask. Presence, highlights and comments
        // all survive, which is the part that matters.
        applyState(state);
      })
      .catch(function () {
        setMode("polling", "no connection");
      });
  }

  // ---------------------------------------------------------------- input wiring

  function wireSelection() {
    if (!docPane) return;
    // selectionchange fires on every mouse move during a drag-select.
    // Serialising walks every text node in the document, so it is
    // debounced -- doing it per pointer sample makes a long document
    // feel sticky to select in, which is the one interaction this page
    // cannot afford to get wrong.
    var selectionTimer = null;
    document.addEventListener("selectionchange", function () {
      window.clearTimeout(selectionTimer);
      selectionTimer = window.setTimeout(function () {
        var payload = serializeSelection();
        pendingSelection = payload;
        if (selectionBar) selectionBar.classList.toggle("d-none", !payload);
      }, 90);
    });

    if (highlightBtn) {
      highlightBtn.addEventListener("click", function () {
        if (!pendingSelection) return;
        send("highlight.add", pendingSelection);
        var sel = window.getSelection();
        if (sel) sel.removeAllRanges();
        pendingSelection = null;
        if (selectionBar) selectionBar.classList.add("d-none");
      });
    }

    docPane.addEventListener("click", function (ev) {
      if (ev.target === docPane) selectHighlight(null);
    });
  }

  function wireCursor() {
    if (!docPane) return;
    var last = 0;
    var queued = null;
    docPane.addEventListener("pointermove", function (ev) {
      var rect = docPane.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      queued = {
        x: clamp((ev.clientX - rect.left) / rect.width, 0, 1),
        y: clamp((ev.clientY - rect.top) / rect.height, 0, 1)
      };
      var now = Date.now();
      // Throttle to ~20/sec. Pointer events fire far faster than that and
      // every one of them would be an HTTP request per participant.
      if (now - last < CURSOR_INTERVAL_MS) return;
      last = now;
      send("cursor.move", queued);
      queued = null;
    });
  }

  function wireComments() {
    if (!commentForm) return;
    commentForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var body = (commentInput.value || "").trim();
      if (!body) return;
      send("comment.add", {
        body: body,
        highlight_id: selectedHighlight || null,
        parent_id: commentInput.dataset.parentId || null
      });
      commentInput.value = "";
      delete commentInput.dataset.parentId;
      commentInput.placeholder = "Add a comment";
    });
    if (clearAnchorBtn) {
      clearAnchorBtn.addEventListener("click", function () { selectHighlight(null); });
    }
  }

  function wireReactions() {
    if (!reactionBar) return;
    reactionBar.querySelectorAll("[data-emoji]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var emoji = btn.dataset.emoji;
        send("reaction.add", {
          emoji: emoji,
          x: 0.2 + Math.random() * 0.6,
          y: 0.55 + Math.random() * 0.3
        });
      });
    });
  }

  function wireShare() {
    if (!copyBtn) return;
    copyBtn.addEventListener("click", function () {
      var value = copyBtn.dataset.copy || "";
      var done = function () {
        var original = copyBtn.textContent;
        copyBtn.textContent = "Copied";
        window.setTimeout(function () { copyBtn.textContent = original; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(value).then(done, function () {
          notice("Couldn't copy. The code is " + value + ".");
        });
      } else {
        notice("Copy this code to invite someone: " + value);
      }
    });
  }

  function wireUnload() {
    window.addEventListener("pagehide", function () {
      closing = true;
      if (source) { try { source.close(); } catch (e) { /* ignore */ } }
      if (!participantId) return;
      // sendBeacon survives the page going away, which a fetch does not.
      // Without it everyone else keeps a ghost avatar until prune() runs.
      var blob = new Blob([JSON.stringify({ participant_id: participantId })], {
        type: "application/json"
      });
      if (navigator.sendBeacon) navigator.sendBeacon(API + "/leave", blob);
    });
  }

  // ---------------------------------------------------------------- boot

  function boot() {
    pristineHTML = docPane ? docPane.innerHTML : "";
    setMode("idle");
    wireSelection();
    wireCursor();
    wireComments();
    wireReactions();
    wireShare();
    wireUnload();
    window.requestAnimationFrame(animate);

    postJSON(API + "/join", { display_name: DISPLAY_NAME })
      .then(function (data) {
        participantId = data.participant_id;
        root.dataset.participantId = participantId;
        (data.recent_events || []).forEach(applyEvent);
        // The room may predate anything still in the replay ring, so take
        // the authoritative view once before going live.
        return fetch(API + "/state", { credentials: "same-origin" }).then(function (r) {
          return r.ok ? r.json() : null;
        });
      })
      .then(function (state) {
        if (state) applyState(state);
        connect();
      })
      .catch(function (err) {
        setMode("polling", "couldn't join");
        notice("Couldn't join this room: " + err.message + " Reload to try again.");
      });
  }

  // Exposed so other pages can start a room without duplicating the call.
  window.SummarEaseRooms = {
    createRoom: function (entryId, title) {
      return postJSON("/api/rooms", { entry_id: entryId, title: title });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
