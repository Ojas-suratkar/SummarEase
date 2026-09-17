/* ---------------------------------------------------------------------------
 * SummarEase - infinite spatial canvas
 *
 * RENDERING APPROACH: a single <canvas> 2D context, not absolutely-positioned
 * DOM nodes. Reasons:
 *   1. The brief is 300+ cards with continuous pan/zoom. With DOM, every pan
 *      frame touches 300 style objects and the compositor re-rasterises text at
 *      each new scale; with canvas it is one transform and one draw loop we
 *      fully control.
 *   2. Culling is trivial and honest here - an off-screen card costs a single
 *      rectangle comparison and is never drawn. In DOM it still exists, still
 *      has layout, still participates in paint invalidation.
 *   3. Level-of-detail (dropping body text, then title, then everything but a
 *      coloured block) is a one-line branch in the draw call. In DOM it means
 *      mutating class names on hundreds of elements mid-gesture, which is
 *      exactly the layout thrash we are trying to avoid.
 *   4. Edges as curves, the marquee, the group regions and the minimap all want
 *      free-form drawing anyway.
 * The only DOM we keep inside the stage is a floating <input> for editing an
 * edge label or a group title - text editing is one thing the platform does far
 * better than we would.
 *
 * PERFORMANCE (what was actually done):
 *   - One requestAnimationFrame loop, driven by a dirty flag. If nothing
 *     changed, no frame is drawn at all. The layout animation and autosave both
 *     just raise the flag.
 *   - Viewport culling in world space before any per-card work.
 *   - Level-of-detail thresholds so small cards skip text measurement entirely,
 *     which is the dominant cost when zoomed out.
 *   - Wrapped text lines are cached per (card, width, detail level) and only
 *     recomputed when one of those changes, so measureText runs on edit, not on
 *     every frame.
 *   - Shadows (a full-canvas blur pass per card) are disabled below 0.4 zoom and
 *     above 200 visible cards.
 *   - The backing store is only resized when the element's pixel size genuinely
 *     changes, via ResizeObserver, never per frame.
 *   - Force-layout repulsion uses a uniform spatial grid with a 3k cutoff once
 *     there are more than 120 nodes, turning an O(n^2) pass into roughly O(n).
 *   - The minimap is a separate small canvas redrawn at most ~8 times a second.
 *   - Hit-testing walks the card list back-to-front and stops at the first hit.
 * ------------------------------------------------------------------------- */

(function () {
  "use strict";

  /* =====================================================================
   * ==== PURE:START ====
   * Everything between these markers is free of DOM and of module state, so
   * it can be lifted out verbatim and exercised by a Node test harness.
   * ===================================================================== */

  var MIN_ZOOM = 0.1;
  var MAX_ZOOM = 4;

  function clamp(value, lo, hi) {
    return value < lo ? lo : value > hi ? hi : value;
  }

  /* The view is stored as the world coordinate sitting at the top-left pixel of
   * the viewport, plus a scale. Keeping it in that form (rather than a pixel
   * offset) means the two conversions below are exact inverses with no
   * accumulated rounding, which matters because we round-trip constantly while
   * dragging. */
  function worldToScreen(view, wx, wy) {
    return { x: (wx - view.x) * view.zoom, y: (wy - view.y) * view.zoom };
  }

  function screenToWorld(view, sx, sy) {
    return { x: sx / view.zoom + view.x, y: sy / view.zoom + view.y };
  }

  /* Zoom anchored at the pointer. The classic bug is scaling around the origin
   * and then trying to correct with a pan derived from the OLD zoom. Solve it
   * properly instead: let w be the world point under the cursor before the
   * change, w = s/z0 + x0. We require the same screen pixel s to map to the same
   * w afterwards, so w = s/z1 + x1, hence x1 = w - s/z1. No approximation, and
   * it holds for any factor including clamped ones - which is why the clamp
   * happens BEFORE x1 is computed. */
  function zoomAt(view, sx, sy, factor) {
    var z1 = clamp(view.zoom * factor, MIN_ZOOM, MAX_ZOOM);
    var w = screenToWorld(view, sx, sy);
    return { x: w.x - sx / z1, y: w.y - sy / z1, zoom: z1 };
  }

  function normalizeRect(x0, y0, x1, y1) {
    return {
      x: Math.min(x0, x1),
      y: Math.min(y0, y1),
      w: Math.abs(x1 - x0),
      h: Math.abs(y1 - y0)
    };
  }

  function rectIntersects(a, b) {
    return !(
      a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y
    );
  }

  function pointInRect(px, py, r) {
    return px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h;
  }

  /* Marquee semantics: a card is selected if the lasso touches it at all.
   * Requiring full containment makes big cards nearly impossible to catch, and
   * every spatial tool people already know (Figma, Miro, Finder) uses touch. */
  function lassoSelect(nodes, rect) {
    var out = [];
    for (var i = 0; i < nodes.length; i++) {
      if (rectIntersects(nodes[i], rect)) out.push(nodes[i].entry_id);
    }
    return out;
  }

  function boundsOf(nodes) {
    if (!nodes || !nodes.length) return null;
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x + n.w > maxX) maxX = n.x + n.w;
      if (n.y + n.h > maxY) maxY = n.y + n.h;
    }
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
  }

  /* Fit-all: scale so the content bounds fit inside the viewport with padding,
   * then centre. Guard the empty case so pressing F on a blank canvas is a
   * no-op rather than a NaN view we can never recover from. */
  function fitView(nodes, viewW, viewH, pad) {
    pad = pad == null ? 80 : pad;
    var b = boundsOf(nodes);
    if (!b || viewW <= 0 || viewH <= 0) return null;
    var zoom = clamp(
      Math.min(
        (viewW - pad * 2) / Math.max(b.w, 1),
        (viewH - pad * 2) / Math.max(b.h, 1)
      ),
      MIN_ZOOM,
      MAX_ZOOM
    );
    return {
      x: b.x + b.w / 2 - viewW / (2 * zoom),
      y: b.y + b.h / 2 - viewH / (2 * zoom),
      zoom: zoom
    };
  }

  /* Level of detail. The numbers are chosen from the card's own type sizes:
   * body text is 12px in world units, so below 0.55 zoom it renders under ~6.6
   * device pixels - past the point of being read, but still the most expensive
   * thing on the frame. The title is 14px, unreadable below about 0.25. Under
   * that we draw a coloured block, which is still useful: shape, position and
   * colour are what spatial memory actually uses at that scale. */
  function lodLevel(zoom) {
    if (zoom < 0.25) return 0; // coloured block only
    if (zoom < 0.55) return 1; // title only, single line
    return 2; // title + summary excerpt
  }

  /* Where a line from an outside point meets a rectangle's border. Used to park
   * edge endpoints on the card edge instead of its centre, so an edge never
   * appears to sprout from underneath a card. */
  function rectBorderPoint(rect, towardX, towardY) {
    var cx = rect.x + rect.w / 2;
    var cy = rect.y + rect.h / 2;
    var dx = towardX - cx;
    var dy = towardY - cy;
    if (dx === 0 && dy === 0) return { x: cx, y: cy };
    var hw = rect.w / 2;
    var hh = rect.h / 2;
    /* Scale the direction vector until it first leaves the box on either axis.
     * The smaller of the two ratios is the side it exits through. */
    var tx = dx === 0 ? Infinity : hw / Math.abs(dx);
    var ty = dy === 0 ? Infinity : hh / Math.abs(dy);
    var t = Math.min(tx, ty);
    return { x: cx + dx * t, y: cy + dy * t };
  }

  /* A quadratic curve bulging perpendicular to the straight run. If the straight
   * midpoint lands inside some other card we push the bulge harder, trying both
   * sides and keeping the first that is clear - cheap, and enough to stop edges
   * tunnelling through cards in the common case. Full routing is not worth the
   * frame budget. */
  function edgeRoute(a, b, obstacles) {
    var p1 = rectBorderPoint(a, b.x + b.w / 2, b.y + b.h / 2);
    var p2 = rectBorderPoint(b, a.x + a.w / 2, a.y + a.h / 2);
    var mx = (p1.x + p2.x) / 2;
    var my = (p1.y + p2.y) / 2;
    var dx = p2.x - p1.x;
    var dy = p2.y - p1.y;
    var len = Math.sqrt(dx * dx + dy * dy) || 1;
    var nx = -dy / len;
    var ny = dx / len;
    var base = Math.min(40, len * 0.12);
    var offsets = [base, -base, base * 3, -base * 3, base * 6, -base * 6];
    var blocked = function (x, y) {
      if (!obstacles) return false;
      for (var i = 0; i < obstacles.length; i++) {
        var o = obstacles[i];
        if (o === a || o === b) continue;
        if (pointInRect(x, y, o)) return true;
      }
      return false;
    };
    for (var i = 0; i < offsets.length; i++) {
      var cx = mx + nx * offsets[i];
      var cy = my + ny * offsets[i];
      /* Sample the curve's own apex, which for a quadratic sits halfway between
       * the chord midpoint and the control point. */
      if (!blocked((mx + cx) / 2, (my + cy) / 2)) {
        return { x1: p1.x, y1: p1.y, cx: cx, cy: cy, x2: p2.x, y2: p2.y };
      }
    }
    return {
      x1: p1.x, y1: p1.y,
      cx: mx + nx * base, cy: my + ny * base,
      x2: p2.x, y2: p2.y
    };
  }

  function quadPoint(r, t) {
    var mt = 1 - t;
    return {
      x: mt * mt * r.x1 + 2 * mt * t * r.cx + t * t * r.x2,
      y: mt * mt * r.y1 + 2 * mt * t * r.cy + t * t * r.y2
    };
  }

  function distanceToCurve(r, px, py) {
    /* 16 samples is plenty for a click test on a gentle quadratic and costs
     * nothing next to a real subdivision solve. */
    var best = Infinity;
    for (var i = 0; i <= 16; i++) {
      var p = quadPoint(r, i / 16);
      var dx = p.x - px;
      var dy = p.y - py;
      var d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  /* Deterministic hash: group colours must be stable across reloads, and the
   * server contract only stores a group NAME on each node, so the colour has to
   * be a function of that name rather than extra persisted state. */
  function hashString(str) {
    var h = 2166136261;
    str = String(str == null ? "" : str);
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h >>> 0;
  }

  /* Deterministic jitter in [-0.5, 0.5], used to break perfectly coincident
   * nodes apart. Determinism matters: a layout that depends on Math.random is
   * untestable and irreproducible for the user. */
  function pseudoJitter(i, j) {
    var h = hashString(i + ":" + j);
    return (h % 100000) / 100000 - 0.5;
  }

  /* ---------------------------------------------------------------------
   * Fruchterman-Reingold.
   *
   * k is the ideal separation, sqrt(area / n) - the side of the square each
   * node would get if the frame were divided up evenly. Repulsion between every
   * pair is k^2/d, attraction along an edge is d^2/k; those two are equal at
   * d = k, which is what makes k the resting length.
   *
   * kScale 1.05 spreads slightly wider than the bare formula because our nodes
   * are real cards with size, not points, and cards at exactly k overlap.
   *
   * Cooling: the step a node may take is capped by a temperature that decays
   * 0.975 per iteration. Starting at maxDim/8 the first moves are large and
   * coarse, and after ~250 iterations the cap is under a pixel, so the layout
   * settles instead of jittering forever. We stop at temp < 0.6 or maxIter.
   *
   * gravity 0.006 is a gentle pull to the centroid. Pure F-R lets disconnected
   * components drift apart indefinitely, which on an infinite canvas means they
   * sail off and the user has to hunt for them. It is deliberately weak enough
   * not to distort a connected cluster's shape.
   *
   * Similarity pairs are treated as soft edges: an attraction weighted by the
   * similarity above a 0.35 floor, so documents about the same thing drift
   * together even when nobody drew a line.
   * ------------------------------------------------------------------- */
  var MIN_SEP2 = 1e-6;

  function createForceLayout(nodes, links, opts) {
    opts = opts || {};
    var n = nodes.length;
    var width = opts.width || 1600;
    var height = opts.height || 1000;
    var kScale = opts.kScale == null ? 1.05 : opts.kScale;
    var k = n > 0 ? kScale * Math.sqrt((width * height) / n) : 1;

    var layout = {
      nodes: nodes,
      links: links || [],
      k: k,
      cutoff: k * 3,
      temp: opts.temp0 == null ? Math.max(width, height) / 8 : opts.temp0,
      cooling: opts.cooling == null ? 0.975 : opts.cooling,
      gravity: opts.gravity == null ? 0.006 : opts.gravity,
      iter: 0,
      maxIter: opts.maxIter == null ? 400 : opts.maxIter,
      dispX: new Float64Array(Math.max(n, 1)),
      dispY: new Float64Array(Math.max(n, 1)),
      done: n <= 1,
      step: null
    };

    layout.step = function () {
      if (layout.done) return false;
      var i, j;
      var dx = layout.dispX;
      var dy = layout.dispY;
      for (i = 0; i < n; i++) { dx[i] = 0; dy[i] = 0; }

      var kk = layout.k * layout.k;

      /* Repulsion. Beyond ~3k the k^2/d term is under a ninth of the resting
       * force and contributes nothing visible, so past 120 nodes we bucket into
       * a uniform grid and only consult the 9 neighbouring cells. Below that the
       * exact all-pairs pass is cheaper than building the grid. */
      if (n > 120) {
        var cell = layout.cutoff;
        var buckets = new Map();
        var keys = new Array(n);
        for (i = 0; i < n; i++) {
          var gx = Math.floor(nodes[i].x / cell);
          var gy = Math.floor(nodes[i].y / cell);
          var key = gx + "," + gy;
          keys[i] = [gx, gy];
          var arr = buckets.get(key);
          if (!arr) { arr = []; buckets.set(key, arr); }
          arr.push(i);
        }
        for (i = 0; i < n; i++) {
          var cx0 = keys[i][0];
          var cy0 = keys[i][1];
          for (var ox = -1; ox <= 1; ox++) {
            for (var oy = -1; oy <= 1; oy++) {
              var near = buckets.get((cx0 + ox) + "," + (cy0 + oy));
              if (!near) continue;
              for (var m = 0; m < near.length; m++) {
                j = near[m];
                if (j <= i) continue;
                applyRepulsion(i, j);
              }
            }
          }
        }
      } else {
        for (i = 0; i < n; i++) {
          for (j = i + 1; j < n; j++) applyRepulsion(i, j);
        }
      }

      function applyRepulsion(a, b) {
        var vx = nodes[a].x - nodes[b].x;
        var vy = nodes[a].y - nodes[b].y;
        var d2 = vx * vx + vy * vy;
        if (d2 < MIN_SEP2) {
          /* Perfectly coincident nodes: the true force is undefined, so nudge
           * them apart deterministically rather than emitting Infinity. */
          vx = pseudoJitter(a, b) || 0.5;
          vy = pseudoJitter(b, a + 1) || -0.5;
          d2 = vx * vx + vy * vy;
          if (d2 < MIN_SEP2) { vx = 1; vy = 0; d2 = 1; }
        }
        var d = Math.sqrt(d2);
        var f = kk / d;
        var ux = (vx / d) * f;
        var uy = (vy / d) * f;
        dx[a] += ux; dy[a] += uy;
        dx[b] -= ux; dy[b] -= uy;
      }

      // Attraction along edges and similarity pairs.
      for (i = 0; i < layout.links.length; i++) {
        var L = layout.links[i];
        var a = L.a, b = L.b;
        if (a === b || a == null || b == null) continue;
        if (a < 0 || b < 0 || a >= n || b >= n) continue;
        var vx2 = nodes[a].x - nodes[b].x;
        var vy2 = nodes[a].y - nodes[b].y;
        var d22 = vx2 * vx2 + vy2 * vy2;
        if (d22 < MIN_SEP2) continue; // already together, nothing to pull
        var d3 = Math.sqrt(d22);
        var w = L.weight == null ? 1 : L.weight;
        var fa = ((d3 * d3) / layout.k) * w;
        var ax = (vx2 / d3) * fa;
        var ay = (vy2 / d3) * fa;
        dx[a] -= ax; dy[a] -= ay;
        dx[b] += ax; dy[b] += ay;
      }

      // Weak centring pull.
      var ccx = 0, ccy = 0;
      for (i = 0; i < n; i++) { ccx += nodes[i].x; ccy += nodes[i].y; }
      ccx /= n; ccy /= n;
      for (i = 0; i < n; i++) {
        dx[i] -= (nodes[i].x - ccx) * layout.gravity * layout.k;
        dy[i] -= (nodes[i].y - ccy) * layout.gravity * layout.k;
      }

      // Move, capped by temperature. Pinned nodes are anchors: others flow
      // around them, which is the whole point of pinning.
      for (i = 0; i < n; i++) {
        if (nodes[i].pinned) continue;
        var mag = Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
        if (!(mag > 0) || !isFinite(mag)) continue;
        var limited = Math.min(mag, layout.temp);
        var nx = nodes[i].x + (dx[i] / mag) * limited;
        var ny = nodes[i].y + (dy[i] / mag) * limited;
        if (isFinite(nx) && isFinite(ny)) { nodes[i].x = nx; nodes[i].y = ny; }
      }

      layout.temp *= layout.cooling;
      layout.iter++;
      if (layout.temp < 0.6 || layout.iter >= layout.maxIter) layout.done = true;
      return true;
    };

    return layout;
  }

  /* Bounded undo history. Both stacks are capped so a long session cannot grow
   * memory without limit; pushing a new action clears the redo branch, which is
   * what every editor does and what users expect. */
  function UndoStack(limit) {
    this.limit = Math.max(1, limit || 50);
    this.past = [];
    this.future = [];
  }
  UndoStack.prototype.push = function (snapshot) {
    this.past.push(snapshot);
    while (this.past.length > this.limit) this.past.shift();
    this.future.length = 0;
  };
  UndoStack.prototype.undo = function (current) {
    if (!this.past.length) return null;
    var s = this.past.pop();
    this.future.push(current);
    while (this.future.length > this.limit) this.future.shift();
    return s;
  };
  UndoStack.prototype.redo = function (current) {
    if (!this.future.length) return null;
    var s = this.future.pop();
    this.past.push(current);
    while (this.past.length > this.limit) this.past.shift();
    return s;
  };
  UndoStack.prototype.canUndo = function () { return this.past.length > 0; };
  UndoStack.prototype.canRedo = function () { return this.future.length > 0; };
  UndoStack.prototype.depth = function () { return this.past.length; };

  /* ==== PURE:END ==== */

  // =======================================================================
  // Application layer
  // =======================================================================

  var root = document.getElementById("se-canvas-root");
  if (!root) return; // canvas.js is inert on any other page

  var stage = root.querySelector("[data-canvas-stage]");
  var surface = root.querySelector("[data-canvas-surface]");
  var minimapEl = root.querySelector("[data-canvas-minimap]");
  if (!stage || !surface || !minimapEl) return;

  var ctx = surface.getContext("2d");
  var mmCtx = minimapEl.getContext("2d");

  var API = {
    layout: root.getAttribute("data-api-canvas") || "/api/canvas",
    items: root.getAttribute("data-api-items") || "/api/canvas/items"
  };

  var CARD_W = 236;
  var CARD_H = 132;
  var MIN_CARD_W = 140;
  var MIN_CARD_H = 84;
  var GRID_SNAP = 8;

  var PALETTE = [
    { key: "slate", label: "Slate", fill: "#ffffff", ink: "#1a1d29", edge: "#d7d9e8" },
    { key: "indigo", label: "Indigo", fill: "#eef0ff", ink: "#2b2a6b", edge: "#c3c6f5" },
    { key: "teal", label: "Teal", fill: "#e6f8fa", ink: "#0d5560", edge: "#a9e3ea" },
    { key: "amber", label: "Amber", fill: "#fdf3e3", ink: "#6b4708", edge: "#f0d6a4" },
    { key: "rose", label: "Rose", fill: "#fdeaf0", ink: "#6d1533", edge: "#f4bccd" },
    { key: "green", label: "Green", fill: "#e9f7ee", ink: "#14532d", edge: "#b3e2c4" },
    { key: "violet", label: "Violet", fill: "#f4ecfd", ink: "#4a1d78", edge: "#dcc6f4" }
  ];
  var PALETTE_BY_KEY = {};
  PALETTE.forEach(function (p) { PALETTE_BY_KEY[p.key] = p; });

  var GROUP_TINTS = [
    "#4f46e5", "#06b6d4", "#16a34a", "#d97706", "#db2777", "#7c3aed", "#0891b2"
  ];

  function colorFor(key) {
    return PALETTE_BY_KEY[key] || PALETTE_BY_KEY.slate;
  }
  function groupTint(name) {
    return GROUP_TINTS[hashString(name) % GROUP_TINTS.length];
  }

  // ---- state -----------------------------------------------------------

  var state = {
    nodes: [],            // {entry_id,x,y,w,h,color,group,pinned}
    edges: [],            // {from_id,to_id,label}
    view: { x: -400, y: -300, zoom: 1 },
    items: {},            // entry_id -> item record
    itemOrder: [],
    selection: new Set(), // entry_ids
    selectedEdge: -1,
    hoverId: null,
    loaded: false
  };

  var nodeById = new Map();
  function reindex() {
    nodeById.clear();
    for (var i = 0; i < state.nodes.length; i++) {
      nodeById.set(state.nodes[i].entry_id, state.nodes[i]);
    }
  }

  var undo = new UndoStack(60); // brief asks for 50; a little headroom is free
  var dirty = true;
  var minimapDirty = true;
  var lastMinimapDraw = 0;
  var layoutRun = null;
  var textCache = new Map();
  var visible = [];         // cards surviving the cull this frame
  var dpr = 1;
  var stageW = 0;
  var stageH = 0;

  function markDirty() { dirty = true; minimapDirty = true; }

  // ---- DOM handles -----------------------------------------------------

  function el(sel) { return root.querySelector(sel); }
  var saveIndicator = el("[data-save-indicator]");
  var emptyState = el("[data-empty-state]");
  var libraryPanel = el("[data-library]");
  var libraryList = el("[data-library-list]");
  var librarySearch = el("[data-library-search]");
  var libraryCount = el("[data-library-count]");
  var helpPanel = el("[data-help]");
  var zoomLabel = el("[data-zoom-label]");
  var statusLine = el("[data-status-line]");
  var layoutBtn = el("[data-act='layout']");
  var floatInput = el("[data-float-input]");

  // ---- snapshots / undo ------------------------------------------------

  function snapshot() {
    return JSON.stringify({
      nodes: state.nodes.map(function (n) {
        return {
          entry_id: n.entry_id, x: n.x, y: n.y, w: n.w, h: n.h,
          color: n.color, group: n.group, pinned: n.pinned
        };
      }),
      edges: state.edges.map(function (e) {
        return { from_id: e.from_id, to_id: e.to_id, label: e.label };
      })
    });
  }

  function restore(json) {
    var data = JSON.parse(json);
    state.nodes = data.nodes;
    state.edges = data.edges;
    reindex();
    textCache.clear();
    // Drop selections that no longer exist.
    var next = new Set();
    state.selection.forEach(function (id) { if (nodeById.has(id)) next.add(id); });
    state.selection = next;
    state.selectedEdge = -1;
    refreshChrome();
    markDirty();
  }

  var pendingCommit = null;
  function beginChange() {
    // Capture once per gesture, not once per mousemove.
    if (pendingCommit === null) pendingCommit = snapshot();
  }
  function commitChange() {
    if (pendingCommit === null) return;
    var before = pendingCommit;
    pendingCommit = null;
    if (before !== snapshot()) {
      undo.push(before);
      scheduleSave();
      refreshChrome();
    }
  }
  function changeNow(fn) {
    beginChange();
    fn();
    commitChange();
    markDirty();
  }

  function doUndo() {
    var current = snapshot();
    var prev = undo.undo(current);
    if (prev === null) { setStatus("Nothing to undo"); return; }
    restore(prev);
    scheduleSave();
    setStatus("Undone");
  }
  function doRedo() {
    var current = snapshot();
    var next = undo.redo(current);
    if (next === null) { setStatus("Nothing to redo"); return; }
    restore(next);
    scheduleSave();
    setStatus("Redone");
  }

  // ---- persistence -----------------------------------------------------

  var saveTimer = null;
  var saveInFlight = false;
  var saveQueued = false;

  function setSaveState(text, tone) {
    if (!saveIndicator) return;
    saveIndicator.textContent = text;
    saveIndicator.setAttribute("data-tone", tone || "idle");
  }

  function scheduleSave() {
    setSaveState("Unsaved", "pending");
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(doSave, 1500);
  }

  function doSave() {
    saveTimer = null;
    if (saveInFlight) { saveQueued = true; return; }
    saveInFlight = true;
    setSaveState("Saving", "pending");
    fetch(API.layout, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes: state.nodes.map(function (n) {
          return {
            entry_id: n.entry_id,
            x: round2(n.x), y: round2(n.y), w: round2(n.w), h: round2(n.h),
            color: n.color || "slate", group: n.group || "", pinned: !!n.pinned
          };
        }),
        edges: state.edges.map(function (e) {
          return { from_id: e.from_id, to_id: e.to_id, label: e.label || "" };
        }),
        view: { x: round2(state.view.x), y: round2(state.view.y), zoom: round4(state.view.zoom) }
      })
    })
      .then(function (r) {
        if (!r.ok) throw new Error("save failed");
        return r.json();
      })
      .then(function () { setSaveState("Saved", "ok"); })
      .catch(function () { setSaveState("Save failed, retrying", "warn"); saveQueued = true; })
      .then(function () {
        saveInFlight = false;
        if (saveQueued) { saveQueued = false; saveTimer = setTimeout(doSave, 2500); }
      });
  }

  function round2(v) { return Math.round(v * 100) / 100; }
  function round4(v) { return Math.round(v * 10000) / 10000; }

  // ---- loading ---------------------------------------------------------

  function load() {
    setStatus("Loading");
    Promise.all([
      fetch(API.layout, { credentials: "same-origin" }).then(okJson).catch(function () {
        return { nodes: [], edges: [], view: null };
      }),
      fetch(API.items, { credentials: "same-origin" }).then(okJson).catch(function () {
        return { items: [] };
      })
    ]).then(function (res) {
      var layout = res[0] || {};
      var itemsRes = res[1] || {};
      var items = Array.isArray(itemsRes.items) ? itemsRes.items : [];

      state.items = {};
      state.itemOrder = [];
      items.forEach(function (it) {
        if (!it || it.id == null) return;
        state.items[it.id] = it;
        state.itemOrder.push(it.id);
      });

      var rawNodes = Array.isArray(layout.nodes) ? layout.nodes : [];
      state.nodes = rawNodes
        .filter(function (n) { return n && n.entry_id != null; })
        .map(function (n) {
          return {
            entry_id: n.entry_id,
            x: num(n.x, 0), y: num(n.y, 0),
            w: Math.max(MIN_CARD_W, num(n.w, CARD_W)),
            h: Math.max(MIN_CARD_H, num(n.h, CARD_H)),
            color: n.color || "slate",
            group: n.group || "",
            pinned: !!n.pinned
          };
        });
      // Drop layout rows whose document has since been deleted.
      state.nodes = state.nodes.filter(function (n) { return state.items[n.entry_id]; });
      reindex();

      state.edges = (Array.isArray(layout.edges) ? layout.edges : [])
        .filter(function (e) {
          return e && nodeById.has(e.from_id) && nodeById.has(e.to_id) && e.from_id !== e.to_id;
        })
        .map(function (e) {
          return { from_id: e.from_id, to_id: e.to_id, label: e.label || "" };
        });

      if (layout.view && isFinite(layout.view.zoom) && layout.view.zoom > 0) {
        state.view = {
          x: num(layout.view.x, 0),
          y: num(layout.view.y, 0),
          zoom: clamp(num(layout.view.zoom, 1), MIN_ZOOM, MAX_ZOOM)
        };
      } else if (state.nodes.length) {
        var fv = fitView(state.nodes, stageW, stageH, 90);
        if (fv) state.view = fv;
      }

      state.loaded = true;
      setSaveState("Saved", "ok");
      setStatus("");
      refreshChrome();
      renderLibrary();
      markDirty();
    });
  }

  function okJson(r) {
    if (!r.ok) throw new Error("http " + r.status);
    return r.json();
  }
  function num(v, fallback) {
    var n = typeof v === "number" ? v : parseFloat(v);
    return isFinite(n) ? n : fallback;
  }

  // ---- chrome (empty state, counts, buttons) ---------------------------

  var statusTimer = null;
  function setStatus(text) {
    if (!statusLine) return;
    statusLine.textContent = text || "";
    if (statusTimer) clearTimeout(statusTimer);
    if (text) statusTimer = setTimeout(function () { statusLine.textContent = ""; }, 2600);
  }

  function refreshChrome() {
    var placed = state.nodes.length;
    var total = state.itemOrder.length;
    if (emptyState) {
      var showEmpty = state.loaded && placed === 0;
      emptyState.classList.toggle("d-none", !showEmpty);
      if (showEmpty) {
        var noDocs = emptyState.querySelector("[data-empty-nodocs]");
        var someDocs = emptyState.querySelector("[data-empty-somedocs]");
        if (noDocs) noDocs.classList.toggle("d-none", total !== 0);
        if (someDocs) someDocs.classList.toggle("d-none", total === 0);
      }
    }
    if (libraryCount) {
      var unplaced = total - placed;
      libraryCount.textContent = unplaced > 0 ? String(unplaced) : "";
      libraryCount.classList.toggle("d-none", unplaced <= 0);
    }
    if (zoomLabel) zoomLabel.textContent = Math.round(state.view.zoom * 100) + "%";
    root.querySelectorAll("[data-needs-selection]").forEach(function (b) {
      b.disabled = state.selection.size === 0;
    });
    var undoBtn = el("[data-act='undo']");
    if (undoBtn) undoBtn.disabled = !undo.canUndo();
    var redoBtn = el("[data-act='redo']");
    if (redoBtn) redoBtn.disabled = !undo.canRedo();
  }

  // ---- library panel ---------------------------------------------------

  function renderLibrary() {
    if (!libraryList) return;
    var q = (librarySearch && librarySearch.value || "").trim().toLowerCase();
    var rows = [];
    for (var i = 0; i < state.itemOrder.length; i++) {
      var id = state.itemOrder[i];
      if (nodeById.has(id)) continue;
      var it = state.items[id];
      if (!it) continue;
      if (q && (it.title || "").toLowerCase().indexOf(q) === -1 &&
          (it.summary || "").toLowerCase().indexOf(q) === -1) continue;
      rows.push(it);
    }
    libraryList.innerHTML = "";
    if (!rows.length) {
      var p = document.createElement("p");
      p.className = "se-cv-lib-empty";
      p.textContent = state.itemOrder.length === 0
        ? "Your knowledge base is empty. Summarize something first."
        : (q ? "Nothing matches that." : "Everything is on the canvas.");
      libraryList.appendChild(p);
      return;
    }
    var frag = document.createDocumentFragment();
    rows.slice(0, 300).forEach(function (it) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "se-cv-lib-item";
      btn.setAttribute("data-place", String(it.id));
      var t = document.createElement("span");
      t.className = "se-cv-lib-title";
      t.textContent = it.title || ("Entry " + it.id);
      var m = document.createElement("span");
      m.className = "se-cv-lib-meta";
      m.textContent = [it.source_type || "note", shortDate(it.created_at)]
        .filter(Boolean).join(" · ");
      btn.appendChild(t);
      btn.appendChild(m);
      frag.appendChild(btn);
    });
    libraryList.appendChild(frag);
  }

  function shortDate(s) {
    if (!s) return "";
    var d = new Date(s);
    if (isNaN(d.getTime())) return String(s).slice(0, 10);
    return d.toISOString().slice(0, 10);
  }

  /* Place new cards on a spiral around the centre of the current viewport, so a
   * burst of additions never stacks perfectly on top of itself. */
  function placeItem(id, index) {
    if (nodeById.has(id) || !state.items[id]) return null;
    var c = screenToWorld(state.view, stageW / 2, stageH / 2);
    var i = index == null ? state.nodes.length : index;
    var angle = i * 2.39996; // golden angle, avoids visible spokes
    var radius = 26 * Math.sqrt(i + 1) + 30;
    var node = {
      entry_id: id,
      x: Math.round(c.x + Math.cos(angle) * radius * 3 - CARD_W / 2),
      y: Math.round(c.y + Math.sin(angle) * radius * 2 - CARD_H / 2),
      w: CARD_W, h: CARD_H,
      color: "slate", group: "", pinned: false
    };
    state.nodes.push(node);
    nodeById.set(id, node);
    return node;
  }

  // ---- geometry helpers on live state ----------------------------------

  function selectedNodes() {
    var out = [];
    state.selection.forEach(function (id) {
      var n = nodeById.get(id);
      if (n) out.push(n);
    });
    return out;
  }

  function groupRegions() {
    var byName = new Map();
    for (var i = 0; i < state.nodes.length; i++) {
      var g = state.nodes[i].group;
      if (!g) continue;
      var arr = byName.get(g);
      if (!arr) { arr = []; byName.set(g, arr); }
      arr.push(state.nodes[i]);
    }
    var out = [];
    byName.forEach(function (members, name) {
      var b = boundsOf(members);
      if (!b) return;
      var pad = 22;
      out.push({
        name: name,
        tint: groupTint(name),
        members: members,
        x: b.x - pad, y: b.y - pad - 22,
        w: b.w + pad * 2, h: b.h + pad * 2 + 22
      });
    });
    return out;
  }

  function hitNode(wx, wy) {
    // Back to front: the last drawn card is the one on top.
    for (var i = state.nodes.length - 1; i >= 0; i--) {
      if (pointInRect(wx, wy, state.nodes[i])) return state.nodes[i];
    }
    return null;
  }

  function hitGroupHeader(wx, wy) {
    var regions = groupRegions();
    for (var i = 0; i < regions.length; i++) {
      var r = regions[i];
      if (wx >= r.x && wx <= r.x + r.w && wy >= r.y && wy <= r.y + 26) return r;
    }
    return null;
  }

  function hitEdge(wx, wy) {
    var tol = 8 / state.view.zoom;
    for (var i = state.edges.length - 1; i >= 0; i--) {
      var a = nodeById.get(state.edges[i].from_id);
      var b = nodeById.get(state.edges[i].to_id);
      if (!a || !b) continue;
      var r = edgeRoute(a, b, visible);
      if (distanceToCurve(r, wx, wy) <= tol) return i;
    }
    return -1;
  }

  // Screen-space handle sizes: these must stay a constant size on screen, so
  // they are divided by zoom when compared in world units.
  function resizeHandleRect(n) {
    var s = 12 / state.view.zoom;
    return { x: n.x + n.w - s, y: n.y + n.h - s, w: s, h: s };
  }
  function linkHandleCenter(n) {
    return { x: n.x + n.w, y: n.y + n.h / 2 };
  }
  function nearLinkHandle(n, wx, wy) {
    var c = linkHandleCenter(n);
    var r = 9 / state.view.zoom;
    return Math.abs(wx - c.x) <= r && Math.abs(wy - c.y) <= r;
  }
  /* The handle straddles the card's right border, so half of it falls outside
   * the card and hitNode alone would never find it. Scan for it separately,
   * back to front, so the affordance is reachable from either side. */
  function hitLinkHandle(wx, wy) {
    for (var i = state.nodes.length - 1; i >= 0; i--) {
      if (nearLinkHandle(state.nodes[i], wx, wy)) return state.nodes[i];
    }
    return null;
  }

  // ---- interaction -----------------------------------------------------

  var drag = null;     // active gesture descriptor
  var spaceDown = false;
  var marquee = null;  // {x0,y0,x1,y1} in world coords
  var linking = null;  // {from, x, y}

  function stagePoint(evt) {
    var r = surface.getBoundingClientRect();
    return { x: evt.clientX - r.left, y: evt.clientY - r.top };
  }

  surface.addEventListener("contextmenu", function (e) { e.preventDefault(); });

  surface.addEventListener("pointerdown", function (e) {
    if (floatInput && !floatInput.classList.contains("d-none")) closeFloatInput(true);
    surface.focus();
    var sp = stagePoint(e);
    var wp = screenToWorld(state.view, sp.x, sp.y);
    var panning = e.button === 1 || spaceDown || e.button === 2;

    if (!panning && e.button === 0) {
      var handleNode = hitLinkHandle(wp.x, wp.y);
      var node = hitNode(wp.x, wp.y) || handleNode;

      // Resize corner of an already-selected card wins over a plain drag.
      if (node && state.selection.has(node.entry_id) &&
          pointInRect(wp.x, wp.y, resizeHandleRect(node))) {
        beginChange();
        drag = {
          kind: "resize", node: node,
          ox: wp.x - (node.x + node.w), oy: wp.y - (node.y + node.h)
        };
        surface.setPointerCapture(e.pointerId);
        return;
      }

      // Edge handle -> start drawing a connection.
      if (handleNode) {
        linking = { from: handleNode, x: wp.x, y: wp.y, target: null };
        surface.setPointerCapture(e.pointerId);
        markDirty();
        return;
      }

      if (node) {
        if (e.shiftKey) {
          if (state.selection.has(node.entry_id)) state.selection.delete(node.entry_id);
          else state.selection.add(node.entry_id);
        } else if (!state.selection.has(node.entry_id)) {
          state.selection.clear();
          state.selection.add(node.entry_id);
        }
        state.selectedEdge = -1;
        // Raise to top so it draws above its neighbours while being moved.
        var idx = state.nodes.indexOf(node);
        if (idx > -1 && idx !== state.nodes.length - 1) {
          state.nodes.splice(idx, 1);
          state.nodes.push(node);
        }
        beginChange();
        drag = { kind: "move", start: wp, moved: false, origin: captureOrigins(selectedNodes()) };
        surface.setPointerCapture(e.pointerId);
        refreshChrome();
        markDirty();
        return;
      }

      // Group header -> drag the whole region.
      var header = hitGroupHeader(wp.x, wp.y);
      if (header) {
        state.selection.clear();
        header.members.forEach(function (m) { state.selection.add(m.entry_id); });
        beginChange();
        drag = { kind: "move", start: wp, moved: false, origin: captureOrigins(header.members) };
        surface.setPointerCapture(e.pointerId);
        refreshChrome();
        markDirty();
        return;
      }

      var edgeIdx = hitEdge(wp.x, wp.y);
      if (edgeIdx >= 0) {
        state.selectedEdge = edgeIdx;
        state.selection.clear();
        refreshChrome();
        markDirty();
        return;
      }

      // Empty space: shift starts a marquee, otherwise pan.
      if (e.shiftKey) {
        marquee = { x0: wp.x, y0: wp.y, x1: wp.x, y1: wp.y, additive: true };
        surface.setPointerCapture(e.pointerId);
        markDirty();
        return;
      }
      if (!e.shiftKey && (state.selection.size || state.selectedEdge >= 0)) {
        state.selection.clear();
        state.selectedEdge = -1;
        refreshChrome();
        markDirty();
      }
    }

    drag = { kind: "pan", sx: sp.x, sy: sp.y, vx: state.view.x, vy: state.view.y };
    surface.setPointerCapture(e.pointerId);
    surface.classList.add("is-panning");
  });

  function captureOrigins(nodes) {
    return nodes.map(function (n) { return { node: n, x: n.x, y: n.y }; });
  }

  surface.addEventListener("pointermove", function (e) {
    var sp = stagePoint(e);
    var wp = screenToWorld(state.view, sp.x, sp.y);

    if (linking) {
      linking.x = wp.x;
      linking.y = wp.y;
      var t = hitNode(wp.x, wp.y);
      linking.target = t && t !== linking.from ? t : null;
      markDirty();
      return;
    }

    if (marquee) {
      marquee.x1 = wp.x;
      marquee.y1 = wp.y;
      markDirty();
      return;
    }

    if (!drag) {
      var handleHover = hitLinkHandle(wp.x, wp.y);
      var h = hitNode(wp.x, wp.y) || handleHover;
      var id = h ? h.entry_id : null;
      if (id !== state.hoverId) { state.hoverId = id; markDirty(); }
      var cursor = "default";
      if (handleHover) cursor = "crosshair";
      else if (h) {
        if (state.selection.has(h.entry_id) && pointInRect(wp.x, wp.y, resizeHandleRect(h))) cursor = "nwse-resize";
        else cursor = "grab";
      } else if (spaceDown) cursor = "grab";
      surface.style.cursor = cursor;
      return;
    }

    if (drag.kind === "pan") {
      /* Pan in world units: one screen pixel of pointer travel must always be
       * one screen pixel of content travel, whatever the zoom. */
      state.view.x = drag.vx - (sp.x - drag.sx) / state.view.zoom;
      state.view.y = drag.vy - (sp.y - drag.sy) / state.view.zoom;
      markDirty();
      return;
    }

    if (drag.kind === "move") {
      var dx = wp.x - drag.start.x;
      var dy = wp.y - drag.start.y;
      if (!drag.moved && Math.abs(dx) + Math.abs(dy) > 1 / state.view.zoom) drag.moved = true;
      for (var i = 0; i < drag.origin.length; i++) {
        var o = drag.origin[i];
        o.node.x = o.x + dx;
        o.node.y = o.y + dy;
      }
      markDirty();
      return;
    }

    if (drag.kind === "resize") {
      var n = drag.node;
      var nw = Math.max(MIN_CARD_W, wp.x - drag.ox - n.x);
      var nh = Math.max(MIN_CARD_H, wp.y - drag.oy - n.y);
      if (n.w !== nw || n.h !== nh) {
        n.w = nw; n.h = nh;
        textCache.delete(cacheKey(n, 2));
        textCache.delete(cacheKey(n, 1));
      }
      markDirty();
    }
  });

  function endGesture(e) {
    if (e && e.pointerId != null && surface.hasPointerCapture && surface.hasPointerCapture(e.pointerId)) {
      surface.releasePointerCapture(e.pointerId);
    }
    surface.classList.remove("is-panning");

    if (linking) {
      var target = linking.target;
      var from = linking.from;
      linking = null;
      if (target) {
        var exists = state.edges.some(function (ed) {
          return (ed.from_id === from.entry_id && ed.to_id === target.entry_id) ||
                 (ed.from_id === target.entry_id && ed.to_id === from.entry_id);
        });
        if (!exists) {
          changeNow(function () {
            state.edges.push({ from_id: from.entry_id, to_id: target.entry_id, label: "" });
          });
          state.selectedEdge = state.edges.length - 1;
          setStatus("Connected. Double-click the line to label it.");
        } else {
          setStatus("Those are already connected");
        }
      }
      markDirty();
      return;
    }

    if (marquee) {
      var r = normalizeRect(marquee.x0, marquee.y0, marquee.x1, marquee.y1);
      var hits = lassoSelect(state.nodes, r);
      if (!marquee.additive) state.selection.clear();
      hits.forEach(function (id) { state.selection.add(id); });
      marquee = null;
      state.selectedEdge = -1;
      refreshChrome();
      markDirty();
      return;
    }

    if (drag) {
      if (drag.kind === "move" && drag.moved) {
        // Snap to a small grid on release: keeps hand-arranged boards tidy
        // without fighting the user during the drag itself.
        for (var i = 0; i < drag.origin.length; i++) {
          var n = drag.origin[i].node;
          n.x = Math.round(n.x / GRID_SNAP) * GRID_SNAP;
          n.y = Math.round(n.y / GRID_SNAP) * GRID_SNAP;
        }
      }
      if (drag.kind === "pan") scheduleSave();
      drag = null;
      commitChange();
      markDirty();
    }
  }

  surface.addEventListener("pointerup", endGesture);
  surface.addEventListener("pointercancel", endGesture);

  surface.addEventListener("dblclick", function (e) {
    var sp = stagePoint(e);
    var wp = screenToWorld(state.view, sp.x, sp.y);
    var idx = hitEdge(wp.x, wp.y);
    if (idx >= 0) {
      openEdgeLabelEditor(idx, sp);
      return;
    }
    var header = hitGroupHeader(wp.x, wp.y);
    if (header) {
      openGroupRenameEditor(header, sp);
      return;
    }
    var node = hitNode(wp.x, wp.y);
    if (node) {
      // Double-click a card toggles pinning - the gesture people reach for
      // when they want one thing to stay put during auto-layout.
      changeNow(function () { node.pinned = !node.pinned; });
      setStatus(node.pinned ? "Pinned" : "Unpinned");
    }
  });

  /* Wheel handling has to serve a mouse (discrete wheel, wants zoom) and a
   * trackpad (continuous two-finger scroll, wants pan; pinch arrives as
   * ctrlKey). The auto mode below uses the usual heuristic - a pinch or a
   * line-mode delta is a zoom; smooth pixel deltas are a pan - and the toolbar
   * offers an explicit override for anyone whose hardware lies. */
  var wheelMode = "auto"; // auto | zoom | pan

  surface.addEventListener("wheel", function (e) {
    e.preventDefault();
    var sp = stagePoint(e);
    var isPinch = e.ctrlKey || e.metaKey;
    var shouldZoom;
    if (wheelMode === "zoom") shouldZoom = !e.shiftKey;
    else if (wheelMode === "pan") shouldZoom = isPinch;
    else {
      shouldZoom = isPinch || e.deltaMode !== 0 ||
        (e.deltaX === 0 && Math.abs(e.deltaY) >= 100 && Math.abs(e.deltaY) % 100 === 0);
    }

    if (shouldZoom) {
      var unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? stageH : 1;
      var delta = e.deltaY * unit;
      // Exponential so each notch changes zoom by a constant ratio; 0.0022 puts
      // a typical 100px notch at about 25%, which feels right on both devices.
      var factor = Math.exp(-delta * 0.0022);
      state.view = zoomAt(state.view, sp.x, sp.y, factor);
      if (zoomLabel) zoomLabel.textContent = Math.round(state.view.zoom * 100) + "%";
    } else {
      state.view.x += e.deltaX / state.view.zoom;
      state.view.y += e.deltaY / state.view.zoom;
    }
    scheduleSave();
    markDirty();
  }, { passive: false });

  // ---- floating text editor -------------------------------------------

  var floatTarget = null;
  function openFloatInput(sp, value, placeholder, onCommit) {
    if (!floatInput) return;
    floatTarget = onCommit;
    floatInput.value = value || "";
    floatInput.placeholder = placeholder || "";
    floatInput.classList.remove("d-none");
    floatInput.style.left = Math.round(clamp(sp.x - 90, 8, Math.max(8, stageW - 196))) + "px";
    floatInput.style.top = Math.round(clamp(sp.y - 16, 8, Math.max(8, stageH - 44))) + "px";
    floatInput.focus();
    floatInput.select();
  }
  function closeFloatInput(commit) {
    if (!floatInput || floatInput.classList.contains("d-none")) return;
    var cb = floatTarget;
    var value = floatInput.value;
    floatTarget = null;
    floatInput.classList.add("d-none");
    if (commit && cb) cb(value);
    surface.focus();
    markDirty();
  }
  if (floatInput) {
    floatInput.addEventListener("keydown", function (e) {
      e.stopPropagation();
      if (e.key === "Enter") { e.preventDefault(); closeFloatInput(true); }
      else if (e.key === "Escape") { e.preventDefault(); closeFloatInput(false); }
    });
    floatInput.addEventListener("blur", function () { closeFloatInput(true); });
  }

  function openEdgeLabelEditor(idx, sp) {
    var edge = state.edges[idx];
    if (!edge) return;
    state.selectedEdge = idx;
    openFloatInput(sp, edge.label || "", "Label this link", function (value) {
      changeNow(function () { edge.label = value.trim().slice(0, 60); });
    });
  }

  function openGroupRenameEditor(region, sp) {
    var oldName = region.name;
    openFloatInput(sp, oldName, "Group name", function (value) {
      var name = value.trim().slice(0, 48);
      if (!name || name === oldName) return;
      changeNow(function () {
        state.nodes.forEach(function (n) { if (n.group === oldName) n.group = name; });
      });
    });
  }

  // ---- keyboard --------------------------------------------------------

  function isTypingTarget(t) {
    if (!t) return false;
    var tag = t.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
  }

  window.addEventListener("keydown", function (e) {
    if (isTypingTarget(e.target)) return;
    var mod = e.metaKey || e.ctrlKey;

    if (e.code === "Space") {
      // A focused button or link owns Space; do not hijack it.
      var tag = e.target && e.target.tagName;
      if (tag === "BUTTON" || tag === "A" || tag === "SELECT") return;
      if (spaceDown) return;
      spaceDown = true;
      surface.style.cursor = "grab";
      e.preventDefault();
      return;
    }

    if (mod && (e.key === "z" || e.key === "Z")) {
      e.preventDefault();
      if (e.shiftKey) doRedo(); else doUndo();
      return;
    }
    if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); doRedo(); return; }
    if (mod && (e.key === "a" || e.key === "A")) {
      e.preventDefault();
      state.selection.clear();
      state.nodes.forEach(function (n) { state.selection.add(n.entry_id); });
      state.selectedEdge = -1;
      refreshChrome();
      markDirty();
      return;
    }
    if (mod && (e.key === "g" || e.key === "G")) { e.preventDefault(); groupSelection(); return; }

    if (e.key === "Escape") {
      if (layoutRun) { stopLayout(); return; }
      state.selection.clear();
      state.selectedEdge = -1;
      marquee = null;
      linking = null;
      refreshChrome();
      markDirty();
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      deleteSelection();
      return;
    }
    if ((e.key === "f" || e.key === "F") && !mod) { e.preventDefault(); fitAll(); return; }

    var step = e.shiftKey ? 40 : 8;
    var dx = 0, dy = 0;
    if (e.key === "ArrowLeft") dx = -step;
    else if (e.key === "ArrowRight") dx = step;
    else if (e.key === "ArrowUp") dy = -step;
    else if (e.key === "ArrowDown") dy = step;
    if (dx || dy) {
      e.preventDefault();
      var sel = selectedNodes();
      if (!sel.length) {
        // Nothing selected: nudge the view instead, so arrows are never dead.
        state.view.x += dx; state.view.y += dy;
        scheduleSave();
        markDirty();
        return;
      }
      changeNow(function () {
        sel.forEach(function (n) { n.x += dx; n.y += dy; });
      });
    }
  });

  window.addEventListener("keyup", function (e) {
    if (e.code === "Space") {
      spaceDown = false;
      surface.style.cursor = "default";
    }
  });

  // ---- commands --------------------------------------------------------

  function deleteSelection() {
    if (state.selectedEdge >= 0) {
      var idx = state.selectedEdge;
      changeNow(function () { state.edges.splice(idx, 1); });
      state.selectedEdge = -1;
      setStatus("Link removed");
      renderLibrary();
      return;
    }
    if (!state.selection.size) { setStatus("Nothing selected"); return; }
    var ids = new Set(state.selection);
    changeNow(function () {
      state.nodes = state.nodes.filter(function (n) { return !ids.has(n.entry_id); });
      state.edges = state.edges.filter(function (e) {
        return !ids.has(e.from_id) && !ids.has(e.to_id);
      });
      reindex();
    });
    state.selection.clear();
    state.selectedEdge = -1;
    setStatus(ids.size === 1 ? "Card removed from the canvas" : ids.size + " cards removed");
    renderLibrary();
    refreshChrome();
  }

  function groupSelection() {
    var sel = selectedNodes();
    if (sel.length < 2) { setStatus("Select two or more cards to group"); return; }
    var existing = sel[0].group;
    var same = sel.every(function (n) { return n.group === existing; });
    if (same && existing) {
      changeNow(function () { sel.forEach(function (n) { n.group = ""; }); });
      setStatus("Group removed");
      return;
    }
    var base = "Group";
    var used = new Set(state.nodes.map(function (n) { return n.group; }));
    var name = base;
    var i = 2;
    while (used.has(name)) { name = base + " " + i; i++; }
    changeNow(function () { sel.forEach(function (n) { n.group = name; }); });
    setStatus("Grouped. Double-click the group title to rename it.");
  }

  function applyColor(key) {
    var sel = selectedNodes();
    if (!sel.length) { setStatus("Select a card first"); return; }
    changeNow(function () { sel.forEach(function (n) { n.color = key; }); });
  }

  function togglePin() {
    var sel = selectedNodes();
    if (!sel.length) { setStatus("Select a card first"); return; }
    var anyUnpinned = sel.some(function (n) { return !n.pinned; });
    changeNow(function () { sel.forEach(function (n) { n.pinned = anyUnpinned; }); });
    setStatus(anyUnpinned ? "Pinned. Auto-layout will leave these alone." : "Unpinned");
  }

  function fitAll() {
    if (!state.nodes.length) { setStatus("Nothing to fit yet"); return; }
    var fv = fitView(state.nodes, stageW, stageH, 90);
    if (fv) { state.view = fv; scheduleSave(); refreshChrome(); markDirty(); }
  }

  function zoomBy(factor) {
    state.view = zoomAt(state.view, stageW / 2, stageH / 2, factor);
    refreshChrome();
    scheduleSave();
    markDirty();
  }

  function zoomReset() {
    var c = screenToWorld(state.view, stageW / 2, stageH / 2);
    state.view = { x: c.x - stageW / 2, y: c.y - stageH / 2, zoom: 1 };
    refreshChrome();
    scheduleSave();
    markDirty();
  }

  // ---- auto layout -----------------------------------------------------

  var SIMILARITY_FLOOR = 0.35;

  function buildLinks() {
    var index = new Map();
    state.nodes.forEach(function (n, i) { index.set(n.entry_id, i); });
    var links = [];
    var seen = new Set();
    function add(a, b, weight) {
      if (a === b) return;
      var key = a < b ? a + "-" + b : b + "-" + a;
      if (seen.has(key)) return;
      seen.add(key);
      links.push({ a: a, b: b, weight: weight });
    }
    state.edges.forEach(function (e) {
      var a = index.get(e.from_id);
      var b = index.get(e.to_id);
      if (a == null || b == null) return;
      add(a, b, 1); // hand-drawn links are the strongest signal
    });
    // Similarity pairs pull at a fraction of a real edge, scaled by how similar
    // they are; anything under the floor is noise and would just mush the
    // layout into a ball.
    state.nodes.forEach(function (n) {
      var item = state.items[n.entry_id];
      if (!item || !item.similarity_to) return;
      var a = index.get(n.entry_id);
      Object.keys(item.similarity_to).forEach(function (otherId) {
        var b = index.get(parseInt(otherId, 10));
        if (b == null) return;
        var sim = parseFloat(item.similarity_to[otherId]);
        if (!isFinite(sim) || sim < SIMILARITY_FLOOR) return;
        add(a, b, Math.min(1, sim) * 0.55);
      });
    });
    return links;
  }

  function startLayout() {
    if (layoutRun) { stopLayout(); return; }
    if (state.nodes.length < 2) { setStatus("Place a few cards first"); return; }
    beginChange();
    layoutRun = createForceLayout(state.nodes, buildLinks(), {
      width: Math.max(stageW / state.view.zoom, 1400),
      height: Math.max(stageH / state.view.zoom, 900)
    });
    if (layoutBtn) { layoutBtn.textContent = "Stop"; layoutBtn.classList.add("is-active"); }
    setStatus("Arranging. Press Escape or Stop to end it.");
    markDirty();
  }

  function stopLayout() {
    if (!layoutRun) return;
    layoutRun = null;
    if (layoutBtn) { layoutBtn.textContent = "Auto-arrange"; layoutBtn.classList.remove("is-active"); }
    commitChange();
    fitAll();
    setStatus("Arranged");
  }

  // ---- toolbar wiring --------------------------------------------------

  root.addEventListener("click", function (e) {
    var actionEl = e.target.closest("[data-act]");
    if (actionEl && root.contains(actionEl)) {
      var act = actionEl.getAttribute("data-act");
      switch (act) {
        case "library": toggleLibrary(); break;
        case "layout": startLayout(); break;
        case "fit": fitAll(); break;
        case "zoom-in": zoomBy(1.25); break;
        case "zoom-out": zoomBy(1 / 1.25); break;
        case "zoom-reset": zoomReset(); break;
        case "group": groupSelection(); break;
        case "pin": togglePin(); break;
        case "delete": deleteSelection(); break;
        case "undo": doUndo(); break;
        case "redo": doRedo(); break;
        case "help": toggleHelp(); break;
        case "place-all": placeAll(); break;
        case "close-library": toggleLibrary(false); break;
        case "close-help": toggleHelp(false); break;
      }
      return;
    }
    var swatch = e.target.closest("[data-color]");
    if (swatch && root.contains(swatch)) {
      applyColor(swatch.getAttribute("data-color"));
      return;
    }
    var placeBtn = e.target.closest("[data-place]");
    if (placeBtn && root.contains(placeBtn)) {
      var id = parseInt(placeBtn.getAttribute("data-place"), 10);
      changeNow(function () {
        var n = placeItem(id);
        if (n) { state.selection.clear(); state.selection.add(n.entry_id); }
      });
      renderLibrary();
      refreshChrome();
    }
  });

  function placeAll() {
    var toPlace = state.itemOrder.filter(function (id) { return !nodeById.has(id); });
    if (!toPlace.length) { setStatus("Everything is already on the canvas"); return; }
    changeNow(function () {
      toPlace.forEach(function (id, i) { placeItem(id, state.nodes.length + i); });
    });
    renderLibrary();
    refreshChrome();
    fitAll();
    setStatus(toPlace.length + " cards placed");
  }

  function toggleLibrary(force) {
    if (!libraryPanel) return;
    var show = force == null ? libraryPanel.classList.contains("d-none") : force;
    libraryPanel.classList.toggle("d-none", !show);
    if (show) { renderLibrary(); if (librarySearch) librarySearch.focus(); }
  }

  function toggleHelp(force) {
    if (!helpPanel) return;
    var show = force == null ? helpPanel.classList.contains("d-none") : force;
    helpPanel.classList.toggle("d-none", !show);
  }

  var wheelModeSelect = el("[data-wheel-mode]");
  if (wheelModeSelect) {
    wheelModeSelect.addEventListener("change", function () {
      wheelMode = wheelModeSelect.value;
    });
  }

  if (librarySearch) {
    librarySearch.addEventListener("input", renderLibrary);
    librarySearch.addEventListener("keydown", function (e) { e.stopPropagation(); });
  }

  // ---- minimap ---------------------------------------------------------

  var mmTransform = null;

  function jumpFromMinimap(evt) {
    if (!mmTransform) return;
    var r = minimapEl.getBoundingClientRect();
    var px = evt.clientX - r.left;
    var py = evt.clientY - r.top;
    var wx = (px - mmTransform.ox) / mmTransform.scale + mmTransform.wx;
    var wy = (py - mmTransform.oy) / mmTransform.scale + mmTransform.wy;
    state.view.x = wx - stageW / (2 * state.view.zoom);
    state.view.y = wy - stageH / (2 * state.view.zoom);
    scheduleSave();
    markDirty();
  }

  var mmDragging = false;
  minimapEl.addEventListener("pointerdown", function (e) {
    mmDragging = true;
    minimapEl.setPointerCapture(e.pointerId);
    jumpFromMinimap(e);
  });
  minimapEl.addEventListener("pointermove", function (e) {
    if (mmDragging) jumpFromMinimap(e);
  });
  minimapEl.addEventListener("pointerup", function (e) {
    mmDragging = false;
    if (minimapEl.hasPointerCapture && minimapEl.hasPointerCapture(e.pointerId)) {
      minimapEl.releasePointerCapture(e.pointerId);
    }
  });

  // ---- sizing ----------------------------------------------------------

  function resize() {
    var rect = stage.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width));
    var h = Math.max(1, Math.round(rect.height));
    var nextDpr = Math.min(window.devicePixelRatio || 1, 2);
    // Only touch the backing store when it genuinely changed: assigning
    // canvas.width clears the surface and reallocates, so doing it per frame
    // would be the single most expensive thing here.
    if (w !== stageW || h !== stageH || nextDpr !== dpr) {
      stageW = w; stageH = h; dpr = nextDpr;
      surface.width = Math.round(w * dpr);
      surface.height = Math.round(h * dpr);
      surface.style.width = w + "px";
      surface.style.height = h + "px";
      var mw = minimapEl.clientWidth || 180;
      var mh = minimapEl.clientHeight || 120;
      minimapEl.width = Math.round(mw * dpr);
      minimapEl.height = Math.round(mh * dpr);
      markDirty();
    }
  }

  if (window.ResizeObserver) {
    new ResizeObserver(resize).observe(stage);
  } else {
    window.addEventListener("resize", resize);
  }

  // ---- text layout cache ----------------------------------------------

  function cacheKey(n, lod) {
    return n.entry_id + "|" + Math.round(n.w) + "|" + Math.round(n.h) + "|" + lod;
  }

  function wrapText(text, maxWidth, maxLines, font) {
    ctx.font = font;
    var words = String(text || "").split(/\s+/).filter(Boolean);
    var lines = [];
    var line = "";
    for (var i = 0; i < words.length; i++) {
      var trial = line ? line + " " + words[i] : words[i];
      if (ctx.measureText(trial).width <= maxWidth || !line) {
        line = trial;
      } else {
        lines.push(line);
        line = words[i];
        if (lines.length === maxLines) break;
      }
    }
    if (lines.length < maxLines && line) lines.push(line);
    if (lines.length === maxLines && (line || words.length > 1)) {
      var last = lines[lines.length - 1];
      while (last.length > 1 && ctx.measureText(last + "…").width > maxWidth) {
        last = last.slice(0, -1);
      }
      var joined = lines.join(" ");
      if (joined.length < String(text || "").length) lines[lines.length - 1] = last + "…";
    }
    return lines;
  }

  function cardText(n, lod) {
    var key = cacheKey(n, lod);
    var cached = textCache.get(key);
    if (cached) return cached;
    var item = state.items[n.entry_id] || {};
    var inner = n.w - 24;
    var out = { title: [], body: [] };
    out.title = wrapText(item.title || ("Entry " + n.entry_id), inner, lod === 1 ? 1 : 2, "600 14px Inter, sans-serif");
    if (lod === 2) {
      var titleLines = out.title.length;
      var room = Math.max(0, Math.floor((n.h - 34 - titleLines * 18 - 16) / 15));
      out.body = room > 0 ? wrapText(item.summary || "", inner, Math.min(room, 5), "400 12px Inter, sans-serif") : [];
    }
    // Bounded cache: 1200 entries covers any realistic board and stops a long
    // resize session from growing without limit.
    if (textCache.size > 1200) textCache.clear();
    textCache.set(key, out);
    return out;
  }

  // ---- drawing ---------------------------------------------------------

  function viewportWorldRect() {
    var tl = screenToWorld(state.view, 0, 0);
    var br = screenToWorld(state.view, stageW, stageH);
    return { x: tl.x, y: tl.y, w: br.x - tl.x, h: br.y - tl.y };
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, stageW, stageH);

    var z = state.view.zoom;
    var lod = lodLevel(z);
    var vp = viewportWorldRect();

    drawBackdrop(vp, z);

    // Cull once, reuse the list for cards, edges and handles.
    visible.length = 0;
    var pad = 80;
    var cullRect = { x: vp.x - pad, y: vp.y - pad, w: vp.w + pad * 2, h: vp.h + pad * 2 };
    for (var i = 0; i < state.nodes.length; i++) {
      if (rectIntersects(state.nodes[i], cullRect)) visible.push(state.nodes[i]);
    }

    ctx.save();
    ctx.translate(-state.view.x * z, -state.view.y * z);
    ctx.scale(z, z);

    drawGroups(z);
    drawEdges(z, cullRect);
    drawCards(lod, z);
    drawOverlays(z);

    ctx.restore();

    if (marquee) drawMarquee();
    drawScaleHint(z);
  }

  function drawBackdrop(vp, z) {
    ctx.fillStyle = cssVar("--se-bg", "#f4f5fb");
    ctx.fillRect(0, 0, stageW, stageH);
    if (z < 0.35) return; // dots become moire at small scales, and cost a lot
    var spacing = 48 * z;
    while (spacing < 22) spacing *= 2; // keep dot density roughly constant
    var ox = (-state.view.x * z) % spacing;
    var oy = (-state.view.y * z) % spacing;
    if (ox < 0) ox += spacing;
    if (oy < 0) oy += spacing;
    ctx.fillStyle = "rgba(26,29,41,0.07)";
    for (var x = ox; x < stageW; x += spacing) {
      for (var y = oy; y < stageH; y += spacing) {
        ctx.fillRect(x, y, 1.5, 1.5);
      }
    }
  }

  function drawGroups(z) {
    var regions = groupRegions();
    for (var i = 0; i < regions.length; i++) {
      var r = regions[i];
      ctx.fillStyle = hexToRgba(r.tint, 0.07);
      ctx.strokeStyle = hexToRgba(r.tint, 0.4);
      ctx.lineWidth = 1.5 / z;
      roundRect(r.x, r.y, r.w, r.h, 18);
      ctx.fill();
      ctx.stroke();
      // Below ~0.35 the 13px world-space title renders under 5 device pixels,
      // so it is noise rather than information.
      if (z >= 0.35) {
        ctx.fillStyle = r.tint;
        ctx.font = "600 13px Inter, sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText(r.name, r.x + 14, r.y + 15);
      }
    }
  }

  function drawEdges(z, cullRect) {
    ctx.lineWidth = Math.max(1.2, 1.6 / Math.max(z, 0.4));
    for (var i = 0; i < state.edges.length; i++) {
      var e = state.edges[i];
      var a = nodeById.get(e.from_id);
      var b = nodeById.get(e.to_id);
      if (!a || !b) continue;
      var span = {
        x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
        w: Math.abs(a.x - b.x) + Math.max(a.w, b.w),
        h: Math.abs(a.y - b.y) + Math.max(a.h, b.h)
      };
      if (!rectIntersects(span, cullRect)) continue;
      var r = edgeRoute(a, b, visible);
      var selected = state.selectedEdge === i;
      ctx.strokeStyle = selected ? cssVar("--se-primary", "#4f46e5") : "rgba(91,96,114,0.45)";
      ctx.lineWidth = (selected ? 2.4 : 1.6) / Math.max(z, 0.35);
      ctx.beginPath();
      ctx.moveTo(r.x1, r.y1);
      ctx.quadraticCurveTo(r.cx, r.cy, r.x2, r.y2);
      ctx.stroke();
      drawArrow(r, z, selected);
      if (e.label && z >= 0.5) {
        var mid = quadPoint(r, 0.5);
        ctx.font = "500 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        var tw = ctx.measureText(e.label).width;
        ctx.fillStyle = cssVar("--se-surface", "#ffffff");
        roundRect(mid.x - tw / 2 - 6, mid.y - 9, tw + 12, 18, 9);
        ctx.fill();
        ctx.strokeStyle = selected ? cssVar("--se-primary", "#4f46e5") : cssVar("--se-border", "#e6e7f0");
        ctx.lineWidth = 1 / z;
        ctx.stroke();
        ctx.fillStyle = cssVar("--se-ink-muted", "#5b6072");
        ctx.fillText(e.label, mid.x, mid.y);
        ctx.textAlign = "left";
      }
    }
  }

  function drawArrow(r, z, selected) {
    if (z < 0.3) return;
    var tip = quadPoint(r, 0.97);
    var back = quadPoint(r, 0.86);
    var ang = Math.atan2(tip.y - back.y, tip.x - back.x);
    var size = 8 / Math.max(z, 0.4);
    ctx.fillStyle = selected ? cssVar("--se-primary", "#4f46e5") : "rgba(91,96,114,0.55)";
    ctx.beginPath();
    ctx.moveTo(tip.x, tip.y);
    ctx.lineTo(tip.x - Math.cos(ang - 0.4) * size, tip.y - Math.sin(ang - 0.4) * size);
    ctx.lineTo(tip.x - Math.cos(ang + 0.4) * size, tip.y - Math.sin(ang + 0.4) * size);
    ctx.closePath();
    ctx.fill();
  }

  function drawCards(lod, z) {
    var useShadow = z >= 0.4 && visible.length <= 200;
    for (var i = 0; i < visible.length; i++) {
      var n = visible[i];
      var pal = colorFor(n.color);
      var selected = state.selection.has(n.entry_id);

      if (useShadow) {
        ctx.shadowColor = "rgba(26,29,41,0.10)";
        ctx.shadowBlur = 12 / z;
        ctx.shadowOffsetY = 3 / z;
      }
      ctx.fillStyle = pal.fill;
      roundRect(n.x, n.y, n.w, n.h, lod === 0 ? 4 : 12);
      ctx.fill();
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      ctx.strokeStyle = selected ? cssVar("--se-primary", "#4f46e5") : pal.edge;
      ctx.lineWidth = (selected ? 2.2 : 1) / z;
      roundRect(n.x, n.y, n.w, n.h, lod === 0 ? 4 : 12);
      ctx.stroke();

      if (lod === 0) {
        // A coloured block still carries position, size and tag - which is what
        // spatial memory works with at this scale.
        if (n.color !== "slate") {
          ctx.fillStyle = hexToRgba(pal.ink, 0.35);
          ctx.fillRect(n.x, n.y, n.w, Math.min(6, n.h));
        }
        continue;
      }

      var text = cardText(n, lod);
      ctx.fillStyle = pal.ink;
      ctx.font = "600 14px Inter, sans-serif";
      ctx.textBaseline = "alphabetic";
      var ty = n.y + 26;
      for (var t = 0; t < text.title.length; t++) {
        ctx.fillText(text.title[t], n.x + 12, ty);
        ty += 18;
      }

      if (lod === 2 && text.body.length) {
        ctx.fillStyle = hexToRgba(pal.ink, 0.62);
        ctx.font = "400 12px Inter, sans-serif";
        var by = ty + 6;
        for (var bI = 0; bI < text.body.length; bI++) {
          ctx.fillText(text.body[bI], n.x + 12, by);
          by += 15;
        }
      }

      if (n.pinned && lod >= 1) {
        ctx.fillStyle = cssVar("--se-primary", "#4f46e5");
        ctx.beginPath();
        ctx.arc(n.x + n.w - 12, n.y + 12, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawOverlays(z) {
    // Selection handles, link handles, in-progress connection. Drawn after the
    // cards so nothing can cover them.
    var hover = state.hoverId != null ? nodeById.get(state.hoverId) : null;
    var primary = cssVar("--se-primary", "#4f46e5");

    for (var i = 0; i < visible.length; i++) {
      var n = visible[i];
      var selected = state.selection.has(n.entry_id);
      if (!selected && n !== hover) continue;
      if (z < 0.3) continue;

      var lc = linkHandleCenter(n);
      ctx.fillStyle = cssVar("--se-surface", "#ffffff");
      ctx.strokeStyle = primary;
      ctx.lineWidth = 1.6 / z;
      ctx.beginPath();
      ctx.arc(lc.x, lc.y, 5 / z, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      if (selected) {
        var rh = resizeHandleRect(n);
        ctx.fillStyle = primary;
        ctx.fillRect(rh.x + rh.w * 0.25, rh.y + rh.h * 0.25, rh.w * 0.6, rh.h * 0.6);
      }
    }

    if (linking) {
      ctx.strokeStyle = primary;
      ctx.lineWidth = 2 / z;
      ctx.setLineDash([6 / z, 5 / z]);
      var from = linkHandleCenter(linking.from);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(linking.x, linking.y);
      ctx.stroke();
      ctx.setLineDash([]);
      if (linking.target) {
        ctx.strokeStyle = primary;
        ctx.lineWidth = 2.4 / z;
        roundRect(linking.target.x, linking.target.y, linking.target.w, linking.target.h, 12);
        ctx.stroke();
      }
    }
  }

  function drawMarquee() {
    var a = worldToScreen(state.view, marquee.x0, marquee.y0);
    var b = worldToScreen(state.view, marquee.x1, marquee.y1);
    var r = normalizeRect(a.x, a.y, b.x, b.y);
    ctx.fillStyle = "rgba(79,70,229,0.10)";
    ctx.strokeStyle = cssVar("--se-primary", "#4f46e5");
    ctx.lineWidth = 1;
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
  }

  function drawScaleHint(z) {
    if (z >= 0.25 || !state.nodes.length) return;
    ctx.fillStyle = cssVar("--se-ink-muted", "#5b6072");
    ctx.font = "500 12px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Zoomed out. Cards show as blocks below 25%.", 14, stageH - 14);
  }

  function roundRect(x, y, w, h, r) {
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  var cssCache = {};
  function cssVar(name, fallback) {
    if (cssCache[name] !== undefined) return cssCache[name];
    var v = "";
    try {
      v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    } catch (err) { v = ""; }
    cssCache[name] = v || fallback;
    return cssCache[name];
  }

  function hexToRgba(hex, alpha) {
    if (!hex || hex.charAt(0) !== "#") return "rgba(26,29,41," + alpha + ")";
    var h = hex.slice(1);
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var n = parseInt(h, 16);
    if (!isFinite(n)) return "rgba(26,29,41," + alpha + ")";
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + alpha + ")";
  }

  // ---- minimap drawing -------------------------------------------------

  function drawMinimap() {
    var mw = minimapEl.width / dpr;
    var mh = minimapEl.height / dpr;
    mmCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mmCtx.clearRect(0, 0, mw, mh);
    mmCtx.fillStyle = "rgba(255,255,255,0.86)";
    mmCtx.fillRect(0, 0, mw, mh);

    var vp = viewportWorldRect();
    var all = state.nodes.slice();
    // Include the viewport in the world bounds, so the minimap still shows you
    // where you are when you pan away from every card.
    var b = boundsOf(all.concat([{ x: vp.x, y: vp.y, w: vp.w, h: vp.h }]));
    if (!b) { mmTransform = null; return; }
    var pad = 10;
    var scale = Math.min((mw - pad * 2) / Math.max(b.w, 1), (mh - pad * 2) / Math.max(b.h, 1));
    var ox = pad + ((mw - pad * 2) - b.w * scale) / 2;
    var oy = pad + ((mh - pad * 2) - b.h * scale) / 2;
    mmTransform = { scale: scale, ox: ox, oy: oy, wx: b.x, wy: b.y };

    function mx(x) { return ox + (x - b.x) * scale; }
    function my(y) { return oy + (y - b.y) * scale; }

    groupRegions().forEach(function (r) {
      mmCtx.fillStyle = hexToRgba(r.tint, 0.12);
      mmCtx.fillRect(mx(r.x), my(r.y), r.w * scale, r.h * scale);
    });

    for (var i = 0; i < state.nodes.length; i++) {
      var n = state.nodes[i];
      var pal = colorFor(n.color);
      mmCtx.fillStyle = state.selection.has(n.entry_id)
        ? cssVar("--se-primary", "#4f46e5")
        : hexToRgba(pal.ink, 0.45);
      mmCtx.fillRect(mx(n.x), my(n.y), Math.max(2, n.w * scale), Math.max(2, n.h * scale));
    }

    mmCtx.strokeStyle = cssVar("--se-primary", "#4f46e5");
    mmCtx.lineWidth = 1.5;
    mmCtx.strokeRect(mx(vp.x) + 0.5, my(vp.y) + 0.5, vp.w * scale, vp.h * scale);
  }

  // ---- frame loop ------------------------------------------------------

  function frame(now) {
    if (layoutRun) {
      /* Two solver iterations per frame: one alone looks sluggish on big
       * boards, more than two and the cards visibly teleport. */
      layoutRun.step();
      layoutRun.step();
      if (layoutRun.done) stopLayout();
      markDirty();
    }

    if (dirty) {
      dirty = false;
      draw();
    }

    // The minimap changes slowly and is small; 8fps is imperceptible here and
    // keeps a second canvas off the critical path during a drag.
    if (minimapDirty && now - lastMinimapDraw > 125) {
      minimapDirty = false;
      lastMinimapDraw = now;
      drawMinimap();
    }

    requestAnimationFrame(frame);
  }

  // ---- boot ------------------------------------------------------------

  surface.setAttribute("tabindex", "0");
  resize();
  refreshChrome();
  requestAnimationFrame(frame);
  load();

  window.addEventListener("beforeunload", function () {
    if (saveTimer) {
      clearTimeout(saveTimer);
      // Best effort: a synchronous keepalive save so a quick tab close does not
      // lose the last 1.5 seconds of arranging.
      try {
        fetch(API.layout, {
          method: "POST",
          credentials: "same-origin",
          keepalive: true,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            nodes: state.nodes,
            edges: state.edges,
            view: state.view
          })
        });
      } catch (err) { /* nothing useful to do while unloading */ }
    }
  });

  // Exposed for the math test harness and for debugging from the console.
  window.SECanvas = {
    worldToScreen: worldToScreen,
    screenToWorld: screenToWorld,
    zoomAt: zoomAt,
    fitView: fitView,
    lassoSelect: lassoSelect,
    lodLevel: lodLevel,
    createForceLayout: createForceLayout,
    UndoStack: UndoStack,
    state: state
  };
})();
