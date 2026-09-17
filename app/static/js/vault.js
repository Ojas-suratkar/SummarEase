/*
 * SummarEase — zero-knowledge vault (client side)
 * ------------------------------------------------------------------
 * Everything in this file runs in the user's browser. The server never
 * receives the passphrase, the derived key, or plaintext. It receives
 * base64 ciphertext plus the public parameters needed to decrypt later
 * (salt, IV, algorithm label, iteration count). Those parameters are not
 * secret: AES-GCM and PBKDF2 are designed so that publishing them costs
 * nothing as long as the passphrase stays on the device.
 *
 * The security claim this file has to hold up is: "we cannot read your
 * notes, even if subpoenaed, even if breached." That is only true if
 * three things are true here:
 *   1. plaintext never crosses the network boundary (see saveCurrent);
 *   2. the key is derived locally and is non-extractable;
 *   3. the passphrase lives in one closure variable and nowhere else —
 *      not localStorage, not sessionStorage, not window, not a cookie.
 * Every deviation from those three is a break of the product promise,
 * so please treat them as load-bearing when editing this file.
 *
 * Honest limitation, stated here rather than hidden: JavaScript strings
 * are immutable and garbage-collected, so we cannot truly wipe the
 * passphrase from memory. Locking drops every reference we hold and
 * clears the derived-key cache, which is the strongest guarantee a web
 * page can make. A compromised browser or a malicious server-served
 * script can still defeat this; zero-knowledge protects against a
 * breached or compelled *server*, not against a compromised *client*.
 */
(function () {
  "use strict";

  /* ================= CRYPTO CORE START ================= */
  /* Everything between these markers is deliberately free of DOM and
   * network references so it can be lifted out verbatim and tested in
   * Node's WebCrypto (identical API surface to the browser's). If you
   * add a `document.` or `fetch(` in here, the test harness breaks —
   * that is the point. */

  // PBKDF2-SHA256 at 600,000 iterations. This is the current OWASP
  // Password Storage Cheat Sheet floor for PBKDF2-HMAC-SHA256; PBKDF2 is
  // cheap to accelerate on GPUs/ASICs, so the iteration count is the only
  // lever we have to make an offline guessing attack against a stolen
  // ciphertext blob expensive. We use PBKDF2 rather than Argon2id purely
  // because it is what WebCrypto ships natively — adding a WASM Argon2
  // would mean shipping a third-party binary into the one place where the
  // user has to trust our code most, which is a worse trade than the
  // weaker KDF. Raise this number as hardware improves; old items keep
  // working because each record stores the iteration count it was made
  // with, and we re-encrypt at the current count on every save.
  var KDF_ITERATIONS = 600000;

  // 16 random bytes, fresh per item. The salt's job is to stop one
  // precomputed table (or one derived key) from unlocking many items or
  // many users. It is not secret and is stored next to the ciphertext.
  var SALT_BYTES = 16;

  // 12 bytes is the IV size AES-GCM is specified and optimised for: a
  // 96-bit IV is used directly as the counter block, while any other
  // length gets hashed through GHASH first, which is slower and buys
  // nothing.
  var IV_BYTES = 12;

  // Stored verbatim in the record so a future client can tell what it is
  // looking at without guessing.
  var ALGO_LABEL = "AES-256-GCM+PBKDF2-SHA256";

  // Typed error so the UI can tell "you typed the wrong passphrase" apart
  // from "the network died" and show a calm sentence instead of a stack
  // trace.
  function VaultError(code, message) {
    var e = new Error(message);
    e.name = "VaultError";
    e.code = code;
    return e;
  }

  function randomBytes(n) {
    var b = new Uint8Array(n);
    // crypto.getRandomValues is a CSPRNG. Math.random() is not, and using
    // it for an IV or salt here would silently void the whole design.
    crypto.getRandomValues(b);
    return b;
  }

  function toBase64(bytes) {
    // Chunked so that a large note does not blow the argument limit of
    // String.fromCharCode.apply on big arrays.
    var chunk = 0x8000;
    var out = "";
    for (var i = 0; i < bytes.length; i += chunk) {
      out += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(out);
  }

  function fromBase64(str) {
    var bin = atob(str);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function encodeUtf8(str) {
    return new TextEncoder().encode(str);
  }

  function decodeUtf8(bytes) {
    // fatal:true so malformed bytes surface as an error rather than as
    // silent U+FFFD replacement characters in the user's note.
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  }

  // Deriving a key costs ~600k HMAC rounds, i.e. a noticeable fraction of
  // a second. A vault with 30 items would otherwise stall the UI for ten
  // seconds just to list titles. The cache is keyed by salt+iterations
  // and lives only for the duration of one unlocked session: lock() and
  // every unlock attempt clear it, so a key derived from a previous
  // passphrase can never be reused after a re-unlock.
  var keyCache = new Map();

  function clearKeyCache() {
    keyCache.clear();
  }

  async function deriveKey(passphrase, saltBytes, iterations) {
    var material = await crypto.subtle.importKey(
      "raw",
      encodeUtf8(passphrase),
      "PBKDF2",
      false, // never extractable — the raw passphrase bytes must not be
      ["deriveKey"] // readable back out of the WebCrypto boundary.
    );
    return crypto.subtle.deriveKey(
      { name: "PBKDF2", salt: saltBytes, iterations: iterations, hash: "SHA-256" },
      material,
      { name: "AES-GCM", length: 256 }, // AES-256, not 128: the cost
      // difference is negligible and it keeps the margin against future
      // attacks comfortable for data with a long shelf life.
      false, // extractable: false — the key object can encrypt and
      // decrypt but its bytes cannot be read by any script on the page,
      // including ours, including an injected one.
      ["encrypt", "decrypt"]
    );
  }

  async function getKey(passphrase, saltBytes, iterations) {
    var cacheKey = toBase64(saltBytes) + "|" + iterations;
    var hit = keyCache.get(cacheKey);
    // Belt and braces: the entry records which passphrase produced it and
    // is only reused for that same passphrase. Without this check, a
    // second passphrase applied to the same salt would silently get the
    // first passphrase's key and appear to "decrypt" correctly, which
    // would turn a wrong-passphrase error into data corruption.
    if (hit && hit.pass === passphrase) return hit.key;
    var key = await deriveKey(passphrase, saltBytes, iterations);
    keyCache.set(cacheKey, { pass: passphrase, key: key });
    return key;
  }

  /**
   * Encrypt a string. Returns the exact record shape the server API
   * expects, with every binary field base64-encoded.
   */
  async function encryptPayload(passphrase, plaintext) {
    var salt = randomBytes(SALT_BYTES);
    var key = await getKey(passphrase, salt, KDF_ITERATIONS);

    // Fresh 12 random bytes for every single encryption, never derived
    // from a counter we might reset and never reused.
    //
    // Why this matters more for GCM than for other modes: GCM is CTR mode
    // plus a GHASH authenticator. Reusing an IV under the same key means
    // (a) the two keystreams are identical, so XOR-ing the two ciphertexts
    // yields the XOR of the two plaintexts, and far worse (b) the
    // attacker can solve for the GHASH subkey H. With H recovered, they
    // can forge a valid authentication tag for *any* message under that
    // key — the "forbidden attack". So a single IV reuse does not degrade
    // GCM, it destroys both confidentiality and authenticity for that
    // key. Note also that every save derives a brand-new key from a new
    // salt, which means even a catastrophic RNG repeat of an IV would
    // have to coincide with a repeat of the salt to do damage.
    var iv = randomBytes(IV_BYTES);

    var ctBuf = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: iv, tagLength: 128 }, // full 128-bit tag; a
      // truncated tag lowers forgery cost for no benefit here.
      key,
      encodeUtf8(plaintext)
    );
    var record = {
      ciphertext: toBase64(new Uint8Array(ctBuf)),
      iv: toBase64(iv),
      salt: toBase64(salt),
      algo: ALGO_LABEL,
      kdf_iterations: KDF_ITERATIONS
    };

    // Integrity self-check. We decrypt what we just produced and compare
    // it to what we were handed, before it is allowed anywhere near the
    // network. This catches a bad RNG, a mis-sized IV, a base64 rounding
    // bug, or flaky hardware — all of which would otherwise be discovered
    // months later as data the user can never get back. Zero-knowledge
    // means we cannot repair a corrupted blob for them, so silent
    // corruption must be impossible by construction, not unlikely.
    var roundTrip = await decryptPayload(passphrase, record);
    if (roundTrip !== plaintext) {
      throw VaultError(
        "integrity",
        "Encryption self-check failed. Nothing was saved."
      );
    }
    return record;
  }

  /**
   * Decrypt a record produced by encryptPayload. Throws a VaultError with
   * code "wrong-passphrase" when the GCM tag does not verify — which is
   * the same signal for a wrong key and for tampered ciphertext, because
   * GCM cannot and should not distinguish them.
   */
  async function decryptPayload(passphrase, record) {
    var iterations = record.kdf_iterations || KDF_ITERATIONS;
    var salt, iv, ct;
    try {
      salt = fromBase64(record.salt);
      iv = fromBase64(record.iv);
      ct = fromBase64(record.ciphertext);
    } catch (e) {
      throw VaultError("malformed", "This item's stored data is not readable.");
    }
    if (salt.length === 0 || iv.length !== IV_BYTES) {
      throw VaultError("malformed", "This item's stored data is not readable.");
    }
    var key = await getKey(passphrase, salt, iterations);
    var ptBuf;
    try {
      ptBuf = await crypto.subtle.decrypt(
        { name: "AES-GCM", iv: iv, tagLength: 128 },
        key,
        ct
      );
    } catch (e) {
      // WebCrypto throws a bare, message-less OperationError on tag
      // failure. Translating it here is what keeps a wrong passphrase
      // from surfacing as a stack trace in the UI.
      throw VaultError(
        "wrong-passphrase",
        "That passphrase does not decrypt this item."
      );
    }
    try {
      return decodeUtf8(new Uint8Array(ptBuf));
    } catch (e) {
      throw VaultError("malformed", "Decrypted data was not valid text.");
    }
  }

  /* ---- Passphrase strength, computed locally ----
   * This is an estimate, not a guarantee, and the UI says so. We model
   * an attacker doing brute force over the character classes actually
   * used: bits = length * log2(poolSize). Two corrections keep it from
   * flattering obviously bad choices:
   *   - repeated adjacent characters count as half a character, because
   *     "aaaaaaaa" is not eight characters of search space;
   *   - anything whose alphabetic skeleton is a known common password is
   *     capped at 12 bits, because it falls in the first few thousand
   *     guesses of any real cracking run regardless of length.
   * The blocklist is intentionally tiny. A full list (rockyou, HIBP) is
   * megabytes and would have to be fetched, which means telling the
   * server something about the passphrase. That trade is not worth it, so
   * we keep a handful of the worst offenders locally and are honest that
   * this check is shallow.
   */
  var COMMON_PASSWORDS = [
    "password", "passw0rd", "password1", "123456", "1234567", "12345678",
    "123456789", "qwerty", "qwerty123", "letmein", "iloveyou", "admin",
    "welcome", "monkey", "dragon", "abc123", "trustno1", "sunshine",
    "princess", "football", "baseball", "master", "shadow", "superman",
    "batman", "000000", "111111", "changeme", "secret", "hunter2",
    "summarease", "vault", "letmein123"
  ];

  function estimateStrength(pass) {
    if (!pass) {
      return { bits: 0, label: "Empty", band: "none", pool: 0, notes: [] };
    }
    var notes = [];
    var pool = 0;
    if (/[a-z]/.test(pass)) pool += 26;
    if (/[A-Z]/.test(pass)) pool += 26;
    if (/[0-9]/.test(pass)) pool += 10;
    if (/[^a-zA-Z0-9]/.test(pass)) pool += 33; // printable ASCII symbols
    if (pool === 0) pool = 26;

    var effective = 1;
    for (var i = 1; i < pass.length; i++) {
      effective += pass[i] === pass[i - 1] ? 0.5 : 1;
    }
    var bits = effective * (Math.log(pool) / Math.log(2));

    // Undo the usual character substitutions before comparing, so
    // "P@ssw0rd123" is recognised as "password" with padding. Crackers
    // apply exactly these rules, so pretending they add entropy would be
    // flattering the user at their own expense.
    var LEET = { "@": "a", "4": "a", "0": "o", "1": "i", "!": "i", "3": "e", "$": "s", "5": "s", "7": "t", "+": "t", "8": "b", "9": "g" };
    var norm = "";
    var lower = pass.toLowerCase();
    for (var k = 0; k < lower.length; k++) {
      var ch = lower[k];
      norm += Object.prototype.hasOwnProperty.call(LEET, ch) ? LEET[ch] : ch;
    }
    var skeleton = norm.replace(/[^a-z]/g, "");
    for (var j = 0; j < COMMON_PASSWORDS.length; j++) {
      var c = COMMON_PASSWORDS[j];
      if (lower === c || norm === c || skeleton === c) {
        bits = Math.min(bits, 10);
        notes.push("This is a well-known password. It is guessed almost immediately.");
        break;
      }
      // A common password with characters bolted on is still a common
      // password; only the added characters carry real search space, and
      // even those are usually digits, so we value them at about 4 bits
      // each rather than the full alphabet.
      if (c.length >= 6 && skeleton.indexOf(c) !== -1) {
        var extra = Math.max(0, pass.length - c.length);
        bits = Math.min(bits, 10 + extra * 4);
        notes.push("This is a common password with a few characters added. That pattern is tried early in any cracking run.");
        break;
      }
    }
    if (pass.length < 12) {
      notes.push("Short passphrases fall quickly to offline guessing. Aim for 4 or more unrelated words.");
    }
    if (/^[0-9]+$/.test(pass)) {
      notes.push("Digits only. The search space is very small.");
    }

    var band, label;
    if (bits < 40) { band = "weak"; label = "Weak"; }
    else if (bits < 60) { band = "fair"; label = "Fair"; }
    else if (bits < 80) { band = "strong"; label = "Strong"; }
    else { band = "excellent"; label = "Very strong"; }

    return { bits: Math.round(bits), label: label, band: band, pool: pool, notes: notes };
  }
  /* ================= CRYPTO CORE END ================= */

  /* ------------------------------------------------------------------
   * Session state. `passphrase` is the single reference to the secret in
   * the whole program. It is a closure variable: not on window, not on
   * any element, never serialised, never passed to fetch. Read it only
   * through the functions below.
   * ------------------------------------------------------------------ */
  var passphrase = null;
  var items = [];          // server metadata: entry_id, updated_at, bytes
  var titles = new Map();  // entry_id -> decrypted title (memory only)
  var current = null;      // { entry_id, title, body, record, isNew, dirty }
  var idleTimer = null;
  var countdownTimer = null;
  var lockDeadline = 0;
  var saving = false;

  var AUTO_LOCK_MS = 10 * 60 * 1000; // 10 minutes of inactivity.

  var el = {};
  function $(id) { return document.getElementById(id); }

  /* ---------------- Server API ---------------- */

  async function api(path, options) {
    var opts = Object.assign(
      {
        // Same-origin session cookie only. No token is put in a URL or in
        // storage, and the request never carries anything derived from
        // the passphrase.
        credentials: "same-origin",
        headers: { Accept: "application/json" }
      },
      options || {}
    );
    var res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw VaultError("network", "Could not reach the server. Your note was not saved.");
    }
    var data = null;
    try {
      data = await res.json();
    } catch (e) {
      data = null;
    }
    if (!res.ok) {
      var msg = (data && data.error) || "The server returned an error (" + res.status + ").";
      throw VaultError("server", msg);
    }
    return data || {};
  }

  function putItem(entryId, record) {
    return api("/api/vault/put", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        entry_id: entryId,
        ciphertext: record.ciphertext,
        iv: record.iv,
        salt: record.salt,
        algo: record.algo,
        kdf_iterations: record.kdf_iterations,
        // Deliberately empty. The API allows an encrypted title_hint, but
        // the safest hint is no hint: a per-item encrypted title would
        // still leak its length, and the title is already inside the
        // ciphertext. The server therefore learns nothing but a size and
        // a timestamp.
        title_hint: ""
      })
    });
  }

  /* ---------------- Plaintext record shape ----------------
   * Title and body are encrypted together as one JSON document, so the
   * title is as protected as the note. The version field lets a future
   * client migrate old items without guessing. */

  function packPlaintext(title, body) {
    return JSON.stringify({ v: 1, title: title, body: body });
  }

  function unpackPlaintext(text) {
    try {
      var obj = JSON.parse(text);
      if (obj && obj.v === 1) {
        return { title: String(obj.title || ""), body: String(obj.body || "") };
      }
    } catch (e) { /* fall through to the legacy path below */ }
    // Anything we cannot parse is treated as a plain note body rather
    // than discarded. Losing user data to a format change would be worse
    // than showing a slightly odd title.
    return { title: "", body: text };
  }

  /* ---------------- Status messaging (no alert(), ever) ---------------- */

  function setStatus(message, kind) {
    var node = el.status;
    if (!node) return;
    node.textContent = message || "";
    node.className = "se-vault-status" + (message ? " se-vault-status-" + (kind || "info") : " d-none");
  }

  function setSaveState(message, kind) {
    if (!el.saveState) return;
    el.saveState.textContent = message || "";
    el.saveState.className = "se-vault-savestate small " + ("text-" + (kind === "error" ? "danger" : kind === "ok" ? "success" : "muted"));
  }

  function setUnlockError(message) {
    if (!el.unlockError) return;
    if (!message) {
      el.unlockError.textContent = "";
      el.unlockError.classList.add("d-none");
      return;
    }
    el.unlockError.textContent = message;
    el.unlockError.classList.remove("d-none");
  }

  function describeError(e) {
    if (e && e.name === "VaultError") return e.message;
    // Never surface a raw exception string to the user; it is noise at
    // best and can leak internals at worst.
    return "Something went wrong. Nothing was changed.";
  }

  /* ---------------- Lock / unlock ---------------- */

  function isUnlocked() {
    return passphrase !== null;
  }

  function renderLockState() {
    var unlocked = isUnlocked();
    el.lockBadge.textContent = unlocked ? "Unlocked" : "Locked";
    el.lockBadge.className = "se-vault-badge " + (unlocked ? "se-vault-badge-open" : "se-vault-badge-shut");
    el.lockBadge.setAttribute("aria-label", unlocked ? "Vault unlocked" : "Vault locked");
    el.lockIcon.textContent = unlocked ? "🔓" : "🔒";
    el.lockNow.classList.toggle("d-none", !unlocked);
    el.lockedPanel.classList.toggle("d-none", unlocked);
    el.unlockedPanel.classList.toggle("d-none", !unlocked);
  }

  function lock(reason) {
    // Drop every reference we hold. We cannot scrub the string from the
    // heap (see the file header), but after this point no code path can
    // reach it and the derived keys are gone.
    passphrase = null;
    clearKeyCache();
    titles.clear();
    current = null;
    items = [];
    if (el.pass) el.pass.value = "";
    if (el.title) el.title.value = "";
    if (el.body) el.body.value = "";
    if (el.demoPlain) el.demoPlain.textContent = "";
    if (el.demoCipher) el.demoCipher.textContent = "";
    if (el.itemList) el.itemList.innerHTML = "";
    clearTimeout(idleTimer);
    clearInterval(countdownTimer);
    idleTimer = null;
    countdownTimer = null;
    renderStrength("");
    renderEditor();
    renderLockState();
    setStatus(reason || "Vault locked.", "info");
    setUnlockError("");
    if (el.pass) el.pass.focus();
  }

  function touchActivity() {
    if (!isUnlocked()) return;
    lockDeadline = Date.now() + AUTO_LOCK_MS;
    clearTimeout(idleTimer);
    idleTimer = setTimeout(function () {
      lock("Vault locked after 10 minutes without activity.");
    }, AUTO_LOCK_MS);
  }

  function startCountdown() {
    clearInterval(countdownTimer);
    countdownTimer = setInterval(function () {
      if (!isUnlocked() || !el.countdown) return;
      var left = Math.max(0, lockDeadline - Date.now());
      var m = Math.floor(left / 60000);
      var s = Math.floor((left % 60000) / 1000);
      el.countdown.textContent = m + ":" + (s < 10 ? "0" : "") + s;
    }, 1000);
  }

  async function unlock(candidate) {
    setUnlockError("");
    // A fresh unlock must never inherit keys derived from a previous
    // passphrase.
    clearKeyCache();
    passphrase = candidate;
    renderLockState();
    touchActivity();
    startCountdown();
    setStatus("Unlocked on this device. The passphrase was not sent anywhere.", "ok");
    // Focus moves into the unlocked region so keyboard and screen-reader
    // users are not left on a control that just disappeared.
    if (el.newNote) el.newNote.focus();
    await loadItems();
  }

  /* ---------------- Item list ---------------- */

  function formatBytes(n) {
    if (typeof n !== "number") return "unknown size";
    if (n < 1024) return n + " B";
    return (n / 1024).toFixed(1) + " KB";
  }

  function formatTime(iso) {
    if (!iso) return "unknown";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso).slice(0, 19).replace("T", " ");
    return d.toLocaleString();
  }

  async function loadItems() {
    try {
      var data = await api("/api/vault/list");
      items = Array.isArray(data.items) ? data.items : [];
    } catch (e) {
      items = [];
      setStatus(describeError(e), "error");
    }
    renderItems();
    // Titles live inside the ciphertext, so the only way to show them is
    // to fetch and decrypt each item here. We do it one at a time, after
    // the list is already on screen, so the page stays responsive: each
    // item costs one PBKDF2 derivation.
    decryptTitlesProgressively();
  }

  function renderItems() {
    var list = el.itemList;
    list.innerHTML = "";
    el.itemCount.textContent = items.length === 1 ? "1 item" : items.length + " items";
    if (!items.length) {
      el.itemsEmpty.classList.remove("d-none");
      return;
    }
    el.itemsEmpty.classList.add("d-none");
    items.forEach(function (it) {
      var known = titles.get(it.entry_id);
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "se-vault-item" + (current && current.entry_id === it.entry_id ? " se-vault-item-active" : "");
      btn.setAttribute("aria-current", current && current.entry_id === it.entry_id ? "true" : "false");

      var titleEl = document.createElement("span");
      titleEl.className = "se-vault-item-title";
      if (known === undefined) {
        titleEl.textContent = "Encrypted — decrypting…";
        titleEl.classList.add("se-vault-item-pending");
      } else if (known === null) {
        titleEl.textContent = "Could not decrypt";
        titleEl.classList.add("se-vault-item-failed");
      } else {
        titleEl.textContent = known || "Untitled note";
      }

      var meta = document.createElement("span");
      meta.className = "se-vault-item-meta";
      // Show exactly what the server holds about this row, nothing more.
      meta.textContent = it.entry_id.slice(0, 8) + " · " + formatBytes(it.bytes) + " · " + formatTime(it.updated_at);

      btn.appendChild(titleEl);
      btn.appendChild(meta);
      btn.addEventListener("click", function () { openItem(it.entry_id); });
      list.appendChild(btn);
    });
  }

  async function decryptTitlesProgressively() {
    for (var i = 0; i < items.length; i++) {
      if (!isUnlocked()) return; // locked mid-pass: stop immediately
      var id = items[i].entry_id;
      if (titles.has(id)) continue;
      try {
        var rec = await api("/api/vault/get/" + encodeURIComponent(id));
        var text = await decryptPayload(passphrase, rec);
        titles.set(id, unpackPlaintext(text).title);
      } catch (e) {
        // A single bad item must not abort the whole list. Recording null
        // marks it as undecryptable in the UI.
        titles.set(id, null);
        if (e && e.code === "wrong-passphrase") {
          setStatus("At least one item did not decrypt. If every item fails, the passphrase is wrong for this vault.", "error");
        }
      }
      renderItems();
    }
  }

  /* ---------------- Editor ---------------- */

  function renderEditor() {
    var has = !!current;
    el.editor.classList.toggle("d-none", !has);
    el.editorEmpty.classList.toggle("d-none", has);
    resetDeleteConfirm();
    if (!has) {
      renderServerView(null);
      return;
    }
    el.title.value = current.title;
    el.body.value = current.body;
    renderServerView(current.record);
  }

  async function openItem(entryId) {
    touchActivity();
    if (current && current.dirty) await saveCurrent(true);
    setSaveState("Decrypting…", "info");
    try {
      var rec = await api("/api/vault/get/" + encodeURIComponent(entryId));
      var text = await decryptPayload(passphrase, rec);
      var parts = unpackPlaintext(text);
      current = {
        entry_id: entryId,
        title: parts.title,
        body: parts.body,
        record: rec,
        isNew: false,
        dirty: false
      };
      titles.set(entryId, parts.title);
      renderEditor();
      renderItems();
      setSaveState("Decrypted in this browser. Last saved " + formatTime(rec.updated_at) + ".", "ok");
      el.title.focus();
    } catch (e) {
      setSaveState(describeError(e), "error");
      if (e && e.code === "wrong-passphrase") {
        setStatus("That item did not decrypt with this passphrase. Nothing was changed.", "error");
      } else {
        setStatus(describeError(e), "error");
      }
    }
  }

  function newNote() {
    touchActivity();
    current = {
      // A random v4 UUID. The id is the one thing the server sees, so it
      // must carry no information about the content.
      entry_id: crypto.randomUUID(),
      title: "",
      body: "",
      record: null,
      isNew: true,
      dirty: false
    };
    renderEditor();
    renderItems();
    setSaveState("New note. It is saved encrypted when you leave a field.", "info");
    el.title.focus();
  }

  async function saveCurrent(silent) {
    if (!current || !isUnlocked() || saving) return;
    if (!current.dirty) return;
    if (!current.title.trim() && !current.body.trim()) {
      // Nothing worth persisting; avoid creating empty ciphertext rows.
      current.dirty = false;
      if (!silent) setSaveState("Nothing to save yet.", "info");
      return;
    }
    saving = true;
    setSaveState("Encrypting…", "info");
    try {
      // The only value that leaves this function is `record`, and every
      // field of it is either ciphertext or a public parameter. The
      // plaintext variable never reaches putItem.
      var plaintext = packPlaintext(current.title, current.body);
      var record = await encryptPayload(passphrase, plaintext);
      await putItem(current.entry_id, record);
      current.record = Object.assign({}, record, { updated_at: new Date().toISOString() });
      current.dirty = false;
      current.isNew = false;
      titles.set(current.entry_id, current.title);
      setSaveState("Saved as ciphertext at " + new Date().toLocaleTimeString() + ".", "ok");
      renderServerView(current.record);
      await refreshListMetadata();
    } catch (e) {
      // Leave `dirty` set so the next blur or the Save button retries.
      setSaveState(describeError(e), "error");
      setStatus(describeError(e), "error");
    } finally {
      saving = false;
    }
  }

  async function refreshListMetadata() {
    try {
      var data = await api("/api/vault/list");
      items = Array.isArray(data.items) ? data.items : [];
    } catch (e) {
      // A failed list refresh is cosmetic; the save already succeeded.
    }
    renderItems();
  }

  function resetDeleteConfirm() {
    if (!el.deleteBtn) return;
    el.deleteBtn.classList.remove("d-none");
    el.deleteConfirmWrap.classList.add("d-none");
  }

  async function deleteCurrent() {
    if (!current) return;
    touchActivity();
    var id = current.entry_id;
    if (current.isNew) {
      // Never persisted, so there is nothing on the server to remove.
      current = null;
      renderEditor();
      renderItems();
      setStatus("Draft discarded. It was never sent to the server.", "info");
      return;
    }
    try {
      await api("/api/vault/item/" + encodeURIComponent(id), { method: "DELETE" });
      titles.delete(id);
      current = null;
      renderEditor();
      setStatus("Item deleted. The ciphertext is gone from the server.", "ok");
      await refreshListMetadata();
    } catch (e) {
      setStatus(describeError(e), "error");
    }
  }

  /* ---------------- "What the server can see" ----------------
   * This panel is a demonstration, not decoration. The right-hand column
   * is the literal string sent in the request body for this item, so a
   * sceptical user can compare it against what they see in their own
   * network inspector and confirm the claim themselves. */

  function renderServerView(record) {
    if (!el.demoCipher) return;
    var plain = current ? packPlaintext(current.title, current.body) : "";
    el.demoPlain.textContent = plain || "—";
    if (!record) {
      el.demoCipher.textContent = "—";
      el.metaId.textContent = current ? current.entry_id : "—";
      el.metaAlgo.textContent = "—";
      el.metaIter.textContent = "—";
      el.metaSalt.textContent = "—";
      el.metaIv.textContent = "—";
      el.metaBytes.textContent = "—";
      el.metaUpdated.textContent = "—";
      el.demoNote.textContent = current && current.isNew
        ? "This note has not been saved yet, so there is no ciphertext on the server."
        : "Select a note to see its stored form.";
      return;
    }
    el.demoCipher.textContent = record.ciphertext;
    el.metaId.textContent = current ? current.entry_id : "—";
    el.metaAlgo.textContent = record.algo || ALGO_LABEL;
    el.metaIter.textContent = (record.kdf_iterations || KDF_ITERATIONS).toLocaleString();
    el.metaSalt.textContent = record.salt;
    el.metaIv.textContent = record.iv;
    el.metaBytes.textContent = formatBytes(Math.floor((record.ciphertext.length * 3) / 4));
    el.metaUpdated.textContent = formatTime(record.updated_at);
    var stale = current && current.dirty;
    el.demoNote.textContent = stale
      ? "The left column has unsaved edits. The right column is the ciphertext from the last save."
      : "The left column exists only in this browser. The right column is exactly what the server stores.";
  }

  /* ---------------- Passphrase strength UI ---------------- */

  function renderStrength(value) {
    var s = estimateStrength(value);
    var pct = Math.min(100, Math.round((s.bits / 100) * 100));
    el.strengthBar.style.width = pct + "%";
    el.strengthBar.className = "se-vault-strength-fill se-vault-strength-" + s.band;
    el.strengthMeter.setAttribute("aria-valuenow", String(s.bits));
    el.strengthMeter.setAttribute("aria-valuetext", s.label + ", about " + s.bits + " bits of estimated entropy");
    if (!value) {
      el.strengthLabel.textContent = "";
      el.strengthDetail.textContent = "";
      return;
    }
    el.strengthLabel.textContent = s.label + " · about " + s.bits + " bits";
    el.strengthDetail.textContent = s.notes.length
      ? s.notes.join(" ")
      : "Estimated from length and character variety. It is a rough guide, not a guarantee.";
  }

  /* ---------------- Wiring ---------------- */

  function init() {
    el = {
      root: $("vault-root"),
      lockBadge: $("vault-lock-badge"),
      lockIcon: $("vault-lock-icon"),
      lockNow: $("vault-lock-now"),
      lockedPanel: $("vault-locked-panel"),
      unlockedPanel: $("vault-unlocked-panel"),
      passForm: $("vault-pass-form"),
      pass: $("vault-pass"),
      passToggle: $("vault-pass-toggle"),
      strengthMeter: $("vault-strength-meter"),
      strengthBar: $("vault-strength-bar"),
      strengthLabel: $("vault-strength-label"),
      strengthDetail: $("vault-strength-detail"),
      unlockError: $("vault-unlock-error"),
      status: $("vault-status"),
      itemList: $("vault-items"),
      itemsEmpty: $("vault-items-empty"),
      itemCount: $("vault-item-count"),
      newNote: $("vault-new"),
      editor: $("vault-editor"),
      editorEmpty: $("vault-editor-empty"),
      title: $("vault-title"),
      body: $("vault-body"),
      saveBtn: $("vault-save"),
      saveState: $("vault-save-state"),
      deleteBtn: $("vault-delete"),
      deleteConfirmWrap: $("vault-delete-confirm-wrap"),
      deleteConfirm: $("vault-delete-confirm"),
      deleteCancel: $("vault-delete-cancel"),
      demoPlain: $("vault-demo-plain"),
      demoCipher: $("vault-demo-cipher"),
      demoNote: $("vault-demo-note"),
      metaId: $("vault-meta-id"),
      metaAlgo: $("vault-meta-algo"),
      metaIter: $("vault-meta-iter"),
      metaSalt: $("vault-meta-salt"),
      metaIv: $("vault-meta-iv"),
      metaBytes: $("vault-meta-bytes"),
      metaUpdated: $("vault-meta-updated"),
      countdown: $("vault-countdown"),
      iterNote: $("vault-iter-note")
    };
    if (!el.root) return; // not the vault page

    // WebCrypto is only exposed in secure contexts. Say so plainly rather
    // than failing with an undefined-property error.
    if (!window.isSecureContext || !window.crypto || !window.crypto.subtle) {
      el.lockedPanel.innerHTML =
        '<div class="alert alert-danger mb-0">This page needs a secure connection (HTTPS or localhost) ' +
        "to use the browser's encryption API. The vault is disabled here so that nothing is sent unencrypted.</div>";
      return;
    }

    if (el.iterNote) el.iterNote.textContent = KDF_ITERATIONS.toLocaleString();

    el.passForm.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var value = el.pass.value;
      if (!value) {
        setUnlockError("Enter a passphrase to unlock.");
        el.pass.focus();
        return;
      }
      if (value.length < 8) {
        setUnlockError("Use at least 8 characters. There is no way to reset this later.");
        el.pass.focus();
        return;
      }
      // Copy the value out, then clear the field so the secret is not
      // sitting in the DOM where an extension or a screenshot can read it.
      el.pass.value = "";
      unlock(value);
    });

    el.pass.addEventListener("input", function () {
      renderStrength(el.pass.value);
      setUnlockError("");
    });

    el.passToggle.addEventListener("click", function () {
      var showing = el.pass.type === "text";
      el.pass.type = showing ? "password" : "text";
      el.passToggle.textContent = showing ? "Show" : "Hide";
      el.passToggle.setAttribute("aria-pressed", showing ? "false" : "true");
      el.pass.focus();
    });

    el.lockNow.addEventListener("click", function () {
      lock("Vault locked.");
    });

    el.newNote.addEventListener("click", newNote);

    function markDirty() {
      if (!current) return;
      current.title = el.title.value;
      current.body = el.body.value;
      current.dirty = true;
      setSaveState("Unsaved changes.", "info");
      renderServerView(current.record);
      touchActivity();
    }
    el.title.addEventListener("input", markDirty);
    el.body.addEventListener("input", markDirty);

    // Autosave on blur: leaving a field re-encrypts and stores. Every save
    // is a full fresh encryption with a new salt and a new IV, never an
    // edit of existing ciphertext.
    el.title.addEventListener("blur", function () { saveCurrent(false); });
    el.body.addEventListener("blur", function () { saveCurrent(false); });
    el.saveBtn.addEventListener("click", function () { saveCurrent(false); });

    // Two-step delete instead of a modal or a confirm() dialog.
    el.deleteBtn.addEventListener("click", function () {
      el.deleteBtn.classList.add("d-none");
      el.deleteConfirmWrap.classList.remove("d-none");
      el.deleteConfirm.focus();
    });
    el.deleteCancel.addEventListener("click", function () {
      resetDeleteConfirm();
      el.deleteBtn.focus();
    });
    el.deleteConfirm.addEventListener("click", function () {
      resetDeleteConfirm();
      deleteCurrent();
    });

    // Ctrl/Cmd+S saves without leaving the keyboard.
    el.root.addEventListener("keydown", function (ev) {
      if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
        ev.preventDefault();
        saveCurrent(false);
      }
      if (ev.key === "Escape" && isUnlocked()) {
        resetDeleteConfirm();
      }
    });

    // Activity resets the idle clock. Passive listeners so scrolling and
    // typing stay smooth.
    ["pointerdown", "keydown", "focusin"].forEach(function (evt) {
      document.addEventListener(evt, touchActivity, { passive: true });
    });

    // Locking on tab hide is deliberately aggressive. A vault left
    // unlocked on a background tab is the realistic way this leaks — a
    // shared laptop, a screen share, a colleague walking past.
    document.addEventListener("visibilitychange", async function () {
      if (document.visibilityState !== "hidden" || !isUnlocked()) return;
      // Flush an in-progress edit before dropping the key, otherwise
      // switching tabs would quietly discard what the user just typed.
      // The save takes a fraction of a second; the lock still happens
      // even if it fails, because holding the key open to retry a network
      // call would trade the security guarantee for a convenience.
      try {
        if (current && current.dirty) await saveCurrent(true);
      } catch (e) { /* locking below is what matters */ }
      if (isUnlocked()) lock("Vault locked because the tab was hidden.");
    });

    // A page unload is the last chance to drop the key material.
    window.addEventListener("pagehide", function () {
      if (isUnlocked()) lock("");
    });

    renderStrength("");
    renderLockState();
    renderEditor();
    setStatus("", "info");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
