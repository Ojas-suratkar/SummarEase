// Progressive-enhancement layer -- no build step / framework needed.
// Every form here still works as a classic full-page POST if JavaScript
// is unavailable (routes.py renders the same result server-side); this
// file adds a faster, live-progress path on top when JS is available.

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

// ---------------------------------------------------------------------
// Client-side rendering of a summarize-job result -- mirrors
// _result.html (and youtube.html's extra blocks) closely enough that
// the existing delegated click/submit handlers below "just work" on the
// HTML this injects.
// ---------------------------------------------------------------------

function renderResultHTML(data) {
  if (!data || !data.summary) return "";

  let html = `<div class="card mt-4 shadow-sm result-card">
    <div class="card-body">
      <h5 class="card-title d-flex justify-content-between align-items-center">
        Summary
        <button type="button" class="btn btn-sm btn-outline-secondary speak-btn" data-text="${escapeHtml(data.summary)}">Listen</button>
      </h5>`;

  if (data.degraded) {
    html += `<div class="alert alert-warning small py-2 mb-2">⚠ Gemini was unavailable, so this is a local extractive summary instead of the usual AI-written one -- accurate but less polished. Try again shortly for the full version.</div>`;
  }
  if (data.used_textrank) {
    html += `<p class="text-muted small mb-2">Long source -- filtered to its key sentences before summarizing.</p>`;
  }

  html += `<p class="card-text">${escapeHtml(data.summary)}</p><audio class="w-100 mt-2 d-none" controls></audio>`;

  if (data.keywords && data.keywords.length) {
    html += `<hr><h6>Keywords</h6><div class="d-flex flex-wrap gap-2">`;
    data.keywords.forEach((kw) => {
      html += `<button type="button" class="btn btn-sm btn-outline-primary explain-btn" data-term="${escapeHtml(kw)}">${escapeHtml(kw)}</button>`;
    });
    html += `</div><div class="explain-output mt-2 small text-muted"></div>`;
  }
  html += `</div></div>`;

  if (data.method) {
    const note =
      data.method === "transcript"
        ? "Summarized from the video's captions."
        : "No captions were available -- summarized by having Gemini watch the video directly.";
    html += `<p class="text-muted small">${note}</p>`;
  }
  if (data.video_id) {
    html += `<div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title d-flex justify-content-between align-items-center">
          Comment sentiment
          <button type="button" class="btn btn-sm btn-outline-secondary sentiment-btn" data-video-id="${escapeHtml(data.video_id)}">Analyze comments</button>
        </h6>
        <div class="sentiment-output small text-muted"></div>
      </div>
    </div>`;
  }

  if (data.doc_id) {
    html += `<div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title">Ask this document</h6>
        <p class="text-muted small">Answers come only from this source -- Gemini says so if it can't find the answer, instead of guessing.</p>
        <form class="ask-form d-flex gap-2" data-doc-id="${escapeHtml(data.doc_id)}">
          <input type="text" class="form-control form-control-sm ask-input" placeholder="Ask a question about this source..." required>
          <button type="submit" class="btn btn-sm btn-primary text-nowrap">Ask</button>
        </form>
        <div class="ask-output mt-2 small"></div>
      </div>
    </div>

    <div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title d-flex justify-content-between align-items-center">
          Credibility &amp; framing lens
          <button type="button" class="btn btn-sm btn-outline-secondary lens-btn" data-doc-id="${escapeHtml(data.doc_id)}">Analyze</button>
        </h6>
        <p class="text-muted small mb-2">Scores the source's language -- loaded wording, emotional tone, reading level. Flagged claims link out for you to verify.</p>
        <div class="lens-output small"></div>
      </div>
    </div>

    <div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title d-flex justify-content-between align-items-center">
          Faithfulness check
          <button type="button" class="btn btn-sm btn-outline-secondary faithfulness-btn" data-doc-id="${escapeHtml(data.doc_id)}" data-summary="${escapeHtml(data.summary)}">Check</button>
        </h6>
        <p class="text-muted small mb-2">Gemini re-checks its own summary against the source and flags anything unsupported.</p>
        <div class="faithfulness-output small"></div>
      </div>
    </div>

    <div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title d-flex justify-content-between align-items-center">
          Spaced-repetition flashcards
          <button type="button" class="btn btn-sm btn-outline-secondary flashcards-btn" data-doc-id="${escapeHtml(data.doc_id)}" data-source-ref="${escapeHtml(data.source_ref || "")}">Generate flashcards</button>
        </h6>
        <p class="text-muted small mb-2">Turns this source into a few flashcards, scheduled with the same SM-2 algorithm behind Anki. Review them any time on the <a href="/review">Review</a> page.</p>
        <div class="flashcards-output small"></div>
      </div>
    </div>

    <div class="card mt-3 shadow-sm">
      <div class="card-body">
        <h6 class="card-title d-flex justify-content-between align-items-center">
          Steelmanned perspectives
          <button type="button" class="btn btn-sm btn-outline-secondary perspectives-btn" data-doc-id="${escapeHtml(data.doc_id)}">View perspectives</button>
        </h6>
        <p class="text-muted small mb-2">For contested topics, generates the strongest good-faith version of each major viewpoint -- plus the assumption it rests on and what would change its mind.</p>
        <div class="perspectives-output small"></div>
      </div>
    </div>`;
  }

  return html;
}

function renderProgressHTML(steps) {
  const items = (steps || [])
    .map((s) => `<li class="list-group-item small">${escapeHtml(s)}</li>`)
    .join("");
  return `<div class="card mt-4 shadow-sm progress-card">
    <div class="card-body">
      <div class="d-flex align-items-center gap-2 mb-2">
        <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
        <strong>Working...</strong>
      </div>
      <ul class="list-group list-group-flush">${items}</ul>
    </div>
  </div>`;
}

async function pollJob(jobId, container) {
  for (;;) {
    let res, data;
    try {
      res = await fetch(`/api/jobs/${jobId}`);
      data = await res.json();
    } catch (err) {
      container.innerHTML = `<div class="alert alert-danger mt-4">Lost connection while checking progress.</div>`;
      return;
    }

    if (!res.ok) {
      container.innerHTML = `<div class="alert alert-danger mt-4">${escapeHtml(data.error || "This job could not be found.")}</div>`;
      return;
    }

    if (data.status === "done") {
      container.innerHTML = renderResultHTML(data.result);
      return;
    }
    if (data.status === "error") {
      container.innerHTML = `<div class="alert alert-danger mt-4">${escapeHtml(data.error)}</div>`;
      return;
    }

    container.innerHTML = renderProgressHTML(data.progress);
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

document.querySelectorAll("form[data-job-kind]").forEach((form) => {
  form.addEventListener("submit", async (event) => {
    const container = document.getElementById("result-container");
    if (!container || typeof fetch !== "function" || typeof FormData !== "function") {
      // No JS support for the fast path -- let the classic full-page
      // POST (routes.py) handle it, exactly as if this listener weren't here.
      return;
    }
    event.preventDefault();

    const submitBtn = form.querySelector('button[type="submit"]');
    const originalLabel = submitBtn ? submitBtn.textContent : "";
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Working...";
    }

    const formData = new FormData(form);
    formData.set("kind", form.dataset.jobKind);
    container.innerHTML = renderProgressHTML(["Starting..."]);

    try {
      const res = await fetch("/api/jobs/start", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || data.error) {
        container.innerHTML = `<div class="alert alert-danger mt-4">${escapeHtml(data.error || "Something went wrong.")}</div>`;
      } else {
        await pollJob(data.job_id, container);
      }
    } catch (err) {
      container.innerHTML = `<div class="alert alert-danger mt-4">Something went wrong starting this job.</div>`;
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = originalLabel;
      }
    }
  });
});

// ---------------------------------------------------------------------
// Delegated click/submit handlers -- work on both server-rendered and
// client-rendered (renderResultHTML) markup, since they match on class
// names rather than a specific DOM subtree.
// ---------------------------------------------------------------------

function renderCredibility(data) {
  const ll = data.loaded_language;
  const readability = data.readability;

  const termsLine = (label, terms) =>
    terms && terms.length
      ? `<div><span class="text-muted">${label}:</span> ${terms.map(escapeHtml).join(", ")}</div>`
      : "";

  const claimsHtml =
    data.claims && data.claims.length
      ? "<ul class=\"mb-0\">" +
        data.claims
          .map(
            (c) =>
              `<li>${escapeHtml(c.text)} — <a href="${c.search_url}" target="_blank" rel="noopener">verify</a></li>`
          )
          .join("") +
        "</ul>"
      : `<p class="mb-0 text-muted">${escapeHtml(data.claims_error || "No distinct factual claims found.")}</p>`;

  return `
    <div class="row g-2 mb-2">
      <div class="col-sm-4"><span class="text-muted">Loaded language:</span> <strong>${ll.level}</strong> (${ll.density_per_100_words}/100 words)</div>
      <div class="col-sm-4"><span class="text-muted">Emotional tone:</span> <strong>${data.sentiment.tone}</strong></div>
      <div class="col-sm-4"><span class="text-muted">Reading level:</span> <strong>${readability.level}</strong> (grade ${readability.flesch_kincaid_grade ?? "-"})</div>
    </div>
    ${termsLine("Absolutist wording", ll.absolutist_terms)}
    ${termsLine("Emotionally loaded", ll.emotional_terms)}
    ${termsLine("Hedging phrases", ll.hedging_phrases)}
    <div class="mt-2"><span class="text-muted">Claims worth verifying:</span></div>
    ${claimsHtml}
  `;
}

function renderFaithfulness(data) {
  if (data.faithful) {
    return `<span class="text-success">✓ Every claim in the summary is supported by the source.</span>`;
  }
  const items = data.unsupported_claims
    .map((c) => `<li>${escapeHtml(c)}</li>`)
    .join("");
  return `<span class="text-warning">⚠ Possibly unsupported:</span><ul class="mb-0 mt-1">${items}</ul>`;
}

function renderPerspectives(data) {
  if (!data.is_contested || !data.perspectives || !data.perspectives.length) {
    return `<p class="mb-0 text-muted">This source doesn't appear to engage a genuinely contested question -- no manufactured debate here.</p>`;
  }
  const cards = data.perspectives
    .map(
      (p) => `
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body">
            <h6 class="card-title">${escapeHtml(p.label)}</h6>
            <p class="small mb-2">${escapeHtml(p.steelman)}</p>
            <p class="small text-muted mb-1"><strong>Key assumption:</strong> ${escapeHtml(p.key_assumption)}</p>
            <p class="small text-muted mb-0"><strong>Would change its mind:</strong> ${escapeHtml(p.would_change_mind)}</p>
          </div>
        </div>
      </div>`
    )
    .join("");
  return `<p class="mb-2"><strong>Contested question:</strong> ${escapeHtml(data.topic)}</p><div class="row g-2">${cards}</div>`;
}

function renderAnnotations(output, entryId, list) {
  const items = list
    .map(
      (a) => `
      <div class="border-start border-3 ps-2 mb-2" style="border-color: var(--se-accent) !important;">
        ${a.quote ? `<div class="fst-italic text-muted mb-1">"${escapeHtml(a.quote)}"</div>` : ""}
        <div>${escapeHtml(a.note)}</div>
        <button type="button" class="btn btn-sm btn-link text-danger p-0 delete-annotation-btn" data-annotation-id="${a.id}" data-entry-id="${entryId}">remove</button>
      </div>`
    )
    .join("");
  output.innerHTML = `
    ${items || '<p class="text-muted mb-2">No annotations yet.</p>'}
    <form class="annotation-form d-flex flex-column gap-1 mt-2" data-entry-id="${entryId}">
      <input type="text" class="form-control form-control-sm annotation-quote" placeholder="Quote (optional)">
      <div class="d-flex gap-2">
        <input type="text" class="form-control form-control-sm annotation-note" placeholder="Your note..." required>
        <button type="submit" class="btn btn-sm btn-primary text-nowrap">Save</button>
      </div>
    </form>`;
}

function relationshipBadgeClass(rel) {
  return { supports: "text-bg-success", contradicts: "text-bg-danger", updates: "text-bg-info" }[rel] || "text-bg-secondary";
}

document.addEventListener("click", async (event) => {
  const flashcardsBtn = event.target.closest(".flashcards-btn");
  if (flashcardsBtn) {
    const docId = flashcardsBtn.dataset.docId;
    const sourceRef = flashcardsBtn.dataset.sourceRef || "this document";
    const output = flashcardsBtn.closest(".card-body").querySelector(".flashcards-output");
    flashcardsBtn.disabled = true;
    output.textContent = "Generating flashcards...";
    try {
      const res = await fetch("/api/flashcards/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, source_ref: sourceRef }),
      });
      const data = await res.json();
      output.innerHTML = data.error
        ? `<span class="text-danger">${escapeHtml(data.error)}</span>`
        : `<span class="text-success">✓ ${data.count} flashcard${data.count === 1 ? "" : "s"} added -- review them on the <a href="/review">Review</a> page.</span>`;
    } catch (err) {
      output.textContent = "Something went wrong generating flashcards.";
    } finally {
      flashcardsBtn.disabled = false;
    }
    return;
  }

  const perspectivesBtn = event.target.closest(".perspectives-btn");
  if (perspectivesBtn) {
    const docId = perspectivesBtn.dataset.docId;
    const output = perspectivesBtn.closest(".card-body").querySelector(".perspectives-output");
    perspectivesBtn.disabled = true;
    output.innerHTML = `<span class="text-muted">Finding the strongest version of each viewpoint...</span>`;
    try {
      const res = await fetch("/api/perspectives", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId }),
      });
      const data = await res.json();
      output.innerHTML = data.error
        ? `<span class="text-danger">${escapeHtml(data.error)}</span>`
        : renderPerspectives(data);
    } catch (err) {
      output.textContent = "Something went wrong generating perspectives.";
    } finally {
      perspectivesBtn.disabled = false;
    }
    return;
  }

  const relatedBtn = event.target.closest(".related-btn");
  if (relatedBtn) {
    const entryId = relatedBtn.dataset.entryId;
    const output = relatedBtn.parentElement.querySelector(".related-output");
    const wasShowing = relatedBtn.dataset.showing === "1";
    if (wasShowing) {
      output.innerHTML = "";
      relatedBtn.dataset.showing = "0";
      relatedBtn.textContent = "Related";
      return;
    }
    relatedBtn.disabled = true;
    output.innerHTML = `<span class="text-muted">Loading related entries...</span>`;
    try {
      const res = await fetch(`/api/graph/${entryId}`);
      const data = await res.json();
      if (!data.links || !data.links.length) {
        output.innerHTML = `<span class="text-muted">No auto-detected links yet.</span>`;
      } else {
        output.innerHTML = data.links
          .map(
            (l) => `<div class="mb-1"><span class="badge ${relationshipBadgeClass(l.relationship)}">${escapeHtml(l.relationship)}</span> ${escapeHtml(l.source_ref || l.source_type)} -- <span class="text-muted">${escapeHtml((l.summary || "").slice(0, 140))}${(l.summary || "").length > 140 ? "..." : ""}</span></div>`
          )
          .join("");
        relatedBtn.dataset.showing = "1";
        relatedBtn.textContent = "✕ Hide related";
      }
    } catch (err) {
      output.innerHTML = `<span class="text-danger">Something went wrong loading related entries.</span>`;
    } finally {
      relatedBtn.disabled = false;
    }
    return;
  }

  const annotateBtn = event.target.closest(".annotate-btn");
  if (annotateBtn) {
    const entryId = annotateBtn.dataset.entryId;
    const output = annotateBtn.closest(".list-group-item").querySelector(".annotate-output");
    const wasShowing = annotateBtn.dataset.showing === "1";
    if (wasShowing) {
      output.innerHTML = "";
      annotateBtn.dataset.showing = "0";
      annotateBtn.textContent = "Annotate";
      return;
    }
    annotateBtn.dataset.showing = "1";
    annotateBtn.textContent = "✕ Hide annotations";
    output.innerHTML = `<div class="text-muted">Loading...</div>`;
    try {
      const res = await fetch(`/api/annotations/${entryId}`);
      const data = await res.json();
      renderAnnotations(output, entryId, data.annotations || []);
    } catch (err) {
      output.innerHTML = `<span class="text-danger">Something went wrong loading annotations.</span>`;
    }
    return;
  }

  const deleteAnnotationBtn = event.target.closest(".delete-annotation-btn");
  if (deleteAnnotationBtn) {
    const annotationId = deleteAnnotationBtn.dataset.annotationId;
    const entryId = deleteAnnotationBtn.dataset.entryId;
    const output = deleteAnnotationBtn.closest(".annotate-output");
    await fetch(`/api/annotations/${annotationId}/delete`, { method: "POST" });
    const res = await fetch(`/api/annotations/${entryId}`);
    const data = await res.json();
    renderAnnotations(output, entryId, data.annotations || []);
    return;
  }

  const shareBtn = event.target.closest(".share-btn");
  if (shareBtn) {
    const entryId = shareBtn.dataset.entryId;
    const output = shareBtn.closest(".list-group-item").querySelector(".share-output");
    shareBtn.disabled = true;
    output.textContent = "Creating link...";
    try {
      const res = await fetch(`/api/share/${entryId}`, { method: "POST" });
      const data = await res.json();
      if (data.error) {
        output.innerHTML = `<span class="text-danger">${escapeHtml(data.error)}</span>`;
      } else {
        output.innerHTML = `<input type="text" class="form-control form-control-sm d-inline-block" style="width: auto; max-width: 100%;" readonly value="${escapeHtml(data.url)}" onclick="this.select()">`;
      }
    } catch (err) {
      output.textContent = "Something went wrong creating that link.";
    } finally {
      shareBtn.disabled = false;
    }
    return;
  }

  const watchCheckBtn = event.target.closest(".watch-check-btn");
  if (watchCheckBtn) {
    const watchId = watchCheckBtn.dataset.watchId;
    const originalLabel = watchCheckBtn.textContent;
    watchCheckBtn.disabled = true;
    watchCheckBtn.textContent = "Checking...";
    try {
      const res = await fetch(`/api/watchlist/${watchId}/check`, { method: "POST" });
      const data = await res.json();
      if (data.error) {
        watchCheckBtn.textContent = "Error";
      } else if (data.changed) {
        watchCheckBtn.textContent = "Changed!";
        setTimeout(() => window.location.reload(), 900);
        return;
      } else {
        watchCheckBtn.textContent = "No change";
      }
      setTimeout(() => {
        watchCheckBtn.textContent = originalLabel;
        watchCheckBtn.disabled = false;
      }, 1800);
    } catch (err) {
      watchCheckBtn.textContent = "Error";
      watchCheckBtn.disabled = false;
    }
    return;
  }

  const sortHeader = event.target.closest(".se-sortable");
  if (sortHeader) {
    const table = sortHeader.closest("table");
    const col = parseInt(sortHeader.dataset.col, 10);
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const ascending = sortHeader.dataset.dir !== "asc";
    table.querySelectorAll(".se-sortable").forEach((h) => delete h.dataset.dir);
    sortHeader.dataset.dir = ascending ? "asc" : "desc";
    rows.sort((a, b) => {
      const av = a.children[col].textContent.trim().toLowerCase();
      const bv = b.children[col].textContent.trim().toLowerCase();
      return ascending ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach((r) => tbody.appendChild(r));
    return;
  }

});

document.addEventListener("click", async (event) => {
  const explainBtn = event.target.closest(".explain-btn");
  if (explainBtn) {
    const term = explainBtn.dataset.term;
    const output = explainBtn.closest(".card-body").querySelector(".explain-output");
    output.textContent = "Looking that up...";
    try {
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ term }),
      });
      const data = await res.json();
      output.textContent = data.explanation || data.error || "No explanation available.";
    } catch (err) {
      output.textContent = "Something went wrong looking that up.";
    }
    return;
  }

  const speakBtn = event.target.closest(".speak-btn");
  if (speakBtn) {
    const text = speakBtn.dataset.text;
    const audio = speakBtn.closest(".card-body").querySelector("audio");
    speakBtn.disabled = true;
    speakBtn.textContent = "Generating audio...";
    try {
      const res = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (data.audio_url) {
        audio.src = data.audio_url;
        audio.classList.remove("d-none");
        audio.play();
      }
    } catch (err) {
      // Fail quietly -- the summary text itself is still visible.
    } finally {
      speakBtn.disabled = false;
      speakBtn.textContent = "Listen";
    }
    return;
  }

  const sentimentBtn = event.target.closest(".sentiment-btn");
  if (sentimentBtn) {
    const videoId = sentimentBtn.dataset.videoId;
    const output = sentimentBtn.closest(".card-body").querySelector(".sentiment-output");
    sentimentBtn.disabled = true;
    output.textContent = "Fetching and scoring comments...";
    try {
      const res = await fetch("/api/youtube/sentiment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId }),
      });
      const data = await res.json();
      output.textContent = data.error
        ? data.error
        : `${data.label} (average score ${data.average_score} across ${data.sample_size} comments)`;
    } catch (err) {
      output.textContent = "Something went wrong analyzing comments.";
    } finally {
      sentimentBtn.disabled = false;
    }
    return;
  }

  const lensBtn = event.target.closest(".lens-btn");
  if (lensBtn) {
    const docId = lensBtn.dataset.docId;
    const output = lensBtn.closest(".card-body").querySelector(".lens-output");
    lensBtn.disabled = true;
    output.textContent = "Analyzing language, tone, and claims...";
    try {
      const res = await fetch("/api/credibility", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId }),
      });
      const data = await res.json();
      output.innerHTML = data.error
        ? `<span class="text-danger">${escapeHtml(data.error)}</span>`
        : renderCredibility(data);
    } catch (err) {
      output.textContent = "Something went wrong analyzing this.";
    } finally {
      lensBtn.disabled = false;
    }
    return;
  }

  const faithBtn = event.target.closest(".faithfulness-btn");
  if (faithBtn) {
    const docId = faithBtn.dataset.docId;
    const summary = faithBtn.dataset.summary;
    const output = faithBtn.closest(".card-body").querySelector(".faithfulness-output");
    faithBtn.disabled = true;
    output.textContent = "Re-checking the summary against the source...";
    try {
      const res = await fetch("/api/faithfulness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId, summary }),
      });
      const data = await res.json();
      output.innerHTML = data.error
        ? `<span class="text-danger">${escapeHtml(data.error)}</span>`
        : renderFaithfulness(data);
    } catch (err) {
      output.textContent = "Something went wrong checking this.";
    } finally {
      faithBtn.disabled = false;
    }
    return;
  }

  const topicsBtn = event.target.closest("#topics-btn");
  if (topicsBtn) {
    const output = document.getElementById("topics-output");
    topicsBtn.disabled = true;
    const wasShowing = topicsBtn.dataset.showing === "1";
    if (wasShowing) {
      output.innerHTML = "";
      topicsBtn.dataset.showing = "0";
      topicsBtn.textContent = "View by topic";
      topicsBtn.disabled = false;
      return;
    }
    output.innerHTML = `<div class="text-muted small">Clustering your knowledge base and labeling topics...</div>`;
    try {
      const res = await fetch("/api/topics");
      const data = await res.json();
      if (data.error) {
        output.innerHTML = `<div class="alert alert-danger">${escapeHtml(data.error)}</div>`;
      } else if (!data.clusters || !data.clusters.length) {
        output.innerHTML = `<p class="text-muted">Not enough in your knowledge base yet to find topics.</p>`;
      } else {
        output.innerHTML =
          `<div class="row g-3">` +
          data.clusters
            .map(
              (c) => `
              <div class="col-md-4">
                <div class="card h-100 topic-card">
                  <div class="card-body">
                    <h6 class="card-title d-flex justify-content-between align-items-center">
                      ${escapeHtml(c.label)}
                      <span class="badge text-bg-light text-muted">${c.items.length}</span>
                    </h6>
                    <ul class="list-unstyled small mb-0">
                      ${c.items.slice(0, 4).map((it) => `<li class="mb-1 text-truncate">${escapeHtml(it.summary)}</li>`).join("")}
                    </ul>
                  </div>
                </div>
              </div>`
            )
            .join("") +
          `</div>`;
        topicsBtn.dataset.showing = "1";
        topicsBtn.textContent = "✕ Hide topics";
      }
    } catch (err) {
      output.innerHTML = `<div class="alert alert-danger">Something went wrong finding topics.</div>`;
    } finally {
      topicsBtn.disabled = false;
    }
  }
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest(".ask-form");
  if (!form) return;
  event.preventDefault();

  const docId = form.dataset.docId;
  const input = form.querySelector(".ask-input");
  const question = input.value.trim();
  if (!question) return;

  const output = form.closest(".card-body").querySelector(".ask-output");
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  output.textContent = "Thinking...";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, question }),
    });
    const data = await res.json();
    if (data.error) {
      output.innerHTML = `<span class="text-danger">${escapeHtml(data.error)}</span>`;
    } else {
      const sourceNote = (data.passages || [])
        .map((p, i) => `[${i + 1}] similarity ${p.similarity}`)
        .join(" · ");
      output.innerHTML =
        `<p class="mb-1"><strong>Q:</strong> ${escapeHtml(question)}</p>` +
        `<p class="mb-1">${escapeHtml(data.answer)}</p>` +
        `<p class="text-muted mb-0">${sourceNote}</p>`;
    }
  } catch (err) {
    output.textContent = "Something went wrong answering that.";
  } finally {
    button.disabled = false;
  }
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest(".annotation-form");
  if (!form) return;
  event.preventDefault();

  const entryId = form.dataset.entryId;
  const quote = form.querySelector(".annotation-quote").value.trim();
  const note = form.querySelector(".annotation-note").value.trim();
  if (!note) return;

  try {
    await fetch("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entry_id: entryId, quote, note }),
    });
    const res = await fetch(`/api/annotations/${entryId}`);
    const data = await res.json();
    renderAnnotations(form.closest(".annotate-output"), entryId, data.annotations || []);
  } catch (err) {
    // Leave the form as-is so the user can retry.
  }
});

// Briefing mode: show only the input matching each row's selected source
// kind (text / URL / PDF), rather than three stacked fields at once.
function updateBriefingRow(select) {
  const row = select.dataset.row;
  const kind = select.value;
  document.querySelectorAll(`.source-input[data-row="${row}"]`).forEach((el) => {
    el.classList.toggle("d-none", el.dataset.kind !== kind);
  });
}

document.addEventListener("change", (event) => {
  const select = event.target.closest(".kind-select");
  if (select) updateBriefingRow(select);
});

document.querySelectorAll(".kind-select").forEach(updateBriefingRow);

// ---------------------------------------------------------------------
// Command palette (Cmd/Ctrl+K) -- instant client-side page navigation
// plus a live search over the knowledge base. SE_PAGES is the static
// page list; kept here (rather than server-rendered) so filtering it as
// you type needs no round trip at all.
// ---------------------------------------------------------------------

const SE_PAGES = [
  { label: "Text summarizer", url: "/text", group: "Summarize" },
  { label: "PDF summarizer", url: "/pdf", group: "Summarize" },
  { label: "YouTube summarizer", url: "/youtube", group: "Summarize" },
  { label: "Article summarizer", url: "/article", group: "Summarize" },
  { label: "Audio summarizer", url: "/audio", group: "Summarize" },
  { label: "Image / screenshot analysis", url: "/image", group: "Summarize" },
  { label: "Video summarizer", url: "/video", group: "Summarize" },
  { label: "Briefing mode", url: "/briefing", group: "Organize" },
  { label: "Compare sources", url: "/compare", group: "Organize" },
  { label: "Knowledge base", url: "/history", group: "Organize" },
  { label: "Search", url: "/search", group: "Organize" },
  { label: "Knowledge graph", url: "/graph", group: "Organize" },
  { label: "Duplicates", url: "/duplicates", group: "Organize" },
  { label: "Weekly digest", url: "/digest", group: "Organize" },
  { label: "Review flashcards", url: "/review", group: "Retain" },
  { label: "Watchlist", url: "/watchlist", group: "Retain" },
  { label: "My analytics", url: "/analytics", group: "Retain" },
  { label: "Compose / draft", url: "/compose", group: "Create" },
  { label: "Export center", url: "/export", group: "Create" },
  { label: "Dashboard", url: "/dashboard", group: "Admin" },
  { label: "Profile & account", url: "/profile", group: "Admin" },
  { label: "Home", url: "/", group: "Admin" },
];

(function () {
  const overlay = document.getElementById("cmdk-overlay");
  if (!overlay) return;
  const input = document.getElementById("cmdk-input");
  const resultsEl = document.getElementById("cmdk-results");
  let items = [];
  let activeIndex = 0;
  let searchTimer = null;

  function escapeHtmlLocal(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }

  function renderItems() {
    if (!items.length) {
      resultsEl.innerHTML = `<div class="se-palette-empty">No matches.</div>`;
      return;
    }
    resultsEl.innerHTML = items
      .map(
        (item, i) => `
        <div class="se-palette-item ${i === activeIndex ? "active" : ""}" data-index="${i}">
          <span class="se-palette-label">${escapeHtmlLocal(item.label)}</span>
          <span class="se-palette-meta">${escapeHtmlLocal(item.group)}</span>
        </div>`
      )
      .join("");
  }

  function open() {
    overlay.classList.remove("d-none");
    input.value = "";
    activeIndex = 0;
    items = SE_PAGES.slice(0, 8).map((p) => ({ label: p.label, group: p.group, url: p.url }));
    renderItems();
    setTimeout(() => input.focus(), 10);
  }

  function close() {
    overlay.classList.add("d-none");
  }

  async function search(query) {
    const q = query.trim().toLowerCase();
    const pageMatches = q
      ? SE_PAGES.filter((p) => p.label.toLowerCase().includes(q))
      : SE_PAGES.slice(0, 8);
    items = pageMatches.map((p) => ({ label: p.label, group: p.group, url: p.url }));
    activeIndex = 0;
    renderItems();

    if (!q) return;
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      const kbMatches = (data.results || []).map((r) => ({
        label: (r.summary || "").slice(0, 70),
        group: r.source_type,
        url: "/history",
      }));
      items = [...items, ...kbMatches];
      renderItems();
    } catch (err) {
      // Page matches are still shown even if the knowledge-base search fails.
    }
  }

  document.addEventListener("keydown", (event) => {
    const isOpen = !overlay.classList.contains("d-none");
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      isOpen ? close() : open();
      return;
    }
    if (!isOpen) return;
    if (event.key === "Escape") {
      close();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      renderItems();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      renderItems();
    } else if (event.key === "Enter") {
      event.preventDefault();
      const item = items[activeIndex];
      if (item) window.location.href = item.url;
    }
  });

  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => search(input.value), 150);
  });

  resultsEl.addEventListener("click", (event) => {
    const el = event.target.closest(".se-palette-item");
    if (!el) return;
    const item = items[parseInt(el.dataset.index, 10)];
    if (item) window.location.href = item.url;
  });

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  document.querySelectorAll(".se-cmdk-trigger").forEach((btn) => {
    btn.addEventListener("click", open);
  });

  document.querySelectorAll(".se-cmdk-trigger-card").forEach((card) => {
    card.addEventListener("click", (event) => {
      event.preventDefault();
      open();
    });
  });
})();

// Live nav badges -- knowledge-base count, unread watchlist digest items,
// and flashcards due for review. Plain counts, no emoji-as-icon: the
// labels next to them (in the dropdown items / badge title) already say
// what they are.
(async () => {
  const badge = document.getElementById("kb-stat");
  if (!badge) return;
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    badge.textContent = String(data.entry_count);
  } catch (err) {
    badge.textContent = "--";
  }
})();

(async () => {
  const badge = document.getElementById("watch-stat-inline");
  if (!badge) return;
  try {
    const res = await fetch("/api/watchlist/unread_count");
    const data = await res.json();
    badge.textContent = String(data.unread_count);
  } catch (err) {
    badge.textContent = "--";
  }
})();

(async () => {
  const badge = document.getElementById("review-stat-inline");
  if (!badge) return;
  try {
    const res = await fetch("/api/flashcards/due_count");
    const data = await res.json();
    badge.textContent = String(data.due_count);
  } catch (err) {
    badge.textContent = "--";
  }
})();
