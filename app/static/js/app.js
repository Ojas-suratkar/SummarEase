// Small progressive-enhancement layer for the "Explain keyword" and
// "Listen to summary" buttons. No build step / framework needed.

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
      speakBtn.textContent = "🔊 Listen";
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
      if (data.error) {
        output.textContent = data.error;
      } else {
        output.textContent =
          `${data.label} (average score ${data.average_score} across ${data.sample_size} comments)`;
      }
    } catch (err) {
      output.textContent = "Something went wrong analyzing comments.";
    } finally {
      sentimentBtn.disabled = false;
    }
  }
});
