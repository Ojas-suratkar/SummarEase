// In-browser mic/webcam/video capture -- no separate app, no plugin,
// just the standard MediaRecorder + getUserMedia APIs already built into
// every modern browser. A recorded clip is attached to the page's normal
// file <input> via the DataTransfer API, so the existing upload form
// (and the async /api/jobs path behind it) needs zero special-casing --
// a recording IS a file, exactly like one picked from disk.
(function () {
  function attachFile(input, blob, filename) {
    const file = new File([blob], filename, { type: blob.type || "application/octet-stream" });
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function stopStream(stream) {
    if (stream) stream.getTracks().forEach((t) => t.stop());
  }

  function initAudio(root) {
    const input = document.getElementById(root.dataset.targetInput);
    const startBtn = root.querySelector(".se-rec-start");
    const stopBtn = root.querySelector(".se-rec-stop");
    const status = root.querySelector(".se-rec-status");
    const audioPreview = root.querySelector("audio.se-rec-preview");
    let mediaRecorder, chunks = [], stream;

    startBtn.addEventListener("click", async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        status.textContent = "Couldn't access the microphone: " + err.message;
        return;
      }
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
        attachFile(input, blob, "recording.webm");
        if (audioPreview) {
          audioPreview.src = URL.createObjectURL(blob);
          audioPreview.classList.remove("d-none");
        }
        status.textContent = "Recorded -- ready to summarize.";
        stopStream(stream);
      };
      mediaRecorder.start();
      status.textContent = "Recording...";
      startBtn.classList.add("d-none");
      stopBtn.classList.remove("d-none");
    });

    stopBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
      startBtn.classList.remove("d-none");
      stopBtn.classList.add("d-none");
    });
  }

  function initImage(root) {
    const input = document.getElementById(root.dataset.targetInput);
    const liveVideo = root.querySelector("video.se-rec-live");
    const canvas = root.querySelector("canvas");
    const startBtn = root.querySelector(".se-rec-start");
    const captureBtn = root.querySelector(".se-rec-capture");
    const status = root.querySelector(".se-rec-status");
    const preview = root.querySelector("img.se-rec-preview");
    let stream;

    startBtn.addEventListener("click", async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (err) {
        status.textContent = "Couldn't access the webcam: " + err.message;
        return;
      }
      liveVideo.srcObject = stream;
      liveVideo.classList.remove("d-none");
      await liveVideo.play();
      startBtn.classList.add("d-none");
      captureBtn.classList.remove("d-none");
      status.textContent = "Camera on -- click Capture when ready.";
    });

    captureBtn.addEventListener("click", () => {
      canvas.width = liveVideo.videoWidth;
      canvas.height = liveVideo.videoHeight;
      canvas.getContext("2d").drawImage(liveVideo, 0, 0);
      canvas.toBlob((blob) => {
        attachFile(input, blob, "capture.png");
        if (preview) {
          preview.src = URL.createObjectURL(blob);
          preview.classList.remove("d-none");
        }
        status.textContent = "Captured -- ready to analyze.";
      }, "image/png");
      stopStream(stream);
      liveVideo.classList.add("d-none");
      captureBtn.classList.add("d-none");
      startBtn.classList.remove("d-none");
    });
  }

  function initVideo(root) {
    const input = document.getElementById(root.dataset.targetInput);
    const liveVideo = root.querySelector("video.se-rec-live");
    const preview = root.querySelector("video.se-rec-preview");
    const startBtn = root.querySelector(".se-rec-start");
    const stopBtn = root.querySelector(".se-rec-stop");
    const status = root.querySelector(".se-rec-status");
    let mediaRecorder, chunks = [], stream;

    startBtn.addEventListener("click", async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      } catch (err) {
        status.textContent = "Couldn't access the camera/microphone: " + err.message;
        return;
      }
      liveVideo.srcObject = stream;
      liveVideo.classList.remove("d-none");
      await liveVideo.play();
      chunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "" });
      mediaRecorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        attachFile(input, blob, "recording.webm");
        if (preview) {
          preview.src = URL.createObjectURL(blob);
          preview.classList.remove("d-none");
        }
        liveVideo.classList.add("d-none");
        status.textContent = "Recorded -- ready to summarize.";
        stopStream(stream);
      };
      mediaRecorder.start();
      status.textContent = "Recording...";
      startBtn.classList.add("d-none");
      stopBtn.classList.remove("d-none");
    });

    stopBtn.addEventListener("click", () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
      startBtn.classList.remove("d-none");
      stopBtn.classList.add("d-none");
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices || !window.MediaRecorder) return; // graceful no-op -- file upload still works
    document.querySelectorAll(".se-recorder[data-kind='audio']").forEach(initAudio);
    document.querySelectorAll(".se-recorder[data-kind='image']").forEach(initImage);
    document.querySelectorAll(".se-recorder[data-kind='video']").forEach(initVideo);
  });
})();
