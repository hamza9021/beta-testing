// ─────────────────────────────────────────────
//  app.js  —  Client-side camera capture + server YOLO detection
// ─────────────────────────────────────────────
//
//  Flow:
//    1. Ask browser for webcam via getUserMedia()
//    2. Every TARGET_INTERVAL_MS:
//         a. Draw video frame onto hidden captureCanvas
//         b. Encode to JPEG blob (toBlob)
//         c. POST blob to /detect
//         d. Decode response JPEG → draw onto displayCanvas
//         e. Read X-FPS / X-Total / X-Detections headers → update sidebar
// ─────────────────────────────────────────────

const TARGET_INTERVAL_MS = 100;  // max ~10 frames/sec sent to server
const CAPTURE_QUALITY = 0.7;  // JPEG quality for frames sent to server (0–1)
const MAX_FPS = 30;   // for FPS bar scaling

// ── DOM refs ──────────────────────────────────────────────────────────────────
const rawVideo = document.getElementById("rawVideo");
const captureCanvas = document.getElementById("captureCanvas");
const displayCanvas = document.getElementById("displayCanvas");
const camOverlay = document.getElementById("camOverlay");
const camOverlayTxt = document.getElementById("camOverlayText");
const startCamBtn = document.getElementById("startCamBtn");

const fpsValueEl = document.getElementById("fpsValue");
const fpsBarEl = document.getElementById("fpsBar");
const totalValueEl = document.getElementById("totalValue");
const confDisplay = document.getElementById("confDisplay");
const confSlider = document.getElementById("confSlider");
const lastUpdateEl = document.getElementById("lastUpdate");

const classCountEls = {
    "Mobile Phone": document.getElementById("cnt-phone"),
    "Laptop / Notebook": document.getElementById("cnt-laptop"),
    "Smart Watch": document.getElementById("cnt-watch"),
};
const classRowEls = {
    "Mobile Phone": document.getElementById("row-phone"),
    "Laptop / Notebook": document.getElementById("row-laptop"),
    "Smart Watch": document.getElementById("row-watch"),
};

// ── State ─────────────────────────────────────────────────────────────────────
let streaming = false;
let busy = false;   // guard: only one in-flight request at a time
let loopTimer = null;

// ── Camera init ───────────────────────────────────────────────────────────────
async function startCamera() {
    camOverlayTxt.textContent = "Requesting camera access…";
    startCamBtn.style.display = "none";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
            audio: false,
        });

        rawVideo.srcObject = stream;
        await new Promise(resolve => { rawVideo.onloadedmetadata = resolve; });

        // Size the display canvas to match video
        const vw = rawVideo.videoWidth || 1280;
        const vh = rawVideo.videoHeight || 720;
        captureCanvas.width = vw;
        captureCanvas.height = vh;
        displayCanvas.width = vw;
        displayCanvas.height = vh;

        // Hide overlay
        camOverlay.style.display = "none";
        streaming = true;

        // Draw placeholder until first result arrives
        const ctx = displayCanvas.getContext("2d");
        ctx.fillStyle = "#0a0a0f";
        ctx.fillRect(0, 0, vw, vh);

        // Kick off the detection loop
        scheduleNext();

    } catch (err) {
        console.error("Camera error:", err);
        camOverlayTxt.textContent = err.name === "NotAllowedError"
            ? "Camera permission denied. Please allow access and reload."
            : `Camera error: ${err.message}`;
        startCamBtn.style.display = "inline-block";
    }
}

// ── Detection loop ────────────────────────────────────────────────────────────
function scheduleNext() {
    if (!streaming) return;
    loopTimer = setTimeout(detectFrame, TARGET_INTERVAL_MS);
}

async function detectFrame() {
    if (!streaming || busy) { scheduleNext(); return; }
    busy = true;

    try {
        // 1. Capture frame from hidden video
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(rawVideo, 0, 0, captureCanvas.width, captureCanvas.height);

        // 2. Encode to JPEG blob
        const blob = await new Promise(resolve =>
            captureCanvas.toBlob(resolve, "image/jpeg", CAPTURE_QUALITY)
        );
        if (!blob) { busy = false; scheduleNext(); return; }

        // 3. POST to server
        const res = await fetch("/detect", {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: blob,
        });

        if (!res.ok) { busy = false; scheduleNext(); return; }

        // 4. Read stats from headers
        const fps = parseFloat(res.headers.get("X-FPS") || "0");
        const total = parseInt(res.headers.get("X-Total") || "0", 10);
        const detsRaw = res.headers.get("X-Detections") || "{}";

        // Parse the Python repr dict that comes back, e.g.
        // "{'Mobile Phone': 1, 'Laptop / Notebook': 2}"
        let dets = {};
        try {
            // Replace single quotes with double quotes for JSON.parse
            dets = JSON.parse(detsRaw.replace(/'/g, '"'));
        } catch (_) { }

        // 5. Decode annotated JPEG → draw on display canvas
        const arrayBuf = await res.arrayBuffer();
        const imgBlob = new Blob([arrayBuf], { type: "image/jpeg" });
        const bitmapUrl = URL.createObjectURL(imgBlob);
        const img = new Image();
        img.onload = () => {
            displayCanvas.getContext("2d").drawImage(img, 0, 0);
            URL.revokeObjectURL(bitmapUrl);
        };
        img.src = bitmapUrl;

        // 6. Update sidebar
        updateStats(fps, total, dets);

    } catch (err) {
        console.warn("Detection error:", err);
    }

    busy = false;
    scheduleNext();
}

// ── Sidebar updates ───────────────────────────────────────────────────────────
function updateStats(fps, total, dets) {
    // FPS
    fpsValueEl.textContent = fps.toFixed(1);
    fpsBarEl.style.width = Math.min((fps / MAX_FPS) * 100, 100) + "%";

    // Total objects
    totalValueEl.textContent = total;

    // Per-class
    for (const [cls, el] of Object.entries(classCountEls)) {
        const cnt = dets[cls] ?? 0;
        el.textContent = cnt;
        classRowEls[cls].classList.toggle("active", cnt > 0);
    }

    // Timestamp
    lastUpdateEl.textContent = new Date().toLocaleTimeString();
}

// ── Confidence Slider ─────────────────────────────────────────────────────────
function updateSliderFill() {
    const val = parseFloat(confSlider.value);
    const pct = ((val - 0.05) / (0.95 - 0.05)) * 100;
    confSlider.style.background =
        `linear-gradient(90deg, var(--cyan) ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
}

async function loadConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        const val = (data.confidence ?? 0.40).toFixed(2);
        confSlider.value = val;
        confDisplay.textContent = val;
        updateSliderFill();
    } catch (_) { }
}

let confTimer = null;
confSlider.addEventListener("input", () => {
    const val = parseFloat(confSlider.value).toFixed(2);
    confDisplay.textContent = val;
    updateSliderFill();

    clearTimeout(confTimer);
    confTimer = setTimeout(async () => {
        try {
            await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ confidence: parseFloat(val) }),
            });
        } catch (_) { }
    }, 300);
});

// ── Start button (shown if permission denied initially) ───────────────────────
startCamBtn.addEventListener("click", startCamera);

// ── Boot ─────────────────────────────────────────────────────────────────────
loadConfig();
startCamera();
