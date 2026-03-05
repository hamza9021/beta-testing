// ─────────────────────────────────────────────
//  app.js  —  Live Stats & Controls
// ─────────────────────────────────────────────

const POLL_INTERVAL_MS = 800;   // how often to fetch /api/stats (ms)
const MAX_FPS = 60;    // for the FPS bar fill %

// DOM refs
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

// ── Stats polling ──────────────────────────────────────────────────────────
async function fetchStats() {
    try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const data = await res.json();

        // FPS
        const fps = data.fps ?? 0;
        fpsValueEl.textContent = fps.toFixed(1);
        const pct = Math.min((fps / MAX_FPS) * 100, 100);
        fpsBarEl.style.width = pct + "%";

        // Update slider gradient to reflect fill
        const sliderVal = parseFloat(confSlider.value);
        const sliderPct = ((sliderVal - 0.05) / (0.95 - 0.05)) * 100;
        confSlider.style.background =
            `linear-gradient(90deg, var(--cyan) ${sliderPct}%, rgba(255,255,255,0.08) ${sliderPct}%)`;

        // Total count
        const total = data.total ?? 0;
        totalValueEl.textContent = total;

        // Per-class counts
        const dets = data.detections ?? {};
        for (const [cls, el] of Object.entries(classCountEls)) {
            const cnt = dets[cls] ?? 0;
            el.textContent = cnt;
            classRowEls[cls].classList.toggle("active", cnt > 0);
        }

        // Timestamp
        lastUpdateEl.textContent = new Date().toLocaleTimeString();

    } catch (err) {
        // Silently ignore network errors while server is starting
    }
}

// Start polling
setInterval(fetchStats, POLL_INTERVAL_MS);
fetchStats();   // immediate first call

// ── Confidence Slider ──────────────────────────────────────────────────────

// Fetch current config on load
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

function updateSliderFill() {
    const val = parseFloat(confSlider.value);
    const pct = ((val - 0.05) / (0.95 - 0.05)) * 100;
    confSlider.style.background =
        `linear-gradient(90deg, var(--cyan) ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
}

// Debounced POST to /api/config
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

// Reconnect video stream if it stalls (e.g. server restart)
const videoEl = document.getElementById("videoFeed");
let videoStallTimer = null;

videoEl.addEventListener("load", () => {
    clearTimeout(videoStallTimer);
    startStallWatcher();
});

function startStallWatcher() {
    videoStallTimer = setTimeout(() => {
        // Force reload the src if no new frame arrived in 5 s
        const src = videoEl.src.split("?")[0];
        videoEl.src = src + "?t=" + Date.now();
    }, 5000);
}

// Init
loadConfig();
