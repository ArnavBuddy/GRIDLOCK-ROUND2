const fileInput = document.getElementById("trafficVideo");
const startButton = document.getElementById("startEngine");
const stopButton = document.getElementById("stopEngine");
const refreshButton = document.getElementById("refreshDashboard");
const statusText = document.getElementById("engineStatus");
const video = document.getElementById("annotatedVideo");
const liveStream = document.getElementById("liveStream");
const fsdStream = document.getElementById("fsdStream");
const videoEmpty = document.getElementById("videoEmpty");
const videoState = document.getElementById("videoState");
const fsdState = document.getElementById("fsdState");
const fsdMessage = document.getElementById("fsdMessage");
const themeToggle = document.getElementById("themeToggle");
const violationRows = document.getElementById("violationRows");
const evidenceCount = document.getElementById("evidenceCount");
const evidenceList = document.getElementById("evidenceList");

let activeJobId = null;
let pollTimer = null;

const metricFields = {
    vehicles: document.getElementById("metricVehicles"),
    violations: document.getElementById("metricViolations"),
    wrong_way: document.getElementById("metricWrongWay"),
    helmet: document.getElementById("metricHelmet"),
    seatbelt: document.getElementById("metricSeatbelt"),
    parking: document.getElementById("metricParking"),
    density: document.getElementById("metricDensity"),
    fps: document.getElementById("metricFps"),
};

function setTheme(theme) {
    document.body.classList.toggle("theme-light", theme === "light");
    document.body.classList.toggle("theme-dark", theme !== "light");
    themeToggle.textContent = theme === "light" ? "Dark Theme" : "Light Theme";
    localStorage.setItem("trafficiq-theme", theme);
}

function updateMetrics(metrics = {}) {
    metricFields.vehicles.textContent = metrics.vehicles ?? 0;
    metricFields.violations.textContent = metrics.violations ?? 0;
    metricFields.wrong_way.textContent = metrics.wrong_way ?? 0;
    metricFields.helmet.textContent = metrics.helmet ?? 0;
    metricFields.seatbelt.textContent = metrics.seatbelt ?? 0;
    metricFields.parking.textContent = metrics.parking ?? 0;
    metricFields.density.textContent = Number(metrics.density ?? 0).toFixed(2);
    metricFields.fps.textContent = Math.round(metrics.fps ?? 0);
}

function renderViolations(rows = []) {
    if (!rows.length) {
        violationRows.innerHTML = '<tr><td colspan="6">No violations logged yet.</td></tr>';
        return;
    }

    violationRows.innerHTML = rows.map((row) => `
        <tr>
            <td>${row.time || "-"}</td>
            <td>${row.vehicle || "-"}</td>
            <td>${row.tracker || "-"}</td>
            <td>${row.violation || "-"}</td>
            <td>${row.plate || "UNKNOWN"}</td>
            <td>${row.confidence || "0%"}</td>
        </tr>
    `).join("");
}

function renderEvidence(items = []) {
    evidenceCount.textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
    if (!items.length) {
        evidenceList.textContent = "No evidence matches the current filters.";
        return;
    }

    evidenceList.innerHTML = items.map((item) => `
        <article class="evidence-item">
            <strong>${item.violation || "Evidence"}</strong>
            <span>${item.vehicle || ""} ${item.tracker || ""}</span>
        </article>
    `).join("");
}

function showRealtimeStreams(jobId) {
    const expectedStream = `/api/stream/${jobId}`;
    if (!liveStream.src.includes(expectedStream)) {
        liveStream.src = `${expectedStream}?t=${Date.now()}`;
    }
    const expectedFsd = `/api/fsd/${jobId}`;
    if (!fsdStream.src.includes(expectedFsd)) {
        fsdStream.src = `${expectedFsd}?t=${Date.now()}`;
    }
    liveStream.style.display = "block";
    fsdStream.style.display = "block";
    video.style.display = "none";
    videoEmpty.style.display = "none";
    fsdMessage.style.display = "none";
}

function showVideoPreview(src) {
    video.src = src;
    video.load();
    video.style.display = "block";
    liveStream.style.display = "none";
    fsdStream.style.display = "none";
    videoEmpty.style.display = "none";
    fsdMessage.style.display = "block";
}

function setRunningState(job) {
    const status = job.status || "offline";
    const progress = job.progress ?? 0;

    if (status === "queued") {
        statusText.textContent = "AI engine queued. Preparing video analysis...";
        videoState.textContent = "Queued";
        fsdState.textContent = "AI engine starting";
        fsdMessage.textContent = "PREPARING FSD VISUALIZATION";
    } else if (status === "processing") {
        statusText.textContent = `AI engine processing uploaded video: ${progress}%`;
        videoState.textContent = "Realtime Detection";
        fsdState.textContent = "AI engine online";
        fsdMessage.textContent = "FSD VISUALIZATION ACTIVE";
        if (activeJobId) {
            showRealtimeStreams(activeJobId);
        }
    } else if (status === "complete") {
        statusText.textContent = "Annotated video ready.";
        videoState.textContent = "Realtime Complete";
        fsdState.textContent = "AI engine complete";
        fsdMessage.textContent = "ANALYSIS COMPLETE";
        if (job.output_url) {
            video.src = job.output_url;
            video.load();
            video.style.display = "block";
            liveStream.style.display = "none";
            videoEmpty.style.display = "none";
            fsdStream.style.display = "block";
            fsdMessage.style.display = "none";
        }
        stopPolling();
    } else if (status === "stopped") {
        statusText.textContent = "AI engine stopped.";
        videoState.textContent = "Stopped";
        fsdState.textContent = "AI engine offline";
        fsdMessage.textContent = "WAITING FOR FSD VISUALIZATION";
        fsdMessage.style.display = "block";
        stopPolling();
    } else if (status === "error") {
        statusText.textContent = job.error || "AI engine error.";
        videoState.textContent = "Error";
        fsdState.textContent = "AI engine offline";
        fsdMessage.textContent = "PROCESSING ERROR";
        stopPolling();
    }
}

async function pollJob() {
    if (!activeJobId) return;
    const response = await fetch(`/api/status/${activeJobId}`);
    const job = await response.json();
    if (!response.ok) {
        statusText.textContent = job.error || "Could not read engine status.";
        stopPolling();
        return;
    }

    updateMetrics(job.metrics);
    renderViolations(job.violations);
    renderEvidence(job.evidence);
    setRunningState(job);
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollJob, 1200);
    pollJob();
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function startEngine() {
    if (!fileInput.files.length) {
        statusText.textContent = "Choose a traffic video before starting the engine.";
        return;
    }

    const formData = new FormData();
    formData.append("video", fileInput.files[0]);
    statusText.textContent = "Uploading video...";
    videoEmpty.style.display = "grid";
    videoEmpty.textContent = "Uploading and preparing annotated output...";
    video.style.display = "none";
    liveStream.style.display = "none";
    fsdStream.style.display = "none";
    fsdMessage.style.display = "block";

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });
        const job = await response.json();
        if (!response.ok) {
            statusText.textContent = job.error || "Upload failed.";
            return;
        }

        activeJobId = job.id;
        updateMetrics(job.metrics);
        renderViolations(job.violations);
        renderEvidence(job.evidence);
        setRunningState(job);
        startPolling();
    } catch (error) {
        statusText.textContent = "Could not start the engine.";
    }
}

async function stopEngine() {
    if (!activeJobId) {
        statusText.textContent = "No active upload job to stop.";
        return;
    }

    await fetch(`/api/stop/${activeJobId}`, { method: "POST" });
    await pollJob();
}

function refreshDashboard() {
    updateMetrics();
    renderViolations();
    renderEvidence();
    statusText.textContent = "Dashboard refreshed. Upload a traffic video to begin.";
    videoState.textContent = "Video Preview";
    fsdState.textContent = "AI engine offline";
    fsdMessage.textContent = "WAITING FOR FSD VISUALIZATION";
    fsdMessage.style.display = "block";
    liveStream.style.display = "none";
    fsdStream.style.display = "none";
    video.style.display = "none";
    videoEmpty.style.display = "grid";
}

if (themeToggle) {
    const savedTheme = localStorage.getItem("trafficiq-theme") || "dark";
    setTheme(savedTheme);
    themeToggle.addEventListener("click", () => {
        setTheme(document.body.classList.contains("theme-light") ? "dark" : "light");
    });
}

fileInput?.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    statusText.textContent = `${file.name} selected. Start the engine to annotate.`;
    showVideoPreview(URL.createObjectURL(file));
});

startButton?.addEventListener("click", startEngine);
stopButton?.addEventListener("click", stopEngine);
refreshButton?.addEventListener("click", refreshDashboard);
