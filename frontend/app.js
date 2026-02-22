/**
 * HNDSR Frontend — app.js
 * Handles image upload, API communication, result display, and error handling.
 */

// --- Configuration ---
// If running on localhost, use local API. Otherwise, use your Hugging Face URL.
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://the-harsh-vardhan-hndsr-production.hf.space';
const POLL_INTERVAL = 5000; // 5 seconds

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewImage = document.getElementById('previewImage');
const inferBtn = document.getElementById('inferBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');
const resultsSection = document.getElementById('resultsSection');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const errorBanner = document.getElementById('errorBanner');
const errorText = document.getElementById('errorText');
const downloadBtn = document.getElementById('downloadBtn');

let currentImageB64 = null;
let outputImageB64 = null;

// ─────────────────────────────────────────────────────────────────────
// Health Check
// ─────────────────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();

        if (data.model_loaded) {
            statusBadge.className = 'status-badge status-online';
            statusText.textContent = data.gpu_available
                ? `Online • ${data.gpu_name || 'GPU'}`
                : 'Online • CPU';
        } else {
            statusBadge.className = 'status-badge status-loading';
            statusText.textContent = 'Loading model...';
        }
    } catch {
        statusBadge.className = 'status-badge status-offline';
        statusText.textContent = 'API Offline';
    }
}

checkHealth();
setInterval(checkHealth, 15000);

// ─────────────────────────────────────────────────────────────────────
// Upload Handling
// ─────────────────────────────────────────────────────────────────────

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showError('Image too large. Max file size: 20 MB.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        previewImage.src = dataUrl;
        previewImage.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');

        // Extract base64 (remove data:image/...;base64, prefix)
        currentImageB64 = dataUrl.split(',')[1];
        inferBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────

inferBtn.addEventListener('click', runInference);

async function runInference() {
    if (!currentImageB64) return;

    // UI: loading state
    inferBtn.disabled = true;
    btnText.textContent = 'Inferring...';
    btnSpinner.classList.remove('hidden');
    hideError();

    const scale = parseInt(document.getElementById('scaleSelect').value);
    const ddim = parseInt(document.getElementById('ddimSteps').value);
    const seedVal = document.getElementById('seedInput').value;

    const body = {
        image: currentImageB64,
        scale_factor: scale,
        ddim_steps: ddim,
        return_metadata: true,
    };
    if (seedVal) body.seed = parseInt(seedVal);

    try {
        const res = await fetch(`${API_BASE}/infer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const detail = err.detail || `HTTP ${res.status}`;

            if (res.status === 429) {
                showError(`Rate limited: ${detail}. Wait before retrying.`);
            } else if (res.status === 503) {
                showError(`Server overloaded: ${detail}. Try again in a few seconds.`);
            } else if (res.status === 413) {
                showError(`Image too large: ${detail}`);
            } else if (res.status === 504) {
                showError(`Timeout: ${detail}`);
            } else {
                showError(`Error: ${detail}`);
            }
            return;
        }

        const data = await res.json();
        displayResult(data);

    } catch (err) {
        showError(`Connection error: ${err.message}. Is the API running?`);
    } finally {
        inferBtn.disabled = false;
        btnText.textContent = 'Super-Resolve';
        btnSpinner.classList.add('hidden');
    }
}

// ─────────────────────────────────────────────────────────────────────
// Display Result
// ─────────────────────────────────────────────────────────────────────

function displayResult(data) {
    outputImageB64 = data.image;

    document.getElementById('resultInput').src = previewImage.src;
    document.getElementById('resultOutput').src = `data:image/png;base64,${data.image}`;

    document.getElementById('inputInfo').textContent =
        `${data.metadata?.input_size || '?'}`;
    document.getElementById('outputInfo').textContent =
        `${data.width}×${data.height}`;

    // Metadata
    if (data.metadata) {
        document.getElementById('metaLatency').textContent = `${data.metadata.latency_ms} ms`;
        document.getElementById('metaScale').textContent = `${data.metadata.scale_factor}×`;
        document.getElementById('metaDDIM').textContent = data.metadata.ddim_steps;
        document.getElementById('metaDevice').textContent = data.metadata.device || '—';
        document.getElementById('metaModel').textContent = 'HNDSR';
        document.getElementById('metaFP16').textContent = data.metadata.fp16 ? 'Yes' : 'No';
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ─────────────────────────────────────────────────────────────────────
// Download
// ─────────────────────────────────────────────────────────────────────

downloadBtn.addEventListener('click', () => {
    if (!outputImageB64) return;
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${outputImageB64}`;
    link.download = 'hndsr_super_resolved.png';
    link.click();
});

// ─────────────────────────────────────────────────────────────────────
// Error Handling
// ─────────────────────────────────────────────────────────────────────

function showError(msg) {
    errorText.textContent = msg;
    errorBanner.classList.remove('hidden');
}

function hideError() {
    errorBanner.classList.add('hidden');
}
