/**
 * HNDSR Frontend — app.js
 * Handles navbar, health polling, image upload, API inference, result display.
 */

// ── Configuration ────────────────────────────────────────────────────
const API_BASE =
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : 'https://the-harsh-vardhan-hndsr-production.hf.space';

// ── DOM refs ─────────────────────────────────────────────────────────
const $id = (id) => document.getElementById(id);

const fileInput       = $id('fileInput');
const uploadArea      = $id('uploadArea');
const uploadPlaceholder = $id('uploadPlaceholder');
const previewImage    = $id('previewImage');
const inferBtn        = $id('inferBtn');
const btnText         = $id('btnText');
const btnSpinner      = $id('btnSpinner');
const resultsSection  = $id('resultsSection');
const errorBanner     = $id('errorBanner');
const errorText       = $id('errorText');
const downloadBtn     = $id('downloadBtn');
const navToggle       = $id('navToggle');
const navLinks        = $id('navLinks');

let currentImageB64 = null;
let outputImageB64  = null;

// ── Navbar ───────────────────────────────────────────────────────────
navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));

// Close mobile menu when a link is clicked
navLinks.querySelectorAll('a').forEach((a) =>
    a.addEventListener('click', () => navLinks.classList.remove('open'))
);

// Shrink navbar on scroll
window.addEventListener('scroll', () => {
    document.querySelector('.navbar').classList.toggle('scrolled', window.scrollY > 40);
});

// ── Health check ─────────────────────────────────────────────────────
async function checkHealth() {
    const heroStatus      = $id('heroStatus');
    const heroStatusLabel = $id('heroStatusLabel');
    try {
        const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(8000) });
        const data = await res.json();
        if (data.model_loaded) {
            heroStatus.textContent      = '\u25CF';
            heroStatus.style.color      = 'var(--success)';
            heroStatusLabel.textContent = data.gpu_available
                ? `Online \u2022 ${data.gpu_name || 'GPU'}`
                : 'Online \u2022 CPU';
        } else {
            heroStatus.textContent      = '\u25CF';
            heroStatus.style.color      = 'var(--warning)';
            heroStatusLabel.textContent = 'Loading model\u2026';
        }
    } catch {
        heroStatus.textContent      = '\u25CF';
        heroStatus.style.color      = 'var(--danger)';
        heroStatusLabel.textContent = 'API Offline';
    }
}
checkHealth();
setInterval(checkHealth, 15000);

// ── Upload handling ──────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const OK_TYPES = [
        'image/png', 'image/jpeg', 'image/webp',
        'image/avif', 'image/bmp', 'image/tiff',
    ];
    if (!file.type.startsWith('image/')) return showError('Please select an image file.');
    if (!OK_TYPES.includes(file.type))   return showError(`Unsupported format (${file.type}). Use PNG, JPEG, or WebP.`);
    if (file.size > 20 * 1024 * 1024)    return showError('File too large. Max 20 MB.');

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        currentImageB64 = e.target.result.split(',')[1];
        inferBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ── Inference ────────────────────────────────────────────────────────
inferBtn.addEventListener('click', runInference);

async function runInference() {
    if (!currentImageB64) return;

    inferBtn.disabled = true;
    btnText.textContent = 'Processing\u2026';
    btnSpinner.classList.remove('hidden');
    hideError();

    const body = {
        image: currentImageB64,
        scale_factor: parseInt($id('scaleSelect').value),
        ddim_steps:   parseInt($id('ddimSteps').value),
        return_metadata: true,
    };
    const seedVal = $id('seedInput').value;
    if (seedVal) body.seed = parseInt(seedVal);

    try {
        const res = await fetch(`${API_BASE}/infer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const d = err.detail || `HTTP ${res.status}`;
            if (res.status === 429) return showError(`Rate limited: ${d}`);
            if (res.status === 503) return showError(`Server busy: ${d}`);
            if (res.status === 413) return showError(`Image too large: ${d}`);
            if (res.status === 504) return showError(`Timeout: ${d}`);
            return showError(`Error: ${d}`);
        }

        displayResult(await res.json());
    } catch (err) {
        showError(`Connection error: ${err.message}. Is the API running?`);
    } finally {
        inferBtn.disabled = false;
        btnText.textContent = 'Super-Resolve';
        btnSpinner.classList.add('hidden');
    }
}

// ── Display result ───────────────────────────────────────────────────
function displayResult(data) {
    outputImageB64 = data.image;

    $id('resultInput').src  = previewImage.src;
    $id('resultOutput').src = `data:image/png;base64,${data.image}`;
    $id('inputInfo').textContent  = data.metadata?.input_size || '?';
    $id('outputInfo').textContent = `${data.width}\u00D7${data.height}`;

    if (data.metadata) {
        $id('metaLatency').textContent = `${data.metadata.latency_ms} ms`;
        $id('metaScale').textContent   = `${data.metadata.scale_factor}\u00D7`;
        $id('metaDDIM').textContent    = data.metadata.ddim_steps;
        $id('metaDevice').textContent  = data.metadata.device || '\u2014';
        $id('metaModel').textContent   = 'HNDSR';
        $id('metaFP16').textContent    = data.metadata.fp16 ? 'Yes' : 'No';
    }

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ── Download ─────────────────────────────────────────────────────────
downloadBtn.addEventListener('click', () => {
    if (!outputImageB64) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${outputImageB64}`;
    a.download = 'hndsr_super_resolved.png';
    a.click();
});

// ── Error helpers ────────────────────────────────────────────────────
function showError(msg) {
    errorText.textContent = msg;
    errorBanner.classList.remove('hidden');
}
function hideError() {
    errorBanner.classList.add('hidden');
}
