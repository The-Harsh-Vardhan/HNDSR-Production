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
let fallbackBanner = null;
const CONVERT_TO_PNG_TYPES = new Set(['image/avif', 'image/bmp', 'image/tiff']);

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
    if (e.dataTransfer.files.length) void handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) void handleFile(fileInput.files[0]);
});

async function handleFile(file) {
    const OK_TYPES = [
        'image/png', 'image/jpeg', 'image/webp',
        'image/avif', 'image/bmp', 'image/tiff',
    ];
    hideError();
    inferBtn.disabled = true;
    currentImageB64 = null;
    if (!file.type.startsWith('image/')) return showError('Please select an image file.');
    if (!OK_TYPES.includes(file.type))   return showError(`Unsupported format (${file.type}). Use PNG, JPEG, or WebP.`);
    if (file.size > 20 * 1024 * 1024)    return showError('File too large. Max 20 MB.');

    try {
        let previewDataUrl = await fileToDataUrl(file);
        if (CONVERT_TO_PNG_TYPES.has(file.type)) {
            previewDataUrl = await convertDataUrlToPng(previewDataUrl);
        }

        const imageB64 = extractBase64FromDataUrl(previewDataUrl);
        if (!imageB64) {
            throw new Error("Could not prepare image payload.");
        }

        previewImage.src = previewDataUrl;
        previewImage.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        currentImageB64 = imageB64;
        inferBtn.disabled = false;
    } catch (err) {
        const msg = err && err.message ? err.message : 'Image conversion failed.';
        showError(`Could not process this image: ${msg}`);
    }
}

// ── Inference ────────────────────────────────────────────────────────
inferBtn.addEventListener('click', runInference);

async function runInference() {
    if (!currentImageB64) return;

    inferBtn.disabled = true;
    btnText.textContent = 'Processing\u2026';
    btnSpinner.classList.remove('hidden');
    hideError();
    hideFallbackBanner();

    const body = {
        image: currentImageB64,
        scale_factor: sanitizeInteger($id('scaleSelect').value, 4, 2, 8),
        ddim_steps:   sanitizeInteger($id('ddimSteps').value, 10, 10, 200),
        return_metadata: true,
    };
    const seedVal = $id('seedInput').value.trim();
    if (seedVal !== '') {
        const seed = Number.parseInt(seedVal, 10);
        if (Number.isInteger(seed)) {
            body.seed = seed;
        }
    }

    try {
        const res = await fetch(`${API_BASE}/infer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const detail = formatErrorDetail(err.detail);
            const msg = typeof err.message === 'string' ? err.message : '';
            const d = detail || msg || `HTTP ${res.status}`;
            if (res.status === 429) return showError(`Rate limited: ${d}`);
            if (res.status === 503) return showError(`Server busy: ${d}`);
            if (res.status === 413) return showError(`Image too large: ${d}`);
            if (res.status === 504) return showError(`Timeout: ${d}`);
            if (res.status === 422) return showError(`Validation error: ${d}`);
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

        if (data.metadata.inference_mode === 'bicubic_fallback') {
            showFallbackBanner(data.metadata.fallback_reason);
        } else {
            hideFallbackBanner();
        }
    } else {
        hideFallbackBanner();
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

function ensureFallbackBanner() {
    if (fallbackBanner) return fallbackBanner;

    fallbackBanner = document.createElement('div');
    fallbackBanner.id = 'fallbackBanner';
    fallbackBanner.classList.add('hidden');
    fallbackBanner.style.margin = '0 0 1rem 0';
    fallbackBanner.style.padding = '0.75rem 1rem';
    fallbackBanner.style.border = '1px solid rgba(255, 179, 0, 0.45)';
    fallbackBanner.style.borderRadius = '10px';
    fallbackBanner.style.background = 'rgba(255, 179, 0, 0.12)';
    fallbackBanner.style.color = '#ffd77a';
    fallbackBanner.style.fontWeight = '600';
    fallbackBanner.style.fontSize = '0.95rem';

    const title = resultsSection.querySelector('h3');
    if (title) {
        title.insertAdjacentElement('afterend', fallbackBanner);
    } else {
        resultsSection.prepend(fallbackBanner);
    }
    return fallbackBanner;
}

function showFallbackBanner(reason) {
    const banner = ensureFallbackBanner();
    const reasonText = reason ? ` Reason: ${reason}` : '';
    banner.textContent = `Model checkpoints were not trusted; serving bicubic fallback.${reasonText}`;
    banner.classList.remove('hidden');
}

function hideFallbackBanner() {
    if (fallbackBanner) {
        fallbackBanner.classList.add('hidden');
    }
}

function sanitizeInteger(rawValue, fallback, min, max) {
    const parsed = Number.parseInt(rawValue, 10);
    if (!Number.isInteger(parsed)) return fallback;
    if (parsed < min || parsed > max) return fallback;
    return parsed;
}

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = () => reject(new Error('Failed to read input file.'));
        reader.readAsDataURL(file);
    });
}

function convertDataUrlToPng(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                reject(new Error('Canvas is not available in this browser.'));
                return;
            }
            ctx.drawImage(img, 0, 0);
            resolve(canvas.toDataURL('image/png'));
        };
        img.onerror = () => reject(new Error('Browser could not decode this image format.'));
        img.src = dataUrl;
    });
}

function extractBase64FromDataUrl(dataUrl) {
    if (typeof dataUrl !== 'string') return null;
    const splitIdx = dataUrl.indexOf(',');
    if (splitIdx === -1) return null;
    const payload = dataUrl.slice(splitIdx + 1).trim();
    return payload || null;
}

function formatErrorDetail(detail) {
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        const rows = detail
            .map(formatValidationEntry)
            .filter((line) => line.length > 0);
        return rows.length ? rows.join('; ') : 'Request validation failed.';
    }
    if (detail && typeof detail === 'object') {
        try {
            return JSON.stringify(detail);
        } catch {
            return 'Request validation failed.';
        }
    }
    return '';
}

function formatValidationEntry(entry) {
    if (!entry || typeof entry !== 'object') return '';
    const loc = Array.isArray(entry.loc) ? entry.loc.join('.') : '';
    const msg = typeof entry.msg === 'string' ? entry.msg : '';
    if (!loc && !msg) return '';
    if (!loc) return msg;
    if (!msg) return loc;
    return `${loc}: ${msg}`;
}
