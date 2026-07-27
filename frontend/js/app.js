// GlassFit wizard: state machine + wiring. Session state lives here.

import { getHealth, postFeedback, postRecommendations, postScan } from './api.js';
import { Camera } from './camera.js';
import { buildAnalyzePayload, buildFeedbackControls, buildFeedbackPayload, initForms } from './forms.js';
import { drawScanOverlay, renderQualityChips } from './overlay.js';
import { renderResults } from './results.js';

const $ = (id) => document.getElementById(id);
const session = { frames: [], scan: null, analysis: null };
const camera = new Camera($('preview'));

const STEP_DOTS = { camera: 'camera', review: 'review', analyzing: 'review', results: 'results', feedback: 'results', done: 'results' };

function announce(text) {
  $('status').textContent = text;
}

function setStep(step, { focus = true } = {}) {
  document.querySelectorAll('main > section[data-step]').forEach((section) => {
    section.hidden = section.dataset.step !== step;
  });
  document.querySelectorAll('.steps li').forEach((li) => {
    li.classList.toggle('current', li.dataset.stepdot === STEP_DOTS[step]);
  });
  const heading = document.querySelector(`section[data-step="${step}"] h1`);
  if (heading && focus) heading.focus();
}

function showCameraError(title, message) {
  $('camera-error-title').textContent = title;
  $('camera-error-msg').textContent = message;
  $('camera-error').hidden = false;
  $('scan-error').hidden = true;
  announce(`${title}. ${message}`);
}

function showScanError(err) {
  const tips = {
    no_face: ['Face the camera straight on', 'Move to even, front-facing light', 'Only one face in frame'],
    poor_quality: ['Hold still during the capture', 'Look directly at the camera', 'Move closer — your face should fill the oval'],
    unavailable: ['Run: uv sync --extra scan', 'Run: uv run python scripts/download_models.py', 'Restart the server'],
    server: ['Check the server terminal for errors'],
    network: ['Start the backend: uv run glassfit'],
    validation: [],
  };
  $('scan-error-title').textContent = 'Scan failed';
  $('scan-error-msg').textContent = err.message;
  const list = $('scan-error-tips');
  list.textContent = '';
  for (const tip of tips[err.kind] || []) {
    const li = document.createElement('li');
    li.textContent = tip;
    list.appendChild(li);
  }
  $('scan-error').hidden = false;
  $('camera-error').hidden = true;
  announce(`Scan failed. ${err.message}`);
}

async function startCamera() {
  $('camera-error').hidden = true;
  $('scan-error').hidden = true;
  try {
    await camera.start();
    $('btn-scan').disabled = false;
    $('btn-start').hidden = true;
    announce('Camera ready. Press Scan when you are aligned.');
  } catch (err) {
    showCameraError('Camera unavailable', err.message);
  }
}

async function countdown() {
  const overlay = $('countdown');
  overlay.hidden = false;
  for (const n of ['3', '2', '1']) {
    overlay.textContent = n;
    announce(n);
    await new Promise((r) => setTimeout(r, 700));
  }
  overlay.hidden = true;
}

async function runScan() {
  if (!camera.ready) return;
  $('scan-error').hidden = true;
  $('btn-scan').disabled = true;
  try {
    await countdown();
    announce('Capturing…');
    const { blobs, width, height, frameIntervalMs } = await camera.captureFrames(6, 200);
    session.frames = blobs;
    announce('Uploading scan…');
    const scan = await postScan(blobs, {
      width,
      height,
      captured_at: new Date().toISOString(),
      frame_interval_ms: frameIntervalMs,
    });
    session.scan = scan;
    camera.stop();
    $('btn-start').hidden = false;
    setStep('review');
    announce(`Face detected with ${scan.landmarks.points.length} landmarks. Confirm your scan, then enter your PD.`);
    await drawScanOverlay($('overlay-canvas'), blobs[scan.best_frame_index], scan);
    renderQualityChips($('quality-chips'), scan.quality);
  } catch (err) {
    showScanError(err);
  } finally {
    $('btn-scan').disabled = !camera.ready;
  }
}

function backToCamera() {
  session.scan = null;
  session.frames = [];
  setStep('camera');
  startCamera();
}

async function analyze(event) {
  event.preventDefault();
  const payload = buildAnalyzePayload(session.scan ? session.scan.scan_id : null);
  if (!payload) return;
  setStep('analyzing');
  announce('Analyzing your fit…');
  try {
    const analysis = await postRecommendations(payload);
    session.analysis = analysis;
    setStep('results');
    renderResults($('results-root'), analysis);
    announce('Analysis complete. Your measurements and frame recommendation are ready.');
  } catch (err) {
    setStep('review', { focus: false });
    if (err.kind === 'server' && /scan .* not found/i.test(err.message)) {
      showScanError(err);
      setStep('camera');
    } else {
      const errEl = $('pd-err');
      errEl.textContent = err.message;
      errEl.hidden = false;
      announce(`Analysis failed: ${err.message}`);
    }
  }
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(session.analysis, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `glassfit_${session.analysis.recommendation_id.slice(0, 8)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function submitFeedback(event) {
  event.preventDefault();
  const payload = buildFeedbackPayload(session.analysis.recommendation_id);
  if (!payload) return;
  try {
    await postFeedback(payload);
    setStep('done');
    announce('Feedback saved. Thank you.');
  } catch (err) {
    const errEl = $('fb-err');
    errEl.textContent = err.message;
    errEl.hidden = false;
  }
}

function wire() {
  initForms();
  buildFeedbackControls();
  $('btn-start').addEventListener('click', startCamera);
  $('btn-retry-camera').addEventListener('click', startCamera);
  $('btn-scan').addEventListener('click', runScan);
  $('btn-retry-scan').addEventListener('click', () => {
    $('scan-error').hidden = true;
    runScan();
  });
  $('btn-retake').addEventListener('click', backToCamera);
  $('inputs-form').addEventListener('submit', analyze);
  $('btn-download').addEventListener('click', downloadJson);
  $('btn-feedback').addEventListener('click', () => setStep('feedback'));
  $('feedback-form').addEventListener('submit', submitFeedback);
  $('btn-rescan-results').addEventListener('click', backToCamera);
  $('btn-rescan-done').addEventListener('click', backToCamera);

  getHealth()
    .then((health) => {
      if (!health.mediapipe_available) {
        const banner = $('banner');
        banner.textContent =
          'Face detection is not ready on the server — run `uv sync --extra scan` and `uv run python scripts/download_models.py`, then restart.';
        banner.hidden = false;
      }
    })
    .catch(() => {
      const banner = $('banner');
      banner.textContent = 'Backend not reachable — start it with `uv run glassfit`.';
      banner.hidden = false;
    });
}

wire();
setStep('camera', { focus: false });
