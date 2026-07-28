// GlassFit wizard: state machine + wiring. Session state lives here.

import {
  getHealth,
  postFeedback,
  postFrameMatch,
  postMatchRating,
  postRecommendations,
  postScan,
} from './api.js';
import { Camera } from './camera.js';
import { $, setError } from './dom.js';
import {
  buildAnalyzePayload,
  buildFeedbackControls,
  buildFeedbackPayload,
  buildFrameChoice,
  initForms,
} from './forms.js';
import { drawScanOverlay, renderQualityChips } from './overlay.js';
import { renderFrameMatches, renderResults } from './results.js';

// bestFrameBlob: the one captured frame kept after upload (for the review overlay) —
// the raw bursts (~1-2 MB of JPEGs) are released as soon as the scan succeeds.
// matchSeq: generation token so a stale in-flight /frames/match response can never
// overwrite a newer analysis's shortlist (or mislabel feedback training data).
const session = {
  bursts: { front: [], left: [], right: [] },
  bestFrameBlob: null,
  scan: null,
  analysis: null,
  matchSeq: 0,
};
const camera = new Camera($('preview'));

// The guided capture choreography: frontal burst, then two head turns whose side
// views let the backend measure hinge-to-ear geometry far better than a frontal
// frame can. Counts stay within the backend's 15-frame total budget.
const PHASES = [
  { key: 'front', label: '1/3 · Face forward — hold still', dir: null, count: 6, holdMs: 900 },
  { key: 'left', label: '2/3 · Turn your head to your LEFT', dir: 'left', count: 3, holdMs: 1900 },
  { key: 'right', label: '3/3 · Now turn to your RIGHT', dir: 'right', count: 3, holdMs: 1900 },
];

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
    announce('Camera ready. Center your face in the oval, then press Begin scan.');
  } catch (err) {
    showCameraError('Camera unavailable', err.message);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function countdown() {
  const overlay = $('countdown');
  overlay.hidden = false;
  for (const n of ['3', '2', '1']) {
    overlay.innerHTML = '';
    const tick = document.createElement('span');
    tick.textContent = n;
    overlay.appendChild(tick);
    announce(n);
    await sleep(700);
  }
  overlay.innerHTML = ''; // don't let the last digit linger in the DOM
  overlay.hidden = true;
}

function showPhase(phase) {
  const bar = $('phase-bar');
  $('phase-label').textContent = phase.label;
  if (phase.dir) bar.dataset.dir = phase.dir;
  else delete bar.dataset.dir;
  bar.hidden = false;
  announce(phase.label);
}

function hidePhase() {
  $('phase-bar').hidden = true;
}

function setPoseState(key, state) {
  document.querySelectorAll('.pose-strip li').forEach((li) => {
    if (li.dataset.pose === key) {
      li.classList.remove('active', 'done');
      if (state) li.classList.add(state);
    }
  });
}

function resetPoseStrip() {
  document.querySelectorAll('.pose-strip li').forEach((li) => li.classList.remove('active', 'done'));
}

async function runScan() {
  if (!camera.ready) {
    // e.g. "Scan again" after the stream was stopped — restart instead of a silent no-op
    await startCamera();
    if (!camera.ready) return; // startCamera surfaced its own error panel
  }
  $('scan-error').hidden = true;
  $('btn-scan').disabled = true;
  try {
    resetPoseStrip();
    await countdown();
    for (const phase of PHASES) {
      setPoseState(phase.key, 'active');
      showPhase(phase);
      await sleep(phase.holdMs); // give the user time to read and turn
      session.bursts[phase.key] = await camera.captureFrames(phase.count, 180);
      setPoseState(phase.key, 'done');
      announce('Pose captured.');
    }
    hidePhase();
    announce('Uploading scan…');
    const scan = await postScan(session.bursts);
    session.scan = scan;
    // keep only the frame the overlay needs; release the rest of the JPEG bursts
    session.bestFrameBlob = session.bursts.front[scan.best_frame_index];
    session.bursts = { front: [], left: [], right: [] };
    camera.stop();
    $('btn-start').hidden = false;
    setStep('review');
    announce(
      `Face detected with ${scan.landmarks.points.length} landmarks. ` +
        'Confirm your scan, then enter your PD.'
    );
    renderQualityChips($('quality-chips'), scan.quality); // instant — before the bitmap decode
    await drawScanOverlay($('overlay-canvas'), session.bestFrameBlob, scan);
  } catch (err) {
    hidePhase();
    showScanError(err);
  } finally {
    $('btn-scan').disabled = !camera.ready;
  }
}

function backToCamera() {
  session.scan = null;
  session.bestFrameBlob = null;
  session.bursts = { front: [], left: [], right: [] };
  session.matchSeq += 1; // orphan any in-flight match fetch
  $('matches-root').textContent = '';
  buildFrameChoice([]);
  resetPoseStrip();
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
    loadFrameMatches(analysis.recommendation_id); // non-blocking shortlist
  } catch (err) {
    setStep('review', { focus: false });
    if (err.kind === 'server' && /scan .* not found/i.test(err.message)) {
      showScanError(err);
      setStep('camera');
    } else {
      setError($('pd-err'), err.message);
      announce(`Analysis failed: ${err.message}`);
    }
  }
}

async function loadFrameMatches(recommendationId) {
  const seq = ++session.matchSeq;
  const container = $('matches-root');
  container.textContent = '';
  buildFrameChoice([]);
  try {
    const data = await postFrameMatch({ recommendation_id: recommendationId, limit: 5 });
    if (seq !== session.matchSeq) return; // a newer analysis/rescan superseded this fetch
    // Ratings snapshot the matcher's own claims (fit_score + components) so the
    // learning loop can compare them against human judgment after re-tuning.
    const onRate = (match, rating) =>
      postMatchRating({
        recommendation_id: recommendationId,
        frame_id: match.frame.frame_id,
        rating,
        fit_score: match.fit_score,
        components: match.components,
      });
    renderFrameMatches(container, data, onRate);
    buildFrameChoice(data.matches);
  } catch {
    // the shortlist is an extra — the measurements/recommendation above stand alone
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
  const button = $('btn-submit-feedback');
  if (button.disabled) return; // in-flight guard: no double submissions
  button.disabled = true;
  try {
    await postFeedback(payload);
    setStep('done');
    announce('Feedback saved. Thank you.');
  } catch (err) {
    setError($('fb-err'), err.message);
  } finally {
    button.disabled = false;
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
