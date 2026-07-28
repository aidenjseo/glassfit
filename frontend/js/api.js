// REST client for the GlassFit backend (same origin, /api/v1).
// All failures are normalized to { kind, message, details } and thrown.

const BASE = '/api/v1';

export class ApiError extends Error {
  constructor(kind, message, details = {}) {
    super(message);
    this.kind = kind; // 'network' | 'no_face' | 'poor_quality' | 'validation' | 'unavailable' | 'server'
    this.details = details;
  }
}

const CODE_TO_KIND = {
  NO_FACE_DETECTED: 'no_face',
  MULTIPLE_FACES: 'no_face',
  POOR_SCAN_QUALITY: 'poor_quality',
  VALIDATION_ERROR: 'validation',
  MEDIAPIPE_UNAVAILABLE: 'unavailable',
  NOT_FOUND: 'server',
  INVALID_IMAGE: 'server',
};

async function request(path, options = {}) {
  let resp;
  try {
    resp = await fetch(BASE + path, options);
  } catch {
    throw new ApiError('network', 'Backend not reachable — is the server running? (uv run glassfit)');
  }
  let body = null;
  try {
    body = await resp.json();
  } catch {
    /* non-JSON body */
  }
  if (resp.ok) return body;
  const err = body && body.error ? body.error : {};
  const kind = CODE_TO_KIND[err.code] || 'server';
  throw new ApiError(kind, err.message || `Request failed (${resp.status})`, err.details || {});
}

export function getHealth() {
  return request('/health');
}

/**
 * Upload the capture bursts.
 * @param {{front: Blob[], left: Blob[], right: Blob[]}} bursts — front is required;
 *   left/right are the head-turn bursts (may be empty).
 */
export function postScan(bursts) {
  const form = new FormData();
  bursts.front.forEach((blob, i) => form.append('frames', blob, `front_${i}.jpg`));
  (bursts.left || []).forEach((blob, i) => form.append('frames_left', blob, `left_${i}.jpg`));
  (bursts.right || []).forEach((blob, i) => form.append('frames_right', blob, `right_${i}.jpg`));
  return request('/scan', { method: 'POST', body: form });
}

export function postRecommendations(payload) {
  return request('/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function postFeedback(payload) {
  return request('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function postFrameMatch(payload) {
  return request('/frames/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function postScanProbe(blob) {
  const form = new FormData();
  form.append('frame', blob, 'probe.jpg');
  return request('/scan/probe', { method: 'POST', body: form });
}

export function postScanTrack(blob) {
  const form = new FormData();
  form.append('frame', blob, 'track.jpg');
  return request('/scan/track', { method: 'POST', body: form });
}

export function postMatchRating(payload) {
  return request('/frames/ratings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
