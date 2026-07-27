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

export function postScan(blobs, meta) {
  const form = new FormData();
  blobs.forEach((blob, i) => form.append('frames', blob, `frame_${i}.jpg`));
  form.append('meta', JSON.stringify(meta));
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
