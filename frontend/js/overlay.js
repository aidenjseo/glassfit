// Review-canvas drawing: the kept captured frame + server-computed landmark geometry.
// One mirror transform is applied to BOTH the image and the raw pixel coordinates so
// the still matches the mirrored viewfinder; text is drawn unmirrored (x' = w - x).

const ACCENT = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#0f6b62';

export async function drawScanOverlay(canvas, frameBlob, scan) {
  const { width, height } = scan.image_size;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const bitmap = await createImageBitmap(frameBlob);

  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1); // mirror: match the selfie viewfinder
  ctx.drawImage(bitmap, 0, 0, width, height);

  // all landmarks: faint dots (pixel positions derived from the normalized set)
  ctx.fillStyle = ACCENT;
  ctx.globalAlpha = 0.35;
  for (const [nx, ny] of scan.landmarks.points) {
    ctx.beginPath();
    ctx.arc(nx * width, ny * height, Math.max(1.5, width / 640), 0, Math.PI * 2);
    ctx.fill();
  }

  // measurement segments
  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = Math.max(2, width / 500);
  for (const seg of scan.overlay_segments) {
    ctx.beginPath();
    ctx.moveTo(seg.start[0], seg.start[1]);
    ctx.lineTo(seg.end[0], seg.end[1]);
    ctx.stroke();
  }

  // key points: solid dots
  for (const point of Object.values(scan.key_points)) {
    ctx.beginPath();
    ctx.arc(point[0], point[1], Math.max(3, width / 300), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // labels, unmirrored text at mirrored positions
  ctx.font = `${Math.max(13, Math.round(width / 55))}px -apple-system, "Segoe UI", Roboto, sans-serif`;
  ctx.textBaseline = 'bottom';
  for (const seg of scan.overlay_segments) {
    const mx = width - (seg.start[0] + seg.end[0]) / 2;
    const my = Math.min(seg.start[1], seg.end[1]) - 6;
    const metrics = ctx.measureText(seg.label);
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(mx - metrics.width / 2 - 4, my - 16, metrics.width + 8, 20);
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText(seg.label, mx, my + 2);
  }
  bitmap.close();
}

export function renderQualityChips(el, quality) {
  el.textContent = '';
  const add = (text, cls) => {
    const chip = document.createElement('span');
    chip.className = `chip ${cls}`;
    chip.textContent = text;
    el.appendChild(chip);
  };
  add(`Frames ${quality.frames_used}/${quality.frames_received}`, quality.ok ? 'good' : 'warn');
  const best = quality.frame_reports.find((r) => r.accepted);
  if (best && best.yaw_deg != null) {
    const posed = Math.abs(best.yaw_deg) < 8 && Math.abs(best.pitch_deg || 0) < 8;
    add(`Pose ${posed ? 'good' : 'tilted'}`, posed ? 'good' : 'warn');
  }
  const jitter = quality.landmark_dispersion_px;
  add(`Stability ${jitter < 3 ? 'high' : jitter < 6 ? 'ok' : 'low'}`, jitter < 6 ? 'good' : 'warn');
  const side = quality.side_views;
  if (side) {
    // subject-right side is exposed by the LEFT head turn, and vice versa
    add(`Left turn ${side.right_frames > 0 ? '✓' : '—'}`, side.right_frames > 0 ? 'good' : 'warn');
    add(`Right turn ${side.left_frames > 0 ? '✓' : '—'}`, side.left_frames > 0 ? 'good' : 'warn');
  }
  if (!quality.ok) add('Consider rescanning', 'warn');
}
