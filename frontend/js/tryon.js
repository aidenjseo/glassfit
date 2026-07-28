// Virtual try-on: composite a frame onto the user's scan photo at TRUE scale.
//
// Everything needed is already measured: the scan keeps pupil positions in pixels
// and the analysis provides mm_per_unit (mm per photo pixel), so a frame with a
// 52mm lens width is drawn exactly 52/mm_per_unit pixels wide, centered on the
// pupil line and rotated to the head's roll. Art comes from either:
//   1. /tryon/<frame_id>.png — a real product photo dropped in locally
//      (e.g. via scripts/fetch_warby_frames.py; personal use, never committed), or
//   2. a parametric SVG rendered from the frame's catalog shape + dimensions.

const MATERIAL_STYLE = {
  acetate: { stroke: '#2b2118', width: 2.4 },
  'TR-90': { stroke: '#202020', width: 2.0 },
  nylon: { stroke: '#262b2e', width: 1.8 },
  combo: { stroke: '#2b2118', width: 2.2 },
  metal: { stroke: '#6e6a5e', width: 1.0 },
  titanium: { stroke: '#7d838a', width: 0.85 },
};
const LENS_FILL = 'rgba(185, 205, 220, 0.12)';
const ENDPIECE_MM = 4; // visual endpiece stub per side

function materialStyle(frame) {
  const style = MATERIAL_STYLE[frame.material] || MATERIAL_STYLE.acetate;
  if (frame.rim === 'rimless') return { stroke: style.stroke, width: 0.4 };
  if (frame.rim === 'semi') return { stroke: style.stroke, width: style.width * 0.75 };
  return style;
}

/** Rounded-rect path with per-corner radii — the basis for most lens shapes. */
function lensPath(x, y, w, h, [tl, tr, br, bl]) {
  return (
    `M ${x + tl} ${y}` +
    ` H ${x + w - tr} Q ${x + w} ${y} ${x + w} ${y + tr}` +
    ` V ${y + h - br} Q ${x + w} ${y + h} ${x + w - br} ${y + h}` +
    ` H ${x + bl} Q ${x} ${y + h} ${x} ${y + h - bl}` +
    ` V ${y + tl} Q ${x} ${y} ${x + tl} ${y} Z`
  );
}

/** Corner radii per catalog shape; `outer` = which side faces the temple. */
function shapeRadii(shape, b, outer) {
  const r = (f) => b * f;
  const sym = (t, bo) => [r(t), r(t), r(bo), r(bo)];
  switch (shape) {
    case 'round':
      return sym(0.5, 0.5);
    case 'oval':
      return sym(0.48, 0.48);
    case 'panto':
      return sym(0.28, 0.5);
    case 'aviator':
      return outer === 'left' ? [r(0.42), r(0.3), r(0.52), r(0.46)] : [r(0.3), r(0.42), r(0.46), r(0.52)];
    case 'cat-eye':
      return outer === 'left' ? [r(0.1), r(0.38), r(0.45), r(0.4)] : [r(0.38), r(0.1), r(0.4), r(0.45)];
    case 'square':
      return sym(0.12, 0.12);
    case 'geometric':
      return sym(0.08, 0.08);
    case 'wayfarer':
      return sym(0.24, 0.32);
    case 'browline':
      return sym(0.18, 0.42);
    case 'wrap':
      return sym(0.3, 0.35);
    default: // rectangle & friends
      return sym(0.18, 0.22);
  }
}

/** Front-view SVG of a frame, drawn in MILLIMETER units (viewBox = mm). */
export function frameSVG(frame) {
  const a = frame.a_mm;
  const b = frame.b_mm;
  const dbl = frame.dbl_mm;
  const { stroke, width } = materialStyle(frame);
  const pad = 3;
  const totalW = 2 * a + dbl + 2 * ENDPIECE_MM + 2 * pad;
  const totalH = b + 2 * pad;
  const y = pad;
  const leftX = pad + ENDPIECE_MM;
  const rightX = leftX + a + dbl;
  const bridgeY = y + b * 0.24;

  const parts = [
    `<path d="${lensPath(leftX, y, a, b, shapeRadii(frame.shape, b, 'left'))}"` +
      ` fill="${LENS_FILL}" stroke="${stroke}" stroke-width="${width}"/>`,
    `<path d="${lensPath(rightX, y, a, b, shapeRadii(frame.shape, b, 'right'))}"` +
      ` fill="${LENS_FILL}" stroke="${stroke}" stroke-width="${width}"/>`,
    // bridge
    `<path d="M ${leftX + a} ${bridgeY + 1.5} Q ${leftX + a + dbl / 2} ${bridgeY - 2.5}` +
      ` ${rightX} ${bridgeY + 1.5}" fill="none" stroke="${stroke}"` +
      ` stroke-width="${Math.max(width, 1.4)}"/>`,
    // endpiece stubs
    `<rect x="${pad}" y="${bridgeY - 1}" width="${ENDPIECE_MM}" height="2" fill="${stroke}"/>`,
    `<rect x="${rightX + a}" y="${bridgeY - 1}" width="${ENDPIECE_MM}" height="2" fill="${stroke}"/>`,
  ];
  if (frame.shape === 'browline') {
    const barH = b * 0.2;
    parts.push(
      `<rect x="${leftX - 1}" y="${y - 1}" width="${a + 2}" height="${barH}" rx="${barH / 2}" fill="#2b2118"/>`,
      `<rect x="${rightX - 1}" y="${y - 1}" width="${a + 2}" height="${barH}" rx="${barH / 2}" fill="#2b2118"/>`
    );
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${totalW} ${totalH}">` +
    parts.join('') +
    `</svg>`
  );
}

const artCache = new Map(); // frame_id -> {img, wMM} (wMM: real width the art spans)

/** Frame art: a local product photo when present, else the parametric SVG. */
async function frameArt(frame) {
  if (artCache.has(frame.frame_id)) return artCache.get(frame.frame_id);
  const totalWMM = 2 * frame.a_mm + frame.dbl_mm + 2 * ENDPIECE_MM + 6;
  let art = null;
  try {
    const resp = await fetch(`/tryon/${frame.frame_id}.png`, { method: 'GET' });
    if (resp.ok && (resp.headers.get('content-type') || '').startsWith('image')) {
      const img = new Image();
      img.src = URL.createObjectURL(await resp.blob());
      await img.decode();
      art = { img, wMM: totalWMM };
    }
  } catch {
    /* no local product art — use the SVG */
  }
  if (!art) {
    const img = new Image();
    img.src = `data:image/svg+xml;utf8,${encodeURIComponent(frameSVG(frame))}`;
    await img.decode();
    art = { img, wMM: totalWMM };
  }
  artCache.set(frame.frame_id, art);
  return art;
}

/**
 * Draw the try-on composite: mirrored scan photo + the frame at true scale.
 * @param canvas   target canvas (intrinsic size set to the photo's)
 * @param photo    ImageBitmap of the best scan frame
 * @param scan     ScanResponse (key_points in photo pixels)
 * @param analysis RecommendationResponse (measurements.mm_per_unit, optics)
 * @param frame    CatalogFrame to wear
 */
export async function drawTryOn(canvas, photo, scan, analysis, frame) {
  const { width, height } = scan.image_size;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  const pr = scan.key_points.pupil_right;
  const pl = scan.key_points.pupil_left;
  const cx = (pr[0] + pl[0]) / 2;
  const cy = (pr[1] + pl[1]) / 2;
  const roll = Math.atan2(pl[1] - pr[1], pl[0] - pr[0]);
  const mmPerPx = analysis.measurements.mm_per_unit;

  const art = await frameArt(frame);
  const drawW = art.wMM / mmPerPx;
  const drawH = drawW * (art.img.naturalHeight / art.img.naturalWidth);
  // pupil sits OC-height above the lens bottom; the lens's vertical center is
  // (OC - B/2) BELOW the pupil line (y grows downward)
  const oc = (analysis.optics.oc_height_mm.right + analysis.optics.oc_height_mm.left) / 2;
  const dropPx = (oc - frame.b_mm / 2) / mmPerPx;

  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1); // match the mirrored viewfinder, photo and frame alike
  ctx.drawImage(photo, 0, 0, width, height);
  ctx.translate(cx, cy + dropPx);
  ctx.rotate(roll);
  ctx.drawImage(art.img, -drawW / 2, -drawH / 2, drawW, drawH);
  ctx.restore();
}
