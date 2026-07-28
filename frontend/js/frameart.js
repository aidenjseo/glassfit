/* Dimension-true product art for catalog frames.
 *
 * One SVG unit = 1 mm and every render shares the same 150×76 viewBox with the
 * front centered at (75, 38), so relative frame sizes are physically comparable
 * across cards — a 44 mm Piper genuinely draws smaller than a 58 mm Bolt.
 * Drawn in currentColor strokes on transparent; the card tile supplies the bg.
 */

const STROKE = {
  acetate: 3.2,
  'TR-90': 2.6,
  nylon: 2.2,
  combo: 1.2, // browline eyewire; the brow bar is drawn separately
  metal: 1.4,
  titanium: 1.2,
};

const fx = (n) => String(Math.round(n * 100) / 100);

function safeNum(v, fallback) {
  return Number.isFinite(v) ? v : fallback;
}

/* Closed rounded box centered at the origin; separate top/bottom corner radii.
   Also returns the open top-half path used for semi-rimless treatments. */
function roundedBox(w, h, rt, rb) {
  const d = [
    `M ${fx(-w + rt)} ${fx(-h)}`,
    `H ${fx(w - rt)}`,
    `Q ${fx(w)} ${fx(-h)} ${fx(w)} ${fx(-h + rt)}`,
    `V ${fx(h - rb)}`,
    `Q ${fx(w)} ${fx(h)} ${fx(w - rb)} ${fx(h)}`,
    `H ${fx(-w + rb)}`,
    `Q ${fx(-w)} ${fx(h)} ${fx(-w)} ${fx(h - rb)}`,
    `V ${fx(-h + rt)}`,
    `Q ${fx(-w)} ${fx(-h)} ${fx(-w + rt)} ${fx(-h)}`,
    'Z',
  ].join(' ');
  const top =
    `M ${fx(-w)} 0 V ${fx(-h + rt)} Q ${fx(-w)} ${fx(-h)} ${fx(-w + rt)} ${fx(-h)} ` +
    `H ${fx(w - rt)} Q ${fx(w)} ${fx(-h)} ${fx(w)} ${fx(-h + rt)} V 0`;
  return { d, top };
}

function ellipseLens(w, h) {
  return {
    d: `M ${fx(-w)} 0 A ${fx(w)} ${fx(h)} 0 1 1 ${fx(w)} 0 A ${fx(w)} ${fx(h)} 0 1 1 ${fx(-w)} 0 Z`,
    top: `M ${fx(-w)} 0 A ${fx(w)} ${fx(h)} 0 0 1 ${fx(w)} 0`,
  };
}

/* Closed polygon with every corner rounded by radius r. */
function roundedPoly(pts, r) {
  const n = pts.length;
  const parts = [];
  for (let i = 0; i < n; i++) {
    const p = pts[i];
    const prev = pts[(i - 1 + n) % n];
    const next = pts[(i + 1) % n];
    const lIn = Math.hypot(p[0] - prev[0], p[1] - prev[1]);
    const lOut = Math.hypot(next[0] - p[0], next[1] - p[1]);
    const rIn = Math.min(r, lIn / 2);
    const rOut = Math.min(r, lOut / 2);
    const ax = p[0] - ((p[0] - prev[0]) / lIn) * rIn;
    const ay = p[1] - ((p[1] - prev[1]) / lIn) * rIn;
    const bx = p[0] + ((next[0] - p[0]) / lOut) * rOut;
    const by = p[1] + ((next[1] - p[1]) / lOut) * rOut;
    parts.push(`${i === 0 ? 'M' : 'L'} ${fx(ax)} ${fx(ay)} Q ${fx(p[0])} ${fx(p[1])} ${fx(bx)} ${fx(by)}`);
  }
  return { d: `${parts.join(' ')} Z` };
}

function pantoLens(w, h) {
  const rt = 0.2 * h;
  const d = [
    `M ${fx(-w + rt)} ${fx(-h)}`,
    `H ${fx(w - rt)}`,
    `Q ${fx(w)} ${fx(-h)} ${fx(w)} ${fx(-h + rt)}`,
    `V ${fx(-0.15 * h)}`,
    `C ${fx(w)} ${fx(0.62 * h)} ${fx(0.58 * w)} ${fx(h)} 0 ${fx(h)}`,
    `C ${fx(-0.58 * w)} ${fx(h)} ${fx(-w)} ${fx(0.62 * h)} ${fx(-w)} ${fx(-0.15 * h)}`,
    `V ${fx(-h + rt)}`,
    `Q ${fx(-w)} ${fx(-h)} ${fx(-w + rt)} ${fx(-h)}`,
    'Z',
  ].join(' ');
  return { d };
}

/* Lens generators, keyed by catalog shape. Coordinates are lens-relative:
   origin at the lens center, temporal edge at -x, nasal edge at +x (the left
   lens is drawn; the right lens is a mirror). w = A/2, h = B/2 in mm. */
const LENS = {
  rectangle: (w, h) => roundedBox(w, h, 0.22 * h, 0.3 * h),
  square: (w, h) => roundedBox(w, h, 0.16 * h, 0.16 * h),
  panto: pantoLens,
  round: ellipseLens,
  oval: ellipseLens,
  aviator: (w, h) => ({
    d: [
      `M ${fx(-w)} ${fx(-0.45 * h)}`,
      `C ${fx(-w)} ${fx(-0.85 * h)} ${fx(-0.6 * w)} ${fx(-h)} ${fx(-0.1 * w)} ${fx(-h)}`,
      `C ${fx(0.5 * w)} ${fx(-h)} ${fx(w)} ${fx(-0.85 * h)} ${fx(w)} ${fx(-0.45 * h)}`,
      `C ${fx(w)} ${fx(-0.05 * h)} ${fx(0.85 * w)} ${fx(0.35 * h)} ${fx(0.5 * w)} ${fx(0.68 * h)}`,
      `C ${fx(0.15 * w)} ${fx(h)} ${fx(-0.5 * w)} ${fx(h)} ${fx(-0.72 * w)} ${fx(0.72 * h)}`,
      `C ${fx(-0.93 * w)} ${fx(0.46 * h)} ${fx(-w)} ${fx(0.05 * h)} ${fx(-w)} ${fx(-0.45 * h)}`,
      'Z',
    ].join(' '),
  }),
  'cat-eye': (w, h) => ({
    d: [
      `M ${fx(-0.98 * w)} ${fx(-0.72 * h)}`,
      `C ${fx(-0.55 * w)} ${fx(-1.05 * h)} ${fx(-0.05 * w)} ${fx(-0.95 * h)} ${fx(0.42 * w)} ${fx(-0.82 * h)}`,
      `C ${fx(0.8 * w)} ${fx(-0.72 * h)} ${fx(w)} ${fx(-0.45 * h)} ${fx(w)} ${fx(-0.08 * h)}`,
      `C ${fx(w)} ${fx(0.5 * h)} ${fx(0.55 * w)} ${fx(h)} ${fx(-0.02 * w)} ${fx(h)}`,
      `C ${fx(-0.6 * w)} ${fx(h)} ${fx(-0.9 * w)} ${fx(0.62 * h)} ${fx(-w)} ${fx(0.12 * h)}`,
      `C ${fx(-1.04 * w)} ${fx(-0.28 * h)} ${fx(-1.06 * w)} ${fx(-0.55 * h)} ${fx(-0.98 * w)} ${fx(-0.72 * h)}`,
      'Z',
    ].join(' '),
  }),
  browline: (w, h) => ({
    ...pantoLens(w, h),
    extra: [
      {
        d: `M ${fx(-w - 1)} ${fx(-h + 0.6)} H ${fx(w + 1)}`,
        attrs: 'stroke-width="3.4" stroke-linecap="round"',
      },
    ],
  }),
  wayfarer: (w, h) =>
    roundedPoly(
      [
        [-w, -h],
        [w, -0.82 * h],
        [0.82 * w, h],
        [-0.9 * w, h],
      ],
      0.24 * h,
    ),
  geometric: (w, h) =>
    roundedPoly(
      [
        [-0.5 * w, -h],
        [0.5 * w, -h],
        [w, -0.4 * h],
        [w, 0.4 * h],
        [0.5 * w, h],
        [-0.5 * w, h],
        [-w, 0.4 * h],
        [-w, -0.4 * h],
      ],
      1.6,
    ),
  rimless: (w, h) => roundedBox(w, h, 0.4 * h, 0.45 * h),
  wrap: (w, h) => ({
    d: [
      `M ${fx(-w)} ${fx(-0.7 * h)}`,
      `C ${fx(-0.7 * w)} ${fx(-1.02 * h)} ${fx(-0.2 * w)} ${fx(-h)} ${fx(0.3 * w)} ${fx(-h)}`,
      `C ${fx(0.72 * w)} ${fx(-h)} ${fx(w)} ${fx(-0.8 * h)} ${fx(w)} ${fx(-0.42 * h)}`,
      `C ${fx(w)} ${fx(0.18 * h)} ${fx(0.85 * w)} ${fx(0.72 * h)} ${fx(0.5 * w)} ${fx(0.88 * h)}`,
      `C ${fx(0.12 * w)} ${fx(h)} ${fx(-0.42 * w)} ${fx(0.92 * h)} ${fx(-0.68 * w)} ${fx(0.62 * h)}`,
      `C ${fx(-0.88 * w)} ${fx(0.38 * h)} ${fx(-w)} ${fx(-0.05 * h)} ${fx(-w)} ${fx(-0.7 * h)}`,
      'Z',
    ].join(' '),
  }),
};

function bridgePaths(style, x1, x2, h, sw) {
  const dblW = x2 - x1;
  const y = (f) => fx(38 + f * h);
  const arc = (yEdge, yTop, width) =>
    `<path d="M ${fx(x1)} ${y(yEdge)} C ${fx(x1 + 0.3 * dblW)} ${y(yTop)} ` +
    `${fx(x2 - 0.3 * dblW)} ${y(yTop)} ${fx(x2)} ${y(yEdge)}" stroke-width="${fx(width)}"/>`;
  const heavy = Math.min(sw, 2.4);
  switch (style) {
    case 'keyhole':
      return arc(-0.5, -0.8, heavy) + arc(-0.26, 0.08, heavy);
    case 'double':
      return (
        `<path d="M ${fx(x1 - 2)} ${y(-0.8)} L ${fx(x2 + 2)} ${y(-0.8)}" stroke-width="1.4"/>` +
        arc(-0.35, -0.58, 1.4)
      );
    case 'pad-arm':
      return arc(-0.42, -0.72, 1.1);
    default: // saddle
      return arc(-0.42, -0.75, heavy);
  }
}

function padPaths(x1, x2, h) {
  const cy = 38 + 0.18 * h;
  const pad = (cx, angle) =>
    `<ellipse cx="${fx(cx)}" cy="${fx(cy)}" rx="1.3" ry="2.3" stroke-width="0.8" ` +
    `transform="rotate(${angle} ${fx(cx)} ${fx(cy)})"/>`;
  return pad(x1 + 1.7, -18) + pad(x2 - 1.7, 18);
}

/** Render a catalog frame (or a match's `frame` object) as an inline SVG string. */
export function frameArtSVG(frame) {
  const a = safeNum(frame?.a_mm, 52);
  const b = safeNum(frame?.b_mm, 38);
  const dbl = safeNum(frame?.dbl_mm, 18);
  const gen = (LENS[frame?.shape] ?? LENS.rectangle)(a / 2, b / 2);
  const rim = frame?.rim ?? 'full';
  const sw = STROKE[frame?.material] ?? 2;
  const w = a / 2;
  const h = b / 2;
  const cx = 75 - dbl / 2 - w; // left lens center
  const x1 = 75 - dbl / 2; // nasal edges
  const x2 = 75 + dbl / 2;

  const lens = [];
  if (rim === 'rimless') {
    // The lens itself is glass — faint outline, with the mount hardware in full ink.
    lens.push(`<path d="${gen.d}" stroke-width="0.6" stroke-opacity="0.55"/>`);
    lens.push(`<rect x="${fx(w - 2.6)}" y="-1.5" width="2.4" height="3" rx="0.8" fill="currentColor" stroke="none"/>`);
    lens.push(`<rect x="${fx(-w + 0.2)}" y="-1.5" width="2.4" height="3" rx="0.8" fill="currentColor" stroke="none"/>`);
  } else if (rim === 'semi' && gen.top) {
    lens.push(`<path d="${gen.d}" stroke-width="0.7"/>`); // nylon cord
    lens.push(`<path d="${gen.top}" stroke-width="${fx(sw)}"/>`);
  } else {
    lens.push(`<path d="${gen.d}" stroke-width="${fx(sw)}"/>`);
  }
  for (const ex of gen.extra ?? []) lens.push(`<path d="${ex.d}" ${ex.attrs}/>`);
  // Endpiece stub hinting at the temple without drawing it.
  lens.push(
    `<path d="M ${fx(-w)} ${fx(-0.62 * h)} L ${fx(-w - 4.5)} ${fx(-0.74 * h)}" ` +
      `stroke-width="${fx(Math.min(sw, 2.6))}" stroke-linecap="round"/>`,
  );
  const lensMarkup = lens.join('');

  const pads =
    frame?.nose_pads === 'adjustable' || frame?.bridge_style === 'pad-arm'
      ? padPaths(x1, x2, h)
      : '';

  return (
    `<svg class="frame-art" viewBox="0 0 150 76" aria-hidden="true" focusable="false" ` +
    `fill="none" stroke="currentColor" stroke-linejoin="round">` +
    `<g transform="translate(${fx(cx)} 38)">${lensMarkup}</g>` +
    `<g transform="translate(${fx(150 - cx)} 38) scale(-1 1)">${lensMarkup}</g>` +
    bridgePaths(frame?.bridge_style, x1, x2, h, sw) +
    pads +
    `</svg>`
  );
}
