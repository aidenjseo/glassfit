// Results rendering: measurements table + recommendation cards + fitting guidance.
// A dumb walk over the /recommendations response — field names mirror the backend schemas.

const mm = (v) => Number(v).toFixed(1); // one decimal — used for degrees too
const deg = mm;
const pct = (v) => `${Math.round(v * 100)}`;
const rl = (ps, f = mm) => `${f(ps.right)} / ${f(ps.left)}`;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function measurementRows(m) {
  return [
    ['Pupillary geometry', [
      ['Binocular PD', mm(m.pd_binocular_mm), 'mm', 'pd_binocular_mm'],
      ['Monocular PD — your right (OD)', mm(m.pd_monocular_mm.right), 'mm', 'pd_monocular_mm'],
      ['Monocular PD — your left (OS)', mm(m.pd_monocular_mm.left), 'mm', 'pd_monocular_mm'],
      ['Pupil height in eye opening (R / L)', rl(m.pupil_height_ratio, pct), '%', 'pupil_height_ratio'],
      ['Canthal tilt (R / L)', rl(m.canthal_tilt_deg), 'deg', 'canthal_tilt_deg'],
    ]],
    ['Nose bridge', [
      ['Bridge width at crest', mm(m.bridge.at_crest_mm), 'mm', 'bridge'],
      ['Bridge width 10 mm below', mm(m.bridge.below_10mm_mm), 'mm', 'bridge'],
      ['Bridge width 15 mm below', mm(m.bridge.below_15mm_mm), 'mm', 'bridge'],
      ['Bridge crest height', mm(m.bridge_crest_height_mm), 'mm', 'bridge_crest_height_mm'],
    ]],
    ['Face widths', [
      ['Cheekbone (zygoma) width', mm(m.zygoma_width_mm), 'mm', 'zygoma_width_mm'],
      ['Temple-to-temple width', mm(m.temple_width_mm), 'mm', 'temple_width_mm'],
      ['Face wrap radius', mm(m.face_wrap_radius_mm), 'mm', 'face_wrap_radius_mm'],
      ['Cheek clearance (R / L)', rl(m.cheek_clearance_mm), 'mm', 'cheek_clearance_mm'],
    ]],
    ['Ears & temples', [
      ['Hinge to ear (R / L)', rl(m.hinge_to_ear_mm), 'mm', 'hinge_to_ear_mm'],
      ['Ear height asymmetry', mm(m.ear_height_asymmetry_mm), 'mm', 'ear_height_asymmetry_mm'],
      ['Behind-ear drop (R / L)', `${mm(m.behind_ear.right.drop_mm)} / ${mm(m.behind_ear.left.drop_mm)}`, 'mm', 'behind_ear'],
    ]],
    ['Depth', [
      ['Vertex distance estimate', mm(m.vertex_estimate_mm), 'mm', 'vertex_estimate_mm'],
    ]],
  ];
}

function buildMeasureTable(m) {
  const lowConfidence = new Set(m.quality.low_confidence_fields || []);
  const refinedFromSides = (m.quality.warnings || []).includes(
    'hinge_to_ear_refined_from_side_views'
  );
  const table = el('table', { class: 'measures' });
  table.appendChild(el('caption', { text: 'Your measurements' }));
  for (const [group, rows] of measurementRows(m)) {
    const tbody = el('tbody');
    const th = el('th', { scope: 'rowgroup', text: group });
    tbody.appendChild(el('tr', {}, [th]));
    th.setAttribute('colspan', '3');
    for (const [label, value, unit, field] of rows) {
      const name = el('td', { text: label });
      if (field === 'hinge_to_ear_mm' && refinedFromSides) {
        name.appendChild(el('span', {
          class: 'badge-refined',
          text: 'side-scan',
          title: 'Measured from your head-turn views',
        }));
      } else if (lowConfidence.has(field)) {
        name.appendChild(el('span', {
          class: 'badge-est',
          text: 'est.',
          title: 'Estimated — the face mesh has no ear landmarks',
        }));
      }
      tbody.appendChild(el('tr', {}, [
        name,
        el('td', { class: 'val', text: value }),
        el('td', { class: 'unit', text: unit }),
      ]));
    }
    table.appendChild(tbody);
  }
  return el('div', { class: 'measure-block' }, [table]);
}

const FRAME_SVG = `
<svg class="frame-svg" viewBox="0 0 220 76" role="img" aria-label="Frame diagram: A is lens width, B is lens height, DBL is bridge width">
  <g fill="none" stroke="currentColor" stroke-width="2">
    <rect x="12" y="14" width="80" height="44" rx="12"/>
    <rect x="128" y="14" width="80" height="44" rx="12"/>
    <path d="M92 26 Q110 16 128 26"/>
  </g>
  <g fill="currentColor" font-size="10" text-anchor="middle">
    <text x="52" y="70">A</text><text x="110" y="40">DBL</text><text x="4" y="40">B</text>
  </g>
</svg>`;

function card(title, children) {
  return el('div', { class: 'card' }, [el('h2', { text: title }), ...children]);
}

function dl(entries) {
  const node = el('dl');
  for (const [term, val, target] of entries) {
    node.appendChild(el('dt', { text: term }));
    const dd = el('dd', { text: val });
    if (target) dd.appendChild(el('span', { class: 'target', text: ` ${target}` }));
    node.appendChild(dd);
  }
  return node;
}

function meter(label, value) {
  const wrap = el('div', { class: 'meter' });
  wrap.appendChild(el('div', { class: 'label' }, [
    el('span', { text: label }),
    el('span', { text: `${Math.round(value * 100)}%` }),
  ]));
  const track = el('div', { class: 'track' });
  const fill = el('div', { class: 'fill' });
  fill.style.width = `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`;
  track.appendChild(fill);
  wrap.appendChild(track);
  return wrap;
}

function buildCards(data) {
  const { frame, as_worn: asWorn, optics, nose_pads: pads, temples, comfort } = data;
  const cards = el('div', { class: 'cards' });

  const frameCard = card('Frame size', [
    el('div', { class: 'big', text: `${Math.round(frame.a_mm)}▪${Math.round(frame.dbl_mm)}▪${Math.round(frame.temple_length_mm)}` }),
    dl([
      ['A — lens width', `${mm(frame.a_mm)} mm`],
      ['B — lens height', `${mm(frame.b_mm)} mm`],
      ['DBL — bridge', `${mm(frame.dbl_mm)} mm`],
      ['ED — effective diameter', `${mm(frame.ed_mm)} mm`],
      ['Temple length', `${mm(frame.temple_length_mm)} mm`],
    ]),
  ]);
  frameCard.insertAdjacentHTML('beforeend', FRAME_SVG);
  cards.appendChild(frameCard);

  cards.appendChild(card('As worn', [dl([
    ['Pantoscopic tilt', `${deg(asWorn.pantoscopic_deg)}°`, 'target 5–8°'],
    ['Face-form (wrap)', `${deg(asWorn.face_form_deg)}°`, 'target 5–10°'],
    ['Vertex distance', `${mm(asWorn.vertex_mm)} mm`, 'target 12–14'],
  ])]));

  cards.appendChild(card('Optics', [dl([
    ['Monocular PD (R / L)', `${rl(optics.pd_monocular_mm)} mm`],
    ['OC height (R / L)', `${rl(optics.oc_height_mm)} mm`],
    ['Inset (R / L)', `${rl(optics.inset_mm)} mm`],
  ])]));

  cards.appendChild(card('Nose pads', [dl([
    ['Pad size', pads.size],
    ['Splay', `${deg(pads.splay_deg)}°`],
    ['Flare', `${deg(pads.flare_deg)}°`],
    ['Drop', `${mm(pads.drop_mm)} mm`],
  ])]));

  cards.appendChild(card('Temples', [dl([
    ['Bend from hinge (R / L)', `${rl(temples.bend_point_mm_from_hinge)} mm`],
    ['Tip angle', `${deg(temples.tip_angle_deg)}°`],
    ['Raise (R / L)', `${rl(temples.raise_mm)} mm`],
  ])]));

  cards.appendChild(card('Comfort forecast', [
    meter('Slip risk', comfort.predicted_slip),
    meter('Nose pressure', comfort.nose_pressure),
    meter('Temple pressure', comfort.temple_pressure),
  ]));

  return cards;
}

const fmtDelta = (d) => (d === 0 ? '' : ` (${d > 0 ? '+' : ''}${Number(d).toFixed(1)})`);

/** "Frames that fit you" — the ranked catalog shortlist from /frames/match. */
export function renderFrameMatches(root, data) {
  root.textContent = '';
  if (!data.matches.length) {
    root.appendChild(el('p', {
      class: 'hint',
      text: 'No catalog frames clear the fit filters for your dimensions — the measurements above still apply to any frame.',
    }));
    return;
  }
  root.appendChild(el('h2', { class: 'section-title', text: 'Frames that fit you' }));
  const grid = el('div', { class: 'cards' });
  for (const match of data.matches) {
    const { frame } = match;
    const cardEl = el('div', { class: 'card match-card' });
    const head = el('div', { class: 'match-head' });
    head.appendChild(el('span', { class: 'match-name', text: frame.name }));
    head.appendChild(el('span', { class: 'match-score', text: `${Math.round(match.fit_score * 100)}% fit` }));
    cardEl.appendChild(head);
    const chips = el('div', { class: 'chips' });
    for (const tag of [frame.shape, frame.material, frame.rim, frame.nose_pads.replace('_', ' ')]) {
      chips.appendChild(el('span', { class: 'chip', text: tag }));
    }
    cardEl.appendChild(chips);
    cardEl.appendChild(el('div', {
      class: 'match-dims',
      text:
        `A ${frame.a_mm}${fmtDelta(match.deltas.a_mm)} · ` +
        `DBL ${frame.dbl_mm}${fmtDelta(match.deltas.dbl_mm)} · ` +
        `B ${frame.b_mm}${fmtDelta(match.deltas.b_mm)} · ` +
        `temple ${frame.temple_mm}${fmtDelta(match.deltas.temple_length_mm)}`,
    }));
    cardEl.appendChild(meter('Dimensional fit', match.fit_score));
    if (match.flags.length) {
      const list = el('ul', { class: 'match-flags' });
      for (const flag of match.flags) list.appendChild(el('li', { text: flag }));
      cardEl.appendChild(list);
    }
    grid.appendChild(cardEl);
  }
  root.appendChild(grid);
}

export function renderResults(root, data) {
  root.textContent = '';
  if (data.measurements.quality.scale_suspect) {
    const warn = el('div', { class: 'panel error' });
    warn.appendChild(el('h2', { text: 'Check your PD entry' }));
    warn.appendChild(el('p', {
      text: 'The implied face size looks unusual for the PD you typed — the numbers below may be scaled wrong.',
    }));
    root.appendChild(warn);
  }
  root.appendChild(buildCards(data));
  const guidance = el('div', { class: 'guidance' });
  guidance.appendChild(el('h2', { text: 'Fitting guidance for your optician' }));
  const list = el('ul');
  for (const note of data.notes) list.appendChild(el('li', { text: note }));
  guidance.appendChild(list);
  root.appendChild(guidance);
  root.appendChild(buildMeasureTable(data.measurements));
}
