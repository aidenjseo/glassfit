// Results rendering: measurements table + recommendation cards + fitting guidance.
// A dumb walk over the /recommendations response — field names mirror the backend schemas.

const mm = (v) => `${Number(v).toFixed(1)}`;
const deg = (v) => `${Number(v).toFixed(1)}`;
const pct = (v) => `${Math.round(v * 100)}`;

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
  const side = (ps, fmt = mm) => ({ right: fmt(ps.right), left: fmt(ps.left) });
  const pdm = side(m.pd_monocular_mm);
  const tilt = side(m.canthal_tilt_deg, deg);
  const cheek = side(m.cheek_clearance_mm);
  const ear = side(m.hinge_to_ear_mm);
  const ratio = side(m.pupil_height_ratio, pct);
  return [
    ['Pupillary geometry', [
      ['Binocular PD', mm(m.pd_binocular_mm), 'mm', 'pd_binocular_mm'],
      ['Monocular PD — your right (OD)', pdm.right, 'mm', 'pd_monocular_mm'],
      ['Monocular PD — your left (OS)', pdm.left, 'mm', 'pd_monocular_mm'],
      ['Pupil height in eye opening (R / L)', `${ratio.right} / ${ratio.left}`, '%', 'pupil_height_ratio'],
      ['Canthal tilt (R / L)', `${tilt.right} / ${tilt.left}`, 'deg', 'canthal_tilt_deg'],
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
      ['Cheek clearance (R / L)', `${cheek.right} / ${cheek.left}`, 'mm', 'cheek_clearance_mm'],
    ]],
    ['Ears & temples', [
      ['Hinge to ear (R / L)', `${ear.right} / ${ear.left}`, 'mm', 'hinge_to_ear_mm'],
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
  const table = el('table', { class: 'measures' });
  table.appendChild(el('caption', { text: 'Your measurements' }));
  for (const [group, rows] of measurementRows(m)) {
    const tbody = el('tbody');
    const th = el('th', { scope: 'rowgroup', text: group });
    tbody.appendChild(el('tr', {}, [th]));
    th.setAttribute('colspan', '3');
    for (const [label, value, unit, field] of rows) {
      const name = el('td', { text: label });
      if (lowConfidence.has(field)) {
        name.appendChild(el('span', { class: 'badge-est', text: 'est.', title: 'Estimated — the face mesh has no ear landmarks' }));
      }
      tbody.appendChild(el('tr', {}, [name, el('td', { class: 'val', text: value }), el('td', { class: 'unit', text: unit })]));
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
    ['Monocular PD (R / L)', `${mm(optics.pd_monocular_mm.right)} / ${mm(optics.pd_monocular_mm.left)} mm`],
    ['OC height (R / L)', `${mm(optics.oc_height_mm.right)} / ${mm(optics.oc_height_mm.left)} mm`],
    ['Inset (R / L)', `${mm(optics.inset_mm.right)} / ${mm(optics.inset_mm.left)} mm`],
  ])]));

  cards.appendChild(card('Nose pads', [dl([
    ['Pad size', pads.size],
    ['Splay', `${deg(pads.splay_deg)}°`],
    ['Flare', `${deg(pads.flare_deg)}°`],
    ['Drop', `${mm(pads.drop_mm)} mm`],
  ])]));

  cards.appendChild(card('Temples', [dl([
    ['Bend from hinge (R / L)', `${mm(temples.bend_point_mm_from_hinge.right)} / ${mm(temples.bend_point_mm_from_hinge.left)} mm`],
    ['Tip angle', `${deg(temples.tip_angle_deg)}°`],
    ['Raise (R / L)', `${mm(temples.raise_mm.right)} / ${mm(temples.raise_mm.left)} mm`],
  ])]));

  cards.appendChild(card('Comfort forecast', [
    meter('Slip risk', comfort.predicted_slip),
    meter('Nose pressure', comfort.nose_pressure),
    meter('Temple pressure', comfort.temple_pressure),
  ]));

  return cards;
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
