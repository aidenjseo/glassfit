// PD / Rx / lens-intent form: validation + analyze-payload assembly.

import { $, setError } from './dom.js';

export function initForms() {
  const pd = $('pd');
  pd.addEventListener('input', () => {
    const v = parseFloat(pd.value);
    $('pd-warn').hidden = !(Number.isFinite(v) && v >= 45 && v <= 80 && (v < 54 || v > 74));
    setError($('pd-err'), '');
  });
}

/** Validate the review form; returns the /recommendations payload or null (errors shown inline). */
export function buildAnalyzePayload(scanId) {
  let ok = true;
  const pdVal = parseFloat($('pd').value);
  if (!Number.isFinite(pdVal) || pdVal < 45 || pdVal > 80) {
    setError($('pd-err'), 'Enter your PD in millimeters (45–80). It is printed on most Rx forms.');
    $('pd').focus();
    ok = false;
  } else {
    setError($('pd-err'), '');
  }

  const payload = {
    scan_id: scanId,
    pd_mm: pdVal,
    lens_intent: $('lens-intent').value,
  };

  const od = parseFloat($('pd-od').value);
  const os = parseFloat($('pd-os').value);
  const monoErr = $('mono-err');
  setError(monoErr, '');
  if (Number.isFinite(od) || Number.isFinite(os)) {
    if (!Number.isFinite(od) || !Number.isFinite(os)) {
      setError(monoErr, 'Enter both monocular values, or clear both.');
      ok = false;
    } else if (Math.abs(od + os - pdVal) > 1.0) {
      setError(monoErr, `OD + OS should equal your PD within 1 mm (got ${(od + os).toFixed(1)}).`);
      ok = false;
    } else {
      payload.pd_monocular = { right: od, left: os }; // right = OD, left = OS
    }
  }

  const rxErr = $('rx-err');
  setError(rxErr, '');
  const rxRaw = {
    od: { sphere: $('rx-od-sph').value, cyl: $('rx-od-cyl').value, axis: $('rx-od-axis').value },
    os: { sphere: $('rx-os-sph').value, cyl: $('rx-os-cyl').value, axis: $('rx-os-axis').value },
  };
  const anyRx = Object.values(rxRaw).some((eye) => Object.values(eye).some((v) => v !== ''));
  if (anyRx) {
    const rx = {};
    for (const [eye, vals] of Object.entries(rxRaw)) {
      const cyl = parseFloat(vals.cyl) || 0;
      const axis = parseInt(vals.axis, 10);
      if (cyl !== 0 && !Number.isFinite(axis)) {
        setError(rxErr, `Axis is required when ${eye.toUpperCase()} has a cylinder value.`);
        ok = false;
      }
      rx[eye] = {
        sphere: parseFloat(vals.sphere) || 0,
        cyl,
        axis: Number.isFinite(axis) ? axis : 0,
      };
    }
    if (ok) payload.rx = rx;
  }

  return ok ? payload : null;
}

/** Build the two 1–5 segmented controls and the two yes/no controls. */
export function buildFeedbackControls() {
  document.querySelectorAll('.segmented[data-name]').forEach((group) => {
    const name = group.dataset.name;
    const options = group.classList.contains('yesno')
      ? [
          { value: 'false', label: 'No' },
          { value: 'true', label: 'Yes' },
        ]
      : [1, 2, 3, 4, 5].map((n) => ({ value: String(n), label: String(n) }));
    group.textContent = '';
    for (const opt of options) {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = name;
      input.value = opt.value;
      const span = document.createElement('span');
      span.textContent = opt.label;
      label.append(input, span);
      group.appendChild(label);
    }
  });
}

/** Populate the "frame you tried" radio list from the match shortlist. */
export function buildFrameChoice(matches) {
  const fieldset = $('frame-choice-fieldset');
  const container = $('frame-choice');
  container.textContent = '';
  if (!matches || !matches.length) {
    fieldset.hidden = true;
    return;
  }
  const options = [
    ...matches.map((m) => ({ value: m.frame.frame_id, label: `${m.frame.name} (${m.frame.frame_id})` })),
    { value: '', label: 'My own / another frame' },
  ];
  for (const [i, opt] of options.entries()) {
    const label = document.createElement('label');
    label.className = 'frame-option';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'tried_frame';
    input.value = opt.value;
    input.checked = i === options.length - 1; // default: own frame
    const span = document.createElement('span');
    span.textContent = opt.label;
    label.append(input, span);
    container.appendChild(label);
  }
  fieldset.hidden = false;
}

/** Validate + assemble the /feedback payload, or null with an inline error. */
export function buildFeedbackPayload(recommendationId) {
  const read = (name) => {
    const checked = document.querySelector(`input[name="${name}"]:checked`);
    return checked ? checked.value : null;
  };
  const nose = read('nose_pressure');
  const temple = read('temple_pressure');
  const slips = read('slips');
  const cheek = read('cheek_touch');
  const err = $('fb-err');
  if (nose == null || temple == null || slips == null || cheek == null) {
    setError(err, 'Answer all four fit questions first.');
    return null;
  }
  setError(err, '');
  const payload = {
    recommendation_id: recommendationId,
    nose_pressure: parseInt(nose, 10),
    temple_pressure: parseInt(temple, 10),
    slips: slips === 'true',
    cheek_touch: cheek === 'true',
  };
  const triedFrame = read('tried_frame');
  if (triedFrame) payload.frame_id = triedFrame;
  const adjustments = {
    panto_delta_deg: parseFloat($('adj-panto').value),
    temple_bend_delta_mm: parseFloat($('adj-bend').value),
    left_raise_delta_mm: parseFloat($('adj-raise').value),
    nose_pad_splay_delta_deg: parseFloat($('adj-splay').value),
  };
  const entered = Object.fromEntries(
    Object.entries(adjustments).filter(([, v]) => Number.isFinite(v))
  );
  if (Object.keys(entered).length) payload.adjustments = entered;
  const comments = $('fb-comments').value.trim();
  if (comments) payload.comments = comments;
  return payload;
}
