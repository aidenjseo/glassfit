// Tiny shared DOM helpers.

export const $ = (id) => document.getElementById(id);

/** Show `message` in an inline error/warning element, or hide it when falsy. */
export function setError(el, message) {
  el.textContent = message || '';
  el.hidden = !message;
}
