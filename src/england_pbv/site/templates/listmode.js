// Per-list List/Thumbnails toggles.
//
// Any element with data-list="<id>" is an independent toggle scope: it contains a
// .view-toggle with two buttons (.mode-list / .mode-thumbs), and the chosen mode is
// stored per list (localStorage "pbv-mode:<id>", falling back to the legacy global
// "pbv-list-mode" so an old preference carries over as the default everywhere).
// Server-rendered lists switch via a .thumbs class on the scope element (CSS shows
// .mode-list-only or .mode-thumbs-only); client-rendered lists register an onChange
// callback and consult ListMode.get(id) when they build their HTML.
const ListMode = (function () {
  const KEY_PREFIX = 'pbv-mode:';
  const LEGACY_KEY = 'pbv-list-mode';
  const callbacks = {};
  // In-memory choices take precedence over storage: with site data blocked
  // (getItem/setItem throw) the toggle must still switch — only persistence
  // across visits is lost.
  const chosen = {};

  function stored(id) {
    try {
      return localStorage.getItem(KEY_PREFIX + id) || localStorage.getItem(LEGACY_KEY);
    } catch (e) { return null; }
  }
  function get(id) {
    const mode = chosen[id] !== undefined ? chosen[id] : stored(id);
    return mode === 'thumbs' ? 'thumbs' : 'list';
  }

  function apply(el, id) {
    const thumbs = get(id) === 'thumbs';
    el.classList.toggle('thumbs', thumbs);
    const listBtn = el.querySelector('.view-toggle .mode-list');
    const thumbsBtn = el.querySelector('.view-toggle .mode-thumbs');
    if (listBtn) {
      listBtn.classList.toggle('active', !thumbs);
      listBtn.setAttribute('aria-pressed', String(!thumbs));
    }
    if (thumbsBtn) {
      thumbsBtn.classList.toggle('active', thumbs);
      thumbsBtn.setAttribute('aria-pressed', String(thumbs));
    }
  }

  function set(id, mode, el) {
    chosen[id] = mode;
    try { localStorage.setItem(KEY_PREFIX + id, mode); } catch (e) { /* private mode */ }
    apply(el, id);
    (callbacks[id] || []).forEach(fn => { try { fn(mode); } catch (e) { /* list not on screen */ } });
  }

  function wireAll() {
    document.querySelectorAll('[data-list]').forEach(el => {
      const id = el.getAttribute('data-list');
      apply(el, id);
      const listBtn = el.querySelector('.view-toggle .mode-list');
      const thumbsBtn = el.querySelector('.view-toggle .mode-thumbs');
      if (listBtn) listBtn.addEventListener('click', () => set(id, 'list', el));
      if (thumbsBtn) thumbsBtn.addEventListener('click', () => set(id, 'thumbs', el));
    });
  }

  function onChange(id, fn) { (callbacks[id] = callbacks[id] || []).push(fn); }

  return { get: get, wireAll: wireAll, onChange: onChange };
})();

// Public-access badge for client-rendered lists. Mirrors the access_badge Jinja
// macro (_badges.j2); codes come from points.js rows (r[6]=code, r[7]=path metres):
// 0 open-access land, 1 on a public path, 2 path nearby, 3 no recorded access.
function accessBadgeHtml(code, pathM) {
  // No data (degraded fallback rows, cached old points.js) -> no badge: a missing
  // fact must not render as the definitive "no recorded access" claim.
  if (code !== 0 && code !== 1 && code !== 2 && code !== 3) return '';
  if (code === 0) {
    return '<span class="access-pill access-open" title="Open-access land (CRoW Act 2000) — right to roam on foot. Computed from Natural England open data; check local restrictions.">open access</span>';
  }
  if (code === 1) {
    return '<span class="access-pill access-path" title="A public right of way passes within ' + pathM + ' m (OpenStreetMap). Computed — verify locally.">footpath</span>';
  }
  if (code === 2) {
    return '<span class="access-pill access-near" title="Nearest public right of way about ' + pathM + ' m away (OpenStreetMap). The last stretch may cross private land.">path ' + pathM + '&nbsp;m</span>';
  }
  return '<span class="access-pill access-none" title="No public right of way or access land recorded at this exact spot — it may be private land. Check before visiting.">access?</span>';
}
