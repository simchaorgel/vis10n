// Strata: edge-to-edge bands of colour, bent by a warped noise field, each one laid over the
// next like cut paper with a soft shadow beneath. Computed small (it's all soft), upscaled, grained.
// Usage: const seed = strataSeed(STRATA_PRESETS.dusk);
//        mountCanvasBg(canvas, c => paintStrataBg(c, STRATA_PRESETS.dusk, seed)) — needs canvas-bg.js.
// The seed is picked once, so repaints on resize keep the same spread.

const STRATA_PRESETS = {
  dusk: {
    seed: null,          // a number keeps one spread; null picks a new one each page load
    bands: 14,           // how many layers cross the canvas
    warp: 1.1,           // how far the layers bend
    edge: 0.2,           // softness of each layer's edge, in band widths
    shadow: 0.16,        // depth of the shadow under each layer's edge
    grain: 24,
    colours: [           // walked through in order, then back again
      [34, 38, 112],     // ultramarine
      [92, 104, 196],    // periwinkle
      [164, 160, 228],   // lilac
      [232, 196, 220],   // pink haze
      [248, 226, 176],   // butter
      [242, 146, 116],   // coral
      [214, 88, 96],     // rosewood
    ],
  },
};

function strataSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}

function paintStrataBg(canvas, preset, seed) {
  const { bands: BANDS, warp: WARP, edge: EDGE, shadow: SHADOW, grain: GRAIN, colours: COLOURS } = preset;

  // --- seeded value noise ---
  const makeRandom = s0 => {
    let s = s0 >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  const PERM = new Uint8Array(512), VALS = new Float32Array(256);
  const rand = makeRandom(seed);
  for (let i = 0; i < 256; i++) { PERM[i] = i; VALS[i] = rand(); }
  for (let i = 255; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [PERM[i], PERM[j]] = [PERM[j], PERM[i]]; }
  for (let i = 0; i < 256; i++) PERM[i + 256] = PERM[i];
  function noise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = PERM[(xi & 255) + PERM[yi & 255]], b = PERM[((xi + 1) & 255) + PERM[yi & 255]];
    const c = PERM[(xi & 255) + PERM[(yi + 1) & 255]], d = PERM[((xi + 1) & 255) + PERM[(yi + 1) & 255]];
    const top = VALS[a] + (VALS[b] - VALS[a]) * u, bot = VALS[c] + (VALS[d] - VALS[c]) * u;
    return top + (bot - top) * v;
  }
  function fbm(x, y) {   // layered noise, 0..1
    let sum = 0, amp = 0.5, norm = 0;
    for (let o = 0; o < 4; o++) { sum += amp * noise(x, y); norm += amp; x = x * 2.03 + 17.1; y = y * 2.03 + 9.7; amp *= 0.35; }
    return sum / norm;
  }

  // colour of band k: walk forward through the colours, then back, so neighbours always sit well together
  function bandColour(k, offset) {
    const n = COLOURS.length, period = 2 * (n - 1);
    let i = ((k + offset) % period + period) % period;
    if (i >= n) i = period - i;
    return COLOURS[i];
  }
  const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  // --- painting ---
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { w, h } = fitBgCanvas(canvas);
  const r = makeRandom(seed ^ 0x5bd1e995);

  // the layers lean along a random direction rather than lying flat
  const angle = (r() - 0.5) * 1.4 + Math.PI / 2;
  const dx = Math.cos(angle), dy = Math.sin(angle);
  const offset = Math.floor(r() * 12);

  const sw = Math.max(2, Math.round(w / 3)), sh = Math.max(2, Math.round(h / 3));
  const small = document.createElement('canvas');
  small.width = sw; small.height = sh;
  const sctx = small.getContext('2d');
  const id = sctx.createImageData(sw, sh);
  const aspect = sw / sh;

  for (let j = 0; j < sh; j++) {
    for (let i = 0; i < sw; i++) {
      const px = (i / sw - 0.5) * aspect, py = j / sh - 0.5;
      // fold the coordinates once through the noise, then measure distance along the lean
      const qx = fbm(px * 0.8 + 3.1, py * 0.8 + 7.7), qy = fbm(px * 0.8 + 8.4, py * 0.8 + 1.9);
      const f = (px * dx + py * dy) + WARP * (fbm(px * 0.7 + qx * 1.2, py * 0.7 + qy * 1.2) - 0.5);
      const pos = (f + 0.9) * BANDS * 0.62;
      const k = Math.floor(pos), frac = pos - k;
      // soft step into the next layer near the boundary
      const blend = smooth(1 - EDGE, 1, frac);
      const c0 = bandColour(k, offset), c1 = bandColour(k + 1, offset);
      const cr = c0[0] + (c1[0] - c0[0]) * blend, cg = c0[1] + (c1[1] - c0[1]) * blend, cb = c0[2] + (c1[2] - c0[2]) * blend;
      // each layer sits on the one before: a shadow just inside its leading edge, then a faint lift
      const shade = 1 - SHADOW * (1 - smooth(0, 0.35, frac)) + 0.04 * smooth(0.4, 0.9, frac);
      // gentle vignette
      const vx = i / sw - 0.5, vy = j / sh - 0.5;
      const vig = 1 - 0.3 * (vx * vx + vy * vy) * 2;
      const o = (j * sw + i) * 4;
      id.data[o] = cr * shade * vig; id.data[o + 1] = cg * shade * vig; id.data[o + 2] = cb * shade * vig; id.data[o + 3] = 255;
    }
  }
  sctx.putImageData(id, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w, h);

  // fine grain over the top, also hides any upscale banding
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    let n = (Math.random() + Math.random() - 1) * GRAIN;
    if (n < 0) n *= 0.85;
    d[i] += n; d[i + 1] += n * 0.98; d[i + 2] += n * 1.05;
  }
  ctx.putImageData(img, 0, 0);
}
