// Marbled silk: smooth noise, warped through itself twice, mapped onto a colour ramp.
// Computed at low resolution (it's all soft), upscaled, then finished with fine grain.
// Usage: mountCanvasBg(canvas, c => paintSilkBg(c, SILK_PRESETS.silk, seed)) — needs canvas-bg.js.
// Pass the same seed on every repaint, so a resize redraws the same pattern rather than a new one.

const SILK_PRESETS = {
  // silk.html: navy through teal and seafoam to pale sand, with a copper thread
  silk: {
    seed: null,     // set a number to get the same pattern every time (see silkSeed)
    scale: 1.5,     // how many "folds" fit across the canvas
    warp: 2.2,      // how strongly the field folds back on itself
    grain: 18,
    ramp: [         // position 0..1 → colour
      [0.00, [10, 18, 32]],    // ink navy
      [0.35, [18, 62, 78]],    // deep teal
      [0.58, [44, 118, 118]],  // sea green
      [0.76, [138, 190, 172]], // seafoam
      [0.92, [214, 208, 186]], // pale sand
      [1.00, [232, 226, 206]],
    ],
    copper: [196, 112, 70],
  },
};

// Seeded value noise; each paint gets its own tables, so several canvases can use different seeds
function makeSilkNoise(seed) {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const PERM = new Uint8Array(512), VALS = new Float32Array(256);
  for (let i = 0; i < 256; i++) { PERM[i] = i; VALS[i] = rand(); }
  for (let i = 255; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [PERM[i], PERM[j]] = [PERM[j], PERM[i]]; }
  for (let i = 0; i < 256; i++) PERM[i + 256] = PERM[i];

  function noise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = PERM[(xi & 255) + PERM[yi & 255]], b = PERM[((xi + 1) & 255) + PERM[yi & 255]];
    const c = PERM[(xi & 255) + PERM[(yi + 1) & 255]], d = PERM[((xi + 1) & 255) + PERM[(yi + 1) & 255]];
    const top = VALS[a] + (VALS[b] - VALS[a]) * u, bot = VALS[c] + (VALS[d] - VALS[c]) * u;
    return top + (bot - top) * v;
  }
  return function fbm(x, y) {   // layered noise, 0..1
    let sum = 0, amp = 0.5, norm = 0;
    for (let o = 0; o < 4; o++) { sum += amp * noise(x, y); norm += amp; x = x * 2.03 + 17.1; y = y * 2.03 + 9.7; amp *= 0.42; }
    return sum / norm;
  };
}

function silkRamp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [p0, c0] = stops[i - 1], [p1, c1] = stops[i];
      const k = (t - p0) / (p1 - p0), s = k * k * (3 - 2 * k);
      return [c0[0] + (c1[0] - c0[0]) * s, c0[1] + (c1[1] - c0[1]) * s, c0[2] + (c1[2] - c0[2]) * s];
    }
  }
  return stops[stops.length - 1][1];
}

function paintSilkBg(canvas, preset, seed) {
  const { scale: SCALE, warp: WARP, grain: GRAIN, ramp: RAMP, copper: COPPER } = preset;
  const fbm = makeSilkNoise(seed);
  const ctx = canvas.getContext('2d');
  const { w, h } = fitBgCanvas(canvas);

  // the field is soft, so compute it small and let the upscale smooth it
  const sw = Math.max(2, Math.round(w / 4)), sh = Math.max(2, Math.round(h / 4));
  const small = document.createElement('canvas');
  small.width = sw; small.height = sh;
  const sctx = small.getContext('2d');
  const id = sctx.createImageData(sw, sh);
  const aspect = sw / sh;

  // pass 1: fold the field through itself twice (domain warping)
  const F = new Float32Array(sw * sh), RX = new Float32Array(sw * sh);
  for (let j = 0; j < sh; j++) {
    for (let i = 0; i < sw; i++) {
      const px = (i / sw) * SCALE * aspect, py = (j / sh) * SCALE;
      const qx = fbm(px, py), qy = fbm(px + 5.2, py + 1.3);
      const rx = fbm(px + WARP * qx + 1.7, py + WARP * qy + 9.2);
      const ry = fbm(px + WARP * qx + 8.3, py + WARP * qy + 2.8);
      F[j * sw + i] = fbm(px + WARP * rx, py + WARP * ry);
      RX[j * sw + i] = rx;
    }
  }
  // auto-levels: every pattern gets the same spread of dark to light, whatever its shape
  const sorted = Float32Array.from(F).sort();
  const lo = sorted[Math.floor(sorted.length * 0.08)], hi = sorted[Math.floor(sorted.length * 0.995)];

  // pass 2: colour
  for (let j = 0; j < sh; j++) {
    for (let i = 0; i < sw; i++) {
      const f = F[j * sw + i], rx = RX[j * sw + i];
      // stretch contrast so the folds read as light catching silk
      const t = Math.pow(Math.min(1, Math.max(0, (f - lo) / (hi - lo))), 1.6);
      let [r, g, b] = silkRamp(RAMP, t);
      // a thread of copper where the second fold runs high
      const cu = Math.max(0, (rx - 0.62) / 0.2) * Math.max(0, 1 - Math.abs(t - 0.5) * 2.5) * 0.55;
      r += (COPPER[0] - r) * cu; g += (COPPER[1] - g) * cu; b += (COPPER[2] - b) * cu;
      // gentle vignette
      const vx = i / sw - 0.5, vy = j / sh - 0.5;
      const vig = 1 - 0.35 * (vx * vx + vy * vy) * 2;
      const k = (j * sw + i) * 4;
      id.data[k] = r * vig; id.data[k + 1] = g * vig; id.data[k + 2] = b * vig; id.data[k + 3] = 255;
    }
  }
  sctx.putImageData(id, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w, h);

  // fine grain over the top, also hides any upscale banding
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const rand = makeBgRandom();
  const uniform = new Float32Array(w * 2); // one row's worth, two per pixel
  for (let y = 0, i = 0; y < h; y++) {
    rand.fill(uniform);
    for (let x = 0; x < w * 2; x += 2, i += 4) {
      const n = (uniform[x] + uniform[x + 1] - 1) * GRAIN;
      d[i] += n; d[i + 1] += n; d[i + 2] += n;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// The preset's seed, or a random one; pick it once per canvas and pass it to every paint
function silkSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}
