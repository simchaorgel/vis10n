// Equal-light colour field: every pixel sits at (nearly) the same perceived lightness, the point
// where white and black text have about the same contrast. All the movement is in hue and chroma.
// Colours are built in OKLCH so "same lightness" really looks the same.
// Usage: mountCanvasBg(canvas, c => paintEquiluxBg(c, EQUILUX_PRESETS.equilux, seed)) — needs canvas-bg.js.
// Pass the same seed on every repaint, so a resize redraws the same pattern rather than a new one.

const EQUILUX_PRESETS = {
  // equilux.html: dusty rose through mauve and violet to slate blue
  equilux: {
    seed: null,           // set a number to keep a pattern (see equiluxSeed)
    lightness: 0.575,     // OKLCH L; ~0.565 is the white/black balance point
    lightnessDrift: 0.025, // how far lightness may wander, kept small on purpose
    hueFrom: 20,          // dusty rose…
    hueSpan: -140,        // …through mauve and violet to slate blue
    chroma: [0.035, 0.085],
    scale: 1.4,
    warp: 2.0,
    grain: 7,
  },
};

// Seeded value noise; each paint gets its own tables, so several canvases can use different seeds
function makeEquiluxNoise(seed) {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const PERM = new Uint8Array(512), VALS = new Float32Array(256);
  for (let i = 0; i < 256; i++) { PERM[i] = i; VALS[i] = rand(); }
  for (let i = 255; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [PERM[i], PERM[j]] = [PERM[j], PERM[i]]; }
  for (let i = 0; i < 256; i++) PERM[i + 256] = PERM[i];
  const hash = (x, y) => PERM[(x & 255) + PERM[y & 255]];

  function noise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = VALS[hash(xi, yi)], b = VALS[hash(xi + 1, yi)], c = VALS[hash(xi, yi + 1)], d = VALS[hash(xi + 1, yi + 1)];
    const top = a + (b - a) * u, bot = c + (d - c) * u;
    return top + (bot - top) * v;
  }
  return function fbm(x, y, octaves = 4) {
    let sum = 0, amp = 0.5, norm = 0;
    for (let o = 0; o < octaves; o++) { sum += amp * noise(x, y); norm += amp; x = x * 2.03 + 17.1; y = y * 2.03 + 9.7; amp *= 0.45; }
    return sum / norm;
  };
}

const equiluxSmooth = (a, b, t) => { t = Math.min(1, Math.max(0, (t - a) / (b - a))); return t * t * (3 - 2 * t); };

// --- OKLCH → sRGB ---------------------------------------------------------------
function oklchToLinear(L, C, hDeg) {
  const h = hDeg * Math.PI / 180, a = C * Math.cos(h), b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}
const linearToSrgb = c => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
function oklch(L, C, h) {
  // pull chroma in until the colour fits on screen, so lightness never has to give
  let lin = oklchToLinear(L, C, h);
  for (let k = 0; k < 8 && (Math.min(...lin) < 0 || Math.max(...lin) > 1); k++) { C *= 0.85; lin = oklchToLinear(L, C, h); }
  return lin.map(c => linearToSrgb(Math.min(1, Math.max(0, c))));
}

function paintEquiluxBg(canvas, preset, seed) {
  const {
    lightness: LIGHTNESS, lightnessDrift: L_DRIFT, hueFrom: HUE_FROM, hueSpan: HUE_SPAN,
    chroma: CHROMA, scale: SCALE, warp: WARP, grain: GRAIN,
  } = preset;
  const fbm = makeEquiluxNoise(seed);
  const smooth = equiluxSmooth;
  const ctx = canvas.getContext('2d');
  const { w, h } = fitBgCanvas(canvas);
  const sw = Math.max(2, Math.round(w / 4)), sh = Math.max(2, Math.round(h / 4));
  const small = document.createElement('canvas');
  small.width = sw; small.height = sh;
  const sctx = small.getContext('2d');
  const id = sctx.createImageData(sw, sh);
  const aspect = sw / sh;

  // pass 1: warped field → hue position, plus a second field for chroma
  const F = new Float32Array(sw * sh), G = new Float32Array(sw * sh), E = new Float32Array(sw * sh);
  for (let j = 0; j < sh; j++) {
    for (let i = 0; i < sw; i++) {
      const px = (i / sw) * SCALE * aspect, py = (j / sh) * SCALE;
      const qx = fbm(px, py), qy = fbm(px + 5.2, py + 1.3);
      const rx = fbm(px + WARP * qx + 1.7, py + WARP * qy + 9.2), ry = fbm(px + WARP * qx + 8.3, py + WARP * qy + 2.8);
      const k = j * sw + i;
      F[k] = fbm(px + WARP * rx, py + WARP * ry);
      G[k] = fbm(px * 1.7 + WARP * qy + 31.0, py * 1.7 + WARP * qx + 7.0, 3);
      E[k] = rx;
    }
  }
  // auto-levels so every pattern uses the whole hue sweep
  const sorted = Float32Array.from(F).sort();
  const lo = sorted[Math.floor(sorted.length * 0.03)], hi = sorted[Math.floor(sorted.length * 0.97)];

  // pass 2: colour, all at one lightness
  for (let k = 0; k < sw * sh; k++) {
    const t = smooth(0, 1, (F[k] - lo) / (hi - lo));
    const hue = HUE_FROM + HUE_SPAN * t;
    const chroma = CHROMA[0] + (CHROMA[1] - CHROMA[0]) * smooth(0.3, 0.7, G[k]);
    const L = LIGHTNESS + L_DRIFT * (E[k] - 0.5) * 2;   // a whisper of light and shade in the folds
    const [r, g, b] = oklch(L, chroma, hue);
    id.data[k * 4] = r; id.data[k * 4 + 1] = g; id.data[k * 4 + 2] = b; id.data[k * 4 + 3] = 255;
  }
  sctx.putImageData(id, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w, h);

  // light grain, kept small so it doesn't push pixels toward white or black
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

// The preset's seed, or a random one; pick it once per canvas and pass it to every paint.
// Logged, so a pattern worth keeping can be pinned by putting its seed in the preset.
function equiluxSeed(preset) {
  const seed = preset.seed ?? Math.floor(Math.random() * 1e9);
  console.log(`equilux seed: ${seed}`);
  return seed;
}
