// Ember smear behind a fine honeycomb mesh: a colour field streaked sideways like a long exposure,
// a band of terracotta deepening to burgundy on the right, peach mist above and below, seen through a
// hexagonal screen, with a little dust and a few hairline scratches.
// Usage: const seed = meshSeed(MESH_PRESETS.ember);
//        mountCanvasBg(canvas, c => paintMeshBg(c, MESH_PRESETS.ember, seed)) — needs canvas-bg.js.
// The seed is picked once, so repaints on resize keep the same pattern.

const MESH_PRESETS = {
  ember: {
    seed: null,       // a number keeps one pattern; null picks a new one each page load
    hex: 5,           // honeycomb cell width in CSS px
    mesh: 1,          // mesh strength (0 hides it)
    grain: 8,
    ramp: [
      [0.00, [248, 238, 228]], // peach mist
      [0.22, [240, 206, 180]], // pale apricot
      [0.45, [222, 150, 112]], // apricot
      [0.66, [176, 88, 66]],   // terracotta
      [0.84, [112, 40, 48]],   // burgundy
      [1.00, [58, 22, 44]],    // plum
    ],
  },
};

function meshSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}

function paintMeshBg(canvas, preset, seed) {
  const { hex: HEX, mesh: MESH, grain: GRAIN, ramp: RAMP } = preset;

  // --- seeded noise ---
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
  function fbm(x, y, octaves = 4) {
    let sum = 0, amp = 0.5, norm = 0;
    for (let o = 0; o < octaves; o++) { sum += amp * noise(x, y); norm += amp; x = x * 2.03 + 17.1; y = y * 2.03 + 9.7; amp *= 0.5; }
    return sum / norm;
  }
  const smooth = (a, b, t) => { t = Math.min(1, Math.max(0, (t - a) / (b - a))); return t * t * (3 - 2 * t); };
  function ramp(t) {
    t = Math.min(1, Math.max(0, t));
    for (let i = 1; i < RAMP.length; i++) if (t <= RAMP[i][0]) {
      const [p0, c0] = RAMP[i - 1], [p1, c1] = RAMP[i], k = (t - p0) / (p1 - p0);
      return [c0[0] + (c1[0] - c0[0]) * k, c0[1] + (c1[1] - c0[1]) * k, c0[2] + (c1[2] - c0[2]) * k];
    }
    return RAMP[RAMP.length - 1][1];
  }

  // Distance from a hexagon's centre to its edge is 0.5; returns how far inside (0 on the wall).
  const SQ3 = Math.sqrt(3);
  function hexInside(x, y) {
    const ax = ((x % 1) + 1) % 1 - 0.5, ay = ((y % SQ3) + SQ3) % SQ3 - SQ3 / 2;
    const bx = (((x - 0.5) % 1) + 1) % 1 - 0.5, by = (((y - SQ3 / 2) % SQ3) + SQ3) % SQ3 - SQ3 / 2;
    const [gx, gy] = ax * ax + ay * ay < bx * bx + by * by ? [ax, ay] : [bx, by];
    const px = Math.abs(gx), py = Math.abs(gy);
    return 0.5 - Math.max(px * 0.5 + py * SQ3 / 2, px);
  }

  // --- painting ---
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { w, h, dpr } = fitBgCanvas(canvas);

  // 1. the smeared colour field, small (it's all soft), stretched sideways
  const sw = Math.max(2, Math.round(w / 6)), sh = Math.max(2, Math.round(h / 3));
  const small = document.createElement('canvas');
  small.width = sw; small.height = sh;
  const sctx = small.getContext('2d');
  const id = sctx.createImageData(sw, sh);
  const aspect = w / h;
  for (let j = 0; j < sh; j++) {
    for (let i = 0; i < sw; i++) {
      const u = i / sw, v = j / sh, x = u * aspect;
      // the band wanders up and down a little across the canvas
      const centre = 0.5 + (fbm(x * 0.5 + 3.0, 1.7, 3) - 0.5) * 0.25;
      const band = Math.exp(-(((v - centre) / 0.27) ** 2));
      // long horizontal streaks: noise stretched hard along x
      const streak = fbm(x * 0.45 + 11.0, v * 4.8 + 2.0, 5) - 0.5;
      // a deep pool of colour toward the right
      const pool = Math.exp(-(((u - 0.86) / 0.2) ** 2) - (((v - 0.56) / 0.16) ** 2));
      const glowL = Math.exp(-(((u - 0.0) / 0.12) ** 2) - (((v - 0.72) / 0.07) ** 2)); // dark streak, left edge
      const t = band * 0.62 + streak * 0.5 + pool * 0.28 + glowL * 0.3;
      const [r, g, b] = ramp(t);
      const k = (j * sw + i) * 4;
      id.data[k] = r; id.data[k + 1] = g; id.data[k + 2] = b; id.data[k + 3] = 255;
    }
  }
  sctx.putImageData(id, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, w, h);

  // 2. dust and hairline scratches, very faint
  ctx.save();
  for (let n = 0; n < 220; n++) {
    ctx.globalAlpha = 0.08 + rand() * 0.18;
    ctx.fillStyle = '#fff';
    const r = (0.4 + rand() * 1.1) * dpr;
    ctx.beginPath(); ctx.arc(rand() * w, rand() * h, r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 0.6 * dpr;
  for (let n = 0; n < 14; n++) {
    ctx.globalAlpha = 0.05 + rand() * 0.08;
    const x0 = rand() * w, y0 = rand() * h, len = (20 + rand() * 90) * dpr, ang = (rand() - 0.5) * 1.2;
    ctx.beginPath(); ctx.moveTo(x0, y0);
    ctx.quadraticCurveTo(x0 + Math.cos(ang) * len * 0.5 + (rand() - 0.5) * 10 * dpr, y0 + Math.sin(ang) * len * 0.5,
                         x0 + Math.cos(ang) * len, y0 + Math.sin(ang) * len);
    ctx.stroke();
  }
  ctx.restore();

  // 3. honeycomb mesh + grain, per pixel
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const cell = HEX * dpr;
  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i += 4) {
      const e = hexInside(x / cell, y / cell);
      // cells catch a little light in the middle, walls sit in shadow
      const m = MESH * (0.035 * smooth(0.05, 0.4, e) - 0.075 * (1 - smooth(0.0, 0.07, e)));
      const n = (Math.random() + Math.random() - 1) * GRAIN;
      d[i]     = d[i]     * (1 + m) + n;
      d[i + 1] = d[i + 1] * (1 + m) + n;
      d[i + 2] = d[i + 2] * (1 + m) + n;
    }
  }
  ctx.putImageData(img, 0, 0);
}
