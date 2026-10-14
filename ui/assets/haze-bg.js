// Haze: ridgelines receding into a dusk sky, each one lost a little more in the mist.
// Usage: const seed = hazeSeed(HAZE_PRESETS.dusk);
//        mountCanvasBg(canvas, c => paintHazeBg(c, HAZE_PRESETS.dusk, seed)) — needs canvas-bg.js.
// The seed is picked once, so repaints on resize keep the same range.

const HAZE_PRESETS = {
  dusk: {
    seed: null,                   // a number keeps one range; null picks a new one each page load
    ridges: 7,
    grain: 30,
    sky: [                        // position 0 (top) .. 1 (horizon) → colour
      [0.00, [34, 38, 74]],       // indigo
      [0.45, [112, 82, 116]],     // dusk violet
      [0.75, [204, 128, 122]],    // rose
      [1.00, [244, 192, 146]],    // apricot
    ],
    haze: [240, 196, 164],        // the mist each ridge fades into
    far: [186, 132, 140],         // colour of the furthest ridge
    near: [22, 20, 42],           // colour of the nearest ridge
    sun: [254, 236, 206],
  },
};

function hazeSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}

function paintHazeBg(canvas, preset, seed) {
  const { ridges: RIDGES, grain: GRAIN, sky: SKY, haze: HAZE, far: FAR, near: NEAR, sun: SUN } = preset;

  // --- seeded 1D value noise ---
  const makeRandom = s0 => {
    let s = s0 >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  const rand = makeRandom(seed);
  const VALS = new Float32Array(1024);
  for (let i = 0; i < VALS.length; i++) VALS[i] = rand();
  function noise(x) {
    const xi = Math.floor(x), xf = x - xi, u = xf * xf * (3 - 2 * xf);
    const a = VALS[xi & 1023], b = VALS[(xi + 1) & 1023];
    return a + (b - a) * u;
  }
  function fbm(x, octaves) {   // layered noise, 0..1
    let sum = 0, amp = 0.5, norm = 0;
    for (let o = 0; o < octaves; o++) { sum += amp * noise(x); norm += amp; x = x * 2.1 + 13.7; amp *= 0.5; }
    return sum / norm;
  }

  const mix = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
  const rgba = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;

  // --- painting ---
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { w, h } = fitBgCanvas(canvas);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const big = Math.max(w, h);
  const r = makeRandom(seed ^ 0x5bd1e995);   // layout choices, separate from the noise tables
  const horizon = h * 0.62;

  // sky
  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  for (const [p, c] of SKY) sky.addColorStop(p, rgba(c));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // sun, low and off-centre, with a wide bloom
  const sx = w * (0.25 + r() * 0.5), sy = horizon - h * (0.1 + r() * 0.14), sr = big * 0.032;
  const bloom = ctx.createRadialGradient(sx, sy, 0, sx, sy, big * 0.55);
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    bloom.addColorStop(t, rgba(HAZE, (0.55 * Math.exp(-5 * t * t) * (1 - t)).toFixed(4)));
  }
  ctx.fillStyle = bloom;
  ctx.fillRect(0, 0, w, h);
  const disc = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr * 1.25);
  disc.addColorStop(0, rgba(SUN));
  disc.addColorStop(0.6, rgba(SUN, 0.92));
  disc.addColorStop(1, rgba(SUN, 0));
  ctx.fillStyle = disc;
  ctx.fillRect(sx - sr * 2, sy - sr * 2, sr * 4, sr * 4);

  // thin streaks of cloud, drawn through a blur
  ctx.filter = `blur(${Math.round(big * 0.008)}px)`;
  for (let i = 0; i < 6; i++) {
    const cy = horizon * (0.25 + r() * 0.6), cw = w * (0.2 + r() * 0.45), ch = h * (0.006 + r() * 0.01);
    const cx = r() * (w - cw * 0.5) - cw * 0.25;
    const lit = mix([150, 96, 128], HAZE, cy / horizon);
    ctx.fillStyle = rgba(lit, 0.25 + r() * 0.3);
    ctx.fillRect(cx, cy, cw, ch);
  }
  ctx.filter = 'none';

  // ridges, far to near; each fades to mist at its foot so the next one stands clear of it
  for (let i = 0; i < RIDGES; i++) {
    const t = i / (RIDGES - 1);
    const base = horizon + (h - horizon) * 0.9 * Math.pow(t, 1.5) - h * 0.02;
    const amp = h * (0.04 + 0.1 * t);
    const freq = (1.4 + r() * 1.2) / w;     // folds across the canvas
    const off = r() * 900;
    const oct = 3 + Math.round(t * 3);       // nearer ridges are more jagged
    const col = mix(FAR, NEAR, Math.pow(t, 0.9));
    const ys = [];
    let top = h;
    for (let x = 0; x <= w + 4; x += 4) {
      const y = base - amp * (fbm(x * freq + off, oct) - 0.35) * 2;
      ys.push(y); if (y < top) top = y;
    }
    const fill = ctx.createLinearGradient(0, top, 0, base + amp * 1.2);
    fill.addColorStop(0, rgba(col));
    fill.addColorStop(0.55, rgba(mix(col, HAZE, 0.12 * (1 - t))));
    fill.addColorStop(1, rgba(mix(col, HAZE, 0.55 - 0.35 * t)));
    ctx.fillStyle = fill;
    // far ridges are softer, as if seen through the haze
    ctx.filter = `blur(${(big * 0.0035 * (1 - t)).toFixed(1)}px)`;
    ctx.beginPath();
    // run past the edges so the blur doesn't soften the sides and bottom
    const pad = big * 0.05;
    ctx.moveTo(-pad, h + pad);
    ctx.lineTo(-pad, ys[0]);
    ys.forEach((y, k) => ctx.lineTo(k * 4, y));
    ctx.lineTo(w + pad, ys[ys.length - 1]);
    ctx.lineTo(w + pad, h + pad);
    ctx.closePath();
    ctx.fill();
    ctx.filter = 'none';

    // a soft band of mist lying in the valley in front of this ridge
    if (i < RIDGES - 1) {
      const my = base + amp * 0.4;
      const mist = ctx.createLinearGradient(0, my - amp * 0.6, 0, my + amp * 0.6);
      mist.addColorStop(0, rgba(HAZE, 0));
      mist.addColorStop(0.5, rgba(HAZE, (0.22 * (1 - t)).toFixed(3)));
      mist.addColorStop(1, rgba(HAZE, 0));
      ctx.fillStyle = mist;
      ctx.fillRect(0, my - amp * 0.6, w, amp * 1.2);
    }
  }

  // the sun's glow spills back over the ridges in front of it
  ctx.globalCompositeOperation = 'screen';
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = bloom;
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;

  // fine grain over the top, also hides gradient banding
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    let n = (Math.random() + Math.random() - 1) * GRAIN;
    if (n < 0) n *= 0.8;
    d[i] += n * 0.95; d[i + 1] += n; d[i + 2] += n * 1.08;
  }
  ctx.putImageData(img, 0, 0);
}
