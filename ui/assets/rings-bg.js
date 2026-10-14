// Rings: an out-of-focus field of colour blocks, printed through a screen of concentric lines.
// Each colour channel is screened on its own, so lines thicken in the dark, vanish in the light,
// and split into coloured fringes in between.
// Usage: const seed = ringsSeed(RINGS_PRESETS.forest);
//        mountCanvasBg(canvas, c => paintRingsBg(c, RINGS_PRESETS.forest, seed)) — needs canvas-bg.js.
// The preset's seed fixes the composition; null would pick a new one each page load.

const RINGS_PRESETS = {
  forest: {
    seed: 88,                      // always the same composition; null for a new one each page load
    ground: [18, 50, 24],          // the field behind the blocks
    blocks: [                      // colour, weight
      [[192, 86, 32], 2],          // burnt orange
      [[206, 178, 48], 2],         // mustard
      [[12, 44, 18], 5],           // bottle green
      [[36, 78, 40], 2],           // moss
      [[226, 218, 190], 1],        // pale cream
    ],
    stretch: 1.0,   // tone contrast before screening; above 1 blows out lights and closes darks
    keep: 0.32,     // how much of the underlying colour shows through the lines (0 = hard black/white lines)
    level: 0.9,     // overall brightness
    ring: 0.0105,   // ring spacing, as a fraction of the shorter side
    soft: 0.18,     // softness of each line's edge
    grit: 0.16,     // noise added before screening, roughens the line edges
    blur: 0.032,    // how out of focus the blocks are, as a fraction of the shorter side
  },
  // forest, darker, with a more muted cream, so black shapes sit well on top
  forestDark: {
    seed: 88,
    ground: [18, 50, 24],
    blocks: [
      [[192, 86, 32], 2],          // burnt orange
      [[206, 178, 48], 2],         // mustard
      [[12, 44, 18], 5],           // bottle green
      [[36, 78, 40], 2],           // moss
      [[200, 192, 164], 1],        // muted cream
    ],
    stretch: 1.0,
    keep: 0.32,
    level: 1,
    ring: 0.0105,
    soft: 0.18,
    grit: 0.16,
    blur: 0.032,
  },
};

function ringsSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}

function paintRingsBg(canvas, preset, seed) {
  const { ground: GROUND, blocks: BLOCKS, stretch: STRETCH, keep: KEEP, level: LEVEL,
          ring: RING, soft: SOFT, grit: GRIT, blur: BLUR } = preset;

  const makeRandom = s0 => {
    let s = s0 >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { w, h, dpr } = fitBgCanvas(canvas);
  const u = Math.min(w, h);
  const r = makeRandom(seed);
  const total = BLOCKS.reduce((s, b) => s + b[1], 0);
  const pickColour = () => { let x = r() * total; for (const [c, wt] of BLOCKS) { if ((x -= wt) < 0) return c; } return BLOCKS[0][0]; };

  // 1. the field: overlapping blocks, mostly wide bars, thrown out of focus.
  //    Drawn at quarter size, since it's all blur anyway.
  const q = 4, fw = Math.ceil(w / q), fh = Math.ceil(h / q);
  const field = document.createElement('canvas');
  field.width = fw; field.height = fh;
  const f = field.getContext('2d', { willReadFrequently: true });
  f.fillStyle = `rgb(${GROUND})`;
  f.fillRect(0, 0, fw, fh);
  // everything is sized off the shorter side, so a wide canvas gets more blocks, not stretched ones
  const fu = Math.min(fw, fh);
  f.filter = `blur(${(BLUR * fu).toFixed(1)}px)`;
  // a loose patchwork underneath so the whole field is covered, no two neighbours alike
  const cols = Math.max(2, Math.round(fw / (fu * 0.42))), rows = Math.max(2, Math.round(fh / (fu * 0.32)));
  const grid = [];
  for (let gy = 0; gy < rows; gy++) for (let gx = 0; gx < cols; gx++) {
    let c;
    do { c = pickColour(); } while (c === grid[(gy - 1) * cols + gx] || (gx > 0 && c === grid[gy * cols + gx - 1]));
    grid[gy * cols + gx] = c;
    const cw = fw / cols, chh = fh / rows;
    f.fillStyle = `rgb(${c})`;
    f.fillRect((gx + (r() - 0.5) * 0.3) * cw, (gy + (r() - 0.5) * 0.3) * chh, cw * (1 + r() * 0.3), chh * (1 + r() * 0.3));
  }
  // then bars and blocks thrown across it
  const n = Math.round((fw * fh) / (fu * fu) * (4 + r() * 3));
  for (let i = 0; i < n; i++) {
    const wide = r() < 0.5;
    const bw = fu * (wide ? 0.4 + r() * 0.35 : 0.25 + r() * 0.25);
    const bh = fu * (wide ? 0.1 + r() * 0.08 : 0.2 + r() * 0.2);
    const bx = r() * (fw + bw) - bw * 0.7, by = r() * (fh + bh) - bh * 0.7;
    f.fillStyle = `rgb(${pickColour()})`;
    f.fillRect(bx, by, bw, bh);
  }
  f.filter = 'none';
  const F = f.getImageData(0, 0, fw, fh).data;

  // 2. the screen: rings around a point near the middle, slightly squashed so they read as a lens
  const cx = w * (0.4 + r() * 0.2), cy = h * (0.42 + r() * 0.16);
  const squash = 0.92 + r() * 0.16;
  const period = Math.max(3.2 * dpr, u * RING);
  const k = Math.PI * 2 / period;
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const sm = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

  for (let y = 0, o = 0; y < h; y++) {
    // bilinear lookup into the small field
    const fy = Math.min(fh - 1.001, y / q), j0 = fy | 0, ty = fy - j0;
    const ry = (y - cy) * squash;
    for (let x = 0; x < w; x++, o += 4) {
      const fx = Math.min(fw - 1.001, x / q), i0 = fx | 0, tx = fx - i0;
      const a = (j0 * fw + i0) * 4, b = a + 4, c = a + fw * 4, e = c + 4;
      const rx = x - cx;
      const dist = Math.sqrt(rx * rx + ry * ry);
      // the screen value runs 0..1 across each ring; lines appear where the ink is darker than it
      const wave = 0.5 + 0.5 * Math.cos(dist * k);
      const grit = (Math.random() - 0.5) * GRIT;
      for (let ch = 0; ch < 3; ch++) {
        const v = ((F[a + ch] * (1 - tx) + F[b + ch] * tx) * (1 - ty) + (F[c + ch] * (1 - tx) + F[e + ch] * tx) * ty) / 255;
        const tone = Math.min(1.1, Math.max(-0.1, (v - 0.5) * STRETCH + 0.5)) + grit;
        const s = sm(wave - SOFT, wave + SOFT, tone);
        // keep some of the underlying colour so the lines aren't pure black and white
        d[o + ch] = 255 * LEVEL * (s * (1 - KEEP) + v * KEEP);
      }
      d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
