// Glow: an out-of-focus field of warm blocks seen through frosted glass. Every pixel samples the
// field from a small random offset, so edges dissolve into a sandy stipple instead of a smooth blur.
// Usage: const seed = glowSeed(GLOW_PRESETS.ember);
//        mountCanvasBg(canvas, c => paintGlowBg(c, GLOW_PRESETS.ember, seed)) — needs canvas-bg.js.
// The preset's seed fixes the composition; null would pick a new one each page load.

const GLOW_PRESETS = {
  ember: {
    seed: 88,                      // always the same composition; null for a new one each page load
    ground: [58, 16, 16],          // the field behind the blocks
    blocks: [                      // colour, weight
      [[44, 10, 12], 4],           // deep maroon
      [[236, 84, 92], 3],          // coral red
      [[248, 232, 160], 3],        // warm cream
      [[196, 50, 60], 1],          // brick
    ],
    blur: 0.035,     // how out of focus the blocks are, as a fraction of the shorter side
    scatter: 0.03,   // how far each pixel looks for its colour, as a fraction of the shorter side
    grain: 22,
    glow: 0.18,      // light bleeding out of the bright blocks
  },
};

function glowSeed(preset) {
  return preset.seed ?? Math.floor(Math.random() * 1e9);
}

function paintGlowBg(canvas, preset, seed) {
  const { ground: GROUND, blocks: BLOCKS, blur: BLUR, scatter: SCATTER, grain: GRAIN, glow: GLOW } = preset;

  const makeRandom = s0 => {
    let s = s0 >>> 0;
    return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };

  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { w, h } = fitBgCanvas(canvas);
  const u = Math.min(w, h);
  const r = makeRandom(seed);
  const total = BLOCKS.reduce((s, b) => s + b[1], 0);
  const pickColour = () => { let x = r() * total; for (const [c, wt] of BLOCKS) { if ((x -= wt) < 0) return c; } return BLOCKS[0][0]; };

  // 1. the field: a patchwork of blocks with bars thrown across, drawn at quarter size and blurred.
  //    Everything is sized off the shorter side, so a wide canvas gets more blocks, not stretched ones.
  const q = 4, fw = Math.ceil(w / q), fh = Math.ceil(h / q), fu = Math.min(fw, fh);
  const field = document.createElement('canvas');
  field.width = fw; field.height = fh;
  const f = field.getContext('2d', { willReadFrequently: true });
  f.fillStyle = `rgb(${GROUND})`;
  f.fillRect(0, 0, fw, fh);
  f.filter = `blur(${(BLUR * fu).toFixed(1)}px)`;
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
  const n = Math.round((fw * fh) / (fu * fu) * (4 + r() * 3));
  for (let i = 0; i < n; i++) {
    const wide = r() < 0.5;
    const bw = fu * (wide ? 0.4 + r() * 0.35 : 0.2 + r() * 0.2);
    const bh = fu * (wide ? 0.1 + r() * 0.08 : 0.16 + r() * 0.16);
    const bx = r() * (fw + bw) - bw * 0.7, by = r() * (fh + bh) - bh * 0.7;
    f.fillStyle = `rgb(${pickColour()})`;
    f.fillRect(bx, by, bw, bh);
  }
  // the bright blocks glow: a wide, faint copy of the field laid back over it
  f.filter = 'none';
  const copy = document.createElement('canvas');
  copy.width = fw; copy.height = fh;
  copy.getContext('2d').drawImage(field, 0, 0);
  f.filter = `blur(${(BLUR * fu * 3).toFixed(1)}px)`;
  f.globalCompositeOperation = 'screen';
  f.globalAlpha = GLOW;
  f.drawImage(copy, 0, 0);
  f.globalCompositeOperation = 'source-over';
  f.globalAlpha = 1;
  f.filter = 'none';
  const F = f.getImageData(0, 0, fw, fh).data;

  // 2. frosted glass: each pixel takes its colour from a random nearby point (gaussian-ish spread)
  const spread = SCATTER * u / q;   // in field pixels
  const img = ctx.createImageData(w, h);
  const d = img.data;
  for (let y = 0, o = 0; y < h; y++) {
    for (let x = 0; x < w; x++, o += 4) {
      const gx = (Math.random() + Math.random() + Math.random() - 1.5) * spread;
      const gy = (Math.random() + Math.random() + Math.random() - 1.5) * spread;
      const sx = Math.min(fw - 1, Math.max(0, (x / q + gx) | 0));
      const sy = Math.min(fh - 1, Math.max(0, (y / q + gy) | 0));
      const k = (sy * fw + sx) * 4;
      const g = (Math.random() + Math.random() - 1) * GRAIN;
      d[o] = F[k] + g; d[o + 1] = F[k + 1] + g * 0.9; d[o + 2] = F[k + 2] + g * 0.8; d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
