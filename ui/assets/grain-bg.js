// Soft blurred blobs of colour, finished with fine film grain.
// Usage: mountCanvasBg(canvas, c => paintGrainBg(c, PRESETS.blue)) — needs canvas-bg.js.

const PRESETS = {
  // x, y as fractions of width/height; r as fraction of the larger side.
  // Optional darkGrain (0–1) tones down the dark half of the grain.
  // Optional ink is the text colour to use on top (defaults to light text).
  // Optional core (0–1) widens the solid centre, sx/sy stretch a blob into an ellipse, rot rotates it (radians).
  // Optional pixel (CSS px) makes the grain chunky; chroma + chromaDir add colour speckle in one hue
  // direction; grainMask fades grain out where the colour is close to the base.
  // Optional rects are soft-edged blocks drawn after the blobs (see cyanotype).
  blue: {
    base: [72, 110, 136],
    grain: 26,
    blobs: [
      { x: 0.15, y: 0.28, r: 0.55, c: [22, 50, 76],    a: 0.85 }, // dark, upper left
      { x: 0.95, y: 0.02, r: 0.35, c: [30, 60, 90],    a: 0.6  }, // dark, top right corner
      { x: 0.92, y: 0.46, r: 0.45, c: [176, 200, 212], a: 0.75 }, // light, right
      { x: 0.40, y: 0.70, r: 0.40, c: [166, 192, 206], a: 0.7  }, // light, lower middle
      { x: 0.05, y: 0.62, r: 0.35, c: [40, 72, 104],   a: 0.55 }, // shade, left
      { x: 0.75, y: 1.02, r: 0.40, c: [36, 64, 92],    a: 0.6  }, // dark, bottom
    ],
  },
  // Coral: warm coral ground, mustard sun top right, plum low on the left (MNIST test window)
  coral: {
    base: [226, 88, 62],
    grain: 40,
    blobs: [
      { x: 0.20, y: 0.04, r: 0.40, c: [150, 44, 52],   a: 0.6,  core: 0.2, sx: 1.4, sy: 0.7, rot: 0.3 }, // deep red, top left
      { x: 0.80, y: 0.28, r: 0.45, c: [250, 140, 70],  a: 0.5 },                                       // warm glow around the sun
      { x: 0.88, y: 0.16, r: 0.28, c: [242, 177, 52],  a: 0.92, core: 0.4 },                           // mustard sun, top right
      { x: 1.00, y: 0.88, r: 0.26, c: [245, 160, 90],  a: 0.6,  core: 0.2 },                           // peach, bottom right
      { x: 0.55, y: 0.98, r: 0.30, c: [120, 40, 60],   a: 0.55, core: 0.2, sx: 1.6, sy: 0.6 },         // plum band, bottom
      { x: 0.04, y: 1.04, r: 0.42, c: [74, 35, 64],    a: 0.9,  core: 0.35 },                          // plum, bottom left
    ],
  },
  green: {
    base: [22, 42, 58],
    grain: 42,
    blobs: [
      { x: 0.60, y: 0.22, r: 0.50, c: [20, 32, 52],    a: 0.85, core: 0.3 },                     // navy, upper middle
      { x: 0.10, y: 0.05, r: 0.22, c: [58, 160, 124],  a: 0.8, core: 0.35, sx: 1.4, sy: 0.6, rot: -0.6 }, // teal arc, top left
      { x: 0.06, y: 0.34, r: 0.30, c: [132, 208, 154], a: 0.92, core: 0.4, sx: 0.75, sy: 1.25, rot: 0.55 }, // mint streak, left
      { x: 0.45, y: 0.44, r: 0.20, c: [38, 140, 112],  a: 0.8, core: 0.3 },                     // teal, centre
      { x: 0.62, y: 0.58, r: 0.20, c: [150, 215, 160], a: 0.92, core: 0.4, sy: 1.25 },           // bright, centre
      { x: 1.00, y: 0.46, r: 0.20, c: [38, 132, 110],  a: 0.75,  core: 0.3 },                     // teal, right edge
      { x: 0.85, y: 0.80, r: 0.34, c: [20, 34, 52],    a: 0.8,  core: 0.3 },                     // navy, lower right
      { x: 0.12, y: 0.68, r: 0.30, c: [22, 40, 58],    a: 0.75, core: 0.3, sx: 1.5, sy: 0.55, rot: 0.3 }, // dark band, left
      { x: 0.18, y: 0.95, r: 0.34, c: [142, 212, 154], a: 0.92, core: 0.4, sx: 1.2, sy: 0.85 }, // bright, bottom left
      { x: 0.58, y: 1.00, r: 0.24, c: [122, 200, 146], a: 0.85,  core: 0.35 },                    // mint, bottom middle
      { x: 0.36, y: 0.90, r: 0.12, c: [24, 50, 64],    a: 0.5, core: 0.2, sx: 0.55, sy: 1.6 },  // shadow between
    ],
  },
  // Bloom: pale ice white with a red and orange bloom low on the right. Light, so it needs dark text.
  bloom: {
    base: [230, 247, 249],
    grain: 56,
    darkGrain: 0.35,
    ink: [34, 40, 52],
    blobs: [
      { x: 0.08, y: 0.72, r: 0.34, c: [196, 230, 236], a: 0.6 },                                   // cool cyan tint, left
      { x: 0.76, y: 0.80, r: 0.55, c: [232, 170, 192], a: 0.6,  sx: 1.2 },                         // pink haze around the bloom
      { x: 0.86, y: 0.70, r: 0.40, c: [236, 34, 48],   a: 0.85, core: 0.1, sx: 1.0, sy: 1.1 },     // red, right
      { x: 0.64, y: 0.94, r: 0.34, c: [232, 76, 52],   a: 0.8,  core: 0.1, sx: 1.3, sy: 0.9 },      // orange-red, bottom
      { x: 1.00, y: 1.00, r: 0.18, c: [220, 200, 210], a: 0.5 },                                   // pale corner, bottom right
    ],
  },
  // Cyanotype: soft-edged blue blocks on white paper, after a blurred print. Blocks sit to the sides
  // so the centre stays pale for the text.
  cyanotype: {
    base: [243, 247, 250],
    grain: 16,
    darkGrain: 0.6,
    ink: [22, 50, 92],
    blobs: [],
    // x, y, w, h as fractions of width/height; blur as a fraction of the larger side
    rects: [
      { x: 0.00, y: 0.00, w: 0.40, h: 0.26, c: [96, 162, 216],  a: 0.75, blur: 0.045 },  // pale band, top left
      { x: 0.03, y: 0.34, w: 0.12, h: 0.72, c: [112, 174, 224], a: 0.6,  blur: 0.045 },  // pale strip, left
      { x: 0.70, y: 0.03, w: 0.26, h: 0.42, c: [30, 118, 198],  a: 0.95, blur: 0.055 }, // deep block, top right
      { x: 0.71, y: 0.51, w: 0.27, h: 0.38, c: [28, 114, 194],  a: 0.95, blur: 0.055 }, // deep block, lower right
      { x: 0.42, y: 0.90, w: 0.36, h: 0.16, c: [124, 182, 228], a: 0.45, blur: 0.045 },  // faint wash, bottom
    ],
  },
  // Dissolve: a deep blue mass breaking up into cyan and white through chunky colour grain.
  dissolve: {
    base: [250, 253, 255],
    grain: 95,
    pixel: 2,                 // grain cells in CSS px, for a coarse digital look
    chroma: 0.9,
    chromaDir: [0.9, 0.8, 0.15], // speckles shift between deep blue and cyan, never into pink
    grainMask: 90,            // no grain on the clean white
    darkGrain: 0.8,
    ink: [14, 38, 86],
    blobs: [
      { x: 0.03, y: 0.50, r: 0.60, c: [92, 214, 242],  a: 0.8,  sx: 0.85, sy: 1.0 },               // cyan fringe
      { x: -0.04, y: 0.48, r: 0.50, c: [26, 96, 178],   a: 1.0,  core: 0.45, sx: 0.8, sy: 1.0 },     // deep blue mass
      { x: 0.55, y: 1.08, r: 0.48, c: [70, 196, 236],  a: 0.8,  core: 0.2, sx: 1.9, sy: 0.55 },     // cyan band, bottom
      { x: 0.92, y: 1.10, r: 0.30, c: [26, 84, 156],   a: 0.85, core: 0.3, sx: 1.6, sy: 0.6 },      // blue, bottom right
    ],
  },
};

function paintGrainBg(canvas, preset) {
  const ctx = canvas.getContext('2d');
  const { w, h, dpr } = fitBgCanvas(canvas);
  const big = Math.max(w, h);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = `rgb(${preset.base})`;
  ctx.fillRect(0, 0, w, h);

  for (const b of preset.blobs) {
    const r = b.r * big;
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    // several stops approximate a gaussian falloff so edges stay very soft
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      const core = b.core || 0; // fraction of the radius kept at full strength
      const u = Math.max(0, (t - core) / (1 - core));
      const alpha = b.a * Math.exp(-4.5 * u * u) * (1 - u);
      g.addColorStop(t, `rgba(${b.c},${alpha.toFixed(4)})`);
    }
    ctx.setTransform(1, 0, 0, 1, b.x * w, b.y * h);
    ctx.rotate(b.rot || 0);
    ctx.scale(b.sx || 1, b.sy || 1);
    ctx.fillStyle = g;
    ctx.fillRect(-r, -r, 2 * r, 2 * r);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // Soft-edged blocks: solid rectangles drawn through a blur filter.
  for (const q of preset.rects || []) {
    ctx.filter = `blur(${Math.round(q.blur * big)}px)`;
    ctx.globalAlpha = q.a;
    ctx.fillStyle = `rgb(${q.c})`;
    ctx.fillRect(q.x * w, q.y * h, q.w * w, q.h * h);
  }
  ctx.filter = 'none';
  ctx.globalAlpha = 1;

  // Grain: noise per cell (one device pixel unless preset.pixel is set), also hides gradient banding.
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const s = preset.grain;
  const dark = preset.darkGrain ?? 1;   // scales the darkening speckles; < 1 leaves mostly light sparkle
  const chroma = preset.chroma ?? 0;    // strength of colour speckle along chromaDir
  const dir = preset.chromaDir || [0, 0, 0];
  const mask = preset.grainMask || 0;   // > 0: grain fades out where the colour is close to the base
  const cell = Math.max(1, Math.round((preset.pixel || 0) * dpr)) || 1;
  const cols = Math.ceil(w / cell);
  // Noise is made one row of cells at a time, so there's never a buffer the size of the image
  const rand = makeBgRandom();
  const uniform = new Float32Array(cols * 4); // two per triangular sample, two samples with chroma
  let u = 0;
  const tri = () => {
    const n = (uniform[u++] + uniform[u++] - 1) * s; // triangular distribution
    return n < 0 ? n * dark : n;
  };
  const noise = new Float32Array(cols * 3);
  const [br, bg, bb] = preset.base;
  for (let y = 0, i = 0; y < h; y++) {
    if (y % cell === 0) {
      rand.fill(uniform);
      u = 0;
      for (let k = 0; k < noise.length; k += 3) {
        const n = tri();
        const c = chroma ? tri() * chroma : 0;
        noise[k]     = n * 0.9  + c * dir[0];
        noise[k + 1] = n        + c * dir[1];
        noise[k + 2] = n * 1.05 + c * dir[2];
      }
    }
    for (let x = 0; x < w; x++, i += 4) {
      const k = ((x / cell) | 0) * 3;
      let f = 1;
      if (mask) {
        const dist = Math.abs(d[i] - br) + Math.abs(d[i + 1] - bg) + Math.abs(d[i + 2] - bb);
        f = Math.min(1, dist / mask);
      }
      d[i]     += noise[k] * f;
      d[i + 1] += noise[k + 1] * f;
      d[i + 2] += noise[k + 2] * f;
    }
  }
  ctx.putImageData(img, 0, 0);
}
