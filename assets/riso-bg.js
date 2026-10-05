// Two-ink risograph print: each ink is a soft density field, printed as stochastic dots,
// multiplied onto paper. Overlaps mix, the layers sit slightly off register.
// Usage: mountCanvasBg(canvas, c => paintRisoBg(c, RISO_PRESETS.dark)) — needs canvas-bg.js.
//
// preset = { paper: [r, g, b], dot: CSS px, inks: [ink, ...],
//           blend: 'multiply' (ink on light paper, default) or 'screen' (ink glowing on dark stock),
//           fibre: strength of the paper texture (default 8) }
// ink = { color: [r, g, b], patchiness: 0..1 (uneven coverage), offset: [x, y] misregistration in CSS px,
//         shapes: [{ x, y as fractions of width/height; r of the larger side; a strength;
//                    core = solid fraction of the radius; optional sx, sy stretch }] }

const RISO_PRESETS = {
  // riso-dark.html: the same print on dark stock, the inks screened on so they glow
  dark: {
    paper: [24, 22, 31],
    dot: 1.5,
    blend: 'screen',
    fibre: 6,
    inks: [
      {
        color: [24, 110, 214], // blue
        patchiness: 0.18,
        offset: [0, 0],
        shapes: [
          { x: 0.32, y: 0.62, r: 0.26, a: 1.0,  core: 0.72 },               // big disc, lower left
          { x: 0.00, y: 0.00, r: 0.55, a: 0.55, core: 0.0 },                // wash from the top-left corner
          { x: 0.88, y: 0.92, r: 0.10, a: 0.9,  core: 0.6 },                // small disc, bottom right
        ],
      },
      {
        color: [255, 72, 176], // fluorescent pink
        patchiness: 0.2,
        offset: [5, -3],
        shapes: [
          { x: 0.55, y: 0.40, r: 0.21, a: 1.0,  core: 0.7 },                // sun, overlapping the disc
          { x: 0.50, y: 1.12, r: 0.62, a: 0.95, core: 0.25, sx: 2.0, sy: 0.4 }, // glow rising from the bottom
          { x: 0.95, y: 0.08, r: 0.30, a: 0.45, core: 0.0 },                // faint wash, top right
        ],
      },
    ],
  },
};

// Smooth value noise in 0..1 at w x h: a tiny random image upscaled with smoothing.
function risoSmoothNoise(w, h, cells) {
  const tiny = document.createElement('canvas');
  tiny.width = cells; tiny.height = Math.max(2, Math.round(cells * h / w));
  const t = tiny.getContext('2d');
  const id = t.createImageData(tiny.width, tiny.height);
  for (let i = 0; i < id.data.length; i += 4) {
    const v = Math.random() * 255;
    id.data[i] = id.data[i + 1] = id.data[i + 2] = v; id.data[i + 3] = 255;
  }
  t.putImageData(id, 0, 0);
  return tiny;
}

// One ink's coverage at full size.
function risoInkDensity(ink, w, h, dpr) {
  const big = Math.max(w, h);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const x = c.getContext('2d', { willReadFrequently: true });
  x.fillStyle = '#000';
  x.fillRect(0, 0, w, h);
  x.globalCompositeOperation = 'lighter'; // shapes of the same ink add up

  const [ox, oy] = (ink.offset || [0, 0]).map(v => v * dpr);
  for (const s of ink.shapes) {
    const r = s.r * big;
    const g = x.createRadialGradient(0, 0, 0, 0, 0, r);
    for (let i = 0; i <= 10; i++) {
      const t = i / 10;
      const u = Math.max(0, (t - s.core) / (1 - s.core));
      const v = Math.round(255 * s.a * Math.exp(-5 * u * u) * (1 - u));
      g.addColorStop(t, `rgb(${v},${v},${v})`);
    }
    x.setTransform(1, 0, 0, 1, s.x * w + ox, s.y * h + oy);
    x.scale(s.sx || 1, s.sy || 1);
    x.fillStyle = g;
    x.fillRect(-r, -r, 2 * r, 2 * r);
  }
  x.setTransform(1, 0, 0, 1, 0, 0);

  // patchy coverage: knock the density down by smooth noise at two scales
  x.globalCompositeOperation = 'multiply';
  x.imageSmoothingQuality = 'high';
  for (const [cells, amt] of [[9, ink.patchiness], [40, ink.patchiness * 0.5]]) {
    x.globalAlpha = amt;
    x.drawImage(risoSmoothNoise(w, h, cells), 0, 0, w, h);
  }
  x.globalAlpha = 1;

  // Raw pixels: the red channel of pixel i (at i * 4) is the coverage, 0..255
  return x.getImageData(0, 0, w, h).data;
}

function paintRisoBg(canvas, preset) {
  const ctx = canvas.getContext('2d');
  const { w, h, dpr } = fitBgCanvas(canvas);
  const rand = makeBgRandom();
  const uniform = new Float32Array(w * 2); // a row's worth of random numbers at a time
  const cell = Math.max(1, Math.round((preset.dot || 1.5) * dpr));
  const paper = preset.paper;
  const fibre = preset.fibre ?? 8;
  const screen = preset.blend === 'screen';

  const img = ctx.createImageData(w, h);
  const out = img.data;

  // paper, with a faint fibre texture
  for (let y = 0, i = 0; y < h; y++) {
    rand.fill(uniform);
    for (let x = 0; x < w; x++, i += 4) {
      const n = (uniform[x] - 0.5) * fibre;
      out[i] = paper[0] + n; out[i + 1] = paper[1] + n; out[i + 2] = paper[2] + n; out[i + 3] = 255;
    }
  }

  for (const ink of preset.inks) {
    const dens = risoInkDensity(ink, w, h, dpr);
    const [ir, ig, ib] = ink.color.map(v => v / 255);
    // one random decision per dot cell: printed or not, with probability = coverage
    for (let cy = 0; cy < h; cy += cell) {
      rand.fill(uniform); // per dot cell: one number to decide if it prints, one for its strength
      for (let cx = 0, u = 0; cx < w; cx += cell, u += 2) {
        const p = (1.18 / 255) * dens[(Math.min(h - 1, cy + (cell >> 1)) * w + Math.min(w - 1, cx + (cell >> 1))) * 4];
        if (uniform[u] >= p) continue;
        const strength = 0.82 + uniform[u + 1] * 0.18; // ink doesn't lay down perfectly evenly
        if (screen) {
          // screen: the ink lightens the dark paper, and overlapping inks glow brighter
          const sr = strength * ir, sg = strength * ig, sb = strength * ib;
          for (let y = cy; y < Math.min(h, cy + cell); y++) {
            for (let x = cx, i = (y * w + cx) * 4; x < Math.min(w, cx + cell); x++, i += 4) {
              out[i]     = 255 - (255 - out[i])     * (1 - sr);
              out[i + 1] = 255 - (255 - out[i + 1]) * (1 - sg);
              out[i + 2] = 255 - (255 - out[i + 2]) * (1 - sb);
            }
          }
        } else {
          const mr = 1 - strength * (1 - ir), mg = 1 - strength * (1 - ig), mb = 1 - strength * (1 - ib);
          for (let y = cy; y < Math.min(h, cy + cell); y++) {
            for (let x = cx, i = (y * w + cx) * 4; x < Math.min(w, cx + cell); x++, i += 4) {
              out[i] *= mr; out[i + 1] *= mg; out[i + 2] *= mb; // multiply, like ink on paper
            }
          }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
