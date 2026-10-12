// Orbital sunrise: a big dark planet, the sun just cresting its rim, the atmosphere burning in bands
// from red-orange at the surface through gold and white into deep blue, a bloom, a starburst and a
// lens streak. Space stays pure black. Smooth, with only a sub-step dither against banding.
// Usage: mountCanvasBg(canvas, c => paintSunriseBg(c, SUNRISE_PRESETS.sunrise)) — needs canvas-bg.js.

const SUNRISE_PRESETS = {
  // Lengths are in canvas heights.
  sunrise: {
    top: 0.66,          // top of the curve, as a fraction of the height
    radius: 1.5,        // planet radius (smaller = more curve)
    tilt: -0.25,        // planet centre offset sideways
    sunAt: 0.95,        // where the sun crests, along the rim (-1 left … 1 right)
    atmos: 0.012,       // thickness of the coloured atmosphere band
    haze: 0.07,         // how far the blue haze reaches into space
    bloom: 0.16,        // size of the sun's soft bloom
    rays: 12,           // starburst spikes
    cap: 246,           // brightest a channel may get
    // atmosphere layers from the surface outward: [height in atmos units, colour]
    layers: [
      [0.0, [255, 90, 40]],     // red-orange, at the surface
      [0.6, [255, 170, 70]],    // gold
      [1.2, [255, 238, 200]],   // white-hot
      [2.2, [120, 190, 255]],   // sky blue
      [4.0, [40, 90, 210]],     // deep blue
      [7.0, [10, 24, 80]],      // indigo, fading into space
    ],
    night: [4, 8, 18],
  },
};

function paintSunriseBg(canvas, preset) {
  const { top: TOP, radius: RADIUS, tilt: TILT, sunAt: SUN_AT, atmos: ATMOS, haze: HAZE,
          bloom: BLOOM, rays: RAYS, cap: CAP, layers: LAYERS, night: NIGHT } = preset;

  // Colour at t atmos-units above the surface
  function layer(t) {
    if (t <= 0) return LAYERS[0][1];
    for (let i = 1; i < LAYERS.length; i++) if (t <= LAYERS[i][0]) {
      const [t0, c0] = LAYERS[i - 1], [t1, c1] = LAYERS[i], k = (t - t0) / (t1 - t0), s = k * k * (3 - 2 * k);
      return [c0[0] + (c1[0] - c0[0]) * s, c0[1] + (c1[1] - c0[1]) * s, c0[2] + (c1[2] - c0[2]) * s];
    }
    return LAYERS[LAYERS.length - 1][1];
  }

  const ctx = canvas.getContext('2d');
  const { w, h } = fitBgCanvas(canvas);
  const img = ctx.createImageData(w, h);
  const d = img.data;
  const aspect = w / h, px = 1 / h;
  const cx = aspect / 2 + TILT, cy = TOP + RADIUS;
  // the sun sits on the rim at sunAt, a hair above the surface
  const sa = SUN_AT * 0.55;
  const sx = cx + Math.sin(sa) * (RADIUS + ATMOS * 0.4), sy = cy - Math.cos(sa) * (RADIUS + ATMOS * 0.4);

  for (let j = 0, i = 0; j < h; j++) {
    const y = j / h;
    for (let k = 0; k < w; k++, i += 4) {
      d[i + 3] = 255;
      const x = k / h, dx = x - cx, dy = y - cy;
      const alt = Math.sqrt(dx * dx + dy * dy) - RADIUS;
      const along = Math.atan2(dx, -dy) / 0.55;

      // how strongly the sun lights this part of the atmosphere: a hot core, a long tail
      const off = along - SUN_AT;
      const lit = Math.exp(-(off * off) / 0.025) * 1.6 + Math.exp(-(off * off) / 0.6) * 0.55 + 0.05;

      let r = 0, g = 0, b = 0;
      if (alt >= 0) {
        // the coloured band, brightest near the surface, thinning with height
        const t = alt / ATMOS;
        const band = Math.exp(-t / 2.2);
        const [lr, lg, lb] = layer(t);
        const haze = Math.exp(-alt / HAZE) * 0.35;
        r = (lr * band * 0.9 + 40 * haze) * lit;
        g = (lg * band * 0.9 + 90 * haze) * lit;
        b = (lb * band * 0.9 + 210 * haze) * lit;
      } else {
        // night side, with the sunlit rim bleeding a little into the edge of the planet
        const depth = -alt;
        const rim = Math.exp(-depth / (ATMOS * 0.5)) * Math.min(1, lit) * 0.5;
        const aa = Math.max(0, 1 - depth / px);   // anti-alias the limb
        r = NIGHT[0] + 255 * (rim * 0.9 + aa * 0.6 * Math.min(1, lit));
        g = NIGHT[1] + 120 * rim + 200 * aa * 0.6 * Math.min(1, lit);
        b = NIGHT[2] + 50 * rim + 160 * aa * 0.6 * Math.min(1, lit);
      }

      // the sun: core, bloom, starburst and a horizontal lens streak; it sits over everything
      const ux = x - sx, uy = y - sy, rs = Math.sqrt(ux * ux + uy * uy);
      const core = Math.exp(-rs / 0.006);
      const bloom = Math.exp(-rs / BLOOM) * 0.55 + Math.exp(-rs / (BLOOM * 0.3)) * 0.6;
      const ang = Math.atan2(uy, ux);
      const rays = Math.pow(Math.abs(Math.cos(ang * RAYS / 2)), 90) * Math.exp(-rs / 0.09) * 0.32;
      // the lens streak is horizontal, like an anamorphic flare, whatever the rim is doing
      const streak = Math.exp(-Math.abs(uy) / 0.0022) * Math.exp(-Math.abs(ux) / 0.35) * 0.7;
      // don't let the sun's light leak onto the night side below the limb
      const shade = alt < 0 ? Math.exp(alt / 0.004) : 1;
      const sun = (core * 3 + bloom + rays) * shade;
      r += 255 * sun + 150 * streak;
      g += 225 * sun + 210 * streak;
      b += 180 * sun + 255 * streak;

      if (r + g + b > 0.6) {
        const n = Math.random() - 0.5;
        d[i] = Math.min(CAP, r + n); d[i + 1] = Math.min(CAP, g + n); d[i + 2] = Math.min(CAP, b + n);
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
