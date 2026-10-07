// Shared plumbing for the canvas backgrounds (grain-bg.js, riso-bg.js, silk-bg.js).
// Each painter is just paint(canvas, ...); mountCanvasBg keeps it painted at the canvas's size.

// Size the canvas's pixels to its laid-out size (device pixels, capped at 2x). Returns { w, h, dpr }.
function fitBgCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = canvas.width = Math.max(1, Math.round(rect.width * dpr));
  const h = canvas.height = Math.max(1, Math.round(rect.height * dpr));
  return { w, h, dpr };
}

// Random numbers for grain and dots, made in bulk: fill(buffer) fills a Float32Array with uniform
// 0..1 values from xorshift32. Much quicker than a call per number in the per-pixel loops, and
// statistically the same noise. Seeded from Math.random unless given a seed.
function makeBgRandom(seed = Math.random() * 4294967296) {
  let state = (seed >>> 0) || 1;
  return {
    fill(out) {
      let s = state; // a local is much faster to update than the captured state
      for (let i = 0; i < out.length; i++) {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
        out[i] = (s >>> 0) / 4294967296;
      }
      state = s;
      return out;
    },
  };
}

// Paint now, and again (debounced) whenever the canvas's size really changes; returns a function
// that stops listening. ResizeObserver also fires once on observe; the size check skips that,
// so the random grain doesn't get redrawn and flicker.
function mountCanvasBg(canvas, paint) {
  let t;
  paint(canvas);
  let size = `${canvas.width}x${canvas.height}`;
  const ro = new ResizeObserver(() => {
    clearTimeout(t);
    t = setTimeout(() => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const next = `${Math.max(1, Math.round(rect.width * dpr))}x${Math.max(1, Math.round(rect.height * dpr))}`;
      if (next === size) return;
      paint(canvas);
      size = next;
    }, 150);
  });
  ro.observe(canvas);
  return () => { clearTimeout(t); ro.disconnect(); };
}
