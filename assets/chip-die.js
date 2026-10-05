// Model chips: every saved model drawn as a silicon die, generated from its own layers.
// The module tree (stem, block groups, blocks, branches...) becomes nested regions of the floorplan,
// and each layer a block sized by its parameter count. Convolutions draw as rows of bars, linear
// layers as banded stripes; big layers split into a grid of cells, so they read as dense silicon. Same model, same chip: the layout comes
// from the layers, the colours from the name, and the glow from the accuracy.
// Usage: drawChipDie(canvas, { name, info }, cssWidth, cssHeight)

function chipHash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

function chipRandom(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

// Layers grouped into a tree by their dotted names; a leaf is a module with its own parameters
function chipTree(layers) {
  const makeNode = name => ({ name, children: new Map(), params: 0, shape: null });
  const root = makeNode('');
  for (const [path, shape] of Object.entries(layers)) {
    const parts = path.split('.');
    const param = parts.pop();   // weight, bias, ...
    let node = root;
    for (const part of parts) {
      if (!node.children.has(part)) node.children.set(part, makeNode(part));
      node = node.children.get(part);
    }
    node.params += shape.reduce((a, b) => a * b, 1);
    if (param === 'weight' || !node.shape) node.shape = shape;
  }

  function finish(node) {
    let children = [...node.children.values()].map(finish);
    // Norm layers are tiny next to the layers they follow; leave them out rather than draw slivers
    const isNorm = c => !c.children.length && (c.shape || []).length <= 1;
    if (children.some(c => !isNorm(c))) children = children.filter(c => !isNorm(c));
    // A module with its own parameters as well as submodules keeps them as one more block
    if (node.params && children.length) children.push(finish({ ...makeNode(''), params: node.params, shape: node.shape }));
    let done = { ...node, children };
    // Collapse chains of single children, so wrappers don't add empty nesting
    while (done.children.length === 1) done = done.children[0];
    done.weight = done.children.length
      ? done.children.reduce((sum, c) => sum + c.weight, 0)
      : Math.sqrt(Math.max(1, done.params));   // compressed, so small layers still show
    return done;
  }
  return finish(root);
}

// Something chip-like for a model saved without its layer list: a few regions of random blocks
function chipStandIn(name) {
  const rand = chipRandom(chipHash(name));
  const layers = {};
  const groups = 3 + Math.floor(rand() * 4);
  for (let g = 0; g < groups; g++) {
    const blocks = 1 + Math.floor(rand() * 4);
    for (let b = 0; b < blocks; b++) {
      const out = 2 ** (4 + Math.floor(rand() * 5));
      layers[`g${g}.b${b}.weight`] = rand() < 0.6 ? [out, out, 3, 3] : [out, out * 2];
      layers[`g${g}.n${b}.weight`] = [out];
    }
  }
  return layers;
}

// Strip treemap: children laid out in rows (about square overall), widths shared by weight
function chipLayout(node, x, y, w, h, depth, out) {
  out.push({ node, x, y, w, h, depth });
  const kids = node.children;
  if (!kids.length || w < 4 || h < 4) return;
  const pad = depth === 0 ? 3 : depth < 3 ? 2 : 1;   // inset inside the region's outline
  const gap = depth < 2 ? 2 : 1;
  const ix = x + pad, iy = y + pad, iw = w - 2 * pad, ih = h - 2 * pad;
  if (iw < 3 || ih < 3) return;

  const perRow = Math.max(1, Math.min(kids.length, Math.round(Math.sqrt(kids.length * iw / ih))));
  const rows = [];
  for (let i = 0; i < kids.length; i += perRow) rows.push(kids.slice(i, i + perRow));
  const total = kids.reduce((s, c) => s + c.weight, 0);
  const rowSpace = ih - gap * (rows.length - 1);

  let cy = iy;
  for (const row of rows) {
    const rowWeight = row.reduce((s, c) => s + c.weight, 0);
    const rh = rowSpace * rowWeight / total;
    const colSpace = iw - gap * (row.length - 1);
    let cx = ix;
    for (const child of row) {
      const cw = colSpace * child.weight / rowWeight;
      chipLayout(child, cx, cy, cw, rh, depth + 1, out);
      cx += cw + gap;
    }
    cy += rh + gap;
  }
}

// One block's pattern, by layer kind
function chipPattern(ctx, shape, x, y, w, h, line, fill) {
  if (w < 3 || h < 3) return;
  ctx.strokeStyle = line;
  ctx.beginPath();
  if (shape.length >= 3) {
    // convolution: a row of bars per kernel row, as many bars as the channels suggest
    const n = Math.max(2, Math.min(Math.floor(w / 2.5), Math.round(Math.log2(shape[0] + 1) * 1.6)));
    const bands = h > 16 ? Math.max(1, Math.min(shape[2] || 1, Math.floor(h / 8))) : 1;
    const bh = (h - 4) / bands;
    for (let b = 0; b < bands; b++) {
      const top = y + 2 + b * bh, bottom = top + bh - (bands > 1 ? 2 : 0);
      for (let i = 1; i <= n; i++) {
        const lx = Math.round(x + (w * i) / (n + 1)) + 0.5;
        ctx.moveTo(lx, top); ctx.lineTo(lx, bottom);
      }
      if (b < bands - 1) { ctx.moveTo(x + 2, bottom + 1); ctx.lineTo(x + w - 2, bottom + 1); }
    }
  } else if (shape.length === 2) {
    // linear: stripes across, as many as the outputs suggest
    const n = Math.max(2, Math.min(Math.floor(h / 2.5), Math.round(Math.log2(shape[0] + 1) * 1.2)));
    for (let i = 1; i <= n; i++) {
      const ly = Math.round(y + (h * i) / (n + 1)) + 0.5;
      ctx.moveTo(x + 2, ly); ctx.lineTo(x + w - 2, ly);
    }
  } else {
    // anything else: a lit strip
    ctx.fillStyle = fill;
    ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
  }
  ctx.stroke();
}

function drawChipDie(canvas, model, cssW, cssH) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const info = model.info;
  const layers = info?.parameters?.layers && Object.keys(info.parameters.layers).length
    ? info.parameters.layers
    : chipStandIn(model.name);
  const accuracy = info?.results?.final_accuracy ?? 0.5;

  // Colour sweeps across the die like the neon frame on a die shot; where it starts is the model's own
  const baseHue = chipHash(model.name) % 360;
  const hueAt = (x, y) => baseHue + 200 * (x / cssW) + 50 * (y / cssH);
  const color = (x, y, alpha) => `hsla(${hueAt(x, y) % 360}, 90%, 62%, ${alpha})`;

  ctx.fillStyle = '#07070b';
  ctx.fillRect(0, 0, cssW, cssH);

  const frame = 3;
  const rects = [];
  chipLayout(chipTree(layers), frame, frame, cssW - 2 * frame, cssH - 2 * frame, 0, rects);

  // Better models glow brighter
  const glow = 0.45 + 0.55 * Math.max(0, Math.min(1, accuracy));
  ctx.lineWidth = 1;
  ctx.shadowBlur = 3 * glow;

  for (const { node, x, y, w, h, depth } of rects) {
    if (depth === 0 || w < 2 || h < 2) continue;
    const cx = x + w / 2, cy = y + h / 2;
    ctx.strokeStyle = ctx.shadowColor = color(cx, cy, node.children.length ? 0.5 * glow : 0.85 * glow);
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    if (node.children.length) continue;

    // Big layers are split into a grid of cells, each with the layer's pattern
    const cols = Math.max(1, Math.round(w / 26)), rows = Math.max(1, Math.round(h / 20));
    const cw = w / cols, ch = h / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const sx = x + c * cw, sy = y + r * ch;
        if (cols * rows > 1) {
          ctx.strokeStyle = color(sx + cw / 2, sy + ch / 2, 0.45 * glow);
          ctx.strokeRect(sx + 1.5, sy + 1.5, cw - 3, ch - 3);
        }
        const inset = cols * rows > 1 ? 2 : 0;
        chipPattern(ctx, node.shape || [], sx + inset, sy + inset, cw - 2 * inset, ch - 2 * inset,
          color(sx + cw / 2, sy + ch / 2, 0.7 * glow), color(sx + cw / 2, sy + ch / 2, 0.25 * glow));
      }
    }
  }

  // The frame: the colour sweep all the way round
  ctx.shadowBlur = 6 * glow;
  const grad = ctx.createLinearGradient(0, 0, cssW, cssH);
  for (let i = 0; i <= 6; i++) {
    const t = i / 6;
    grad.addColorStop(t, `hsl(${(baseHue + 250 * t) % 360}, 90%, 60%)`);
  }
  ctx.strokeStyle = grad;
  ctx.shadowColor = `hsl(${(baseHue + 125) % 360}, 90%, 60%)`;   // shadows take a colour, not a gradient
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, cssW - 2, cssH - 2);
  ctx.shadowBlur = 0;
}
