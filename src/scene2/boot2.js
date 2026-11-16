// Scene two's lifecycle — the growth universe, drawn live.
//
// The room, the ring, the red light-painting, the twelve cards and the embers
// are the procedural scene (universe.js): the cards carry Dinesh's own tools,
// the figure is his matted walk, and the whole thing answers the pointer. The
// sequence is time-based and plays once on arrival, then holds a living loop.
//
// This module owns everything around that render: the canvas, the drawing
// loop, the arrival from black, when to decode and when to stop, and the
// reduced-motion still.

import { Universe } from './universe.js';

const DPR_CAP = 1.75;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

export async function initUniverse() {
  const section = document.getElementById('universe');
  const canvas = document.getElementById('uniStage');
  const veil = document.getElementById('uniVeil');
  if (!section || !canvas || !veil) return null;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    powerPreference: 'high-performance',
  });
  if (!gl) {
    section.classList.add('is-fallback');
    return null;
  }

  const uni = new Universe(canvas, gl);
  try {
    await uni.load();
  } catch (err) {
    // a missing texture is not worth a blank section: the CSS fallback plate
    // still carries the title and the room's colour
    section.classList.add('is-fallback');
    return null;
  }

  const state = {
    on: false, entered: false, raf: 0, t: 0, last: 0,
    dpr: Math.min(window.devicePixelRatio || 1, DPR_CAP),
  };

  function resize() {
    const w = section.clientWidth || window.innerWidth;
    const h = section.clientHeight || window.innerHeight;
    // portrait phones cannot afford a full-resolution room at 60fps; the
    // spark count and the pixel budget both step down together
    state.dpr = Math.min(window.devicePixelRatio || 1,
      w < 760 ? 1.25 : DPR_CAP);
    uni.quality = w < 760 ? 0.55 : 1;
    uni.resize(w, h, state.dpr);
    if (reduced) drawStill();
  }

  function drawStill() {
    // honour the preference: the settled composition, held as a frame
    uni.render(8.2, 0.016, 0.5);
    section.classList.add('is-on');
    veil.style.opacity = '0';
    state.entered = true;
  }

  function frame(now) {
    if (!state.on) return;
    state.raf = requestAnimationFrame(frame);
    const dt = state.last ? Math.min(0.05, (now - state.last) / 1000) : 0.016;
    state.last = now;
    state.t += dt;

    const rect = section.getBoundingClientRect();
    // the camera dollies through the scroll of the section, but the sequence
    // itself stays time-based so its choreography cannot be scrubbed away
    const scroll = clamp((window.innerHeight - rect.top)
      / (window.innerHeight + rect.height), 0, 1);

    uni.render(state.t, dt, scroll);

    if (!state.entered && state.t > 0.35) {
      state.entered = true;
      section.classList.add('is-on');
    }
    veil.style.opacity = state.t > 2.4 ? '0' : '1';
  }

  function start() {
    if (state.on || reduced) return;
    state.on = true;
    state.last = 0;
    state.raf = requestAnimationFrame(frame);
  }

  function stop() {
    state.on = false;
    cancelAnimationFrame(state.raf);
  }

  // ---- pointer -------------------------------------------------------------
  // He is the only hit target: the cards lift and brighten under the cursor
  // and the room's parallax follows it.
  const p = uni.pointer;
  section.addEventListener('pointermove', (e) => {
    const r = section.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    p.tx = (x / r.width) * 2 - 1;
    p.ty = 1 - (y / r.height) * 2;
    p.inside = true;
    uni.hovered = reduced ? -1 : uni.pick(x, y);
  }, { passive: true });
  section.addEventListener('pointerleave', () => {
    p.inside = false;
    uni.hovered = -1;
  });

  new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        if (reduced) {
          if (!state.entered) drawStill();
        } else start();
      } else stop();
    }
  }, { threshold: 0.05 }).observe(section);

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') stop();
    else if (section.getBoundingClientRect().top < window.innerHeight
      && section.getBoundingClientRect().bottom > 0) start();
  });

  let resizeId = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeId);
    resizeId = requestAnimationFrame(resize);
  });

  resize();

  // one frame is drawn before the section is ever reached, so the arrival has
  // something to fade up from instead of a flash of empty canvas
  uni.render(0.02, 0.016, 0);

  return { uni, section };
}
