// MaCE-Lenia in the browser: a line-for-line port of lenia.sim.Batch.step (flow "mace", kn 1, gn 1) for one world.
// Float64 throughout; FFT size N = p * 2^q (here 160 = 5 * 32).
(function (root) {
  "use strict";

  // ---------- FFT: N = p * 2^q, p odd ----------
  function makeFFT(N) {
    let p = N, M = 1;
    while (p % 2 === 0) { p /= 2; M *= 2; }
    const bits = Math.round(Math.log2(M));
    const rev = new Uint32Array(M);
    for (let i = 0; i < M; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
    const twR = new Float64Array(M / 2), twI = new Float64Array(M / 2);
    for (let k = 0; k < M / 2; k++) { twR[k] = Math.cos(-2 * Math.PI * k / M); twI[k] = Math.sin(-2 * Math.PI * k / M); }
    const WR = new Float64Array(N), WI = new Float64Array(N);   // W_N^j
    for (let j = 0; j < N; j++) { WR[j] = Math.cos(-2 * Math.PI * j / N); WI[j] = Math.sin(-2 * Math.PI * j / N); }
    const sr = new Float64Array(N), si = new Float64Array(N);   // p sub-transforms of length M, stored [b*M + k]
    const cR = new Float64Array(p * N), cI = new Float64Array(p * N), cS = new Uint32Array(p * N);   // combine tables
    for (let k = 0; k < N; k++) for (let b = 0; b < p; b++) {
      const w = (b * k) % N; cR[k * p + b] = WR[w]; cI[k * p + b] = WI[w]; cS[k * p + b] = b * M + (k % M);
    }
    const gather = new Uint32Array(N); for (let b = 0; b < p; b++) for (let a = 0; a < M; a++) gather[b * M + a] = p * a + b;
    function fftM(re, im, off) {                                  // in-place radix-2 on sr/si[off .. off+M)
      for (let i = 0; i < M; i++) { const j = rev[i]; if (j > i) { let t = re[off + i]; re[off + i] = re[off + j]; re[off + j] = t; t = im[off + i]; im[off + i] = im[off + j]; im[off + j] = t; } }
      for (let size = 2; size <= M; size *= 2) {
        const half = size / 2, step = M / size;
        for (let s = 0; s < M; s += size) for (let k = 0; k < half; k++) {
          const a = off + s + k, b = a + half, wr = twR[k * step], wi = twI[k * step];
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
    // forward DFT of (re, im)[base + i*stride], i < N, in place; inverse = conjugate trick (no 1/N here)
    function fft(re, im, base, stride, inverse) {
      for (let j = 0; j < N; j++) {
        const idx = base + gather[j] * stride;
        sr[j] = re[idx]; si[j] = inverse ? -im[idx] : im[idx];
      }
      for (let b = 0; b < p; b++) fftM(sr, si, b * M);
      for (let k = 0, t = 0; k < N; k++) {
        let xr = 0, xi = 0;
        for (let b = 0; b < p; b++, t++) {
          const q = cS[t], yr = sr[q], yi = si[q], wr = cR[t], wi = cI[t];
          xr += yr * wr - yi * wi; xi += yr * wi + yi * wr;
        }
        const idx = base + k * stride;
        re[idx] = xr; im[idx] = inverse ? -xi : xi;
      }
    }
    return fft;
  }

  function transpose(a, N) { for (let y = 0; y < N; y++) for (let x = y + 1; x < N; x++) { const i = y * N + x, j = x * N + y, t = a[i]; a[i] = a[j]; a[j] = t; } }
  function fft2(fft, re, im, N, inverse) {        // rows, transpose, rows, transpose; inverse includes 1/N^2
    for (let y = 0; y < N; y++) fft(re, im, y * N, 1, inverse);
    transpose(re, N); transpose(im, N);
    for (let y = 0; y < N; y++) fft(re, im, y * N, 1, inverse);
    transpose(re, N); transpose(im, N);
    if (inverse) { const s = 1 / (N * N); for (let i = 0; i < N * N; i++) { re[i] *= s; im[i] *= s; } }
  }

  // ---------- the rule ----------
  function kernelFields(rule, N) {
    const D = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dy = Math.min(y, N - y), dx = Math.min(x, N - x);
      D[y * N + x] = Math.sqrt(dy * dy + dx * dx) / rule.R;
    }
    return rule.kernels.map(k => {
      const bs = k.b, B = bs.length, kk = new Float64Array(N * N); let tot = 0;
      for (let i = 0; i < N * N; i++) {
        if (!(D[i] < k.r)) continue;
        const Br = B * D[i] / k.r, ring = bs[Math.min(Math.floor(Br), B - 1)];
        const r = Math.min(Br % 1, 1), c = Math.pow(4 * r * (1 - r), 4);   // kernel core 1 (quad4)
        kk[i] = c * ring; tot += kk[i];
      }
      if (tot > 0) for (let i = 0; i < N * N; i++) kk[i] /= tot;
      return kk;
    });
  }

  class World {
    constructor(rule, N, A0) {
      if (rule.flow !== "mace" || rule.kn !== 1 || rule.gn !== 1) throw new Error("only MaCE, kn 1, gn 1");
      this.N = N; this.C = rule.channels; this.rule = rule;
      const NN = N * N; this.fft = makeFFT(N);
      this.K = rule.kernels.length;
      this.c0 = rule.kernels.map(k => k.c0); this.c1 = rule.kernels.map(k => k.c1);
      this.m = rule.kernels.map(k => k.m); this.s = rule.kernels.map(k => k.s); this.h = rule.kernels.map(k => k.h);
      this.hsum = new Float64Array(this.C);
      rule.kernels.forEach(k => { this.hsum[k.c1] += k.h; });
      for (let c = 0; c < this.C; c++) this.hsum[c] = Math.max(this.hsum[c], 1e-12);
      this.beta = rule.beta;
      // kernel spectra (real kernels, full complex FFT)
      this.kR = []; this.kI = [];
      for (const kk of kernelFields(rule, N)) {
        const re = Float64Array.from(kk), im = new Float64Array(NN);
        fft2(this.fft, re, im, N, false); this.kR.push(re); this.kI.push(im);
      }
      this.A = Float64Array.from(A0);              // (C, N, N)
      this.FR = []; this.FI = []; for (let c = 0; c < this.C; c++) { this.FR.push(new Float64Array(NN)); this.FI.push(new Float64Array(NN)); }
      this.tr = new Float64Array(NN); this.ti = new Float64Array(NN);
      this.Dg = new Float64Array(this.C * NN); this.E = new Float64Array(this.C * NN);
      this.T1 = new Float64Array(NN); this.T2 = new Float64Array(NN);
      this.t = 0;
    }

    // spectra of the C real channels, two at a time packed as re + i*im
    spectra() {
      const N = this.N, NN = N * N, { tr, ti, A } = this;
      for (let c = 0; c < this.C; c += 2) {
        const two = c + 1 < this.C;
        for (let i = 0; i < NN; i++) { tr[i] = A[c * NN + i]; ti[i] = two ? A[(c + 1) * NN + i] : 0; }
        fft2(this.fft, tr, ti, N, false);
        const aR = this.FR[c], aI = this.FI[c], bR = two ? this.FR[c + 1] : null, bI = two ? this.FI[c + 1] : null;
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          const i = y * N + x, j = ((N - y) % N) * N + ((N - x) % N);
          const zr = tr[i], zi = ti[i], cr = tr[j], ci = -ti[j];          // conj(Z[-k])
          aR[i] = 0.5 * (zr + cr); aI[i] = 0.5 * (zi + ci);
          if (two) { bR[i] = 0.5 * (zi - ci); bI[i] = -0.5 * (zr - cr); }  // (Z - conj Z-)/(2i)
        }
      }
    }

    step() {
      const N = this.N, NN = N * N, C = this.C, { tr, ti, Dg, E, T1, T2, A } = this;
      this.spectra();
      Dg.fill(0);
      for (let k = 0; k < this.K; k += 2) {                 // two kernels per inverse FFT
        const two = k + 1 < this.K;
        const a0R = this.FR[this.c0[k]], a0I = this.FI[this.c0[k]], kaR = this.kR[k], kaI = this.kI[k];
        const a1R = two ? this.FR[this.c0[k + 1]] : null, a1I = two ? this.FI[this.c0[k + 1]] : null;
        const kbR = two ? this.kR[k + 1] : null, kbI = two ? this.kI[k + 1] : null;
        for (let i = 0; i < NN; i++) {
          let pr = a0R[i] * kaR[i] - a0I[i] * kaI[i], pi = a0R[i] * kaI[i] + a0I[i] * kaR[i];
          if (two) {
            const qr = a1R[i] * kbR[i] - a1I[i] * kbI[i], qi = a1R[i] * kbI[i] + a1I[i] * kbR[i];
            pr -= qi; pi += qr;                                // P + i*Q
          }
          tr[i] = pr; ti[i] = pi;
        }
        fft2(this.fft, tr, ti, N, true);
        for (const [kk, U] of two ? [[k, tr], [k + 1, ti]] : [[k, tr]]) {
          const m = this.m[kk], s9 = 9 * this.s[kk] * this.s[kk], h = this.h[kk], off = this.c1[kk] * NN;
          for (let i = 0; i < NN; i++) {
            const d = U[i] - m, g = Math.max(0, 1 - d * d / s9);
            Dg[off + i] += h * (g * g * g * g * 2 - 1);      // growth 1 (quad4)
          }
        }
      }
      // MaCE: E = exp(beta * D / hsum); A' = E * box3(A / box3(E))
      for (let c = 0; c < C; c++) {
        const off = c * NN, hs = this.hsum[c];
        for (let i = 0; i < NN; i++) E[off + i] = Math.exp(this.beta * Dg[off + i] / hs);
        box3(E, off, T1, N, T2);                            // T1 = box3(E)
        for (let i = 0; i < NN; i++) T1[i] = A[off + i] / T1[i];
        box3(T1, 0, T1, N, T2);                             // T1 = box3(A / box3(E))
        for (let i = 0; i < NN; i++) A[off + i] = E[off + i] * T1[i];
      }
      this.t++;
    }
  }

  function box3(src, off, dst, N, tmp) {                   // 3x3 Moore sum on the torus (dst may equal src's buffer)
    for (let y = 0; y < N; y++) {
      const ym = ((y - 1 + N) % N) * N, y0 = y * N, yp = ((y + 1) % N) * N;
      for (let x = 0; x < N; x++) tmp[y0 + x] = src[off + ym + x] + src[off + y0 + x] + src[off + yp + x];
    }
    for (let y = 0; y < N; y++) {
      const y0 = y * N;
      for (let x = 0; x < N; x++) {
        const xm = (x - 1 + N) % N, xp = (x + 1) % N;
        dst[y0 + x] = tmp[y0 + xm] + tmp[y0 + x] + tmp[y0 + xp];
      }
    }
  }

  const api = { World, makeFFT, fft2 };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MaCE = api;
})(typeof self !== "undefined" ? self : this);
