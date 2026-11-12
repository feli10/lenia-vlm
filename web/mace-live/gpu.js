// MaCE-Lenia on the GPU (WebGL2): same update as lenia.sim.Batch.step (flow "mace", kn 1, gn 1), float32.
// Per step: (1) all kernel sums by direct convolution + growth + affinity E = exp(beta * D / hsum),
// (2) S = box3(E), (3) A' = E * box3(A / S). Kernels are radially symmetric, so direct and FFT convolution agree.
(function (root) {
  "use strict";
  const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;
  const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
out vec4 o;
ivec2 wrap(ivec2 q, int n){ return ivec2((q.x % n + n) % n, (q.y % n + n) % n); }
`;
  const FS_GROWTH = HEAD + `
uniform sampler2D A; uniform sampler2D K; uniform int N; uniform int KR; uniform int nK;
uniform float m[12]; uniform float s9[12]; uniform float h[12]; uniform vec3 sel[12]; uniform vec3 tgt[12];
uniform vec3 hsum; uniform float beta;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int W = 2 * KR + 1;
  float U[12];
  for (int k = 0; k < 12; k++) U[k] = 0.0;
  for (int dy = -KR; dy <= KR; dy++) for (int dx = -KR; dx <= KR; dx++) {
    ivec2 kc = ivec2(dx + KR, dy + KR);
    vec4 w0 = texelFetch(K, kc, 0);
    if (w0.a < 0.5) continue;                       // alpha of block 0 marks offsets inside some kernel
    vec3 a = texelFetch(A, wrap(p - ivec2(dx, dy), N), 0).rgb;
    vec4 w1 = texelFetch(K, kc + ivec2(0, W), 0), w2 = texelFetch(K, kc + ivec2(0, 2 * W), 0),
         w3 = texelFetch(K, kc + ivec2(0, 3 * W), 0);
    float w[12] = float[12](w1.r, w1.g, w1.b, w1.a, w2.r, w2.g, w2.b, w2.a, w3.r, w3.g, w3.b, w3.a);
    for (int k = 0; k < 12; k++) { if (k >= nK) break; U[k] += w[k] * dot(a, sel[k]); }
  }
  vec3 D = vec3(0.0);
  for (int k = 0; k < 12; k++) {
    if (k >= nK) break;
    float d = U[k] - m[k]; float g = max(0.0, 1.0 - d * d / s9[k]); g = g * g; g = g * g;
    D += h[k] * (g * 2.0 - 1.0) * tgt[k];
  }
  o = vec4(exp(beta * D / hsum), 1.0);
}`;
  const FS_BOX = HEAD + `
uniform sampler2D E; uniform int N;
void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 s = vec3(0.0);
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) s += texelFetch(E, wrap(p + ivec2(dx, dy), N), 0).rgb;
  o = vec4(s, 1.0); }`;
  const FS_UPDATE = HEAD + `
uniform sampler2D A; uniform sampler2D E; uniform sampler2D S; uniform int N;
void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 s = vec3(0.0);
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    ivec2 q = wrap(p + ivec2(dx, dy), N);
    s += texelFetch(A, q, 0).rgb / texelFetch(S, q, 0).rgb;
  }
  o = vec4(texelFetch(E, p, 0).rgb * s, 1.0); }`;
  const FS_BRUSH = HEAD + `
uniform sampler2D A; uniform int N; uniform vec2 c; uniform float sig; uniform int erase; uniform vec3 amt;
void main(){ ivec2 p = ivec2(gl_FragCoord.xy); vec3 a = texelFetch(A, p, 0).rgb;
  float n = float(N); vec2 d = mod(vec2(p) - c + n * 0.5, n) - n * 0.5;
  float b = exp(-dot(d, d) / (2.0 * sig * sig));
  o = vec4(erase == 1 ? a * (1.0 - 0.97 * b) : a + amt * b, 1.0); }`;
  const FS_SHOW = HEAD + `
uniform sampler2D A; uniform int N; uniform float scale; uniform vec2 view;
vec3 at(ivec2 q){ return texelFetch(A, wrap(q, N), 0).rgb; }
void main(){
  vec2 uv = gl_FragCoord.xy / view; uv.y = 1.0 - uv.y;            // data row 0 at the top
  vec2 f = uv * float(N) - 0.5; ivec2 i = ivec2(floor(f)); vec2 t = f - floor(f);
  vec3 v = mix(mix(at(i), at(i + ivec2(1, 0)), t.x), mix(at(i + ivec2(0, 1)), at(i + ivec2(1, 1)), t.x), t.y);
  o = vec4(clamp(v / scale, 0.0, 1.0), 1.0); }`;

  function supported(gl) { return !!(gl && gl.getExtension("EXT_color_buffer_float")); }

  class GPUWorld {
    constructor(gl, N) {
      this.gl = gl; this.N = N;
      const prog = fs => {
        const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
          if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
        const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
        gl.bindAttribLocation(p, 0, "p"); gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
        const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < n; i++) { const nm = gl.getActiveUniform(p, i).name; u[nm.replace(/\[0\]$/, "")] = gl.getUniformLocation(p, nm); }
        return { p, u };
      };
      this.P = { growth: prog(FS_GROWTH), box: prog(FS_BOX), update: prog(FS_UPDATE), brush: prog(FS_BRUSH), show: prog(FS_SHOW) };
      this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
      const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      const tex = () => { const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, N, N, 0, gl.RGBA, gl.FLOAT, null);
        for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
        const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("float framebuffer incomplete");
        return { t, f }; };
      this.A = [tex(), tex()]; this.E = tex(); this.S = tex(); this.cur = 0; this.t = 0;
      this.buf = new Float32Array(4 * N * N);
    }

    load(rule, A0) {                                   // A0: (C, N, N), data row 0 = texture row 0
      const gl = this.gl, N = this.N, NN = N * N;
      if (rule.flow !== "mace" || rule.kn !== 1 || rule.gn !== 1 || rule.kernels.length > 12) throw new Error("unsupported rule");
      this.rule = rule; this.t = 0; this.cur = 0;
      const px = new Float32Array(4 * NN);
      for (let i = 0; i < NN; i++) { for (let c = 0; c < 3; c++) px[4 * i + c] = c < rule.channels ? A0[c * NN + i] : 0; px[4 * i + 3] = 1; }
      gl.bindTexture(gl.TEXTURE_2D, this.A[0].t); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, px);
      // kernels (lenia.sim.kernel_fields) as weights per offset; block 0 alpha = "inside some kernel"
      const maxr = Math.max(...rule.kernels.map(k => k.r)), KR = Math.ceil(rule.R * maxr), W = 2 * KR + 1;
      const kw = new Float32Array(4 * W * 4 * W);
      rule.kernels.forEach((k, ki) => {
        const bs = k.b, B = bs.length, w = new Float64Array(W * W); let tot = 0;
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {        // torus kernel, as in Python
          const dy = Math.min(y, N - y), dx = Math.min(x, N - x), D = Math.sqrt(dx * dx + dy * dy) / rule.R;
          if (!(D < k.r)) continue;
          const Br = B * D / k.r, r = Math.min(Br % 1, 1), v = Math.pow(4 * r * (1 - r), 4) * bs[Math.min(Math.floor(Br), B - 1)];
          tot += v;
          const ox = x <= N / 2 ? x : x - N, oy = y <= N / 2 ? y : y - N;
          if (Math.abs(ox) <= KR && Math.abs(oy) <= KR) w[(oy + KR) * W + (ox + KR)] += v;
        }
        const blk = 1 + Math.floor(ki / 4), ch = ki % 4;
        for (let j = 0; j < W * W; j++) {
          const val = tot > 0 ? w[j] / tot : 0, ox = j % W, oy = Math.floor(j / W);
          kw[4 * ((oy + blk * W) * W + ox) + ch] = val;
          if (val !== 0) kw[4 * (oy * W + ox) + 3] = 1;
        }
      });
      if (this.K) gl.deleteTexture(this.K);
      this.K = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.K);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, 4 * W, 0, gl.RGBA, gl.FLOAT, kw);
      for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
      this.KR = KR;
      const hs = [0, 0, 0]; rule.kernels.forEach(k => { hs[k.c1] += k.h; });
      const one = c => [0, 1, 2].map(i => (i === c ? 1 : 0));
      const pad = (a, n) => a.concat(Array(n - a.length).fill(0));
      this.U = {
        m: new Float32Array(pad(rule.kernels.map(k => k.m), 12)),
        s9: new Float32Array(pad(rule.kernels.map(k => 9 * k.s * k.s), 12)).map(v => v || 1),
        h: new Float32Array(pad(rule.kernels.map(k => k.h), 12)),
        sel: new Float32Array(pad(rule.kernels.flatMap(k => one(k.c0)), 36)),
        tgt: new Float32Array(pad(rule.kernels.flatMap(k => one(k.c1)), 36)),
        hsum: new Float32Array(hs.map(v => Math.max(v, 1e-12)).map((v, i) => (i < rule.channels ? v : 1))),
      };
    }

    _pass(P, target, bind) {
      const gl = this.gl;
      gl.useProgram(P.p); gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.f : null);
      gl.viewport(0, 0, target ? this.N : gl.drawingBufferWidth, target ? this.N : gl.drawingBufferHeight);
      let unit = 0;
      for (const [name, v] of Object.entries(bind)) {
        const loc = P.u[name]; if (loc === undefined) continue;
        if (v && v.tex) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v.tex); gl.uniform1i(loc, unit++); }
        else if (v && v.i !== undefined) gl.uniform1i(loc, v.i);
        else if (v && v.v3) gl.uniform3fv(loc, v.v3);
        else if (v && v.v2) gl.uniform2fv(loc, v.v2);
        else if (v && v.f1) gl.uniform1fv(loc, v.f1);
        else gl.uniform1f(loc, v);
      }
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    step(n = 1) {
      const N = this.N, U = this.U, r = this.rule;
      for (let i = 0; i < n; i++) {
        const A = this.A[this.cur], B = this.A[1 - this.cur];
        this._pass(this.P.growth, this.E, { A: { tex: A.t }, K: { tex: this.K }, N: { i: N }, KR: { i: this.KR }, nK: { i: r.kernels.length },
          m: { f1: U.m }, s9: { f1: U.s9 }, h: { f1: U.h }, sel: { v3: U.sel }, tgt: { v3: U.tgt }, hsum: { v3: U.hsum }, beta: r.beta });
        this._pass(this.P.box, this.S, { E: { tex: this.E.t }, N: { i: N } });
        this._pass(this.P.update, B, { A: { tex: A.t }, E: { tex: this.E.t }, S: { tex: this.S.t }, N: { i: N } });
        this.cur = 1 - this.cur; this.t++;
      }
    }

    brush(fx, fy, erase, amt) {
      const N = this.N, A = this.A[this.cur], B = this.A[1 - this.cur];
      this._pass(this.P.brush, B, { A: { tex: A.t }, N: { i: N }, c: { v2: [fx * N, fy * N] }, sig: this.rule.R * (erase ? 1.0 : 0.8),
        erase: { i: erase ? 1 : 0 }, amt: { v3: amt || [0, 0, 0] } });
      this.cur = 1 - this.cur;
    }

    read() {                                          // -> (C, N, N) Float32Array, data order
      const gl = this.gl, N = this.N, NN = N * N;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.A[this.cur].f);
      gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, this.buf);
      const C = this.rule.channels, out = new Float32Array(C * NN);
      for (let i = 0; i < NN; i++) for (let c = 0; c < C; c++) out[c * NN + i] = this.buf[4 * i + c];
      return out;
    }

    show(scale) {
      const gl = this.gl;
      this._pass(this.P.show, null, { A: { tex: this.A[this.cur].t }, N: { i: this.N }, scale,
        view: { v2: [gl.drawingBufferWidth, gl.drawingBufferHeight] } });
    }
  }
  root.MaCEGPU = { GPUWorld, supported };
})(typeof self !== "undefined" ? self : this);
