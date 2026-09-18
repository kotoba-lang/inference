// A full Nex-N2.5-mini (qwen35moe) decode, weights from the served GGUF, every op on the GPU,
// ONE command buffer per token, the only readback being the argmax token id. Greedy tokens are
// compared with llama.cpp (the control host) on the same GGUF and prompt ids.
// Root ADR-2609182100 D1 / co-scientist iteration 2 -- the composed step. Deno WebGPU is the
// bench/oracle HARNESS here, not the runtime (the finished path is Kotoba + amu native).
//
//   deno run --unstable-webgpu --allow-read verify/nex_decode_step.js <model.gguf> <prompt ids csv> <n_predict> [expected ids csv]
//
// Memory: the whole 17.4 GB of weights is uploaded to the GPU; run on a device with >= 24 GB
// free (a 64 GB Apple Silicon Mac, or the B70 with its llama-server stopped).
const here = new URL("./", import.meta.url);
const [modelPath, promptCsv, nPredictArg, expectedCsv] = Deno.args;
if (!modelPath || !promptCsv) { console.error("usage: nex_decode_step.js <model.gguf> <prompt ids csv> <n_predict> [expected ids csv]"); Deno.exit(2); }
const promptIds = promptCsv.split(",").map(Number), nPredict = Number(nPredictArg ?? 8);
const expected = expectedCsv ? expectedCsv.split(",").map(Number) : null;
const T0 = performance.now();
const log = (...a) => console.error(`[${((performance.now() - T0) / 1000).toFixed(1)}s]`, ...a);

// ---------------- GGUF header ----------------
const file = await Deno.open(modelPath, {read: true});
const fstat = await file.stat();
async function readAt(off, len) { const out = new Uint8Array(len); let got = 0; await file.seek(off, Deno.SeekMode.Start); while (got < len) { const n = await file.read(out.subarray(got)); if (n === null) throw new Error("eof"); got += n; } return out; }
const head = await readAt(0, Math.min(fstat.size, 96 * 1024 * 1024));
const hv = new DataView(head.buffer); let pos = 0;
const u32 = () => { const v = hv.getUint32(pos, true); pos += 4; return v; };
const u64 = () => { const v = Number(hv.getBigUint64(pos, true)); pos += 8; return v; };
const gstr = () => { const n = u64(); const s = new TextDecoder().decode(head.subarray(pos, pos + n)); pos += n; return s; };
const SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8};
function readValue(t) {
  if (t === 8) return gstr();
  if (t === 9) { const et = u32(), n = u64(); if (n > 4096) { if (et === 8) { for (let i = 0; i < n; i++) { const l = u64(); pos += l; } } else pos += n * SZ[et]; return {skipped: n}; } const arr = []; for (let i = 0; i < n; i++) arr.push(readValue(et)); return arr; }
  let v; switch (t) { case 0: v = hv.getUint8(pos); break; case 1: v = hv.getInt8(pos); break; case 2: v = hv.getUint16(pos, true); break; case 3: v = hv.getInt16(pos, true); break; case 4: v = hv.getUint32(pos, true); break; case 5: v = hv.getInt32(pos, true); break; case 6: v = hv.getFloat32(pos, true); break; case 7: v = hv.getUint8(pos) !== 0; break; case 10: v = Number(hv.getBigUint64(pos, true)); break; case 11: v = Number(hv.getBigInt64(pos, true)); break; case 12: v = hv.getFloat64(pos, true); break; default: throw new Error("gguf type " + t); }
  pos += SZ[t]; return v;
}
if (new TextDecoder().decode(head.subarray(0, 4)) !== "GGUF") throw new Error("not gguf");
pos = 4; const version = u32(), nTensors = u64(), nKv = u64();
const kv = {}; for (let i = 0; i < nKv; i++) { const k = gstr(); const t = u32(); kv[k] = readValue(t); }
const tensors = {}; for (let i = 0; i < nTensors; i++) { const name = gstr(); const nd = u32(); const dims = []; for (let d = 0; d < nd; d++) dims.push(u64()); const type = u32(); const off = u64(); tensors[name] = {dims, type, off}; }
const alignment = kv["general.alignment"] ?? 32; const dataOff = Math.ceil(pos / alignment) * alignment;
const arch = kv["general.architecture"]; if (arch !== "qwen35moe") throw new Error("expected qwen35moe, got " + arch);
const H = {
  nLayer: kv["qwen35moe.block_count"], nEmbd: kv["qwen35moe.embedding_length"], nHead: kv["qwen35moe.attention.head_count"], nHeadKv: kv["qwen35moe.attention.head_count_kv"],
  headDim: kv["qwen35moe.attention.key_length"], nRot: kv["qwen35moe.rope.dimension_count"], ropeBase: kv["qwen35moe.rope.freq_base"], eps: kv["qwen35moe.attention.layer_norm_rms_epsilon"],
  nExpert: kv["qwen35moe.expert_count"], nExpertUsed: kv["qwen35moe.expert_used_count"], nFfExp: kv["qwen35moe.expert_feed_forward_length"], nFfShexp: kv["qwen35moe.expert_shared_feed_forward_length"],
  convK: kv["qwen35moe.ssm.conv_kernel"], dInner: kv["qwen35moe.ssm.inner_size"], dState: kv["qwen35moe.ssm.state_size"], nKHeads: kv["qwen35moe.ssm.group_count"], nVHeads: kv["qwen35moe.ssm.time_step_rank"],
  fullInterval: kv["qwen35moe.full_attention_interval"] ?? 4, vocab: tensors["output.weight"].dims[1],
};
const isRecr = (il) => ((il + 1) % H.fullInterval) !== 0;
const BLOCK_BYTES = {12: 144, 13: 176, 14: 210, 23: 136, 0: 4};
const rowBytes = (t) => t.type === 0 ? t.dims[0] * 4 : (t.dims[0] / 256) * BLOCK_BYTES[t.type];
const tensorBytes = (t) => t.dims.slice(1).reduce((a, b) => a * b, 1) * rowBytes(t);
log("gguf", {version, nTensors, arch, dataOff}, H);

// ---------------- GPU ----------------
const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error("no adapter");
const device = await adapter.requestDevice({requiredLimits: {maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30), maxStorageBuffersPerShaderStage: Math.max(8, adapter.limits.maxStorageBuffersPerShaderStage >= 8 ? 8 : adapter.limits.maxStorageBuffersPerShaderStage)}});
device.pushErrorScope("validation");
const src = async (f) => await Deno.readTextFile(new URL("../shaders/" + f, here));
const mod = async (f) => device.createShaderModule({code: await src(f)});
const kdotPipe = device.createComputePipeline({layout: "auto", compute: {module: await mod("ggml_kdot_wg.wgsl"), entryPoint: "main"}});
const moePipe = device.createComputePipeline({layout: "auto", compute: {module: await mod("ggml_kdot_moe.wgsl"), entryPoint: "main"}});
const q8Pipe = device.createComputePipeline({layout: "auto", compute: {module: await mod("q8k_quantize.wgsl"), entryPoint: "main"}});
const dnPipe = device.createComputePipeline({layout: "auto", compute: {module: await mod("deltanet_step.wgsl"), entryPoint: "main"}});
const opsMod = await mod("nex_ops.wgsl");
const bgl = device.createBindGroupLayout({entries: [
  {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
  {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
  {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
  {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
  {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
  {binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
  {binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}]});
const opsLayout = device.createPipelineLayout({bindGroupLayouts: [bgl]});
const OPS = {}; for (const e of ["rmsnorm", "l2norm", "gated_rmsnorm", "silu_mul", "conv1d_step", "softmax_topk", "gate_decay", "weighted_sum", "rope_neox", "attn_decode", "argmax_partial", "argmax_final", "add", "f32_matvec"]) OPS[e] = device.createComputePipeline({layout: opsLayout, compute: {module: opsMod, entryPoint: e}});
const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, UU = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const gbuf = (bytes, usage = SU) => device.createBuffer({size: Math.max(16, Math.ceil(bytes / 4) * 4), usage});
const f32buf = (n) => gbuf(n * 4);
const uni = (arr) => { const b = gbuf(Math.max(32, arr.byteLength), UU); device.queue.writeBuffer(b, 0, arr); return b; };
const meta4 = (a, b, c, d) => uni(new Uint32Array([a, b, c, d]));
const meta8 = (a, b, c, d, e, f, g, h) => uni(new Uint32Array([a, b, c, d, e, f, g, h]));
const opsMeta = (n, rows, aux, aux2, eps, scale) => { const u = new ArrayBuffer(32), dv = new DataView(u); dv.setUint32(0, n, true); dv.setUint32(4, rows, true); dv.setUint32(8, aux, true); dv.setUint32(12, aux2, true); dv.setFloat32(16, eps, true); dv.setFloat32(20, scale, true); return uni(new Uint8Array(u)); };
const setOpsMeta = (buf, n, rows, aux, aux2, eps, scale) => { const u = new ArrayBuffer(32), dv = new DataView(u); dv.setUint32(0, n, true); dv.setUint32(4, rows, true); dv.setUint32(8, aux, true); dv.setUint32(12, aux2, true); dv.setFloat32(16, eps, true); dv.setFloat32(20, scale, true); device.queue.writeBuffer(buf, 0, u); };
const dummies = [0, 1, 2, 3, 4].map(() => f32buf(4));
const ent = (bufs) => bufs.map((b, i) => ({binding: i, resource: Array.isArray(b) ? {buffer: b[0], offset: b[1], size: b[2]} : {buffer: b}}));

// upload a tensor's raw bytes to a GPU buffer (chunked reads)
let uploaded = 0;
async function upload(name) {
  const t = tensors[name]; if (!t) throw new Error("missing tensor " + name);
  const bytes = tensorBytes(t); const b = gbuf(bytes);
  const CH = 64 << 20; for (let o = 0; o < bytes; o += CH) { const n = Math.min(CH, bytes - o); const chunk = await readAt(dataOff + t.off + o, n); device.queue.writeBuffer(b, o, chunk, 0, n); }
  uploaded += bytes; return {buf: b, t};
}
// f32 tensor to a JS array (small ones)
async function f32Tensor(name) { const t = tensors[name]; const bytes = tensorBytes(t); const raw = await readAt(dataOff + t.off, bytes); return new Float32Array(raw.buffer, raw.byteOffset, bytes / 4); }
// IQ4_XS row dequant (token embedding)
const KV4 = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const f16 = (u) => { const s = (u >> 15) & 1, e = (u >> 10) & 31, m = u & 1023; if (e === 0) return (s ? -1 : 1) * m * 2 ** -24; if (e === 31) return m ? NaN : (s ? -Infinity : Infinity); return (s ? -1 : 1) * (1 + m / 1024) * 2 ** (e - 15); };
async function embedRow(id) {
  const t = tensors["token_embd.weight"]; if (t.type !== 23) throw new Error("token_embd type " + t.type + " not handled (IQ4_XS only)");
  const rb = rowBytes(t); const raw = await readAt(dataOff + t.off + id * rb, rb); const dv = new DataView(raw.buffer, raw.byteOffset);
  const out = new Float32Array(t.dims[0]); const nb = t.dims[0] / 256;
  for (let b = 0; b < nb; b++) { const o = b * 136; const d = f16(dv.getUint16(o, true)); const sh = dv.getUint16(o + 2, true);
    for (let ib = 0; ib < 8; ib++) { const ls = ((raw[o + 4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4); const dl = d * (ls - 32);
      for (let j = 0; j < 16; j++) { const q = raw[o + 8 + ib * 16 + j]; out[b * 256 + ib * 32 + j] = dl * KV4[q & 0xf]; out[b * 256 + ib * 32 + 16 + j] = dl * KV4[q >> 4]; } } }
  return out;
}

// ---------------- activation buffers ----------------
const E = H.nEmbd, QKV = 2 * H.nKHeads * H.dState + H.nVHeads * H.dState, VD = H.nVHeads * H.dState, KD = H.nKHeads * H.dState;
const A = {
  x: f32buf(E), h: f32buf(E), q8x: gbuf((E / 256) * 292), resid: f32buf(E), h2: f32buf(E), q8h2: gbuf((E / 256) * 292), ffnOut: f32buf(E), hn: f32buf(E), q8hn: gbuf((E / 256) * 292),
  qkv: f32buf(QKV), z: f32buf(VD), alpha: f32buf(H.nVHeads), beta: f32buf(H.nVHeads), g: f32buf(H.nVHeads), conv: f32buf(QKV), qn: f32buf(KD), kn: f32buf(KD), dnOut: f32buf(VD), gnorm: f32buf(VD), q8vd: gbuf((VD / 256) * 292), attnOut: f32buf(E),
  router: f32buf(H.nExpert), topIds: gbuf(H.nExpertUsed * 4), topW: f32buf(H.nExpertUsed), gateE: f32buf(H.nExpertUsed * H.nFfExp), upE: f32buf(H.nExpertUsed * H.nFfExp), actE: f32buf(H.nExpertUsed * H.nFfExp), q8e: gbuf(H.nExpertUsed * (H.nFfExp / 256) * 292), downE: f32buf(H.nExpertUsed * E),
  shGate: f32buf(H.nFfShexp), shUp: f32buf(H.nFfShexp), shAct: f32buf(H.nFfShexp), q8sh: gbuf((H.nFfShexp / 256) * 292), shDown: f32buf(E), shLogit: f32buf(1),
  qFull: f32buf(H.nHead * H.headDim * 2), kProj: f32buf(H.nHeadKv * H.headDim), vProj: f32buf(H.nHeadKv * H.headDim), qNorm: f32buf(H.nHead * H.headDim), kNorm: f32buf(H.nHeadKv * H.headDim), qRope: f32buf(H.nHead * H.headDim), kRope: f32buf(H.nHeadKv * H.headDim), attnO: f32buf(H.nHead * H.headDim), q8attn: gbuf(((H.nHead * H.headDim) / 256) * 292),
  logits: f32buf(H.vocab), amaxPart: f32buf(Math.ceil(H.vocab / 4096)), amaxIds: gbuf((Math.ceil(H.vocab / 4096) + 1) * 4), amaxVal: f32buf(4),
};
const T_MAX = 512;
const zero = (b, n) => device.queue.writeBuffer(b, 0, new Float32Array(n));

// ---------------- per-layer resources ----------------
const layers = [];
for (let il = 0; il < H.nLayer; il++) {
  const L = {il, recr: isRecr(il)};
  const w = async (suffix) => await upload(`blk.${il}.${suffix}`);
  L.attnNorm = await w("attn_norm.weight"); L.postNorm = await w("post_attention_norm.weight");
  if (L.recr) {
    L.qkv = await w("attn_qkv.weight"); L.gate = await w("attn_gate.weight"); L.alpha = await w("ssm_alpha.weight"); L.beta = await w("ssm_beta.weight");
    L.conv1d = await w("ssm_conv1d.weight"); L.dt = await w("ssm_dt.bias"); L.ssmA = await w("ssm_a"); L.ssmNorm = await w("ssm_norm.weight"); L.out = await w("ssm_out.weight");
    L.convRing = f32buf(3 * QKV); zero(L.convRing, 3 * QKV); L.state = f32buf(H.nVHeads * H.dState * H.dState); zero(L.state, H.nVHeads * H.dState * H.dState);
  } else {
    L.q = await w("attn_q.weight"); L.k = await w("attn_k.weight"); L.v = await w("attn_v.weight"); L.o = await w("attn_output.weight"); L.qNormW = await w("attn_q_norm.weight"); L.kNormW = await w("attn_k_norm.weight");
    L.kCache = f32buf(T_MAX * H.nHeadKv * H.headDim); L.vCache = f32buf(T_MAX * H.nHeadKv * H.headDim);
  }
  L.gateInp = await w("ffn_gate_inp.weight"); L.gateInpSh = await w("ffn_gate_inp_shexp.weight");
  L.gateExps = await w("ffn_gate_exps.weight"); L.upExps = await w("ffn_up_exps.weight"); L.downExps = await w("ffn_down_exps.weight");
  L.gateSh = await w("ffn_gate_shexp.weight"); L.upSh = await w("ffn_up_shexp.weight"); L.downSh = await w("ffn_down_shexp.weight");
  layers.push(L);
  if (il % 5 === 4) log(`layer ${il} uploaded, ${(uploaded / 1e9).toFixed(2)} GB so far`);
}
const outNorm = await upload("output_norm.weight"); const outW = await upload("output.weight");
await device.queue.onSubmittedWorkDone();
log(`weights uploaded: ${(uploaded / 1e9).toFixed(2)} GB`);

// ---------------- bind groups (static) ----------------
const kdotBG = (W, q8, out, rows, cols, outOffsetBytes = 0, weightOffsetBytes = 0, weightBytes = null) => {
  const m = meta4(rows, cols, W.t.type, 1);
  const wres = weightBytes ? [W.buf, weightOffsetBytes, weightBytes] : W.buf;
  const ores = outOffsetBytes ? [out, outOffsetBytes, rows * 4] : out;
  return {bg: device.createBindGroup({layout: kdotPipe.getBindGroupLayout(0), entries: ent([m, wres, q8, ores])}), rows};
};
const moeBG = (W, q8, out, rowsPerExpert, cols, perExpertInput) => ({bg: device.createBindGroup({layout: moePipe.getBindGroupLayout(0), entries: ent([meta8(rowsPerExpert, cols, W.t.type, H.nExpertUsed, perExpertInput ? 1 : 0, 0, 0, 0), W.buf, q8, out, A.topIds])}), rows: rowsPerExpert});
const q8BG = (x, q8, blocks, rows) => ({bg: device.createBindGroup({layout: q8Pipe.getBindGroupLayout(0), entries: ent([meta4(blocks, rows, 0, 0), x, q8])}), blocks, rows});
const opBG = (metaBuf, a, b, c, o, s, ids) => device.createBindGroup({layout: bgl, entries: ent([metaBuf, a, b ?? dummies[0], c ?? dummies[1], o, s ?? dummies[2], ids ?? dummies[3]])});
const kdotOutBytes = 0;
for (const L of layers) {
  L.bg = {};
  L.bg.attnNorm = opBG(opsMeta(E, 1, 0, 0, H.eps, 0), A.x, L.attnNorm.buf, null, A.h);
  L.bg.q8x = q8BG(A.h, A.q8x, E / 256, 1);
  if (L.recr) {
    L.bg.qkv = kdotBG(L.qkv, A.q8x, A.qkv, QKV, E); L.bg.z = kdotBG(L.gate, A.q8x, A.z, VD, E);
    L.bg.alpha = kdotBG(L.alpha, A.q8x, A.alpha, H.nVHeads, E); L.bg.beta = kdotBG(L.beta, A.q8x, A.beta, H.nVHeads, E);
    L.bg.gateDecay = opBG(opsMeta(0, H.nVHeads, 0, 0, 0, 0), A.alpha, L.dt.buf, L.ssmA.buf, A.g, A.beta);
    L.bg.conv = opBG(opsMeta(QKV, 1, 0, 0, 0, 0), A.qkv, L.conv1d.buf, null, A.conv, L.convRing);
    L.bg.qn = opBG(opsMeta(H.dState, H.nKHeads, 0, 0, H.eps, 0), [A.conv, 0, KD * 4], null, null, A.qn);
    L.bg.kn = opBG(opsMeta(H.dState, H.nKHeads, 0, 0, H.eps, 0), [A.conv, KD * 4, KD * 4], null, null, A.kn);
    L.bg.dn = device.createBindGroup({layout: dnPipe.getBindGroupLayout(0), entries: ent([meta4(H.nVHeads, H.dState, H.dState, H.nVHeads / H.nKHeads), A.qn, A.kn, [A.conv, 2 * KD * 4, VD * 4], A.g, A.beta, L.state, A.dnOut])});
    L.bg.gnorm = opBG(opsMeta(H.dState, H.nVHeads, 0, 0, H.eps, 0), A.dnOut, L.ssmNorm.buf, A.z, A.gnorm);
    L.bg.q8vd = q8BG(A.gnorm, A.q8vd, VD / 256, 1);
    L.bg.out = kdotBG(L.out, A.q8vd, A.attnOut, E, VD);
  } else {
    L.bg.q = kdotBG(L.q, A.q8x, A.qFull, H.nHead * H.headDim * 2, E); L.bg.k = kdotBG(L.k, A.q8x, A.kProj, H.nHeadKv * H.headDim, E); L.bg.v = kdotBG(L.v, A.q8x, A.vProj, H.nHeadKv * H.headDim, E);
    L.bg.qNorm = opBG(opsMeta(H.headDim, H.nHead, 2 * H.headDim, 0, H.eps, 0), A.qFull, L.qNormW.buf, null, A.qNorm);
    L.bg.kNorm = opBG(opsMeta(H.headDim, H.nHeadKv, 0, 0, H.eps, 0), A.kProj, L.kNormW.buf, null, A.kNorm);
    L.ropeQMeta = opsMeta(H.headDim, H.nHead, H.nRot, 0, 0, H.ropeBase); L.ropeKMeta = opsMeta(H.headDim, H.nHeadKv, H.nRot, 0, 0, H.ropeBase);
    L.bg.ropeQ = opBG(L.ropeQMeta, A.qNorm, null, null, A.qRope); L.bg.ropeK = opBG(L.ropeKMeta, A.kNorm, null, null, A.kRope);
    L.attnMeta = opsMeta(H.headDim, H.nHead, 1, H.nHeadKv, -1.0, 1 / Math.sqrt(H.headDim));
    L.bg.attn = opBG(L.attnMeta, A.qRope, L.kCache, L.vCache, A.attnO, A.qFull);
    L.bg.q8attn = q8BG(A.attnO, A.q8attn, (H.nHead * H.headDim) / 256, 1);
    L.bg.o = kdotBG(L.o, A.q8attn, A.attnOut, E, H.nHead * H.headDim);
  }
  L.bg.resid = opBG(opsMeta(E, 1, 0, 0, 0, 0), A.x, A.attnOut, null, A.resid);
  L.bg.postNorm = opBG(opsMeta(E, 1, 0, 0, H.eps, 0), A.resid, L.postNorm.buf, null, A.h2);
  L.bg.q8h2 = q8BG(A.h2, A.q8h2, E / 256, 1);
  L.bg.router = opBG(opsMeta(E, H.nExpert, 0, 0, 0, 0), L.gateInp.buf, A.h2, null, A.router);
  L.bg.shLogit = opBG(opsMeta(E, 1, 0, 0, 0, 0), L.gateInpSh.buf, A.h2, null, A.shLogit);
  L.bg.topk = opBG(opsMeta(H.nExpert, 1, H.nExpertUsed, 0, 0, 1.0), A.router, null, null, A.topW, null, A.topIds);
  L.bg.gateE = moeBG(L.gateExps, A.q8h2, A.gateE, H.nFfExp, E, false); L.bg.upE = moeBG(L.upExps, A.q8h2, A.upE, H.nFfExp, E, false);
  L.bg.actE = opBG(opsMeta(H.nExpertUsed * H.nFfExp, 1, 0, 0, 0, 0), A.gateE, A.upE, null, A.actE);
  L.bg.q8e = q8BG(A.actE, A.q8e, H.nFfExp / 256, H.nExpertUsed);
  L.bg.downE = moeBG(L.downExps, A.q8e, A.downE, E, H.nFfExp, true);
  L.bg.shGate = kdotBG(L.gateSh, A.q8h2, A.shGate, H.nFfShexp, E); L.bg.shUp = kdotBG(L.upSh, A.q8h2, A.shUp, H.nFfShexp, E);
  L.bg.shAct = opBG(opsMeta(H.nFfShexp, 1, 0, 0, 0, 0), A.shGate, A.shUp, null, A.shAct);
  L.bg.q8sh = q8BG(A.shAct, A.q8sh, H.nFfShexp / 256, 1);
  L.bg.shDown = kdotBG(L.downSh, A.q8sh, A.shDown, E, H.nFfShexp);
  L.bg.wsum = opBG(opsMeta(E, 1, H.nExpertUsed, 0, 0, 0), A.downE, A.topW, A.shDown, A.ffnOut, A.shLogit);
  L.bg.resid2 = opBG(opsMeta(E, 1, 0, 0, 0, 0), A.resid, A.ffnOut, null, A.x);
}
const finalBG = {
  norm: opBG(opsMeta(E, 1, 0, 0, H.eps, 0), A.x, outNorm.buf, null, A.hn), q8: q8BG(A.hn, A.q8hn, E / 256, 1),
  lmChunks: [], parts: Math.ceil(H.vocab / 4096),
};
{ const CH = 62080; const rb = rowBytes(outW.t); for (let r0 = 0; r0 < H.vocab; r0 += CH) { const rows = Math.min(CH, H.vocab - r0); finalBG.lmChunks.push(kdotBG(outW, A.q8hn, A.logits, rows, E, r0 * 4, r0 * rb, rows * rb)); } }
finalBG.amaxP = opBG(opsMeta(H.vocab, 1, finalBG.parts, 0, 0, 0), A.logits, null, null, A.amaxVal, A.amaxPart, A.amaxIds);
finalBG.amaxF = opBG(opsMeta(H.vocab, 1, finalBG.parts, 0, 0, 0), A.logits, null, null, A.amaxVal, A.amaxPart, A.amaxIds);
const idStaging = device.createBuffer({size: (finalBG.parts + 1) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
log("bind groups ready");

// ---------------- one token = one command buffer ----------------
const wgN = (n) => Math.ceil(n / 256);
// The attention layers need this position's k/v in the cache before attn_decode reads them, and
// the copy is a transfer command, so a token is: [pass A: everything up to and including rope for
// every layer]? No -- layers are sequential, so the copy must sit between rope and attn of the same
// layer. WebGPU allows copyBufferToBuffer between compute passes in one command buffer, so the step
// is one command buffer with 10 short pass breaks (one per attention layer), still one submit.
function encodeToken(enc, position) {
  let p = enc.beginComputePass();
  for (const L of layers) {
    p.setPipeline(OPS.rmsnorm); p.setBindGroup(0, L.bg.attnNorm); p.dispatchWorkgroups(1);
    p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8x.bg); p.dispatchWorkgroups(L.bg.q8x.blocks, 1);
    if (L.recr) {
      p.setPipeline(kdotPipe);
      for (const k of [L.bg.qkv, L.bg.z, L.bg.alpha, L.bg.beta]) { p.setBindGroup(0, k.bg); p.dispatchWorkgroups(k.rows, 1, 1); }
      p.setPipeline(OPS.gate_decay); p.setBindGroup(0, L.bg.gateDecay); p.dispatchWorkgroups(1);
      p.setPipeline(OPS.conv1d_step); p.setBindGroup(0, L.bg.conv); p.dispatchWorkgroups(wgN(QKV));
      p.setPipeline(OPS.l2norm); p.setBindGroup(0, L.bg.qn); p.dispatchWorkgroups(H.nKHeads); p.setBindGroup(0, L.bg.kn); p.dispatchWorkgroups(H.nKHeads);
      p.setPipeline(dnPipe); p.setBindGroup(0, L.bg.dn); p.dispatchWorkgroups(H.nVHeads);
      p.setPipeline(OPS.gated_rmsnorm); p.setBindGroup(0, L.bg.gnorm); p.dispatchWorkgroups(H.nVHeads);
      p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8vd.bg); p.dispatchWorkgroups(L.bg.q8vd.blocks, 1);
      p.setPipeline(kdotPipe); p.setBindGroup(0, L.bg.out.bg); p.dispatchWorkgroups(L.bg.out.rows, 1, 1);
    } else {
      p.setPipeline(kdotPipe);
      for (const k of [L.bg.q, L.bg.k, L.bg.v]) { p.setBindGroup(0, k.bg); p.dispatchWorkgroups(k.rows, 1, 1); }
      p.setPipeline(OPS.rmsnorm); p.setBindGroup(0, L.bg.qNorm); p.dispatchWorkgroups(H.nHead); p.setBindGroup(0, L.bg.kNorm); p.dispatchWorkgroups(H.nHeadKv);
      p.setPipeline(OPS.rope_neox); p.setBindGroup(0, L.bg.ropeQ); p.dispatchWorkgroups(H.nHead); p.setBindGroup(0, L.bg.ropeK); p.dispatchWorkgroups(H.nHeadKv);
      p.end();
      const kvBytes = H.nHeadKv * H.headDim * 4;
      enc.copyBufferToBuffer(A.kRope, 0, L.kCache, position * kvBytes, kvBytes);
      enc.copyBufferToBuffer(A.vProj, 0, L.vCache, position * kvBytes, kvBytes);
      p = enc.beginComputePass();
      p.setPipeline(OPS.attn_decode); p.setBindGroup(0, L.bg.attn); p.dispatchWorkgroups(H.nHead);
      p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8attn.bg); p.dispatchWorkgroups(L.bg.q8attn.blocks, 1);
      p.setPipeline(kdotPipe); p.setBindGroup(0, L.bg.o.bg); p.dispatchWorkgroups(L.bg.o.rows, 1, 1);
    }
    p.setPipeline(OPS.add); p.setBindGroup(0, L.bg.resid); p.dispatchWorkgroups(wgN(E));
    p.setPipeline(OPS.rmsnorm); p.setBindGroup(0, L.bg.postNorm); p.dispatchWorkgroups(1);
    p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8h2.bg); p.dispatchWorkgroups(L.bg.q8h2.blocks, 1);
    p.setPipeline(OPS.f32_matvec); p.setBindGroup(0, L.bg.router); p.dispatchWorkgroups(H.nExpert); p.setBindGroup(0, L.bg.shLogit); p.dispatchWorkgroups(1);
    p.setPipeline(OPS.softmax_topk); p.setBindGroup(0, L.bg.topk); p.dispatchWorkgroups(1);
    p.setPipeline(moePipe); p.setBindGroup(0, L.bg.gateE.bg); p.dispatchWorkgroups(H.nFfExp, H.nExpertUsed); p.setBindGroup(0, L.bg.upE.bg); p.dispatchWorkgroups(H.nFfExp, H.nExpertUsed);
    p.setPipeline(OPS.silu_mul); p.setBindGroup(0, L.bg.actE); p.dispatchWorkgroups(wgN(H.nExpertUsed * H.nFfExp));
    p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8e.bg); p.dispatchWorkgroups(L.bg.q8e.blocks, L.bg.q8e.rows);
    p.setPipeline(moePipe); p.setBindGroup(0, L.bg.downE.bg); p.dispatchWorkgroups(E, H.nExpertUsed);
    p.setPipeline(kdotPipe); p.setBindGroup(0, L.bg.shGate.bg); p.dispatchWorkgroups(H.nFfShexp, 1, 1); p.setBindGroup(0, L.bg.shUp.bg); p.dispatchWorkgroups(H.nFfShexp, 1, 1);
    p.setPipeline(OPS.silu_mul); p.setBindGroup(0, L.bg.shAct); p.dispatchWorkgroups(wgN(H.nFfShexp));
    p.setPipeline(q8Pipe); p.setBindGroup(0, L.bg.q8sh.bg); p.dispatchWorkgroups(L.bg.q8sh.blocks, 1);
    p.setPipeline(kdotPipe); p.setBindGroup(0, L.bg.shDown.bg); p.dispatchWorkgroups(E, 1, 1);
    p.setPipeline(OPS.weighted_sum); p.setBindGroup(0, L.bg.wsum); p.dispatchWorkgroups(wgN(E));
    p.setPipeline(OPS.add); p.setBindGroup(0, L.bg.resid2); p.dispatchWorkgroups(wgN(E));
  }
  p.setPipeline(OPS.rmsnorm); p.setBindGroup(0, finalBG.norm); p.dispatchWorkgroups(1);
  p.setPipeline(q8Pipe); p.setBindGroup(0, finalBG.q8.bg); p.dispatchWorkgroups(finalBG.q8.blocks, 1);
  p.setPipeline(kdotPipe); for (const c of finalBG.lmChunks) { p.setBindGroup(0, c.bg); p.dispatchWorkgroups(c.rows, 1, 1); }
  p.setPipeline(OPS.argmax_partial); p.setBindGroup(0, finalBG.amaxP); p.dispatchWorkgroups(finalBG.parts);
  p.setPipeline(OPS.argmax_final); p.setBindGroup(0, finalBG.amaxF); p.dispatchWorkgroups(1);
  p.end();
  enc.copyBufferToBuffer(A.amaxIds, 0, idStaging, 0, (finalBG.parts + 1) * 4);
}
async function step(tokenId, position) {
  device.queue.writeBuffer(A.x, 0, await embedRow(tokenId));
  for (const L of layers) if (!L.recr) { setOpsMeta(L.ropeQMeta, H.headDim, H.nHead, H.nRot, position, 0, H.ropeBase); setOpsMeta(L.ropeKMeta, H.headDim, H.nHeadKv, H.nRot, position, 0, H.ropeBase); setOpsMeta(L.attnMeta, H.headDim, H.nHead, position + 1, H.nHeadKv, -1.0, 1 / Math.sqrt(H.headDim)); }
  const enc = device.createCommandEncoder(); encodeToken(enc, position);
  const t0 = performance.now(); device.queue.submit([enc.finish()]);
  await idStaging.mapAsync(GPUMapMode.READ); const ids = new Uint32Array(idStaging.getMappedRange().slice(0)); idStaging.unmap();
  return {next: ids[finalBG.parts], ms: performance.now() - t0};
}

// ---------------- run ----------------
const generated = []; const times = [];
let position = 0; let next = null;
for (const id of promptIds) { const r = await step(id, position++); next = r.next; times.push(r.ms); }
log(`prompt ${promptIds.length} tokens processed, first prediction ${next}`);
for (let i = 0; i < nPredict; i++) { generated.push(next); const r = await step(next, position++); next = r.next; times.push(r.ms); }
const validation = await device.popErrorScope();
const sorted = [...times].sort((a, b) => a - b);
const report = {gpu: adapter.info?.description, model: modelPath.split("/").pop(), weightsGB: +(uploaded / 1e9).toFixed(2), promptIds, generated, expected, match: expected ? generated.slice(0, expected.length).filter((v, i) => v === expected[i]).length + "/" + Math.min(expected.length, generated.length) : "no expected given", msPerToken: {median: +sorted[Math.floor(sorted.length / 2)].toFixed(1), min: +sorted[0].toFixed(1)}, "tok/s@median": +(1000 / sorted[Math.floor(sorted.length / 2)]).toFixed(2), validation: validation?.message, submitsPerToken: 1};
console.log(JSON.stringify(report));
if (validation) Deno.exit(1);
