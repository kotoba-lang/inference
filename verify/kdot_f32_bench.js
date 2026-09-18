// Bandwidth + parity of the f32-activation K-quant dots on real Nex tensors: ggml_kdot_f32 (1 row/wg)
// vs ggml_kdot_f32_r8 (8 rows/wg). deno run --unstable-webgpu --allow-read verify/kdot_f32_bench.js <attn_gate.iq4xs.bin> <attn_qkv.q5k.bin> [<ffn_gate_exps.iq4xs.bin>]
const here = new URL("./", import.meta.url);
const [gateP, qkvP, expP] = Deno.args;
const adapter = await navigator.gpu.requestAdapter(); const device = await adapter.requestDevice({requiredLimits: {maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30)}}); device.pushErrorScope("validation");
const pipes = {}; for (const [n, f] of [["r1", "ggml_kdot_f32.wgsl"], ["r8", "ggml_kdot_f32_r8.wgsl"], ["v3", "ggml_kdot_f32_v3.wgsl"], ["v3b", "ggml_kdot_f32_v3b.wgsl"]]) pipes[n] = device.createComputePipeline({layout: "auto", compute: {module: device.createShaderModule({code: await Deno.readTextFile(new URL("../shaders/" + f, here))}), entryPoint: "main"}});
const SU = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, UU = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const buf = (d, u = SU) => { const b = device.createBuffer({size: Math.max(16, Math.ceil(d.byteLength / 4) * 4), usage: u}); device.queue.writeBuffer(b, 0, d); return b; };
let seed = 9; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
async function read(b, n) { const st = device.createBuffer({size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}); const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); device.queue.submit([e.finish()]); await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); return a; }
async function bench(name, w, rows, cols, type, experts = null) {
  const nsel = experts ? experts.length : 1; const x = Float32Array.from({length: cols}, () => (rnd() * 2 - 1));
  const meta = buf(new Uint32Array([rows, cols, type, nsel, 0, 0, 0, 0]), UU), wb = buf(w), xb = buf(x), ids = buf(experts ?? new Uint32Array([0]));
  const outs = {}, res = {};
  for (const [n, pipe] of Object.entries(pipes)) {
    const ob = buf(new Float32Array(rows * nsel)); outs[n] = ob;
    const bg = device.createBindGroup({layout: pipe.getBindGroupLayout(0), entries: [meta, wb, xb, ob, ids].map((b, i) => ({binding: i, resource: {buffer: b}}))});
    const gx = n === "r1" ? rows : Math.ceil(rows / 8);
    const run = async (K) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); for (let i = 0; i < K; i++) p.dispatchWorkgroups(gx, nsel, 1); p.end(); const t0 = performance.now(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); return performance.now() - t0; };
    await run(1); const one = Math.min(await run(1), await run(1), await run(1)), many = Math.min(await run(21), await run(21), await run(21));
    const per = (many - one) / 20; const bytes = rows * nsel * (cols / 256) * {12: 144, 13: 176, 14: 210, 23: 136}[type];
    res[n] = {ms: +per.toFixed(3), "GB/s": +((bytes / 1e9) / (per / 1000)).toFixed(1)};
  }
  const a = await read(outs.r1, rows * nsel); for (const n of ["r8", "v3", "v3b"]) { const b = await read(outs[n], rows * nsel); let worst = 0; for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]) - 1e-5 * Math.abs(a[i])); res[n].parityExcess = +worst.toPrecision(3); res[n].speedup = +(res.r1.ms / res[n].ms).toFixed(2); }
  console.log(name, JSON.stringify(res));
}
await bench("attn_gate IQ4_XS 4096x2048", await Deno.readFile(gateP), 4096, 2048, 23);
await bench("attn_qkv Q5_K 8192x2048", await Deno.readFile(qkvP), 8192, 2048, 13);
if (expP) await bench("ffn_gate_exps IQ4_XS 8 experts x 512x2048", await Deno.readFile(expP), 512, 2048, 23, new Uint32Array([0, 255, 7, 64, 128, 199, 31, 250]));
const v = await device.popErrorScope(); if (v) { console.log("validation:", v.message); Deno.exit(1); }
