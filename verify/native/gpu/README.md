# GPU dispatch from a native Kotoba binary — `:gpu/compute` (wire 42)

Root ADR-2609182400 C. No JVM, no Deno, no Node at run time: the `.kotoba`
programs here are compiled by amu to a kexe, and the kexe asks the host GPU
for work through `typed-cap-call :gpu/compute :string :string`. The Vulkan
mechanism lives in amu's loader (`tools/kexe_gpu_vulkan.c`, supervisor side);
every decision — which kernel, which shape, how many dispatches per command
buffer — is in the `.kotoba` file.

| file | what |
|---|---|
| `probe.kotoba` | `INFO` → `<device>\|<api>\|<maxWgX>\|<unified>` |
| `dot_rows.comp` | the first kernel: `o[row] = Σ w[row,k]·x[k]`, one workgroup per row (GLSL; `glslangValidator -V dot_rows.comp -o dot_rows.spv` at build time — the SPIR-V is data, not a runtime dependency) |
| `dot.kotoba` | MAP a 4096×1024 f32 matrix and a vector by path, one DISPATCH, READ the 4096 outputs back as hex — checked against an f64 twin |
| `bench.kotoba` | one command buffer (`BEGIN … SUBMIT`) of 40 dispatches over a 64 MiB matrix: 2.68 GB of weight reads, ~1.4 Nex tokens |
| `policy.edn` | `{:allow #{[:cap/call 42]}}` |

```
amu compile verify/native/gpu/bench.kotoba --target x86_64-linux --jvm-free --policy verify/native/gpu/policy.edn --output bench.kexe
amu extract-native bench.kexe --symbol main --output bench.bin      # note :offset
cc -std=c11 -O2 -Wall -Wextra -Werror -DKEXE_GPU_VULKAN <amu>/tools/kexe_loader.c -o kexe-loader-gpu -lvulkan
KEXE_RESULT_TYPE=string KEXE_STRUCTURED_REPORT=1 KEXE_FUEL=100000 KEXE_WALL_SECONDS=300 ./kexe-loader-gpu bench.bin <offset> 0 x86_64 42
```

Measured 2026-09-18 (`bench.kotoba`, 3 runs each, submit→fence on the host clock):

| box | GPU / driver | kexe ISA | 40 × 64 MiB | GB/s |
|---|---|---|---|---|
| aiueos-6600hs-1 | Intel Arc Pro B70, ANV 1.4 | x86_64 | 16.8 ms | 160 |
| aiueos K16 | AMD Radeon 680M iGPU, RADV 1.4 | x86_64 | 61 ms | 44 |
| xavier | NVIDIA Tegra Xavier, nvgpu 1.3 | aarch64 | 49 ms (first run 160, cold) | 54 |

`dot.kotoba` returned the same bits on all three (max rel err 2.2e-4 against
the f64 twin; the f32 tree in the shader is the order). Single small dispatch
(16 MiB): B70 0.68 ms, K16 0.80 ms, Xavier 2.47 ms — launch-bound, which is
why the serving path records one command buffer per token (ADR-2609182100 D1).

## The Nex K-quant kernels on the same path (2026-09-19)

`kdot_f32_r8.comp` / `kdot_f32_r1.comp` are the GLSL twins of
`shaders/ggml_kdot_f32_r8.wgsl` / `ggml_kdot_f32.wgsl` (Q4_K 12 / Q5_K 13 /
Q6_K 14 / IQ4_XS 23, f32 activations, five storage bindings). `gen_kdot_guest.cljk`
reads a GGUF's tensor directory and writes a guest that `MAP`s the tensor by
file offset; `kdot_ref.py` is the CPU dequant oracle (first 64 rows),
`kdot_check.py` the comparison. All four formats agree with the oracle on all
three boxes (max rel err ≤ 1.4e-5). Real Nex tensors, 10 dispatches in one
command buffer:

| tensor (type, bytes) | B70 r8 / r1 | K16 iGPU r8 / r1 | Xavier r8 / r1 |
|---|---|---|---|
| `blk.0.attn_qkv` (Q5_K, 11.5 MB) | 25 / **98** GB/s | **21** / 21 | **13** / 1.2 |
| `blk.0.attn_gate` (IQ4_XS, 4.5 MB) | 14 / **57** | **17** / 13 | **11** / 4.4 |
| `blk.0.ffn_gate_exps` expert 0 (IQ4_XS, 0.56 MB) | 2.2 / **16** | 6.5 / **16** | 4.8 / 3.6 |
| `output` lm_head (Q6_K, 417 MB) | **135** / 107 | 38 / **43** | 6.7 / **12** |

Read across: the best layout is per (backend, tensor) — the same finding the
Deno-era table had — and the small tensors are launch-bound (B70 ≈ 0.1 ms per
dispatch inside a buffer, Xavier ≈ 0.5). Two NVIDIA-specific fixes on the way:
a dynamically indexed `ivec4`/constant array goes to local memory on nvgpu
(Q5_K at 1 GB/s until the loops were written out and the IQ4_XS codebook became
four packed words), and a live `VkDevice` at process exit segfaults in the
driver's atexit path (amu #1028 tears down in order). Xavier is still far from
its 54 GB/s f32 number — its own layout is the next co-scientist item.

## One whole recurrent layer on the native path (2026-09-19)

`nex_ops.comp` (15 entry points, `-DOP=<name>`, GLSL twin of `shaders/nex_ops.wgsl`
plus `gate_decay2` and an input offset for `l2norm`) and `deltanet_step.comp`
complete the kernel set for a gated-delta-net layer. `gen_layer0_guest.cljk`
reads the GGUF directory and writes the guest: 20 tensors `MAP`ped by file
offset, 37 buffers, 27 metas written, then **27 dispatches in ONE command
buffer** (`BEGIN … SUBMIT`) — rmsnorm → qkv/gate/alpha/beta kdots → gate_decay
→ conv1d → q/k l2norm → delta-net → gated rmsnorm → out kdot → add_rmsnorm →
router → softmax-top8 → 8-expert gate/up kdots (one dispatch each, `positions=8`)
→ silu·up → 8-expert down → shared expert → weighted sum → residual.
`layer0_ref.py` is the f64 numpy port of the retired Deno reference; token 9707,
position 0, `blk.0`:

| box | command buffer (27 dispatches) | x_out vs f64 | top-8 experts |
|---|---|---|---|
| B70 (ANV) | **0.93 ms** | rel 1.1e-6 | identical |
| K16 iGPU (RADV) | **1.55 ms** | rel 1.1e-6 | identical |
| Xavier (nvgpu, aarch64 kexe) | 48.9 ms | rel 1.0e-6 | identical |

Every intermediate read back (h, qkv, g/β, conv, delta-net out, resid, router,
top-k weights) matches to f32 precision. Estimate from this: 30 recurrent layers
at ~1 ms + 10 attention layers + the 417 MB lm_head ≈ 35–40 ms/token on B70
before any tuning (~25 tok/s class; llama.cpp Vulkan 60, vLLM 103); K16 ~16
tok/s class (llama.cpp CPU-only there: 11); Xavier needs its own kernel layout
(the small kdots are 10× slower than on Mesa).

Found on the way, now refused by the loader: a `WRITE`/`MAP`/`READ` while a
`BEGIN` is open reuses the one command buffer and silently discards the
recorded dispatches (the first layer run answered zeros) — amu #1028 refuses
them by name, and the generator hoists every constant write before `BEGIN`.

## The decode step through N layers, one command buffer (2026-09-19)

`gen_decode_guest.cljk <gguf> <layers> <dir> <out>` generalises the layer
generator: every tensor of layers `0..N-1` MAPped once, recurrent AND attention
layers (q/k/v kdots → per-head rmsnorm → NEOX rope → `attn_decode` with the
T = 1 cache → output kdot), then output norm → Q6_K lm_head (r8) → two-stage
argmax — all recorded into **one command buffer**. `decode_ref.py` is the f64
oracle (token 9707, position 0), `decode_check.py` compares the last hidden
state and the argmax. Depth is bounded by what the device can hold beside the
box's serving process (the full 40 layers are 18.7 GB):

| box | layers (of 40) | dispatches | command buffer | x vs f64 | argmax |
|---|---|---|---|---|---|
| B70 (ANV; vLLM holds 27 of 30 GB) | 4 (3 recurrent + 1 attention) + lm_head | 125 | **3.94 ms** | rel 2.9e-6 | 59315 = ref, logit 8.50742 = ref |
| K16 iGPU (RADV) | 12 + lm_head | 349 | **28.96 ms** | rel 4.6e-6 | 163967 = ref, logit 12.82849 = ref |
| Xavier (nvgpu, aarch64 kexe) | 16 + lm_head | 462 | 140.5 ms | rel 6.5e-6 | 204887 = ref, logit 13.07352 = ref |

Straight-line extrapolation to 40 layers (untuned kernels, position 0):
B70 ≈ 40 ms/token (~25 tok/s; llama.cpp Vulkan 60, vLLM 103), K16 iGPU ≈ 70 ms
(~14 tok/s; llama.cpp CPU-only there 11), Xavier ≈ 350 ms (~3 tok/s; llama.cpp
CUDA 18 — the nvgpu kernel layout is the open item).

Three bounds surfaced by name on the way and raised (amu #1028): 64 → 2048
dispatches per command buffer, 256 → 4096 buffers, the 1 MiB SOURCE bound that
`extract-native` applied to a 1.09 MB artifact (now the 8 MiB EDN bound); and
the guest needs `KEXE_PAIRS=1048576` (one pair per string) beyond ~300 requests.

## Greedy decode over tokens, entirely on the device (2026-09-19, kaizen loop tick 1)

`gen_decode_tokens_guest.cljk <gguf> <layers> <tokens> <prompt-id> <dir> <out>`:
the prompt id is `WRITE`n once; `embed_iq4xs.comp` dequantises the token's
embedding row on the device, reading the id from the prompt buffer for token 0
and from the **argmax buffer** for every later token — no id ever crosses back
to the host. State stays in device buffers across steps: conv ring, delta-net S,
and the KV cache (`copy_at` appends this token's rope'd k and v at row `pos`;
`attn_decode` runs with `T = pos + 1` and rope with `aux2 = pos`, all as
per-position metas written before the first `BEGIN`). One command buffer per
token. `decode_tokens_ref.py` is the stateful f64 oracle (ring / S / KV
carried), `decode_tokens_check.py` compares every step:

| box | layers | tokens | per-token command buffer | argmax per step |
|---|---|---|---|---|
| B70 (vLLM resident) | 4 | 3 | 16.8 / 11.2 / 6.7 ms | 59315, 149044, 169222 = ref |
| K16 iGPU | 12 | 3 | 28.7 / 27.8 / 27.8 ms | 163967, 1320, 11278 = ref |
| Xavier (aarch64 kexe) | 12 | 3 | 139.8 / 103.3 / 88.6 ms | 163967, 1320, 11278 = ref |

x after the last layer: rel ≤ 7.5e-6 at every step on every box.

A language ceiling met and routed around: binding every handle in one `let`
(358 bindings + the request temporaries over ~1,400 requests) hits kotoba-mir's
`:spill-frame-too-large` at 4,095 slots (the AArch64 `ldr [sp, #imm12*8]` reach).
The loader assigns buffer and pipeline handles sequentially per kind, so the
generator writes every handle as a literal and the program keeps nothing live
between requests — 5× less machine code, and the bound is not approached.

## Xavier (2026-09-19, kaizen loop tick 2)

Two findings, one lever pulled:

- **The GPU clock governor was the largest Xavier cost.** `nvpmodel` said MAXN
  but the GPU ran `nvhost_podgov` from 114 MHz, ramping per burst; our command
  buffers are bursts. With `jetson_clocks` (min = max = 1377 MHz) the 12-layer ×
  3-token chain went from 89–140 ms/token to a flat **59.7 ms/token** (1.5–2.3×).
  kotoba-lang/murakumo already ships `murakumo-xavier-performance.service` for
  exactly this; it was *disabled* on the box. Enabled and started 2026-09-19
  (persists across boots). The production llama-server there: 18.4 → 18.7 tok/s
  (sustained decode was already ramped; bursts were not).
- **The r1 structure, not the dequant arithmetic, is what nvgpu dislikes.**
  `kdot_ctrl_r1.comp` (r1 with a trivial one-load dequant — a probe, not a
  kernel) reaches only 5.8 GB/s where the f32 `dot_rows` reaches 54. `kdot_x8.comp`
  (256 threads per row, eight consecutive values per thread, `vec4` x loads,
  contiguous warp loads) brings Q5_K attn_qkv from 1.3 → **14.1 GB/s** and IQ4_XS
  attn_gate 4.4 → **10.5** — parity with r8 on nvgpu, and correct (rel ≤ 1.6e-5).
  In the composed 12-layer step it is *slower* than r1 (69 vs 60 ms): the many
  tiny kdots (32-row alpha/beta, 512-row experts) pay for 256-thread groups.
  The lm_head Q6_K stays at 17–20 GB/s on every layout (the 210-byte blocks are
  misaligned; a load-time repack is the next idea). Memory pressure is a confound
  on this box: 26 of 31 GB used with swap active while the 17 GB llama-server is
  resident.

**Correction (tick 3, same day).** The x8 numbers above were taken BEFORE the
clocks were pinned and are governor-confounded. With `jetson_clocks` in force,
r1 ≈ x8 ≈ r8 on Xavier: Q5_K attn_qkv 13.8 / 14.0 / 11.8 GB/s, IQ4_XS gate
10.8 / 10.4, lm_head 20.4 / 17.6 / 19.5. The structure claim survives in a
weaker form: the control probe reaches 33 GB/s and the f32 kernel 79 GB/s
(pinned), so the K-quant kdot on nvgpu loses ~2.4× to dequant arithmetic and
loads and ~2.4× more to the 64-thread structure. `kdot_s1_r1.comp` (the block's
16-byte header staged through shared memory once per block, scales read from
there) is the first real gain on the arithmetic side: 14 → **17 GB/s**, correct.

`gen_decode_tokens_guest.cljk` now takes a backend (`anv | radv | nvgpu`) and picks
the kernel per tensor class (`:wide` / `:narrow` / `:lm`) from `layout-table`. In the
composed 12-layer × 3-token step the table changes nothing measurable (K16
28.0 vs 27.8 ms; Xavier 60.4 vs 59.7): at this depth K16 is already near its
bandwidth (~0.9 GB of weights actually touched per token — 8 of 256 experts —
in 28 ms ≈ 32 GB/s of ~44), and Xavier's 60 ms is ~15 GB/s, i.e. the kdot
ceiling above plus per-dispatch cost. The next Xavier levers are the dequant
path (s1, then fewer loads per value) and fewer dispatches per layer.

**Tick 4 (iteration 9): `kdot_s2_r1.comp`** = s1 for every format (Q4/Q5 header,
IQ4_XS 8-byte header, Q6_K 16 scales + d through `weight_u32_at`) plus the
activation as one `vec4` load. Xavier, pinned clocks, 10 dispatches in one
command buffer, all oracle-exact (rel ≤ 1e-5):

| tensor | r1 | s1 | **s2** | x8 | s2-on-x8 (`kdot_s2_x8.comp`) |
|---|---|---|---|---|---|
| attn_qkv Q5_K | 14.0 GB/s | 17.0 | **18.9** | 14.0 | 14.4 |
| attn_gate IQ4_XS | 11.0 | 11.8 | **12.6** | 10.5 | 10.3 |
| ffn_gate_exps[0] IQ4_XS | 7.2 | 8.5 | **9.2** | 6.5 | 8.0 |
| output Q6_K | 20.1 | 20.1 | **20.6** | 17.6 | 14.8 |

Header staging does nothing for x8 (a warp's header loads were already one
broadcast); the gain is in the 64-thread r1 shape. With the nvgpu row of
`layout-table` set to s2 for all three classes the composed 12-layer × 3-token
step goes **60.0 → 54.0 ms/token** (argmax chain unchanged: 163967 → 1320 →
11278). On the K16 iGPU (RADV, one wave64 per workgroup) s2 is *slower* than r1
(attn_qkv 31.9 → 22.5 GB/s, experts 16.5 → 9.5; lm_head equal), so s2 stays
nvgpu-only — the layout table is no longer neutral, it carries this. Remaining
Xavier gap: 54 ms ≈ 17 GB/s against the 33 GB/s control probe.

**Tick 5 (iteration 10): barrier groups.** The "~0.5 ms per dispatch" above was
the single-dispatch *submit* latency, not the cost of a dispatch inside a
command buffer. Measured with a probe of N trivial dispatches in one buffer
(amu `test/fixtures/gpu/gpu-dispatch-floor*.kotoba`, slope 270 → 1080):

| box | per dispatch, with the loader's compute→compute barrier | without (`DISPATCHC`) |
|---|---|---|
| B70 ANV | **13 µs** | 0.18 µs |
| K16 RADV | 2.7 µs | 0.15 µs |
| Xavier nvgpu | 3.5 µs | 0.35 µs |

So dispatch count is a B70 lever (125 dispatches ≈ 1.6 of 3.9 ms at 4 layers),
a small one elsewhere, and the barrier — not the launch — is what costs. amu
PR #1029 adds `DISPATCHC`: recorded like `DISPATCH` but with no barrier before
it; the guest asserts the dispatch reads nothing written, and writes nothing
touched, since the last plain `DISPATCH`. The generator marks 18 of the 27
per-layer dispatches concurrent (qkv|gate|alpha|beta, gate_decay2×2|conv1d,
l2norm q|k, q|k|v, rmsnorm q|k, rope q|k|copy v, router|gate_sh|up_sh|shlogit,
softmax_topk|silu_sh, gate_exps|up_exps|down_sh) and orders the MoE tail so
the groups are contiguous — 27 → 15 barriers per layer. Results bit-identical
on all three boxes (same `x rel`, same argmax chains):

| box | before | after |
|---|---|---|
| B70 4L × 3T | 3.93 ms/token | **3.63** |
| K16 iGPU 12L × 3T | 28.4 | **27.4** |
| Xavier 12L × 3T | 54.0 | **52.8** |

Fusing kernels (rmsnorm+kdot etc.) would remove the same barriers and is now
worth less than it looked; the Xavier budget is still the kdot bandwidth
(~12.6 GB/s over the layers, 20 on the lm_head).

**Tick 6 (iteration 11): what the Xavier kdot ceiling actually is.** Three
probes on top of s2, all oracle-exact (Xavier, pinned clocks, GB/s):

| tensor | s2 | s3 (`subgroupAdd` reduction) | s3_r8 (8 rows/wg, named accumulators) | **s4** (all headers staged once, 2 blocks/iter) |
|---|---|---|---|---|
| attn_qkv Q5_K | 19.0 | 19.7 | 20.1 | 19.6 |
| attn_gate IQ4_XS | 12.1 | 13.0 | 11.3 | **13.7** |
| ffn_gate_exps[0] IQ4_XS | 8.4 | 9.2 | 6.4 | **9.6** |
| output Q6_K | 21.0 | 21.6 | 16.9 | **23.7** |

Refuted: the reduction tree (s3: +3%), the workgroup count (s3_r8: no
better), and the Q5_K "one word per four values" repack premise — the r1 layout
already touches each qs word twice and each qh word eight times from L1, so
the traffic is minimal and a 20-bit packing would add 55% bytes. What the
numbers say instead: in **values per second** every format sits at 26–29 G/s
(Q5_K 28.5, IQ4_XS 25.8, Q6_K 28.9) while the f32 kernel does 19.8 and the
trivial-dequant control 48 — an **instruction-issue ceiling** (~25
instructions per value on 512 cores × 1.377 GHz), not a memory one. s4 is the
nvgpu row now: composed 12-layer × 3-token **52.8 → 48.6 ms/token**.

Next levers are per-value instruction count: the Xavier driver reports
`shaderFloat16` + `shaderInt8` + 16-bit storage but **no
`VK_KHR_shader_integer_dot_product`**, so packed `f16vec2` FMAs (two values per
instruction) and the exponent-trick int→float conversion are the candidates;
`dp4a`-style int8 activations are not available on this driver.

**Tick 7 (iteration 12): packed-half dequant, `kdot_h2_r1.comp`** — Q4_K / Q5_K
codes become two `f16vec2` by the exponent trick (`0x6400 | q` is the half
1024 + q), IQ4_XS through a 16-entry half codebook in shared memory (one LDS
per value instead of three selects), the activation converted to halves once
per block, products as two packed multiplies. Q6_K keeps the f32 path. Xavier,
pinned clocks:

| tensor | s4 (exact) | h2 | h2 max rel err (random-normal x, 1e-2 floor) |
|---|---|---|---|
| attn_qkv Q5_K | 19.1 GB/s | **21.9** | 4.0e-2 |
| attn_gate IQ4_XS | 13.5 | **22.2** | 1.8e-2 |
| ffn_gate_exps[0] IQ4_XS | 8.7 | **13.7** | 4.5e-3 |
| output Q6_K | 23.7 | 24.1 (same path) | 1.4e-6 |

The error is the format, not a bug: against a reference computed with the
activation rounded to f16 the kernel's rel-RMS error is 8.6e-4 (attn_qkv), of
which the f16 rounding of x alone is 2.2e-4; the max-rel column is the
worst-case row of a random-normal dot with Σ|p|/|Σp| ≈ 120–170. The first
version summed the four products in a packed FMA chain (`qlo*ylo + qhi*yhi`
in f16) and drifted **3e-2** (`x rel`) over 12 layers; summing the four
products in f32 (two packed multiplies, four converts) brings that to
**6e-4 – 3e-3** at the same speed. Composed 12-layer × 3-token on Xavier:
**48.5 → 39.3 ms/token**, argmax chain unchanged.

This is a numerics decision, so h2 is an **opt-in row** (`nvgpu-h2`) and the
`nvgpu` default stays the exact s4. For comparison llama.cpp's Vulkan path
quantizes activations to Q8 per 32-block (≈4e-3 relative per activation),
coarser than f16; a like-for-like distribution comparison against llama.cpp
is the open parity item.

What this is not yet: the 40-layer model on a device with room (the serving
processes own the memory), prompt-side prefill (tokens are fed one at a time),
sampling other than argmax, distribution parity with llama.cpp over a real
prompt, and the per-backend kernel layouts (Xavier).