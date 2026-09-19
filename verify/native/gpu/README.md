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

**Tick 8 (iteration 13): the parity probe, and why the first one was worthless.**
The f64 oracle now runs all 40 layers (K16, ~1 min/token) and saves the full
logits; llama-server (K16, CPU, the same GGUF) and vLLM (B70, W4A16 AutoRound)
were asked for the next-token distribution after the same single token 9707
(`prompt` as a token array, no BOS, `n_probs` / `logprobs`). Three engines, three
unrelated answers: oracle top-1 198 (p 0.134), llama.cpp top-1 19 (p 0.119,
198 at 0.099; KL(llama‖oracle) over its top-40 = **1.04 nats**), vLLM top-1 a
CJK fragment at p 0.0048. A single token at position 0 with no BOS is a
degenerate input on which nothing can be concluded — the fact that this is
the only prompt the guests could run was the real gap.

So the generator and the oracle take a **comma-separated prompt**: the first P
steps are forced to those tokens (prefill at decode speed, one command buffer
per token, KV/recurrent state carried), greedy decode continues from step P.
`decode_tokens_ref.py nex.gguf 9707,198,220 12 4` / the guest with the same
argument: K16 12 layers, all four steps match (argmax 163967 / 112516 /
169222 / 169484, `x rel` ≤ 6e-6, 27–28 ms per step). The helper modules
(`layer0_ref.py`, `decode_ref.py`) no longer run their own layer when
imported with a token list. The like-for-like llama.cpp comparison needs the
40-layer guest, which needs per-layer MAP/FREE streaming on the K16's 13 GiB
GTT — the next item.

**Tick 9 (iteration 14): the whole model on the K16 iGPU, and llama.cpp parity.**
`gen_decode_tokens_guest.cljk … radv stream` emits a *function-shaped* program:
`recurrent-layer [xin xout ring S]` and `attention-layer [xin xout kc vc]`
hold the layer's dispatches once; `layer-N [g]` MAPs its tensors, calls the
body, FREEs them; `main` calls the 40 layer functions per token. Two ceilings
forced that shape and both are real: the ABI allows **5 parameters** per
function (kotoba-sema `max-parameters`) and a module **64 KiB of string
literals** (`kotoba.kir.value/string-value-byte-limit`; the straight-line
40 × 5 program was 589 KB of source). Two consequences: the position is now
read on the device (`nex_ops.comp` `dyn_pos`: `p1 != 0` → `ids[0]` from a
16-byte `posbuf` the guest WRITEs once per token; `rope_neox`, `attn_decode`,
`copy_at`), so a layer's dispatches are the same request every token; and the
weight handles are literal even though they are MAPped per layer per token —
the loader hands out the lowest free slot, so every recurrent layer lands on
the same 19 slots and every attention layer on the lowest 16 of them, which the
generator simulates. The non-stream guest is unchanged in behaviour (K16 12 L
re-verified with the new kernels).

Result, K16 iGPU (RADV, GTT 13 GiB; the 17.4 GB GGUF is re-uploaded every
token from the page cache), prompt "The capital of France is" = `760,6511,314,9338,369`:

| | |
|---|---|
| 5 tokens × 40 layers, wall | 25.8 s (uploads; the lm_head buffer alone is 9.8 ms) |
| every step vs f64 oracle | argmax 271 / 314 / 279 / 369 / **11751** all equal; `x rel` ≤ 1.7e-4 (step 0), ≤ 1.5e-5 after |
| final logits vs oracle | max\|Δ\| 1.5e-5, KL(oracle‖gpu) 7e-13 nats |
| **vs llama-server (same GGUF, CPU)** | top-1 **11751 " Paris"** on both; KL(llama‖gpu) over llama's top-40 = **0.0143 nats**; llama mass on its top-40 0.907, gpu 0.911; gpu top-10 ⊂ llama top-40 10/10; p(Paris) 0.549 vs 0.474 |

So the native path computes the same model as llama.cpp; the 0.014 nats are
the two engines' rounding (llama.cpp quantizes activations to Q8 per 32-block).
`parity_check.py <loader output> <ref npz> <llama json>` prints all of the above.
The single-token probe of tick 8 was indeed the input, not the model.

**Tick 10 (iteration 15): the same on Xavier, for the exact and the packed-half
kernels — h2 becomes the nvgpu default.** Streaming fits beside the 17 GB
llama-server (weights re-read from NVMe every token: 85 s for 5 tokens; the
lm_head buffer is 18.1 ms). Same prompt, oracle copied from K16, CUDA
llama-server on the box:

| Xavier, 40 layers | s4 (exact f32) | **h2 (packed half)** |
|---|---|---|
| argmax chain vs oracle | 5/5 | 5/5 |
| `x rel` per step | 8.7e-5, 5e-6, 4e-6, 1e-5, 6e-6 | 6.5e-2, 4.2e-3, 4.8e-3, 4.8e-3, 1.2e-1 |
| final logits vs oracle | max\|Δ\| 6.7e-6, KL 8e-13 | max\|Δ\| 0.32, **KL(oracle‖gpu) 8.5e-4 nats** |
| vs llama-server (CUDA) top-40 | top-1 Paris, **KL 0.0167** | top-1 Paris, **KL 0.0142** |
| p(Paris) | 0.549 | 0.557 (llama 0.474) |

h2's distance from the f64 oracle (8.5e-4 nats) is twenty times smaller than
llama.cpp's own (~0.015), and its distance from llama.cpp equals the exact
kernel's. So `nvgpu` now means h2 for the wide/narrow classes (the lm_head
stays f32), and `nvgpu-exact` keeps s4. The hidden-state `x rel` of h2 is
large on some coordinates (1.2e-1 at step 4) while the distribution moves
8.5e-4 nats — the number that matters for serving is the second; both are
recorded.

**Tick 11 (iteration 16): the guest is a program, not a transcript.** The
default (`fn` mode; `flat` keeps the 2026-09-19 straight-line form for A/B)
computes its handles: `string-from-i64` exists in the profile, so
`recurrent-layer [w0 xin xout ring S]` / `attention-layer [w0 xin xout kc vc]`
build every request from `(+ w0 k)`; weights are MAPped at a stride of 19
handles per layer (attention layers pad with three 16-byte ALLOCs);
`layers [il n]` recurses over the layers (attention when `(bit-and il 3) = 3`,
state index `il - il/4` or `il/4`); `token-step [pos n acc]` recurses over the
tokens and accumulates the answers. The position and the tokens live on the
device: `embed_iq4xs` reads `tokbuf[pos[0]]` (binding 4, `dyn`), `argmax_final`
appends the next token to `tokbuf[pos + 1]` once past the prompt (binding 2 is
writable now), `pos_incr` bumps the counter. So the program no longer depends
on T except for the token buffer's size, and a 12-layer × 4-token guest is
30 KB of source instead of 200 KB. Trap found on the way: an nbb map literal
with more than 8 entries evaluates its values in *hash* order, so the layer's
MAPs came out scrambled — `layer-weights!` now MAPs in `rec-keys` / `att-keys`
order explicitly. Verified: K16 12 L, `9707,198,220` + 1 greedy — all four
steps match the oracle at 27.2–27.7 ms/token (flat: 27.0–27.8); Xavier 12 L × 3
(h2 default) 39.4 ms/token, chain 163967 → 1320 → 11278.

**Tick 12 (iteration 17): the third box.** B70 (Intel Arc Pro B70, ANV) ran the
same 40-layer streaming guest beside vLLM (27 of 30 GB held; the box has only
15 GB of host RAM, so the 17.4 GB GGUF is re-read from disk every token:
44.5 s for 5 tokens, lm_head buffer 6.6–8.6 ms). Same prompt, oracle from K16,
llama.cpp reference from K16:

| box, 40 layers, exact kernels | argmax vs oracle | final logits vs oracle | KL(llama‖gpu), top-40 | p(Paris) |
|---|---|---|---|---|
| K16 RADV | 5/5 | max\|Δ\| 1.5e-5 | 0.0143 | 0.549 |
| Xavier nvgpu (s4) | 5/5 | 6.7e-6 | 0.0167 | 0.549 |
| Xavier nvgpu (h2, default) | 5/5 | 0.32 (KL 8.5e-4) | 0.0142 | 0.557 |
| **B70 ANV** | **5/5** | **1.7e-5** | **0.0143** | **0.549** |

Three GPU architectures, three drivers, two ISAs of the same `.kotoba` program,
one distribution. The llama.cpp distribution-parity item is closed for the
fleet. Streaming stays a correctness tool: on a box that cannot page-cache the
model it is disk-bound.

**Tick 13 (iteration 18): sampling on the device.** `nex_ops.comp` `sample_topp`:
one workgroup over the vocabulary — `p = softmax(logits / T)`, the top-p set is
`{p ≥ τ}` for the largest τ whose mass reaches `top_p` (30-step bisection; the
sorted definition with boundary ties included), the draw is `u · mass` in index
order over 256 contiguous chunks, `u = mix32(seed ^ pos·0x9E3779B9) >> 8 / 2²⁴`
with `pos` from `posbuf`; the token goes to `ids[61]` (read back) and to
`tokbuf[pos+1]` past the prompt. `sample_test.kotoba` draws 256 times from one
fixed logits vector (the oracle's last step, `logits_last.f32`) advancing the
counter with `pos_incr`; `sample_check.py` recomputes every draw in f64 with
the same hash: K16, T 0.8, top-p 0.95 — top-p set 14 tokens (mass 0.9523),
**256/256 draws equal to the oracle**, all inside the set, empirical-vs-exact TV
0.052 (≈ the 256-sample noise). In the token loop (`fn` mode, 9th argument
`sample:T:top_p:seed`; `decode_tokens_ref.py … --sample T top_p seed` is the
oracle) K16 12 L, 3 forced + 3 sampled: **6/6 tokens equal** (18171 / 84249 /
236687 / 112516 / 21356 / 226585). Cost: the single-workgroup sampler is 4.8 ms
per draw on K16 (27.5 → 32 ms/token). Not implemented: top-k, repetition
penalties, min-p.

**Tick 14 (iteration 19): the sampler on 61 workgroups.** Same definition,
split into `sample_part` (61 wg: partial max and partial Z relative to it) →
`sample_probs` (61 wg: global max/Z from the 61 partials, `p_i` into a 1 MB
scratch, lo/hi reset) → 30 × (`sample_bisect_part` (61 wg: mass of `p ≥ mid`
per chunk) + `sample_bisect_step` (1 wg: sum, move lo or hi)) →
`sample_final` (1 wg: index-order draw over the set). 63 dispatches, one
scratch of 1 KB for partials + lo/hi and one for `p_i`. `sample_test.kotoba`
now runs this chain (65 dispatches per draw, one command buffer per draw
because of the 2048-dispatch cap): **256/256 draws equal the oracle**, same
picks as the single-workgroup op, **1.25 ms** per draw command buffer (was
4.8). In the token loop: K16 12 L 3 forced + 3 sampled 6/6, **28.6 ms/token**
(greedy 27.5, single-workgroup sampler 32.0); Xavier 12 L with the exact
kernels 6/6 at 49.7 ms (greedy 48.5); Xavier with the **h2** default 40.7 ms but
the *sampled* tokens differ from the oracle from step 0 (18185 vs 18171 —
neighbouring ids): at 12 layers the distribution is flat, the top-p set is
huge, and h2's ~1e-3 probability shifts move the index-order draw to a nearby
token. That is the format's numerics, not the sampler (the same SPIR-V picks
256/256 on exact probabilities); sampled-output comparisons against an oracle
must use the exact kernels or compare distributions, not token chains. The
single-workgroup `sample_topp` stays in `nex_ops.comp` as the reference form.

**Tick 15 (iteration 20): the whole model resident on one GPU.** The owner
stopped Xavier's llama-server (`murakumo-xavier-nex-n25-mini.service`, 21 GB
of device memory) for this path (2026-09-19, order A → B → C). The `fn`-mode
guest with 40 layers and the `nvgpu` (h2) row then runs resident — 17.4 GB of
weights MAPped once in 19 s, then one command buffer per token:

| Xavier, 40 layers resident, h2 | |
|---|---|
| ms/token (12 tokens: 5 prompt + 7 greedy) | **89.3–89.6, mean 89.5 → 11.2 tok/s** |
| vs oracle | 12/12 argmax equal; final logits as tick 10 |
| greedy continuation of "The capital of France is" | ` Paris.<\|im_end\|>\n<think>\n\n</think>\n\n` |
| llama.cpp CUDA on the same box (stopped) | 18.7 tok/s |

The first full-model number of the native path, and it is what the 12-layer
extrapolation said (1.8 ms/layer × 40 + 18 ms lm_head). The token budget is
72 ms of layers (kdot at the nvgpu instruction-issue ceiling) + 18 ms of
Q6_K lm_head on the f32 path; those two are the A items that follow.

**Tick 16 (iteration 21, A-2): the Q6_K lm_head on the half path — and a
regression caught by A/B.** `kdot_h2_r1.comp -DQ6_HALF` → `kdot_h2q6_r1.spv`:
the 6-bit codes as bytes (ql nibble | qh 2 bits), exponent trick with 1056,
packed multiplies, f32 sums. Bench (Xavier, lm_head Q6_K): s4 23.7 → **26.6
GB/s** (17.7 → 15.9 ms per dispatch; Q6_K's 210-byte blocks keep it
unaligned-load bound), max rel err 3.4e-3 on random x. The first attempt put
the arm into the same binary as the Q5_K / IQ4_XS arms, and the 40-layer
token went 88.9 → **94.7 ms** although the lm_head itself got faster: the extra
arm raised the register pressure of the whole uniform-branch kernel and cost
the other formats ~5 ms per token. Caught only because the previous guest was
re-run in the same session (89.5 → 94.7 with the *same* guest binary, since it
names the `.spv` by path). So the Q6 arm is a separate binary
(`kdot_h2q6_r1.spv`, the `:lm` row), the wide/narrow binary is byte-identical
to tick 15, and the resident 40-layer token is **88.9 → 86.9 ms (11.5 tok/s)**,
12/12 oracle, KL(oracle‖gpu) 1e-7 at the last step. `fn`-mode guests now
append the final logits to the last token's answer, so `parity_check.py` works
on them too (needs `KEXE_STRING_POOL` ≥ 8 MiB).

**Tick 17 (iteration 22, A-3 preparation): where the 71 ms of layers go.**
`profile_guest.py` turns a flat one-token guest into one whose every dispatch
runs alone in its own command buffer (grouped `timed` functions; 112 nested
lets exhausted the desugarer's stack) and reports per op. Xavier, 4 layers,
min of two runs, each figure carrying the ~0.07 ms single-submit floor:

| recurrent layer (sum 4.8 alone; ~1.8 in-buffer) | ms | attention layer (4.2 alone) | ms |
|---|---|---|---|
| qkv 8192×2048 Q5_K | 0.82 | attn_out 2048×4096 | 0.47 |
| gate 4096×2048 IQ4_XS | 0.40 | q 8192×2048 | 0.44 |
| down_exps 2048×8 | 0.37 | down_exps 2048×8 | 0.37 |
| gate_exps / up_exps 512×8 | 0.26 / 0.26 | gate_exps / up_exps | 0.26 / 0.26 |
| deltanet | 0.26 | gate_sh / up_sh 512 | 0.21 / 0.22 |
| ssm_out 2048×4096 | 0.24 | softmax_topk | 0.19 |
| gate_decay2 (beta) | 0.23 | router f32 | 0.16 |
| alpha 32 / beta 32 | 0.22 / 0.15 | add_rmsnorm | 0.14 |

Reading: the big K-quant kdots (qkv, gate, ssm_out/attn_out, the three expert
kdots) are ~75% of a layer and all sit at the nvgpu instruction-issue ceiling
measured in tick 6 (~28 G values/s); fusing the element-wise ops around them
would buy 10–15%. The tiny kdots (alpha/beta 32 rows, gate_sh/up_sh 512) cost
0.1–0.2 ms each because 32–512 workgroups of 64 threads cannot fill 8 SMs —
a split-K layout for narrow kdots is the one cheap win left (~0.25 ms/layer,
≈10 ms/token). Budget arithmetic for the A target: the token touches ~2.6 G
weight values; at 28 G values/s that is ~90 ms, and llama.cpp CUDA's 18.7
tok/s (53 ms) needs ~50 G values/s — what `dp4a` gives it and what this
Vulkan driver does not expose (`VK_KHR_shader_integer_dot_product` absent,
tick 6). Realistic Vulkan ceiling on Xavier with the remaining items: ~75 ms
≈ 13 tok/s, i.e. 0.7× llama.cpp CUDA. That is a decision for the owner, not a
kernel.

**Tick 18 (iteration 23, B-1): the same profile on B70, and the barrier
halved.** `profile_guest.py` on B70 (ANV, vLLM resident): the single-submit
floor is ~0.11 ms and almost every op sits *at* it — subtracting it, a
recurrent layer is qkv 0.09, deltanet 0.07, down_exps 0.06, gate 0.05,
gate_exps / up_exps 0.05 each, ssm_out 0.04, everything else ≈ 0; the lm_head
(r8) 1.59 ms = 262 GB/s, so the B70's memory is faster than the 160 GB/s the
f32 probe suggested (that probe was compute-bound). Kernel work per layer
≈ 0.35 ms, in-buffer layer ≈ 0.46 ms → the difference is the **15 barriers ×
13 µs**, i.e. ~40% of a layer — on B70 the barrier is the item, the opposite
of Xavier. amu PR #1031: when the device is Vulkan 1.3 the loader records a
`synchronization2` barrier naming `SHADER_STORAGE_WRITE → STORAGE_READ|WRITE`
only; the 1.0 `SHADER_READ` also covers uniform / sampled reads and ANV flushes
twice as much for it. Floor per dispatch **13 → 6.6 µs**; the 4-layer × 3-token
guest **3.81 → 3.50 ms/token**, bit-identical. RADV: no difference; Xavier's
JetPack headers are Vulkan 1.2, so the block compiles out there. Corrected
B70 estimate for 40 layers: 40 × ~0.40 + 1.8 ≈ **18 ms/token ≈ 55 tok/s**
(the earlier "~28 tok/s" multiplied the 4-layer figure without separating the
lm_head). Remaining B levers, in order: fusing the ~20 element-wise dispatches
per layer into the kdots (the other half of the barrier cost), expert kdot
bandwidth (90 → 200+ GB/s), qkv at 128 GB/s.

**Tick 19 (iteration 24, B-2): the fused delta-net step — correct, and
neutral.** `deltanet_fused.comp` does, per head, in one dispatch of 128
threads: conv1d + silu on the head's q / k / v channels (the conv ring is
double-buffered by the position's parity, because q / k channels are shared by
two heads), the l2 norms, the gate and beta from alpha / betaL / dt / a, the
delta rule on S, and the gated rmsnorm × silu(z) — seven dispatches and four
barriers of the recurrent path become one (27 → 21 dispatches per recurrent
layer; 13 bindings, so amu #1032 raises the cap to 16). Oracle-exact on both
boxes. Timing, A/B twice each: B70 4 L × 3 T **3.51 → 3.49 ms/token**; Xavier
12 L × 3 T **37.3 → 37.3 ms**. So the "barriers are 40% of a B70 layer" reading
of tick 18 was the floor probe's artefact: between *real* dispatches the drain
overlaps with work and a removed barrier is worth ~1 µs, not 6.6. Fusion is
the default (`<mode>-nofuse` keeps the seven-dispatch path for A/B) because it
is simpler and never slower, but it is not a B lever; the B70 layer's ~0.46 ms
is kernel time plus launch gaps of small dispatches, and the next B levers are
the kdots themselves (expert kdots at ~90 GB/s and qkv at ~130 on a ≥ 260 GB/s
device).

**Tick 20 (iteration 25, B-3): the B70 kernel zoo at a pinned clock.** The B70
idles at **400 MHz** (`/sys/class/drm/card0/device/tile0/gt0/freq0`, max 2800)
and a ten-dispatch bench does not ramp it: the same kernels measured 19–26 GB/s
cold and 120–260 GB/s with `min_freq` pinned — the Xavier governor story again
(tick 2). Composed decode steps ramp the clock themselves (4 L × 3 T 3.50 ms
either way), so only the micro-benches were wrong. Pinned, GB/s:

| tensor | r1 | s2 | s4 | h2 | x8 | **r8** | s3_r8 |
|---|---|---|---|---|---|---|---|
| attn_qkv Q5_K | 121 | 120 | 141 | 134 | 143 | **170** | 105 |
| attn_gate IQ4_XS | 90 | 79 | 91 | 102 | 97 | 91 | 104 |
| ffn_gate_exps[0] (1 expert) | 30 | 29 | 31 | 31 | 30 | 14 | 15 |
| output Q6_K | 151 | 133 | 195 | 192 | 173 | **261** | 187 |

The `anv` row's `:wide` class moves to r8 (qkv +40%); `:narrow` stays r1 (r8
starves on 512-row tensors); `:lm` was r8 already. Composed 4 L × 3 T on B70:
**3.50 → 3.45 ms/token** — the wide kdots are a small share of the layer. The
B70 budget, restated with these numbers: a recurrent layer's ~36 MB of weights
at 260 GB/s is 0.14 ms; the layer takes ~0.46. The rest is latency: the
delta-net step is 32 workgroups × 128 threads each walking two serial 128-step
loops (~0.065 ms, ~2 ms per 40-layer token), the expert kdots run at ~90 GB/s,
and ~20 small dispatches per layer each cost their launch. Those three, in that
order, are the remaining B levers.

**Tick 21 (iteration 26, B-4): the delta-net step on 512 threads.**
`deltanet_fused.comp` now runs 512 threads per head: thread `(j = t & 127,
part = t >> 7)` owns value column j and a quarter of the 128 keys in both state
loops, the four partial `kv_mem` / `o` are summed through shared memory, the
conv / l2 / gated-rmsnorm parts stay on part 0. Oracle-exact on all three boxes
(B70 `x rel` 4.9e-6, Xavier 3.1e-3 with h2, K16 chain equal). Per token,
A/B'd: **B70 4 L × 3 T 3.46 → 3.30 ms** (−0.053 ms per recurrent layer, ≈ −1.6 ms
on 40 layers), **Xavier 12 L × 3 T 37.1 → 36.4**, **K16 12 L 27.3 → 27.3** (RADV's
wave64 already had the parallelism). `profile_guest.py --report` knows the
fused layer's 21 labels.

**Tick 22 (iteration 27, B-5): the expert kdots get their own class.** The
layout table has a fourth class, `:expert` (the three 8-position expert kdots;
falls back to `:narrow` when absent). On B70 the experts move to r8: 4 L × 3 T
**3.30 → 3.06 ms/token** (−0.06 ms per layer, ≈ −2.4 ms on 40 layers),
oracle-exact; the same for the *narrow* class (alpha/beta 32 rows, k/v/shared
experts 512–2048) **loses** (3.15 — small row counts starve r8's 8-rows-per-
workgroup shape), so `anv` is now wide r8 / narrow r1 / expert r8 / lm r8. On
K16 (RADV) experts on r8 are neutral (27.2 vs 27.3, bandwidth-bound), so the
`radv` row is unchanged. B70's 40-layer estimate: ~14 ms/token, ~70 tok/s class
(still an extrapolation from 4 layers — a resident run needs vLLM's memory).

**Tick 23 (iteration 28, B-6): consolidation measured, one class split found.**
`kdot_f32_r8_dual.comp` computes gate and up for the same eight (expert, row)
pairs from one x read and writes `silu(gate)·up` directly — three dispatches
(gate_exps, up_exps, silu_mul) become one. Correct, and **slower**: 3.06 → 3.09
ms on B70 (two weight streams and four accumulator arrays per thread; the two
separate dispatches were already concurrent). Kept as `:expert-dual` in the
table for A/B, not used. The per-op profile of the current layer (pinned
clock, floor-subtracted) then showed the real item: **ssm_out / attn_output
(2048 rows × 4096 cols) on r8 get only 256 workgroups** and had slowed from
0.04 to 0.063 ms when `:wide` moved to r8 in tick 20 — the "wide" class was
two shapes. A fifth class `:out` puts them back on r1: **3.05 → 2.96 ms/token**
(oracle-exact). `anv` = wide r8 / narrow r1 / expert r8 / out r1 / lm r8.
Session tally for B70's 4-layer token: 3.63 (tick 5) → 2.96; 40-layer estimate
≈ 40 × 0.315 + 1.7 ≈ **14 ms, ~70 tok/s**. Small-dispatch consolidation is
closed as a B lever (ticks 19 and 23 both measured it at 0 to −1%).

**Tick 24 (iteration 29): the memory floor, and the row sweep.** `mem_probe.comp`
+ `mem_probe.kotoba` stream 1 GiB with pure `vec4` loads (4096 workgroups × 256
threads × 64 vec4, ten dispatches in one buffer): **B70 598 GB/s** (pinned),
**K16 46 GB/s**, **Xavier 103 GB/s**. Against that floor the K-quant kdots sit
at 28–44% on B70 (qkv 170, lm_head 261), 70–95% on K16 (bandwidth-bound, as
tick 3 said) and ~20% on Xavier (instruction-bound, as tick 6 said). The token
floor on B70 is 1.3 GB / 598 GB/s ≈ 2.2 ms — vLLM's 9.7 ms is far from it too.
`kdot_f32_r8.comp` takes `-DROWS`; the sweep on B70 (pinned, GB/s): attn_qkv
r1 120 / **r4 172** / r8 168 / r16 128 / r32 69; lm_head r1 152 / r4 256 /
**r8 261** / r16 131 / r32 132 — rows in flight saturate at 4–8 and spill past
16, so the remaining 2.3× to the floor on ANV is the per-value dequant
arithmetic, the same wall Xavier hit at a lower height. `gen_kdot_guest.cljk`
accepts `r1 | r4 | r8 | r16 | r32`. Next kernel: an r8-shaped packed-half kdot
for ANV (h2's arithmetic, r8's rows in flight), with the 40-layer parity
measurement before it can become the `anv` default.

**Tick 25 (iteration 30, B-7): packed half on ANV — refuted; and the lever
ANV does have.** `kdot_h2_r8.comp` (r8's eight rows in flight, h2's exponent-
trick halves and packed multiplies, shared half codebook for IQ4_XS). B70,
pinned, vs the f32 r8: attn_qkv Q5_K 162 → **103** GB/s, attn_gate IQ4_XS 90 →
111, single expert 13 → 16, lm_head Q6_K 261 → **191**. Packed half is not
cheaper on Xe2 in this shape — the byte composition of the codes costs what the
half multiplies save, and Q5_K / Q6_K lose outright. The `anv` row stays f32;
the kernel is kept as a measured negative. The floor-to-kernel gap on ANV is
therefore not "arithmetic that halves in fp16". What Xe2 *does* expose, and the
Xavier driver does not: **`VK_KHR_shader_integer_dot_product` with
`integerDotProduct4x8BitPackedSignedAccelerated` (B70 and K16 both)** — the
`dp4a` path llama.cpp's CUDA kernels live on: quantize the activation to int8
per 32-block once per token, then one packed 4×8-bit dot per four values with no
int→float conversions (the K-quant `codes` words are already byte-per-code, so
the dot consumes them directly; mins and the Q6_K −32 fold into a Σx term).
That is the next kernel, and its numerics are llama.cpp's own.

**Tick 26 (iteration 31, B-8): the dp4a kdot, measured.** `nex_ops.comp`
`quant_q8` (per 32-value block: scale = max|x|/127, int8 codes four to a word;
thread = block) and `kdot_i8_r8.comp` (r8 shape; `dotPacked4x8EXT` on the
byte-per-code K-quant words against the int8 activation word; the K-quant mins
and Q6_K's −32 use one more dot with `0x01010101`; IQ4_XS packs the four
codebook values into an int; six bindings). The system glslang (15.1) does not
know `GL_EXT_integer_dot_product`; glslang 16.6 from the Khronos release does
(`/opt/glslang-16.6`, `glslang16` on B70). `gen_kdot_guest.cljk … i8r8` runs the
quantizer then the kdot. B70, pinned:

| tensor | f32 r8 | **i8 r8** | i8 error vs f64 |
|---|---|---|---|
| attn_qkv Q5_K | 168 GB/s | **193** | rel-RMS 6.1e-3 (max-rel 0.20 on the 1e-2 floor) |
| attn_gate IQ4_XS | 90 | 99 | |
| ffn_gate_exps[0] | 13 | 14 | |
| output Q6_K | 261 | **289** | rel-RMS 4.5e-3 |

Correct at Q8-activation precision (the error is the quantization of a
random-normal activation, llama.cpp's own regime) and **+11–15%** — not the 2×
the floor suggested, so on ANV the K-quant kdot's remaining cost is not the
multiply-accumulate either: what stays is the code extraction (shifts, masks)
and the 4-byte-granular loads (the 598 GB/s floor was measured with 16-byte
`vec4` loads; the block layout gives each thread 4-byte words). The lever that
is left is layout: 16-byte loads per thread (`uvec4`, 16 values per thread per
block) with rows in flight — an "r8 × x16" kernel. Wiring `i8r8` into the token
loop needs a `quant_q8` before each of a layer's six kdot inputs; that is the
generator work of the next tick if the layout idea does not beat it.

**Tick 27 (iteration 32, B-9): three more hypotheses for ANV's kdot ceiling,
all measured, none it.** (1) *Load width*: `mem_probe_u32` streams the same
1 GiB with 4-byte loads — **593 GB/s**, same as `vec4` (598); granularity is not
the cause. (2) *Occupancy*: `kdot_f32_r8k4.comp`, the r8 kernel with the block
loop split across four thread groups (256 threads per workgroup, same rows in
flight): attn_qkv 162 → 162, attn_gate 91 → 111, lm_head 261 → 236. (3) *Load
instruction count*: `kdot_s4_r8.comp`, block headers staged through shared
memory once per block for all eight rows: attn_qkv 162 → **132**, lm_head 261 →
196 — the added shared memory and per-block barriers cost more than the header
loads they remove (the lm_head takes no staged path and still lost, so the
shared allocation itself reduced occupancy). What is left, and consistent with
every number so far: the *coalescing pattern inside a block*. In the r8 layout
a 64-thread load instruction for one row touches 128 B of `qs` with 50%
redundancy (threads t and t+8 read the same word for the two nibble halves)
plus 32 B of `qh`, while the probe's instructions read 256 contiguous bytes;
`kdot_x8` (eight consecutive values per thread, no redundancy) was the best
r1-shaped kernel on B70 (143 vs 121) but is one row per workgroup. The next
kernel is x8's thread mapping with r8's rows in flight ("x8 × r8"). Both
negatives are kept for A/B.

**Tick 28 (iteration 33, B-10): x8 × rows.** `kdot_x8r8.comp` (`-DROWS`): 256
threads = 8 block slots × 32 lanes, a lane owns eight consecutive values, R rows
per workgroup share the activation's two `vec4`. B70, pinned, GB/s:

| tensor | f32 r8 | x8 × r8 | **x8 × r4** |
|---|---|---|---|
| attn_qkv Q5_K | 163 | 159 | **199** |
| attn_gate IQ4_XS | 90 | 104 | |
| ffn_gate_exps[0] (1 expert) | 14 | **29** | |
| output Q6_K | 261 | 173 | **272** |

Composed 4 L × 3 T: wide + lm on x8r4 **2.97 → 2.82 ms/token** (oracle-exact);
experts on x8r8 lose in the 8-position shape (3.27), so `anv` = wide x8r4 /
narrow r1 / expert r8 / out r1 / lm x8r4. Rows-per-workgroup is now read from
the kernel's name (`…r8.spv` → 8, `…r4.spv` → 4, else 1). Coalescing inside the
block was a real factor (qkv +22%), not the whole gap: 199 of 598. Session
tally, B70 4-layer token: 3.63 → 2.82 (−22%); 40-layer estimate ≈ 40 × 0.28 +
1.7 ≈ **13 ms, ~75 tok/s**.

What this is not yet: the 40-layer model on a device with room (the serving
processes own the memory), prompt-side prefill (tokens are fed one at a time),
sampling other than argmax, distribution parity with llama.cpp over a real
prompt, and the per-backend kernel layouts (Xavier).