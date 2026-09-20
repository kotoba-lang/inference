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

**Tick 29 (iteration 34, B-11): dp4a on x8r4, wired into the token loop, and
the parity that makes it the ANV default.** `kdot_i8_x8r4.comp` (two
`dotPacked4x8EXT` per lane per block on the x8 mapping, 4 rows): B70 pinned
attn_qkv **221 GB/s** (f32 x8r4 199, session start r8 163), attn_gate 127 (90),
lm_head **321** (272 / 261) — the coalescing and the integer-dot gains stack.
The generator quantizes an activation once per layer (`quant_q8` into a paired
`[xq xscale]` buffer, rows `:wide-i8` / `:lm-i8` / `:expert-i8` name the int8
kernels). Composed 4 L × 3 T: **2.83 → 2.55 ms/token (−10%)**; int8 experts add
nothing (2.58) and stay f32 r8. 40-layer streaming parity on B70, same prompt:
5/5 argmax = oracle, **KL(oracle‖gpu) 3.4e-3 nats, KL(llama.cpp‖gpu) 0.0054** —
closer to llama.cpp than the exact kernels' 0.0143, as it should be (the same
Q8-activation regime); p(Paris) 0.535 (llama 0.474, exact 0.549). The hidden
state's `x rel` reaches 0.42 at the last step while the distribution moves
3.4e-3 nats — the same shape as h2 on Xavier, and the same rule: decide on the
distribution. `anv` = int8 wide / lm, `anv-exact` keeps f32. Session tally, B70
4-layer token: 3.63 → **2.55 (−30%)**; 40-layer estimate ≈ 40 × 0.21 + 1.7 ≈
**10 ms, ~100 tok/s class** — vLLM's 103 is within reach of the estimate; a
resident run is what turns it into a measurement.

**Tick 30 (iteration 35): the same int8 path on K16 — measured, not adopted.**
RADV also exposes the 4×8 integer dot; glslang 16.6 installed on K16 and the
same kernels compiled. K16 (GB/s): attn_qkv f32 r8 25.7 / r1 22.3 / **i8 x8r4
21.9** / i8 r8 20.0; lm_head f32 r1 43.1 / r8 38.2 / **i8 x8r4 44.2** / i8 r8
38.4; attn_gate i8 x8r4 17.4. Against a 46 GB/s floor the K16 kdots were already
at 55–95%, and the int8 path buys nothing (qkv loses, lm_head +2%) while
adding the Q8 numerics — so `radv` stays f32 (wide r8 / narrow r1 / lm r1).
Per box, the kdot story is now: K16 bandwidth-bound (nothing left in the
kernel), B70 issue/coalescing-bound (int8 × x8r4 landed), Xavier
instruction-bound without an integer dot (h2 landed, dp4a unavailable).

**Tick 31 (iteration 36, C-1 groundwork): the kernels are prefill-ready.** A
real prefill runs the prompt's P tokens through each layer as *one* batch where
the math allows and P sequential steps where it does not: the kdots take
`positions = P` with `input_per_expert = 1` (the activation is `[P × n]`, the
weights are read **once** instead of P times — the whole point); rmsnorm /
rope / softmax / silu are row-wise already; the delta-net step and causal
attention are recurrences and run P dispatches each, one per token. Two kernel
changes make that possible and change nothing for decode: `deltanet_fused`
reads its **row** from `pos[1]` (which token of the batch; 0 in decode) and
offsets `qkv / alpha / betaL / z / out` by it; every kdot's `in_row` is
`position / pad0` when `Meta.pad0 > 1` (the MoE prefill has 8 expert positions
per token). Decode re-verified on all three boxes after the rebuild (B70 2.58
ms, K16 27.3, Xavier 36.6). Dispatch arithmetic for a P-token prompt on 40
layers: decode-as-prefill = P × ~21 dispatches and P weight passes; batched =
~21 + 2P dispatches (the two recurrences) and one weight pass — for P = 16 the
weight traffic per prompt drops 16×, the dispatch count 5×. The generator's
`prefill` mode (batched layer bodies, a `stepbuf` for the recurrences, MoE ids
`[P × 8]`, lm_head on the last row) is the next tick.

**Tick 32 (iteration 37, C-1): batched prefill, recurrent layers.**
`gen_decode_tokens_guest.cljk … <backend> prefill` (the prompt is the batch,
`tokens` = 1): every buffer that holds an activation is `P` rows; the kdots run
with `positions = P`, `input_per_expert = 1` and a zero `expert_ids` of P
entries (the MoE experts with `8P` positions and `pad0 = 8`); `rmsnorm`,
`add_rmsnorm`, `f32_matvec`, `softmax_topk`, `weighted_sum` and the embedding
gained a row dimension (workgroup x or y = row; decode dispatches shape 1 and is
unchanged); the delta-net step runs P dispatches on a per-layer `stepbuf`
advanced by `pos_incr2` (absolute position and batch row together); the final
norm runs on all rows and the lm_head reads the last one (`pad1`). The answer
is `ns | x of all P rows | argmax`; `prefill_check.py` compares every row with
the oracle's per-step hidden states (same math, so they must agree). K16,
3 recurrent layers, the 8-token prompt "The capital of France is. The capital":

| | dispatches | time | vs oracle |
|---|---|---|---|
| decode, 8 steps | 8 × (21 + tail) | 118.6 ms (14.1–16.1 per token) | 8/8 argmax, `x rel` ≤ 5e-6 |
| **prefill, one command buffer** | 113 | **40.2 ms** | **8/8 rows `x rel` ≤ 5e-6**, argmax 261 = oracle |

2.9× on the whole; the lm_head (~10 ms on K16) is paid once either way, so the
layers went from ~37 ms to ~30 — the batched kdots read the weights once but
the delta-net's P sequential dispatches remain the prefill's cost on a
recurrent layer. Attention layers are not batched yet (the generator refuses
them in prefill mode; `layers ≤ 3` on this model), so this is the recurrent
half of C-1. Decode paths re-verified on K16 (3 L, 8/8) after the kernel
changes; B70 and Xavier kernels rebuilt in tick 31's pass.

**Tick 33 (iteration 38, C-1 second half): attention layers batched.** In
prefill the attention layer is fully parallel over the prompt: q / k / v kdots
with `positions = P`, the head-wise rmsnorm over `16P` / `2P` rows, `rope_neox`
takes the token from the head row (`h / rows`, position `base + token`), the
KV copies are one contiguous `512P` write, and `attn_decode` runs `(16, P)`
workgroups — query row `y`, causal `T = base + 1 + y`, since all P keys are in
the cache before the dispatch. Unlike the delta-net there is no sequential
step. K16, 12 layers (9 recurrent + 3 attention), the same 8-token prompt:
decode 8 steps **217.3 ms** (26.9–27.6 per token), prefill **122.5 ms** (1.8×),
**8/8 rows `x rel` ≤ 1.3e-5**, argmax 240548 = oracle. The prefill's cost is
now the recurrent layers' P delta-net steps (9 layers × 8 dispatches); the
attention layers are one pass. Decode re-verified on all three boxes after the
kernel change (B70 2.58 ms, Xavier 36.8, K16 chain equal).

**Tick 34 (iteration 39): where the prefill's time went, and the kdot that
reads weights once.** `profile_guest.py` on the 12-layer prefill (407
dispatches): the kdots were **105 of 122 ms**, the delta-net steps 12 — because
the `(rows, positions)` dispatch gives every position its own workgroup and
re-reads the row, so a P-token prefill read the weights P times exactly like
decode (the 1.8× had come from amortised launches and the single lm_head).
`kdot_f32_p.comp` puts the positions in the *inner loop*: one workgroup per
row, the block dequantized once, multiplied against every position's
activation (P ≤ 16), reduction per position. It serves every kdot whose
weights do not depend on the position (all but the MoE experts). K16, 12
layers, 8 tokens: prefill **122.5 → 84.5 ms** (decode 8 steps 217.3), all 8
rows `x rel` ≤ 1.3e-5. Left in the prefill: the MoE expert kdots (their
weights differ per position; sharing needs grouping by expert), the delta-net's
P steps (~12 ms), the lm_head (~10 ms).

**Tick 35 (iteration 40, C-2): the tokenizer.** `gguf_tokenizer.cljk` (kbb) reads
`tokenizer.ggml.tokens` / `merges` / `pre` from the GGUF header alone and
implements byte-level BPE — the qwen2/gpt2 pre-tokenizer regex (`\p{L}` /
`\p{N}` classes, the contraction and whitespace alternatives), GPT-2's
byte→unicode map, greedy lowest-rank merges — and the inverse for decode.
`tokenizer_check.cljk` runs a corpus through it and through llama-server's
`/tokenize` (`add_special false`): **12/12 lines equal** on
`tokenizer_corpus.txt` (English, Japanese, Korean/Russian, German with
apostrophes, Python, prices and dates, emoji, contractions, leading/trailing
and multiple spaces, a URL), and decode round-trips the text. This is the
serving shell's text ↔ ids; the guest keeps taking ids. Not covered yet:
special tokens (`<|im_start|>` etc. must be spliced by the chat-template layer,
not the BPE), and `add_special` (Qwen adds no BOS).

**Tick 36 (iteration 41, C-3-1): the prompt at run time.** One guest binary
now serves any prompt: `… <backend> fn - runtime` emits an exported
`serve [plen ntok]` beside `main` (the frontend requires `main` to take no
arguments; the loader passes i64 argv to an exported entry —
`extract-native --symbol serve`, arity 2). `serve` reads the prompt as a
decimal list from `:env/read NEX_PROMPT` and writes it to the token buffer
with the loader's new `WRITEDEC` (amu #1034; the guest formats decimals but
has no hex, no `string->i64`, no `string-length` on the native slice — those
three refusals shaped the design), rewrites `argmax_final`'s prompt-length
meta the same way, and runs the token loop for `plen + ntok` steps; the token
buffer holds 4096 slots and the KV caches a 4096-token context. Policy gains
`[:cap/call 33]`. K16, 12 layers, one binary: `NEX_PROMPT=9707,198,220 … 3 1`
→ 163967 / 112516 / 169222 / 169484 (4/4 = oracle); `NEX_PROMPT=760,6511,314,
9338,369 … 5 3` → 8/8 = oracle, 26.6–27.8 ms/token. The first attempt sized
the KV cache from the build-time token count and the second prompt's step 4
read past it — argmax happened to match while `x rel` was 1.2; the row check
caught it.

What this is not yet: the 40-layer model on a device with room (the serving
processes own the memory), prompt-side prefill (tokens are fed one at a time),
sampling other than argmax, distribution parity with llama.cpp over a real
prompt, and the per-backend kernel layouts (Xavier).
## Tick 37 (2026-09-20, C-3-2): the HTTP shell — `serve_http.cljk`

The serving surface is a kbb script (`node /opt/kbb-engine/cli.js serve_http.cljk ...` on the box; the
kbb engine is `org-babashka-nbb` rsynced to `/opt/kbb-engine`, no kbb launcher there): `POST /v1/completions`
→ `tokenizer_core.cljk` (encode, loaded once with `nbb.core/load-file`; the CLI `gguf_tokenizer.cljk` now
loads the same core) → `kexe-loader-gpu` with `NEX_PROMPT=<ids>` + argv `<plen> <max_tokens>` (the runtime
guest of tick 36, WRITEDEC, amu #1035 merged this tick and the loader promoted on all three boxes,
previous kept as `kexe-loader-gpu.pre1035`) → parse `:result-utf8-hex`, split `#`-joined `ns|x|id` parts,
take the argmax after the last prompt token and each generated one → decode → OpenAI-shaped JSON with
`token_ids`, `usage`, `timing`. One process per request; greedy only.

Two things node 18 (the boxes) would not run: destructuring `def` (kbb/sci wants `(def a ..)` per name)
and the qwen2 pre-tokenizer's `(?i:'s|...)` inline modifier (V8 12.5+) — spelled out case by case in
`tokenizer_core.cljk`; `tokenizer_check.cljk` still 12/12 against llama-server after the change.

K16, `rt12.bin` (12 layers, runtime prompt): `curl -d '{"prompt":"The capital of France is","max_tokens":8}'`
→ ids `128186,116769,166224,2752,2752,132819,176133,4032`, 27.0 ms/token on the 13 steps, HTTP wall
3.8–5.2 s over 3 runs; the bare loader run is 3.95 s (3 runs: 3.92–3.97) of which the 13 token steps are
0.35 s — **the other 3.6 s is process start + MAP of the 12 layers + pipeline creation, paid per request.**
That is the cost the resident form (guest loops over requests, weights mapped once) removes, and it is the
next item. The 12-layer text is not meant to read (a 12-layer prefix is not the model); the check is the id
list against the f64 oracle (`decode_tokens_ref.py nex.gguf 760,6511,314,9338,369 12 13`) — the oracle's steps 4–11 are `128186,116769,166224,2752,2752,132819,176133,4032`: 8/8 exact through
the whole HTTP → tokenizer → kexe → detokenize path (prompt steps 0–3 are forced, so 5 prompt + 8 generated = 13 steps).

## Tick 38 (2026-09-20, C-3-3): the resident guest — weights mapped once, one message per token step

`… <backend> fn - resident` (generator) emits a guest whose `main` runs the constant phase (MAP, ALLOC,
PIPELINE) once and then `serve-loop`: `:io/read "64"` (wire 41, the pipe's atomic write) → one message
`"<token id>,<reset flag>"` → `WRITEDEC` into a 16-byte control buffer → the token step's command buffer,
now opened by the `ctl` op (`nex_ops.comp`: flag ≠ 0 rewinds `pos` to 0, then `tokbuf[pos] = token`) →
`SUBMIT` → `READ` the argmax → `:io/write` `"<ns>|<hex id>\n"` + an empty `:io/write-error` (the loader
buffers standard output 64 KiB and flushes it before any diagnostic write — the first run hung in `read(0)`
with the reply sitting in that buffer) → recur until EOF. Policy `policy_resident.edn` (wires 42 33 41 37 39).
No state is zeroed on a reset: `deltanet_fused.comp` reads the ring and S as zero when `pos[0] == 0`
(`fresh`), `attn_decode` reads `T = pos + 1` entries, `copy_at` rewrites the cache at `pos` — position 0
has no history by definition, so the rewind is the whole reset. `argmax_final`'s prompt-length meta is
4096 in this mode (the shell is the token buffer's only writer).

`serve_http.cljk` is now this guest's shell: spawns it once (`KEXE_STRING_POOL` 1 GiB, `KEXE_PAIRS` 2^26,
`KEXE_WALL_SECONDS` 86400 — the loader's ceilings; larger values are refused at start), serializes
requests, feeds the prompt (flag 1 on the first token) then each argmax back, respawns after
`steps-per-life` (15000) steps or on exit. The pool and pair heap are bump allocators: measured 11.3 KiB
and 3.5 K pairs per 12-layer step (120 steps: 1,360,741 B / 420,849 pairs), so 2^26 pairs is ~19 k steps.

Measured, 12 layers, prompt 760,6511,314,9338,369 + 8 greedy (`resident_drive.py`, 3 requests each box):

| box | GPU ms/token (SUBMIT ns) | wall per 12-step request | first request (MAP + steps) | ids |
|---|---|---|---|---|
| K16 radv | 27.1 | 0.374 s | 1.69 s | oracle 8/8, ×10 identical |
| B70 anv (int8) | 4.97 | 0.112 s | 2.00 s | oracle 8/8 |
| Xavier nvgpu (h2) | 37.2 | 0.669 s | 2.99 s | oracle 8/8 |

HTTP on K16 (`curl … max_tokens 8`, 5 runs): **wall 381–395 ms, was 3.8–5.2 s in tick 37**; the same 8 ids.
`resident_mix.py` alternates the two oracle prompts (5-token/8 and 3-token/2) on one guest: **6/6** equal
to the single-prompt oracle — the rewind leaks nothing across requests of different lengths.

**The number that fell out**: wall per step minus GPU ns = host cost per step, and it is one round trip
per `gpu` request (12 layers = 291 requests per step: 22 per recurrent layer, 27 per attention, 12 in the
loop): K16 4.1 ms (14 µs/request), B70 4.4 ms (15 µs), Xavier 18.6 ms (64 µs). Every earlier "ms/token"
in this README is the SUBMIT ns — GPU time — so the served rate is lower than those tables say: Xavier
40 layers ≈ 942 requests ≈ 60 ms on top of 86.9 ms GPU (≈ 6.8 tok/s served, not 11.5); B70 40 layers
≈ 14 ms on top of the ~10 ms GPU estimate. The broker pipe round trip and the Vulkan recording per
dispatch are not separated yet. Since fn mode made every step's dispatches identical (device-side pos
and token), the whole step is ONE reusable command buffer: record once, replay per step — a loader
request (`BEGIN keep` / `REPLAY`) that turns 291–942 round trips into 1. That is the next item.

## Tick 39 (2026-09-20, B / C-3-4): the step as one kept command buffer — `resident-replay`

The loader (amu #1036, `kexe-loader-gpu6` on the boxes until merged) gains kept command buffers:
`BEGINK <slot>` records a replayable buffer, `SUBMIT` submits it once and keeps it, `REPLAY <slot>`
submits it again, `DROP <slot>` frees it; `FREE` of a buffer a kept buffer binds is refused by name.
`… fn - resident-replay` uses it: `record-step` records the token step once (`BEGINK 0` … `SUBMIT`: one
warm-up step on zero state that the first request's rewind discards), and `serve-loop` is 4 `gpu`
requests per step — `WRITEDEC` ctrl, `REPLAY 0`, `READ` argmax, plus the two io writes — instead of 291.

Measured (12 layers, France prompt + 8, `resident_drive.py`, 3 requests; GPU ns unchanged, ids oracle 8/8
on every box, `resident_mix.py` 6/6 on K16):

| box | GPU ms/token | wall / 12 steps | host per step, tick 38 → now | guest pool + pairs per step |
|---|---|---|---|---|
| K16 radv | 27.1 | 0.374 → **0.340 s** | 4.1 → **1.2 ms** | 11.3 KiB / 3.5 K → 0.42 KiB / 142 |
| B70 anv | 4.97 | 0.112 → **0.064 s** | 4.4 → **0.36 ms** | same |
| Xavier nvgpu | 37.5 | 0.669 → **0.462 s** | 18.6 → **1.0 ms** | same |

HTTP on K16 (`curl … max_tokens 8`, 5 runs): **0.331–0.354 s** (tick 38: 0.381–0.395). The remaining
host cost per step is the 4 round trips + the `READ` (its own copy command buffer + fence) + the pipe;
the guest's life is no longer bounded by its allocators in practice (2^26 pairs / 142 ≈ 470 k steps).
For the served rate this is the correction of tick 38's correction: Xavier 40 layers ≈ 87 ms GPU + ~3 ms
host, B70 40 layers ≈ 10 ms GPU + ~1 ms.

## Tick 40 (2026-09-20, A / C-3-5): the 40-layer model served natively on Xavier — chat included

`… 40 4 9707 /root/kgpu … nvgpu fn - resident-replay` (1123 constant requests + the step; kept command buffer of
~940 dispatches) on Xavier with `kexe-loader-gpu` (= #1036): the France prompt + 8 greedy →
`11751,13,248046,198,248068,271,248069,271` = the 40-layer f64 oracle (`decode_tokens_ref40-12.npz`) 8/8,
**85.5–86.2 ms/token GPU, 1.04 s wall per 12 steps (86.7 ms/step → 11.5 tok/s served)**, first request 13.1 s
(the 18.7 GB MAP). Text: " Paris.<|im_end|>…" — llama-server (K16 :8097) continues " Paris.\n\nThe capital of
France is": after "." it has "\n\n" 0.128 vs `<|im_end|>` 0.120 (its own top_logprobs), a near-tie the f64 oracle
resolves our way. Not a bug to chase; it is what KL 0.014 looks like at an argmax.

The tokenizer now matches control / user-defined tokens (`token_type` 3 and 4: `<|im_start|>` `<|im_end|>`
`<think>` `</think>` …) verbatim before the pre-tokenizer, longest first, like llama.cpp's `parse_special`:
the ChatML prefix tokenizes identically to llama-server's `/tokenize` (13/13 ids); the corpus stays 12/12.
`serve_http.cljk` gained `POST /v1/chat/completions` (the template as `/apply-template` renders it for this
GGUF, checked on two message shapes; `reasoning_content` split at `</think>`) and stops at `<|im_end|>` /
`<|endoftext|>` (`finish_reason "stop"`). Xavier, 40 layers, chat "What is the capital of France? Answer in
one word." → `"Paris"`, stop, 21 prompt + 4 generated, 2.26 s (llama-server: "Paris", 5 tokens). A haiku
request: 200 tokens with reasoning, coherent, 216 steps in 19.0 s = **88.2 ms/step wall, 11.3 tok/s**.

The surface is a systemd unit on Xavier, `murakumo-xavier-nex-native.service` (:8091, node 18 + the kbb
engine at `/opt/kbb-engine`, `steps-per-life` 400000), enabled. It is NOT yet the gateway head (:8090 of the
stopped llama-server unit): the shell has no `stream: true` (SSE) and prompt tokens are still fed one step
each (21 prompt tokens = 1.8 s of the 2.26 s above) — the prefill mode of tick 33 has to enter the resident
protocol first. Those two are the next items; B70's 40 layers still wait on the vLLM 27 GB decision.

## Tick 41 (2026-09-20, A / C-3-6): streaming, and Xavier's native surface takes the gateway's fallback head

`serve_http.cljk` `stream: true`: SSE, one `data:` chunk per generated token as the guest returns it — text
through `TextDecoder {stream: true}` over `token-bytes` (a token can end inside a UTF-8 sequence; 日本の首都は →
「東京」 arrived whole), chat as `delta.reasoning_content` until `</think>` then `delta.content` (the template's
newlines after the think block dropped), a final chunk with `finish_reason` + `usage` + `timing`, `data: [DONE]`.
Also `GET /health`, `/slots` (llama-server's shape, one slot, `is_processing` while a step is in flight),
`/v1/models`; OpenAI content-parts arrays (text joined; an `image_url` part is refused with 400 — no vision on
the native path, the mmproj stays with llama.cpp); a thrown refusal answers 400 instead of ending the server.

**The gateway's Xavier head is now this surface.** cloud-murakumo-api `src/xavier_hosted_model.js` routes
`nex-n2.5-mini-uncensored` to heads in order 6600hs1 (B70 vLLM, 64 slots) → xavier (`192.168.1.28:8090`,
1 slot, taken when the primary is unreachable) → k16 (:8097 llama-server). Xavier's llama-server unit has been
stopped since tick 36 (owner: 「xavier の serving を止めて ok」), so :8090 was dead; `murakumo-xavier-nex-native.service`
now listens there (moved from :8091). Through the gateway the model answered `system_fingerprint: vllm-0.29.0`
both times — the primary is up and takes the traffic, as designed; the native head is the fallback, reachable
and answering `/health` `/slots` `/v1/models` locally. Not exercised via the gateway this tick (that would mean
taking the primary down). Note for the gateway repo: `worker.cljk`'s `probe-xavier` still expects model id
`murakumo-main` / alias `qwen3.8-27b` / 27.3 B params — stale since Xavier moved to Nex, independent of this work.

**Prefill on Xavier, measured before building it into the resident protocol**: a 40-layer `prefill` guest does
not compile (`module string literals exceed UTF-8 byte limit` — prefill mode is still the literal program; it
needs fn mode's functions). 12 layers, the 5-token France prompt, Xavier: **174 ms batched vs 5 × 36.6 =
183 ms decode — 5 % faster**, where K16 got 2.6× (84.5 vs 217 ms for 8 tokens). Xavier is instruction-bound and
`kdot_f32_p` is f32: the batched kdot saves weight reads Xavier was never short of. Also found: Xavier's
`kdot_h2_r1.spv` / `kdot_h2q6_r1.spv` predated the `pad1` row offset (built 23:56, source changed 00:36) — the
prefill's lm_head read row 0 (argmax 163967 = the prompt's first step) until the rebuild; then 128186 = oracle,
rows `x rel` ≤ 2.2e-3 (h2 experts). Decode re-checked after the rebuild (36.6 ms/token, 8/8). Conclusion for
the next item: on Xavier a resident prefill needs an h2 positions-inner-loop kdot before it is worth its
complexity; on B70 / K16 the f32 one already pays.

## Tick 42 (2026-09-20, C-1 / C-3-7): prefill as a fn-mode program — 40 layers compile; measured on three boxes

`… <backend> prefill` is now the FUNCTION program (the literal one is `prefill-flat`): the layer functions run
`PF` rows; a recurrent layer seeds its stepbuf from the position (`pos_seed`: `[pos, 0]`, handle = `S + (SB0 − S0)`
since states and stepbufs both run one per recurrent layer — the 5-parameter ABI has no room for a sixth) and
runs its P delta-net steps; the step ends with `pos_add PF`; the dyn `copy_at` meta covers `512 · PF` (the
first fn-mode run had rows 1–4 at `x rel` 8–12 with row 0 exact — only row 0 of K/V reached the cache).
The embed already read `pos + row`. The 40-layer guest is 66 KB of source (1155 constant requests) and compiles.

12 layers, the 5-token France prompt, rows vs the f64 oracle and the last row's argmax (3 runs each):

| box | prefill (one command buffer) | decode 5 steps | ratio | rows | argmax |
|---|---|---|---|---|---|
| K16 radv | **67.3–68.1 ms** | 5 × 27.1 = 135 ms | **2.0×** | `x rel` ≤ 1.3e-5 | 128186 = oracle |
| B70 anv | **10.3 ms** | 5 × 5.0 = 25 ms | **2.4×** | ≤ 2.7e-5 | 128186 = oracle (after the rebuild below) |
| Xavier nvgpu | 174.4 ms | 5 × 36.6 = 183 ms | 1.05× | ≤ 2.2e-3 | 128186 = oracle |

**40 layers on Xavier**: 545–558 ms for 5 tokens vs 5 × 86 = 430 ms decode — **1.27× slower than decode**;
rows `x rel` ≤ 1.35e-2 (the h2 experts, 40 layers deep), argmax 11751 = the 40-layer oracle. Same conclusion as
tick 41 with the sign now negative: on Xavier the f32 positions-inner-loop kdot loses to the h2 decode kernels
it replaces. The next item is the h2 twin of `kdot_f32_p` (`-DHALF`: f16x2 dequant and activations, HFMA2 on
Volta); until then the Xavier shell must keep feeding prompts one step at a time.

**Stale kernels, second time.** B70's `kdot_x8r4.spv` / `kdot_f32_r1.spv` / `kdot_f32_r8.spv` / `kdot_i8_*` were
built between the two prefill commits (they had `pad0`, not `pad1`), so the prefill's lm_head read row 0
(argmax 163967) while every row was right. Rebuilt all seven from the current sources (glslang 16.6 for the
int8 pair, `--target-env vulkan1.3`); decode re-checked 5.06 ms/token 8/8. The rule that follows: a kernel's
`.spv` on a box is stale whenever its mtime precedes the source's last commit — check that before reading a
mismatch as a bug (twice now: Xavier tick 41, B70 tick 42). `nex_pos_add.spv` / `nex_pos_seed.spv` built on all
three boxes.

## Tick 43 (2026-09-20, C-1 / kernel): the prefill kdot's positions in registers — Xavier 545 → 292 ms

`kdot_f32_p.comp` kept `acc[MAXP]` / `mins[MAXP]` indexed by a runtime-bounded loop; on NVIDIA that is LOCAL
MEMORY (the trap the kernel's own q6 comment names). `-DPFIX=<P>` makes the position count a compile-time
constant (`kdot_f32_p<P>.spv`, P = 4 5 8 16 built on the three boxes; a mismatched `positions` writes a NaN
instead of a wrong number), and the activation is one `vec4` load per position per block instead of four
scalars. The generator's prefill mode names `kdot_f32_p<PF>.spv`.

| | before (tick 42) | constant P | + vec4 loads | vs decode |
|---|---|---|---|---|
| Xavier 12 L, 5 tokens | 174.4 ms | 106.3 | **99.5 ms** | 183 ms → **1.84×** |
| Xavier 40 L, 5 tokens | 545 ms | 317 | **292–308 ms** | 430 ms → **1.4–1.47×** |
| K16 12 L | 67.3 | — | 63.2–65.8 | 135 → 2.1× |
| B70 12 L | 10.3 | — | 9.69 | 25 → 2.6× |

Rows unchanged (Xavier 40 L `x rel` ≤ 1.35e-2, K16 ≤ 1.3e-5, B70 ≤ 2.7e-5), argmax = oracle everywhere. The
NVIDIA compiler was the one that needed the constant; RADV/ANV already kept the arrays in registers (4–6 %).

**`-DHALF` measured and not adopted**: the packed-f16 products (kdot_h2_r1's form) on top of constant P gave
Xavier 12 L 99.5 → 99.5 ms (nothing) and 40 L 317 → 296 ms (7 %) while the rows moved to `x rel` 3.7e-3 (12 L)
and **0.115 (40 L)** — the f16 rounding compounds through 40 layers of P-row activations. The kernel was
never issue-bound the way the decode h2 kernel is; it was spilling. The `HALF` arm stays in the source as the
measured negative (not built).

Xavier's 40-layer prefill is now 1.4× decode, so the prompt half of a chat request (21 tokens = 1.8 s fed
one step each) can drop toward ~1.3 s with batched prefill once it is in the resident protocol — which is the
next item, for all three boxes.

## Tick 44 (2026-09-20, C-1 / C-3-8): prefill in the resident protocol — the prompt as P-token steps

`… fn - resident-replay prefill:<P>`: the guest records TWO kept command buffers — 0 the decode step, 1 a
prefill step of P rows — and its `serve-loop` tells the message kinds apart by the one parse it has,
`WRITEDEC`'s word count: `"t,f"` (2 words) → `REPLAY 0`, `"t0,…,tP-1,f"` (P+1 words) → `REPLAY 1`. `ctl` now
takes `P.n` tokens (`tokbuf[pos+i] = c[i]`, flag at `c[n]`; decode's meta has n = 1, so it is unchanged). To
emit both, the generator's batch-dependent state became a PHASE: `PF`, `prefill?`, the kdot-p pipeline and
the 19 metas that depend on the batch are dynamic vars bound by `with-phase`; every activation buffer is sized
by `PMAX`; the layer bodies are emitted twice (`recurrent-layer-p` / `attention-layer-p` / `layers-p`). The
legacy modes regenerate to the same programs (K16 replay 0.340 s, standalone prefill 61.6 ms, both oracle).
`serve_http.cljk` takes `prefill-bucket` (arg 8) and feeds the prompt as chunks of P then single tokens;
`resident_drive.py` / `resident_mix.py` take the bucket as their last argument.

The bug the refactor exposed: `shlogit` (the shared-expert gate logit, one float per row) was still
`max(16, 4·PF)` bytes with the emission-time PF = 1 — 16 bytes, room for 4 rows — so P = 4 was exact and P = 8
read past the buffer (K16 `40923,56380` for the oracle's `4032,268`; the debug reply showed the tokens and
the position landing correctly, which pointed at the compute, and a handle-by-handle comparison of the
resident and standalone P = 8 programs found the one allocation that differed). Sized by PMAX now; the
grep `alloc!.*\bPF\b` is empty.

Measured (ids = oracle in every row; "steps" = messages):

| box, guest | prompt | chunked | decode only |
|---|---|---|---|
| K16 12 L, P = 4 | France 5 + 8 | 9 steps, 270 ms GPU, **0.281 s** | 12 steps, 326 ms, 0.339 s |
| K16 12 L, P = 4 | 12-token oracle prompt + 2 | 4 steps, 184 ms, **0.189 s** | 13 steps, 355 ms, 0.371 s |
| K16 12 L, P = 8 | 12-token + 2 | 6 steps, 219 ms, **0.225 s** | 13 steps, 355 ms |
| B70 12 L, P = 4 | France 5 + 8 | 9 steps, 49.1 ms, **0.052 s** | 12 steps, 60.7 ms, 0.065 s |
| Xavier 40 L, P = 8 | 12-token 40-L oracle prompt + 2 | 6 steps, 871 ms, **0.877 s** | 13 steps, 1143 ms, 1.157 s |

Mixed prompts with the bucket (K16, P = 4): 6/6. On a garbage prompt (`…,1,2,3,4`) the chunked and
decode-only Xavier runs agree for 6 generated tokens and then diverge — the prefill kdot is f32 where decode
is h2 (rows `x rel` ≤ 1.35e-2 at 40 L), so near-ties fall differently; on the oracle prompt they agree.

**Xavier's unit now runs `rp40-nv.bin` with `prefill-bucket 8`** (:8090): chat "What is the capital of France?
Answer in one word." (21 prompt tokens = 2 chunks + 5) → `"Paris"`, 11 steps, **1.65–1.70 s (tick 40: 2.26 s)**;
the haiku (17 tokens) 163 steps in 15.0 s. Prompt cost per token on Xavier is now ~292/8 ≈ 37 ms in the
chunks against 86 ms per decode step; a longer bucket (16) would take it further for long prompts.

## Tick 45 (2026-09-20, C-3-9): sampling in the resident protocol — temperature / top_p / seed per request

Every `resident-replay` guest now records a THIRD kept command buffer (2): the decode step with the
`nex_sample_*` chain (tick 19) in place of argmax, preceded by a `params` op that copies words 2..4 of the
control buffer into the sampler meta's `eps / scale / p0` (T, top_p, seed). The message is
`"t,f,Tbits,top_p_bits,seed"` — 5 words — so `serve-loop` routes by `WRITEDEC`'s count: 2 → `REPLAY 0`
(greedy), 5 → `REPLAY 2` (sampled), else → `REPLAY 1` (prefill; the generator refuses `prefill:4` and `:1`,
whose P+1 would collide). The draw is `u = hash(seed, position)` as before, so `decode_tokens_ref.py
--sample T top_p seed` is the oracle for any (T, top_p, seed).

K16 12 L, France prompt + 8, T 0.7 / top_p 0.9 / seed 42: the guest answers **80072, 80072, 35222, 35533,
143898, 131822, 194349, 157676 — the f64 oracle's eight picks, 8/8**, across 3 requests; greedy on the same
guest still 8/8; the sampled step costs 335 vs 326 ms per 12 steps (+0.75 ms/token). B70 12 L: greedy 8/8;
sampled 80072, 80072 then diverges (33546 …) — the int8 anv path's logits differ at KL 0.0054 and the draw
falls on the other side of a boundary; expected, the f32 path reproduces the oracle exactly.

`serve_http.cljk`: `temperature` (default 1.0; 0 = greedy), `top_p` (default 1.0), `seed` (default a fresh
random u32, echoed in the answer with `temperature` and `top_p`); the last prompt token and every generated
token ride the 5-word message, so a prefill chunk is taken only while strictly more than P prompt tokens
remain (the reply of the last prompt message is the first pick). Over HTTP on K16: T 0 → the greedy ids;
T 0.7 / 0.9 / seed 42 twice → the oracle's eight, identical; no seed → a different sequence with the seed
reported. `resident_drive.py` takes `T top_p seed` as its last three arguments.

**Xavier's head (:8090) runs the sampling guest** (`rs40-nv.bin`, 40 layers, P = 8): T 0 → `"Paris"` in 1.67 s
as before; the haiku at T 0.7 / top_p 0.9: seed 7 → a finished haiku in 22 tokens (2.9 s), seed 8 → 120
tokens of reasoning about syllables (11.5 s) — sampled steps cost ≈ 90–95 ms against 86 greedy. When the
120-token cap lands inside the think block, the whole text comes back as `content` (there was no `</think>`
to split on) — same as a truncated llama-server answer, noted.

## Tick 46 (2026-09-20, C-4): batching — B independent sequences in one step

`… resident-replay prefill:<P> batch:<B>` records kept command buffer 3, a decode step of B rows where the
rows are SEQUENCES: each has its own absolute position (`rowpos[row]`), token-buffer row (stride 4096), KV
cache slice (`row × 4096 × 512`), delta-net ring and state (`row × their single-sequence size`). Kernel twins
under `-DBATCH` (`nex_<op>_b.spv` for copy_at / rope_neox / attn_decode / argmax_partial / argmax_final /
pos_incr / ctl, `deltanet_fused_b.spv`, `embed_iq4xs_b.spv`): row = workgroup y, position from `pos[row]`,
per-row offsets; the B-row kdots are the prefill phase's (`kdot_f32_p<B>`, MoE experts with `8B` positions),
the lm_head runs `positions = B`, argmax keeps 64 slots per row (the pick at slot 61). The message is
`B tokens, B flags` (2B words: flag = rewind that row), so 2B ∉ {2, 5, P+1}; the reply is every row's 64 slots.
Buffers grow by NSEQ = B (rings, states, caches, token buffer, logits, argmax partials). `batch_drive.py` runs
B prompts with their own budgets; an idle row is fed token 0 with flag 1.

**Correct on all three boxes**: K16 B=2 (France + 8 and Hello + 2) both rows = their single-sequence oracle;
K16 B=4 with the two prompts duplicated → the duplicates produce identical sequences and all four match
(4/4); B70 B=2 2/2; Xavier B=2 2/2 and B=4 4/4 (the two continued Hello rows agree across boxes).

**Throughput, 12 layers, GPU ms per step** (the batch step's kdots are the f32 positions-inner-loop kernel, not
each backend's tuned decode layout — that is what the numbers say):

| box | single decode step | batch step B=1 | B=2 | B=4 | seq-tokens/s: single → B=2 → B=4 |
|---|---|---|---|---|---|
| K16 radv | 27.1 ms | 48.5 | 49.2 | 84.3 | 36.9 → 40.7 → 47.4 (+10 %, +29 %) |
| B70 anv (int8 decode) | 5.0 | 7.83 | 7.84 | — | 200 → 255 (+28 %) |
| Xavier nvgpu (h2 decode) | 36.6 | 77.7 | 77.4 | 135.3 | 27.3 → 25.8 → 29.6 (−5 %, +8 %) |

The batch step at B=1 is 1.6–2.1× a decode step: `kdot_f32_p` (one row per workgroup, f32) is what the batch
phase can use today, and it loses to the r8 / int8 / h2 decode layouts it replaces; B=2 costs the same as B=1
(weights read once), B=4 about 1.7×. So the mechanism is right and the kernel is the lever: a batch-aware kdot
per backend (positions inner loop on the r8 / x8r4 / i8 / h2 layouts) would put B=2 near 1.0× a decode step and
the throughput near 2×. That, and the shell's scheduler (rows as slots), are the next items. Wall time in the
harness includes the warm-up before the first message and is not the served figure.

## Tick 47 (2026-09-20, C-4): where the batch step's time went — the lm_head read B times

Profiling by diffing the decode and batch layer functions kernel-by-kernel (`recurrent-layer` vs
`recurrent-layer-b`): the layers were equivalent per row; the difference was outside them. The batch step ran
the lm_head — 248320 × 2048 Q6_K, **425 MB**, the largest tensor in the step — with the positions DISPATCH
dimension (workgroup y = row), which reads it B times: on K16 (46 GB/s) that is 9.2 ms per extra row. Reading it
once through the positions kernel (`kdot_f32_p<B>`, rows as the inner loop, `:lm-once` in the layout table for
radv and nvgpu):

| box | B | before | after | seq-tokens/s vs single |
|---|---|---|---|---|
| K16 | 2 | 49.2 ms | **36.9 ms** | 54.2 / 36.9 = **1.47×** |
| K16 | 4 | 84.3 | **58.1** | 68.9 / 36.9 = **1.87×** |
| Xavier | 2 | 77.4 | **60.8** | 32.9 / 27.3 = 1.20× |
| Xavier | 4 | 135.3 | **88.9** | 45.0 / 27.3 = **1.65×** |
| B70 | 2 | 7.84 | 8.62 (worse) | kept the positions dispatch: 255 / 200 = 1.28× |

ANV is not bandwidth-bound on the lm_head; its int8 x8r4 read twice beats the f32 kernel read once, so the flag is
per backend. All rows oracle-exact throughout (K16 4/4, Xavier 2/2 and 4/4, B70 2/2).

Also tried: `kdot_f32_p` with `-DROWS=8` (the r8 shape with the positions inner loop, `kdot_f32_p<P>r8.spv`,
`NEX_PROWS` / layout `:prows`): K16 B=2 50.3 vs 49.2 ms for the layer kdots, and 46.0 vs 36.9 when it also
took the lm_head — the r1 shape wins on radv for this kernel; kept in the source as a measured negative, default
`:prows 1`. What remains between the B=1 batch step (36.8) and a decode step (27.1) on K16: the MoE expert kdots
at 8B positions (their weights differ per position, ~3 ms), the delta-net's per-row state traffic (~1.5 ms), and
the rest of the row-doubled ops. Those are per-row costs, so they scale with B and cap the gain — B=4 at 1.87×
is near what this shape gives on K16.

## Tick 48 (2026-09-20, C-4): the shell scheduler — rows as slots

`serve_http.cljk … <prefill-bucket> <batch-rows>`: with `batch-rows` B (a `… batch:<B>` guest) the shell is a
scheduler. B rows are slots; an arriving request waits in `pending` and takes a free row at the next tick with
its reset flag; every tick is ONE 2B-word message that advances every occupied row (an empty row gets `0,1`);
a row leaves on `<|im_end|>` / `<|endoftext|>` (`stop`) or its `max_tokens` (`length`), and its promise
resolves — streaming per row works through the same `on-token`. The batch step is greedy and one token per
row per step, so batch mode uses neither prefill chunks nor sampling (the single-row steps share row 0's state
with the batch step and cannot be interleaved). `load_check.py` fires N concurrent requests.

K16, 12 layers, `rb12lm4.bin` (B = 4), prompts of 5 / 1 tokens + 8 generated, vs the serial shell (`rs12s.bin`,
prefill bucket 8):

| N concurrent | serial shell | batch scheduler (B = 4) | ratio |
|---|---|---|---|
| 1 | 0.376 s, 34.6 seq-tokens/s | 0.661 s, 19.7 | 0.57× |
| 4 | 1.155 s, 38.1 | 0.762 s, **57.7** | **1.51×** |
| 8 | 2.327 s, 37.8 | 1.276 s, **69.0** | **1.83×** |

Xavier, 40 layers, `rb40-nv.bin` (B = 4) on :8090 with the head unit stopped for the measurement (the primary
vLLM head was up; the unit was restarted and answered `"Paris"` in 1.69 s afterwards): N=1 3.9 seq-tokens/s
(the France request stops at "Paris." after 7 tokens), N=4 **14.8**, N=8 **15.8** — against the serial head's
single stream of 11.5 tokens/s, 1.3–1.4×. The batch guest's step always computes B rows, so a lone request pays
the B-row step and loses the prefill chunks: the serving policy that follows is "serial shell under light load,
batch shell under heavy load", or an adaptive guest with kept buffers for B ∈ {1, 2, 4} — that is the next item.
Ids in batch mode equal the greedy single-row ids on every request checked.

## Tick 49 (2026-09-20, C-4): adaptive batch — one kept step per B, the scheduler pays for the rows it occupies

`… resident-replay - batch:1,2,4`: an ADAPTIVE guest records one batch step per B into kept buffers 0, 1, 2 (the
single-row decode / prefill / sampled steps are left out — they share row 0's memory with the batch rows and
cannot be interleaved), with layer bodies per B (`recurrent-layer-b1` / `-b2` / `-b`) and the same row layout
(`rowpos`, token rows, KV slices, per-row ring / state), so a B=1 step advances row 0 and a B=4 step rows 0–3 of
the same state. `serve_http.cljk … 0 1,2,4`: each tick takes the smallest B that covers the highest occupied
row (admission fills the lowest free row), the message is 2B words, the reply is read for B rows.

K16, 12 layers, `ra12.bin`, `load_check.py` (prompts of 5 / 1 tokens + 8 generated):

| N | serial shell (tick 48) | B=4-only scheduler (tick 48) | **adaptive {1,2,4}** |
|---|---|---|---|
| 1 | 0.376 s, 34.6 seq-tokens/s | 0.661 s, 19.7 | **0.392 s, 33.2** |
| 2 | — | — | **0.436 s, 50.5** |
| 4 | 1.155 s, 38.1 | 0.762 s, 57.7 | **0.742 s, 59.3** |
| 8 | 2.327 s, 37.8 | 1.276 s, 69.0 | 1.25–1.49 s, 59–71 (3 runs) |

The lone request is back to a single-row step (0.96× the serial shell; the 4 % is the missing prefill chunk),
N=2 is 1.5×, N=4–8 1.6–1.9×; ids unchanged. The N=8 spread is the K16 iGPU's clock (three consecutive runs),
not the scheduler. Not measured on Xavier this tick (the head would have to stop again; its B=4 figures are
tick 48's). What batch mode still lacks against the serial head: prefill chunks per row and sampling — both
are per-row variants of steps that exist; the batch step's message would carry them per row.

## Tick 50 (2026-09-20, C-4): per-row sampling in the batch step

The batch step now ends with the `nex_sample_*` chain over B rows (`-DBATCH`: row = workgroup y, per-row T /
top_p / seed from a params buffer, position from `rowpos`, logits / p_i at `row·n`, scratch at `row·256`, the
pick at slot 61 of the row) after the argmax; a row with **T = 0 keeps its argmax pick** (every sampler op
returns early for it), so greedy and sampled rows share one step. The batch message is 5B words — tokens,
flags, then (T, top_p, seed) per row — written by a batch `params` op into the params buffer; 2B-word messages
still route (no sampler words = all greedy). The adaptive guest holds the three steps as before.

K16 12 L, `ra12s.bin` (`batch:1,2,4`), over HTTP through the scheduler: T 0 → the greedy oracle ids; T 0.7 /
top_p 0.9 / seed 42 → `80072, 80072, 35222, 35533, 143898, 131822, 194349, 157676` = the sampled f64 oracle, twice;
**four concurrent requests mixing two greedy and two seeded-42 rows in one step: all four rows exact**
(0.76 s for the four). The sampler adds ~1 ms to a 12-layer step.

Xavier 40 L, `ra40-nv.bin` on :8090 with the head stopped for the run (restored after; "Paris" 1.68 s): greedy
France 7 tokens in 0.81 s (B=1 step ≈ 118 ms — the f32 positions kdot is 1.37× the h2 decode step on Xavier),
N=4 15.7 and N=8 16.3 seq-tokens/s vs the serial head's 11.5 single stream (1.4×); the seeded haiku (T 0.7,
seed 7) came back in 21 tokens / 4.3 s. **The serial guest stays the Xavier head**: on this box the batch
guest's single-request path is 1.4× slower (kernel, not protocol) and lacks prefill chunks; the adaptive guest
is ready for a bandwidth-bound box (K16) where its B=1 step equals a decode step. The two items that remain
for the head switch on Xavier: an h2 batch kdot (positions inner loop on the h2 layout) and per-row prefill.

## Tick 51 (2026-09-20, C-4): the B=1 batch step is the decode step — the kdot form per B

Each batch step's non-MoE kdots now take one of two forms by B (`:batch-layout-upto`, default 1, env
`NEX_BATCH_LAYOUT_UPTO`): up to that B the backend's DECODE kernels with the rows as the positions dispatch
dimension (weights read B times; for B = 1 this is the decode step itself, int8 / h2 / r8 included — the int8 and
dual forms are enabled for the B = 1 step), above it the positions kernel (weights once, rows inner). Measured
12 layers, adaptive guests `batch:1,2,4`, GPU ms/step, rows oracle-exact throughout:

| box | decode step | B=1 positions → **decode form** | B=2 positions / decode form | B=4 positions / decode form |
|---|---|---|---|---|
| K16 radv | 27.1 | 36.8 → **29.1** | **36.7** / 44.4 | **57.8** / 77.4 |
| B70 anv | 5.0 | 7.8 → **5.23** (int8) | **9.02** / 9.55 | **13.3** / 17.3 |
| Xavier nvgpu | 36.6 | 45.6 → **39.7** (h2) | 60.8 / 61.7 | **88.5** / 108.4 |

So the lone request in the adaptive guest costs 1.05–1.08× a decode step (the difference is the sampler chain
and the batch ops), and B ≥ 2 keeps the positions form on every box. Xavier 40 layers (`ra40-nv.bin`, head
swapped for the run and restored): greedy chat 25 steps in 2.27–2.31 s = 91 ms/step (the serial head's 86 ms
decode + ~5 ms sampler; its 1.65 s comes from the prefill chunks the batch step still lacks), N=4 15.9 and
N=8 **16.7 seq-tokens/s** vs 11.5. The serial guest stays the Xavier head until per-row prefill lands — the
one remaining gap. The h2 positions-inner-loop kernel that tick 50 named is not needed: at B = 1 the h2 decode
kernel is the right form, and at B ≥ 2 the f32 positions kernel already wins on Xavier.

## Tick 52 (2026-09-20, C-4): per-row prefill in the batch step — the adaptive guest takes the Xavier head

`… resident-replay prefill:<P> batch:1,2,4`: the adaptive guest records one more kept buffer, a BATCH PREFILL —
P positions of the ONE sequence the message names (`P tokens, row, flag, pad` = P+3 words; the generator refuses
a P whose P+3 collides with a 2B / 5B message). `-DBPREFILL` kernel twins add the sequence's slice offsets to the
prefill kernels: `ctl` writes the P tokens into the row's token buffer at its base, rewinds it on the flag, and
leaves `rowsel = [base, row]`; `embed` reads `tokbuf[row·4096 + base + i]`; `copy_at` / `attn_decode` address the
row's KV slice (`row · 4096 · 512`); `pos_seed` seeds each layer's stepbuf `[base, 0, row]` and `deltanet_fused`
offsets its ring and state by `pos[2]`; `pos_add` advances `rowpos[row]` by P. The prefill's argmax lands in
row 0's slot as before (the shell ignores it unless the prompt ends there — it never does: chunks are taken
only while more than P tokens remain). The scheduler runs a prefill tick for one such row before each batch tick.

Two things had to give for the 40-layer program to compile: the single-row functions are not emitted for
adaptive guests (never called there), and the 733 `MAP` literals no longer each carry the weight file's path
(`(M "off len")` through one helper) — 66.7 → 49.1 KB of the 64 KiB string budget.

K16 12 L (`rap12.bin`, `batch_drive.py --prefill 8`): a 12-token prompt row (one 8-chunk + 4 singles) next to a
3-token row → `4032, 268` and `169222, 169484`, both = oracle, 6 steps / 267 ms vs 13 steps / 476 ms without
chunks; the same with the long prompts in rows 1 and 3 of a 4-row batch → 4/4. Over HTTP through the scheduler
(`… 8 1,2,4`): a 19-token prompt alone in 338 ms (2 chunks + 3 singles + 3 generated), three of them plus a short
one concurrently in 1.04 s, identical ids.

**Xavier's head unit now runs `rap40-nv.bin` (`prefill:8 batch:1,2,4`)** — the previous unit file is kept as
`murakumo-xavier-nex-native.service.serial`. Against the serial head on the same requests: greedy chat 21 prompt
tokens → `"Paris"` in **1.70–1.74 s** (serial 1.65–1.70), the seeded haiku (T 0.7 / top_p 0.9 / seed 7) → the
**same three lines** in 22 tokens / 3.05 s (serial 2.9 s), load N=4 15.8 and N=8 **17.2 seq-tokens/s** (serial
11.5). No feature is lost: greedy, sampling with seed, streaming, chunked prefill, and now batching (1.5× at N=8).

## Tick 53 (2026-09-20, C-4): int8 in the batch step on anv; the pipeline table

The batch steps' int8 forms (anv's `kdot_i8_x8r4` for the wide kdots and the lm_head) now work for B rows: the
B rows' activations are quantized as ONE vector (`quant_q8` over `B·n` elements; the int8 kernels index the
per-32 scale by element, so contiguous rows need nothing else), the int8 lm_head dispatches `positions = B`,
and the int8 / dual forms are enabled for batch steps up to `:batch-layout-upto`. B70, 12 layers, adaptive
guests (`prefill:8 batch:1,2,4`), GPU ms/step, rows oracle-exact:

| `:batch-layout-upto` | B=1 | B=2 | B=4 |
|---|---|---|---|
| 1 (int8 only at B=1) | 5.21 | 8.95 | 13.32 |
| **2** (anv default now) | 5.21 | **8.65** | **13.24** |
| 4 (int8 layout everywhere) | 5.21 | 8.66 | 15.57 |

So anv takes the int8 decode form up to B = 2 and the positions form above; seq-tokens/s 200 (single) → 231
(B=2) → **302 (B=4, 1.51×)**. The 425 MB lm_head read twice at B = 2 in int8 beats the f32 positions kernel
reading it once — the item tick 47 left open closes as a measurement, not a new kernel.

The adaptive anv guest hit the loader's `pipeline table full` (64): 23 ops + 13 batch twins + 5 batch-prefill
twins + the kdot layouts with their int8 forms + a positions kernel per phase + three embed / delta-net variants
+ the quantizer ≈ 70. amu #1039 raises `KGPU_MAX_PIPELINES` to 128; built on the three boxes as `kexe-loader-gpu7`
(promoted to `kexe-loader-gpu` when the PR lands). K16 and Xavier were under the old limit by a few entries.
