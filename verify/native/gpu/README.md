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

What this is not yet: the composed decode step. The matvec kernels are ported;
delta-net, `nex_ops` (norms, rope, softmax-topk, gate) and the MoE gather are the
remaining WGSL → GLSL ports, then the 40-layer decode step as a `.kotoba` program
issuing ~20 dispatches per token (the intended end is amu's accelerator KIR
emitting SPIR-V). Xavier
runs this path instead of CUDA (JetPack 5.1.2 cannot run a vLLM that knows
Nex; owner 2026-09-18).
