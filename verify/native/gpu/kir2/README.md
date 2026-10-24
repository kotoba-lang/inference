# GPU kernels from Amu: the KIR v2 artifact table (JUN-375)

Amu ADR 0374 (option D, accepted 2026-10-24) splits the work:
* **Amu's side** ends at verified GPU artifacts: the SPIR-V modules that Amu
  selfhost writes with KIR v2. They are checked by the Kotoba checker and
  `spirv-val`, and pinned by digest.
* **Inference's side** is to consume those modules in place of
  `kernel-dsl/emit-kernel.cljk` and glslang. It keeps choosing kernels by
  name in the layout table (`gen_decode_tokens_guest.cljk`, unchanged).

ADR 0371 Decision 9 makes shaders data: each is identified by SHA-256 in an
artifact table of path, digest and kind. This directory holds inference's
table, the gate that checks it, and the fetch that stages the modules.

| file | what |
|---|---|
| `artifacts.edn` | The table. Each entry gives the name the guests load (`:spv`), Amu's module (`:amu`), its byte count and SHA-256 (equal to Amu's receipt, `:amu/receipt` at `:amu/commit`, whose own digest is `:amu/receipt-sha256`), and the glslang build it replaces. |
| `kir2_artifacts.kotoba` | The native gate, built by Amu's pinned root. It checks the receipt against its pin, every module's pin against the receipt, and every module's bytes against its pin. Then it stages each module under its `:spv` name and reads it back. It computes SHA-256 itself, with wire 3 as its oracle. Its answer is the number of failures. |
| `fetch.sh` | Runs Amu's phases (refs, build, oracle), then the gate and three negative controls. |

## Fetch and stage

```
verify/native/gpu/kir2/fetch.sh <amu-clone> <new-work-dir> <empty-stage-dir>
```

This needs podman (or `CONTAINER=docker`), python3 for Amu's harness, and
glslangValidator and spirv-val on the host for Amu's two oracle phases. Most
of the 30 s it takes is Amu's build. In order:

1. **refs.** `git archive` of `:amu/commit`, then Amu's `refs` phase: glslang's
   builds of the same kernels, which Amu's build compares against.
2. **build.** Amu's `build --suite kir2`, in an arm64 Linux container, with
   `seed/bootstrap/aarch64-linux-static-root.elf`. It writes all 105 modules.
3. **oracle.** Amu's `oracle` phase: `spirv-val` and `spirv-dis` on every
   module.
4. **gate.** The same root compiles the gate. The gate runs with
   `KEXE_ALLOW=3,35,37,38,39`, scoped to the inference tree, Amu's tree, the
   build output and the stage directory, all read-only except the stage. Three
   negative controls must each fail by name, with exactly one FAIL line:
   * byte 100 of the first artifact's module changed (`FAIL sha256 …`);
   * the first hex digit of its pin changed (`FAIL receipt …`);
   * the first hex digit of `:amu/receipt-sha256` changed (`FAIL receipt sha256 …`).

glslang and spirv-val run only inside Amu's phases, as Amu's oracles. What
the gate stages are Amu's KIR v2 bytes. To use them on a box, copy the stage
directory's files into the guests' `KDOT_SPV_DIR`. The names are the ones the
layout table already loads.

## What the eleven modules replace

For each entry, building inference's own source with glslang (the
`:replaces` column, with Amu's flags `-V --target-env vulkan1.1`) gives a
module byte-identical to Amu's oracle reference of that kernel. Amu's provider
runs on gad (RADV) and the B70 (ANV) showed each KIR v2 module computing the
same output bytes as its reference (amu `docs/evidence/gpu-kernel-kir2-h2-20261024/`,
`gpu-kernel-exec-20261024/`). Those runs are historical receipts of a route
amu main no longer builds. The record for this table is
`verify/evidence/kir2-artifacts-20261024.json`.

## Not covered yet (JUN-375's remaining slices)

* **Running them.** No route can execute a module yet. The numeric gates (f32
  rel ≤1e-3, h2 rms ≤1e-2, the argmax pick equal to f64's) and the speed gate
  (≤1.25× glslang's) wait for Amu's executor routes: a Kotoba loader for the
  system Vulkan stack, a native kernel driver, or browser WebGPU.
* **The other kernels Amu builds**: the nex ops and their batch twins,
  deltanet, embed, `attn_copy_dense`, `dot_rows` and the rope-bias variant.
  They are the next entries for this table.
* **Kernels Amu has no module for**, which still come from glslang: the
  prefill `kdot_f32_p<P>[r<R>]`, `kdot_s*` (`nvgpu-exact`, Xavier's dense
  path), `kdot_f32_r8_dual`, `kdot_ptq1_dual_r1`, `hadamard_signed`,
  `kdot_h2_swiglu` and `kdot_h2_swres`.
* **`emit-kernel.cljk`** (its MSL twins for Metal included), and the CLJK
  generators of the guests.
* **Xavier** (aarch64, nvgpu) and **macOS** (Metal).
