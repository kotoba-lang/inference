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
| `artifacts.edn` | The table. Each entry gives:<br>• the name the guests load (`:spv`) and Amu's module (`:amu`), with its byte count and SHA-256;<br>• Amu's glslang module of the same kernel (`:ref`);<br>• the receipt's GPU pair of the two (`:gpu`);<br>• the glslang build the module replaces (`:replaces`).<br>The pins come from one passing GPU receipt of Amu's build (`:amu/receipt` at `:amu/commit`, whose own digest is `:amu/receipt-sha256`). |
| `kir2_artifacts.kotoba` | The native gate, built by Amu's pinned root. It checks:<br>• the receipt against its pin, and that its int8 and h2 kernels ran;<br>• each module's pin against the receipt's build entry and against its GPU pair;<br>• the pair's glslang half against `:ref`;<br>• today's `:ref` against the receipt's, by source and define;<br>• each module's bytes against its pin;<br>• the table's own shape.<br>Only then does it stage each module under its `:spv` name and read it back. It computes SHA-256 itself, with wire 3 as its oracle. Its answer is the number of failures. |
| `fetch.sh` | Runs Amu's phases (refs, build, oracle), then inference's glslang oracle and its control, then the gate and its five negative controls. It copies into STAGE only when everything is green. |

## Fetch and stage

```
verify/native/gpu/kir2/fetch.sh <amu-clone> <new-work-dir> <empty-stage-dir>
```

Requirements:
* podman, or `CONTAINER=docker`. The image is pinned by digest.
* python3, for Amu's harness.
* glslangValidator, spirv-val and spirv-dis on the host, for the oracle steps.

It takes about 30 s, mostly Amu's build. In order:

1. **refs.** `git archive` of `:amu/commit`, then Amu's `refs` phase: glslang's
   builds of Amu's copies of the kernels (`tools/gpu_kernel_refs`). Amu's
   build checks those with the Kotoba checker.
2. **build.** Amu's `build --suite kir2`, in an arm64 Linux container, with
   `seed/bootstrap/aarch64-linux-static-root.elf`. It writes all 105 modules.
3. **oracle.** Two checks on the host:
   * Amu's `oracle` phase runs `spirv-val` and `spirv-dis` on every module.
   * Inference's own: building each `:replaces` with glslang (Amu's flags
     `-V --target-env vulkan1.1`) must give exactly Amu's `:ref`. That ties
     each name the guests load to the kernel Amu ran on the GPU. Its control:
     the first artifact with a define, built without it, must differ.
4. **gate.** The same root compiles the gate. The gate runs with
   `KEXE_ALLOW=3,35,37,38,39`. Its file scope is:
   * read-only: the inference tree, Amu's tree and the build output;
   * writable: WORK's `gate` directory.

   Five negative controls must each fail by name, with exactly one FAIL line:
   * byte 100 of the first artifact's module changed (`FAIL sha256 …`). The
     digest the gate reports must equal `sha256sum` of the changed file.
   * the first hex digit of its pin changed (`FAIL receipt …`);
   * the first hex digit of `:amu/receipt-sha256` changed
     (`FAIL receipt sha256 …`);
   * the first artifact pointing at the second's module, bytes and pin
     (`FAIL pair … v2-…`);
   * the first artifact naming the second's glslang reference
     (`FAIL pair … ref-…`).

   Then the gate checks every artifact, and stages them inside WORK only
   when nothing failed.

The staged modules are Amu's KIR v2 bytes; glslang and spirv-val only check
them. To use them on a box, copy STAGE's files into the guests' SPIR-V
directory (`gen_decode_tokens_guest`'s directory argument, or `KDOT_SPV_DIR`
for `gen_kdot_guest`).

## What the eleven modules replace

Ten of the names are in the decode guest's layout table. `kdot_x8r8` is
`gen_kdot_guest`'s `x8r8` layout.

For every entry, three things hold:
* **glslang of `:replaces` is `:ref`.** Inference's own source, built by
  glslang, is byte-identical to Amu's glslang module of that kernel.
  `fetch.sh` checks this on every fetch.
* **Amu ran both halves of the pair.** The receipt, a provider run on gad's
  RADV with its int8 and h2 kernels run, ran `:ref` and `:amu` as one GPU pair
  and found identical output bytes. Today's `:ref` has the receipt's source
  and define. The B70's ANV receipt (`b70-anv.json`) has the same pairs.
* **The table pins the receipt's module.**

Those GPU runs are historical receipts, of a route amu main no longer builds.
The record of this table is `verify/evidence/kir2-artifacts-20261024.json`.

## Not covered yet (JUN-375's remaining slices)

* **Running them.** No route can execute a module yet. The numeric gates (f32
  rel ≤1e-3, h2 rms ≤1e-2, the argmax pick equal to f64's) and the speed gate
  (≤1.25× glslang's) wait for Amu's executor routes: a Kotoba loader for the
  system Vulkan stack, a native kernel driver, or browser WebGPU.
* **The other kernels Amu builds**:
  * the nex ops and their batch twins, deltanet, embed, `attn_copy_dense`,
    `dot_rows` and the rope-bias variant;
  * the kdot row variants `gen_kdot_guest` can name: `kdot-f32-r8-rows4/16/32`,
    `kdot-x8r16/32` and `kdot-i8-x8r8/16/32`.

  They are the next entries for this table.
* **Kernels Amu has no module for**, which still come from glslang: the
  prefill `kdot_f32_p<P>[r<R>]`, `kdot_s*` (`nvgpu-exact`, Xavier's dense
  path), `kdot_f32_r8_dual`, `kdot_ptq1_dual_r1`, `hadamard_signed`,
  `kdot_h2_swiglu` and `kdot_h2_swres`.
* **`emit-kernel.cljk`** (its MSL twins for Metal included), and the CLJK
  generators of the guests.
* **Xavier** (aarch64, nvgpu) and **macOS** (Metal).
