# kotoba-lang/inference

CLJC-first inference runtime contracts and verification for Kotoba/Kotodama.

This repository owns the portable inference layer:

- `kotodama.inference.*` CLJC runtime specs and host ports
- `torch` model graph contracts
- `num` tensor compute contracts for CPU, WASM, WGSL, and WebGPU hosts
- explicit adapter declarations for browser, wasm, native, remote, and edge hosts
- real-model verification for local Gemma GGUF artifacts

The core foundation is `kotoba-lang/torch` plus `kotoba-lang/num`. JavaScript
model runtimes such as Transformers.js, ONNX, and ONNX Runtime Web are not
portable foundations for this repo.

## Layout

```text
cljc/src/    portable runtime contracts
cljc/test/   CLJC contract tests
verify/      maturity gates and local-model verification
browser/     browser worker/client surface
shaders/     WGSL kernels
docs/        ADRs
```

## Verify

```sh
clojure -M:test
clojure -M:verify-maturity
clojure -M:verify-torch-num
```

Local model verification expects a local Ollama `gemma4:e4b` GGUF artifact:

```sh
clojure -M:verify-gguf
KOTODAMA_VERIFY_FULL_MLP=1 KOTODAMA_VERIFY_FULL_LAYERS=2 KOTODAMA_VERIFY_FULL_VOCAB=1 clojure -M:verify-gemma-num
```

`gemma4-e4b-num-smoke` proves 2 composed transformer blocks against golden
logits; it does not tokenize, sample, or run the full model. End-to-end
single-request text generation (real tokenizer, every real transformer block,
greedy sampling, no KV-cache — recompute-from-scratch) is:

```sh
clojure -M:verify-gemma-generate
# optional env: KOTODAMA_VERIFY_MODEL, KOTODAMA_VERIFY_GGUF_PATH,
# KOTODAMA_VERIFY_PROMPT, KOTODAMA_VERIFY_MAX_TOKENS
```

This is CPU-bound recompute-from-scratch (no KV-cache — see
`kotodama.inference.decode`'s docstring) against the model's full real layer
count, so it is slow (see the PR that introduced it for observed tokens/sec).

`kotoba-lang/num` and `kotoba-lang/torch` are sibling local dependencies. A
standalone checkout should place them next to this repository, or use a monorepo
layout that preserves `../num` and `../torch`.

## Native Runtime Status

The former Rust WebGPU/native runtime experiment, Cargo metadata, and Rust
integration tests have been removed from this repository. Runtime backends should
live in adapter repositories and consume the CLJC contracts here.
