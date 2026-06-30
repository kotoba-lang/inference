# kotoba-lang/inference

CLJC-first inference runtime contracts and verification for Kotoba/Kotodama.

This repository owns the portable inference layer:

- `kotodama.inference.*` CLJC runtime specs and host ports
- `torch` model graph contracts
- `num` tensor compute contracts for CPU, WASM, WGSL, and WebGPU hosts
- Rust `kotoba-kotodama-inference` WebGPU/native runtime experiments
- real-model verification for local Gemma GGUF artifacts

The core foundation is `kotoba-lang/torch` plus `kotoba-lang/num`. JavaScript
model runtimes such as Transformers.js, ONNX, and ONNX Runtime Web are not
portable foundations for this repo.

## Layout

```text
cljc/src/    portable runtime contracts
cljc/test/   CLJC contract tests
verify/      maturity gates and local-model verification
src/         Rust inference runtime
browser/     browser worker/client surface
shaders/     WGSL kernels
tests/       Rust integration tests
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

Browser wasm package:

```sh
wasm-pack build --target web --no-default-features --out-dir pkg-web
```
