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

## Stable JVM production entry point (`kotodama.inference.host.jvm`)

Downstream JVM hosts (e.g. `kotoba-lang/murakumo-studio`) should call the
promoted host adapter, **not** the `verify/` smoke code:

```clojure
(require '[kotodama.inference.host.jvm :as gemma])

;; one-shot (loads + closes the GGUF around the call):
(gemma/generate
  {:kotodama/model        "gemma4:e4b"   ; local Ollama tag, resolved via `ollama show`
   ;; :kotodama/model-path "/path/to.gguf" ; or a resolved GGUF path instead of :model
   :kotodama/prompt       "The capital of France is"
   :kotodama/max-tokens   16
   :kotodama/cache-weights? false        ; true keeps dequant weights in RAM (~20GB, faster)
   :kotodama/use-kv-cache?  true         ; default; pure speed optimisation
   :kotodama/on-token     (fn [id text nanos] ...)}) ; optional streaming callback
;; => {:kotodama/text "..." :kotodama/generated-token-ids [...]
;;     :kotodama/stop-reason :eos|:max-tokens :gemma4/block-count 42 ...}

;; reuse a session across calls (load once, generate many):
(let [s (gemma/load-model {:kotodama/model "gemma4:e4b" :kotodama/cache-weights? true})]
  (gemma/generate {:kotodama/session s :kotodama/prompt "..." :kotodama/max-tokens 16})
  (gemma/close-model s))
```

`host.jvm` implements the real Gemma4 **per-layer-embedding (PLE)** architecture
(the `inp_gate` / `proj` / `post_norm` / `per_layer_*` tensors the older
`stable-block-forward` omits), plus QK/V norms, rotate-half NEOX RoPE (partial
rotary on global layers), the gated GELU MLP, KV-sharing, and a KV-cache. The
`.cljc` kernel stays pure (no file IO); all GGUF reading lives in this
`.clj`-only namespace, per `kotodama.inference.ports`.

Verify:

```sh
CACHE_WEIGHTS=1 clojure -M:verify-gemma-ple-generate   # end-to-end via host.jvm/generate
CACHE_WEIGHTS=1 clojure -M:verify-gemma-kvcache         # KV-cache == from-scratch, + tok/s
```

> Status: the decode loop, tokenizer (SentencePiece `add_dummy_prefix`), GGUF
> weight decode and output projection are verified correct (a 0-layer forward
> greedily predicts the input token); the KV-cache is verified to be output-
> identical to a from-scratch run. The full 42-layer PLE forward is still
> **numerically different from Ollama**. Reference-correct zero-based RoPE,
> attention scale `1.0`, and checkpoint layer-output scales removed the former
> repeated multilingual garbage, but the fixed one-token probe currently gives
> `" hopefully"` where Ollama raw completion gives `" Paris"`. This is improved
> but not a quality pass.

The weight-cached verifier has a 24 GiB JVM heap contract because the current
reference implementation materializes roughly 20 GiB of dequantized weights:

```sh
CACHE_WEIGHTS=1 KOTODAMA_VERIFY_MAX_TOKENS=1 clojure -M:verify-gemma-ple-generate
```

For numerical parity work, emit compact activation fingerprints for every
attention, MLP, PLE, and output stage without retaining full activations:

```sh
CACHE_WEIGHTS=1 \
KOTODAMA_VERIFY_MAX_TOKENS=1 \
KOTODAMA_TRACE_EDN=/tmp/gemma4-trace.edn \
clojure -M:verify-gemma-ple-generate
```

`KOTODAMA_FLOAT32=1` additionally uses explicit float32 projection
accumulation for comparison with ggml. It is a diagnostic bridge; exact Ollama
parity still requires native Q4_K/Q6_K block-dot reduction order.

## Local MLX host adapter (`kotodama.inference.mlx`, Apple Silicon)

A thin `IModelRuntime` host adapter (same shape as `kotodama.inference.ollama`)
for a local, already-running OpenAI-compatible MLX server — either Apple's
own `mlx_lm.server` or mu-hashmi/mlx-moe's `serve` (single-node MoE, SSD-paged
experts; the fleet planner that launches it lives in `kotoba-lang/murakumo`'s
`murakumo.infer.moe`). Both speak the same `/v1/chat/completions` wire shape,
so one client covers both — `:backend` only labels which process convention
the caller launched:

```clojure
(require '[kotodama.inference.mlx :as mlx]
         '[kotodama.inference.ports :as ports])

(def rt (mlx/mlx-runtime {:base-url "http://127.0.0.1:8080"
                          :model "mlx-community/Qwen3.6-35B-A3B-4bit"
                          :backend :mlx-lm}))    ; or :mlx-moe
(def session (ports/load! rt {}))
(ports/generate! rt session "..." {:kotodama/max-new-tokens 64 :kotodama/temperature 0.2})
```

`generate!` is the only method that talks over HTTP; `forward!` throws (no
raw-tensor endpoint on either server), matching the Ollama adapter's own
limitation. See `cljc/test/kotodama/inference/mlx_test.cljc` for the pure
(no-I/O) coverage — the HTTP path is verified by hand against a running
`mlx_lm.server`/`mlx-moe serve`, the same standard the Ollama adapter holds
itself to.

`kotoba-lang/num` and `kotoba-lang/torch` are sibling local dependencies. A
standalone checkout should place them next to this repository, or use a monorepo
layout that preserves `../num` and `../torch`.

## Native Runtime Status

The former Rust WebGPU/native runtime experiment, Cargo metadata, and Rust
integration tests have been removed from this repository. Runtime backends should
live in adapter repositories and consume the CLJC contracts here.
