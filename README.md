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
> identical to a from-scratch run. The native Q8_K × Q4_K/Q6_K path now matches
> Ollama on the fixed full-42-layer raw-completion probe: both produce token
> `9079`, `" Paris"`. The former dequantize-then-double-matmul reference path
> produces `" hopefully"`; this proved that ggml's activation quantization and
> reduction order are model semantics for parity, not merely an optimization.

The legacy weight-cached verifier retains its 24 GiB JVM heap contract. The
native path mmaps quantized tensor windows, requires no whole-tensor
dequantization, passed with a 4 GiB heap, and measured 1.94 GiB maximum RSS on
Apple M4:

```sh
CACHE_WEIGHTS=1 KOTODAMA_VERIFY_MAX_TOKENS=1 clojure -M:verify-gemma-ple-generate
scripts/build-native-kdot.sh
clojure -M:verify-native-kdot
clojure -M:verify-native-gemma-parity
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
parity still requires the same quantized reduction order. Set
`KOTODAMA_GGML_K_DOT=1` to use the scalar reference implementation of ggml's
Q8_K activation quantization and Q4_K/Q6_K block-dot directly from GGUF bytes.
This avoids whole-tensor dequantization and is correctness-oriented; native
SIMD/Metal kernels are still required for production throughput. After running
`scripts/build-native-kdot.sh`, set `KOTODAMA_NATIVE_K_DOT=1` to mmap each
quantized tensor and execute the same contract through JDK FFM without copying
or materializing its weights.

Current Apple M4 fixed-probe measurements (one six-token prompt plus one output
token) are 19.4 s for the pthread CPU K-dot path versus 0.566 s wall time for a
warm Ollama/Metal run (`prompt_eval_duration` 0.141 s). Output parity is real;
throughput parity is not. Metal K-dot, persistent GPU buffers, and fused decode
remain required.

The experimental persistent Metal path uses the same GGUF bytes and K-dot
contract through a binary JVM↔Deno worker. Weights are uploaded once and cached
as WebGPU buffers:

```sh
deno run --unstable-webgpu --allow-read verify/metal_kdot.js
clojure -M:verify-metal-kdot-worker
KOTODAMA_METAL_K_DOT=1 KOTODAMA_VERIFY_MAX_TOKENS=2 \
  clojure -M:verify-gemma-ple-generate
```

On Apple M4 it preserves `" Paris."` generation and measures roughly 15–20 ms
for one `[10240,2560]` Q4_K projection. The binary protocol now groups Q/K/V
and MLP gate/up projections that share an activation: each group performs one
Q8_K quantization/upload, command submission, and combined readback while each
K-dot retains official ggml reduction order. This reduced the fixed probe from
about 17.1 to 15.1 s for the first token and from 11.2 to 8.4 s for the warm
second token on Apple M4. Readbacks between attention, MLP, and PLE stages plus
JVM-side normalization/elementwise work remain dominant; full-layer GPU
residency/fusion is still required.

## Ollama-compatible HTTP server

The JVM host can run behind Ollama's local HTTP wire contract. Model sessions
are loaded lazily and retained across requests, so the GGUF mmap, KV/weight
caches, and persistent Metal worker are reused instead of recreated:

```sh
# defaults to 127.0.0.1:11434; an optional first argument overrides the port
KOTODAMA_METAL_K_DOT=1 clojure -M:serve-ollama 11434

curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/api/generate \
  -d '{"model":"gemma4:e4b","prompt":"The capital of France is"}'
```

Implemented endpoints are `GET /api/version`, `GET /api/tags`,
`POST /api/show`, `GET /api/ps`, and `POST /api/generate`. Generate supports
Ollama's default newline-delimited streaming and `"stream": false`, plus
`num_predict`, `temperature`, `top_k`, `top_p`, and `seed`. The transport has
real loopback HTTP tests for response shape, streaming chunks, lazy loading,
session reuse, model-not-found errors, and disposal. Model pull/push/copy/delete,
chat, embeddings, blobs, scheduling across multiple models, and exact behavior
for every Ollama option remain future compatibility work.

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
