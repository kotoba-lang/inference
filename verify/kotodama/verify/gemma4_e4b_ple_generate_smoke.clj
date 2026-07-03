(ns kotodama.verify.gemma4-e4b-ple-generate-smoke
  "Real Gemma4 GGUF -> end-to-end generation through the promoted, PLE-aware
  library host adapter `kotodama.inference.host.jvm` (KV-cache on by default).

  This is the successor to `gemma4-e4b-generate-smoke` (which runs the older
  `stable-block-forward`, a plain transformer block without per-layer
  embeddings). It exercises the stable production entry point that
  murakumo-studio calls: `kotodama.inference.host.jvm/generate`.

  Env: KOTODAMA_VERIFY_PROMPT, KOTODAMA_VERIFY_MAX_TOKENS,
       CACHE_WEIGHTS=1 (keep dequantized weights in RAM, ~20GB, much faster)."
  (:require [kotodama.inference.host.jvm :as host]))

(defn -main [& _]
  (let [prompt (or (System/getenv "KOTODAMA_VERIFY_PROMPT") "The capital of France is")
        max-tokens (Long/parseLong (or (System/getenv "KOTODAMA_VERIFY_MAX_TOKENS") "16"))
        cache-weights? (= "1" (System/getenv "CACHE_WEIGHTS"))
        t0 (System/nanoTime)
        result (host/generate {:kotodama/model "gemma4:e4b"
                               :kotodama/prompt prompt
                               :kotodama/max-tokens max-tokens
                               :kotodama/cache-weights? cache-weights?
                               :kotodama/on-token
                               (fn [id text nanos]
                                 (println (format "  [+%.1fs] %d %s" (/ nanos 1.0e9) id (pr-str text)))
                                 (flush))})
        secs (/ (- (System/nanoTime) t0) 1.0e9)]
    (println "prompt:    " (pr-str prompt))
    (println "generated: " (pr-str (:kotodama/text result)))
    (println "stop:      " (:kotodama/stop-reason result))
    (println "tokens/sec:" (/ (count (:kotodama/generated-token-ids result)) secs))
    (prn {:kotodama/gemma4-e4b-ple-generate-smoke :ok
          :kotodama/prompt prompt
          :kotodama/generated-text (:kotodama/text result)
          :gemma4/block-count (:gemma4/block-count result)})))
