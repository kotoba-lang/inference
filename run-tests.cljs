(ns run-tests
  "Run the portable part of this suite under nbb (ClojureScript on Node via
  SCI), so the .cljc here is checked by a second runtime.

  Coverage is partial and the exclusions are deliberate. There are ten test
  namespaces: six are .clj (the JVM host adapters, the ollama server, the
  ggml kdot and the kotoba parity gates) and are JVM-only by construction,
  and of the four .cljc ones two are listed here and two are not:

    kotodama.inference.core-test    listed -- portable
    kotodama.inference.shard-test   listed -- portable

    kotodama.inference.generation-test  NOT listed. It asserts that an
      unmapped character round-trips through the tokenizer's <0xXX>
      byte-fallback. That path is JVM-only on purpose in the source:
      tokenizer/symbol->ids reads #?(:clj (.getBytes ...) :cljs
      [unknown-token-id]) and tokenizer/decode reassembles UTF-8 through a
      java.io.ByteArrayOutputStream with no cljs equivalent. So under
      ClojureScript 'z' encodes to <unk> and the assertion fails honestly.
      Closing that gap means implementing byte-fallback encode/decode on
      TextEncoder/TextDecoder, which is a real change to production
      behaviour rather than a test fix, and is left as follow-up. The other
      19 assertions in that namespace do pass.

    kotodama.inference.mlx-test  NOT listed. mlx-runtime is a host adapter
      and its cljs branch deliberately throws 'mlx-runtime is a host
      adapter; provide a CLJS fetch-backed IModelRuntime instead'. The test
      is testing the JVM adapter, so there is nothing here for nbb to check.

  Measured 2026-08-20, the two runtimes agree exactly on what is listed:

    clojure -M:test -n core-test -n shard-test
                             Ran 12 tests containing 125 assertions, 0 failures
    nbb run-tests.cljs           12 tests containing 125 assertions, 0 failures

  Namespaces are listed explicitly. clojure -M:test discovers them by
  scanning the test path and a cljs runner cannot, so one left off this list
  would silently never run rather than fail.

    nbb --classpath \"$(clojure -Spath -M:test)\" run-tests.cljs"
  (:require [cljs.test :as t]
            [kotodama.inference.core-test]
            [kotodama.inference.shard-test]))

(defmethod t/report [:cljs.test/default :end-run-tests] [m]
  (println (str "\nnbb: " (:test m) " tests, " (:pass m) " passed, "
                (:fail m) " failed, " (:error m) " errors"))
  ;; Without this a failing suite exits 0 and the gate is green forever.
  (when (pos? (+ (or (:fail m) 0) (or (:error m) 0)))
    (set! (.-exitCode js/process) 1)))

(t/run-tests 'kotodama.inference.core-test
             'kotodama.inference.shard-test)
