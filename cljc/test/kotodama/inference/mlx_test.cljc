(ns kotodama.inference.mlx-test
  "Covers the pure (no-I/O) surface of kotodama.inference.mlx: probe!/load!
   shape and defaulting. generate!/forward! open a real HTTP connection (same
   as kotodama.inference.ollama) so they are exercised by hand against a
   running mlx_lm.server / mlx-moe serve, not here — see the mlx.cljc
   docstring."
  (:require [kotodama.inference.mlx :as mlx]
            [kotodama.inference.ports :as ports]
            #?(:clj [clojure.test :refer [deftest is testing]]
               :cljs [cljs.test :refer-macros [deftest is testing]])))

(deftest mlx-runtime-defaults
  (testing "probe! reports the mlx-lm backend and default base-url by default"
    (let [rt (mlx/mlx-runtime)]
      (is (= {:kotodama/backends [:mlx]
              :kotodama/mlx-backend :mlx-lm
              :kotodama/base-url mlx/default-base-url}
             (ports/probe! rt))))))

(deftest mlx-runtime-mlx-moe-backend
  (testing "the :backend option is surfaced by probe!, for either serving process"
    (let [rt (mlx/mlx-runtime {:backend :mlx-moe :base-url "http://127.0.0.1:9090"})]
      (is (= {:kotodama/backends [:mlx]
              :kotodama/mlx-backend :mlx-moe
              :kotodama/base-url "http://127.0.0.1:9090"}
             (ports/probe! rt))))))

(deftest mlx-runtime-session-shape
  (testing "load! derives a session id from :model + :backend, carries the spec through"
    (let [rt (mlx/mlx-runtime {:model "mlx-community/Qwen3.6-35B-A3B-4bit" :backend :mlx-moe})
          spec {:kotodama/runtime :mlx}
          session (ports/load! rt spec)]
      (is (= "mlx:mlx-moe:mlx-community/Qwen3.6-35B-A3B-4bit" (:kotodama/session-id session)))
      (is (= :mlx (:kotodama/runtime session)))
      (is (= :mlx-moe (:kotodama/mlx-backend session)))
      (is (= "mlx-community/Qwen3.6-35B-A3B-4bit" (:kotodama/model session)))
      (is (= spec (:kotodama/spec session)))))
  (testing "load! falls back to the runtime-spec's :kotodama/model when no :model option was given"
    (let [rt (mlx/mlx-runtime {})
          session (ports/load! rt {:kotodama/model "mlx-community/Qwen2.5-0.5B-Instruct-4bit"})]
      (is (= "mlx-community/Qwen2.5-0.5B-Instruct-4bit" (:kotodama/model session))))))
