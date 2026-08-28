(ns kotodama.inference.qwen4exp-test
  (:require [clojure.test :refer [deftest is testing]]
            [kotodama.inference.qwen4exp :as qwen4exp]))

(def config
  {"architectures" ["Qwen4ExpForConditionalGeneration"]
   "model_type" "qwen4_exp"
   "text_config" {"mtp_num_hidden_layers" 1
                   "mtp" {"num_hidden_layers" 1}}})

(def complete-index
  {"weight_map"
   (into {}
         (map-indexed (fn [i tensor]
                        [tensor (str "model-" (inc i) "-of-00131.safetensors")]))
         (qwen4exp/required-mtp-tensors 1))})

(deftest admits-an-mtp-complete-qwen4exp-checkpoint
  (let [audit (qwen4exp/checkpoint-audit config complete-index)
        spec (qwen4exp/runtime-spec "Qwen/Qwen3.8-Flash-Next"
                                    config complete-index)]
    (is (:kotodama/mtp-admitted? audit))
    (is (= 1 (:kotodama/mtp-layer-count audit)))
    (is (= 31 (:kotodama/mtp-tensor-count audit)))
    (is (= :text-generation/mtp (:kotodama/task spec)))
    (is (= :qwen4exp (:kotodama/architecture spec)))))

(deftest rejects-a-gguf-or-index-that-dropped-the-mtp-head
  (let [audit (qwen4exp/checkpoint-audit
               config {"weight_map" {"model.language_model.embed_tokens.weight"
                                      "model.safetensors"}})]
    (is (false? (:kotodama/mtp-admitted? audit)))
    (is (= 31 (count (:kotodama/mtp-missing audit))))
    (is (thrown-with-msg? #?(:clj clojure.lang.ExceptionInfo :cljs js/Error)
                          #"not MTP-complete"
                          (qwen4exp/runtime-spec "missing-mtp" config
                                                {"weight_map" {}})))))

(deftest rejects-the-wrong-architecture-even-with-lookalike-tensors
  (testing "tensor names alone cannot relabel another model as Qwen4Exp"
    (is (false? (:kotodama/mtp-admitted?
                 (qwen4exp/checkpoint-audit
                  (assoc config "model_type" "qwen3_5") complete-index))))))

(deftest separates-real-token-correctness-from-mtp-speed
  (is (= {:kotodama/mtp-execution-qualified? false
          :kotodama/mtp-optimization-qualified? false
          :kotodama/token-parity? false
          :kotodama/off-deterministic? true
          :kotodama/on-deterministic? false
          :kotodama/end-to-end-speedup 0.5514117336596434
          :kotodama/disqualification :unstable-mtp}
         (qwen4exp/execution-qualification
          {"off_deterministic" true
           "on_deterministic" false
           "token_parity" false
           "end_to_end_speedup" 0.5514117336596434})))
  (is (:kotodama/mtp-optimization-qualified?
       (qwen4exp/execution-qualification
        {:off_deterministic true :on_deterministic true
         :token_parity true :end_to_end_speedup 1.25}))))
