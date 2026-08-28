(ns kotodama.inference.qwen4exp
  "Checkpoint admission for Qwen3.8-Flash-Next / Qwen4Exp MTP execution.

  This namespace consumes config and Safetensors index data, not filenames
  guessed from a quant label. A target checkpoint without its MTP tensors is
  rejected before a host attempts speculative setup."
  (:require [clojure.string :as str]
            [kotodama.inference.runtime :as runtime]))

(def ^:private mtp-root-tensors
  ["mtp.fc_embedding.weight"
   "mtp.fc_hidden.weight"
   "mtp.hyper_connection_mixer.hc_norm.weight"
   "mtp.hyper_connection_mixer.input_mix_weight_down.weight"
   "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
   "mtp.pre_fc_norm_embedding.weight"
   "mtp.pre_fc_norm_hidden.weight"])

(def ^:private mtp-layer-suffixes
  ["attn_hyper_connection.block_inject_weight.weight"
   "attn_hyper_connection.hc_norm.weight"
   "attn_hyper_connection.input_mix_weight_down.weight"
   "attn_hyper_connection.input_mix_weight_up.weight"
   "mlp.experts.down_proj"
   "mlp.experts.gate_up_proj"
   "mlp.gate.weight"
   "mlp.shared_expert.down_proj.weight"
   "mlp.shared_expert.gate_proj.weight"
   "mlp.shared_expert.up_proj.weight"
   "mlp.shared_expert_gate.weight"
   "mlp_hyper_connection.block_inject_weight.weight"
   "mlp_hyper_connection.hc_norm.weight"
   "mlp_hyper_connection.input_mix_weight_down.weight"
   "mlp_hyper_connection.input_mix_weight_up.weight"
   "self_attn.indexer.index_qk_proj.weight"
   "self_attn.indexer.k_layernorm.weight"
   "self_attn.indexer.q_layernorm.weight"
   "self_attn.k_norm.weight"
   "self_attn.k_proj.weight"
   "self_attn.o_proj.weight"
   "self_attn.q_norm.weight"
   "self_attn.q_proj.weight"
   "self_attn.v_proj.weight"])

(defn- get-key [m k]
  (or (get m k) (get m (keyword k))))

(defn- text-config [config]
  (or (get-key config "text_config") config))

(defn mtp-layer-count [config]
  (let [text (text-config config)
        mtp (get-key text "mtp")]
    (long (or (get-key text "mtp_num_hidden_layers")
              (get-key mtp "num_hidden_layers")
              0))))

(defn required-mtp-tensors [layer-count]
  (into mtp-root-tensors
        (for [layer (range layer-count)
              suffix mtp-layer-suffixes]
          (str "mtp.layers." layer "." suffix))))

(defn checkpoint-audit
  "Return a fail-closed Qwen4Exp MTP checkpoint audit.

  `index` may be a complete `model.safetensors.index.json` map or its weight-map.
  The returned shard list is the exact acquisition set for the MTP head."
  [config index]
  (let [text (text-config config)
        architecture (first (or (get-key config "architectures") []))
        model-type (or (get-key config "model_type")
                       (get-key text "model_type"))
        layer-count (mtp-layer-count config)
        weight-map (or (get-key index "weight_map") index {})
        tensor-names (set (map name (keys weight-map)))
        required (if (pos? layer-count)
                   (required-mtp-tensors layer-count)
                   mtp-root-tensors)
        missing (vec (remove tensor-names required))
        mtp-tensors (vec (sort (filter #(str/starts-with? % "mtp.") tensor-names)))
        shards (vec (sort (distinct (keep #(or (get weight-map %)
                                               (get weight-map (keyword %)))
                                          mtp-tensors))))
        qwen4exp? (and (= "qwen4_exp" model-type)
                       (or (nil? architecture)
                           (contains? #{"Qwen4ExpForConditionalGeneration"
                                        "Qwen4ExpForCausalLM"}
                                      architecture)))
        admitted? (and qwen4exp? (pos? layer-count) (empty? missing))]
    {:kotodama/architecture :qwen4exp
     :kotodama/model-type model-type
     :kotodama/mtp-layer-count layer-count
     :kotodama/mtp-tensor-count (count mtp-tensors)
     :kotodama/mtp-shards shards
     :kotodama/mtp-missing missing
     :kotodama/mtp-admitted? admitted?}))

(defn runtime-spec
  "Create an admitted MTP runtime spec from official checkpoint metadata."
  ([model config index] (runtime-spec model config index {}))
  ([model config index opts]
   (let [audit (checkpoint-audit config index)]
     (when-not (:kotodama/mtp-admitted? audit)
       (throw (ex-info "Qwen4Exp checkpoint is not MTP-complete" audit)))
     (merge (runtime/mtp-transformer
             model
             (merge {:kotodama/mtp
                     {:kotodama/draft-token-count 4
                      :kotodama/verify-draft? true}}
                    opts))
            {:kotodama/architecture :qwen4exp
             :kotodama/checkpoint-audit audit}))))
