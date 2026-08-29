(ns kotodama.inference.edge
  "Pinned model and OpenAI wire contract for Murakumo edge replicas.

  inference owns the model/protocol contract, torch builds the host runtime
  resource plan, and num (through torch) decides unified-memory admission."
  (:require [torch.edge-runtime :as torch-edge]))

(def ornith-1-5-9b
  {:model-id "murakumo-edge"
   :upstream-model "ornith-ai/Ornith-1.5-9B"
   :artifact-repo "ornith-ai/Ornith-1.5-9B-GGUF"
   :artifact-revision "abdd624b12ebf020b767fff532ff44fe552b28c3"
   :model-file "Ornith-1.5-9B-Q4_K_M.gguf"
   :model-bytes 5780090816
   :mmproj-file "mmproj-Ornith-1.5-9B-BF16.gguf"
   :mmproj-bytes 921704672
   :native-context 262144
   :edge-context 65536
   :vision? true
   :tool-calling? true})

(defn openai-request
  "Preserve OpenAI messages/content parts/tools; add only bounded edge defaults."
  [request]
  (when-not (and (map? request) (seq (:messages request)))
    (throw (ex-info "edge inference needs non-empty OpenAI messages" {})))
  (-> request
      (assoc :model (:model-id ornith-1-5-9b))
      (update :max_tokens #(min 2048 (or % 512)))
      (update :stream boolean)))

(defn replica-plan
  [{:keys [home llama-server port memory-bytes context-bytes api-key-file]
    :or {port 8092 context-bytes (* 2 1073741824)}}]
  (let [{:keys [model-id model-file model-bytes mmproj-file mmproj-bytes edge-context]}
        ornith-1-5-9b
        root (str home "/.murakumo/models/" model-id)]
    (torch-edge/replica-plan
     {:model-id model-id
      :model-path (str root "/" model-file)
      :mmproj-path (str root "/" mmproj-file)
      :llama-server llama-server :port port :context edge-context :parallel 1
      :memory-bytes memory-bytes
      :os-reserve-bytes (* 3 1073741824)
      :headroom-bytes 1073741824
      :runtime-bytes (+ model-bytes mmproj-bytes 536870912)
      :context-bytes context-bytes
      :api-key-file api-key-file})))
