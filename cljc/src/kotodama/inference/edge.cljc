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

(def ornith-1-5-9b-mtp
  {:model-id "murakumo-edge-mtp-canary"
   :upstream-model "ornith-ai/Ornith-1.5-9B"
   :artifact-repo "protoLabsAI/Ornith-1.5-9B-MTP-GGUF"
   :artifact-revision "5957ee5dcb88e9a9f4cd9a23c649320d20a574cd"
   :model-file "Ornith-1.5-9B-MTP-IQ4_XS.gguf"
   :model-bytes 5454999360
   :model-sha256 "962cf048bd64066003e8253c5b7317fde5fa70f0d4c70dc6e892a33e0f240981"
   :mmproj-file "mmproj-Ornith-1.5-9B-BF16.gguf"
   :mmproj-bytes 921704416
   :mmproj-sha256 "3b8b6b4f357aae0135a2826dbbf2d72cdd096f1878b50709449f5070a0e6d32f"
   :llama-tag "b10472"
   :llama-revision "60eeeb6082c1126bb8bc72902c83123cd056811b"
   :mtp? true
   :draft-token-count 3
   :speculative-bytes 805306368
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

(defn- plan-for
  [artifact {:keys [home llama-server port memory-bytes context-bytes api-key-file]
             :or {port 8092 context-bytes (* 2 1073741824)}}]
  (let [{:keys [model-id model-file model-bytes mmproj-file mmproj-bytes edge-context
                mtp? draft-token-count speculative-bytes]}
        artifact
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
      :speculative-bytes (or speculative-bytes 0)
      :context-bytes context-bytes
      :mtp? (boolean mtp?)
      :draft-token-count (or draft-token-count 3)
      :api-key-file api-key-file})))

(defn replica-plan [options]
  (plan-for ornith-1-5-9b options))

(defn mtp-replica-plan [options]
  (plan-for ornith-1-5-9b-mtp options))
