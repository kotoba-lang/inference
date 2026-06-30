(ns kotodama.inference.validate
  (:require [kotodama.inference.runtime :as runtime]))

(defn- problem [code message data]
  (merge {:kotodama/severity :error
          :kotodama/code code
          :kotodama/message message}
         data))

(defn runtime-problems [runtime-spec]
  (let [rt (:kotodama/runtime runtime-spec)
        backend (:kotodama/backend runtime-spec)
        compute-backend (:kotodama/compute-backend runtime-spec)
        task (:kotodama/task runtime-spec)
        distributed (:kotodama/distributed runtime-spec)
        tensor-parallel-size (:kotodama/tensor-parallel-size distributed)
        pipeline-parallel-size (:kotodama/pipeline-parallel-size distributed)
        scheduler (:kotodama/scheduler runtime-spec)]
    (cond-> []
      (not (contains? runtime/supported-runtimes rt))
      (conj (problem :runtime/unsupported "unsupported runtime"
                     {:kotodama/runtime rt}))

      (not (contains? runtime/supported-tasks task))
      (conj (problem :task/unsupported "unsupported inference task"
                     {:kotodama/task task}))

      (not (contains? runtime/supported-backends backend))
      (conj (problem :backend/unsupported "unsupported backend"
                     {:kotodama/backend backend}))

      (not (contains? runtime/supported-compute-backends compute-backend))
      (conj (problem :compute-backend/unsupported "unsupported num-clj compute backend"
                     {:kotodama/compute-backend compute-backend}))

      (and (= rt :torch-transformer) (= backend :webgpu) (not= compute-backend :num/wgsl)
           (not= compute-backend :num/webgpu))
      (conj (problem :compute-backend/not-webgpu
                     "WebGPU transformer runtime should use num-clj WGSL/WebGPU compute"
                     {:kotodama/backend backend
                      :kotodama/compute-backend compute-backend}))

      (and (= rt :torch-transformer) (= backend :webgl) (not= compute-backend :num/webgl))
      (conj (problem :compute-backend/not-webgl
                     "WebGL transformer runtime requires a host-provided num-clj WebGL backend"
                     {:kotodama/backend backend
                      :kotodama/compute-backend compute-backend}))

      (and (contains? runtime/supported-runtimes rt) (empty? (:kotodama/model runtime-spec)))
      (conj (problem :model/missing "torch-transformer runtime requires :kotodama/model" {}))

      (and (contains? runtime/supported-runtimes rt) (nil? (:kotodama/model-graph runtime-spec)))
      (conj (problem :model-graph/missing
                     "torch runtime requires a torch-clj :kotodama/model-graph"
                     {}))

      (and distributed
           (or (not (integer? tensor-parallel-size))
               (not (pos? tensor-parallel-size))
               (not (integer? pipeline-parallel-size))
               (not (pos? pipeline-parallel-size))))
      (conj (problem :distributed/parallel-size
                     "distributed tensor/pipeline parallel sizes must be positive integers"
                     {:kotodama/distributed distributed}))

      (and (= rt :torch-diffusion)
           (not (contains? #{:image-generation :image-to-image} task)))
      (conj (problem :task/not-diffusion
                     "torch-diffusion runtime requires an image generation task"
                     {:kotodama/runtime rt
                      :kotodama/task task}))

      (and (= rt :torch-diffusion)
           (not (contains? runtime/supported-schedulers scheduler)))
      (conj (problem :scheduler/unsupported
                     "unsupported diffusion scheduler"
                     {:kotodama/scheduler scheduler}))

      (and (= rt :torch-audio)
           (not (contains? #{:speech-to-text :audio-transcription :audio-translation} task)))
      (conj (problem :task/not-audio
                     "torch-audio runtime requires a speech/audio task"
                     {:kotodama/runtime rt
                      :kotodama/task task})))))

(defn generation-problems [generation]
  (let [max-new (:kotodama/max-new-tokens generation)]
    (cond-> []
      (and max-new (or (not (integer? max-new)) (neg? max-new)))
      (conj (problem :generation/max-new-tokens
                     ":kotodama/max-new-tokens must be a non-negative integer"
                     {:kotodama/max-new-tokens max-new})))))

(defn problems [op]
  (case (:kotodama/op op)
    :load (runtime-problems (:kotodama/runtime-spec op))
    :generate (generation-problems (:kotodama/generation op))
    :llm-infer (generation-problems (:kotodama/generation op))
    :forward []
    [(problem :op/unsupported "unsupported inference op" {:kotodama/op (:kotodama/op op)})]))

(defn valid? [op]
  (not-any? #(= :error (:kotodama/severity %)) (problems op)))
