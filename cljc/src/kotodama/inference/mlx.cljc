(ns kotodama.inference.mlx
  "CLJC host adapter for a local MLX-based model runtime on Apple Silicon.

  Structurally mirrors `kotodama.inference.ollama` (same `ports/fn-runtime`
  shape, same hand-rolled minimal JSON — this repo carries no JSON dependency
  for a host adapter this thin) but targets the OpenAI-compatible HTTP server
  both local MLX serving paths expose:

  - `:mlx-lm`  — Apple's own `mlx_lm.server` (dense/standard MLX checkpoints)
  - `:mlx-moe` — mu-hashmi/mlx-moe `serve` (single-node MoE, SSD-paged experts;
                 see kotoba-lang/murakumo's `murakumo.infer.moe` for the fleet
                 planner that launches this)

  Both speak the same wire protocol (`POST /v1/chat/completions`), so one
  client covers both — `:backend` only labels which process convention the
  caller launched, for `probe!`'s benefit; it does not change the request
  shape. This is intentionally a host implementation of
  `kotodama.inference.ports/IModelRuntime`, not the portable model contract
  itself — same division of responsibility as the Ollama adapter."
  (:require [kotodama.inference.ports :as ports])
  #?(:clj
     (:import [java.net URI]
              [java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers
               HttpResponse$BodyHandlers]
              [java.time Duration])))

(def default-base-url "http://127.0.0.1:8080")
(def default-backend :mlx-lm)

(defn- json-escape [s]
  (-> (str s)
      (.replace "\\" "\\\\")
      (.replace "\"" "\\\"")
      (.replace "\n" "\\n")
      (.replace "\r" "\\r")
      (.replace "\t" "\\t")))

(declare json-value)

(defn- json-array [xs]
  (str "[" (->> xs (map json-value) (interpose ",") (apply str)) "]"))

(defn- json-object [m]
  (str "{"
       (->> m
            (keep (fn [[k v]]
                    (when (some? v)
                      (str "\"" (name k) "\":" (json-value v)))))
            (interpose ",")
            (apply str))
       "}"))

(defn- json-value [v]
  (cond
    (string? v) (str "\"" (json-escape v) "\"")
    (keyword? v) (str "\"" (name v) "\"")
    (number? v) (str v)
    (boolean? v) (str v)
    (map? v) (json-object v)
    (sequential? v) (json-array v)
    :else (str "\"" (json-escape v) "\"")))

(defn- json-body [m] (json-object m))

(defn- unescape-json-string [s]
  (loop [xs (seq s), out (StringBuilder.)]
    (if-not xs
      (str out)
      (let [c (first xs)]
        (if (not= c \\)
          (do (.append out c) (recur (next xs) out))
          (let [e (second xs)
                more (nnext xs)]
            (case e
              \" (do (.append out \") (recur more out))
              \\ (do (.append out \\) (recur more out))
              \/ (do (.append out \/) (recur more out))
              \b (do (.append out \backspace) (recur more out))
              \f (do (.append out \formfeed) (recur more out))
              \n (do (.append out \newline) (recur more out))
              \r (do (.append out \return) (recur more out))
              \t (do (.append out \tab) (recur more out))
              (do (.append out e) (recur more out)))))))))

(defn- field-value-start
  "Index just past `\"field\"` + optional whitespace + `:` + optional
   whitespace in `json`, or nil. Whitespace-tolerant because host JSON
   serializers disagree: Go's json.Marshal (Ollama) emits compact
   `\"field\":value`, Python's json.dumps (mlx_lm.server / mlx-moe, both
   OpenAI-shaped Python servers) emits `\"field\": value` with a space."
  [json field]
  (let [m (re-matcher (re-pattern (str "\"" #?(:clj (java.util.regex.Pattern/quote field)
                                              :cljs field)
                                       "\"\\s*:\\s*"))
                      json)]
    (when (.find m) (.end m))))

(defn- extract-json-string
  "First `\"field\": \"...\"` occurrence in `json` → its unescaped value, or
   nil. Good enough for a single-choice, non-streaming chat-completion
   response (the same simplifying assumption the Ollama adapter makes)."
  [json field]
  (when-let [from (field-value-start json field)]
    (when (and (< from (count json)) (= \" (.charAt ^String json from)))
      (let [from (inc from)]
        (loop [i from, escaped? false]
          (when (< i (count json))
            (let [c (.charAt ^String json i)]
              (cond
                escaped? (recur (inc i) false)
                (= c \\) (recur (inc i) true)
                (= c \") (unescape-json-string (subs json from i))
                :else (recur (inc i) false)))))))))

(defn- extract-json-number
  "First `\"field\": <number>` occurrence in `json` → a double, or nil."
  [json field]
  (when-let [from (field-value-start json field)]
    (let [m (re-matcher #"-?\d+(\.\d+)?" json)]
      (when (.find m from)
        (when (= (.start m) from)
          (parse-double (.group m)))))))

#?(:clj
   (defn- post-json [base-url path body timeout-ms]
     (let [client (HttpClient/newHttpClient)
           request (-> (HttpRequest/newBuilder (URI/create (str base-url path)))
                       (.timeout (Duration/ofMillis timeout-ms))
                       (.header "content-type" "application/json")
                       (.POST (HttpRequest$BodyPublishers/ofString body))
                       (.build))
           response (.send client request (HttpResponse$BodyHandlers/ofString))
           status (.statusCode response)
           text (.body response)]
       (if (<= 200 status 299)
         text
         (throw (ex-info "mlx request failed"
                         {:kotodama/status status
                          :kotodama/body text}))))))

(defn mlx-runtime
  "Create a CLJ-side `IModelRuntime` backed by a local OpenAI-compatible MLX
  server — either Apple's `mlx_lm.server` or mu-hashmi/mlx-moe's `serve`.

  Options:
  - `:base-url` defaults to `http://127.0.0.1:8080`
  - `:backend`  `:mlx-lm` (default) or `:mlx-moe` — descriptive only, surfaced
    via `probe!`; both use the same `/v1/chat/completions` wire shape.
  - `:model` defaults to the model in the runtime spec/session
  - `:timeout-ms` defaults to 180000"
  ([] (mlx-runtime {}))
  ([{:keys [base-url backend model timeout-ms]
     :or {base-url default-base-url backend default-backend timeout-ms 180000}}]
   #?(:clj
      (ports/fn-runtime
        {:probe (fn [] {:kotodama/backends [:mlx]
                        :kotodama/mlx-backend backend
                        :kotodama/base-url base-url})
         :load (fn [runtime-spec]
                 {:kotodama/session-id (str "mlx:" (name backend) ":"
                                            (or model (:kotodama/model runtime-spec)))
                  :kotodama/runtime :mlx
                  :kotodama/mlx-backend backend
                  :kotodama/model (or model (:kotodama/model runtime-spec))
                  :kotodama/spec runtime-spec})
         :generate (fn [session prompt generation]
                     (let [model-id (or model (:kotodama/model session))
                           response (post-json
                                      base-url
                                      "/v1/chat/completions"
                                      (json-body {:model model-id
                                                  :messages [{:role "user" :content prompt}]
                                                  :max_tokens (:kotodama/max-new-tokens generation 64)
                                                  :temperature (:kotodama/temperature generation 0.2)})
                                      timeout-ms)
                           text (or (extract-json-string response "content") "")
                           completion-tokens (extract-json-number response "completion_tokens")]
                       (cond-> {:kotodama/text text
                                :kotodama/model model-id
                                :kotodama/runtime :mlx
                                :kotodama/mlx-backend backend
                                :kotodama/raw response}
                         completion-tokens (assoc :kotodama/completion-tokens completion-tokens))))
         :forward (fn [_ _ _]
                    (throw (ex-info "mlx runtime does not expose raw tensor forward"
                                    {:kotodama/runtime :mlx :kotodama/mlx-backend backend})))})
      :cljs
      (throw (js/Error. "mlx-runtime is a host adapter; provide a CLJS fetch-backed IModelRuntime instead")))))
