(ns kotodama.inference.host.ollama-server
  "Ollama-compatible HTTP surface backed by the production JVM host.

  Loaded model sessions are retained across requests. This is important for
  the native and Metal hosts: their mmap handles, GPU worker, weight buffers,
  and other model-scoped resources must not be rebuilt for every generation.

  The POLICY of this surface is not written here. Route admission, request
  option defaults, session expiry and duration accounting live in
  `kotoba/*_core.kotoba` and are executed through
  `kotodama.inference.host.oracle` (ADR-2608130700). What is here is transport:
  sockets, JSON, threads, locks, and the clock."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [kotodama.inference.host.jvm :as host]
            [kotodama.inference.host.oracle :as oracle])
  (:import [com.sun.net.httpserver HttpExchange HttpHandler HttpServer]
           [java.io Closeable InputStreamReader]
           [java.net InetSocketAddress]
           [java.nio.charset StandardCharsets]
           [java.time Instant]
           [java.util Random]
           [java.util.concurrent Executors TimeUnit]))

(def default-port 11434)
(def default-host "127.0.0.1")

;; Ollama reports a far-future instant for a session that never expires; a
;; client sorting by expires_at must not see "already expired".
(def ^:private never-expires-at "9999-12-31T23:59:59Z")

(defn- now-ms [] (System/currentTimeMillis))
(defn- iso [ms] (.toString (Instant/ofEpochMilli ms)))
(defn- now [] (iso (now-ms)))

;; ── Kotoba decision cores ────────────────────────────────────────────

(defn- protocol* [export & args] (oracle/call :ollama-protocol export args))
(defn- options* [export & args] (oracle/call :ollama-options export args))
(defn- session* [export & args] (oracle/call :ollama-session export args))

(defn- json-bytes [value]
  (.getBytes (json/write-str value) StandardCharsets/UTF_8))

(defn- read-json [^HttpExchange exchange]
  (with-open [body (.getRequestBody exchange)
              reader (InputStreamReader. body StandardCharsets/UTF_8)]
    (json/read reader :key-fn keyword)))

(defn- send-bytes! [^HttpExchange exchange status content-type bytes]
  (doto (.getResponseHeaders exchange)
    (.set "content-type" content-type))
  (.sendResponseHeaders exchange status (alength ^bytes bytes))
  (with-open [out (.getResponseBody exchange)]
    (.write out ^bytes bytes)))

(defn- send-json! [exchange status value]
  (send-bytes! exchange status "application/json; charset=utf-8" (json-bytes value)))

(defn- error! [exchange status message]
  (send-json! exchange status {:error message}))

(defn- error-kind! [exchange kind message]
  (error! exchange (protocol* :error-status kind) message))

(defn- default-model-spec [model]
  {:name model
   :model model
   :modified_at (now)
   :size 0
   :digest ""
   :details {:parent_model ""
             :format "gguf"
             :family "gemma4"
             :families ["gemma4"]
             :parameter_size "8.0B"
             :quantization_level "Q4_K_M"}})

(defn service
  "Create reusable Ollama API state.

  Injection options make the transport independently verifiable:
  `:load-fn`, `:generate-fn`, and `:close-fn` have the same shapes as the JVM
  host functions. `:models` maps model names to Ollama metadata plus optional
  `:kotodama/load-opts`."
  ([] (service {}))
  ([{:keys [models load-fn generate-fn close-fn]
     :or {load-fn host/load-model
          generate-fn host/generate
          close-fn host/close-model}}]
   (let [models (or models {host/default-model (default-model-spec host/default-model)})]
     {:models models
      :sessions (atom {})
      :lifecycle-lock (Object.)
      :load-fn load-fn
      :generate-fn generate-fn
      :close-fn close-fn})))

(defn- model-spec [service model]
  (get (:models service) model))

(defn- ensure-session! [service model keep-alive-ms]
  (locking (:lifecycle-lock service)
    (if-let [entry (get @(:sessions service) model)]
      (do (reset! (:keep-alive-ms entry) keep-alive-ms) entry)
      (let [spec (or (model-spec service model)
                     (throw (ex-info "model not found"
                                     {:model model
                                      :kind (protocol* :error-not-found)})))
            session ((:load-fn service)
                     (merge {:kotodama/model model}
                            (:kotodama/load-opts spec)))
            loaded (now-ms)
            entry {:session session
                   :lock (Object.)
                   :loaded-at-ms loaded
                   :last-used-ms (atom loaded)
                   :keep-alive-ms (atom keep-alive-ms)}]
        (swap! (:sessions service) assoc model entry)
        entry))))

(defn- detach-sessions!
  "Remove `models` from the session map and return their entries. Closing runs
  outside the lifecycle lock — a close can be slow (munmap, GPU teardown) and
  must not block an unrelated load."
  [service models]
  (locking (:lifecycle-lock service)
    (let [current @(:sessions service)
          entries (keep #(find current %) models)]
      (swap! (:sessions service) #(apply dissoc % models))
      (vec entries))))

(defn- close-entries! [service entries]
  (doseq [[_model {:keys [session]}] entries]
    ((:close-fn service) session))
  (mapv first entries))

(defn reap-expired!
  "Unload every session whose keep-alive has elapsed as of `at-ms`.

  Exposed (and taking its own clock) so expiry is testable without sleeping.
  Returns the model names unloaded."
  ([service] (reap-expired! service (now-ms)))
  ([service at-ms]
   (let [expired (->> @(:sessions service)
                      (filter (fn [[_model entry]]
                                (session* :expired? at-ms
                                          @(:last-used-ms entry)
                                          @(:keep-alive-ms entry))))
                      (mapv first))]
     (if (seq expired)
       (close-entries! service (detach-sessions! service expired))
       []))))

(defn close-service! [service]
  (let [entries (locking (:lifecycle-lock service)
                  (let [entries (vec @(:sessions service))]
                    (reset! (:sessions service) {})
                    entries))]
    (close-entries! service entries)
    {:closed (count entries)}))

;; ── request options ──────────────────────────────────────────────────

(defn- keep-alive-seconds
  "Ollama accepts `keep_alive` as seconds or as a duration string (\"5m\").
  Normalising the string to seconds is parsing, so it stays here; the policy
  that turns seconds into an expiry rule is in ollama-session-core.

  Returns nil for absent/unparseable — an unparseable duration must not read
  as an explicit request (it takes the default, like absence)."
  [value]
  (cond
    (number? value) (long value)
    (string? value)
    (let [[_ n unit] (re-matches #"(?i)\s*(-?\d+)\s*(ns|us|ms|s|m|h)?\s*" value)]
      (when n
        (let [n (parse-long n)]
          (case (some-> unit str/lower-case)
            "ns" (quot n 1000000000)
            "us" (quot n 1000000)
            "ms" (quot n 1000)
            ("s" nil) n
            "m" (* n 60)
            "h" (* n 3600)))))
    :else nil))

(defn- request-keep-alive-ms [body]
  (let [seconds (keep-alive-seconds (:keep_alive body))]
    (options* :keep-alive-ms (some? seconds) (or seconds 0))))

(defn- request-stream? [body]
  (options* :stream? (some? (:stream body)) (true? (:stream body))))

(defn- generation-options [body]
  (let [options (:options body)
        rng (Random. (long (or (:seed options) (System/nanoTime))))
        top-k (:top_k options)]
    {:kotodama/max-tokens (options* :max-tokens
                                    (some? (:num_predict options))
                                    (long (or (:num_predict options) 0)))
     :kotodama/sample-opts
     (cond-> {}
       (some? (:temperature options))
       (assoc :kotodama/temperature (double (:temperature options))
              :kotodama/rand01 #(.nextDouble rng))
       (some? top-k)
       (assoc :kotodama/top-k (options* :top-k true (long top-k)))
       (some? (:top_p options))
       (assoc :kotodama/top-p (double (:top_p options))))}))

(defn- stop-code [result]
  (case (:kotodama/stop-reason result)
    :eos (protocol* :stop-eos)
    :max-tokens (protocol* :stop-max-tokens)
    (protocol* :stop-unknown)))

(defn- base-generate-response [model]
  {:model model :created_at (now)})

(defn- final-response [model result total-nanos load-nanos eval-nanos]
  (merge (base-generate-response model)
         {:response ""
          :done true
          :done_reason (protocol* :done-reason (stop-code result))
          :context (into (vec (:kotodama/prompt-token-ids result []))
                         (:kotodama/generated-token-ids result []))
          :total_duration total-nanos
          :load_duration load-nanos
          :prompt_eval_count (count (:kotodama/prompt-token-ids result []))
          :prompt_eval_duration 0
          :eval_count (count (:kotodama/generated-token-ids result []))
          :eval_duration eval-nanos}))

(defn- run-generate! [service body on-token]
  (let [model (or (:model body) host/default-model)
        keep-alive-ms (request-keep-alive-ms body)
        started (System/nanoTime)
        existed? (contains? @(:sessions service) model)
        entry (ensure-session! service model keep-alive-ms)
        loaded (System/nanoTime)]
    (locking (:lock entry)
      (let [eval-start (System/nanoTime)
            result ((:generate-fn service)
                    (merge {:kotodama/session (:session entry)
                            :kotodama/model model
                            :kotodama/prompt (or (:prompt body) "")
                            :kotodama/on-token on-token}
                           (generation-options body)))
            ended (System/nanoTime)]
        (reset! (:last-used-ms entry) (now-ms))
        {:model model
         :result result
         :total-nanos (session* :total-duration-nanos started ended)
         :load-nanos (session* :load-duration-nanos existed? started loaded)
         :eval-nanos (session* :eval-duration-nanos eval-start ended)}))))

(defn- settle-session!
  "Apply the request's keep-alive now that it has been answered. `keep_alive: 0`
  means unload immediately, which is a decision the session core makes — the
  same `expired?` rule the reaper uses, asked at this instant."
  [service model]
  (when-let [entry (get @(:sessions service) model)]
    (when (session* :expired? (now-ms) @(:last-used-ms entry) @(:keep-alive-ms entry))
      (close-entries! service (detach-sessions! service [model])))))

(defn- generate-json! [service exchange body]
  (let [{:keys [model result total-nanos load-nanos eval-nanos]}
        (run-generate! service body nil)]
    (send-json! exchange 200
                (assoc (final-response model result total-nanos load-nanos eval-nanos)
                       :response (:kotodama/text result "")))
    (settle-session! service model)))

(defn- generate-stream! [service ^HttpExchange exchange body]
  (when-not (model-spec service (or (:model body) host/default-model))
    (throw (ex-info "model not found" {:kind (protocol* :error-not-found)})))
  (doto (.getResponseHeaders exchange)
    (.set "content-type" "application/x-ndjson; charset=utf-8"))
  (.sendResponseHeaders exchange 200 0)
  (with-open [out (.getResponseBody exchange)]
    (letfn [(write! [value]
              (let [bytes (.getBytes (str (json/write-str value) "\n") StandardCharsets/UTF_8)]
                (.write out bytes)
                (.flush out)))]
      (try
        (let [{:keys [model result total-nanos load-nanos eval-nanos]}
              (run-generate!
               service body
               (fn [_token-id token-text _step-nanos]
                 (write! (merge (base-generate-response (or (:model body) host/default-model))
                                {:response token-text :done false}))))]
          (write! (final-response model result total-nanos load-nanos eval-nanos))
          (settle-session! service model))
        (catch Throwable e
          ;; Response headers may already be committed. Ollama streaming clients
          ;; expect a terminal NDJSON error object in this situation.
          (write! {:error (.getMessage e) :done true}))))))

(defn- tags-response [service]
  {:models (->> (:models service)
                vals
                (sort-by :name)
                (mapv #(dissoc % :kotodama/load-opts)))})

(defn- show-response [service model]
  (when-let [spec (model-spec service model)]
    {:license (or (:license spec) "")
     :modelfile (or (:modelfile spec) "")
     :parameters (or (:parameters spec) "")
     :template (or (:template spec) "")
     :details (:details spec)
     :model_info (or (:model_info spec) {})
     :capabilities (or (:capabilities spec) ["completion"])}))

(defn- expires-at [entry]
  (let [at (session* :expires-at-ms @(:last-used-ms entry) @(:keep-alive-ms entry))]
    (if (neg? at) never-expires-at (iso at))))

(defn- ps-response [service]
  {:models
   (mapv (fn [[model entry]]
           (let [spec (model-spec service model)]
             (merge (select-keys spec [:name :model :size :digest :details])
                    {:expires_at (expires-at entry)
                     :size_vram (long (or (:size_vram spec) 0))
                     :loaded_at (iso (:loaded-at-ms entry))})))
         @(:sessions service))})

(defn handle! [service ^HttpExchange exchange]
  (try
    (let [method (.getRequestMethod exchange)
          path (.getPath (.getRequestURI exchange))
          code (protocol* :route-code method path)
          body (when (protocol* :route-reads-body? code) (read-json exchange))]
      (condp = code
        (protocol* :route-version)
        (send-json! exchange 200 {:version (protocol* :api-version)})

        (protocol* :route-tags)
        (send-json! exchange 200 (tags-response service))

        (protocol* :route-ps)
        (send-json! exchange 200 (ps-response service))

        (protocol* :route-show)
        (if-let [shown (show-response service (or (:model body) (:name body)))]
          (send-json! exchange 200 shown)
          (error-kind! exchange (protocol* :error-not-found) "model not found"))

        (protocol* :route-generate)
        (if (request-stream? body)
          (generate-stream! service exchange body)
          (generate-json! service exchange body))

        (error-kind! exchange (protocol* :error-not-found) "not found")))
    (catch clojure.lang.ExceptionInfo e
      (error! exchange
              (protocol* :error-status
                         (or (:kind (ex-data e)) (protocol* :error-internal)))
              (.getMessage e)))
    (catch Throwable e
      (error-kind! exchange (protocol* :error-internal) (.getMessage e)))
    (finally
      (.close exchange))))

(defn start-server!
  "Start the Ollama-compatible server. Returns a Closeable server handle with
  `:port` (the actual port, useful when starting with port 0) and `:service`.

  A reaper unloads idle sessions on the cadence the session core chooses for
  the default keep-alive; `:reap-interval-ms` overrides it."
  ([] (start-server! {}))
  ([{:keys [host port backlog executor service-opts reap-interval-ms]
     :or {host default-host port default-port backlog 0}}]
   ;; Fail at start, not on the first request, if an artifact is missing.
   (oracle/preload!)
   (let [svc (service service-opts)
         server (HttpServer/create (InetSocketAddress. ^String host (int port)) backlog)
         executor (or executor (Executors/newVirtualThreadPerTaskExecutor))
         reap-ms (or reap-interval-ms
                     (session* :reap-interval-ms (options* :default-keep-alive-ms)))
         reaper (Executors/newSingleThreadScheduledExecutor)]
     (.createContext server "/" (reify HttpHandler
                                  (handle [_ exchange] (handle! svc exchange))))
     (.setExecutor server executor)
     (.scheduleWithFixedDelay reaper
                              ^Runnable (fn [] (try (reap-expired! svc)
                                                    (catch Throwable _ nil)))
                              reap-ms reap-ms TimeUnit/MILLISECONDS)
     (.start server)
     (let [actual-port (.getPort (.getAddress server))]
       {:server server
        :service svc
        :port actual-port
        :reap-interval-ms reap-ms
        :closeable
        (reify Closeable
          (close [_]
            (.stop server 0)
            (.shutdownNow reaper)
            (close-service! svc)
            (when (instance? java.util.concurrent.ExecutorService executor)
              (.close ^java.util.concurrent.ExecutorService executor))))}))))

(defn stop-server! [{:keys [^Closeable closeable]}]
  (.close closeable)
  {:stopped true})

(defn -main [& [port]]
  (let [server (start-server! {:port (if port (parse-long port) default-port)})]
    (println (str "Kotodama Ollama-compatible API listening on http://" default-host ":" (:port server)))
    (.addShutdownHook (Runtime/getRuntime)
                      (Thread. #(stop-server! server)))
    @(promise)))
