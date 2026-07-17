(ns kotodama.inference.host.ollama-server
  "Ollama-compatible HTTP surface backed by the production JVM host.

  Loaded model sessions are retained across requests. This is important for
  the native and Metal hosts: their mmap handles, GPU worker, weight buffers,
  and other model-scoped resources must not be rebuilt for every generation."
  (:require [clojure.data.json :as json]
            [kotodama.inference.host.jvm :as host])
  (:import [com.sun.net.httpserver HttpExchange HttpHandler HttpServer]
           [java.io Closeable InputStreamReader]
           [java.net InetSocketAddress]
           [java.nio.charset StandardCharsets]
           [java.time Instant]
           [java.util Random]
           [java.util.concurrent Executors]))

(def default-port 11434)
(def default-host "127.0.0.1")

(defn- now [] (.toString (Instant/now)))

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

(defn- ensure-session! [service model]
  (locking (:lifecycle-lock service)
    (or (get @(:sessions service) model)
        (let [spec (or (model-spec service model)
                       (throw (ex-info "model not found" {:model model :status 404})))
              session ((:load-fn service)
                       (merge {:kotodama/model model}
                              (:kotodama/load-opts spec)))
              entry {:session session
                     :lock (Object.)
                     :loaded-at (now)
                     :last-used (atom (now))}]
          (swap! (:sessions service) assoc model entry)
          entry))))

(defn close-service! [service]
  (let [entries (locking (:lifecycle-lock service)
                  (let [entries (vals @(:sessions service))]
                    (reset! (:sessions service) {})
                    entries))]
    (doseq [{:keys [session]} entries]
      ((:close-fn service) session))
    {:closed (count entries)}))

(defn- generation-options [body]
  (let [options (:options body)
        rng (Random. (long (or (:seed options) (System/nanoTime))))]
    {:kotodama/max-tokens (long (or (:num_predict options) 32))
     :kotodama/sample-opts
     (cond-> {}
       (some? (:temperature options))
       (assoc :kotodama/temperature (double (:temperature options))
              :kotodama/rand01 #(.nextDouble rng))
       (some? (:top_k options))
       (assoc :kotodama/top-k (long (:top_k options)))
       (some? (:top_p options))
       (assoc :kotodama/top-p (double (:top_p options))))}))

(defn- done-reason [result]
  (case (:kotodama/stop-reason result)
    :eos "stop"
    :max-tokens "length"
    "stop"))

(defn- base-generate-response [model]
  {:model model :created_at (now)})

(defn- final-response [model result total-nanos load-nanos eval-nanos]
  (merge (base-generate-response model)
         {:response ""
          :done true
          :done_reason (done-reason result)
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
        started (System/nanoTime)
        existed? (contains? @(:sessions service) model)
        entry (ensure-session! service model)
        loaded (System/nanoTime)]
    (locking (:lock entry)
      (reset! (:last-used entry) (now))
      (let [eval-start (System/nanoTime)
            result ((:generate-fn service)
                    (merge {:kotodama/session (:session entry)
                            :kotodama/model model
                            :kotodama/prompt (or (:prompt body) "")
                            :kotodama/on-token on-token}
                           (generation-options body)))
            ended (System/nanoTime)]
        {:model model
         :result result
         :total-nanos (- ended started)
         :load-nanos (if existed? 0 (- loaded started))
         :eval-nanos (- ended eval-start)}))))

(defn- generate-json! [service exchange body]
  (let [{:keys [model result total-nanos load-nanos eval-nanos]}
        (run-generate! service body nil)]
    (send-json! exchange 200
                (assoc (final-response model result total-nanos load-nanos eval-nanos)
                       :response (:kotodama/text result "")))))

(defn- generate-stream! [service ^HttpExchange exchange body]
  (when-not (model-spec service (or (:model body) host/default-model))
    (throw (ex-info "model not found" {:status 404})))
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
          (write! (final-response model result total-nanos load-nanos eval-nanos)))
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

(defn- ps-response [service]
  {:models
   (mapv (fn [[model {:keys [loaded-at last-used]}]]
           (let [spec (model-spec service model)]
             (merge (select-keys spec [:name :model :size :digest :details])
                    {:expires_at @last-used
                     :size_vram (long (or (:size_vram spec) 0))
                     :loaded_at loaded-at})))
         @(:sessions service))})

(defn handle! [service ^HttpExchange exchange]
  (try
    (let [method (.getRequestMethod exchange)
          path (.getPath (.getRequestURI exchange))]
      (case [method path]
        ["GET" "/api/version"] (send-json! exchange 200 {:version "0.1.0-kotodama"})
        ["GET" "/api/tags"] (send-json! exchange 200 (tags-response service))
        ["GET" "/api/ps"] (send-json! exchange 200 (ps-response service))
        ["POST" "/api/show"]
        (let [body (read-json exchange)
              shown (show-response service (or (:model body) (:name body)))]
          (if shown (send-json! exchange 200 shown) (error! exchange 404 "model not found")))
        ["POST" "/api/generate"]
        (let [body (read-json exchange)]
          (if (false? (:stream body))
            (generate-json! service exchange body)
            (generate-stream! service exchange body)))
        (error! exchange 404 "not found")))
    (catch clojure.lang.ExceptionInfo e
      (error! exchange (or (:status (ex-data e)) 500) (.getMessage e)))
    (catch Throwable e
      (error! exchange 500 (.getMessage e)))
    (finally
      (.close exchange))))

(defn start-server!
  "Start the Ollama-compatible server. Returns a Closeable server handle with
  `:port` (the actual port, useful when starting with port 0) and `:service`."
  ([] (start-server! {}))
  ([{:keys [host port backlog executor service-opts]
     :or {host default-host port default-port backlog 0}}]
   (let [svc (service service-opts)
         server (HttpServer/create (InetSocketAddress. ^String host (int port)) backlog)
         executor (or executor (Executors/newVirtualThreadPerTaskExecutor))]
     (.createContext server "/" (reify HttpHandler
                                   (handle [_ exchange] (handle! svc exchange))))
     (.setExecutor server executor)
     (.start server)
     (let [actual-port (.getPort (.getAddress server))]
       {:server server
        :service svc
        :port actual-port
        :closeable
        (reify Closeable
          (close [_]
            (.stop server 0)
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
