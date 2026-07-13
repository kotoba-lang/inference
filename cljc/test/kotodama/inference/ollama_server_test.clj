(ns kotodama.inference.ollama-server-test
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [kotodama.inference.host.ollama-server :as server])
  (:import [java.net URI]
           [java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers
            HttpResponse$BodyHandlers]))

(defn- request [port method path body]
  (let [builder (-> (HttpRequest/newBuilder
                     (URI/create (str "http://127.0.0.1:" port path)))
                    (.header "content-type" "application/json"))
        request (case method
                  :get (.GET builder)
                  :post (.POST builder (HttpRequest$BodyPublishers/ofString
                                        (json/write-str body))))
        response (.send (HttpClient/newHttpClient)
                        (.build request)
                        (HttpResponse$BodyHandlers/ofString))]
    {:status (.statusCode response)
     :content-type (some-> (.firstValue (.headers response) "content-type") (.orElse nil))
     :body (.body response)}))

(defn- parsed [response]
  (json/read-str (:body response) :key-fn keyword))

(defn- fixture []
  (let [loads (atom 0)
        closes (atom 0)
        sessions (atom [])
        models {"fixture:latest"
                {:name "fixture:latest"
                 :model "fixture:latest"
                 :modified_at "2026-07-13T00:00:00Z"
                 :size 42
                 :digest "sha256:fixture"
                 :details {:format "gguf" :family "gemma4"
                           :parameter_size "8.0B"
                           :quantization_level "Q4_K_M"}}}
        running
        (server/start-server!
         {:port 0
          :service-opts
          {:models models
           :load-fn (fn [opts]
                      (swap! loads inc)
                      {:id @loads :opts opts})
           :generate-fn
           (fn [{:keys [kotodama/session kotodama/on-token]}]
             (swap! sessions conj session)
             (when on-token
               (on-token 11 " Paris" 10)
               (on-token 12 "." 20))
             {:kotodama/prompt-token-ids [2 3 4]
              :kotodama/generated-token-ids [11 12]
              :kotodama/text " Paris."
              :kotodama/stop-reason :max-tokens})
           :close-fn (fn [_] (swap! closes inc))}})]
    {:running running :loads loads :closes closes :sessions sessions}))

(deftest ollama-compatible-discovery-and-lifecycle
  (let [{:keys [running loads closes]} (fixture)
        port (:port running)]
    (try
      (testing "tags advertises configured GGUF metadata without loading it"
        (let [response (request port :get "/api/tags" nil)
              model (first (:models (parsed response)))]
          (is (= 200 (:status response)))
          (is (= "fixture:latest" (:name model)))
          (is (= "Q4_K_M" (get-in model [:details :quantization_level])))
          (is (zero? @loads))))
      (testing "show and version expose Ollama-compatible objects"
        (is (= "gemma4"
               (get-in (parsed (request port :post "/api/show"
                                        {:model "fixture:latest"}))
                       [:details :family])))
        (is (string? (:version (parsed (request port :get "/api/version" nil))))))
      (testing "ps contains only lazily loaded models"
        (is (empty? (:models (parsed (request port :get "/api/ps" nil))))))
      (finally
        (server/stop-server! running)))
    (is (zero? @closes))))

(deftest generate-supports-json-streaming-and-session-reuse
  (let [{:keys [running loads closes sessions]} (fixture)
        port (:port running)]
    (try
      (testing "non-streaming response includes generation accounting"
        (let [response (request port :post "/api/generate"
                                {:model "fixture:latest"
                                 :prompt "The capital of France is"
                                 :stream false
                                 :options {:num_predict 2}})
              body (parsed response)]
          (is (= 200 (:status response)))
          (is (= "application/json; charset=utf-8" (:content-type response)))
          (is (= " Paris." (:response body)))
          (is (= true (:done body)))
          (is (= "length" (:done_reason body)))
          (is (= 3 (:prompt_eval_count body)))
          (is (= 2 (:eval_count body)))
          (is (= [2 3 4 11 12] (:context body)))))
      (testing "default streaming is newline-delimited and flushes token chunks"
        (let [response (request port :post "/api/generate"
                                {:model "fixture:latest" :prompt "x"})
              chunks (mapv #(json/read-str % :key-fn keyword)
                           (remove str/blank? (str/split-lines (:body response))))]
          (is (= 200 (:status response)))
          (is (= "application/x-ndjson; charset=utf-8" (:content-type response)))
          (is (= [" Paris" "." ""] (mapv :response chunks)))
          (is (= [false false true] (mapv :done chunks)))))
      (testing "both requests shared one loaded model session"
        (is (= 1 @loads))
        (is (= 2 (count @sessions)))
        (is (apply = @sessions))
        (is (= ["fixture:latest"]
               (mapv :name (:models (parsed (request port :get "/api/ps" nil)))))))
      (finally
        (server/stop-server! running)))
    (is (= 1 @closes))))

(deftest unknown-model-is-a-json-404
  (let [{:keys [running]} (fixture)]
    (try
      (let [response (request (:port running) :post "/api/generate"
                              {:model "missing" :prompt "x" :stream false})]
        (is (= 404 (:status response)))
        (is (= "model not found" (:error (parsed response)))))
      (finally
        (server/stop-server! running)))))
