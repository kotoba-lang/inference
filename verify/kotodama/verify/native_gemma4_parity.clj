(ns kotodama.verify.native-gemma4-parity
  "Fixed raw-completion parity gate: native kotodama host versus live Ollama."
  (:require [clojure.string :as str]
            [kotodama.inference.host.jvm :as host])
  (:import (java.net URI)
           (java.net.http HttpClient HttpRequest HttpRequest$BodyPublishers HttpResponse$BodyHandlers)
           (java.time Duration)))

(def model "gemma4:e4b")
(def prompt "The capital of France is")
(def expected-text " Paris")
(def expected-token-id 9079)

(defn- ollama-raw-one-token []
  (let [body (str "{\"model\":\"" model "\",\"prompt\":\"" prompt
                  "\",\"raw\":true,\"stream\":false,"
                  "\"options\":{\"num_predict\":1,\"temperature\":0}}")
        request (-> (HttpRequest/newBuilder (URI/create "http://127.0.0.1:11434/api/generate"))
                    (.timeout (Duration/ofMinutes 10))
                    (.header "content-type" "application/json")
                    (.POST (HttpRequest$BodyPublishers/ofString body))
                    (.build))
        response (.send (HttpClient/newHttpClient) request (HttpResponse$BodyHandlers/ofString))
        json (.body response)
        marker "\"response\":\""
        start (.indexOf json marker)]
    (when-not (= 200 (.statusCode response))
      (throw (ex-info "Ollama parity request failed" {:status (.statusCode response) :body json})))
    (when (neg? start)
      (throw (ex-info "Ollama response field missing" {:body json})))
    (let [from (+ start (count marker))
          end (.indexOf json "\"" from)]
      (subs json from end))))

(defn -main [& _]
  (let [t0 (System/nanoTime)
        local (host/generate {:kotodama/model model
                              :kotodama/prompt prompt
                              :kotodama/max-tokens 1
                              :kotodama/cache-weights? false
                              :kotodama/dbg {:native-k-dot? true}})
        local-seconds (/ (- (System/nanoTime) t0) 1.0e9)
        local-text (:kotodama/text local)
        local-ids (:kotodama/generated-token-ids local)
        ollama-text (ollama-raw-one-token)]
    (when-not (= expected-text ollama-text)
      (throw (ex-info "live Ollama fixed probe changed" {:expected expected-text :actual ollama-text})))
    (when-not (= ollama-text local-text)
      (throw (ex-info "native Gemma4 output differs from Ollama"
                      {:ollama ollama-text :kotodama local-text :ids local-ids})))
    (when-not (= [expected-token-id] local-ids)
      (throw (ex-info "native Gemma4 token id differs from fixed oracle"
                      {:expected [expected-token-id] :actual local-ids})))
    (prn {:kotodama/native-gemma4-parity :ok
          :model model
          :prompt prompt
          :token-id expected-token-id
          :text local-text
          :ollama-text ollama-text
          :seconds local-seconds})))
