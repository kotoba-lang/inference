(ns kotodama.inference.edge-test
  (:require [clojure.test :refer [deftest is]]
            [kotodama.inference.edge :as edge]))

(deftest preserves-multimodal-and-tools
  (let [messages [{:role "user" :content [{:type "text" :text "read"}
                                           {:type "image_url"
                                            :image_url {:url "https://example.test/a.png"}}]}]
        tools [{:type "function" :function {:name "done" :parameters {:type "object"}}}]
        request (edge/openai-request {:model "anything" :messages messages
                                      :tools tools :max_tokens 9999})]
    (is (= "murakumo-edge" (:model request)))
    (is (= messages (:messages request)))
    (is (= tools (:tools request)))
    (is (= 2048 (:max_tokens request)))))

(deftest builds-admitted-16g-plan
  (let [plan (edge/replica-plan
              {:home "/Users/a" :llama-server "/opt/llama-server"
               :memory-bytes (* 16 1073741824)})]
    (is (:admitted? plan))
    (is (= "murakumo-edge" (:model-id plan)))
    (is (some #{"--mmproj"} (:argv plan)))))
