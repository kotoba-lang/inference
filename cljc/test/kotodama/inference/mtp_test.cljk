(ns kotodama.inference.mtp-test
  (:require [clojure.test :refer [deftest is]]
            [kotodama.inference.mtp :as mtp]
            [torch.speculative :as speculative]))

(deftest admitted-runtime-connects-to-torch-execution
  (let [seen (atom nil)
        runtime {:kotodama/task :text-generation/mtp
                 :kotodama/mtp {:kotodama/draft-token-count 4
                                :kotodama/verify-draft? true}
                 :kotodama/checkpoint-audit {:kotodama/mtp-admitted? true}}]
    (with-redefs [speculative/make-mtp-step-fn
                  (fn [options] (reset! seen options) :step)]
      (is (= :step (mtp/make-step-fn
                    runtime {:draft-fn identity :target-fn identity
                             :verify-options {:temperature 0.7}})))
      (is (= 4 (:draft-token-count @seen)))
      (is (= {:temperature 0.7} (:verify-options @seen))))))

(deftest missing-checkpoint-admission-fails-closed
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"not admitted"
       (mtp/make-step-fn
        {:kotodama/task :text-generation/mtp
         :kotodama/mtp {:kotodama/draft-token-count 4
                        :kotodama/verify-draft? true}}
        {:draft-fn identity :target-fn identity}))))
