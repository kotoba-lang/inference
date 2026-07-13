(ns kotodama.inference.host.metal-kdot
  "Persistent Deno WebGPU→Metal K-dot worker with a binary JVM protocol."
  (:import (java.io BufferedInputStream BufferedOutputStream DataInputStream DataOutputStream File)
           (java.lang ProcessBuilder ProcessBuilder$Redirect)
           (java.nio.charset StandardCharsets)))

(def ^:private magic 0x4b444f54)

(defn start-worker
  ([] (start-worker {}))
  ([{:keys [worker-path]
     :or {worker-path "verify/metal_kdot_worker.js"}}]
   (let [path (.getAbsolutePath (File. worker-path))
         process (-> (ProcessBuilder. ^java.util.List
                                      ["deno" "run" "--unstable-webgpu" "--allow-read" path])
                     (.redirectError ProcessBuilder$Redirect/INHERIT)
                     (.start))]
     {:process process
      :in (DataOutputStream. (BufferedOutputStream. (.getOutputStream process) 1048576))
      :out (DataInputStream. (BufferedInputStream. (.getInputStream process) 1048576))})))

(defn alive? [worker] (.isAlive ^Process (:process worker)))

(defn close-worker! [worker]
  (when worker
    (try (.close ^DataOutputStream (:in worker)) (catch Exception _))
    (try (.close ^DataInputStream (:out worker)) (catch Exception _))
    (.destroy ^Process (:process worker)))
  {:kotodama/metal-worker-closed? true})

(defn matvec
  [worker tensor-type model-path tensor-offset tensor-bytes cache-key rows cols inputs]
  (locking worker
    (when-not (alive? worker)
      (throw (ex-info "Metal K-dot worker is not alive" {:exit (.exitValue ^Process (:process worker))})))
    (let [^bytes path-bytes (.getBytes (str model-path) StandardCharsets/UTF_8)
          ^bytes key-bytes (.getBytes (str cache-key) StandardCharsets/UTF_8)
          positions (count inputs)
          ^DataOutputStream out (:in worker)
          ^DataInputStream in (:out worker)]
      (.writeInt out magic)
      (.writeInt out (int tensor-type))
      (.writeLong out (long rows))
      (.writeLong out (long cols))
      (.writeInt out (int positions))
      (.writeLong out (long tensor-offset))
      (.writeLong out (long tensor-bytes))
      (.writeInt out (alength path-bytes))
      (.writeInt out (alength key-bytes))
      (.write out path-bytes)
      (.write out key-bytes)
      (doseq [position inputs, value position] (.writeFloat out (float value)))
      (.flush out)
      (let [status (.readInt in)
            count-or-size (.readInt in)]
        (if (zero? status)
          (let [expected (* positions rows)]
            (when-not (= expected count-or-size)
              (throw (ex-info "Metal worker output size mismatch"
                              {:expected expected :actual count-or-size})))
            (mapv (fn [_] (double-array (repeatedly rows #(double (.readFloat in)))))
                  (range positions)))
          (let [message-bytes (byte-array count-or-size)]
            (.readFully in message-bytes)
            (throw (ex-info "Metal K-dot worker failed"
                            {:status status
                             :message (String. message-bytes StandardCharsets/UTF_8)}))))))))
