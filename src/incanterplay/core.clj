(ns linear-reg
  (:require [clojure.contrib.str-utils2 :as sutils]
            [incanter.core :as icore]
            [incanter.datasets :as datasets] [incanter.charts :as charts])
  (:use clojure.contrib.generic.math-functions))


(defn create-sigmoid
  [beta]
  (icore/div 1 (icore/plus 1 (icore/exp (icore/mult -1 beta)))))

(defn grad-ascent-1
  [data-matrix class-labels]
  (let [my-matrix (icore/matrix data-matrix)
        label-mat (icore/matrix class-labels)
        nbr-rows (icore/nrow my-matrix)
        nbr-cols (icore/ncol my-matrix)
        alpha 0.001
        weights (atom (icore/matrix 1 nbr-cols 1))]
    (doall (for [max-cycles (range 500)]       
             (let [h (create-sigmoid (icore/mmult my-matrix @weights))
                   error (icore/minus label-mat h)]
               (swap! weights (fn [weights] (icore/plus weights (icore/mult alpha (icore/mmult  (icore/trans my-matrix) error))))))))
    @weights))

(defn parse-multiple-to-float
  ""
  [& args]
  (map #(Double/parseDouble %) args))

(defn load-dataset
  []
  (let [data-text (map #(sutils/split % #"\t") (sutils/split-lines (slurp "testSet.txt")))]
    (map #(apply parse-multiple-to-float %) data-text)))

(defn lazy-seq-1
  "small hack to return a lazy-seqs consisting of 100 1s"
  []
  (for [x (range 100)]
    1.0))

(defn calculate-weights
  ""
  []
  (let [whole-matrix (load-dataset)
        data-matrix (icore/trans
                     (icore/matrix [(lazy-seq-1) (map first whole-matrix) (map fnext whole-matrix)]))
        label-matrix (icore/matrix (map last whole-matrix))
        ]
    (grad-ascent-1 data-matrix label-matrix)))

(defn make-linear-reg-graph
  ""
  []
  (let [ whole-matrix  (icore/matrix (load-dataset))
        plot (charts/scatter-plot (icore/sel whole-matrix  :cols 0) (icore/sel whole-matrix :cols 1) :group-by (icore/sel whole-matrix :cols 2))
        x-points (range -4 4 0.5)
        y-points (map linear-fn-weights x-points)]
    (charts/add-lines plot x-points y-points)
    (icore/view plot)))

(defn linear-fn-weights
  ""
  [x]
  (let [weights (calculate-weights)
        a (- (first (icore/sel weights :rows 1)))
        b (- (first (icore/sel weights :rows 0)))
        c (first (icore/sel weights :rows 2))]
    (/ (+ b (* a x)) c)))

; K-nn graphs

(defn k-nn-line
  ""
  []
  (let [ whole-matrix  (icore/matrix (load-dataset))
        x-points (range -4 4 0.5)
        y-points (map calculate-point-near-middle x-points)
        plot (charts/scatter-plot (icore/sel whole-matrix  :cols 0) (icore/sel whole-matrix :cols 1) :group-by (icore/sel whole-matrix :cols 2))]
    (charts/add-lines plot x-points y-points)
    (icore/view plot)))

(defn calculate-point-near-middle
  ""
  [x-point]
  (loop [y-candidates (range -3 15 0.01)]
    (let [y-candidate (first y-candidates)
          point (vector x-point y-candidate)
          knn-point (k-nn 15 point)]
      (if (in-between point knn-point)
        y-candidate
        (recur (next y-candidates))))))

(defn in-between
  ""
  [point knn-point]
  (println point)
  (println knn-point)
  (and (> knn-point 0) (< knn-point 1)))

; K-nn

(defn k-nn
  ""
  [nbr-neighbors point]
  (/ (reduce + (map fnext (euclidian-distance nbr-neighbors point))) nbr-neighbors))

(defn euclidian-distance
  ""
  [nbr-neighbors point]
  (let [whole-matrix (load-dataset)]
    (take nbr-neighbors
          (sort-by first
                   (map #(calculate-euclidan-distance point %) whole-matrix)))))

(def transpose (partial apply map list))

(defn- calculate-euclidan-distance
  "Find a better way to write this pattern of maybe having everythign inside a fn in the map"
  [point tuple]
  (let [x1 (first point) y1 (last point)
        x2 (first tuple) y2 (fnext tuple)]
    [(sqrt (+ (sqr (- x2 x1)) (sqr (- y2 y1)))) (last tuple)]))
