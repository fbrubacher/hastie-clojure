(ns hastie-clj.linear-reg
  (:require [clojure.contrib.str-utils2 :as sutils]
            [incanter.core :as icore]
            [incanter.datasets :as datasets] [incanter.charts :as charts])
  (:use clojure.contrib.generic.math-functions))


(defn create-sigmoid
  [beta]
  (icore/div 1 (icore/plus 1 (icore/exp (icore/mult -1 beta)))))

(defn grad-ascent-1
  "Gradient ascent - max of a function then the best way to move in the direction of the gradient.
  alpha is the step size towards the target, http://people.csail.mit.edu/jrennie/writing/lr.pdf
  for proof on how we get to this"
  [data-matrix class-labels]
  (let [my-matrix (icore/matrix data-matrix)
        label-mat (icore/matrix class-labels)
        nbr-rows (icore/nrow my-matrix)
        nbr-cols (icore/ncol my-matrix)
        alpha 0.001
        weights (atom (icore/matrix 1 nbr-cols 1))]
    (doall
     (for [max-cycles (range 500)]       
       (let [h (create-sigmoid (icore/mmult my-matrix @weights))
             error (icore/minus label-mat h)]
         (swap! weights (fn [weights]
                          (icore/plus weights
                                      (icore/mult alpha
                                                  (icore/mmult
                                                   (icore/trans my-matrix) error))))))))
    @weights))

(defn ridge-regression
  "subset selection often produces a model that is interpretable, but it often exhibits
  high variance because it is a discrete method, and so doesnt reduce the prediction
  error of the full model, shrinkage (or weight decay) as it is a continous doesn't suffer
  from this variance. The idea behind ridge-regression is to penelize regression coefficients
  by imposing a penalty on their size"
  []
  )

(defn eye
  ""
  []
  ())

(defn build-lwr-weights
  ""
  []
  ())

(defn locally-weighted-regression
  "regression in general tends to underfit data. locally-weighted-regression is a non-parametric
algorithm (you need to recalculate the training dataset each time you make a prediction)
w = (Xt W X)^-1 XtWy where is a matrix to weight the data points"
  [test-point dataset]
  (let [x-matrix (icore/matrix (map first dataset))
        y-matrix (icore/matrix (map fnext dataset))
        m (icore/nrow x-matrix)
        weights (icore/matrix (eye m))]
    (let [weights (build-lwr-weights x-matrix test-point)
          xTx (icore/mmult (icore/trans x-matrix) (icore/mmult weights x-matrix))
          I (icore/matrix 1)]
      (icore/mmult (icore/mmult xTx I) (icore/trans x-matrix (icore/mmult weights y-matrix))))))



(defn parse-multiple-to-float
  ""
  [& args]
  (map #(Double/parseDouble %) args))

(defn load-dataset
  []
  (let [data-text (map #(sutils/split % #"\t") (sutils/split-lines (slurp "testSet.txt")))]
    (map #(apply parse-multiple-to-float %) data-text)))

(defn lazy-seq-1
  "small hack to return a lazy-seq consisting of 100 1s"
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


(defn linear-fn-weights
  ""
  [x]
  (let [weights (calculate-weights)
        a (- (first (icore/sel weights :rows 1)))
        b (- (first (icore/sel weights :rows 0)))
        c (first (icore/sel weights :rows 2))]
    (/ (+ b (* a x)) c)))

(defn make-linear-reg-graph
  ""
  []
  (let [ whole-matrix  (icore/matrix (load-dataset))
        plot (charts/scatter-plot (icore/sel whole-matrix  :cols 0) (icore/sel whole-matrix :cols 1) :group-by (icore/sel whole-matrix :cols 2))
        x-points (range -4 4 0.5)
        y-points (map linear-fn-weights x-points)]
    (charts/add-lines plot x-points y-points)
    (icore/view plot)))

; K-nn

(defn- calculate-euclidan-distance
  "Find a better way to write this pattern of maybe having everythign inside a fn in the map"
  [point tuple]
  (let [x1 (first point) y1 (last point)
        x2 (first tuple) y2 (fnext tuple)]
    [(sqrt (+ (sqr (- x2 x1)) (sqr (- y2 y1)))) (last tuple)]))


(defn euclidian-distance
  ""
  [nbr-neighbors point]
  (let [whole-matrix (load-dataset)]
    (take nbr-neighbors
          (sort-by first
                   (map #(calculate-euclidan-distance point %) whole-matrix)))))

(defn k-nn
  ""
  [nbr-neighbors point]
  (/ (reduce + (map fnext (euclidian-distance nbr-neighbors point))) nbr-neighbors))

(def transpose (partial apply map list))


; K-nn graphs

(defn in-between
  ""
  [point knn-point]
  (println point)
  (println knn-point)
  (and (> knn-point 0) (< knn-point 1)))


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

(defn k-nn-line
  ""
  []
  (let [ whole-matrix  (icore/matrix (load-dataset))
        x-points (range -4 4 0.5)
        y-points (map calculate-point-near-middle x-points)
        plot (charts/scatter-plot (icore/sel whole-matrix  :cols 0) (icore/sel whole-matrix :cols 1) :group-by (icore/sel whole-matrix :cols 2))]
    (charts/add-lines plot x-points y-points)
    (icore/view plot)))


; SVM

(defn svm
  "margin is calculated by the label*(wTx + b)"
  []
  )
