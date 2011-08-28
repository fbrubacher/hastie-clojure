(ns incanterplay.test.core
  (:use [hastie-clj.linear-reg] :reload)
  (:use [clojure.test])
  (:require [incanter.core :as core]))

(deftest calculate-weights-test
  (let [result (list 4.124143489627892 0.4800732928842445 -0.6168481970344016)]
    (is (= result (core/to-list (calculate-weights))))))
