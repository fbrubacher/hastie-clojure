(ns linear-reg
  (:require [clojure.contrib.str-utils2 :as sutils]
            [incanter.core :as icore] [incanter.datasets :as datasets]))


(defn create-sigmoid
  [beta]
  (icore/div 1 (icore/plus 1 (icore/exp (icore/mult -1 beta)))))

(defn calculate-weights
  ""
  [weights my-matrix label-mat]
  (icore/mmult my-matr weights))
;  (let [h (create-sigmoid (icore/mmult my-matrix weights))
 ;       error (icore/minus label-mat h)
  ;      alpha 0.001]
    
    ;(icore/plus weights
     ;           (icore/mult alpha
   ;                         (icore/mmult error (icore/trans my-matrix))))))

(defn grad-ascent-1
  [data-matrix class-labels]
  (let [my-matrix (icore/matrix data-matrix)
        label-mat (icore/matrix class-labels)
        nbr-rows (icore/nrow my-matrix)
        nbr-cols (icore/ncol my-matrix)
        alpha 0.001
        max-cycles 500]
    (loop [max-cycles 500
           weights (icore/matrix 1 nbr-cols 1)] 
      (if (> 0 max-cycles)
        weights
        (let [
               h  (create-sigmoid (icore/mmult my-matrix weights))
               error (icore/minus label-mat h)
               new-weights (icore/plus weights (icore/mult alpha (icore/mmult  (icore/trans my-matrix) error)))
               ]
          (recur (dec max-cycles)
                 (icore/plus weights (icore/mult alpha (icore/mmult  (icore/trans my-matrix) error)))))))))

(defn grad-ascent
  [data-matrix class-labels]
  (let [my-matrix (icore/matrix data-matrix)
        label-mat (icore/trans (icore/matrix class-labels))
        nbr-rows (icore/nrow my-matrix)
        nbr-cols (icore/ncol my-matrix)]
    (loop [max-cycles 500
           weights (icore/matrix 1 nbr-cols nbr-cols)]
      (println weights)
      (if (< max-cycles 0)
        weights
        (recur (dec max-cycles)
               (calculate-weights weights my-matrix label-mat))))))

(defn tab-parse-and-cast-int
  [triple-tuple]
  (let [file-output ((comp #(Integer/parseInt %) #(sutils/split % #"\t")) triple-tuple)]
    file-output))

(defn tab-parse-and-cast-int2
  [triple-tuple & args]
  (println triple-tuple args))

(defn calculate-weights
  ""
  [weights matrix label-mat]
  (let [alpha 0.01
        h (create-sigmoid (mmult matrix weights))
        error (- label-mat h)]
    (+ (* alpha (mmult (trans data-matrix) error)) weights)))

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

(defn main-
  ""
  []
  (let [whole-matrix (load-dataset)
        data-matrix (icore/trans
                     (icore/matrix [(lazy-seq-1) (map first whole-matrix) (map fnext whole-matrix)]))
        label-matrix (icore/matrix (map last whole-matrix))]
    (grad-ascent-1 data-matrix label-matrix)))

















(defn initial-stoc-grad-accent
  ""
  [data-matrix classLabels numIter]
  (let [nbr-rows (icore/nrow data-matrix)
        nbr-cols (icore/ncol data-matrix)
        weights (ones nbr-rows)]
    (map #(sutils/split % #"\t") (sutils/split-lines (slurp "testSet.txt")))))

(defn colicTest
  []
  (let [train-data (sutils/split (slurp "horseColicTraining.txt") #"\t|\r\n")
        test-data (sutils/split (slurp "horseColicTest.txt") #"\t|\r\n")]
    (let [in-groups-of-21 (partition 21 train-data)
          train-weights (initial-stoc-grad-accent in-groups-of-21 training-labels 1000)])))


(def a (matrix [            [5,5,0,5],
                            [5,0,3,4],
                            [3,4,0,3],
                            [0,0,5,3],
                            [5,4,4,5],
                            [5,4,5,5] 
                            ]))

(def vtprueba (matrix [[-0.5710 0.2228] [-0.4275, 0.5172] [-0.3846, -0.8246] [-0.5859, 0.0532]]))

(def svdmat (decomp-svd a))

(def bobt (matrix [5 5 0 0 0 5]))

(let [svdecomp (decomp-svd a)]
  (let [{:keys [U V S]} svdecomp
        u1 (sel U :cols [0 1])
        vt (trans V)
        v2 (sel vt :rows [0 1])
]
    (println  u1)))


u, s, vt = m.singular_value_decomposition
vt = vt.transpose
 
# Take the 2-rank approximation of the Matrix
#   - Take first and second columns of u  (6x2)
#   - Take first and second columns of vt (4x2)
#   - Take the first two eigen-values (2x2)
u2 = Linalg::DMatrix.join_columns [u.column(0), u.column(1)]
v2 = Linalg::DMatrix.join_columns [vt.column(0), vt.column(1)]
eig2 = Linalg::DMatrix.columns [s.column(0).to_a.flatten[0,2], s.column(1).to_a.flatten[0,2]]
 
# Here comes Bob, our new user
bob = Linalg::DMatrix[[5,5,0,0,0,5]]
bobEmbed = bob * u2 * eig2.inverse

;; A couple of chaps have asked me to write a clojure macro tutorial, to explain
;; my debugging macro:

(defmacro dbg[x] `(let [x# ~x] (println '~x "=" x#) x#))

;; Which I use to print out intermediate values in functions.

;; Here's an example function that we might want to debug:
(defn pythag [ x y ] (* (* x x) (* y y)))

;; And here is a version enhanced to print out its thought process as it runs
(defn pythag [ x y ]  (dbg (* (dbg (* x x)) (dbg (* y y)))))

(pythag 4 5)
;; (* x x) = 16
;; (* y y) = 25
;; (* (dbg (* x x)) (dbg (* y y))) = 400
;; 400

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; I'm going to try to imagine that I didn't know how to write dbg, and had to
;; go at it by trial and error, to show why it is as it is.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The problem that dbg solves:

;; Often the best way to debug code is by adding print statements.

;; Consider our old friend factorial

(defn factorial [n]
  (if (< n 2) n
      (* n (factorial (dec n)))))

(factorial 5) ;; 120

;; How would we watch it at work?
;; This modified version prints out the value of every recursive call:

(defn factorial [n]
  (if (< n 2) n
      (* n (let [a (factorial (dec n))]
             (println "(factorial (dec n))=" a)
             a))))

(factorial 5)
;;(factorial (dec n))= 1
;;(factorial (dec n))= 2
;;(factorial (dec n))= 6
;;(factorial (dec n))= 24
;;120

;; So now we can watch the stack unwind. This gives us confidence in the inner
;; workings of the function.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The problem with this solution is that I've had to do a fair bit of typing to
;; change the function into a version that prints out its intermediate values.

;; First, let's give n a global value so that we can evaluate fragments out of
;; context:
(def n 5) 

;; Here's the original function again (re-evaluate the definition)
(defn factorial [n]
  (if (< n 2) n
      (* n (factorial (dec n)))))

;; Specifically, what I had to do was change
(factorial (dec n))
;; into
(let [a (factorial (dec n))] (println "(factorial (dec n))=" a) a)

;; Which is an expression which not only evaluates to 24 when n=5 , like the
;; original did, but which prints out (factorial (dec n))= 24 at the same time.

;; Notice that the phrase (factorial (dec n)) has to be repeated in the code.

;; Every time I would like to examine the value returned by an expression as my
;; program runs, I have to make this complicated but mechanical change. Even
;; more annoyingly, I have to tell the compiler the same thing twice.

;; Any time you find that you have to do too much typing and perform mechanical
;; repetitions, you will also find difficulty in reading, and potential for
;; error. It is always to be avoided.

;; As Larry Wall said, the chief virtue of a programmer is laziness.

;; This simple repetitive task should be as easy as changing
(factorial (dec n))
;; to
(dbg (factorial (dec n)))

;; Normally, when one spots a common pattern like this, one makes a function.
;; But a function to do what we want here is problematical, because we need the
;; source code as well as the evaluated value of (factorial (dec n)) We might
;; try something like:
(defn dbgf [s x]
  (println s "=" x)
  x)

;; And use it like this:
(defn factorial [n]
  (if (< n 2) n
      (dbgf "(* n factorial(dec n))" (* n (factorial (dec n))))))
  
;; That's a bit better, but I'm still changing
(* n (factorial (dec n)))
;; into 
(dbgf "(* n factorial(dec n))" (* n (factorial (dec n))))

;; Which is less error prone, but still repetitive.

;; The reason that we need to hold dbgf's hand like this, telling it what to
;; print out in two different ways, is that a function's arguments are evaluated
;; before the function is called.

;; If we want to write
(dbg (* n (factorial (dec n))))
;; and have it work as we intend, then we need to take control of when
;; (* n (factorial (dec n))) is evaluated.

;; And this is the problem that macros solve.

;; We need to work out how to write:
(dbg (* n (factorial (dec n))))
;; And get:
(let [a (factorial (dec n))] (println "(factorial (dec n))=" a) a)
;; Instead.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Generating Code

;; Now because lisp code and lisp data are very similar things, we can easily
;; write a function which will generate the code that we want:

;; Let's define:
(defn dbg-code [s]
  (list 'let ['a s] (list 'println (list 'quote s) "=" 'a) 'a))
;; Which is just a function that takes some code and gives back some code.

;; We can call this function on little pieces of code, to get other little
;; pieces of code

(dbg-code 'x)
;; (let [a x] (println (quote x) "=" a) a)

(dbg-code '(* x x))
;; (let [a (* x x)] (println (quote (* x x)) "=" a) a)

(dbg-code '(* n (factorial (dec n))))
;; (let [a (* n (factorial (dec n)))] (println (quote (* n (factorial (dec n)))) = a) a)

;; Nothing 'macro' has gone on yet! This is just a function, taking advantage of
;; lisp's ability to easily deal with the lists and trees and vectors that make
;; up lisp code.

;; But the function generates exactly the code that we'd like the compiler to
;; substitute in.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Macros

;; Now how shall we turn our code-generating function into a macro?
;; Just change defn to defmacro:

(defmacro dbg-1 [s]
  (list 'let ['a s] (list 'println (list 'quote s) "=" 'a) 'a))

;; Now it's a macro!

;; Let's try it out:
(defn factorial [n]
  (if (< n 2) n
      (dbg-1 (* n (factorial (dec n))))))

(factorial 5)
;; (* n (factorial (dec n))) = 2
;; (* n (factorial (dec n))) = 6
;; (* n (factorial (dec n))) = 24
;; (* n (factorial (dec n))) = 120
;; 120

;; Bingo!

;; When the compiler sees a macro, which is just a function that returns some
;; code, it runs the function, and substitutes the code that is returned into
;; the program.

;; It is like programming the compiler to be an apprentice programmer who will
;; write out the tedious bits longhand for you.

;; We can even ask the compiler what it sees when it expands dbg-1:
(macroexpand-1 '(dbg-1 x))
;; (let [a x] (println (quote x) "=" a) a)
(macroexpand-1 '(dbg-1 (* x x)))
;; (let [a (* x x)] (println (quote (* x x)) "=" a) a)
(macroexpand-1 '(dbg-1 (println x)))
;; (let [a (println x)] (println (quote (println x)) "=" a) a)
(macroexpand-1 '(dbg-1 (inc x)))
;; (let [a (inc x)] (println (quote (inc x)) "=" a) a)
(macroexpand-1 '(dbg-1 (* n (factorial (dec n)))))
;; (let [a (* n (factorial (dec n)))] (println (quote (* n (factorial (dec n)))) "=" a) a)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; So have we won??

;; We have certainly solved our problem as we originally conceived it, but there
;; are potentially a couple of difficulties with our solution, which I shall
;; cover in a later post.

;; Neither are terribly likely to occur in practice, but it is better to write
;; bug-free code than almost-bug-free code.

;; The chief virtues of a programmer are laziness and paranoia.
;; Million to one chances have a way of coming up nine times out of ten.

;; Also, compare our solution with the actual dbg macro:

(defmacro dbg-1 [s]
  (list 'let ['a s] (list 'println (list 'quote s) "=" 'a) 'a))

(defmacro dbg[x] `(let [x# ~x] (println '~x "=" x#) x#))

;; There are obviously similarities, but the complete version looks more like
;; the generated code, and is thus easier to write and to read once you've got
;; the hang of the weird `, #, and ~ things.

;; And the second version also solves the two potential difficulties which I
;; haven't explained yet!


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Things to play with to check you've understood:

;; I. Find the source for clojure.core/when
(clojure.repl/source when)
;; figure out what it does.

;; II. What is wrong with:
(defmacro dbg-oops [s]
  (list 'do (list 'println (list 'quote s) "=" s) s))

;; III. Can you improve dbg so you can write (dbg * 3 2) rather than
;; (dbg (* 3 2)), saving a couple of brackets every time you use it?

;; IV. Using the methods above, can you write a for loop. We'd want
(forloop [i 1 10]
  (print i)
  (print (* i i)))

;; to expand to:
(loop [i 1]
  (when (<= i 10)
    (print i)
    (print (* i i))
    (recur (inc i))))

;; V. (hard) If you managed to do IV above, and write forloop, then congratulations! ;; You have
;; understood this post. But I bet your solution suffers from either or both of
;; the subtle problems that I mentioned above. Can you work out what they are?

;; Hint. How many times does the finish condition get evaluated as you go
;; through the loop? How would you make sure that only happened once? What if a
;; name you'd used in the macro clashed with a name used in the code where the
;; macro is?

;; Both problems are, in fact, easy to solve. See part II.
