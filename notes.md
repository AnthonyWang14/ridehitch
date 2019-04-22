# some problems need to be considered


## the detour when a driver responses more than one request

## what the graph looks like if follows the distribution

## deal with the terminated state

RANK has similar performance as RANDOM.
FIRST is the best one.


#references

##Online stochastic weighted matching: Improved approximation algorithms WINE2011
First online stochastic weighted bipartite matching
##Online Stochastic Matching: Online Actions Based on Offline Statistics 2011
Vahideh H. Manshadi,
Shayan Oveis Gharan,
Amin Saberi†
###setting
iid-known distribution, one-sided online matching
###positive result
0.702 for arbitrary rates, and 0.705 for the i.i.d. model
###hardness result
an upper bound of 0.823 on the competitive ratio of any 
deterministic or randomized online algorithm in the known distribution model
###others
non-adaptive/adaptive definition,
Since f is a fractional matching, standard algorithmic versions of Caratheodory’s theorem (see
e.g. [Geometric Algorithms and Combinatorial Optimization, Theorem 6.5.11]) say that, in polynomial time, we can decompose a feasible solution in the
bipartite matching polytope into a convex combination of polynomially many bipartite matchings.



##Assigning Tasks to Workers based on Historical Data: Online Task Assignment with Two-sided Arrivals 2018
###setting
iid-known distribution(learn from data),
two-sided online learning,
vertex-weighted,
edge-weighted,
two stage birth-death process: k(1)
###algorithm
Non-adaptive algorithm 0.295(k(1)) for edge-weighted,
Greedy 0.295 for unweighted,
Adaptive algorithm 0.32 for node-weighted
###hardness result
no non-adaptive can > k(1),
no adaptive can > 1/(e-1)=0.581
###experiment
dataset: gMission and EverySender,
the result is useless I think.
###others
two stage birth-death process,
LP relaxation technique for stochastic matching


##Online Stochastic Matching: New Algorithms and Bounds 2017
Brian Brubach†
, Karthik A. Sankararaman‡
, Aravind Srinivasan§
, and Pan Xu¶
###setting
integral arrival rates,
vertex-weighted,
edge-weighted
###algorithm




