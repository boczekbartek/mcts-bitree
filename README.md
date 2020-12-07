# mcts-bitree
An implementation of Monte Carlo Tree Search algorithm for binary trees in Python

## Prerequisites
* Python > 3.7

## Installation
```bash
python -m pip install -r requirements.txt
```

## Description
Construct a binary tree (each node has two child nodes) of depth d = 12 (or more – if you’re
feeling lucky) and assign different values to each of the 2^d leaf-nodes. Specifically, pick the leaf
values to be real numbers randomly distributed between 0 and 100 (use the uniform continuous
distribution U(0, 100), so don’t restrict yourself to integer values!).

* Implement the MCTS algorithm and apply it to the above tree to search for the optimal
(i.e. highest) value.
* Collect statistics on the performance and discuss the role of the hyperparameter c in the
UCB-score.

Assume that the number MCTS-iterations starting in a specific root node is limited (e.g. to 10
or 50). Make a similar assumption for the number of roll-outs starting in a particular (”snowcap”)
leaf node (e.g. 1 or 5).
