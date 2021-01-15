#!/usr/bin/env bash

for roll in $(seq 1 9); do
    for i in $(seq 1 15); do
        >&2 echo "C=$c, i=$i"
        python main-mcts.py --n-rollouts 1000 -mls 100 --tree-depth 12 -c 1.$c 2>/dev/null 
    done
done \
| tee -a rollouts.log