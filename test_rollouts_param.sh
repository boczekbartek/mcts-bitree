#!/usr/bin/env bash

for roll in $(seq 100 1000 30000); do
    for i in $(seq 1 5); do
        >&2 echo "C=$c, i=$i"
        python main-mcts.py --n-rollouts $roll -mls 10000 --tree-depth 12 -c 13 2>/dev/null 
    done
done \
| tee -a rollouts3.log
