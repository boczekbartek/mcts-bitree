#!/usr/bin/env bash

for c in $(seq 1 40); do
    for i in $(seq 1 15); do
        >&2 echo "C=$c, i=$i"
        python main.py --n-rollouts 1000 -mls 10000 --tree-depth 12 -c$c 2>/dev/null 
    done
done
done \
| tee -a $0.log
