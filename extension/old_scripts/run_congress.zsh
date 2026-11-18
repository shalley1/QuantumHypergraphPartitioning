##!/bin/zsh

mkdir -p logs

for script in fair_min_congress.py min_congress.py max_congress.py fair_max_congress.py; do
    echo "================================"
    echo "Starting $script"
    echo "================================"

    python -u "$script" > "logs/${script%.py}.txt" 2>&1 &
done

wait

echo "All experiments completed!"#!/bin/zsh

