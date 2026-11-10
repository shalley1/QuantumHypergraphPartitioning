#!/bin/zsh


for script in fair_min_congress_sm.py min_congress_sm.py max_congress_sm.py fair_max_congress_sm.py; do
    echo "================================"
    echo "Running $script"
    echo "================================"

    python -u "$script"

    echo "Finished $script"
done

echo "All experiments completed!"
