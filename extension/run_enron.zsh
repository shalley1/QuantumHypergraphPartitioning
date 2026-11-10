#!/bin/zsh


for script in fair_min_enron.py min_enron.py max_enron.py; do
    echo "================================"
    echo "Running $script"
    echo "================================"

    python -u "$script"

    echo "Finished $script"
done

echo "All experiments completed!"
