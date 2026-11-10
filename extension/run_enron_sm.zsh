
#!/bin/zsh


for script in fair_min_enron_sm.py min_enron_sm.py max_enron_sm.py fair_max_enron_sm.py; do
    echo "================================"
    echo "Running $script"
    echo "================================"

    python -u "$script"

    echo "Finished $script"
done

echo "All experiments completed!"
