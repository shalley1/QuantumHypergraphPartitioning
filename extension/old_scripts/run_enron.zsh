#!/bin/zsh

mkdir -p logs results

for loss in max min fair_max fair_min; do
    echo "Starting Enron | Loss: $loss"

    python -u qaoa.py \
        --dataset enron \
        --runtype aon \
        --loss_type "$loss" \
        --mixer_graph complete \
        --num_layers 5 \
        --max_trials 600 \
        --patience 100 \
        --min_delta 1e-5 \
        --output "results/enron_${loss}_p5.csv" \
        > "logs/enron_${loss}_p5.txt" 2>&1 &
done

wait

echo "All Enron experiments finished!"
