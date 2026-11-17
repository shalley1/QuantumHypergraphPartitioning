
#!/bin/zsh

mkdir -p logs results

for loss in max min fair_max fair_min; do
    echo "Starting Enron | Loss: $loss"

    python -u qaoa.py \
        --dataset enron \
        --runtype aon \
        --loss_type "$loss" \
        --mixer_graph complete \
        --num_layers 6 \
        --max_trials 600 \
        --patience 50 \
        --min_delta 1e-3 \
        --output "results/enron_${loss}_p6.csv" \
        > "logs/enron_${loss}_p6.txt" 2>&1 &
done

wait

echo "All Enron experiments finished!"
