
#!/bin/zsh

mkdir -p logs results

for loss in max min fair_max fair_min; do
    echo "Starting Poisson | Loss: $loss"

    python -u qaoa.py \
        --dataset poisson \
        --runtype aon \
        --loss_type "$loss" \
        --mixer_graph complete \
        --num_layers 3 \
        --max_trials 300 \
        --patience 30 \
        --min_delta 1e-4 \
        --output "results/poisson_${loss}_p3.csv" \
        > "logs/poisson_${loss}_p6.txt" 2>&1 &
done

wait

echo "All Poisson experiments finished!"
