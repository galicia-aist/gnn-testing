#!/bin/bash

eval "$(/mnt/miniconda3/bin/conda shell.bash hook)"
conda activate graph-pooling-v1

# Loop-based script to run all experiments

# ====== CONFIGURATION ======
experiment_type="bag_ratios"   # options: "all_datasets" or "bag_ratios"

models=("gcn" "gat" "sage")
pools=("default")
datasets=("coraml" "citeseer" "photo" "actor")
bag_ratios=("0.9" "0.8" "0.7" "0.6")

class_type="node"
mode=3
target_dataset="coraml"  # used only for bag ratio experiments
# ===========================


if [ "$experiment_type" == "all_datasets" ]; then
    echo "=== Running ALL DATASETS experiment ==="
    for d in "${datasets[@]}"; do
        echo "===== Dataset: $d ====="
        for pool in "${pools[@]}"; do
            for model in "${models[@]}"; do
                echo "Running: model=$model, pool=$pool, dataset=$d"
                python main.py --cuda --model="$model" \
                    --log_level=debug \
                    --pool="$pool" \
                    --d="$d" \
                    --mode="$mode" \
                    --class_type="$class_type"
            done
        done
    done

elif [ "$experiment_type" == "bag_ratios" ]; then
    echo "=== Running BAG RATIO experiment for ALL DATASETS ==="
    for d in "${datasets[@]}"; do
        echo "===== Dataset: $d ====="
        for ratio in "${bag_ratios[@]}"; do
            echo "---- Bag ratio: $ratio ----"
            for pool in "${pools[@]}"; do
                for model in "${models[@]}"; do
                    echo "Running: model=$model, pool=$pool, dataset=$d, bag_ratio=$ratio"
                    python main.py --model="$model" \
                        --log_level=debug \
                        --pool="$pool" \
                        --d="$d" \
                        --mode="$mode" \
                        --class_type="$class_type" \
                        --bag_ratio="$ratio"
                done
            done
        done
    done

else
    echo "Error: Unknown experiment_type '$experiment_type'. Use 'all_datasets' or 'bag_ratios'."
    exit 1
fi
