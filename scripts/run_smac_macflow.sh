algo=${algo:-macflow}
env=${env:-smac_v2}
source=${source:-og_marl}
scenarios=${scenarios:-"terran_10_vs_10" }
datasets=${datasets:-"Replay"}
seed_start=${seed_start:-0}
seed_max=${seed_max:-9}
gpu=${gpu:-1}
dir="/datastor1/dongsu"

for scenario in $scenarios; do
  for dataset in $datasets; do
    for seed in $(seq "$seed_start" "$seed_max"); do
      echo "Running seed=$seed scenario=$scenario dataset=$dataset agent=$algo on gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" \
      python smac_macflow.py \
        --project_name 'VGF-MAC' \
        --agent_name "$algo" \
        --env "$env" \
        --source "$source" \
        --scenario "$scenario" \
        --dataset "$dataset" \
        --seed "$seed" \
        --data_dir "$dir"
      echo
      sleep 1
    done
  done
done