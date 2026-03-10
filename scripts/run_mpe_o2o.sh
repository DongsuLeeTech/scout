algo=${algo:-madflow}
env=${env:-mpe}
source=${source:-omar}
scenarios=${scenarios:-"simple_spread"}
datasets=${datasets:-"Expert"}
seed_start=${seed_start:-15}
seed_max=${seed_max:-19}
gpu=${gpu:-3}

for scenario in $scenarios; do
  for dataset in $datasets; do
    for seed in $(seq "$seed_start" "$seed_max"); do
      echo "Running seed=$seed scenario=$scenario dataset=$dataset agent=$algo on gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" \
      python continuous_o2o_main.py \
        --agent_name "$algo" \
        --env "$env" \
        --source "$source" \
        --scenario "$scenario" \
        --dataset "$dataset" \
        --seed "$seed"
      echo
      sleep 1
    done
  done
done
