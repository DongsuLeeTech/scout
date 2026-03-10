algo=${algo:-mafql}
env=${env:-mamujoco}
source=${source:-omiga}
scenarios=${scenarios:-"6halfcheetah"}
datasets=${datasets:-"Medium"}
seed_start=${seed_start:-0}
seed_max=${seed_max:-3}
gpu=${gpu:-1}

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
