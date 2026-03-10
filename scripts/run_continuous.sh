algo=${algo:-macflow_vgf}
env=${env:-mamujoco}
source=${source:-omiga}
scenarios=${scenarios:-"6halfcheetah"}
datasets=${datasets:-"Medium-Replay"}
seed_start=${seed_start:-0}
seed_max=${seed_max:-7}
gpu=${gpu:-7}
dir="/datastor1/dongsu"

for scenario in $scenarios; do
  for dataset in $datasets; do
    for seed in $(seq "$seed_start" "$seed_max"); do
      echo "Running seed=$seed scenario=$scenario dataset=$dataset agent=$algo on gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" \
      python continuous_main.py \
        --project_name 'VGF-MAC' \
        --agent_name "$algo" \
        --env "$env" \
        --source "$source" \
        --scenario "$scenario" \
        --dataset "$dataset" \
        --seed "$seed" \
        --data_dir "$dir" \
        --project_name "VGF-MAC"
      echo
      sleep 1
    done
  done
done
