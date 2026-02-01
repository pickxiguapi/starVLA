Framework_name=QwenOFT
dit_type="DiT-B"
freeze_module_list='' # fully FT, e.g., freeze_module_list=""
data_mix=bridge_rt_1

## Modify below paths before running ##
date_time=$(date +%m%d_%H%M)
config_yaml=scripts/ER1_5/qwen3vl_bridge_rt1_oft.yaml
base_vlm=/apdcephfs_hldy/share_304012692/er1/saves/Embodied-R1.5-SFT/20260128
data_root_dir=./playground/Datasets/OXE_LEROBOT # local path of dataset root
run_root_dir=/apdcephfs_hldy/share_304012692/er1/starvla/Checkpoints # output root path
run_id=qwen3vl_bridge_rt1_oft_${date_time} # run id
batch_size=8
wandb_project=Qwen3VL_Bridge_RT1_OFT

export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


accelerate launch \
  --main_process_port 29503 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root_dir}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 200 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --output_dir ${output_dir} \
  --wandb.project ${wandb_project} \
  --wandb.entity ${wandb_entity}


# multi-node launch example

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
#   --framework.framework_py QwenGR00T \
#   --framework.qwenvl.base_vlm microsoft/Florence-2-large \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project your_project \
#   --wandb_entity your_name
