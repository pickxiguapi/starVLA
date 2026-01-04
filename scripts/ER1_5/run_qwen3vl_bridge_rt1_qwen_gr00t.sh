Framework_name=QwenGR00T
dit_type="DiT-B"
freeze_module_list='' # fully FT, e.g., freeze_module_list=""
data_mix=bridge_rt_1

## Modify below paths before running ##
date_time=$(date +%m%d_%H%M)
base_vlm=/apdcephfs_hldy/share_304012692/er1/Qwen3-VL-8B-Instruct # local path of VLM
data_root_dir=./playground/Datasets/ # local path of dataset root
run_root_dir=/apdcephfs_hldy/share_304012692/Checkpoints # output root path
run_id=qwen3vl_bridge_rt1_gr00t_${date_time} # run id
batch_size=32
wandb_project=Qwen3VL_Bridge_RT1_GR00T
wandb_entity=your_wandb_entity # set your wandb entity here

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


accelerate launch \
  --main_process_port 29503 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen3vl_bridge_rt1_gr00t.yaml \
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
