Framework_name=QwenGR00T
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct # must be a local path, due to simpler will run in other where

freeze_module_list='' # just for fast debug, sota is under fully FT, e.g., freeze_module_list=""

oxe_data_root=./playground/Datasets/
data_mix=bridge
run_root_dir=./playground/Checkpoints
run_id=1023_qwen3vl_bridge_gr00t

export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/


accelerate launch \
  --main_process_port 29503 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen3vl_bridge_gr00t.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 24 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 200 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --output_dir ${output_dir}


