Framework_name=QwenPI
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # must be a local path, due to simpler will run in other where

freeze_module_list='' # just for fast debug, sota is under fully FT, e.g., freeze_module_list=""

oxe_data_root=./playground/Datasets/
data_mix=bridge
run_root_dir=./playground/Checkpoints
run_id=1019_starvla_qwenpi_bridge

export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --main_process_port 29503 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/simpler_qwenPI.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 15000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \


