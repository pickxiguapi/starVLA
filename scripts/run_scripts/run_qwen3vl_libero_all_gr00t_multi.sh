

config_yaml=./starVLA/config/training/qwen3vl_libero_gr00t.yaml
Framework_name=QwenGR00T
base_vlm=/apdcephfs/share_303838591/hunyuan/iffyuan/models/Qwen3-VL-4B-Instruct # must be a local path, due to simpler will run in other where
freeze_module_list='' # just for fast debug, sota is under fully FT, e.g., freeze_module_list=""
libero_data_root=/qy4/datasets/  
run_root_dir=./playground/Checkpoints
run_id=1023_qwen3vl_libero_all_gr00t
data_mix=libero_all

export WANDB_MODE=online
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

export MASTER_IP=29.232.225.212
export MASTER_PORT=29500
export MACHINE_RANK=0

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_multi.yaml \
  --num_machines 2 \
  --num_processes 16 \
  --machine_rank ${MACHINE_RANK:-0} \
  --main_process_ip ${MASTER_IP} \
  --main_process_port ${MASTER_PORT:-29500} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 15000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --output_dir ${output_dir}

