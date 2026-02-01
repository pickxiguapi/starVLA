#!/bin/bash

# ==================== 2机16卡训练脚本 ====================
# 【NCCL 通信 IP】：必须是 bond1 的内网 IP
MASTER_ADDR="29.160.51.246"
MASTER_PORT=29582

# 【SSH 连接 IP】：用于发送启动命令
NODE0_SSH_IP="29.160.51.246"  # 节点2 (Worker) 的 SSH IP
NODE1_SSH_IP="29.160.42.61"  # 节点3 (Worker) 的 SSH IP

# ==================== 训练配置 ====================
WORK_DIR="/qy4/yyf/starVLA"
CONDA_ENV="starVLA"

# === Training parameters ===
Framework_name=QwenOFT
freeze_module_list=''
date_time=$(date +%m%d_%H%M)
base_vlm=/apdcephfs_hldy/share_304012692/er1/saves/Embodied-R1.5-SFT/20260128
config_yaml=scripts/ER1_5/qwen3vl_libero.yaml
libero_data_root=./playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=/starvla/Checkpoints
run_id=libero4in1_${Framework_name}_2node_${date_time}
batch_size=8
wandb_project=Qwen3VL_libero_all_${Framework_name}_2node

# Multi-node settings
NNODES=2
TOTAL_GPUS=16  # 2 nodes × 8 GPUs

# 日志配置
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# ==================== 自动清理函数 ====================
cleanup() {
    echo -e "\n\033[31m正在终止训练，清理后台进程...\033[0m"
    kill $(jobs -p) 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# ==================== 节点启动函数 ====================
run_node() {
    local NODE_RANK=$1
    local SSH_IP=$2
    local LOG_FILE="${LOG_DIR}/libero_${Framework_name}_${date_time}_node_${NODE_RANK}.log"

    echo "🚀 [Node $NODE_RANK] 正在通过 SSH ($SSH_IP) 启动..."

    CMD="
    source ~/.bashrc && \
    conda activate $CONDA_ENV && \
    cd $WORK_DIR && \

    export http_proxy=http://star-proxy.oa.com:3128 && \
    export https_proxy=http://star-proxy.oa.com:3128 && \
    export ftp_proxy=http://star-proxy.oa.com:3128 && \
    export no_proxy=.woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com && \

    export NCCL_IB_GID_INDEX=3 && \
    export NCCL_IB_SL=3 && \
    export NCCL_CHECK_DISABLE=1 && \
    export NCCL_P2P_DISABLE=0 && \
    export NCCL_IB_DISABLE=0 && \
    export NCCL_LL_THRESHOLD=16384 && \
    export NCCL_IB_CUDA_SUPPORT=1 && \
    export NCCL_SOCKET_IFNAME=bond1 && \
    export UCX_NET_DEVICES=bond1 && \
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 && \
    export NCCL_COLLNET_ENABLE=0 && \
    export SHARP_COLL_ENABLE_SAT=0 && \
    export NCCL_NET_GDR_LEVEL=2 && \
    export NCCL_IB_QPS_PER_CONNECTION=4 && \
    export NCCL_IB_TC=160 && \
    export NCCL_PXN_DISABLE=0 && \
    export NCCL_NVLS_ENABLE=0 && \
    export NCCL_PROFILE_PRIMS_ENABLE=1 && \
    export NCCL_DEBUG=INFO && \
    export NCCL_TIMEOUT=18000000 && \

    export WANDB_MODE=online && \

    echo \"[Node $NODE_RANK] Starting training...\" && \

    accelerate launch \
      --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --machine_rank $NODE_RANK \
      --num_machines $NNODES \
      --num_processes $TOTAL_GPUS \
      starVLA/training/train_starvla.py \
      --config_yaml ${config_yaml} \
      --framework.name ${Framework_name} \
      --framework.qwenvl.base_vlm ${base_vlm} \
      --datasets.vla_data.data_root_dir ${libero_data_root} \
      --datasets.vla_data.data_mix ${data_mix} \
      --datasets.vla_data.per_device_batch_size ${batch_size} \
      --trainer.vla_data.video_backend torchvision_av \
      --trainer.freeze_modules ${freeze_module_list} \
      --trainer.max_train_steps 80000 \
      --trainer.save_interval 10000 \
      --trainer.logging_frequency 100 \
      --trainer.eval_interval 100 \
      --run_root_dir ${run_root_dir} \
      --run_id ${run_id} \
      --wandb_project ${wandb_project}
    "

    ssh $SSH_IP "$CMD" > "$LOG_FILE" 2>&1 &
}

# ==================== 正式运行 ====================
echo "=========================================="
echo "🚀 2机16卡多节点训练启动"
echo "Master IP (NCCL): $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Total Nodes: $NNODES"
echo "Total GPUs: $TOTAL_GPUS"
echo "Run ID: $run_id"
echo "=========================================="

# 启动节点 0 (Master)
run_node 0 "$NODE0_SSH_IP"

# 启动节点 1 (Worker)
run_node 1 "$NODE1_SSH_IP"

# ==================== 日志实时显示 ====================
echo "✅ 任务已后台提交，等待日志生成..."
sleep 5

echo "=========================================="
echo "📺 正在实时监控所有节点日志..."
echo "👉 按 Ctrl+C 可停止监控并终止训练"
echo "=========================================="
echo ""

tail -f ${LOG_DIR}/libero_${Framework_name}_${date_time}_node_0.log \
        ${LOG_DIR}/libero_${Framework_name}_${date_time}_node_1.log &
wait
