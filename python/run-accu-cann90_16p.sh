#!/bin/bash
# GLM-5.2-w8a8 16P two-node PD Mix launch script
# Based on test_npu_glm_5_2_w8a8_16p_gpqa.py configuration

# cpu high performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Model path
MODEL_PATH=/home/weights/GLM-5.2-w8a8

# Node IPs (index 0 = master)
NODE_IPS=('61.47.19.68' '61.47.19.70')
MASTER_ADDR="${NODE_IPS[0]}:5000"
SERVICE_PORT=6677

# Detect local IP
LOCAL_HOST1=$(hostname -I | awk -F ' ' '{print $1}')
LOCAL_HOST2=$(hostname -I | awk -F ' ' '{print $2}')
echo "LOCAL_HOST1=${LOCAL_HOST1}"
echo "LOCAL_HOST2=${LOCAL_HOST2}"

# Auto-detect NIC name (mirrors get_nic_name() logic)
detect_nic() {
    local exclude="lo docker tunl cali veth br- virbr eth0@if kube- flannel weave cilium"
    local best_nic="" best_bytes=0
    while read -r line; do
        ifname=$(echo "$line" | awk -F: '{gsub(/^[ \t]+/,"",$1); print $1}')
        rx=$(echo "$line" | awk '{print $2}')
        tx=$(echo "$line" | awk '{print $10}')
        total=$((rx + tx))
        local skip=false
        for prefix in $exclude; do
            case "$ifname" in
                "$prefix"*) skip=true; break ;;
            esac
        done
        if [ "$skip" = false ] && [ "$total" -gt "$best_bytes" ]; then
            best_bytes=$total
            best_nic="$ifname"
        fi
    done < <(tail -n +3 /proc/net/dev)
    echo "${best_nic:-lo}"
}
NIC_NAME=$(detect_nic)
echo "Detected NIC: $NIC_NAME"

# Environment variables (from test case GLM_5_2_W8A8_16P_TWO_NODE_ENVS)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEPEP_HCCL_BUFFSIZE=2500
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export DEEPEP_HYBRID_DEPLOYMENT=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export HCCL_SOCKET_IFNAME=$NIC_NAME
export GLOO_SOCKET_IFNAME=$NIC_NAME

# Launch on matched node
for i in "${!NODE_IPS[@]}"; do
    if [[ "$LOCAL_HOST1" == "${NODE_IPS[$i]}" || "$LOCAL_HOST2" == "${NODE_IPS[$i]}" ]]; then
        echo "Starting node $i on ${NODE_IPS[$i]}"
        python3 -m sglang.launch_server \
            --model-path $MODEL_PATH \
            --attention-backend ascend \
            --device npu \
            --tp-size 32 \
            --nnodes 2 \
            --dp-size 8 \
            --enable-dp-attention \
            --chunked-prefill-size 65536 \
            --max-prefill-tokens 280000 \
            --trust-remote-code \
            --mem-fraction-static 0.76 \
            --context-length 135000 \
            --served-model-name glm-5.2-w8a8 \
            --cuda-graph-max-bs-decode 4 \
            --max-running-requests 32 \
            --quantization modelslim \
            --moe-a2a-backend deepep \
            --deepep-mode auto \
            --disable-shared-experts-fusion \
            --load-balance-method round_robin \
            --reasoning-parser glm45 \
            --tool-call-parser glm47 \
            --enable-metrics \
            --speculative-algorithm NEXTN \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4 \
            --dist-init-addr $MASTER_ADDR \
            --node-rank $i \
            --host ${NODE_IPS[$i]} \
            --port $SERVICE_PORT
        NODE_RANK=$i
        break
    fi
done
