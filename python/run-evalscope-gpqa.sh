#!/bin/bash
# GLM-5.2-w4a8 GPQA Diamond accuracy evaluation via evalscope CLI
# Based on test_npu_glm_5_2_w4a8_16p_gpqa.py

HOST=61.47.19.68
PORT=6677
MODEL_NAME=glm-5.2-w4a8

evalscope eval \
    --model $MODEL_NAME \
    --api-url http://${HOST}:${PORT}/v1/chat/completions \
    --eval-type openai_api \
    --datasets gpqa_diamond \
    --eval-batch-size 32 \
    --generation-config '{"max_tokens": 65536, "temperature": 1.0, "timeout": 1200, "stream": true}' \
    --limit 100000 \
    --stream \
    --work-dir ./evalscope_result_gpqa
