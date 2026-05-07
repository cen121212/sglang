"""
使用方法：
1.在sglang的容器里直接 pip install ai_infra_bench, 然后改下BASE_URL，然后就可以直接跑了

其他
2.  os.environ["SHAREGPT_DATASET"]="xxx" 这里如果找不到数据集的话，就会默认用去huggingface下，默认用 ShareGPT_V3_unfiltered_cleaned_split.json
3.  --tokenizer deepseek-ai/DeepSeek-V3.2  这个也会去huggingface下，如果网络不通，可以改为本地路径
"""

import os
import math

from ai_infra_bench import client_gen

os.environ["BASE_URL"] = "http://61.47.19.68:6689"
os.environ["SHAREGPT_DATASET"]="/home/luochen/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

# input args
base_url = os.environ["BASE_URL"]
dataset_path = os.environ["SHAREGPT_DATASET"]
input_features = [
    "random_input_len",
    "random_output_len",
    "request_rate",
    "max_concurrency",
]
output_metrics = [
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "output_throughput",
    "completed"
]

# construct client requests
client_template = """
python -m sglang.bench_serving
        --base-url {base_url}
                --backend sglang-oai
        --tokenizer /home/weights/GLM-5.1-w4a8
        --model /home/weights/GLM-5.1-w4a8
                --dataset-path {dataset_path}
        --flush-cache
                --dataset-name random
                --random-range-ratio 1
                --random-input-len {input_len}
                --random-output-len {output_len}
                --request-rate {request_rate}
                --max-concurrency {max_concurrency}
                --num-prompt {num_prompt}
"""
rate_lists = [1, 4, 8, 12, 16, 24, 36, 48, 64]
rate_lists2 = [48, 64, 80]
rate_lists4 = [32, 48, 64]
rate_lists16 = [8, 12, 16]
rate_lists50 = [36]

# ds3.2
sg_rate_lists50 = [24]

# ds3.2 + vllm
vllm_rate_lists50 = [24]

client_cmds = [
    *[
        client_template.format(
            base_url=base_url,
            input_len=2048,
            output_len=256,
            dataset_path=dataset_path,
            request_rate=16,
            max_concurrency=rate,
            num_prompt=min(max(rate * 10, 80), 500),  # clip to [80, 250]
        )
        for rate in rate_lists2
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=4096,
            output_len=256,
            dataset_path=dataset_path,
            request_rate=16,
            max_concurrency=rate,
            num_prompt=min(max(rate * 10, 80), 500),  # clip to [80, 250]
        )
        for rate in rate_lists4
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=16384,
            output_len=256,
            dataset_path=dataset_path,
            request_rate=16,
            max_concurrency=rate,
            num_prompt=min(max(rate * 10, 80), 500),  # clip to [80, 250]
        )
        for rate in rate_lists16
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=51200,
            output_len=256,
            dataset_path=dataset_path,
            request_rate=16,
            max_concurrency=rate,
            num_prompt=min(max(rate * 10, 80), 500),  # clip to [80, 250]
        )
        for rate in rate_lists50
    ],
    *[
        client_template.format(
            base_url=base_url,
            input_len=102400,
            output_len=256,
            dataset_path=dataset_path,
            request_rate=16,
            max_concurrency=rate,
            num_prompt=min(max(rate * 10, 80), 500),  # clip to [80, 250]
        )
        for rate in rate_lists50
    ],
]


if __name__ == "__main__":
    client_gen(
        client_cmds=client_cmds,
        input_features=input_features,
        output_metrics=output_metrics,
        server_label="glm5.1",
        n=1,
        only_last=True,
        output_dir="glm5-outputs_all",
    )