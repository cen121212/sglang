python3 -m sglang_router.launch_router \
--pd-disaggregation \
--policy round_robin \
--prefill http://61.47.19.68:8000 8998 \
--decode http://61.47.19.67:8003 \
--host 61.47.19.68 \
--port 6688 \
--prometheus-port 9004 \
--disable-health-check