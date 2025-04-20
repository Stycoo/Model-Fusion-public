#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download Qwen/Qwen2.5-Math-7B-Instruct   \
--token "" \
--local-dir /GLOBALFS/gznwp_3/qxj/yangzy/models/Qwen2.5-Math-7B-Instruct \
--exclude "*.pth" "*.pt" "consolidated*"  \
--cache-dir /GLOBALFS/gznwp_3/qxj/yangzy/models/hugging_cache \
--local-dir-use-symlinks False
