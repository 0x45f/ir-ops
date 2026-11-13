#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
rm -rf ./perf
mkdir -p ./perf
CUDA_VISIBLE_DEVICES=1 /data/wangzhen/miniconda3/envs/ir-ops/bin/python ./triton_softmax.py
