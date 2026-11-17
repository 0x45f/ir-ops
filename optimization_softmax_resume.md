# Softmax 优化恢复记录（2026-11-17）

目标仍为全部 98 个 shape 的完整 API 调用几何平均加速比 >1.6，沿用原计时、精度和回归验收要求。

## 本轮核验

- `perf_llir_round2/summary.json`：API 几何平均 1.359663209107442×，kernel 几何平均 1.4647074753085638×，98 个 shape，无显著回归，`accepted=false`。
- 使用 `/data/wangzhen/miniconda3/envs/ir-ops/bin/python` 逐项核对正式结果保存的运行时文件 SHA256：无不一致。
- `triton_softmax.py` 与 Git HEAD 原始文件字节一致。
- 上一轮状态说明未产生新优化；此次恢复重新核验了外部运行条件，发现 GPU 已不可用。

## 当前外部阻碍

`nvidia-smi` 报告无法连接 NVIDIA 驱动。上述 Python 环境中的 PyTorch 2.12.0+cu130 返回 `cuda_available=False`、`cuda_device_count=0`；显式 `torch.cuda.init()` 报 CUDA error 304（OS call failed or operation not supported）。`/dev/nvidia*` 仅发现 `/dev/nvidia-caps`。

因此当前无法执行数值验证、编译加载或性能测量。现有性能数字是先前正式测量结果，不是当前环境重新测得。没有修改生产 IR，也没有将未经测量的候选合入。

这是恢复后的首次 GPU 不可用核验，目标保持 active、未完成。下一步需要执行环境恢复 GPU 设备与驱动访问；恢复后先核验 CUDA 初始化，再继续原范围的实验。未尝试重启共享主机或更改驱动。
