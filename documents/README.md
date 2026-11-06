# RWKV-LM-V7 环境配置

先完成本文的环境配置，再按 [数据处理与微调教程](./数据处理与微调.md) 操作。教程包含示例数据、预处理、2.9B/13.3B 微调、评估、续训和导出命令。所有命令在 **Bash、仓库根目录** 执行；`documents` 就是本仓库使用的目录名。

## 1. 这份依赖基于什么环境


| 项目 | 实际读取的版本或配置 |
| --- | --- |
| 操作系统目标 | Linux x86_64，NVIDIA CUDA |
| Python | 3.12.11 |
| PyTorch | 2.7.1+cu128，`torch.version.cuda == '12.8'` |
| PyTorch Lightning | **1.9.5** |
| DeepSpeed | 0.17.1 |
| CUDA 编译器 nvcc | 12.8.93 |
| conda GCC / G++ | 13.4.0 |
| NumPy / Jinja2 / Ninja | 2.3.0 / 3.1.6 / 1.11.1.4 |
| wandb / torchmetrics | 0.20.1 / 1.7.2 |
| 本机硬件参照 | 8 张 H800，单卡 81559 MiB，驱动 610.57.04 |



[`requirements.txt`](./requirements.txt) 固定了本仓库数据处理、训练、导出、参考推理及 CPU 测试所需的 71 个直接和间接依赖，版本来自实机环境。未把环境里无关的 `datasets`、`transformers`、`rwkv-fla`、`torchvision`、`torchaudio` 等一并塞入；当前主训练路径使用仓库自带的 tokenizer 和 CUDA 扩展。

[`environment.yml`](./environment.yml) 提供 Python 和编译工具层，**不含本机绝对路径，也不自动安装 pip 依赖**。根目录旧 `environment.yml` 是另一份环境快照，含本机 `prefix`；本教程统一使用 `documents/` 内的文件。

## 2. 机器准备

需要先安装 Miniconda/Anaconda、Git 和正常工作的 NVIDIA 驱动。以下教程面向 NVIDIA Linux 服务器；仓库虽有 `hip/`，本文依赖和 fused masked CE 配置没有覆盖 AMD/ROCm。

```bash
git clone --branch new_sft https://github.com/TimeMobius/RWKV-LM-V7.git
cd RWKV-LM-V7
```

如果拿到的是完整代码压缩包，解压后进入含 `train.py` 的目录即可。远程分支可能尚未包含本文所引用的本地新增文件，交付代码时需一起带上 `documents/`、`scripts/setup_rwkv_cuda_env.sh`、检查工具及 `run_3b_*.sh`。用下面的命令确认主要入口齐全：

```bash
ls train.py data/make_sft_binidx.py rwkv_vocab_v20260603.txt \
  data/SFT/sample/chat_template.jinja \
  scripts/setup_rwkv_cuda_env.sh scripts/check_rwkv_checkpoint.py \
  scripts/check_sft_mask.py run_3b_sft_zero3_offload.sh \
  run_13b_sft_zero3_offload.sh documents/requirements.txt
nvidia-smi
```

训练使用 BF16 和现场编译的 CUDA 扩展，不能用 CPU 版 Torch 代替。驱动需支持 CUDA 12.8；本机的 610.57.04 是参照版本，不要求逐字相同。`nvidia-smi` 的 CUDA Version 表示驱动支持能力，**不代表已经安装 nvcc**。

ZeRO-3 offload 会使用大量 CPU 内存，checkpoint 还含优化器分片。13.3B 的 BF16 权重本身约 26.5 GB，FP32 权重约 53 GB，实际训练/转换峰值更高；按模型、卡数和 batch 留出主存与磁盘空间。新机器先跑短 profile，不直接照搬 86016/192512 的长上下文。

## 3. 创建环境并安装依赖

### 3.1 创建 Python / CUDA / C++ 环境

在尚未创建同名环境的新机器上执行：

```bash
conda env create -f documents/environment.yml
conda activate rwkv-lm-v7

python -V
command -v python
nvcc --version
command -v c++
```

环境已存在时先激活并检查版本，不必重复创建。也可以创建不同名字的环境：

```bash
conda env create -n rwkv-v7-user -f documents/environment.yml
conda activate rwkv-v7-user
```

### 3.2 先安装 Torch，再安装其余依赖

DeepSpeed 的构建过程可能检查 Torch，因此采用分两步的安装顺序。`DS_BUILD_OPS=0` 仅跳过安装时的算子预编译；运行训练时仍会按需编译 DeepSpeedCPUAdam/FusedAdam 和本仓库 CUDA 扩展。

```bash
python -m pip install pip==25.1.1 setuptools==80.9.0 wheel==0.45.1

python -m pip install torch==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

DS_BUILD_OPS=0 python -m pip install --no-build-isolation \
  -r documents/requirements.txt

python -m pip check
```

`requirements.txt` 中也包含 CUDA 12.8 的 Torch 下载源。不要再无版本限制地安装或升级根目录 `requirements.txt`，否则会改变这里固定的组合。特别是 `train.py` 使用 Lightning 1.x 的 `Trainer.add_argparse_args` 和 `Trainer.from_argparse_args`，升级到 Lightning 2.x 不能直接运行。

### 3.3 明确指定本次使用的解释器

每次开新终端执行，后续数据处理和训练都使用它：

```bash
conda activate rwkv-lm-v7
export PYTHON_BIN="$(command -v python)"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export MAX_JOBS=4
export RWKV_CLEAN_STALE_EXTENSION_LOCKS=0
source scripts/setup_rwkv_cuda_env.sh

"$PYTHON_BIN" -c 'import sys; print(sys.executable)'
```

`PYTHON_BIN` 能覆盖启动脚本中对作者机器环境路径的优先选择。初始化脚本会补齐 conda CUDA、`targets/x86_64-linux`、pip `nvidia/*` 的 header/lib 路径，避免找不到 `cuda_runtime.h`、`curand.h` 或 `-lcublas` 等。

上面将自动清理旧编译锁关闭，避免多任务共用缓存时相互影响。首次启动会出现 Ninja/nvcc 的编译输出，完成后才进入训练；同一环境后续会复用缓存。

### 3.4 检查包、GPU 与编译器

```bash
"$PYTHON_BIN" - <<'PY'
import sys
import importlib.metadata as metadata
import torch

print("Python:", sys.executable, sys.version.split()[0])
for package in ("torch", "pytorch-lightning", "deepspeed", "numpy", "jinja2", "ninja"):
    print(package, metadata.version(package))
print("Torch CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "Torch 无法使用 GPU，请先检查驱动或容器 GPU 挂载"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "capability", torch.cuda.get_device_capability(i), "GiB", round(p.total_memory / 2**30, 1))
assert torch.cuda.is_bf16_supported(), "当前训练配置要求 BF16"
PY

nvcc --version
"$PYTHON_BIN" -c 'import pytorch_lightning, deepspeed; print(pytorch_lightning.__version__, deepspeed.__version__)'
```

启动脚本默认 `TORCH_CUDA_ARCH_LIST=9.0`，对应 H100/H800。其他 GPU 必须按实际计算能力设置，例如 A100 为 `8.0`、RTX 4090 为 `8.9`；能编译不代表该卡有足够显存跑相同模型和上下文。

```bash
export TORCH_CUDA_ARCH_LIST="$("$PYTHON_BIN" -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')"
echo "$TORCH_CUDA_ARCH_LIST"
```

## 4. 如何判断环境已经可用

先按教程跑 [示例预处理](./数据处理与微调.md#3-先用示例数据验证预处理)，它不需要 GPU。随后准备真实模型，执行教程里的 2-step profile，完成前向、反向、优化器更新才说明训练环境可用。

可选的 CPU 回归检查：

```bash
TORCH_WARM_POOL=0 TORCHINDUCTOR_COMPILE_THREADS=1 \
  "$PYTHON_BIN" -m pytest -q \
  tests/test_sft_binidx.py tests/test_sft_prompt.py \
  tests/test_make_data.py tests/test_check_sft_mask.py tests/test_fill_sft_mask.py
```

本次文档编写已在现有环境检查包版本、`pip check` 和导入；示例数据处理与命令检查结果见教程末尾。**没有在全新 conda 环境重新下载安装，也没有为编写文档启动大型 GPU 训练**，因此不把现有环境检查描述成新机器完整安装验收。
