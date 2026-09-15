
<div align="center">

# RWKV-LM-V7
[![English](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-中文版本-red.svg)](./README_CN.md)

</div>

## 项目介绍
让任何研究者在 15 分钟内开始预训练一个完全对齐的 RWKV v7 模型。当然不包括下载数据 :) 。

所有的代码来源于原始 RWKV-LM 项目：https://github.com/BlinkDL/RWKV-LM

此仓库适合快速的在Nvidia和AMD显卡上使用样例数据或私有数据小规模的复现 RWKV v7 系列模型，例如 191M ~ 3B 等大小，我们接下来会重点改善：

- 提供 RWKV 系列模型在多模态等任务的模板代码
- 提供跨平台的内核实现
- 提供可配置的 RWKV Layer 类
- 提供高性能的 Pytorch 推理实现
- 提供适合 3 ~ 70B 的集群训练框架及脚本

我们热爱并回馈开源社区，感谢任何开源社区的实现。如果您发现我们的代码仓库有包含但不限于：代码质量，代码风格，代码可解释性，数值精度误差的问题，欢迎[提交 issue](https://github.com/RWKV-Vibe/RWKV-LM-V7/issues/new)。

> [!WARNING]
> Note: 整个仓库仍处于 WIP 阶段（与基线相比，我们改进了融合算子的使用，计算更高效）。如果您有所顾虑，则可以使用 [RWKV-LM](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp) 作为参考实现。

## 如何开始？

### 准备环境

环境准备，请使用 UV 嘎嘎快而且非常方便，用过的都说好 😉
```
uv venv ./rwkv-lm --python 3.12
source ./rwkv-lm/bin/activate
```
随后安装下列依赖，注意 `pytorch-lightning` 固定使用了 `1.9.5` 版本，此为本仓库特性，请不要升级此依赖包。
```
uv pip install torch
uv pip install -r requirements.txt
```

### 下载数据

```
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 开始训练

1. 初始化空 RWKV-7 模型
```
sh ./demo-training-prepare.sh
```

2. 登录 WandB 账号

3. 开始训练
```
sh ./demo-training-run.sh
```

## 详细解释

此章节包含模型初始化、学习率及细节解释。

RWKV-7 使用了包含经过设计和数学论证的初始化和基于训练结果分析的初始化，加速模型收敛及其性能。

### RWKV G1 模型 Lora 维度

|             | params | 0.1B | 0.4B | 1.5B | 2.9B | 7.2B | 13.3B |  
|-------------|--------|------|------|------|------|------|-------| 
|D_DECAY_LORA |   w    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_AAA_LORA   |   a    |  64  |  64  |  96  |  96  |  128 |  192  |
|D_MV_LORA    |   v    |  32  |  32  |  64  |  64  |  96  |  128  |   
|D_GATE_LORA  |   g    |  128 |  128 |  256 |  320 |  480 |  384  |

### L2Warp
此类惩罚模型，避免模型过度自信，从而缓解 BF16 中间的精度损失。

### 权重及其初始化样例

请严格注意上下文学习率等相关设置：
```
self.k_k = nn.Parameter(torch.zeros(1, 1, C)+0.71 - linear*0.1)
self.k_a = nn.Parameter(torch.zeros(1, 1, C)+1.02)
```

RWKV-7 weight example for 1.5B (L24-D2048, vocab 65536):

| name                | shape         | comment      | initialization  |
|---------------------|---------------|--------------|-----------------|
| emb.weight          | [65536, 2048] | wdecay       | see code        |
| blocks.0.ln0.weight | [2048]        | for layer 0  | 1               |
| blocks.0.ln0.bias   | [2048]        | for layer 0  | 0               |
|                     |               |              |                 |
| blocks.*.ln1.weight | [2048]        |              | 1               |
| blocks.*.ln1.bias   | [2048]        |              | 0               |
| blocks.*.att.x_r    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_w    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_k    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_v    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_a    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.x_g    | [1, 1, 2048]  |              | see code        |
| blocks.*.att.w0     | [1, 1, 2048]  | lr 2x        | see code        |
| blocks.*.att.w1     | [2048, 96]    |              | 0               |
| blocks.*.att.w2     | [96, 2048]    |              | see code        |
| blocks.*.att.a0     | [1, 1, 2048]  |              | 0               |
| blocks.*.att.a1     | [2048, 96]    |              | 0               |
| blocks.*.att.a2     | [96, 2048]    |              | see code        |
| blocks.*.att.v0     | [1, 1, 2048]  | for layer 1+ | 1               |
| blocks.*.att.v1                | [2048, 64]   | for layer 1+ | 0         |
| blocks.*.att.v2                | [64, 2048]   | for layer 1+ | see code  |
| blocks.*.att.g1                | [2048, 256]  |              | 0         |
| blocks.*.att.g2                | [256, 2048]  |              | see code  |
| blocks.*.att.k_k               | [1, 1, 2048] |              | 1         |
| blocks.*.att.k_a               | [1, 1, 2048] |              | 1         |
| blocks.*.att.r_k               | [32, 64]     |              | 0         |
| blocks.*.att.receptance.weight | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.key.weight        | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.value.weight      | [2048, 2048] | wdecay       | see code  |
| blocks.*.att.output.weight     | [2048, 2048] | wdecay       | 0         |
| blocks.*.att.ln_x.weight       | [2048]       |              | see code  |
| blocks.*.att.ln_x.bias         | [2048]       |              | 0         |
|                                |              |              |           |
| blocks.*.ln2.weight            | [2048]       |              | 1         |
| blocks.*.ln2.bias              | [2048]       |              | 0         |
| blocks.*.ffn.x_k               | [1, 1, 2048] |              | see code  |
| blocks.*.ffn.key.weight        | [8192, 2048] | wdecay       | see code  |
| blocks.*.ffn.value.weight      | [2048, 8192] | wdecay       | 0         |
|                                |              |              |           |
| ln_out.weight | [2048]        |        | 1         |
| ln_out.bias   | [2048]        |        | 0         |
| head.weight   | [65536, 2048] | wdecay | see code  |

## 检查结果

在 `out/....../train_log.txt` 路径下，您的损失应该非常接近：

```
0 4.875856 131.0863 0.00059975 2025-04-24 02:23:42.481256 0
1 4.028621 56.1834 0.00059899 2025-04-24 02:28:16.674463 1
2 3.801625 44.7739 0.00059773 2025-04-24 02:32:51.059568 2
3 3.663070 38.9808 0.00059597 2025-04-24 02:37:25.409892 3
4 3.578974 35.8368 0.00059371 2025-04-24 02:41:59.711315 4
5 3.510906 33.4786 0.00059096 2025-04-24 02:46:33.990839 5
6 3.462345 31.8917 0.00058771 2025-04-24 02:51:08.378331 6
7 3.412196 30.3318 0.00058399 2025-04-24 02:55:42.927474 7
8 3.376724 29.2747 0.00057978 2025-04-24 03:00:17.504665 8
9 3.336911 28.1321 0.00057511 2025-04-24 03:04:52.006063 9
10 3.313411 27.4787 0.00056999 2025-04-24 03:09:27.563336 10
11 3.295895 27.0016 0.00056441 2025-04-24 03:14:01.786079 11
```

## 处理训练数据

### 将 jsonl 转换为 binidx 格式

使用 `data/make_data.py` 脚本将你的训练数据从 `.jsonl` 格式转换为 `binidx` 格式。

```
python make_data.py [输入文件] [重复轮数] [上下文长度]
# 示例：
cd data/
python make_data.py demo.jsonl 3 4096
```

`python make_data.py demo.jsonl 3 4096` 命令将会：

- 对 demo.jsonl 进行打乱并复制 3 轮
- 加载 jsonl 并进行 tokenize
- 保存为 demo.bin 和 demo.idx
- 计算 ctxlen=4096 时的 `magic_prime`

假设你的源 jsonl 如下：

- {"text":"aa"}
- {"text":"bb"}
- {"text":"cc"}
- {"text":"dd"}

最终的 binidx 会像这样（这里 "/" 表示 end_of_doc，实际是 token [0]）：`bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/`

> [!WARNING]
> make_data.py 处理大体积 jsonl 时会非常慢，如需处理大文件请参考 [json2binidx_tool](https://github.com/Abel2076/json2binidx_tool)。

### 将 SFT messages jsonl 转换为 binidx + loss mask

SFT 数据处理使用 `data/make_sft_binidx.py`，输入是每行一个样本的 JSONL。每个样本至少需要包含 `messages`，可选包含 `tools`。`messages` 遵循常见的 chat 结构：`system`、`user`、`assistant`、`tool` 等角色按对话顺序排列；`assistant.tool_calls` 会在渲染前统一整理参数格式，字符串 JSON 会被解析成结构化参数，已有 XML 参数片段会原样保留。

示例命令：

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train \
  --vocab rwkv_vocab_v20260603.txt \
  --chat-template data/SFT/sample/chat_template.jinja \
  --ctx-len 4096 \
  --pack \
  --num-workers 8 \
  --shuffle
```

也可以直接传入一个包含 JSONL 分片的文件夹。文件夹输入会递归展开为所有层级里的 `*.jsonl` 文件，并按路径排序：

```bash
python data/make_sft_binidx.py data/sft_shards \
  --output-prefix data/sft_train \
  --ctx-len 4096 \
  --pack \
  --num-workers 8
```

如果你需要保持多个输入文件和每个 epoch 内的原始顺序，可以关闭打乱：

```bash
python data/make_sft_binidx.py data/sft_part_000.jsonl data/sft_part_001.jsonl \
  --out-prefix data/sft_train_ordered \
  --no-shuffle
```

使用 `--out-prefix` 或别名 `--output-prefix` 可以指定输出 binidx 的命名前缀。例如 `--out-prefix data/sft_train` 会写出 `data/sft_train.bin`、`data/sft_train.idx`、`data/sft_train.mask.bin`、`data/sft_train.mask.idx`。传入多个位置参数时必须显式指定输出前缀，因为脚本无法从多个源路径自动推导唯一名字；单个文件默认使用去掉 `.jsonl` 后缀的文件名，单个文件夹默认使用文件夹路径作为前缀。

`--num-workers` 会用在两个阶段，现在这些 worker 是多进程 worker，不是 Python 线程；Jinja 渲染、JSON 解析和 tokenize 会真实并行吃多核。默认路径的读取阶段以“一个 JSONL 文件”为一个任务，所以多个 JSONL 可以并发读取；单个大 JSONL 在读取阶段不会被多个 worker 拆分读取。`--pack-strategy best-fit-decreasing` 会按有界 JSONL shard group 读取和 packing。`--pack-shard-group-size` 默认是 `1`，保持一次只处理一个 shard 的低内存占用，但读取阶段只能读这 1 个 JSONL，best-fit 也只能在这个文件内部优化。调大后，每个 group 可以并发读取多个 JSONL，并在 group 内跨文件做 best-fit packing。group 之间仍然串行：group 1 完成读取、tokenize、packing 并追加写入后，才会开始 group 2。group 内 JSONL 读取并发数是 `min(group_size, num_workers, 当前 group 文件数)`。例如 `--pack-shard-group-size 8 --num-workers 32` 最多同时读取 8 个 JSONL，但渲染和 tokenize 阶段仍最多 32 个进程并行处理样本；`--pack-shard-group-size 64 --num-workers 32` 最多同时读取 32 个 JSONL，剩余文件排队。`--worker-chunksize` 控制多进程渲染/tokenize 每个任务批量处理多少条样本，默认 `64`，通常可以减少进程间调度开销。进度条默认启用，并在 stderr 上单行刷新阶段名、当前文件、已处理/总数、剩余数量和速度；多 worker 时只有主进程统一刷新，worker 不直接输出。渲染/tokenize 使用按输入顺序回收结果的 `ProcessPoolExecutor.map`，所以进度条里的 `file=...part_xxxx.jsonl:line` 表示“已经按输入顺序确认完成到这个位置”，不是 worker 随机完成的最新样本；后面的文件或行可能已经被 worker 算完并暂存在结果队列里，只是要等前面的样本按序回收后才会显示。输出写入仍是单进程 append 到同一套 token binidx 和 mask sidecar。输出仍按确定的样本顺序或 shard group 顺序写入，所以相同输入、`--seed`、`--shuffle` 设置和 group size 会得到可复现结果。当前所有文本文件按 UTF-8 读取，JSONL 额外兼容 UTF-8 BOM，中文内容会按 UTF-8 字节映射到 token span，不会在 mask 推导中丢失。

best-fit 命令示例：

```bash
# 推荐的有界多文件 packing：
# group 之间串行；每个 group 最多 8 个 JSONL；这里读取阶段最多实际用到 8 个 worker。
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 8 \
  --num-workers 32 \
  --shuffle

# 更大的 group，同样的 worker 上限：
# 每个 group 最多 64 个 JSONL；最多 32 个文件并发读取，剩下的文件排队。
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 64 \
  --num-workers 32 \
  --shuffle
```

长任务建议打开 best-fit group 缓存。缓存只支持 `--pack --pack-strategy best-fit-decreasing --no-shuffle`，每个 group 完成后会在 `--pack-cache-dir` 下保存一组小 binidx + mask + meta；中途失败后原命令重跑，会跳过已经完整缓存的 group，再合并成最终输出：

```bash
python data/make_sft_binidx.py /mnt/data/Datas/SFT_RWKV7_13B/sharded_cleaned \
  --out-prefix /mnt/data/Datasets/SFT_RWKV7_13B \
  --ctx-len 86016 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 8 \
  --num-workers 32 \
  --worker-chunksize 64 \
  --no-shuffle \
  --pack-cache-dir /mnt/data/Datasets/SFT_RWKV7_13B.cache \
  --error-log /mnt/data/Datasets/SFT_RWKV7_13B.errors.jsonl \
  --progress
```

如果内存足够，可以把 `--pack-shard-group-size` 试到 `16`、`32` 或 `64`。更大的 group 会让 best-fit 有更多样本可组合，也能让读取阶段并发更多 JSONL，但 group 内 tokenized 样本会暂存在内存中，过大可能撑爆内存。你之前的 `--pack-shard-group-size 1 --num-workers 32` 只有渲染/tokenize 阶段能用 32 个进程，读取阶段每次只有 1 个 JSONL，packing 也只能在单文件内优化；它最省内存，但通常不是最快。

如果需要定位坏 JSON、模板渲染异常或 mask 边界异常，可以加 `--error-log PATH`。它默认不开启；开启后仍然是“遇错停止”，不会跳过坏样本，只会在停止前由主进程追加一条 JSONL 错误记录。日志只记录出错样本，不记录成功样本；每条错误包含 `source_path`、`line_number`、错误类型和错误消息、原始 `source_text`、解析后的完整 `record`，以及便于快速扫结构的 `record_summary`。多进程模式下 worker 不直接写日志，而是把准确的错误样本带回主进程统一写入，避免并发输出互相打架。

整体流程可以抽象为：

1. 读取一个或多个 UTF-8 JSONL，过滤空行，并记录每条样本来自哪个文件和行号，便于定位坏 JSON。
2. 如果输入里有文件夹，就递归展开为所有层级的 `*.jsonl` 文件并排序。
3. 按 `--n-epoch` 重复源样本。这是写 binidx 前的离线重复：`--n-epoch 3` 表示每条源样本会被写入产物 3 次，确实会让输出数据重复 3 份。默认值是 `--n-epoch 1`，所以不想重复数据时不用传这个参数。默认每一份重复都会按 `--seed` 做确定性打乱；使用 `--no-shuffle` 时，每一份都保持输入顺序。
4. 加载权威 chat template：`data/SFT/sample/chat_template.jinja`。根目录模板不是 SFT 数据处理入口，避免误用。
5. 对每条样本先规范化工具调用参数，再用同一个 Jinja template 渲染两次：一次渲染到最后一轮 assistant 之前，用来确定条件上下文边界；一次渲染完整样本，用来得到真正写入训练集的文本。
6. 最后一轮 assistant 会被规范化为始终包含 think 标签。如果原始内容已有 think 结束标签，就保留原始 think；如果没有，就在最终回复前补一个空 think 块。历史 assistant、系统、用户、工具返回都只作为上下文。
7. loss mask 从“最后一轮 assistant 的可训练后缀”推导：assistant 内容边界之前全部为 `0`。样本里真实存在的 think 内容参与训练；无 thinking 样本自动补出的空 think 块只作为格式上下文，仍然是 `0`；可见回复、最终工具调用、assistant 结束段和真实样本结束段为 `1`。
8. 文本只 tokenize 一次。代码用 UTF-8 字节跨度记录每个 token 对应的字符区间，再把字符级可训练区间投影为 token 级 mask。这样中文、多字节符号和特殊片段都走同一套规则。
9. 如果不传 `--pack`、`--pad`、`--pack-length` 或 `--pad-length`，每条重复后的源样本会写成一个变长 binidx document，并同步写入一个同长度的 mask document，不会自动 padding。每个独立 document 末尾仍然会有真实的 `EOD_TOKEN`，并且这个 EOD 参与训练。`--ctx-len`、`--pack-length` 和 `--pad-length` 都是 token 数，不是字符数。推荐用 `--ctx-len N --pack` 或 `--ctx-len N --pad`，实际写出长度会自动使用 `N + 1`，用于匹配训练端 next-token label。使用 `--pack` / `--pack-length` 时，样本默认按输入顺序做不可拆分 packing：多个完整样本可以合并到同一个固定长度 document，样本之间的分隔换行 mask 为 `0`；如果当前 document 放不下下一条完整样本，就先把当前 document 右侧 padding 后写出，再新开 document。`--pack-strategy best-fit-decreasing` 会先按样本 token 长度从大到小排序，再用 best-fit 近似装箱，仍然不拆样本，但会重排样本以减少 padding；这个策略按 JSONL shard group 独立处理。`--pack-shard-group-size 1` 表示每组一个 JSONL；更大的值允许在有界批次内跨文件 best-fit，提高 packing 利用率，同时避免把全量 tokenized 样本一次性放进内存。所有 group 处理完后仍写成同一组 `PREFIX` binidx + mask 输出。使用 `--pad` / `--pad-length` 时不做 packing：每条源样本独立 padding 到固定长度。过长样本会在 tokenize 后、packing/padding 前按目标 token 长度过滤丢弃，不会终止整个构建。
10. 输出包含主 token 数据集和 mask sidecar：`PREFIX.bin`、`PREFIX.idx`、`PREFIX.mask.bin`、`PREFIX.mask.idx`。SFT 训练时主数据集提供 token，mask sidecar 提供哪些 token 参与 loss。

如果想让已有 SFT binidx 数据的所有 token 都参与训练，可以把 mask sidecar 全部填成 `1`。`.mask.bin` 存实际 mask 数组，`.mask.idx` 存 document 长度、offset 和 dtype，所以只改 mask 数值时主 token 文件不需要动。默认脚本会写出一套新的 mask sidecar：

```bash
python scripts/fill_sft_mask.py /mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
  --output-prefix /mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B.all_train.mask
```

之后可以把生成的 `*.all_train.mask.bin/.idx` 重命名或复制成 `DATA_FILE.mask.bin/.idx`，也可以保留原始 mask，并在 `train.py` 里传 `--sft_mask_file /mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B.all_train.mask`。如果确认要原地覆盖原始 mask，加 `--in-place`；这只会改 `PREFIX.mask.bin`。

参数含义：

- `--chat-template`：SFT 渲染模板路径，默认 `data/SFT/sample/chat_template.jinja`。
- `--vocab`：tokenizer vocab，默认 `rwkv_vocab_v20260603.txt`。
- `--out-prefix` / `--output-prefix`：输出 binidx 的命名前缀，四个输出文件都由这个前缀派生。
- `--n-epoch`：离线重复数据次数，默认 `1`；大于 `1` 会让样本在产物中重复出现。
- `--seed`：打乱顺序用的随机种子；关闭 shuffle 时不影响样本顺序。
- `--shuffle` / `--no-shuffle`：是否在每个 epoch 内打乱样本，默认开启。
- `--num-workers`：并发读取、渲染和 tokenize 的多进程 worker 数，默认 `1`。
- `--worker-chunksize`：多进程渲染/tokenize 每个任务批量处理的样本数，默认 `64`；样本很短时可以调大减少调度开销，样本很长或希望进度更细时可以调小。
- `--progress` / `--no-progress`：是否显示单行刷新进度条，默认开启；进度条写到 stderr，最终统计仍写到 stdout。
- `--progress-interval`：进度条最小刷新间隔秒数，默认 `0.2`；设为 `0` 会每条样本都刷新。
- `--error-log`：错误样本 JSONL 日志路径，默认不写；启用后只在异常时追加完整出错样本和摘要，处理仍会停止。
- `--ctx-len`：训练上下文 token 数；配合 `--pack` 或 `--pad` 时，预处理长度自动使用 `ctx_len + 1`。
- `--pack`：启用顺序样本不拆分 packing，长度为 `ctx_len + 1`；不设置时默认关闭。
- `--pack-strategy`：packing 策略，默认 `ordered` 保持样本顺序；`best-fit-decreasing` 会在每个 JSONL shard group 内按长度重排做近似最优打包，减少 padding，并把所有 group 追加到同一组输出文件。
- `--pack-shard-group-size`：每个 best-fit group 包含多少个 JSONL shard，默认 `1`；调大后可在有界 group 内并发读取多个 JSONL，并跨文件 packing。
- `--pack-cache-dir`：best-fit group 缓存目录，只能配合 `--pack --pack-strategy best-fit-decreasing --no-shuffle` 使用；中断后重跑同一命令会复用已完整缓存的 group。
- `--pad`：启用逐样本 padding，长度为 `ctx_len + 1`；不设置时默认关闭，不能和 `--pack` 同时使用。
- `--pack-length`：兼容旧用法，显式指定固定长度 packing 目标 token 数。
- `--pad-length`：兼容旧用法，显式指定逐样本 padding 目标 token 数；不能和 `--pack-length` 同时使用。
- `--current-date`、`--current-location`：可覆盖或注入系统消息里的日期和位置字段。

### 使用 SFT binidx 数据训练

使用 SFT 预处理产物训练时，设置 `--data_type sft_binidx`。`--data_file` 传 binidx 前缀，不带 `.bin` 或 `.idx`；训练代码默认读取 `DATA_FILE.mask` 作为 mask sidecar，也可以用 `--sft_mask_file` 显式指定另一个 mask 前缀。

在 SFT 模式下，`--epoch_steps` 和 `--epoch_count` 由用户直接控制。`epoch_steps` 表示每个 epoch 多少个 optimizer step，`epoch_count` 表示总共训练多少个 epoch。这个语义不同于预训练 `binidx`，预训练仍保留原来的 magic-prime 调度。

如果想完整跑一遍 SFT binidx 数据，可以设置 `--sft_one_pass 1`。训练脚本会只读取 `DATA_FILE.idx` 的 document 数，自动计算 `epoch_steps = ceil(num_documents / effective_bsz)`，并把 `epoch_count` 设为 `1`；其中 `effective_bsz = num_nodes * devices * micro_bsz * accumulate_grad_batches`。向上取整后如果样本数不足一个完整 step，dataset 会按原有确定性顺序从开头 wrap，启动日志会打印重复的尾部样本数。

手动指定训练长度时，直接写 `--epoch_steps` 和 `--epoch_count`：

```bash
python train.py \
  --load_model model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --proj_dir out/sft-0.4b \
  --data_file data/sft_train \
  --data_type sft_binidx \
  --ctx_len 4096 \
  --epoch_steps 1000 \
  --epoch_count 1 \
  --sft_one_pass 0 \
  --micro_bsz 1 \
  --accumulate_grad_batches 1 \
  --vocab_size 65536 \
  --n_layer 24 \
  --n_embd 1024 \
  --dim_ffn 4096 \
  --head_size 64 \
  --d_decay_lora 64 \
  --d_aaa_lora 64 \
  --d_mv_lora 32 \
  --d_gate_lora 128 \
  --my_testing x070 \
  --lr_init 1e-5 \
  --lr_final 1e-6 \
  --lr_wsd_decay_iters 0 \
  --lr_wsd_decay_style cosine \
  --warmup_steps 10 \
  --weight_decay 0 \
  --accelerator gpu \
  --devices 1 \
  --precision bf16 \
  --strategy deepspeed_stage_2 \
  --grad_cp 1
```

如果只想完整跑一遍数据，让训练脚本自动计算 step，可以写 `--sft_one_pass 1`。这种直接调用 `train.py` 的方式可以不写 `--epoch_steps` / `--epoch_count`，因为它们有整数默认值并会被覆盖；不要传空字符串：

```bash
python train.py \
  --load_model model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --proj_dir out/sft-0.4b-one-pass \
  --data_file data/sft_train \
  --data_type sft_binidx \
  --ctx_len 4096 \
  --sft_one_pass 1 \
  --micro_bsz 1 \
  --accumulate_grad_batches 4 \
  --vocab_size 65536 \
  --n_layer 24 \
  --n_embd 1024 \
  --dim_ffn 4096 \
  --head_size 64 \
  --d_decay_lora 64 \
  --d_aaa_lora 64 \
  --d_mv_lora 32 \
  --d_gate_lora 128 \
  --my_testing x070 \
  --lr_init 1e-5 \
  --lr_final 1e-6 \
  --lr_wsd_decay_iters 0 \
  --lr_wsd_decay_style cosine \
  --warmup_steps 10 \
  --weight_decay 0 \
  --accelerator gpu \
  --devices 1 \
  --precision bf16 \
  --strategy deepspeed_stage_2 \
  --grad_cp 1
```

这个通用示例里，`--accelerator gpu` 表示用 CUDA GPU 训练，`--devices 1` 表示当前节点使用 1 张 GPU。如果要在单节点做多卡 DeepSpeed，把 `--devices` 改成 GPU 数量，比如 `--devices 8`；当 `strategy` 包含 `deepspeed`、`num_nodes=1` 且 `devices > 1` 时，`train.py` 会自动用 `torchrun` 重启多卡进程。SFT 调度里，`real_bsz = num_nodes * devices * micro_bsz` 表示每次 forward 的全局样本数，`effective_bsz = real_bsz * accumulate_grad_batches` 表示每个 optimizer step 消耗的样本数。这里故意不写 `--my_exit_tokens`，因为 SFT 由 `--epoch_count` 或 `--sft_one_pass` 控制停止；`my_exit_tokens` 是预训练 token-limit 调度的一部分。`--lr_wsd_decay_iters 0` 表示关闭 SFT 专用末段衰减，warmup 后保持 `lr_init`；如果设为正数，例如 `--lr_wsd_decay_iters 1000 --lr_wsd_decay_style cosine`，训练最后 1000 个 optimizer step 会按 cosine 从 `lr_init` 衰减到 `lr_final`。`lr_wsd_decay_style` 支持 `none`、`linear`、`cosine`。

训练阶段的 SFT document shuffle 默认关闭。需要打乱时设置 `--sft_train_shuffle 1 --sft_train_shuffle_seed 1234`，它会在 `MyDataset` 内部对每个 epoch 的 train sample offset 应用该 epoch 固定的确定性 permutation；这样仍保留原来的多 rank / 梯度累计采样流和断点续训 offset。启用 tail eval 时，train/eval split 也会先按 seed 对全量 document 做固定 permutation 再切分，所以 held-out eval 固定但不再只是原始数据最后一段。如果用 `scripts/probe_sft_binidx_steps.py` 检查已开启 shuffle 的训练 step，也要传相同的 `--sft-train-shuffle 1 --sft-train-shuffle-seed 1234`，否则 probe 报出的 doc index 会按未打乱顺序解释。

训练端使用 next-token label，所以每个 SFT document 需要提供 `ctx_len + 1` 个 token。document 比这个短时，dataloader 会在内存里用 `--sft_pad_token_id` padding，并把 padding mask 设为 `0`；document 更长时会直接报错。为了让训练长度稳定，建议预处理时使用 `--ctx-len CTX_LEN --pack` 或 `--ctx-len CTX_LEN --pad`，预处理会自动写出 `CTX_LEN + 1` 个 token，训练时再设置 `--ctx_len CTX_LEN`。RWKV7 x070 的 `ctx_len` 需要能被 16 整除。

`--sft_masked_ce_chunk` 目前先保持 `0`。`0` 表示关闭实验性的 Python 分块 masked-head CE，SFT 默认走旧的完整 logits masked CE；这个设置本身不会引入额外分块、checkpoint 重算或 ZeRO-3 反复 all-gather 的效率损耗。注意它仍会物化完整 `[batch, ctx_len, vocab]` logits，所以长上下文显存压力仍然很大。正数 Python chunk 路径虽然能降低 Python loss 侧的 logits 峰值，但服务器反馈 `4096/8192` 在 13B + ZeRO-3 长上下文下仍会 timeout，因此生产训练不要开启。

新的 `--sft_masked_fused_ce_chunk N` 是独立的 CUDA fused masked head CE 路径，和上面的 Python chunk 不是一回事。设为正数时，SFT 会一次性调用 fused CUDA op，op 内部按 `N` 行临时 logits buffer 做 `hidden @ head.weight.T`、masked CE、`grad_hidden` 和 `grad_weight`，loss 和梯度都按 `loss_mask.sum()` 归一化，不带 L2Wrap，不改变预训练 `(x, y)` 的 fused CE 路径。这个路径目前只做 CUDA/H800 方向，默认仍为 `0`；建议先跑下面的服务器 smoke，再在 13B 脚本里尝试 `SFT_MASKED_FUSED_CE_CHUNK=4096` 或 `8192`。

CUDA 算子按职责分成两类。模型主体算子会被预训练和 SFT 共用；head / CE loss 算子才区分预训练和 SFT：

| 算子 / 路径 | pretrain 是否使用 | SFT 是否使用 | 说明 |
| --- | --- | --- | --- |
| `rwkv7_clampw_v3` | 是 | 是 | RWKV7 time-mix recurrent core，属于模型主体；`KERNEL=@rwkv3` 时启用，不关心 loss 是预训练 CE 还是 SFT masked CE。 |
| `rwkv7_tmix_mix6_bf16_v5` | 是 | 是 | time-mix 前处理/混合相关，属于模型主体。 |
| `rwkv7_cmix_bf16_v5` | 是 | 是 | channel-mix，属于模型主体。 |
| `rwkv7_l2wrap_ce_bf16_v2` | 是 | 否 | 预训练完整 logits + targets 的 L2Wrap CE 路径。 |
| `rwkv7_head_l2wrap_ce_bf16_v4.forward` | 是 | 否 | 预训练 head fused CE：`hidden @ head.weight.T` + CE + L2Wrap，无 SFT mask。 |
| `rwkv7_head_l2wrap_ce_bf16_v4.forward_masked` | 否 | 是 | SFT 专用 fused masked head CE：`hidden @ head.weight.T` + masked CE，不加 L2Wrap，按 `loss_mask.sum()` 归一化。 |

因此，`rwkv7_clampw_v3` 不是预训练专用；SFT 和预训练都会经过它。真正 pretrain-only 的主要是带 L2Wrap、无 mask 的 head/CE 路径；SFT-only 的主要是新增的 `forward_masked`。

上面示例里的 0.4B checkpoint 是 `L24-D1024`，对应 `dim_ffn=4096`、`vocab_size=65536`、`head_size=64`，RWKV7 G1 LoRA 维度为 `64/64/32/128`。如果换用其他 checkpoint，需要先看对应架构文本，保证这些形状参数和 checkpoint 一致。

SFT mask 训练的实现框架：

1. 离线数据处理阶段只负责生成两套完全对齐的 binidx：主数据 `PREFIX.bin/.idx` 存 token id，sidecar `PREFIX.mask.bin/.idx` 存同长度的 `0/1` loss mask。
2. `train.py` 通过 `--data_type sft_binidx` 进入 SFT 分支。这个分支不使用预训练的 magic-prime 调度，而是保留用户传入的 `--epoch_steps` 和 `--epoch_count`，并把 Lightning `max_epochs` 设为 `epoch_count`，所以 SFT 会按指定 epoch 数正常结束。启用 `--accumulate_grad_batches G` 时，`epoch_steps` 仍然表示 optimizer step 数；dataloader 会为每个 epoch 提供 `epoch_steps * G` 个 micro-batch。
   如果启用 `--sft_one_pass 1`，这个分支会覆盖手写的 `epoch_steps/epoch_count`，自动设置为“完整跑一遍数据”的步数和 `epoch_count=1`。
3. `src/dataset.py` 的 `MyDataset` 会加载 `--data_file` 指向的 token binidx，同时默认加载 `--data_file.mask`。如果你传了 `--sft_mask_file`，就用显式 mask 前缀。初始化时会检查 token 和 mask 的 document 数量、每个 document 长度必须完全一致。`--sft_train_shuffle 1` 会用 `--sft_train_shuffle_seed` 对每个 epoch 内的训练 sample offset 做确定性 epoch permutation；如果启用 tail eval，held-out eval documents 会从同一个 seed 固定的全量 document permutation 里切出，而不是直接取原始文件尾部。
4. 每个 SFT document 最长允许 `ctx_len + 1` 个 token。短 document 会在内存中用 `--sft_pad_token_id` padding 到 `ctx_len + 1`，padding mask 始终为 `0`；长 document 直接报错，避免静默截断破坏 mask。
5. dataloader 返回三元组 `(x, y, loss_mask)`：`x = token_ids[:-1]`，`y = token_ids[1:]`，`loss_mask = raw_mask[1:]`。mask 右移是为了和 next-token label 对齐，也就是 mask 标记的是“这个 target token 是否参与 loss”。
6. `src/model.py` 的 `training_step` 根据 batch 长度分流：普通预训练 `(x, y)` 继续走原来的 fused CE 快路径；SFT `(x, y, loss_mask)` 默认走 `src/sft_loss.py::masked_cross_entropy`。这样 SFT 不影响预训练性能路径。
7. `masked_cross_entropy` 先用标准 CE 得到每个 token 的 loss，再只对 `loss_mask=1` 的位置求平均。如果一个 batch 的 mask 全为 `0`，返回可反传的 0 loss，避免除零和梯度图断裂。
8. `--sft_masked_ce_chunk` 当前建议保持 `0`。正数会改走 `src/sft_loss.py::masked_head_cross_entropy` 的 Python 分块路径：模型主体只输出 hidden，不先生成完整 logits；loss 函数只收集 mask=1 的 target token，并按 `N` 个 trainable token 一块执行 `hidden @ head.weight.T` 和 CE。这个路径可用于小模型/单卡数值对比，但在 13B ZeRO-3 长上下文下会反复触发 head weight all-gather，服务器实测会 timeout，因此不作为生产推荐路径。
9. `--sft_masked_fused_ce_chunk N` 是新的 CUDA fused masked head CE 路径。它直接调用 `rwkv7_head_l2wrap_ce_bf16_v4.forward_masked`，在 CUDA op 内部按 `N` 行 chunk 计算 head logits 和 masked CE，并直接返回 `grad_hidden/grad_weight`。这个路径不带 L2Wrap，目标是和当前 SFT masked CE 数值等价，同时避免完整 logits OOM 和 Python chunk 反复 all-gather。
10. 梯度累计由 Lightning 的 `--accumulate_grad_batches` 执行；SFT dataset 会同步使用这个参数来计算 epoch 长度、完整数据遍历步数和 step checkpoint 的 mid-epoch 恢复偏移。也就是说，从 `rwkv-step-N.pth` 恢复时会跳过 `N * accumulate_grad_batches` 个 micro-batch，而不是只跳过 `N` 个 micro-batch。
11. SFT 学习率默认只做 warmup，warmup 后保持 `lr_init`。启用 `--lr_wsd_decay_iters K` 后，调度器会用 `total_steps = epoch_steps * epoch_count` 定位最后 `K` 个 optimizer step，并按 `--lr_wsd_decay_style linear|cosine` 从 `lr_init` 衰减到 `lr_final`；这个 SFT WSD 调度不依赖 `my_exit_tokens`，也不会触发预训练的 token-limit 退出逻辑。
   WSD 的 step 使用 Lightning 恢复出的 `global_step`，所以从 DeepSpeed/Lightning checkpoint 断点续训时，LR 会继续处在原曲线的位置。断点续训时应保持 `epoch_steps`、`epoch_count`、`lr_wsd_decay_iters`、`lr_wsd_decay_style`、`lr_init`、`lr_final` 和原训练一致；如果你有意改变它们，就等价于从当前 step 接到一条新的 LR 曲线。

WSD 衰减区间的计算方式：

- `epoch_steps` 是每个 SFT epoch 的 optimizer step 数。
- `epoch_count` 是本次 SFT run 的 epoch 数。
- `total_steps = epoch_steps * epoch_count`。
- `K = min(lr_wsd_decay_iters, total_steps)`。
- 衰减起点是第 `total_steps - K` 个 optimizer step，按 0-based step 计数。
- 对应 epoch/step 是：
  - `decay_start_epoch = (total_steps - K) // epoch_steps`
  - `decay_start_step_in_epoch = (total_steps - K) % epoch_steps`
- `lr_wsd_decay_iters=0` 或 `lr_wsd_decay_style=none` 时不衰减，warmup 后保持 `lr_init`。

例子：`epoch_steps=12500`、`epoch_count=1`、`lr_wsd_decay_iters=1000` 时，`total_steps=12500`，从全局 optimizer step `11500` 开始衰减，也就是第 0 个 epoch 的第 `11500` 个 step 开始，最后一个 step 到 `lr_final`。如果 `epoch_count=3`，`total_steps=37500`，同样的 `K=1000` 会从全局 step `36500` 开始，也就是第 2 个 epoch 的第 `11500` 个 step 开始。

测试流程和覆盖内容：

- 默认 CPU/单进程单测：`tests/test_sft_training.py` 覆盖 mask sidecar 加载、mask shift、padding、坏 mask、过长 document、SFT epoch 调度、梯度累计下的 dataset 长度/续训偏移、WSD LR 调度、WSD 断点续训 step 位置、完整 logits masked CE 数学正确性、chunked masked-head CE 等价性和 zero-mask 行为。
- 数据处理回归：`tests/test_sft_binidx.py` 覆盖权威 Jinja 渲染、think/no-think 规则、中文 UTF-8 span、工具调用、packing、padding、递归目录、多 JSONL 并发读取和 CLI 参数。
- CUDA smoke 第一档：`RWKV_RUN_CUDA_SFT_SMOKE=1` 会在服务器加载真实 RWKV7 checkpoint，构造 tiny SFT binidx，跑 CUDA forward/backward，验证 masked SFT loss 可以反传。
- CUDA smoke 第二档：`RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` 会启动 `train.py` 跑 1 个 SFT step，覆盖 Lightning、DeepSpeed、optimizer、多卡 torchrun 链路。可用 `RWKV_SFT_SMOKE_ACCUMULATE_GRAD_BATCHES=2` 之类的环境变量额外覆盖梯度累计路径。
- CUDA smoke 第三档：`RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` 会保存 `rwkv-step-1.pth` 并从它恢复，覆盖 SFT 断点续训和 DeepSpeed 分片 checkpoint 加载。
- CUDA smoke 第四档：`RWKV_RUN_TRAIN_PY_SFT_WSD_RESUME_SMOKE=1` 会用 DeepSpeed 先保存 step checkpoint，再恢复并检查恢复后的 `train_log.txt` 里 LR 已经处在 WSD 衰减后的正确位置。
- CUDA smoke 第五档：`RWKV_RUN_CUDA_SFT_MASKED_CE_CHUNK_EQUIV_SMOKE=1` 会在单卡上对比完整 logits CE 和旧的 `--sft_masked_ce_chunk` Python 分块 CE。这个测试只保留为实验路径的数值对比；13B ZeRO-3 生产训练不要开启正数 Python chunk。
- CUDA smoke 第六档：`RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_OP_EQUIV_SMOKE=1` 会直接校验 `rwkv7_head_l2wrap_ce_bf16_v4.forward_masked` 这个 CUDA op：完整 logits PyTorch masked CE vs fused CUDA 的 loss、`grad_hidden`、`grad_weight`，并额外覆盖全 0 mask。这是最快定位新算子本身精度问题的测试。
- CUDA smoke 第七档：`RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_SMOKE=1` 会在单卡上做训练级对比：完整 logits CE 和新的 `--sft_masked_fused_ce_chunk` CUDA fused masked head CE，记录 loss、梯度范数、参数采样差异、耗时和 CUDA peak memory。
- CUDA smoke 第八档：`RWKV_RUN_TRAIN_PY_SFT_FUSED_CE_SMOKE=1` 会走 `train.py` + DeepSpeed/ZeRO，直接验证 fused masked CE 在多卡 strategy 下能跑完并写出 loss；这个更接近检查 13B 是否还会 timeout。

当前已验证结果：

- 本地默认回归：`133 passed, 7 skipped`。
- SFT 训练相关 targeted：`42 passed`；`src.dataset`、`src.sft_loss`、`src.lr_schedule`、可单测的 `train.py` helper surface 合计 `99%`，其中 `src.lr_schedule.py` 和 `train.py` helper surface 为 `100%`。
- SFT 数据处理覆盖率：`src.sft_binidx`、`data.make_sft_binidx`、`data.tokenizer.rwkv_tokenizer` 合计 `99%`。
- 13.3B 启动脚本语法检查：`bash -n run_13b_sft_zero3_offload.sh` 通过。
- 服务器 8xH800：
  - `RWKV_RUN_CUDA_SFT_SMOKE=1` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_TRAIN_PY_SFT_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_2` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `3 passed, 2 skipped`。
  - `RWKV_RUN_CUDA_SFT_ACCUM_EQUIV_SMOKE=1` -> `1 passed in 14.74s`。
  - `RWKV_RUN_TRAIN_PY_SFT_DP_ZERO_ACCUM_EQUIV_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `1 passed in 81.06s`。
  - `RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1` + `RWKV_SFT_SMOKE_DEVICES=8` + `deepspeed_stage_3_offload` -> `1 passed in 284.01s`。
- 服务器反馈：旧的 Python `--sft_masked_ce_chunk 4096/8192` 在 13B + ZeRO-3 长上下文下仍会 timeout；生产脚本默认保持 `SFT_MASKED_CE_CHUNK=0`。新的 CUDA fused 路径通过 `SFT_MASKED_FUSED_CE_CHUNK` 单独开启，需要先跑下方 smoke。

新增 fused masked CE 服务器测试命令：

```bash
RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_OP_EQUIV_SMOKE=1 \
RWKV_SFT_FUSED_CE_OP_BATCH=2 \
RWKV_SFT_FUSED_CE_OP_TIME=2049 \
RWKV_SFT_FUSED_CE_OP_HIDDEN=4096 \
RWKV_SFT_FUSED_CE_OP_CHUNKS=257,4096,8192 \
RWKV_SFT_FUSED_CE_OP_EQUIV_SUMMARY_FILE=/tmp/rwkv_sft_fused_ce_op_equiv.json \
python -m pytest -q tests/test_sft_cuda_smoke.py::test_cuda_sft_masked_fused_ce_op_matches_full_logits
```

这个 op 级测试不需要模型 checkpoint。它会编译 CUDA extension，和完整 logits PyTorch masked CE 对比 loss 和最大绝对梯度误差，并把结果写到 summary 文件。上面的严格服务器命令使用 BF16、真实 `vocab_size=65536`、4098 行、13.3B hidden size 4096，同时测非整除 chunk `257`、生产常用 chunk `4096`、大于总行数的 chunk `8192`，并覆盖四种 mask：全训练、稀疏训练、单 token 训练、全 0 mask。容差可以用 `RWKV_SFT_FUSED_CE_OP_LOSS_ATOL`、`RWKV_SFT_FUSED_CE_OP_GRAD_ATOL`、`RWKV_SFT_FUSED_CE_OP_ZERO_ATOL` 覆盖。

```bash
RWKV_SFT_SMOKE_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_CUDA_SFT_MASKED_FUSED_CE_SMOKE=1 \
RWKV_SFT_FUSED_CE_PAD_LENGTH=4097 \
RWKV_SFT_FUSED_CE_STEPS=8 \
RWKV_SFT_FUSED_CE_MICRO_BSZ=1 \
RWKV_SFT_FUSED_CE_CHUNK=512 \
RWKV_SFT_FUSED_CE_SUMMARY_FILE=/tmp/rwkv_sft_fused_ce.json \
python -m pytest -q tests/test_sft_cuda_smoke.py::test_cuda_sft_masked_fused_ce_training_matches_full_logits
```

```bash
RWKV_SFT_SMOKE_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_FUSED_CE_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_FUSED_CE_TRAIN_PY_CHUNK=512 \
RWKV_SFT_FUSED_CE_TRAIN_PY_STEPS=2 \
RWKV_SFT_FUSED_CE_TRAIN_PY_SUMMARY_FILE=/tmp/rwkv_sft_fused_ce_zero3.json \
python -m pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_masked_fused_ce_smoke
```

## 13.3B SFT 启动脚本操作手册

完整 13.3B 示例是 [run_13b_sft_zero3_offload.sh](/D:/codes/RWKV-LM-V7-12B-train/run_13b_sft_zero3_offload.sh)。它按 `model/rwkv7-g1f-13.3b.txt` 推导了模型形状：`n_layer=61`、`n_embd=4096`、`dim_ffn=16384`、`vocab_size=65536`、`head_size=64`、LoRA 维度 `192/192/128/384`。默认适配 8 卡 H800、`deepspeed_stage_3_offload`、开启激活检查点，并使用 SFT binidx+mask 数据。脚本里的配置都可以通过同名环境变量覆盖，所以推荐把一次训练的输入、输出、batch、LR 和 checkpoint 策略都写在启动命令前面。

### 1. 准备 SFT binidx 数据

第一步：准备固定长度 SFT 数据。`--ctx-len 8192 --pack` 会写出每个 document `8193` 个 token，因为训练端使用 next-token label：

```bash
python data/make_sft_binidx.py /mnt/data/datasets/sft_jsonl \
  --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
  --ctx-len 8192 \
  --pack \
  --pack-strategy best-fit-decreasing \
  --pack-shard-group-size 8 \
  --num-workers 32 \
  --shuffle
```

### 2. 计算手动 schedule

第二步：计算完整跑一遍 SFT documents 需要多少 `EPOCH_STEPS`：

```bash
python scripts/calc_sft_onepass_steps.py /mnt/data/datasets/sft_train_ctx8192 \
  --num-nodes 1 \
  --devices 8 \
  --micro-bsz 1 \
  --accumulate-grad-batches 1 \
  --eval-tail-ratio 0.005 \
  --eval-include-in-train 0 \
  --n-pass 1 \
  --ctx-len 8192
```

这个脚本只读取 `DATA_FILE.idx`，不会 mmap 大体积 `.bin`，所以在几十 GB 数据上也很快。输出会包含 `total_documents`、`train_documents`、`eval_documents`、`epoch_steps`、`epoch_count`、`total_optimizer_steps`、由 `ceil(...)` 带来的尾部重复样本数，以及按几个常见 seconds/step 估算的保存间隔。`--eval-include-in-train 0` 表示 held-out eval：尾部 eval documents 不参与训练；设为 `1` 表示 overlap 监控模式：训练仍使用所有 documents，同时固定用尾部 split 做 eval。

eval 参数有三种常见写法：

```bash
# held-out：尾部 0.5% 只用于 eval，不参与训练
python scripts/calc_sft_onepass_steps.py /mnt/data/datasets/sft_train_ctx8192 \
  --num-nodes 1 --devices 8 --micro-bsz 1 --accumulate-grad-batches 1 \
  --eval-tail-ratio 0.005 --eval-include-in-train 0 \
  --n-pass 1 --ctx-len 8192

# overlap：训练仍使用全量数据，尾部 0.5% 只作为固定 eval 监控集
python scripts/calc_sft_onepass_steps.py /mnt/data/datasets/sft_train_ctx8192 \
  --num-nodes 1 --devices 8 --micro-bsz 1 --accumulate-grad-batches 1 \
  --eval-tail-ratio 0.005 --eval-include-in-train 1 \
  --n-pass 1 --ctx-len 8192

# 固定 eval 文档数；--eval-tail-docs > 0 时优先于 --eval-tail-ratio
python scripts/calc_sft_onepass_steps.py /mnt/data/datasets/sft_train_ctx8192 \
  --num-nodes 1 --devices 8 --micro-bsz 1 --accumulate-grad-batches 1 \
  --eval-tail-docs 2000 --eval-include-in-train 0 \
  --n-pass 1 --ctx-len 8192
```

输出里 `total_documents` 是总 documents；`eval_documents` 是尾部 eval documents；`train_documents` 是用于计算训练步数的 documents。`eval_include_in_train=0` 时，`train_documents = total_documents - eval_documents`，所以 eval 与训练不重合；`eval_include_in_train=1` 时，`train_documents = total_documents`，所以训练使用全量数据，eval 只作为重叠监控集。`epoch_steps = ceil(train_documents / effective_bsz)`，因此 held-out eval 会比 overlap eval 少一些训练 step。

### 3. 启动 13.3B SFT

第三步：在 8 张 H800 上启动 13.3B SFT：

针对 `ctx_len=86016`、160G 级 SFT 数据、约 20 天窗口的 13.3B 长训，可以直接用这个自包含脚本。它在同一个文件里写明 13.3B 模型结构、SFT 数据、DeepSpeed、LR、checkpoint、eval 和 loss 分块参数；当前默认是手动 schedule：`SFT_ONE_PASS=0`、`EPOCH_STEPS=45000`、`EPOCH_COUNT=1`，尾部 `0.5%` held-out eval，`SAVE_EVERY_N_STEPS=20` 且同频 eval，最后 `15000` step 做 cosine WSD 衰减。默认超参偏保守：`micro_bsz=1`、8 卡、无梯度累计、ZeRO-3-offload、开启 block 级激活检查点、`lr=5e-6`、`weight_decay=0.001`、warmup 200 step。若后续要完整一遍数据或改训练窗口，先用上面的 `scripts/calc_sft_onepass_steps.py` 重新计算，再改 `EPOCH_STEPS` / eval / 保存间隔。

先编辑 `run_13b_sft_ctx86016_onepass.sh` 顶部的 `LOAD_MODEL`、`DATA_FILE`、`PROJ_DIR`、`SAVE_EVERY_N_STEPS`、`LR_WSD_DECAY_ITERS` 等配置，再直接运行：

```bash
bash run_13b_sft_ctx86016_onepass.sh
```

如果后面根据一遍数据的总 step 数决定开启末段衰减，建议把 `LR_WSD_DECAY_ITERS` 设为总 optimizer step 的最后 5%-10%；如果目标约 20 天、希望每半天保存一次，`SAVE_EVERY_N_STEPS` 可以设为总 optimizer step 除以约 `40`，或者用实际前几小时吞吐换算。

如果你要手动控制一遍或多遍数据，用下面这种写法。`EPOCH_STEPS` 按“一遍数据”算，想跑 `N` 遍就把 `EPOCH_COUNT=N`：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth \
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
PROJ_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload \
CTX_LEN=8192 \
N_NODE=1 \
GPU_PER_NODE=8 \
MICRO_BSZ=1 \
ACCUMULATE_GRAD_BATCHES=4 \
EPOCH_STEPS=12500 \
EPOCH_COUNT=1 \
SFT_ONE_PASS=0 \
STRATEGY=deepspeed_stage_3_offload \
GRAD_CP=1 \
LR_INIT=1e-5 \
LR_FINAL=1e-6 \
LR_WSD_DECAY_ITERS=1000 \
LR_WSD_DECAY_STYLE=cosine \
WARMUP_STEPS=10 \
EPOCH_SAVE=1 \
SAVE_EVERY_N_STEPS=0 \
KEEP_LAST_N_CHECKPOINTS=3 \
WANDB_PROJECT=RWKV-13B-SFT \
bash run_13b_sft_zero3_offload.sh
```

如果只想完整跑一遍数据，推荐直接让脚本传 `SFT_ONE_PASS=1`，这时 `train.py` 会覆盖脚本里传入的 `EPOCH_STEPS/EPOCH_COUNT` 占位值：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth \
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
PROJ_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-one-pass \
CTX_LEN=8192 \
N_NODE=1 \
GPU_PER_NODE=8 \
MICRO_BSZ=1 \
ACCUMULATE_GRAD_BATCHES=4 \
SFT_ONE_PASS=1 \
STRATEGY=deepspeed_stage_3_offload \
GRAD_CP=1 \
LR_INIT=1e-5 \
LR_FINAL=1e-6 \
LR_WSD_DECAY_ITERS=1000 \
LR_WSD_DECAY_STYLE=cosine \
WARMUP_STEPS=10 \
EPOCH_SAVE=1 \
SAVE_EVERY_N_STEPS=0 \
KEEP_LAST_N_CHECKPOINTS=3 \
WANDB_PROJECT=RWKV-13B-SFT \
bash run_13b_sft_zero3_offload.sh
```

### 4. 脚本参数说明

关键参数说明：

- `MODEL_TYPE`、`N_LAYER`、`N_EMBD`、`DIM_FFN`、`VOCAB_SIZE`、`HEAD_SIZE`、`D_DECAY_LORA`、`D_AAA_LORA`、`D_MV_LORA`、`D_GATE_LORA`：模型结构参数，默认值对应 RWKV7 G1F 13.3B。换 checkpoint 时必须和对应架构文本一致。
- `LOAD_MODEL`：初始 13.3B checkpoint；也可以指向保存出的 `rwkv-step-N.pth` / `rwkv-N.pth` 做断点续训。DeepSpeed checkpoint 目录需要配合 DeepSpeed strategy 恢复。
- `DATA_FILE`：SFT binidx 前缀，不带 `.bin` 或 `.idx`。脚本会检查 `DATA_FILE.bin`、`DATA_FILE.idx`、`DATA_FILE.mask.bin`、`DATA_FILE.mask.idx`。
- `CTX_LEN`：训练上下文长度，需要和预处理 `--ctx-len` 一致；预处理产物里的 document 应该是 `CTX_LEN + 1` 个 token。
- `N_NODE`、`GPU_PER_NODE`、`MICRO_BSZ`：决定每次 forward 的真实全局 batch size：`real_bsz = N_NODE * GPU_PER_NODE * MICRO_BSZ`。
- `ACCUMULATE_GRAD_BATCHES`：梯度累计步数。SFT 常用它在 `MICRO_BSZ=1` 的情况下提高有效 batch；有效 batch 为 `effective_bsz = real_bsz * ACCUMULATE_GRAD_BATCHES`。
- `EPOCH_STEPS`：每个 SFT epoch 的 optimizer step 数。完整跑一遍建议用 `ceil(num_sft_documents / effective_bsz)`。
- `EPOCH_COUNT`：跑几遍 SFT 数据。想跑 `N` 遍时，`EPOCH_STEPS` 按一遍数据计算，`EPOCH_COUNT=N`。
- `SFT_ONE_PASS`：设为 `1` 时，脚本仍会传入 `EPOCH_STEPS/EPOCH_COUNT` 作为整数占位值，但 `train.py` 会自动读取 `DATA_FILE.idx` 的 document 数并覆盖为 `ceil(num_documents / effective_bsz)` 和 `epoch_count=1`，适合只想完整跑一遍数据的场景。直接调用 `train.py` 时可以省略 `--epoch_steps/--epoch_count`；通过这个脚本调用时不用管它们的默认值。
- `SFT_EVAL_TAIL_RATIO` / `SFT_EVAL_TAIL_DOCS`：从同一份 SFT binidx 尾部切出 eval split。`SFT_EVAL_TAIL_DOCS` 为正数时优先，否则按 ratio 向上取整；不需要额外生成第二份 eval binidx。
- `SFT_EVAL_INCLUDE_IN_TRAIN`：`0` 表示 held-out eval，尾部 eval documents 会从训练集中排除，one-pass step 也按 train documents 计算，此时 tail ratio 必须小于 `1`；`1` 表示 overlap 模式，训练仍使用全量 documents，eval 只作为固定尾部监控集，此时可以设 `SFT_EVAL_TAIL_RATIO=1` 让 eval 也覆盖全量数据。
- `SFT_EVAL_EVERY_N_STEPS` / `SFT_EVAL_STEPS`：每隔 N 个 optimizer step 跑一次 SFT eval，每次每个 rank 跑指定数量的 eval micro-batch。如果同一个 step 同时命中保存和 eval，两者都会执行。eval 会写入 `train_log.txt` 和 wandb 的 `eval/loss`、`eval/ppl`、`eval/mask_tokens`、`eval/docs`。
- `SFT_TRAIN_SHUFFLE` / `SFT_TRAIN_SHUFFLE_SEED`：训练阶段 SFT document shuffle。默认 `0` 和 `1234`；设为 `SFT_TRAIN_SHUFFLE=1` 后，会在不改变 DataLoader 多卡分片方式的前提下，对训练 document 做按 epoch 固定的确定性 permutation。如果启用 tail eval，会先用同一个 seed 对全量 document 做固定 permutation 再切 train/eval，所以 held-out eval 固定但不是原始文件尾部。
- `GRAD_CP`：激活检查点。`1` 表示对 block 开启 checkpointing，省显存但更慢；显存足够时可设 `0`。
- `SFT_MASKED_CE_CHUNK`：SFT masked loss 的实验性 head/CE 分块大小。生产默认保持 `0`，使用完整 logits masked CE；`0` 本身没有额外分块效率损耗，但会物化完整 logits。正数会只对 mask=1 的 target token 分块计算 head 和 CE，目前在 13B ZeRO-3 长上下文下会 timeout，暂不推荐生产使用。
- `SFT_MASKED_FUSED_CE_CHUNK`：新的 CUDA fused masked head CE 内部 chunk 行数。13B 脚本默认 `4096`，这是当前 ctx86016 SFT 的推荐生产路径；若长跑偶发 OOM，可先降到 `2048`。不要和 `SFT_MASKED_CE_CHUNK` 同时设为正数。
- `STRATEGY`：默认 `deepspeed_stage_3_offload`，更省显存；显存足够时可以用 `deepspeed_stage_3` 做纯 ZeRO-3。
- `LR_INIT`、`LR_FINAL`、`WARMUP_STEPS`、`WEIGHT_DECAY`：SFT 学习率计划和正则参数。默认 `LR_WSD_DECAY_ITERS=0` 时，warmup 后保持 `LR_INIT`；设置 `LR_WSD_DECAY_ITERS=K` 后，最后 `K` 个 optimizer step 会按 `LR_WSD_DECAY_STYLE=cosine|linear` 衰减到 `LR_FINAL`。
- 断点续训 LR：从 DeepSpeed/Lightning checkpoint 恢复时，`trainer.global_step` 会恢复，WSD 会按恢复后的 step 继续衰减。恢复时不要随意改 `EPOCH_STEPS/EPOCH_COUNT/LR_WSD_DECAY_ITERS/LR_WSD_DECAY_STYLE/LR_INIT/LR_FINAL`，否则后续 LR 曲线会按新的配置重新解释当前 step。
- `EPOCH_SAVE`、`SAVE_EVERY_N_STEPS`、`KEEP_LAST_N_CHECKPOINTS`：checkpoint 保存频率和保留数量。
- `PROJ_DIR`：训练日志和 checkpoint 输出目录。
- `WANDB_PROJECT`：空字符串表示不启用 wandb；非空则记录到对应项目。
- wandb 训练指标：`train/loss` 是当前 step loss，`train/epoch_loss` 是当前 epoch 内累计平均 loss，`train/lr` / `train/weight_decay` 是当前优化器参数，`train/grad_norm` 是可获取时的全局梯度范数，`train/samples` 是累计训练样本数，`train/tokens` / `train/tokens_b` 是累计训练 token 数及其十亿 token 视图。吞吐相关指标写在 `perf/*` 下，包括 `perf/iteration_time_sec`、`perf/optimizer_steps_per_sec`、`perf/tokens_per_sec`、`perf/ktokens_per_sec` 和 `perf/samples_per_sec`。
- `KERNEL`：RWKV7 CUDA kernel 选择，默认 `@rwkv3`。
- `HEAD_CHUNK`：head 分块设置，默认 `0`，一般先保持默认。
- `DS_BUCKET_MB`：DeepSpeed all-gather / reduce-scatter bucket 大小，13B 脚本默认 `64` MB；实测 bucket-only `128` 没有提速，反而略慢。
- `DS_OFFLOAD_PIN_MEMORY`：DeepSpeed offload 的 `pin_memory` 开关。`-1` 表示保留 strategy 默认值，`0/1` 表示强制关闭/开启。13B 脚本默认设为 `1`，用于减少 CPU offload H2D/D2H 拷贝等待。
- `DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD`：写入 DeepSpeed `stage3_param_persistence_threshold`，单位是参数元素个数，不是字节。13B 脚本默认 `0`，也就是不额外常驻小参数，保留更健康的显存余量。
- `DS_STAGE3_PREFETCH_BUCKET_SIZE`：写入 DeepSpeed `stage3_prefetch_bucket_size`，单位是参数元素个数。13B 脚本默认 `5000000`，这是当前 A/B 中 lowmem 推荐组合的一部分。
- `DS_STAGE3_MAX_LIVE_PARAMETERS`：写入 DeepSpeed `stage3_max_live_parameters`，单位是参数元素个数。13B 脚本默认 `200000000`，用于限制 live 参数量并控制显存峰值。
- `MASTER_ADDR`、`MASTER_PORT`、`CUDA_VISIBLE_DEVICES`：单机多卡 torchrun / distributed 初始化相关参数。
- `TORCH_EXTENSIONS_DIR`、`TORCH_CUDA_ARCH_LIST`、`MAX_JOBS`：CUDA 扩展编译缓存、架构和并行编译设置。H800 常用 `TORCH_CUDA_ARCH_LIST=9.0`。

### 5. 断点续训命令

保存出的 step checkpoint 在 `PROJ_DIR/rwkv-step-N.pth`；DeepSpeed strategy 下它是一个目录，里面包含 ZeRO 分片和 trainer state。续训时把 `LOAD_MODEL` 指向这个目录，其他训练形状、数据、batch、LR/WSD、DeepSpeed strategy 尽量保持和原 run 一致：

```bash
LOAD_MODEL=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.pth \
DATA_FILE=/mnt/data/datasets/sft_train_ctx8192 \
PROJ_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload \
CTX_LEN=8192 \
N_NODE=1 \
GPU_PER_NODE=8 \
MICRO_BSZ=1 \
ACCUMULATE_GRAD_BATCHES=4 \
EPOCH_STEPS=12500 \
EPOCH_COUNT=1 \
SFT_ONE_PASS=0 \
STRATEGY=deepspeed_stage_3_offload \
GRAD_CP=1 \
LR_INIT=1e-5 \
LR_FINAL=1e-6 \
LR_WSD_DECAY_ITERS=1000 \
LR_WSD_DECAY_STYLE=cosine \
WARMUP_STEPS=10 \
SAVE_EVERY_N_STEPS=1000 \
KEEP_LAST_N_CHECKPOINTS=3 \
WANDB_PROJECT=RWKV-13B-SFT \
bash run_13b_sft_zero3_offload.sh
```

如果原训练使用 `SFT_ONE_PASS=1`，续训时也可以继续设 `SFT_ONE_PASS=1`，但仍要保证 `DATA_FILE`、`effective_bsz` 和 LR/WSD 设置没有无意变化。`train.py` 会识别 DeepSpeed checkpoint 目录，并把它作为 Lightning `ckpt_path` 恢复；此时 `epoch_begin` 会被置为 `0`，真正的进度来自 checkpoint 里的 `global_step`。

### 6. 合并 DeepSpeed 分片 checkpoint

训练完成或需要做推理测试时，可以把 DeepSpeed/ZeRO 分片 checkpoint 合并成普通单文件 `.pth`。`--checkpoint-dir` 指向训练保存出的目录，`--output-file` 是合并后的文件；`--summary-file` 会写出参数名、shape、dtype 和总参数量，便于归档：

```bash
python scripts/convert_deepspeed_checkpoint_to_pth.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.pth \
  --output-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.bf16.pth \
  --dtype bf16 \
  --summary-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.summary.txt
```

如果你手里有架构摘要文件，也可以加 `--verify-summary-file model/rwkv7-g1f-13.3b.txt` 做 shape / dtype / 参数总量校验。13.3B 文件很大，建议在服务器上执行合并，并确保输出目录有足够磁盘空间。

### 7. 合并后等价验证

合并后可以用下面的脚本对比“从原 ZeRO checkpoint 重构出的 state_dict”和“合并后的单文件 `.pth`”。`--strict-forward` 会额外跑一次真实 forward；如果只是先检查 tensor 完全一致，可以去掉它：

```bash
python scripts/test_converted_checkpoint_equivalence.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.pth \
  --converted-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.bf16.pth \
  --dtype bf16 \
  --strict-forward \
  --chat-template data/SFT/sample/chat_template.jinja \
  --prompt "你好，请用一句话介绍 RWKV。" \
  --device cuda \
  --demo-vocab-path rwkv_vocab_v20260603.txt \
  --summary-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.equiv.json
```

### 8. 合并后推理测试命令

`scripts/run_converted_rwkv_demo.py` 是轻量 generation demo，会从 `.pth` 自动推断层数、hidden size、LoRA 维度和 head size，不需要手动写 13.3B 结构参数。默认情况下，`--prompt` 表示用户输入，脚本会在内部构造一轮 `user` 消息并用 `data/SFT/sample/chat_template.jinja` 渲染成真正送入模型的 prompt；不需要传入 messages 文件或 messages JSON。SFT 数据处理默认使用 `rwkv_vocab_v20260603.txt`，所以推理测试也建议显式传同一个 vocab：

```bash
python scripts/run_converted_rwkv_demo.py \
  --model-path /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.bf16.pth \
  --vocab-path rwkv_vocab_v20260603.txt \
  --chat-template data/SFT/sample/chat_template.jinja \
  --prompt "你好，请用一句话介绍 RWKV。" \
  --device cuda \
  --dtype auto \
  --topk 10 \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --top-p 0.8 \
  --sample
```

这个 demo 只是验证合并后的 checkpoint 能加载、能 forward、能按 SFT chat template 生成。需要指定系统消息时可以加 `--system-prompt`、`--current-date`、`--current-location`；如果只是想做普通 next-token continuation，不经过 chat template，可以加 `--raw-prompt`。思考前缀有三种模式：

- 默认不加额外参数：chat prompt 会加入空的 `<think>\n\n</think>\n\n`，模型从正式回答处开始续写。
- 强制思考：加 `--force-thinking`，prompt 会以打开的 `<think>\n` 结尾，后面的思考内容和 `</think>` 都由模型生成。
- 强制不思考：加 `--no-add-thinking`，prompt 只以 `<|im_start|>Assistant: ` 结尾，不插入 `<think>` 前缀。

`--force-thinking` 和 `--no-add-thinking` 互斥，不能同时使用。

如果想把“合并 DeepSpeed 分片”和“跑 prompt 推理 smoke”放在一条命令里，可以用整合脚本。默认输出文件已存在时会直接复用，避免每次测试 prompt 都重新合并 13B；需要强制重新合并时加 `--force-convert`：

```bash
python scripts/convert_and_run_rwkv_demo.py \
  --checkpoint-dir /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.pth \
  --output-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.bf16.pth \
  --convert-dtype bf16 \
  --summary-file /mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1000.summary.txt \
  --vocab-path rwkv_vocab_v20260603.txt \
  --chat-template data/SFT/sample/chat_template.jinja \
  --prompt "你好，请用一句话介绍 RWKV。" \
  --device cuda \
  --runtime-dtype auto \
  --topk 10 \
  --max-new-tokens 64 \
  --temperature 1.0 \
  --top-p 0.8 \
  --force-thinking \
  --sample
```

如果要强制不思考，把上面的 `--force-thinking` 换成 `--no-add-thinking` 即可。

### 9. 13.3B SFT profiling / ZeRO 诊断

[run_13b_sft_profile.sh](/D:/codes/RWKV-LM-V7-12B-train/run_13b_sft_profile.sh) 是短跑诊断脚本，不用于正式长训。它复用 13.3B / ctx86016 / SFT mask 参数，默认只跑 `PROFILE_STEPS=8` 个 optimizer step，不保存 checkpoint、不上 wandb，并把诊断文件写到 `PROJ_DIR`：

- `train.log`：完整训练日志和 step time。
- `gpu_monitor.csv`：后台 `nvidia-smi` 采样，包含 GPU 利用率、显存、功耗。
- `vmstat.log`：后台 CPU / IO 粗采样；服务器没有 `vmstat` 时自动跳过。
- `nccl.*.log`：当 `NCCL_DEBUG=INFO` 时记录每个进程的 NCCL 初始化和 collective 日志。
- `nsys_sft_profile.nsys-rep` / `nsys_stats.txt`：当 `PROFILE_MODE=nsys` 时生成 CUDA / cuBLAS 时间线和摘要；NCCL 明细主要看 `NCCL_DEBUG_FILE`，部分 nsys 版本也会把 NCCL kernel 作为 CUDA kernel 显示出来。

先跑一个无 nsys 的短诊断，确认真实 step time 和 GPU 利用率：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth \
DATA_FILE=/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
PROFILE_STEPS=20 \
SFT_MASKED_FUSED_CE_CHUNK=4096 \
STRATEGY=deepspeed_stage_3_offload \
bash run_13b_sft_profile.sh
```

再跑一个 nsys 版本，看 CUDA kernel、cuBLAS，以及可能显示出来的 NCCL kernel 在时间线上各占多少。nsys trace 会比较大，建议先用 4-8 step：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth \
DATA_FILE=/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
PROFILE_STEPS=6 \
PROFILE_MODE=nsys \
NCCL_DEBUG=INFO \
SFT_MASKED_FUSED_CE_CHUNK=4096 \
STRATEGY=deepspeed_stage_3_offload \
bash run_13b_sft_profile.sh
```

看结果时可以按下面判断：

- 如果 `gpu_monitor.csv` 里大部分 GPU util 长时间很低，同时 `train.log` step time 很长，通常是 CPU offload、IO、进程同步或通信等待，不是 CUDA 算子本身算不过来。
- 如果 `NCCL_DEBUG_FILE` 里 collective 很密，或者 nsys 时间线/`nsys_stats.txt` 里 `nccl*` kernel 占比高，说明 ZeRO all-gather / reduce-scatter / 通信同步是主要瓶颈。部分 nsys 版本不支持 `--trace=nccl`，脚本默认只用 `cuda,nvtx,osrt,cublas`，这是正常的。
- 如果 `cublas*gemm*`、`rwkv7_*`、`wkv7*` kernel 占大头且 GPU util 高，说明主要瓶颈在模型主干/矩阵乘/自定义 CUDA 算子。
- 如果显存已经接近满，`SFT_MASKED_FUSED_CE_CHUNK` 不要继续加大；偶发 OOM 时先降到 `2048`。`SFT_MASKED_CE_CHUNK` 仍保持 `0`。
- 对比 offload 影响时，只改 `STRATEGY`，其他参数保持一致：`deepspeed_stage_3_offload` 更省显存但可能慢，`deepspeed_stage_3` 更吃显存但能判断 CPU offload 是否是瓶颈。

一次 13.3B / ctx86016 / 8xH800 / ZeRO-3-offload / `SFT_MASKED_FUSED_CE_CHUNK=4096` 的 6-step nsys 结果如下。这里的百分比是多 GPU kernel 累加时间占比，用来排序瓶颈，不等同于单步 wall-clock：

| 类别 | 占比 | 现象 | 优化方向 |
| --- | ---: | --- | --- |
| `wkv7/clampw` forward/backward | 38.7% | RWKV 时序核心最大单项 | 后续若继续写 CUDA，应优先看 `wkv7_cuda` / `rwkv7_clampw_v3` 在 `B=1,T=86016,H=64,head_size=64` 下的专门调优 |
| cuBLASLt GEMM | 37.9% | 模型 Linear 和 head projection/grad GEMM 很重 | 主要靠 batch/ZeRO/显存策略；CE 小 kernel 已不是瓶颈 |
| NCCL kernels | 11.9% | `AllGather` 约 8.1%，`ReduceScatter` 约 2.2%，小 AllGather 很多 | 测 `deepspeed_stage_3` 非 offload；或调大 bucket、开启 pin memory、调 stage3 persistence/prefetch/live 参数 |
| RWKV 自定义 pointwise | 7.4% | tmix/cmix/vres/a_gate 等 | 不是第一优先级，除非 profiler 显示某个 kernel 异常 |
| PyTorch elementwise/reduce/copy | 2.8% | 零散 tensor 操作 | 低优先级 |
| PyTorch LayerNorm | 1.2% | LN 不是主要瓶颈 | fused LN 收益有限，暂不优先 |
| SFT fused masked CE 小 kernel | 0.1% | mask/softmax 小 kernel 很轻 | 已达到避免 full logits OOM/timeout 的目标，继续优化 CE 收益主要只剩 head GEMM/active-row compact |

实际优化优先级表：

| 优先级 | 瓶颈假设 | 证据 | 下一步实验 | 成功信号 | 主要风险 |
| ---: | --- | --- | --- | --- | --- |
| 1 | ZeRO-3 offload / 参数搬运限制了利用率 | NCCL kernels 约 11.9%，小 AllGather 多，D2H/H2D memops 明显，采样 GPU util 约 66% | 对比纯 `deepspeed_stage_3` 和调参后的 `deepspeed_stage_3_offload`；调 `DS_BUCKET_MB`、`DS_OFFLOAD_PIN_MEMORY`、`DS_STAGE3_*` | step time 下降、GPU util 上升且不 OOM；NCCL / memcopy 压力降低 | 纯 ZeRO-3 或更大 bucket 可能超过 80G 显存 |
| 2 | 模型主干 kernel 是真实计算下限 | `wkv7/clampw` 加 cuBLASLt GEMM 约 76.6% 累计 kernel 时间 | ZeRO/offload 调完后再 profile；仍然如此再考虑 `rwkv7_clampw_v3` / 长上下文专门优化 | nsys 仍被 `wkv7/clampw` / GEMM 主导，同时 GPU util 已较高 | CUDA 改动复杂，且无法解决通信等待 |
| 3 | SFT fused masked CE 主要解决显存，不是当前速度瓶颈 | CE 小 kernel 约 0.1%，但 full logits 在 ctx86016 曾有 OOM/timeout 风险 | 保持 `SFT_MASKED_FUSED_CE_CHUNK=4096`；若 OOM 降到 `2048`；跑 op 级和训练级精度 smoke | 不再 full-logits OOM，loss 有限，op loss/grad 误差在容差内 | fused chunk 越大显存峰值越高 |
| 4 | Python / 数据输入不是训练瓶颈 | 这次 profile 里 CPU 和磁盘 IO 不忙 | 暂不优先优化 dataloader；只有当 `gpu_monitor.csv` 显示 GPU 空转且 NCCL/kernel 时间都低时再看 | 训练仍主要受 GPU/通信限制 | 过早优化 dataloader 不会改善 step time |
| 5 | LayerNorm / 小 PyTorch op 不是一阶问题 | LayerNorm 约 1.2%，PyTorch elementwise/reduce/copy 约 2.8% | 暂缓 fused LayerNorm / 小 op 清理 | 后续 profile 排名发生变化时再重看 | 预期收益低 |

这组结果还显示 GPU 采样约 `66%` 平均利用率、显存峰值约 `80.1GiB / 81.6GiB`，CPU 和磁盘 IO 基本不忙；GPU memops 中 D2H/H2D 拷贝很多，说明 offload 和 ZeRO 参数流动确实在消耗时间。下一步优先比较纯 ZeRO-3 和调参后的 ZeRO-3-offload：

DeepSpeed 参数 A/B 短测结果如下，所有组都使用 13.3B / ctx86016 / 8xH800 / `deepspeed_stage_3_offload` / `SFT_MASKED_FUSED_CE_CHUNK=4096` / `PROFILE_STEPS=20`。`tail avg` 统计 step 10 之后的 tqdm 指标；`active GPU util` 只统计训练活跃采样。这里的 `tuned bucket128` 不是 bucket-only 对照，它同时调大了 stage3 persistence/prefetch/live 参数，所以不能把它的高显存全部归因于 `DS_BUCKET_MB=128`。

| 组名 | `DS_BUCKET_MB` | `pin_memory` | `param_persistence` | `prefetch_bucket` | `max_live` | tail avg s/it | tail avg Kt/s | active GPU util | peak VRAM | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ds-baseline-default` | 64 | default/false | default | default | default | 38.170 | 18.700 | 90.5% | 80077 MiB | 慢且显存高，作为基准 |
| `ds-offload-lowmem` | 32 | 1 | 0 | 5000000 | 200000000 | 35.682 | 19.655 | 96.4% | 78615 MiB | 稳、省显存，比基准快约 6.5% |
| `ds-offload-lowmem-bucket64` | 64 | 1 | 0 | 5000000 | 200000000 | 35.541 | 19.809 | 96.8% | 78675 MiB | 当前推荐，较 bucket32 只多约 60 MiB 峰值显存 |
| `ds-offload-lowmem-bucket128` | 128 | 1 | 0 | 5000000 | 200000000 | 36.196 | 19.409 | 96.3% | 78815 MiB | bucket-only 128 没有提速，反而比 64 慢 |
| `ds-offload-tuned` | 128 | 1 | 100000 | 20000000 | 1000000000 | 35.343 | 20.045 | 95.6% | 80129 MiB | 略快但显存很紧，不适合直接长跑默认 |

当前建议的长跑配置：

```bash
DS_BUCKET_MB=64
DS_OFFLOAD_PIN_MEMORY=1
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD=0
DS_STAGE3_PREFETCH_BUCKET_SIZE=5000000
DS_STAGE3_MAX_LIVE_PARAMETERS=200000000
SFT_MASKED_FUSED_CE_CHUNK=4096
```

这组 bucket-only 对照说明，`DS_BUCKET_MB=128` 本身只比 64 多约 `140 MiB` 峰值显存，但 tail avg s/it 从 `35.541` 变成 `36.196`，吞吐反而下降。`ds-offload-tuned` 的快主要来自更激进的 stage3 persistence / prefetch / live 参数组合，不是单纯来自 bucket=128；但它显存峰值已到 `80129 MiB`，不适合直接作为 20 天长跑默认。

如果后续还想探索更激进配置，可以只在短测里试，不建议直接长跑：

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth \
DATA_FILE=/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
PROFILE_STEPS=20 \
STRATEGY=deepspeed_stage_3_offload \
SFT_MASKED_FUSED_CE_CHUNK=4096 \
DS_BUCKET_MB=128 \
DS_OFFLOAD_PIN_MEMORY=1 \
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD=100000 \
DS_STAGE3_PREFETCH_BUCKET_SIZE=10000000 \
DS_STAGE3_MAX_LIVE_PARAMETERS=500000000 \
RUN_TAG=ds-offload-mid-tuned \
bash run_13b_sft_profile.sh
```

```bash
LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth \
DATA_FILE=/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
PROFILE_STEPS=10 \
STRATEGY=deepspeed_stage_3 \
SFT_MASKED_FUSED_CE_CHUNK=4096 \
RUN_TAG=zero3-no-offload \
bash run_13b_sft_profile.sh

LOAD_MODEL=/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth \
DATA_FILE=/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B \
PROFILE_STEPS=10 \
STRATEGY=deepspeed_stage_3_offload \
SFT_MASKED_FUSED_CE_CHUNK=4096 \
DS_BUCKET_MB=128 \
DS_OFFLOAD_PIN_MEMORY=1 \
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD=100000 \
DS_STAGE3_PREFETCH_BUCKET_SIZE=20000000 \
DS_STAGE3_MAX_LIVE_PARAMETERS=1000000000 \
RUN_TAG=zero3-offload-tuned \
bash run_13b_sft_profile.sh
```

### 10. 服务器 smoke 测试命令

服务器上可以打开可选 CUDA smoke 测试：

```bash
RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_CUDA_SFT_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_CUDA_SFT_ACCUM_EQUIV_SMOKE=1 \
RWKV_SFT_ACCUM_EQUIV_SUMMARY_FILE=/tmp/rwkv_sft_accum_equiv.json \
pytest -q tests/test_sft_cuda_smoke.py::test_cuda_sft_gradient_accumulation_loss_matches_large_batch

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_DP_ZERO_ACCUM_EQUIV_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_DP_ZERO_ACCUM_EQUIV_SUMMARY_FILE=/tmp/rwkv_sft_dp_zero_accum_equiv.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_accumulation_loss_matches_large_micro_batch

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_RESUME_SMOKE=1 \
pytest -q tests/test_sft_cuda_smoke.py

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_WSD_RESUME_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_WSD_RESUME_EPOCH_STEPS=33 \
RWKV_SFT_WSD_RESUME_WARMUP_STEPS=4 \
RWKV_SFT_WSD_RESUME_DECAY_ITERS=9 \
RWKV_SFT_WSD_RESUME_SUMMARY_FILE=/tmp/rwkv_sft_wsd_resume.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_resume_keeps_wsd_lr_position

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_checkpoint_converts_to_pth

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_EVAL_WANDB_EQUIV_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_EVAL_WANDB_EQUIV_SUMMARY_FILE=/tmp/rwkv_sft_eval_wandb_equiv.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_eval_wandb_loss_matches_baseline

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_TAIL_EVAL_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_SHUFFLE_TAIL_EVAL_SUMMARY_FILE=/tmp/rwkv_sft_shuffle_tail_eval.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_shuffle_tail_eval_smoke

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_RESUME_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_SHUFFLE_RESUME_SUMMARY_FILE=/tmp/rwkv_sft_shuffle_resume.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_shuffle_resume_smoke

RWKV_SFT_SMOKE_MODEL=model/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
RWKV_RUN_TRAIN_PY_SFT_SHUFFLE_EVAL_WANDB_EQUIV_SMOKE=1 \
RWKV_SFT_SMOKE_DEVICES=8 \
RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3_offload \
RWKV_SFT_SHUFFLE_EVAL_WANDB_EQUIV_SUMMARY_FILE=/tmp/rwkv_sft_shuffle_eval_wandb_equiv.json \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_shuffle_eval_wandb_loss_matches_baseline
```

第一条命令在进程内跑 CUDA forward/backward，验证 SFT masked loss。第二条命令对比同一批合成 SFT 样本在“大 batch 一次 forward”和“拆成多个 micro-batch 后按梯度累计口径聚合”时的 masked loss 精度，默认对比 `micro_bsz=2, accumulate=1` 与 `micro_bsz=1, accumulate=2` 的 loss，容差为 `RWKV_SFT_ACCUM_EQUIV_ATOL=1e-2`、`RWKV_SFT_ACCUM_EQUIV_RTOL=1e-3`，并可把差值写到 `RWKV_SFT_ACCUM_EQUIV_SUMMARY_FILE`。第三条命令真实启动 `train.py` + DeepSpeed/ZeRO 多卡训练两次，对比 DP/ZeRO 下 `micro_bsz=2, accumulate=1` 和 `micro_bsz=1, accumulate=2` 的首个 epoch loss，容差为 `RWKV_SFT_DP_ZERO_ACCUM_EQUIV_ATOL=1e-2`、`RWKV_SFT_DP_ZERO_ACCUM_EQUIV_RTOL=1e-3`，并可写出 `RWKV_SFT_DP_ZERO_ACCUM_EQUIV_SUMMARY_FILE`。第四条命令启动 `train.py` 跑 1 个 SFT step，额外覆盖 Lightning、DeepSpeed 和 optimizer 链路。第五条命令会先保存 `rwkv-step-1.pth`，再从这个 step checkpoint 恢复，覆盖 SFT 断点续训；使用 DeepSpeed strategy 时，也会覆盖 DeepSpeed 分片 checkpoint 的加载。第六条命令专门验证 WSD LR 断点续训：默认 `epoch_steps=33`、`warmup_steps=4`、`lr_wsd_decay_iters=9`，也就是 step 0-3 warmup、step 4-23 保持 `lr_init`、step 24 进入 WSD decay，step 28 衰减到一半并保存 checkpoint，resume 后继续到 step 32，检查最后记录的 LR 到 `lr_final`；这避免了“还在 warmup”或“一开始就在 decay”的假阳性，也覆盖了衰减中途断点。第七条命令会先产出一个 tiny SFT DeepSpeed checkpoint，然后调用合并脚本把分片 checkpoint 目录转成单文件 `.pth`，再加载并和原 ZeRO checkpoint 重构结果做等价性比较。第八条命令对比“不开 eval/wandb”和“开启 tail eval + fake wandb”的训练 loss 是否对齐；这里 eval 使用 `sft_eval_include_in_train=1` 的重叠模式，所以训练样本数量和顺序不变，fake wandb 只把 `wandb.init`、`train/loss`、`eval/loss` 写到临时 jsonl，不需要服务器登录 wandb，也不会访问网络。多卡服务器可以加 `RWKV_SFT_SMOKE_DEVICES=8`，`train.py` 会自动用 torchrun 重启多卡 DeepSpeed；也可以设置 `RWKV_SFT_SMOKE_STRATEGY=deepspeed_stage_3` 或 `deepspeed_stage_3_offload` 来验证不同分片模式。

最后三条 shuffle smoke 专门覆盖训练阶段 document shuffle：第一条验证 held-out eval split 会从 seed 固定的全量 document permutation 里切出，而不是原始尾部，并检查多卡 DeepSpeed 能正常 eval/save；第二条在 shuffle + held-out eval 下保存 DeepSpeed step checkpoint 并恢复，检查 dataloader step offset 和 trainer state resume；第三条对比“shuffle 训练 + overlap eval/wandb”和同样 shuffle 训练但不开 eval/wandb 的 epoch loss，确认 eval/wandb 不改变训练样本流。

如果要直接测试已有 checkpoint 目录的 pth 合并，而不是先训练 tiny checkpoint：

```bash
RWKV_RUN_TRAIN_PY_SFT_MERGE_SMOKE=1 \
RWKV_SFT_MERGE_CHECKPOINT_DIR=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.pth \
RWKV_SFT_MERGE_OUTPUT_FILE=/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload/rwkv-step-1.bf16.pth \
RWKV_SFT_MERGE_STRICT_FORWARD=1 \
pytest -q tests/test_sft_cuda_smoke.py::test_train_py_sft_deepspeed_checkpoint_converts_to_pth
```

### 为指定 binidx 数据集计算 magic_prime

`data/compute_magic_prime.py` 脚本可为指定的 binidx 数据集和上下文长度（ctx_len）计算正确的 `--my_exit_tokens` 和 `--magic_prime` 值。

1. 在 `data/compute_magic_prime.py` 中修改你的训练数据集和上下文长度（`DATA_NAME` 和 `CTX_LEN`）
2. 运行脚本以获得正确的 `--my_exit_tokens` 和 `--magic_prime` 值

```
cd data/
python compute_magic_prime.py
```

最终输出类似于：

```
### Loading /home/rwkv/RWKV-LM-V7/data/demo

### /home/rwkv/RWKV-LM-V7/data/demo.bin/idx has 200499 tokens, 546 items. Dtype <class 'numpy.uint16'>

### magic_prime = 47 (for ctxlen 4096)

--my_exit_tokens 200499 --magic_prime 47 --ctx_len 4096
```
