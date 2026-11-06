# RWKV-7 2.9B Loss 异常排查结论

## 1. 问题现象

将以下 2.9B 基座用于长上下文 SFT：

```text
/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth
```

在相同训练代码、相同 86K 数据、相同 mask 和相同 CUDA kernel 下，首步 loss 为：

| 模型 | 首步 loss |
|---|---:|
| 2.9B g1j | 10.80 |
| 13.3B g1f | 1.380 |

13.3B 对照模型：

```text
/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth
```

需要确定的是：2.9B 的高 loss 来自模型本身、模型加载、数据、mask、特殊 token、上下文长度，还是 CUDA 前向。

---

## 2. 实验与结果

### 2.1 使用 13.3B 跑完全相同的训练流程

只替换 checkpoint 和对应架构参数，其他数据与训练配置不变：

```text
2.9B loss  = 10.80
13.3B loss = 1.380
```

这说明数据读取、mask 和 loss 实现并不存在会影响所有模型的通用错误。异常与 2.9B checkpoint 的行为有关。

### 2.2 在普通文本上比较 PPL

使用 MiniPile validation、相同旧 tokenizer 和 2048-token 窗口评测 690,761 个 token：

| 模型 | Mean NLL | PPL |
|---|---:|---:|
| 2.9B | 1.8332 | 6.2538 |
| 13.3B | 1.6804 | 5.3677 |

2.9B 只比 13.3B 差约 16.5%，没有出现与 SFT loss `10.x` 对应的整体崩坏。

由此排除：

- checkpoint 完全损坏；
- 加载了随机初始化模型；
- 普通 token 的 tokenizer 映射整体错位。

### 2.3 精确复现训练首批文档

异常训练使用 `sft_train_shuffle=0`，8 个 rank 的首批文档正好是 `0..7`。独立诊断脚本读取同一批 token 和 mask，在 inference 模式下得到：

| 模型 | 全 token loss | Masked loss | 普通 masked token loss |
|---|---:|---:|---:|
| 2.9B | 9.9777 | 10.7661 | 10.1283 |
| 13.3B | 1.2511 | 1.3832 | 1.0015 |

独立评测结果与训练日志一致，证明：

- 模型路径正确；
- 训练 loss 没有把正常值错误计算成 `10.x`；
- 不经过 optimizer 和 backward，也能复现异常。

这里“普通 masked token loss”指：只统计 `mask=1`，并排除目标 ID `65530..65535`。

### 2.4 排查 mask 和特殊输出 token

这 8 篇文档共有 3050 个监督 token，其中新增特殊目标只有 43 个，占 1.41%。

排除特殊目标后，2.9B 普通 token loss 仍为：

```text
10.1283
```

把 mask 全部设为 1 后：

```text
2.9B all-token loss  = 9.9777
13.3B all-token loss = 1.2511
```

另外，在所有 masked 位置中，新增特殊输出 token 成为 top-1 的次数为 0。临时屏蔽输出 logits 的 `65530..65535` 后，普通 token loss 几乎不变。

由此排除：

- SFT mask 是高 loss 的根因；
- 少量特殊目标把均值拉到 `10.x`；
- 新增输出 head 抢占 softmax，导致普通 token loss 异常。

### 2.5 重置 state 和截取后缀

固定相同的监督目标，只从序列后部以全零 state 重新开始：

| 保留的后缀历史 | 2.9B masked loss | 普通 masked loss |
|---:|---:|---:|
| 4K | 1.0948 | 0.3302 |
| 8K | 1.0799 | 0.3188 |
| 16K | 1.0777 | 0.3190 |
| 32K | 1.1461 | 0.3841 |
| 64K | 1.2168 | 0.4611 |
| 完整 86K | 10.7661 | 10.1283 |

这说明异常与序列开头建立的 recurrent state 有关。只要从文档中间的普通 token 重新建立 state，后续很长一段也可以正常预测。

但该实验同时改变了上下文长度和首 token，因此还不能直接认定是“上下文太长”。

### 2.6 删除 System Prompt

每篇文档开头的 System Prompt 长约 7.7K–28.7K token。将它完整删除，但保留后续 User、Assistant 和 Tool 对话：

| 模型 | 条件 | Masked loss | 普通 masked loss |
|---|---|---:|---:|
| 2.9B | 原始 | 10.7661 | 10.1283 |
| 2.9B | 删除 System | 10.9286 | 10.3141 |
| 13.3B | 原始 | 1.3832 | 1.0015 |
| 13.3B | 删除 System | 1.0953 | 0.7041 |

删除 System Prompt 没有修复 2.9B。删除后新的序列仍然以 `<|im_start|>User:` 开头。

由此排除：

- System Prompt 的具体内容是 2.9B 异常的直接原因。

### 2.7 检查长张量 CUDA 因果性

比较同一文档前 2048 token 在以下两种输入中的 hidden：

1. 只输入 2048 token；
2. 输入完整 86K，但只取前 2048 hidden。

结果：

```text
hidden max absolute difference = 0
hidden mean cosine similarity  = 1.0
loss in 2K tensor              = 8.5420
loss in 86K tensor prefix      = 8.5420
```

前缀结果完全一致，说明超长 CUDA forward 没有破坏因果性。高 loss 是模型真实输出，不是 kernel 根据未来 token 错算了前缀。

### 2.8 首 token 最小消融

最后只修改位置 0 的输入条件，其他原始内容和监督目标保持不变。

文档 0 的结果：

| 输入处理 | Masked loss | 普通 masked loss |
|---|---:|---:|
| 原始，以 `<|im_start|>` 开头 | 10.2097 | 9.8629 |
| 首个 `<|im_start|>` 替换成换行 | 1.0873 | 0.6770 |
| 跳过首个 `<|im_start|>` | 1.0846 | 0.6750 |
| 在完整原始序列前增加换行，保留 `<|im_start|>` | 1.0877 | 0.6774 |
| 删除 System，保留新的首个 `<|im_start|>` | 10.3188 | 10.0075 |
| 删除 System，并跳过新的首个 `<|im_start|>` | 1.0773 | 0.6659 |

在全部 8 篇文档上，保留所有原始内容，只在最前面增加普通换行 token：

```text
masked loss:          10.7661 -> 1.2291
ordinary masked loss: 10.1283 -> 0.4728
```

这是决定性的因果证据：

- 不是因为截短了数据；完整内容仍然存在，输入甚至更长；
- `<|im_start|>` 仍然存在；
- 唯一关键变化是 RWKV 不再以近零的 `<|im_start|>` embedding 从全零 state 启动。

---

## 3. 原因分析

### 3.1 2.9B 的 `<|im_start|>` input embedding 近似为零

| Checkpoint | Token | Embedding norm |
|---|---|---:|
| 2.9B | 换行 ID `11` | 0.352938 |
| 2.9B | `<|im_start|>` ID `65530` | `1.4807e-15` |
| 13.3B | 换行 ID `11` | 0.648649 |
| 13.3B | `<|im_start|>` ID `65530` | `9.6864e-8` |

2.9B 的 `<|im_start|>` 输入向量实质上为零，与正常 token embedding 相差巨大。

### 3.2 异常发生在输入 state，不是对应 output logits

需要区分两件事：

- `emb.weight[65530]` 决定 `<|im_start|>` 作为输入时如何影响模型；
- `head.weight[65530]` 决定模型预测 `<|im_start|>` 时的输出 logit。

本次异常主要发生在输入侧：

- 屏蔽新增特殊输出 logits 后，普通 token loss 不变；
- 将首个输入 token 改成普通 token 后，整段普通 token loss 立即恢复。

### 3.3 RWKV state 动力学

RWKV 从全零 recurrent state 开始，后续每个位置都依赖此前累计的 state。

对于 2.9B g1j：

1. 位置 0 输入近零 `<|im_start|>` embedding；
2. checkpoint 从这个特殊启动条件进入异常 state 轨迹；
3. 后续正常 token 未能及时把 state 拉回正常区域；
4. 因此不仅特殊 token，普通文字和代码 token 的 loss 也升到 `10.x`。

从文档中间重置 state 时，第一个 token 通常是已经训练过的文字或代码 token，embedding 处于正常范围，state 因而恢复正常。

### 3.4 为什么 13.3B 没有同样失败

13.3B 使用相同 token ID，但 checkpoint 参数不同：

- 特殊 token embedding 的具体数值不同；
- LayerNorm bias、time decay 和 WKV 权重不同；
- 整体 recurrent dynamics 对近零首 token 更鲁棒。

目前证据不能证明“13.3B 在基模训练中充分见过 `<|im_start|>`”。事实上，它的特殊 token embedding 仍远小于普通 token，特殊目标 NLL 也很高。

严谨结论是：

> 两个基模的训练数据、初始化和权重差异，造成了不同的状态动力学；13.3B 能容忍近零首 token，而 2.9B g1j 不能。不能只把原因归结为参数规模，也不能仅凭现象断言 13.3B 训练过 `<|im_start|>`。

---

## 4. 训练结论

### 4.1 2.9B 可以继续训练，但应先 PT

直接进行稀疏 mask SFT 风险较高，因为监督位置可能距离序列开头很远，修复首 token embedding 的梯度需要穿过很长的 recurrent state。

先使用同一批新-tokenizer 数据做 mask 全为 1 的 PT 更合理：

- 所有位置都有监督；
- `<|im_start|>`、`<|im_end|>` 等特殊 token 会反复出现；
- input embedding 和 output head 都能获得密集梯度；
- PT 有机会先修复特殊 token 和异常 state 启动，再进入 SFT。

### 4.2 当前 PT 配置

脚本：

```text
run_3b_sft_ctx192512_pt.sh
```

关键参数：

```text
initial model = /mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth
data          = /mnt/data/Datasets/SFT_RWKV7_13B_ctx192512_mask_1/results/SFT_RWKV7_13B
ctx_len       = 192512
mask          = all ones
shuffle       = 1, seed 1234
PT lr         = warmup 后保持 4e-6
final lr      = SFT 尾段衰减到 4e-7
warmup        = 100 steps
effective bsz = 32
PT steps      = 1000
save interval = 500 steps
speed estimate= 88 seconds/global optimizer step
```

对应时间约为：

```text
step 500  ~= 12.2 hours
step 1000 ~= 24.4 hours
```

启动：

```bash
screen -S rwkv7-3b-pt
bash run_3b_sft_ctx192512_pt.sh
```

### 4.3 PT 后进行 SFT

脚本：

```text
run_3b_sft_ctx192512_sft-finish-onepass.sh
```

它默认加载：

```text
outs/3b-sft-ctx192512-pt/rwkv-step-1000.pth
```

并恢复模型、optimizer、global step 和相同 seed 的 shuffle 位置，然后切换到普通 SFT mask，继续剩余文档。

启动：

```bash
screen -S rwkv7-3b-sft
bash run_3b_sft_ctx192512_sft-finish-onepass.sh
```

### 4.4 PT 是否修复必须单独验收

不能只用训练进度条上不同 batch 的 loss 判断修复。建议在 step 500 和 step 1000 对固定文档重新评测。

基座参考：

```text
masked loss          = 10.7661
ordinary masked loss = 10.1283
```

前置普通 token 的正常参考：

```text
masked loss          = 1.2291
ordinary masked loss = 0.4728
```

如果 PT checkpoint 在原始、未前置换行的固定批次上已经降到约 `1.x`，说明模型自己学会了正常处理首个 `<|im_start|>`，可以进入 SFT。

如果 step 1000 后仍接近 `8~10`，则不应直接开始长时间 SFT，需要显式处理首 token 或初始化特殊 token embedding。

---

## 5. 关键文件

诊断脚本：

```text
scripts/eval_rwkv_ppl.py
scripts/eval_sft_batch_loss.py
scripts/check_rwkv_checkpoint.py
scripts/convert_deepspeed_checkpoint_to_pth.py
```

训练脚本：

```text
run_3b_sft_ctx192512_pt.sh
run_3b_sft_ctx192512_sft-finish-onepass.sh
```

关键结果：

```text
outs/ppl-minipile-20260916/3b.log
outs/ppl-minipile-20260916/13b.log
outs/sft-loss-diagnosis-20260916/3b-full-binned.json
outs/sft-loss-diagnosis-20260916/13b-full-binned.json
outs/sft-loss-diagnosis-20260916/3b-no-system.json
outs/sft-loss-diagnosis-20260916/13b-no-system.json
outs/sft-loss-diagnosis-20260916/3b-suffix4k.json
outs/sft-loss-diagnosis-20260916/3b-suffix8k.json
outs/sft-loss-diagnosis-20260916/3b-suffix16k.json
outs/sft-loss-diagnosis-20260916/3b-suffix32k.json
outs/sft-loss-diagnosis-20260916/3b-suffix64k.json
```

所有诊断中的删除、截取、前置 token 和 mask 变换均只发生在内存中，原始训练数据未被修改。
