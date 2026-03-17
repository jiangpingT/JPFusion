# JPFusion — 项目知识库（阿策永久记忆）

> 当前状态、版本号 → 见 `~/.claude/projects/.../memory/MEMORY.md`
> 本文件记录：历史里程碑、核心规律、血泪教训、代码片段

---

## 项目定位

托卡马克等离子体 RL 控制四阶段流水线：
```
Phase 1 (PPO) → Phase 2 (SFT/BC) → Phase 3 (CQL) → Phase 4 (MBRL+Dyna)
     ↑                                    ↑
     └── SFT 热启动（辅助）                └── Phase 4 真实轨迹回流（飞轮）
```

目标：mean_reward ≥ 200,000，lawson ≥ 99.5%

---

## 历史里程碑

| 时间 | 版本/事件 | mean_reward | lawson | 备注 |
|------|---------|------------|--------|------|
| 20260228 | run1 PPO | ~185,000 | - | 首次突破 |
| 20260301 | run2-3 | 193,527 | 99% | 三轮积累最优，已丢失 |
| 20260302 | iter2 CQL v3 | -2,427 | - | 飞轮崩溃，Q值爆炸 |
| 20260302 | Recovery | 17,417 | 100% | 复现失败 |
| 20260302 | 新 Stage4b | 192,957 | 100% | 阶段冠军 |
| 20260303 | Stage4b Run2 | **195,815** | **100%** | 最终冠军，Demo 就绪 |

---

## 血泪教训 — Apple Silicon MPS

### ★ d3rlpy 在 MPS 上 epoch 边界 silent crash

**现象**：`nohup python cql_expert_train.py` 跑到第 1-2 个 epoch 后进程静默死亡，日志无任何报错，只有 Gym 弃用警告。

**根因**：d3rlpy（及很多非 PyTorch-native 的 ML 库）对 MPS backend 支持不完整。epoch 结束时的 cleanup/callback 操作触发 MPS 内部错误，macOS 直接 kill 进程，无 stderr 输出。

**诊断思路**（遇到 silent crash 的通用流程）：
1. 进程总在同一位置死 → 不是随机 OOM，是确定性触发
2. 加 `-u` 参数（无缓冲）确认不是缓冲问题
3. 怀疑硬件后端 → 显式指定 `device='cpu'` 测试
4. 确认后加 FileAdapter 替代 NoopAdapter，获取详细日志

**修复**：
```python
# cql_expert_train.py
cql = d3rlpy.algos.CQLConfig(...).create(device='cpu')  # ★ 强制 CPU
```

**通用规则**：Apple Silicon 机器，凡 non-PyTorch-native 库（d3rlpy、stable-retro 等），
训练循环中**默认显式加 `device='cpu'`**，MPS 只用于 PyTorch 原生代码且充分测试后。

---

## 血泪教训 — CQL Q值爆炸

**现象**：Phase 4 高奖励轨迹（111K reward，约 222/step）与 Phase 1 数据（约 180/step）合并后，
CQL critic_loss 从 533K → 17M → 170B → 1.61e17（10 epochs 内发散）。

**根因**：Bellman bootstrap `Q(s,a) = r + γQ(s')` 在高奖励数据上指数放大。

**三步修复**（已写入 `cql_expert_train.py`）：
```python
# 1. Z-score 奖励归一化
rew_norm = (rew - rew.mean()) / (rew.std() + 1e-8)

# 2. 降低 conservative_weight（归一化后量级~1.0，5.0太强→actor发散）
conservative_weight=1.0   # 原 5.0

# 3. 梯度裁剪
actor_optim_factory=d3rlpy.optimizers.AdamFactory(clip_grad_norm=10.0)
critic_optim_factory=d3rlpy.optimizers.AdamFactory(clip_grad_norm=10.0)
```

---

## 血泪教训 — Phase 1 PPO 崩塌

**现象**：PPO 在 step 310K 达到 best（191K），step 920K 时崩塌到 -589，final.zip 保存崩塌状态。

**修复**：`train_rl.py` 的 `FusionRLCallback` 加 `best.zip` 保存机制：
- 每次评估若 mean_reward 刷新最高，保存 `{save_dir}/best.zip`
- `cql_expert_train.py` 优先加载 `best.zip`（忽略可能崩塌的 `final.zip`）

---

## 设计缺口 — SFT 质量门槛

**问题**：pipeline_v2.py Stage 0 只检查 SFT 文件存在，不管质量。v6 SFT（mean_reward=24,806，方差57K）
虽然文件存在，但实际是烂热启动，给 PPO 提供了错误先验。

**修复**（已写入 `pipeline_v2.py` Stage 0）：
- SFT 存在时，用 BC actor 在 FusionEnv 跑 8 episodes
- `mean_reward < 20,000` → 自动降级为随机初始化，打印警告

---

## 飞轮闭环说明

**完整飞轮**（已修复）：
```
Stage 1: Phase 1 PPO（1M steps，best.zip 保存）
Stage 2: CQL v2（Phase 1 专家轨迹，device='cpu'）
Stage 3: BC 蒸馏（CQL → SB3 格式）
Stage 4a: MBRL DeepONet（世界模型 + 100次 Dyna）
Stage 4b: 真实精调（300K steps，输出 finetuned.zip + buffer.npz）
Stage 5: CQL v3（Phase1 + Phase4 回流数据，device='cpu'，奖励归一化）
→ 冠军挑战（提升≥0.5%才更新）
→ 下一轮迭代
```

**飞轮断裂历史**：
- iter2：CQL v3 Q值爆炸（未修奖励归一化）
- Stage 4b 轨迹未能回流给 CQL v3（MPS crash）→ 已修

---

## 关键文件路径

| 文件 | 用途 |
|------|------|
| `scripts/pipeline_v2.py` | 主流水线入口 |
| `scripts/cql_expert_train.py` | Phase 3 CQL 训练（含所有修复）|
| `scripts/mbrl_expert_train.py` | Phase 4 MBRL + Dyna |
| `scripts/phase4_finetune.py` | Stage 4b 真实精调 |
| `backend/rl/train_rl.py` | Phase 1 PPO（含 best.zip）|
| `backend/rl/bc_pretrain.py` | Phase 2 SFT/BC |
| `data/rl_models/p4_mbrl/champion.json` | 冠军记录 |

---

## CQL 实测稳定性数据

CQL v2 20-episode 评估结果（2026-03-03）：
- mean_reward: **24,806 ± 57,801**（极不稳定，std > mean）
- lawson_rate: 100%
- 单次推理 197K 是 +3σ 的偶发值，不代表真实水平

对比 MBRL v3（100-episode）：193,527 ± 19,675（稳定）

---

---

## 项目复盘 — 完整错误清单（2026-03-04）

> 从立项到完成，按问题类别分组。每条：现象 → 根因 → 教训。

---

### 一、模型文件管理：反复丢失最优权重

**错误 1：final.zip 保存崩塌状态**
- 现象：PPO 在 step 310K 达到 best（191K），继续训到 920K 时崩塌至 -589，final.zip 保存了崩塌后的状态。
- 根因：训练脚本只保存最后一步，没有"当前最优"快照机制。
- 教训：**训练脚本必须在第一天就加 best.zip 机制**，final.zip 只作调试用，绝不作生产输入。

**错误 2：finetuned.zip 被新一轮训练覆盖**
- 现象：Stage 4b 精调后保存为 p4_mbrl_v3_finetuned.zip。下一轮迭代（84K 烂模型）再次精调，直接覆盖了上一个 192K 冠军，冠军永久丢失。
- 根因：没有带分数命名的 archive 机制，finetuned.zip 是个"易碎"文件。
- 教训：**每次精调完立刻 copy 一份 archive_timestamp_rewardXXX.zip**，finetuned.zip 只是快捷入口，不是安全存档。（已在 phase4_finetune.py 修复）

**错误 3：run2-3 积累的 193,527 历史最优丢失**
- 现象：三轮迭代积累出的最优成果，因为没有 archive 机制，在某轮覆盖后永久消失。
- 根因：同上，结合：pipeline 没有在每次超越历史最优时自动存档的逻辑。
- 教训：**冠军挑战者机制应该是 pipeline 第一个组件**，而不是事后补救。

---

### 二、错误热启动：从错误的模型出发

**错误 4：Stage 4b 从 finetuned.zip 而非 archive 启动**
- 现象：Run 1 精调前评估只有 88K，远低于期望的 190K。排查后发现 finetuned.zip 已被烂模型覆盖。
- 根因：错误 2 的连锁反应 + 启动命令用了默认路径（finetuned.zip）没有显式指定 archive。
- 教训：**每次重启训练，第一件事是验证 pre-eval 指标是否符合预期**（本次精调前 pre-eval 偏差超 50% 即停止）。

---

### 三、算法稳定性：过训练与方差陷阱

**错误 5：精调前 20-episode 评估的统计欺骗**
- 现象：精调前用 20 episodes 评估得到 198,763 ± 1,491，感觉极好，实际上 100-episode 正式评估只有 194,846 ± 17,261。
- 根因：20 episodes 样本量太小，mean 估计方差大，偶然抽到了高分区间（±1.5K 的 std 本身就是假象）。
- 教训：**pre-eval 至少 50 episodes 才有统计意义**，20 episodes 的结果只能做"模型加载确认"，不能做性能判断。

**错误 6：对高原区持续精调导致退化**
- 现象：策略已在 195K 附近，多次精调（参数从 lr=3e-5/600K 到 lr=5e-6/200K）全部导致最终评估比起点更低。
- 根因：策略已接近环境上限，高方差（±17K）来自 FusionEnv 初始状态分布的多样性，不是策略问题。精调只是在噪声上做梯度下降，不稳定。
- 教训：**mean_reward 高原 + std 大 = 停止精调的信号**。正确方向是减少环境方差（改初始状态采样），而不是换精调参数。

**错误 7：CQL Q 值爆炸**
- 现象：Phase 4 高奖励轨迹（222/step）与 Phase 1 数据（180/step）合并后，CQL critic_loss 在 10 个 epoch 内从 533K 发散到 1.61e17。
- 根因：Bellman bootstrap 在高奖励数据上指数放大，没有奖励归一化。
- 教训：**多来源数据合并前必须做 z-score 归一化**，不同量级数据直接合并必然炸。

---

### 四、工程基础设施：环境与工具链问题

**错误 8：d3rlpy MPS Silent Crash**
- 现象：CQL 训练跑到 1-2 个 epoch 后进程静默死亡，日志无报错。
- 根因：d3rlpy 对 Apple Silicon MPS backend 支持不完整，epoch 边界 cleanup 触发内部错误，macOS 直接 kill 进程。
- 教训：**非 PyTorch-native 库在 Apple Silicon 上默认 `device='cpu'`**，不要假设 MPS 能用。

**错误 9：API 端口混淆（8000 vs 8001）**
- 现象：修复 Dashboard best_reward 后测试 API，一直用 localhost:8000 测试，返回空结果，以为修复没生效。实际上 JPFusion 后端是 8001，8000 是 fusion-platform。
- 根因：两个项目共用相似架构，端口记错了。
- 教训：端口分配要写在 CLAUDE.md 第一行，调试前先确认连的是哪个服务。

**错误 10：champion.json vs v3_results.json 读取错误**
- 现象：Dashboard Phase 4 显示 193.5K，实际冠军是 195.8K。
- 根因：后端 API 读 v3_results.json（pipeline 早期写入，不更新），没读 phase4_finetune.py 写的 champion.json。
- 教训：**多个脚本写不同 JSON 文件时，必须明确哪个是"最权威的当前状态"**，API 只读一个来源。

---

### 五、数据认知：对数据的错误假设

**错误 11：Harvard Dataverse 链接是 404 占位符**
- 现象：east_loader.py 中写了 `doi:10.7910/DVN/OQTQXJ`，实际访问返回 404。我们一度以为是真实数据源。
- 根因：代码写注释时用了一个虚构/过期的链接，没有验证。
- 教训：**代码里的外部数据链接必须验证可访问性**，占位符 DOI 会让人误以为数据存在。

**错误 12：ai4plasma 仓库误认为 EAST 数据**
- 现象：误认为 ai4plasma 仓库包含 EAST 托卡马克放电数据，花时间深入研究。实际上是 SF6/N2/Ar 弧放电数据（低温等离子体），完全不同领域。
- 根因：看到"plasma + AI"就以为是核聚变，没有仔细看数据文件内容。
- 教训：**等离子体物理 ≠ 核聚变**。低温等离子体（弧放电、刻蚀）和高温等离子体（托卡马克）是两个完全不同的方向，数据不可互用。

**错误 13：Phase 2 设计依赖不存在的 EAST 公开数据**
- 现象：Pipeline 设计了 Phase 2A（τ_E 拟合）和 2B（DTW 验证）两个步骤，前提是有 EAST 历史放电数据。实际上 EAST 数据根本不公开（ASIPP 服务器，需机构合作）。
- 根因：架构设计时假设了数据可获得，没有预先验证数据可及性。
- 教训：**先验证数据，再设计依赖该数据的架构**。数据可及性是系统设计的前置条件，不是实现细节。

---

### 六、目标设定：200K 的执念

**错误 14：200K 目标设定后没有 exit criteria**
- 现象：195,815 时明显已到环境方差上限，仍连续跑了多轮精调尝试突破 200K，每轮都退化。
- 根因：目标数字一旦设定就变成执念，没有提前定义"什么情况下放弃"的判断标准。
- 教训：**量化目标必须同时定义 exit criteria**：若 N 轮精调均未提升，且 std/mean > 阈值，则判定为方差瓶颈，停止精调，转而优化环境。实际成功条件（lawson≥99.5% + reward≥150K）早已达到，执念 200K 浪费了多轮计算。

---

### 总结：三个最值得铭记的教训

| 优先级 | 教训 | 一句话 |
|--------|------|--------|
| ★★★ | 模型存档 | **best.zip + archive_XXX.zip 是第一天就要有的，不是事后补救** |
| ★★★ | 数据先行 | **先验证数据可及性，再设计依赖该数据的架构** |
| ★★ | 统计诚实 | **小样本 pre-eval 是谎言，高方差 + 高原期 = 停止精调的信号** |

---

## 前端服务说明

| 服务 | 端口 | 启动命令 |
|------|------|---------|
| JPFusion 后端 | 8001 | `venv/bin/python -m uvicorn backend.main:app --port 8001` |
| JPFusion 前端 | 3000 | `cd frontend && PORT=3000 npm start` |
| fusion-platform 后端 | 8000 | 已在运行（等离子体推理用）|

RLDashboard BASE_URL 已改为 `http://localhost:8001`。
