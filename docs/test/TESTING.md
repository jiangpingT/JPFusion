# 测试文档

> FusionLab 测试策略、用例和验收标准。

---

## 测试策略

FusionLab 采用**三层测试**：

| 层次 | 工具 | 频率 | 覆盖范围 |
|------|------|------|----------|
| **单元测试** | Python 手动验证脚本 | 每次修改后 | 物理函数、奖励计算、破裂检测 |
| **集成测试** | curl / httpx | 每次 API 变更后 | API 端点、WebSocket |
| **端到端测试** | 训练验收标准 | 每个 Phase 完成后 | RL 训练效果、gaming 检测 |

> 当前阶段无自动化测试框架（无 pytest 套件），以手动验证脚本为主。
> 未来计划引入 pytest + hypothesis（属性测试）。

---

## 单元测试

### T-001：dynamics.py — ODE 动力学

**验证目标**：step_plasma_state 输出合法，归一化范围 [0, 1]

```python
source venv/bin/activate
python -c "
from backend.rl.dynamics import step_plasma_state, denormalize_state
import numpy as np

state  = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
action = np.array([0.05, 0.01, -0.02], dtype=np.float32)
next_s = step_plasma_state(state, action)

assert next_s.shape == (7,), f'形状错误：{next_s.shape}'
assert np.all(next_s >= 0.0) and np.all(next_s <= 1.0), f'超出 [0,1]：{next_s}'
s_dict = denormalize_state(next_s)
assert s_dict['q95'] > 0, f'q95 非正：{s_dict[\"q95\"]}'
print('[PASS] dynamics.step_plasma_state OK')
print(f'  q95={s_dict[\"q95\"]:.3f}, P_heat={s_dict[\"P_heat\"]/1e6:.2f} MW')
"
```

**期望结果**：`[PASS] dynamics.step_plasma_state OK`

---

### T-002：disruption.py — 破裂检测

**验证目标**：4 个破裂条件分别能被触发

```python
python -c "
from backend.rl.disruption import check_disruption
import numpy as np

# 正常状态（不破裂）
safe = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
disrupted, reason = check_disruption(safe)
assert not disrupted, f'正常状态误判为破裂：{reason}'
print('[PASS] 正常状态不破裂')

# 低 q95（Kruskal-Shafranov）
low_q95 = safe.copy()
low_q95[3] = 0.0  # q95_norm = 0 → q95 = 2.0（临界）
d, r = check_disruption(low_q95)
print(f'低 q95 破裂={d}, 原因={r}')

# 高 beta_N（Troyon 极限）
high_beta = safe.copy()
high_beta[4] = 1.0  # beta_N_norm = 1.0 → beta_N = 4.0 > 3.5
d, r = check_disruption(high_beta)
print(f'高 beta_N 破裂={d}, 原因={r}')

print('[PASS] disruption.check_disruption OK')
"
```

---

### T-003：rewards.py — 奖励函数

**验证目标**：奖励分量正负符合预期，无 per-step 正奖励（在低 Lawson 状态下）

```python
python -c "
from backend.rl.rewards import compute_reward, compute_disruption_penalty
import numpy as np

state  = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
action = np.zeros(3, dtype=np.float32)

rew, info = compute_reward(state, action)
print(f'总奖励：{rew:.2f}')
print(f'  Lawson 奖励：   {info[\"lawson_reward\"]:.2f}')
print(f'  效率惩罚：      {info[\"efficiency_penalty\"]:.2f}')
print(f'  稳定性惩罚：    {info[\"stability_penalty\"]:.2f}')
print(f'  成功加成：      {info[\"success_bonus\"]:.2f}')

# 效率惩罚必须为负
assert info['efficiency_penalty'] < 0, '效率惩罚应为负值！'
print('[PASS] rewards.compute_reward OK')

# 破裂惩罚
disrupt_penalty = compute_disruption_penalty()
assert disrupt_penalty < 0, '破裂惩罚应为负值！'
print(f'[PASS] 破裂惩罚 = {disrupt_penalty}')
"
```

---

### T-004：FusionEnv — 完整 episode

**验证目标**：FusionEnv 能运行完整 episode，API 兼容 Gymnasium

```python
python -c "
from backend.rl.fusion_env import FusionEnv
import numpy as np

env = FusionEnv(max_steps=100)

# reset
obs, info = env.reset(seed=42)
assert obs.shape == (7,), f'obs 形状错误：{obs.shape}'
assert all(0 <= v <= 1 for v in obs), f'obs 超出 [0,1]：{obs}'
print(f'[PASS] reset OK, q95={info[\"state_dict\"][\"q95\"]:.3f}')

# step loop
total_reward = 0
for i in range(100):
    action = env.action_space.sample() * 0.03
    obs, rew, terminated, truncated, info = env.step(action)
    total_reward += rew
    if terminated or truncated:
        print(f'  Episode 结束（step={i+1}, disrupted={info[\"disrupted\"]}）')
        break

assert obs.shape == (7,)
print(f'[PASS] FusionEnv step loop OK, total_reward={total_reward:.1f}')
"
```

**期望结果**：`[PASS]` 全部通过

---

### T-005：east_loader + replay_buffer — 数据管线

```python
python -c "
from backend.data.east_loader import load_synthetic_east_data
from backend.data.replay_buffer import build_replay_buffer

df = load_synthetic_east_data(n_shots=5, steps_per_shot=30)
assert len(df) == 150, f'数据量错误：{len(df)}'
print(f'[PASS] 合成数据生成 OK：{df.shape}')

buf = build_replay_buffer(df)
assert buf['observations'].shape[1] == 7, '状态维度错误'
assert buf['actions'].shape[1] == 3, '动作维度错误'
assert len(buf['rewards']) == len(buf['observations']), '长度不一致'
print(f'[PASS] Replay Buffer OK：{buf[\"n_transitions\"]} transitions')
"
```

---

### T-006：SFT BC 预训练

```python
python -c "
from backend.rl.bc_pretrain import behavior_clone_sft
from backend.data.east_loader import load_synthetic_east_data

df = load_synthetic_east_data(n_shots=3, steps_per_shot=20)
path = behavior_clone_sft(east_df=df, n_epochs=3)
import os
assert os.path.exists(path), f'权重文件不存在：{path}'
print(f'[PASS] SFT BC 预训练 OK：{path}')
"
```

---

### T-007：World Model 快速训练 + 预测

```python
python -c "
from backend.rl.world_model import WorldModelEnsemble, train_world_model
from backend.data.east_loader import load_synthetic_east_data
import numpy as np

df = load_synthetic_east_data(n_shots=5, steps_per_shot=20)
model = train_world_model(east_df=df, n_epochs=3, n_models=3)

state  = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
action = np.array([0.05, 0.01, -0.02], dtype=np.float32)
next_s, std, rew, unc = model.predict_next(state, action)

assert next_s.shape == (7,), f'next_s 形状错误：{next_s.shape}'
assert unc >= 0, f'不确定性不能为负：{unc}'
print(f'[PASS] World Model 预测 OK：unc={unc:.5f}, reward={rew:.3f}')
"
```

---

## 集成测试（API 端点）

启动后端后运行：

```bash
# 健康检查
curl http://localhost:8000/ | python -m json.tool

# RL 状态
curl http://localhost:8000/api/rl/status | python -m json.tool

# τ_E 系数查询
curl http://localhost:8000/api/calibration/tau-e-coeff | python -m json.tool

# 世界模型不确定性（需先训练）
curl http://localhost:8000/api/rl/world-model-uncertainty | python -m json.tool
```

---

## 端到端训练验收标准

### Phase 1 验收

```bash
python -m backend.rl.train_rl --n_steps 100000 --version e2e_test
python -m backend.rl.eval_rl --model_path data/rl_models/ppo_fusion_e2e_test/final.zip --n_episodes 10
```

| 指标 | 验收标准 | 说明 |
|------|----------|------|
| `disruption_rate` | < 0.2 | 破裂率低于 20%（100K 步后）|
| `gaming_proxy` | < 0.5 | 无明显 gaming |
| `mean_ep_length` | > 300 | 平均 episode 长度 > 300 步 |
| `episode_reward` trend | 上升 | 100K 步的奖励趋势向上 |

### Phase 2A 验收

| 指标 | 验收标准 |
|------|----------|
| τ_E 拟合 R² | > 0.8（合成数据应 = 1.0）|

### Phase 2B 验收

| 指标 | 验收标准 |
|------|----------|
| DTW 分数 | < 10.0（随机策略 > 50.0）|

### Phase 4 验收

| 指标 | 验收标准 |
|------|----------|
| 世界模型测试集误差 | < 5%（需真实 EAST 数据）|
| Dyna MBRL reward | ≥ Phase 1 PPO 的 reward |

---

## 快速全量验证脚本

```bash
#!/bin/bash
source venv/bin/activate
echo "=== FusionLab 单元测试 ==="
python -c "
from backend.rl.dynamics import step_plasma_state, denormalize_state
from backend.rl.disruption import check_disruption
from backend.rl.rewards import compute_reward
from backend.rl.fusion_env import FusionEnv
from backend.data.east_loader import load_synthetic_east_data
from backend.data.replay_buffer import build_replay_buffer
import numpy as np

state  = np.array([0.3, 0.2, 0.5, 0.5, 0.3, 0.4, 0.3], dtype=np.float32)
action = np.array([0.05, 0.01, -0.02], dtype=np.float32)

# T-001
next_s = step_plasma_state(state, action)
assert np.all(next_s >= 0) and np.all(next_s <= 1)
print('[PASS] T-001: dynamics')

# T-002
d, _ = check_disruption(state)
assert not d
print('[PASS] T-002: disruption')

# T-003
rew, info = compute_reward(state, action)
assert info['efficiency_penalty'] < 0
print('[PASS] T-003: rewards')

# T-004
env = FusionEnv(max_steps=20)
obs, _ = env.reset(seed=42)
for _ in range(20):
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample() * 0.03)
    if terminated or truncated: break
print('[PASS] T-004: FusionEnv')

# T-005
df = load_synthetic_east_data(n_shots=3, steps_per_shot=20)
buf = build_replay_buffer(df)
assert buf['observations'].shape[1] == 7
print('[PASS] T-005: east_loader + replay_buffer')

print()
print('=== 所有单元测试通过 ===')
" 2>&1 | grep -E "\[PASS\]|错误|Error|FAIL|==="
```

---

*文档维护：阿策 · 最后更新：2026-02*
