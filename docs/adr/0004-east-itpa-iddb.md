# ADR-0004: 使用 ITPA IDDB 作为 EAST 真实数据来源

**状态**：接受

**日期**：2026-02

**决策人**：姜哥 / 阿策

---

## 背景

Phase 2-4 需要真实的 EAST（东方超环）托卡马克放电数据，用于：
- Phase 2A：拟合 τ_E 经验公式系数
- Phase 2B：DTW 策略有效性验证
- Phase 2C：SFT 行为克隆预训练
- Phase 3：Offline RL Replay Buffer
- Phase 4：World Model 训练

可选数据来源：
1. EAST 官方数据库（MDSplus）——需要机构授权 + VPN
2. ITPA IDDB（Harvard Dataverse）——公开免费
3. IMAS/OMAS 格式数据——需要 ITER 组织访问权限
4. 合成数据（用 FusionEnv 生成）——无真实物理价值

## 决策

使用 **ITPA IDDB（Harvard Dataverse）** 作为主要数据来源，同时实现合成数据生成器作为降级方案。

```
数据源：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OQTQXJ
格式：HDF5 / CSV
内容：1170+ 次放电，含 EAST 数据子集
访问：无需注册，公开免费
```

降级方案：`backend/data/east_loader.py::load_synthetic_east_data()` 生成合成数据，
用于在没有真实数据的情况下测试 Phase 2-4 整体流程。

## 后果

### 正面影响

- 无需机构授权，本机可直接下载使用
- 包含多个托卡马克装置数据（EAST 子集最适合 FusionEnv 的参数范围）
- HDF5 格式标准化，`east_loader.py` 可以统一处理
- 合成数据降级方案确保 CI/CD 和测试不依赖网络

### 负面影响 / 权衡

- IDDB 数据经过预处理，某些高频信号（Mirnov 线圈 dB/dt）不可用
- τ_E 不直接存在于数据集中，需要从 P_heat 和 T_e 时序近似计算（引入误差）
- 数据覆盖的 EAST 运行时段可能不包含最新实验

### 数据质量注意事项

| 问题 | 处理方式 |
|------|----------|
| 单位不统一（eV vs K，MA vs A） | `east_loader.py` 自动检测并转换 |
| NaN 和异常值 | 过滤 n_e/T_e/Ip/P_heat 全非正的行 |
| 放电末期数据噪声大 | 只用每次放电前 80% 时间窗口 |
| 类别不平衡（破裂极少） | terminal 标记确保 done 信号准确 |

## 替代方案

### 方案 A：EAST MDSplus 官方数据库

**放弃原因**：需要中国科学院等离子体物理研究所机构账号 + 内网 VPN，
本机（海外 / 非机构网络）无法访问。

### 方案 B：合成数据（仅用 FusionEnv 生成）

**放弃原因**：合成数据由 FusionEnv 自身的 ODE 动力学生成，
Phase 2-4 用它训练等于"用模型预测自己"，没有真实物理校准价值。
只作为降级方案使用。

### 方案 C：从论文附录数据重建

**放弃原因**：论文附录数据通常只有关键时刻的截面数据，缺少时序连续性，
无法构建 Replay Buffer 所需的 (s, a, r, s') 轨迹。

---

*相关 ADR：[ADR-0005](0005-mlp-ensemble-world-model.md)*
