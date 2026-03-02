"""
backend.rl — FusionRL 强化学习子包

Phase 1: 纯 FusionEnv + PPO（模拟器驱动，不依赖真实数据）
Phase 2: 行为克隆预训练（BC Pretrain）
Phase 3: Offline RL（CQL via d3rlpy）
Phase 4: World Model + Dyna MBRL
"""
