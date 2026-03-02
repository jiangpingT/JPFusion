"""
offline_train.py — CQL 离线训练命令行入口（Phase 3）

运行命令：
    python -m backend.rl.offline_train --east_data data/east_discharge.csv
    python -m backend.rl.offline_train --use_synthetic   # 无真实数据时用合成数据
"""

import argparse
from backend.rl.offline_rl import train_cql

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CQL 离线 RL 训练")
    parser.add_argument("--east_data",           type=str,   default=None)
    parser.add_argument("--use_synthetic",        action="store_true",
                        help="使用合成 EAST 数据（无真实数据时）")
    parser.add_argument("--n_steps",             type=int,   default=100_000)
    parser.add_argument("--conservative_weight", type=float, default=5.0)
    args = parser.parse_args()

    if args.use_synthetic or args.east_data is None:
        from backend.data.east_loader import load_synthetic_east_data
        import tempfile, os
        df = load_synthetic_east_data(n_shots=100)
        # 保存为临时 CSV
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        tmp.close()
        east_data_path = tmp.name
        print(f"[offline_train] 使用合成数据：{tmp.name}")
    else:
        east_data_path = args.east_data

    train_cql(
        east_data_path=east_data_path,
        n_steps=args.n_steps,
        conservative_weight=args.conservative_weight,
    )
