"""
model.py — PyTorch MLP 场预测器

架构：
  输入层  : 5 个特征 (n_e_norm, T_e_norm, B_norm, r, theta)
  隐藏层  : [256, 256, 128]，Leaky ReLU + LayerNorm + Dropout
  输出层  : 1 个值（该坐标点的归一化温度）

设计考量：
  - LayerNorm 代替 BatchNorm：批次小时更稳定
  - Leaky ReLU：避免 dying ReLU，对物理负值更友好
  - Dropout(0.1)：轻量正则化，不过拟合小数据集
  - 输出 Sigmoid：温度已归一化到 [0,1]，Sigmoid 约束范围
"""

import torch
import torch.nn as nn
from typing import Dict


class PlasmaFieldMLP(nn.Module):
    """
    等离子体场预测 MLP
    给定 (n_e, T_e, B, r, theta) → 预测该坐标点的温度
    """

    def __init__(
        self,
        input_dim:   int   = 5,
        hidden_dims: list  = None,
        output_dim:  int   = 1,
        dropout:     float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim

        layers += [
            nn.Linear(prev_dim, output_dim),
            nn.Sigmoid(),   # 输出 [0,1]，对应归一化温度
        ]

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，加速收敛"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device: str = "cpu") -> PlasmaFieldMLP:
    """创建并返回模型，移动到指定设备"""
    model = PlasmaFieldMLP(
        input_dim=5,
        hidden_dims=[256, 256, 128],
        output_dim=1,
        dropout=0.1,
    )
    model = model.to(device)
    print(f"[Model] PlasmaFieldMLP 创建完成，参数量: {model.count_parameters():,}，设备: {device}")
    return model


def save_model(model: PlasmaFieldMLP, path: str, meta: Dict = None):
    """保存模型权重和元数据"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "input_dim":   5,
            "hidden_dims": [256, 256, 128],
            "output_dim":  1,
        },
        "meta": meta or {},
    }
    torch.save(checkpoint, path)
    print(f"[Model] 模型已保存到 {path}")


def load_model(path: str, device: str = "cpu") -> PlasmaFieldMLP:
    """从文件加载模型"""
    checkpoint = torch.load(path, map_location=device)
    arch = checkpoint["architecture"]
    model = PlasmaFieldMLP(
        input_dim=arch["input_dim"],
        hidden_dims=arch["hidden_dims"],
        output_dim=arch["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[Model] 模型从 {path} 加载完成")
    return model
