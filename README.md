# PRML 2026 — 作业 1：回归实验

## 📌 任务描述

对给定二维数据进行线性和非线性回归拟合：
1. 使用 **最小二乘法 (OLS)**、**梯度下降法 (GD)**、**牛顿法** 进行线性拟合
2. 探索更合适的非线性模型（多项式回归 + 高斯基函数回归 + L2 正则化）

## 📂 文件结构

```
├── main.py                  # 主程序：所有方法实现 + 可视化
├── Data4Regression.xlsx     # 原始数据
├── report.pdf               # 实验报告
├── figures/                 # 生成的实验图表
│   ├── data_overview.png
│   ├── linear_ols.png
│   ├── linear_gd.png
│   ├── linear_newton.png
│   ├── linear_comparison.png
│   ├── gd_convergence.png
│   ├── newton_convergence.png
│   ├── nonlinear_poly.png
│   ├── poly_error_curve.png
│   ├── regularization.png
│   ├── nonlinear_gauss.png
│   └── error_comparison.png
└── README.md
```

## 🚀 运行方法

```bash
# 依赖安装
pip install numpy pandas matplotlib openpyxl

# 运行实验
python main.py
```

运行后将在 `figures/` 目录生成所有实验图表。

## 📊 主要结果

| Method | Train MSE | Test MSE |
|--------|-----------|----------|
| Linear (OLS) | 0.6134 | 0.5950 |
| Linear (GD) | 0.6142 | 0.5934 |
| Linear (Newton) | 0.6134 | 0.5950 |
| Poly (M=9) | 0.3541 | 0.3875 |
| Ridge (M=15, λ=0.01) | 0.2967 | 0.3807 |
| **Gauss RBF** | **0.1858** | **0.2868** |

## 🔧 技术细节

- 全部使用 **NumPy 手写实现**，无 sklearn 依赖
- 多项式回归内置 **x 归一化** 防止高阶数值溢出
- 支持 **L2 正则化** (Ridge Regression)
- 高斯基函数回归使用 15 个基函数，σ=0.5
