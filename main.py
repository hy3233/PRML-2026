"""
PRML 作业1 — 回归实验
使用最小二乘法、梯度下降法、牛顿法进行线性拟合，
并探索多项式回归和高斯基函数等非线性模型。

依赖: numpy, pandas, matplotlib, openpyxl
运行: python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 配置 =====================
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data4Regression.xlsx')


# ===================== 数据加载 =====================
def load_data(path=DATA_PATH):
    df_train = pd.read_excel(path, sheet_name='Training Data')
    df_test = pd.read_excel(path, sheet_name='Test Data')
    x_train = df_train['x'].values
    y_train = df_train['y_complex'].values
    x_test = df_test['x_new'].values
    y_test = df_test['y_new_complex'].values
    return x_train, y_train, x_test, y_test


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ===================== Part 1: 线性拟合 =====================

def design_matrix_linear(x):
    """构建线性模型设计矩阵 [1, x]"""
    return np.column_stack([np.ones_like(x), x])


# --- 1.1 最小二乘法 (OLS) ---
def ols_fit(x, y):
    X = design_matrix_linear(x)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


# --- 1.2 梯度下降法 (GD) ---
def gd_fit(x, y, lr=0.001, max_iter=5000):
    X = design_matrix_linear(x)
    N = len(y)
    w = np.zeros(2)
    losses = []
    for i in range(max_iter):
        residual = X @ w - y
        loss = np.mean(residual ** 2)
        losses.append(loss)
        grad = (2.0 / N) * X.T @ residual
        w = w - lr * grad
    return w, losses


# --- 1.3 牛顿法 ---
def newton_fit(x, y, max_iter=20):
    X = design_matrix_linear(x)
    N = len(y)
    w = np.zeros(2)
    losses = []
    for i in range(max_iter):
        residual = X @ w - y
        loss = np.mean(residual ** 2)
        losses.append(loss)
        grad = (2.0 / N) * X.T @ residual
        H = (2.0 / N) * X.T @ X
        w = w - np.linalg.inv(H) @ grad
        # 检查收敛
        if i > 0 and abs(losses[-1] - losses[-2]) < 1e-12:
            break
    return w, losses


def predict_linear(w, x):
    X = design_matrix_linear(x)
    return X @ w


# ===================== Part 2: 非线性拟合 =====================

# --- 2.1 多项式回归（带归一化防止数值溢出）---
def poly_design_matrix(x_norm, degree):
    """构建多项式设计矩阵 [1, x, x^2, ..., x^M], x 应已归一化"""
    return np.column_stack([x_norm ** i for i in range(degree + 1)])


def normalize_x(x, x_mean=None, x_std=None):
    """归一化 x 到零均值单位方差"""
    if x_mean is None:
        x_mean = x.mean()
    if x_std is None:
        x_std = x.std()
    return (x - x_mean) / x_std, x_mean, x_std


def poly_fit(x, y, degree, lam=0.0):
    """多项式拟合，支持L2正则化 (岭回归)，内部归一化"""
    x_norm, x_mean, x_std = normalize_x(x)
    Phi = poly_design_matrix(x_norm, degree)
    I = np.eye(Phi.shape[1])
    I[0, 0] = 0
    w = np.linalg.inv(Phi.T @ Phi + lam * I) @ Phi.T @ y
    return w, x_mean, x_std


def poly_predict(w, x, degree, x_mean, x_std):
    x_norm = (x - x_mean) / x_std
    Phi = poly_design_matrix(x_norm, degree)
    return Phi @ w


# --- 2.2 高斯基函数回归 ---
def gauss_design_matrix(x, centers, sigma):
    """构建高斯基函数设计矩阵"""
    Phi = np.column_stack([np.exp(-0.5 * ((x - c) / sigma) ** 2) for c in centers])
    Phi = np.column_stack([np.ones_like(x), Phi])  # 加偏置
    return Phi


def gauss_fit(x, y, n_basis=15, sigma=0.5, lam=0.01):
    centers = np.linspace(x.min(), x.max(), n_basis)
    Phi = gauss_design_matrix(x, centers, sigma)
    I = np.eye(Phi.shape[1])
    I[0, 0] = 0
    w = np.linalg.inv(Phi.T @ Phi + lam * I) @ Phi.T @ y
    return w, centers, sigma


def gauss_predict(w, x, centers, sigma):
    Phi = gauss_design_matrix(x, centers, sigma)
    return Phi @ w


# ===================== 可视化 =====================

def plot_data_overview(x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, c='#2196F3', s=20, alpha=0.7, label='Training Data')
    ax.scatter(x_test, y_test, c='#FF9800', s=20, alpha=0.7, label='Test Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Data Overview')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'data_overview.png'))
    plt.close(fig)
    print('[✓] data_overview.png')


def plot_linear_fit(x_train, y_train, x_test, y_test, w, method_name, filename):
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, c='#2196F3', s=20, alpha=0.6, label='Train')
    ax.scatter(x_test, y_test, c='#FF9800', s=20, alpha=0.6, label='Test')
    x_line = np.linspace(0, 10, 200)
    y_line = predict_linear(w, x_line)
    ax.plot(x_line, y_line, 'r-', linewidth=2., label=f'{method_name}: y={w[1]:.4f}x+{w[0]:.4f}')
    train_mse = mse(y_train, predict_linear(w, x_train))
    test_mse = mse(y_test, predict_linear(w, x_test))
    ax.set_title(f'Linear Fit — {method_name}\nTrain MSE={train_mse:.4f}, Test MSE={test_mse:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close(fig)
    print(f'[✓] {filename}')


def plot_gd_convergence(losses):
    fig, ax = plt.subplots()
    ax.plot(losses, color='#E91E63', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Gradient Descent Convergence')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'gd_convergence.png'))
    plt.close(fig)
    print('[✓] gd_convergence.png')


def plot_newton_convergence(losses):
    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses, 'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Newton\'s Method Convergence')
    ax.set_xticks(range(len(losses)))
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'newton_convergence.png'))
    plt.close(fig)
    print('[✓] newton_convergence.png')


def plot_linear_comparison(x_train, y_train, w_ols, w_gd, w_newton):
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, c='#607D8B', s=20, alpha=0.6, label='Train Data')
    x_line = np.linspace(0, 10, 200)
    ax.plot(x_line, predict_linear(w_ols, x_line), '-', color='#F44336', linewidth=2, label='OLS')
    ax.plot(x_line, predict_linear(w_gd, x_line), '--', color='#4CAF50', linewidth=2, label='GD')
    ax.plot(x_line, predict_linear(w_newton, x_line), ':', color='#2196F3', linewidth=2.5, label='Newton')
    ax.set_title('Linear Fit Comparison: OLS vs GD vs Newton')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'linear_comparison.png'))
    plt.close(fig)
    print('[✓] linear_comparison.png')


def plot_poly_fits(x_train, y_train, x_test, y_test, degrees):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    x_plot = np.linspace(0, 10, 300)
    colors = ['#E91E63', '#9C27B0', '#3F51B5', '#009688', '#FF5722', '#795548']
    for idx, deg in enumerate(degrees):
        ax = axes[idx]
        w, xm, xs = poly_fit(x_train, y_train, deg)
        y_plot = poly_predict(w, x_plot, deg, xm, xs)
        train_err = mse(y_train, poly_predict(w, x_train, deg, xm, xs))
        test_err = mse(y_test, poly_predict(w, x_test, deg, xm, xs))
        ax.scatter(x_train, y_train, c='#2196F3', s=12, alpha=0.5, label='Train')
        ax.scatter(x_test, y_test, c='#FF9800', s=12, alpha=0.5, label='Test')
        y_clipped = np.clip(y_plot, -5, 5)
        ax.plot(x_plot, y_clipped, '-', color=colors[idx], linewidth=2)
        ax.set_title(f'Degree {deg}\nTrain={train_err:.4f} Test={test_err:.4f}', fontsize=10)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    plt.suptitle('Polynomial Regression with Different Degrees', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'nonlinear_poly.png'))
    plt.close(fig)
    print('[✓] nonlinear_poly.png')


def plot_poly_error_curve(x_train, y_train, x_test, y_test, max_degree=20):
    degrees = range(1, max_degree + 1)
    train_errors = []
    test_errors = []
    for deg in degrees:
        w, xm, xs = poly_fit(x_train, y_train, deg)
        train_errors.append(mse(y_train, poly_predict(w, x_train, deg, xm, xs)))
        test_errors.append(mse(y_test, poly_predict(w, x_test, deg, xm, xs)))
    fig, ax = plt.subplots()
    ax.plot(list(degrees), train_errors, 'o-', color='#2196F3', linewidth=2, label='Train MSE')
    ax.plot(list(degrees), test_errors, 's-', color='#F44336', linewidth=2, label='Test MSE')
    ax.set_xlabel('Polynomial Degree M')
    ax.set_ylabel('MSE')
    ax.set_title('Train/Test Error vs Polynomial Degree')
    ax.set_ylim(0, max(2.0, min(max(test_errors), 5.0)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'poly_error_curve.png'))
    plt.close(fig)
    print('[✓] poly_error_curve.png')
    return train_errors, test_errors


def plot_regularization(x_train, y_train, x_test, y_test, degree=15):
    lambdas = [0, 1e-6, 1e-4, 1e-2, 1, 10]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    x_plot = np.linspace(0, 10, 300)
    for idx, lam in enumerate(lambdas):
        ax = axes[idx]
        w, xm, xs = poly_fit(x_train, y_train, degree, lam=lam)
        y_plot = poly_predict(w, x_plot, degree, xm, xs)
        train_err = mse(y_train, poly_predict(w, x_train, degree, xm, xs))
        test_err = mse(y_test, poly_predict(w, x_test, degree, xm, xs))
        ax.scatter(x_train, y_train, c='#2196F3', s=12, alpha=0.5)
        y_clipped = np.clip(y_plot, -5, 5)
        ax.plot(x_plot, y_clipped, '-', color='#E91E63', linewidth=2)
        ax.set_title(f'$\\lambda$={lam}\nTrain={train_err:.4f} Test={test_err:.4f}', fontsize=10)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'L2 Regularization Effect (Degree={degree})', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'regularization.png'))
    plt.close(fig)
    print('[✓] regularization.png')


def plot_gauss_fit(x_train, y_train, x_test, y_test):
    w, centers, sigma = gauss_fit(x_train, y_train, n_basis=15, sigma=0.5, lam=0.01)
    x_plot = np.linspace(0, 10, 300)
    y_plot = gauss_predict(w, x_plot, centers, sigma)
    train_err = mse(y_train, gauss_predict(w, x_train, centers, sigma))
    test_err = mse(y_test, gauss_predict(w, x_test, centers, sigma))
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, c='#2196F3', s=20, alpha=0.6, label='Train')
    ax.scatter(x_test, y_test, c='#FF9800', s=20, alpha=0.6, label='Test')
    ax.plot(x_plot, y_plot, '-', color='#E91E63', linewidth=2.5, label='Gaussian RBF')
    ax.set_title(f'Gaussian Basis Function Regression\nTrain MSE={train_err:.4f}, Test MSE={test_err:.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIGURE_DIR, 'nonlinear_gauss.png'))
    plt.close(fig)
    print('[✓] nonlinear_gauss.png')
    return train_err, test_err


def plot_error_comparison(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = list(results.keys())
    train_errs = [results[m]['train'] for m in methods]
    test_errs = [results[m]['test'] for m in methods]
    x_pos = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, train_errs, width, color='#2196F3', alpha=0.8, label='Train MSE')
    bars2 = ax.bar(x_pos + width / 2, test_errs, width, color='#F44336', alpha=0.8, label='Test MSE')
    # 在柱子上方标注数值
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('MSE')
    ax.set_title('Error Comparison Across All Methods')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'error_comparison.png'))
    plt.close(fig)
    print('[✓] error_comparison.png')


# ===================== 主函数 =====================
def main():
    print('=' * 60)
    print('PRML Assignment 1 — Regression Experiment')
    print('=' * 60)

    # 加载数据
    x_train, y_train, x_test, y_test = load_data()
    print(f'\nTraining data: {len(x_train)} points, x ∈ [{x_train.min():.2f}, {x_train.max():.2f}]')
    print(f'Test data:     {len(x_test)} points, x ∈ [{x_test.min():.2f}, {x_test.max():.2f}]')

    # 数据总览
    plot_data_overview(x_train, y_train, x_test, y_test)

    # ===================== Part 1: 线性拟合 =====================
    print('\n' + '=' * 60)
    print('Part 1: Linear Regression')
    print('=' * 60)

    # 1.1 最小二乘法
    w_ols = ols_fit(x_train, y_train)
    train_mse_ols = mse(y_train, predict_linear(w_ols, x_train))
    test_mse_ols = mse(y_test, predict_linear(w_ols, x_test))
    print(f'\n[OLS] w0={w_ols[0]:.6f}, w1={w_ols[1]:.6f}')
    print(f'      Train MSE={train_mse_ols:.6f}, Test MSE={test_mse_ols:.6f}')
    plot_linear_fit(x_train, y_train, x_test, y_test, w_ols, 'OLS', 'linear_ols.png')

    # 1.2 梯度下降
    w_gd, gd_losses = gd_fit(x_train, y_train, lr=0.001, max_iter=5000)
    train_mse_gd = mse(y_train, predict_linear(w_gd, x_train))
    test_mse_gd = mse(y_test, predict_linear(w_gd, x_test))
    print(f'\n[GD]  w0={w_gd[0]:.6f}, w1={w_gd[1]:.6f}')
    print(f'      Train MSE={train_mse_gd:.6f}, Test MSE={test_mse_gd:.6f}')
    plot_linear_fit(x_train, y_train, x_test, y_test, w_gd, 'Gradient Descent', 'linear_gd.png')
    plot_gd_convergence(gd_losses)

    # 1.3 牛顿法
    w_newton, newton_losses = newton_fit(x_train, y_train, max_iter=20)
    train_mse_newton = mse(y_train, predict_linear(w_newton, x_train))
    test_mse_newton = mse(y_test, predict_linear(w_newton, x_test))
    print(f'\n[Newton] w0={w_newton[0]:.6f}, w1={w_newton[1]:.6f}')
    print(f'         Train MSE={train_mse_newton:.6f}, Test MSE={test_mse_newton:.6f}')
    print(f'         Converged in {len(newton_losses)} iterations')
    plot_linear_fit(x_train, y_train, x_test, y_test, w_newton, "Newton's Method", 'linear_newton.png')
    plot_newton_convergence(newton_losses)

    # 对比
    plot_linear_comparison(x_train, y_train, w_ols, w_gd, w_newton)

    print('\n--- Linear Method Comparison ---')
    print(f'{"Method":<15} {"w0":>10} {"w1":>10} {"Train MSE":>12} {"Test MSE":>12}')
    print('-' * 60)
    print(f'{"OLS":<15} {w_ols[0]:>10.6f} {w_ols[1]:>10.6f} {train_mse_ols:>12.6f} {test_mse_ols:>12.6f}')
    print(f'{"GD":<15} {w_gd[0]:>10.6f} {w_gd[1]:>10.6f} {train_mse_gd:>12.6f} {test_mse_gd:>12.6f}')
    print(f'{"Newton":<15} {w_newton[0]:>10.6f} {w_newton[1]:>10.6f} {train_mse_newton:>12.6f} {test_mse_newton:>12.6f}')

    # ===================== Part 2: 非线性拟合 =====================
    print('\n' + '=' * 60)
    print('Part 2: Nonlinear Regression')
    print('=' * 60)

    # 2.1 多项式回归
    degrees = [3, 5, 7, 9, 12, 15]
    plot_poly_fits(x_train, y_train, x_test, y_test, degrees)

    print('\n--- Polynomial Regression Results ---')
    print(f'{"Degree":<10} {"Train MSE":>12} {"Test MSE":>12}')
    print('-' * 35)
    results = {}
    for deg in degrees:
        w, xm, xs = poly_fit(x_train, y_train, deg)
        tr_err = mse(y_train, poly_predict(w, x_train, deg, xm, xs))
        te_err = mse(y_test, poly_predict(w, x_test, deg, xm, xs))
        print(f'{deg:<10} {tr_err:>12.6f} {te_err:>12.6f}')
        results[f'Poly(M={deg})'] = {'train': tr_err, 'test': te_err}

    # 误差曲线
    train_errs, test_errs = plot_poly_error_curve(x_train, y_train, x_test, y_test, max_degree=20)

    # 正则化效果
    plot_regularization(x_train, y_train, x_test, y_test, degree=15)

    # 正则化最佳结果
    best_lam = 1e-2
    best_deg = 15
    w_reg, xm_reg, xs_reg = poly_fit(x_train, y_train, best_deg, lam=best_lam)
    reg_train = mse(y_train, poly_predict(w_reg, x_train, best_deg, xm_reg, xs_reg))
    reg_test = mse(y_test, poly_predict(w_reg, x_test, best_deg, xm_reg, xs_reg))
    print(f'\n[Ridge] Degree={best_deg}, lambda={best_lam}')
    print(f'        Train MSE={reg_train:.6f}, Test MSE={reg_test:.6f}')

    # 2.2 高斯基函数
    gauss_train, gauss_test = plot_gauss_fit(x_train, y_train, x_test, y_test)
    print(f'\n[Gauss RBF] n_basis=15, σ=0.5, λ=0.01')
    print(f'            Train MSE={gauss_train:.6f}, Test MSE={gauss_test:.6f}')

    # ===================== 总对比 =====================
    print('\n' + '=' * 60)
    print('Overall Comparison')
    print('=' * 60)

    all_results = {
        'Linear(OLS)': {'train': train_mse_ols, 'test': test_mse_ols},
        'Linear(GD)': {'train': train_mse_gd, 'test': test_mse_gd},
        'Linear(Newton)': {'train': train_mse_newton, 'test': test_mse_newton},
    }
    # 加入最佳多项式
    for deg in [5, 9]:
        w, xm, xs = poly_fit(x_train, y_train, deg)
        all_results[f'Poly(M={deg})'] = {
            'train': mse(y_train, poly_predict(w, x_train, deg, xm, xs)),
            'test': mse(y_test, poly_predict(w, x_test, deg, xm, xs)),
        }
    all_results[f'Ridge(M={best_deg})'] = {'train': reg_train, 'test': reg_test}
    all_results['Gauss RBF'] = {'train': gauss_train, 'test': gauss_test}

    plot_error_comparison(all_results)

    print(f'\n{"Method":<20} {"Train MSE":>12} {"Test MSE":>12}')
    print('-' * 45)
    for name, errs in all_results.items():
        print(f'{name:<20} {errs["train"]:>12.6f} {errs["test"]:>12.6f}')

    print(f'\nAll figures saved to: {FIGURE_DIR}')
    print('Done!')


if __name__ == '__main__':
    main()
