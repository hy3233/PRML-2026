"""
PRML 作业 2 — 分类实验
任务：对 3D Make-Moons 数据集进行分类，并对比 DT, AdaBoost, SVM (多核) 性能。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ===================== 配置 =====================
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (10, 7),
    'figure.dpi': 150,
})
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# ===================== 数据生成 (来自作业 PDF) =====================
def make_moons_3d(n_samples=500, noise=0.1):
    """⽣成了⼀个3D的数据集。"""
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  
    
    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([
        np.column_stack([x, y, z]), 
        np.column_stack([-x, y - 1, -z])
    ])
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)
    return X, labels

def plot_3d_data(X, labels, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close(fig)
    print(f'[OK] Saved: {filename}')

# ===================== 实验主逻辑 =====================
def run_experiment():
    print('='*60)
    print('PRML Assignment 2: 3D Moons Classification')
    print('='*60)

    # 1. 生成数据
    # 训练集: 1000样本 (500 C0, 500 C1)
    # 测试集: 新生成的 500样本 (250 C0, 250 C1)
    X_train, y_train = make_moons_3d(n_samples=500, noise=0.2) # n_samples=500 yields 1000 total points
    X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)   # n_samples=250 yields 500 total points

    print(f'Training size: {X_train.shape[0]} | Test size: {X_test.shape[0]}')
    
    # 可视化训练数据
    plot_3d_data(X_train, y_train, '3D Make Moons - Training Set', 'hw2_data_distribution.png')

    # 2. 定义分类器
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=None),
        'AdaBoost (DT)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), 
            n_estimators=50
        ),
        'SVM (Linear)': SVC(kernel='linear', probability=True),
        'SVM (Poly d=3)': SVC(kernel='poly', degree=3, probability=True),
        'SVM (RBF)': SVC(kernel='rbf', gamma='scale', probability=True)
    }

    results = {}

    # 3. 训练与测评
    print('\nEvaluating Models...')
    print(f'{"Model":<18} | {"Accuracy":>10} | {"F1-Score":>10}')
    print('-'*45)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results[name] = {'acc': acc, 'f1': f1}
        
        print(f'{name:<18} | {acc:>10.4f} | {f1:>10.4f}')

    # 4. 性能对比可视化
    plot_performance(results)

    print(f'\nDone! Figures saved in: {FIGURE_DIR}')

def plot_performance(results):
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    f1s = [results[n]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#3F51B5', alpha=0.8)
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color='#E91E63', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison of Classifiers on 3D Make-Moons')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'hw2_performance_comparison.png'))
    plt.close(fig)
    print(f'[OK] Saved: hw2_performance_comparison.png')

if __name__ == '__main__':
    run_experiment()
