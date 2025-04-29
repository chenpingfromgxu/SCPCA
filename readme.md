# SCPCA-MNIST-Experiment

## 简介

本项目旨在实现一种基于稀疏协同相关熵的鲁棒主成分分析（SCPCA）算法，并在MNIST数据集上进行实验。SCPCA算法能够有效地从数据中提取主成分，同时抑制噪声和异常值的影响。

## 环境要求

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

## 安装依赖

使用pip命令安装所需的库：

```bash
pip install numpy matplotlib scikit-learn
```

## 数据集

MNIST数据集可通过scikit-learn库直接加载。

## 运行步骤

1. 克隆或下载本项目代码到本地。
2. 确保Python环境已安装所有依赖库。
3. 运行`scPCA_mnist.py`脚本。

## 代码结构

```
SCPCA-MNIST-Experiment/
│
├── SCPCA_mnist.py          # 主脚本，包含SCPCA算法实现和MNIST数据集实验
├── README.md            # 项目说明文件
```

## SCPCA_mnist.py 脚本说明

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def scPCA(X, lambd, mu, delta=1e-5, max_iter=100):
    # SCPCA算法实现
    pass  # 算法实现代码

def f(L, E, p, lamb):
    # 目标函数
    pass  # 目标函数代码

# 加载MNIST数据集
digits = load_digits()
X = digits.data

# 设置正则化参数
lambd = 0.1
mu = 0.1

# 运行SCPCA
L, E = scPCA(X, lambd, mu)

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, comp in zip(axes.flat, [L, E]):
    ax.imshow(comp, cmap='viridis')
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()
```

## 运行脚本

在命令行中运行以下命令：

```bash
python SCPCA_mnist.py
```

### 输出

运行脚本后，将显示两个子图，分别表示低秩矩阵 \(L\) 和稀疏矩阵 \(E\) 的可视化结果。

## 贡献

欢迎任何形式的贡献，包括但不限于：
- 报告bug
- 提交PR
- 改进文档

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

请确保在使用本项目代码时遵守相应的许可证和版权规定。
