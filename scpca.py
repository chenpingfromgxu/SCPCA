import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def scPCA(X, lambd, mu, delta=1e-5, max_iter=100):
    d, r = X.shape
    L = np.random.rand(d, r)
    E = np.random.rand(d, r)
    p = np.ones((d, r))
    k = 0

    while k < max_iter:
        p = np.exp(-np.linalg.norm(X - L - E, axis=0)**2 / (2 * p))
        L_k = np.dot(X - E, p) / np.sum(p, axis=1, keepdims=True)
        omega_k = np.dot(p, (L_k - L))
        L = L + omega_k * (L_k - L)
        B = L - (1 / (L + lamb)) * np.dot(np.diag(1 / np.diag(L)), 
                  np.gradient(np.sum(np.multiply(L, p), axis=1), axis=0)) - lamb * L
        L = np.maximum(L - lamb * np.sign(B), 0)
        if np.linalg.norm(f(L, E, p) - lamb * np.linalg.norm(L, 'nuc')) <= delta:
            break
        E = np.maximum(E - mu * np.sign(X - L), 0)
        k += 1
    
    return L, E

def f(L, E, p, lamb):
    return np.linalg.norm(X - L - E, 'fro')**2 + lamb * np.linalg.norm(L, 'nuc')**2

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
