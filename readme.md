# SCPCA (基于稀疏协同相关熵的鲁棒主成分分析) - Python 实现

## 概述

本项目包含鲁棒主成分分析（Robust PCA）算法 SCPCA（Sparsity Cooperated Correntropy based Robust Principal Component Analysis）的 Python 实现。该实现主要依据以下研究论文中描述的方法：

*   **论文题目:** 基于稀疏协同相关熵的鲁棒主成分分析
*   **作者:** 陈平, 刘珂菡, 梁正友, 胡奇兴, 张远鹏
*   **期刊:** 《计算机科学》
*   **网络首发日期:** 2024-12-27
*   **DOI/Link:** [https://link.cnki.net/urlid/50.1075.TP.20241226.1100.009](https://link.cnki.net/urlid/50.1075.TP.20241226.1100.009)

该算法旨在将数据矩阵 `X` 分解为低秩部分 `L` 和稀疏误差部分 `E` (`X ≈ L + E`)，并通过引入基于 Correntropy 的度量来增强对非高斯噪声（包括脉冲噪声和离群点/样本异常值）的鲁棒性。

**重要提示:**

> **此实现是一个简要示例，主要用于在计算资源有限（低配置）的计算机上验证核心算法的基本行为和运行初步实验（如 MNIST）。它可能需要进一步的优化和细致的参数调整，才能在复杂、大规模数据集上达到最佳性能，或完全复现原始论文中的所有精确结果。**

## 主要功能

*   实现了论文中描述的基于加速块坐标更新 (Accelerated BCU) 和 Fenchel 对偶的 SCPCA 迭代优化算法。
*   包含在 MNIST 数据集上运行实验的功能，自动下载数据。
*   生成可视化结果：
    *   展示原始 MNIST 图像、恢复的低秩图像 `L` 和稀疏误差图像 `E`。
    *   绘制算法收敛过程中的关键指标（重建误差、L 的秩、E 的稀疏度、核宽度 Sigma）。
*   将运行参数、最终结果和迭代历史记录导出到 Excel 文件。

## 环境要求

*   Python 3.x
*   所需的 Python 库见 `requirements.txt` 文件。

## 安装

1.  **克隆或下载代码库:**
    ```bash
    # 如果使用 git
    git clone <your-repo-url>
    cd <repository-folder>
    ```
    或者直接下载代码文件。

2.  **安装依赖项:**
    在项目根目录下打开终端或命令行，运行以下命令：
    ```bash
    pip install -r requirements.txt
    ```
    *注意：* 安装 `torch` 和 `torchvision` 可能需要根据您的操作系统和 CUDA 版本（如果使用 GPU）参考 PyTorch 官网 ([https://pytorch.org/](https://pytorch.org/)) 的特定命令。`requirements.txt` 中的版本适用于通用 CPU 安装。

## 如何运行

1.  **(可选) 修改参数:**
    打开 `run_scpca.py` 文件（或您保存代码的文件名）。您可以根据需要修改位于 `if __name__ == "__main__":` 代码块内的参数，例如：
    *   `NUM_MNIST_SAMPLES`: 使用的 MNIST 样本数量。
    *   `DIGITS_TO_USE`: 选择使用的特定 MNIST 数字（例如 `[0, 1, 8]`）或设为 `None` 使用所有数字。
    *   `SCPCA_LAMBDA`: 控制低秩项 `L` 的核范数权重（需要调优）。
    *   `SCPCA_MU`: 控制稀疏项 `E` 的 L1 范数权重（需要调优）。
    *   `SCPCA_TOL`: 算法收敛的容忍度。
    *   `SCPCA_MAX_ITER`: 最大迭代次数。

2.  **运行脚本:**
    在终端中，确保您位于包含脚本的目录下，然后运行：
    ```bash
    python run_scpca.py
    ```

## 输出

脚本运行后，将在项目目录下创建一个名为 `scpca_output` 的文件夹（如果尚不存在），其中包含以下文件：

*   **`scpca_mnist_results.png`:** 显示随机选取的原始 MNIST 样本、对应的低秩分量 `L` 和稀疏分量 `E` 的图像。
*   **`scpca_convergence_plots.png`:** 显示算法在迭代过程中的收敛指标（误差、秩、稀疏度、Sigma）的变化曲线图。
*   **`scpca_results.xlsx`:** 一个 Excel 文件，包含两个工作表：
    *   `Summary & Parameters`: 记录本次运行使用的超参数和最终的汇总结果（如最终误差、秩、稀疏度、总时间等）。
    *   `Iteration History`: 详细记录了每次（或每隔几次）迭代的关键指标值。

## 注意

*   SCPCA 算法的性能（收敛速度、结果质量）对超参数 `lambda_` 和 `mu` 非常敏感，需要根据具体数据和任务进行仔细调整。
*   对于大型数据集，算法（尤其是 SVD 步骤）可能会比较耗时。
*   MNIST 数据集会自动下载（如果本地没有），请确保网络连接正常。
