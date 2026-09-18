# DynaFE-Net 传感器项目

本项目是一个面向动态生物化学 / 化学传感器回归预测任务的机器学习项目，主要用于动态混合气体环境下的传感器信号建模与气体浓度预测。

## 项目信息
- 作者：于天佑、朱赫阳

## 方法简介

**DynaFE-Net：动态特征增强神经网络**  
**Dynamic Feature Enhanced Neural Network**

DynaFE-Net 保持了机器学习 / 深度学习方向的建模思路，但没有直接采用不稳定、计算开销较大的端到端长窗口时序网络。  
该方法通过细致的**因果动态特征工程**提取传感器响应中的时序信息，并结合轻量级**残差多层感知机（Residual MLP）**完成非线性回归预测。

整体思路是：先根据传感器动态响应机理构造多尺度动态特征，再利用神经网络学习这些可解释特征与气体浓度之间的复杂非线性映射关系。

## 选择该方法的原因

UCI 动态气体传感器数据集具有明显的动态响应特征，因此仅使用当前时刻传感器读数进行静态回归往往难以取得理想效果。DynaFE-Net 的设计与数据特点之间的对应关系如下：

| 数据特征 | 方法设计 |
|---|---|
| 传感器响应存在延迟和滞后现象 | 构造因果 lag、slope、EMA、rolling statistics 等动态特征 |
| 浓度转变阶段样本较少 | 使用 transition-weighted Huber loss 提高变化段权重 |
| 混合气体之间存在交叉敏感性 | 使用非线性残差 MLP 建模复杂关系 |
| 16 个通道由 4 类传感器重复组成 | 构造传感器类型级 mean、std、contrast、interaction 特征 |
| 不同气体目标浓度量纲不同 | 对目标浓度进行标准化 |
| 完整数据样本量较大 | 使用表格化动态特征，避免高开销长窗口序列模型 |

## 项目结构

```text
dynafe_sensor_project/
├── README.md
├── requirements.txt
├── data.py
├── features.py
├── train.py
└── run_all.py
```

各文件功能说明如下：

- `README.md`：项目说明文档；
- `requirements.txt`：项目所需 Python 依赖；
- `data.py`：数据读取与预处理接口；
- `features.py`：动态特征与传感器类型结构特征构造；
- `train.py`：单个任务的模型训练与评估脚本；
- `run_all.py`：同时运行多个气体混合任务的入口脚本。

## 数据说明

请将解压后的 UCI 数据集 `.txt` 文件放置在 `data/` 文件夹下。

数据加载器会递归搜索类似如下名称的数据文件：

```text
ethylene_CO.txt
ethylene_methane.txt
```

项目中不包含自动下载器，需要用户自行下载并解压数据集。

## 环境安装

在项目根目录下运行：

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10 或兼容版本，并确保已安装 NumPy、Pandas、Scikit-learn、PyTorch 等依赖库。

## 快速运行

以 Ethylene-CO 任务为例，可以使用快速模式运行：

```bash
python train.py --data-dir data --mixture ethylene_co --preset quick
```

快速模式适合检查环境配置、数据路径和训练流程是否正常。

## 完整运行

分别运行两个动态混合气体任务：

```bash
python train.py --data-dir data --mixture ethylene_co --preset full
python train.py --data-dir data --mixture ethylene_methane --preset full
```

也可以使用 `run_all.py` 一次性运行两个任务：

```bash
python run_all.py --data-dir data --preset full
```

## 输出结果

每次运行后，程序会在 `results/` 文件夹下创建一个带时间戳的结果目录，其中包含：

```text
metrics.json
predictions.csv
training_log.csv
summary.json
model.pt
prediction_<target>.png
```

各输出文件含义如下：

- `metrics.json`：测试集评价指标，包括 RMSE、MAE、R² 等；
- `predictions.csv`：模型预测值与真实值；
- `training_log.csv`：训练过程日志；
- `summary.json`：本次实验配置与结果摘要；
- `model.pt`：训练得到的 PyTorch 模型权重；
- `prediction_<target>.png`：不同目标气体浓度的预测曲线图。

## 方法描述

**DynaFE-Net** 首先将原始传感器读数转换为类似 log-resistance 的响应表示，以缓解不同通道尺度差异和极端值影响。随后，模型构造多尺度因果动态描述符，包括滞后特征、差分特征、变化斜率、指数滑动平均记忆、滚动统计特征以及传感器类型结构特征。最后，使用轻量级残差 MLP 学习这些可解释动态特征到目标气体浓度之间的非线性映射关系。

该设计将传感器特征工程的可解释性与神经网络的非线性建模能力相结合，在预测精度、训练稳定性和计算复杂度之间取得了较好的平衡，适用于动态混合气体浓度在线预测等场景。

## 项目目标

本项目旨在完成以下目标：

1. 理解动态生物化学 / 化学传感器数据的时序响应特性；
2. 掌握传感器信号预处理、动态特征构造和回归建模方法；
3. 对比传统机器学习方法与动态特征增强神经网络在混合气体浓度预测任务中的表现；
4. 探索数据特性驱动的特征工程与深度学习模型结合的有效性。

## 任务说明

本项目主要面向 UCI Gas Sensor Array Under Dynamic Gas Mixtures 数据集中的两个任务：

- Ethylene-CO 动态混合气体浓度预测；
- Ethylene-Methane 动态混合气体浓度预测。

模型输入为 16 通道化学传感器响应信号，输出为两个目标气体组分的浓度预测值。
