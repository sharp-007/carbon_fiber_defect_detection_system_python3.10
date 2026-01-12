## 碳纤维缺陷检测系统

本项目基于YOLO11 和 Streamlit，实现碳纤维缺陷检测，支持图片、视频和摄像头实时检测，生成可视化缺陷统计结果。

### 功能特性

- ✅ 基于 YOLO11 + Streamlit 实现碳纤维缺陷检测
- ✅ 支持图片、视频和摄像头实时缺陷检测，自动生成检测框标注缺陷位置，生成详细的检测结果统计图表
- ✅ 视频检测支持带检测框视频的播放与下载
- ✅ 摄像头实时检测支持实时检测框生成、关键帧图片展示与批量下载
- ✅ 支持自定义模型加载，支持自定义置信度、IoU阈值、检测频率、输出帧率、最大检测帧数，支持CPU和GPU推理加速

### 项目结构

```text
carbon_fiber_defect_detection_system/
  ├─ app.py                 # Streamlit 主应用
  ├─ train.py               # 模型训练脚本
  ├─ test.py                # 模型测试脚本（在测试集上评估）
  ├─ turn.py                # TURN 服务器配置（WebRTC）
  ├─ README.md              # 项目说明文档（本文件）
  ├─ requirements.txt       # Python 依赖包
  ├─ packages.txt           # 系统级依赖包（Streamlit Cloud）
  ├─ runtime.txt            # Python 版本配置（Streamlit Cloud）
  ├─ .gitignore             # Git 忽略文件配置
  ├─ .streamlit/            # Streamlit 配置目录
  │  └─ config.toml         # Streamlit 应用配置
  ├─ dataset/               # 数据集目录
  │  ├─ data.yaml           # 数据集配置文件
  │  ├─ train/              # 训练集（images + labels）
  │  ├─ valid/              # 验证集（images + labels）
  │  └─ test/               # 测试集（images + labels）
  ├─ model/                 # 模型文件目录
  │  ├─ yolo11n.pt          # YOLO11n 预训练模型
  │  ├─ best.pt             # 训练后的最佳模型（训练后生成）
  │  └─ last.pt             # 训练后的最后模型（训练后生成）
  ├─ log/                   # 日志目录（运行时生成）
  ├─ test_image_video/      # 测试图片和视频目录
  ├─ test_demo/             # 演示结果目录
  ├─ test_results/          # 测试集评估结果（运行 test.py 后生成）
  └─ runs/                  # 训练输出目录（训练后生成）
```

### 1. 环境配置

#### 方式一：使用 conda 环境（推荐）

```bash
# 激活 conda 环境（如果使用 defect_detect 环境）
conda activate defect_detect

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：新建虚拟环境

```bash
# 创建新的 conda 环境（使用 Python 3.10，Streamlit Cloud 最低要求）
conda create -n defect_detect python=3.10 -y
conda activate defect_detect

# 安装依赖
pip install -r requirements.txt
```

> **重要提示**：本项目已升级支持 Python 3.10+，以满足 Streamlit Cloud 部署要求。虽然项目最初在 Python 3.9 环境下开发，但所有代码和依赖都已兼容 Python 3.10，可以直接使用。

> **注意**：如果已安装 `streamlit`、`ultralytics` 等基础包，`pip` 会自动跳过已安装的包。

### 2. 运行应用

在项目根目录 `carbon_fiber_defect_detection_system` 下执行：

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`（或在命令行给出的地址）。

### 3. 使用说明

1. **在侧边栏配置：**

   - 选择 **检测模式**：图片检测 / 视频检测 / 摄像头实时检测
   - **模型配置**：
     - 勾选"使用默认模型"使用 `model/best.pt`（训练后生成的最佳模型）
     - 或上传 **YOLO 模型权重文件**（例如 `yolo11n.pt`、`best.pt` 等）
   - 选择设备（`cpu` 或 `cuda`，如果有 GPU）
   - 设置置信度阈值（默认 0.15）
   - 设置 IoU 阈值（默认 0.25，用于非极大值抑制）
2. **上传待检测文件或启动检测：**

   - **图片检测模式**：上传单张或多张图片
   - **视频检测模式**：上传视频文件，可选择抽帧间隔（默认每5帧检测一次）
   - **摄像头实时检测模式**：点击"开始摄像头检测"按钮，支持设置抽帧时间间隔和最大检测帧数
3. **查看检测结果：**

   - 页面中会展示带有检测框与缺陷类型的图片或视频帧
   - 侧边栏显示每类缺陷的数量统计
   - 支持下载检测结果图片、视频和 CSV 数据

### 4. 模型说明

- **支持的 YOLO 版本**：YOLO11、YOLOv8、YOLOv5 等（只要是 `ultralytics` 库支持的模型格式）
- **模型格式**：`.pt` 文件（PyTorch 权重文件）
- **预训练模型**：项目包含 `model/yolo11n.pt` 预训练模型，用于训练时的迁移学习
- **应用默认模型**：Streamlit 应用默认使用 `model/best.pt`（训练后生成的最佳模型），如果该文件不存在，需要手动上传模型文件
- **自定义模型**：你可以使用自己的训练数据训练模型后，在应用中加载使用（上传或使用默认的 `best.pt`）

> **提示**：如果代码中使用模型名称（如 `"yolo11n.pt"`）而不是本地文件路径，`ultralytics` 会在首次运行时自动从网上下载模型权重文件。本项目已包含 `model/yolo11n.pt` 预训练模型，使用本地路径时无需下载。

### 5. 数据集说明

项目包含碳纤维缺陷检测数据集：

- **数据集结构**：

  - `dataset/train/`：训练集（70 张图片）
  - `dataset/valid/`：验证集（9 张图片）
  - `dataset/test/`：测试集（5 张图片）
  - `dataset/data.yaml`：数据集配置文件
- **缺陷类别**：碳纤维缺陷（carbon-fibre-defect）
- **数据来源**：Roboflow（Carbon Fiber Defect Dataset）
- **标签格式**：支持标准 YOLO 边界框格式和多边形格式

### 5.1 模型训练

本项目提供了完整的训练脚本 `train.py`，支持基于 YOLO11n 预训练模型进行微调训练。

#### 训练指令

**快速测试（1轮训练）：**

```bash
python train.py --quick-test
```

**指定训练轮数：**

```bash
python train.py --epochs 1      # 训练1轮
python train.py --epochs 50     # 训练50轮
python train.py --epochs 100    # 训练100轮（默认）
```

**完整训练（默认100轮）：**

```bash
python train.py
```

#### 训练流程

训练脚本会自动执行以下步骤：

1. **训练阶段**：

   - 基于 `model/yolo11n.pt` 预训练模型进行微调
   - 使用 `dataset/train/` 数据集进行训练
   - 训练过程中使用 `dataset/valid/` 数据集进行验证
   - 自动检测设备（GPU/CPU）并调整批次大小
   - 生成 `best.pt`（验证集上表现最好的模型）和 `last.pt`（最后一个 epoch 的模型）
2. **验证阶段**：

   - 使用 `best.pt` 模型在 `dataset/valid/` 数据集上进行完整验证
   - 输出详细的验证指标：
     - mAP@0.5 和 mAP@0.5:0.95
     - 精确度 (Precision)
     - 召回率 (Recall)
     - F1-Score
     - 各类别的详细指标
3. **可视化阶段**：

   - 自动生成预测结果和真实标签的对比图片
   - 绿色框表示真实标签（Ground Truth）
   - 红色框表示预测结果（Prediction），包含类别名称和置信度

#### 输出文件位置

训练完成后，相关文件保存在以下位置（每次训练会创建新的 run 文件夹，如 run1、run2 等）：

- **训练结果目录**：`runs/run1/train/`

  - `weights/best.pt`：最佳模型（验证集上表现最好，推荐使用）
  - `weights/last.pt`：最后一个 epoch 的模型
  - `args.yaml`：训练参数配置
  - `results.csv`：训练指标 CSV 文件（每个 epoch 的 loss 和 metrics）
  - `results.png`：训练曲线图（loss 和 metrics 变化）
  - `confusion_matrix.png`：混淆矩阵（训练时验证集）
  - `confusion_matrix_normalized.png`：归一化混淆矩阵
  - `BoxPR_curve.png`：PR 曲线（Precision-Recall）
  - `BoxF1_curve.png`：F1 曲线
  - `BoxP_curve.png`：Precision 曲线
  - `BoxR_curve.png`：Recall 曲线
- **验证结果目录**：`runs/run1/val/`

  - `validation_summary.json`：验证结果摘要（JSON 格式）
  - `confusion_matrix.png`：验证集混淆矩阵
  - `confusion_matrix_normalized.png`：归一化混淆矩阵
  - `BoxPR_curve.png`：PR 曲线
  - `BoxF1_curve.png`：F1 曲线
  - `BoxP_curve.png`：Precision 曲线
  - `BoxR_curve.png`：Recall 曲线
  - `validation_visualizations/`：预测与真实标签对比图片（10 张示例）

#### 训练参数说明

- **预训练模型**：`model/yolo11n.pt`
- **数据集配置**：`dataset/data.yaml`
- **图像尺寸**：640×640
- **批次大小**：自动调整（GPU: 16, CPU: 4）
- **设备**：自动检测（优先使用 GPU）
- **训练轮数**：可通过命令行参数指定（默认 100 轮）

#### 数据增强配置（针对小数据集优化）

训练脚本已配置以下数据增强策略，以提高模型泛化能力：

- **颜色增强**：

  - HSV-H=0.015：色调增强，增加颜色多样性
  - HSV-S=0.7：饱和度增强，增强颜色鲜艳度
  - HSV-V=0.4：明度/亮度增强，增强亮度变化
- **几何变换**：

  - 旋转：±10.0°（增加旋转鲁棒性）
  - 平移：0.1（10% 平移比例）
  - 缩放：0.5（50% 缩放比例）
  - 左右翻转：0.5（50% 概率）
- **高级增强**：

  - Mosaic=1.0：Mosaic 增强概率 100%（将 4 张图片拼接成 1 张）
  - MixUp=0.1：MixUp 增强概率 10%（混合两张图片和标签）

#### NMS 和置信度阈值配置

- **IoU 阈值**：0.7（用于 NMS 非极大值抑制，去除重叠检测框）
- **置信度阈值**：0.001（用于验证时过滤低置信度检测框，训练/验证时使用较低值以评估所有可能的检测）

#### 注意事项

- 训练 1 轮主要用于快速测试代码和环境是否正确
- 实际训练建议使用更多轮数（如 50-100 轮）以获得更好的模型性能
- 如果有 GPU，训练速度会更快
- 训练过程可以随时按 `Ctrl+C` 中断，模型会保存到当前检查点
- 训练完成后，模型保存在 `runs/run1/train/weights/best.pt`（或 run2、run3 等，取决于运行次数）

**关于 best.pt 文件：**

- `best.pt` 是验证集上表现最好的模型，由 Ultralytics 库在每个 epoch 后自动保存
- `best.pt` 的生成条件：
  - ✅ `val=True`（训练时进行验证）- 已设置
  - ✅ `save=True`（保存模型检查点）- 已设置
  - ✅ 至少完成 1 个完整的 epoch
  - ✅ 验证集验证成功（有有效的 mAP 指标）
- 如果训练轮数过少（如只训练 1 轮）或训练被中断，可能只有 `last.pt` 而没有 `best.pt`
- 训练脚本会自动处理这种情况：
  - 如果 `best.pt` 不存在但 `last.pt` 存在，会自动使用 `last.pt` 进行验证
  - 如果两个文件都不存在，会给出详细的错误提示

**常见问题排查：**

如果 `best.pt` 没有生成，请检查：

1. **训练是否完成**：确保训练至少完成了 1 个完整的 epoch
2. **验证集是否正常**：检查 `dataset/valid/` 目录是否存在且包含图片和标签
3. **训练是否被中断**：如果训练过程中按 `Ctrl+C` 中断，可能只有 `last.pt`
4. **训练轮数**：如果只训练 1 轮，可能还没有生成 `best.pt`（建议至少训练 10 轮以上）

**使用训练后的模型：**

- 方式一：将 `runs/run1/train/weights/best.pt` 复制到 `model/best.pt`，然后在 Streamlit 应用中勾选"使用默认模型"
- 方式二：在 Streamlit 应用中直接上传 `runs/run1/train/weights/best.pt` 文件
- 如果只有 `last.pt`，也可以使用 `last.pt`（但性能可能不如 `best.pt`）

### 5.2 模型测试

本项目提供了完整的测试脚本 `test.py`，用于在测试集上评估训练好的模型性能。

#### 测试指令

**基本使用（自动查找最新的 best.pt 模型）：**

```bash
python test.py
```

**选择模型类型（best 或 last）：**

```bash
# 使用最新的 best.pt（默认）
python test.py --model-type best

# 使用最新的 last.pt
python test.py --model-type last
```

**指定模型路径：**

```bash
python test.py --model runs/run1/train/weights/best.pt
```

**自定义置信度和 IoU 阈值：**

```bash
python test.py --conf 0.3 --iou 0.5
```

**组合使用所有参数：**

```bash
# 使用 last.pt，并自定义阈值
python test.py --model-type last --conf 0.3 --iou 0.5

# 指定模型路径，并自定义阈值
python test.py --model runs/run3/train/weights/best.pt --conf 0.35 --iou 0.65
```

#### 测试流程

测试脚本会自动执行以下步骤：

1. **模型加载**：

   - 自动查找最新的 `best.pt` 模型（从 `runs/run1/train/weights/` 或 `model/` 目录）
   - 或使用指定的模型路径
2. **测试集评估**：

   - 在 `dataset/test/` 测试集上评估模型
   - 计算详细的测试指标：
     - mAP@0.5 和 mAP@0.5:0.95
     - 精确度 (Precision)
     - 召回率 (Recall)
     - F1-Score
     - 各类别的详细指标
3. **可视化阶段**：

   - 自动生成所有测试图片的预测结果和真实标签对比图片
   - 绿色框表示真实标签（Ground Truth）
   - 红色框表示预测结果（Prediction），包含类别名称和置信度

#### 测试输出文件位置

测试完成后，相关文件保存在以下位置：

- **测试结果目录**：`test_results/`（根目录下的单独文件夹）
  - `test_summary.json`：测试结果摘要（JSON 格式）
  - `confusion_matrix.png`：测试集混淆矩阵
  - `confusion_matrix_normalized.png`：归一化混淆矩阵
  - `BoxPR_curve.png`：PR 曲线（Precision-Recall）
  - `BoxF1_curve.png`：F1 曲线
  - `BoxP_curve.png`：Precision 曲线
  - `BoxR_curve.png`：Recall 曲线
  - `test_visualizations/`：预测与真实标签对比图片（所有测试图片）

#### 测试参数说明

**命令行参数：**

- `--model`：模型文件路径（可选）

  - 如果指定，直接使用该路径
  - 如果未指定，自动查找模型（见 `--model-type` 说明）
  - 示例：`--model runs/run3/train/weights/best.pt`
- `--model-type`：模型类型选择（仅在未指定 `--model` 时有效）

  - `best`：使用最佳模型（验证集上 mAP 最高的模型，默认值）
  - `last`：使用最后一个 epoch 的模型
  - 查找顺序：`runs/run{N}/train/weights/{model_type}.pt` → `model/{model_type}.pt`
  - 示例：`--model-type last`
- `--conf`：置信度阈值（默认：0.15）

  - 范围：0.0-1.0
  - 值越高，检测越严格（可能漏检）
  - 值越低，检测越宽松（可能误检）
  - 示例：`--conf 0.3`
- `--iou`：IoU 阈值，用于 NMS（默认：0.25）

  - 范围：0.0-1.0
  - 值越高，重叠框保留越多
  - 值越低，重叠框过滤越严格
  - 示例：`--iou 0.5`

**其他配置：**

- **设备**：自动检测（优先使用 GPU，如果可用）
- **批次大小**：自动调整（GPU: 16, CPU: 4）
- **图像尺寸**：640×640

#### 测试注意事项

- **测试集用途**：测试集用于最终评估模型性能，不应参与训练过程
- **模型选择**：
  - `best.pt`：验证集上表现最好的模型（推荐用于最终测试）
  - `last.pt`：最后一个 epoch 的模型（可能性能不如 best.pt）
- **阈值调整**：
  - 可以根据实际需求调整置信度和 IoU 阈值
  - 建议先用默认值测试，再根据结果调整
- **结果对比**：测试结果可以用于对比不同训练轮数、超参数设置或模型版本的效果
- **自动查找**：测试脚本会自动查找最新的训练模型，也可以手动指定模型路径
- **测试前准备**：测试前请确保已完成模型训练，或手动指定模型路径

#### 使用示例

**示例 1：快速测试（使用默认参数）**

```bash
python test.py
```

**示例 2：测试 last.pt 模型**

```bash
python test.py --model-type last
```

**示例 3：更严格的检测（高置信度，低 IoU）**

```bash
python test.py --conf 0.5 --iou 0.4
```

**示例 4：更宽松的检测（低置信度，高 IoU）**

```bash
python test.py --conf 0.15 --iou 0.8
```

**示例 5：测试特定训练轮次的模型**

```bash
python test.py --model runs/run2/train/weights/best.pt --conf 0.3 --iou 0.6
```

### 6. Ultralytics 库详解

#### 6.1 什么是 Ultralytics？

**Ultralytics** 是一个现代化的、开源的计算机视觉框架，专门用于 YOLO（You Only Look Once）目标检测模型的训练、验证和推理。它提供了统一的 API，支持 YOLOv5、YOLOv8、YOLOv9、YOLOv10、YOLO11 等多个版本的 YOLO 模型。

#### 6.2 Ultralytics 核心功能

1. **模型训练（Training）**

   - 支持从预训练模型进行迁移学习
   - 自动数据增强和数据加载
   - 实时训练监控和指标可视化
   - 支持断点续训（resume training）
   - 自动保存最佳模型（best.pt）和最新模型（last.pt）
2. **模型验证（Validation）**

   - 自动计算 mAP（mean Average Precision）指标
   - 支持 mAP@0.5 和 mAP@0.5:0.95
   - 生成混淆矩阵、PR 曲线等可视化结果
   - 支持每个类别的详细性能指标
3. **模型推理（Inference）**

   - 支持图片、视频、摄像头实时推理
   - 批量处理多张图片
   - 可配置置信度阈值和 IoU 阈值
   - 自动绘制检测框和标签
   - 支持 CPU 和 GPU（CUDA）加速
4. **模型导出（Export）**

   - 支持导出为多种格式：ONNX、TensorRT、CoreML、TensorFlow 等
   - 便于模型部署到不同平台
   - 优化模型大小和推理速度
5. **数据集管理**

   - 支持 YOLO 格式数据集
   - 自动数据集验证和统计
   - 支持多种数据增强策略

#### 6.3 在本项目中的应用

本项目充分利用了 Ultralytics 库的强大功能：

**在 `app.py` 中的应用：**

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("model/yolo11n.pt")

# 图片推理
results = model.predict(
    source=image,
    conf=0.15,      # 置信度阈值（应用默认值：0.15）
    iou=0.25,       # IoU 阈值（应用默认值：0.25，用于 NMS）
    device="cuda",  # 设备选择
    verbose=False   # 是否显示详细信息
)

# 获取检测结果
for result in results:
    boxes = result.boxes.xyxy      # 边界框坐标
    scores = result.boxes.conf     # 置信度分数
    classes = result.boxes.cls     # 类别ID
    names = result.names           # 类别名称字典
```

**在 `train.py` 中的应用：**

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("model/yolo11n.pt")

# 训练模型
results = model.train(
    data="dataset/data.yaml",      # 数据集配置文件
    epochs=100,                    # 训练轮数
    imgsz=640,                     # 图像尺寸
    batch=16,                      # 批次大小
    device="cuda",                 # 设备选择
    project="runs/run1",           # 项目目录（每次训练创建新的run文件夹）
    name="train",                  # 训练结果文件夹名称
    save=True,                     # 保存检查点
    val=True                       # 训练时进行验证
)

# 验证模型
metrics = model.val(
    data="dataset/data.yaml",
    device="cuda"
)
```

#### 6.4 主要应用场景

Ultralytics YOLO 广泛应用于各种计算机视觉任务，以下是主要应用场景：

1. **工业检测与质量控制**

   - 产品缺陷检测（如本项目中的碳纤维缺陷检测）
   - 表面质量检测
   - 零件识别与分类
   - 装配线质量监控
   - 异常检测
2. **安防监控**

   - 人员检测与跟踪
   - 车辆识别与计数
   - 入侵检测
   - 行为分析
   - 异常事件检测
3. **自动驾驶与交通**

   - 车辆检测
   - 行人检测
   - 交通标志识别
   - 车道线检测
   - 障碍物识别
4. **医疗影像分析**

   - 医学影像中的病灶检测
   - X光片异常检测
   - 细胞识别与计数
   - 医疗器械识别
5. **零售与电商**

   - 商品识别与分类
   - 货架监控
   - 顾客行为分析
   - 库存管理
6. **农业与环境**

   - 农作物病虫害检测
   - 植物生长监测
   - 野生动物识别
   - 环境监测
7. **体育与娱乐**

   - 运动员动作分析
   - 球类跟踪
   - 比赛数据分析
   - 视频内容分析
8. **机器人视觉**

   - 物体抓取与操作
   - 导航与避障
   - 场景理解
   - 人机交互

#### 6.5 Ultralytics 主要优势

1. **简单易用**：统一的 API 设计，几行代码即可完成训练和推理
2. **高性能**：优化的模型架构，推理速度快
3. **功能完整**：从数据准备到模型部署的完整流程
4. **持续更新**：活跃的社区支持，定期发布新版本和模型
5. **跨平台**：支持 Windows、Linux、macOS，支持 CPU 和 GPU
6. **文档完善**：详细的官方文档和丰富的示例代码

#### 6.6 常用 API 参考

**模型加载：**

```python
from ultralytics import YOLO

# 加载预训练模型（会自动下载）
model = YOLO("yolo11n.pt")  # nano 版本（最小最快）
model = YOLO("yolo11s.pt")  # small 版本
model = YOLO("yolo11m.pt")  # medium 版本
model = YOLO("yolo11l.pt")  # large 版本
model = YOLO("yolo11x.pt")  # xlarge 版本（最大最准确）

# 加载自定义训练模型
model = YOLO("runs/run1/train/weights/best.pt")
```

**推理参数说明：**

- `source`: 输入源（图片路径、视频路径、摄像头ID、numpy数组等）
- `conf`: 置信度阈值（0.0-1.0），应用默认值 0.15（Ultralytics 库默认 0.25）
- `iou`: IoU 阈值，用于非极大值抑制（NMS），应用默认值 0.25（Ultralytics 库默认 0.7）
- `device`: 设备选择（"cpu"、"cuda"、"0"、"1" 等）
- `imgsz`: 推理图像尺寸，默认 640
- `save`: 是否保存结果图片
- `show`: 是否显示结果（在 Jupyter 中有效）
- `verbose`: 是否打印详细信息

**训练参数说明：**

- `data`: 数据集配置文件路径（YAML 格式）
- `epochs`: 训练轮数
- `batch`: 批次大小（-1 表示自动调整）
- `imgsz`: 训练图像尺寸
- `device`: 设备选择
- `workers`: 数据加载线程数
- `project`: 项目保存目录
- `name`: 实验名称
- `resume`: 是否从检查点恢复训练

#### 6.7 详细使用方法

##### 6.7.1 快速开始 - 图片检测

最简单的使用方式，只需几行代码：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 单张图片检测
results = model("path/to/image.jpg")

# 查看结果
for result in results:
    # 显示结果（在 Jupyter 中）
    result.show()
  
    # 保存结果
    result.save("output.jpg")
  
    # 获取检测框信息
    boxes = result.boxes
    for box in boxes:
        print(f"类别: {box.cls}, 置信度: {box.conf}, 坐标: {box.xyxy}")
```

##### 6.7.2 视频检测

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 视频文件检测
results = model.predict(
    source="path/to/video.mp4",
    conf=0.25,
    save=True,
    save_txt=True  # 保存检测结果到文本文件
)

# 实时摄像头检测
results = model.predict(
    source=0,  # 0 表示默认摄像头
    conf=0.25,
    show=True  # 实时显示结果
)
```

##### 6.7.3 批量图片处理

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

# 批量处理文件夹中的所有图片
image_dir = Path("path/to/images")
results = model.predict(
    source=str(image_dir),
    conf=0.25,
    save=True,
    save_txt=True
)
```

##### 6.7.4 自定义训练

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo11n.pt")

# 训练模型
results = model.train(
    data="dataset/data.yaml",  # 数据集配置文件
    epochs=100,                # 训练轮数
    imgsz=640,                 # 图像尺寸
    batch=16,                  # 批次大小
    device="cuda",             # 使用 GPU
    project="runs/run1",       # 项目目录（每次训练创建新的run文件夹）
    name="train",              # 训练结果文件夹名称
    patience=50,               # 早停耐心值
    save=True,                 # 保存检查点
    save_period=10,            # 每10个epoch保存一次
    val=True,                  # 训练时进行验证
    plots=True                 # 生成训练图表
)

# 训练完成后，最佳模型保存在：
# runs/run1/train/weights/best.pt
```

##### 6.7.5 模型验证

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("runs/run1/train/weights/best.pt")

# 在验证集上验证
metrics = model.val(
    data="dataset/data.yaml",
    conf=0.25,
    iou=0.45,
    device="cuda",
    plots=True  # 生成混淆矩阵、PR曲线等
)

# 查看验证指标
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
print(f"精确度: {metrics.box.mp}")
print(f"召回率: {metrics.box.mr}")
```

##### 6.7.6 模型导出

将训练好的模型导出为其他格式，便于部署：

```python
from ultralytics import YOLO

model = YOLO("best.pt")

# 导出为 ONNX 格式（用于 TensorRT、OpenVINO 等）
model.export(format="onnx")

# 导出为 TensorRT 格式（NVIDIA GPU 加速）
model.export(format="engine", device=0)

# 导出为 CoreML 格式（Apple 设备）
model.export(format="coreml")

# 导出为 TensorFlow 格式
model.export(format="tflite")

# 导出为 OpenVINO 格式（Intel 设备）
model.export(format="openvino")
```

##### 6.7.7 结果处理与可视化

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n.pt")
results = model("image.jpg")

# 获取第一个结果
result = results[0]

# 获取检测框（多种格式）
boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
boxes_xywh = result.boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
boxes_xyxyn = result.boxes.xyxyn.cpu().numpy()  # 归一化坐标

# 获取置信度和类别
confidences = result.boxes.conf.cpu().numpy()
class_ids = result.boxes.cls.cpu().int().numpy()
class_names = [result.names[int(cid)] for cid in class_ids]

# 获取原始图片
original_image = result.orig_img

# 绘制结果（使用 OpenCV）
annotated_image = result.plot()  # 返回绘制了检测框的图片

# 保存结果
cv2.imwrite("annotated_image.jpg", annotated_image)

# 获取每个检测框的详细信息
for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
    x1, y1, x2, y2 = box
    cls_name = result.names[int(cls_id)]
    print(f"检测 {i+1}: {cls_name} (置信度: {conf:.2f}) 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

##### 6.7.8 高级功能

**1. 断点续训：**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 训练时如果中断，可以从检查点恢复
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    resume=True  # 从最新的检查点恢复训练
)
```

**2. 自定义数据增强：**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    hsv_h=0.015,      # 色调增强
    hsv_s=0.7,        # 饱和度增强
    hsv_v=0.4,        # 明度增强
    degrees=10.0,     # 旋转角度
    translate=0.1,    # 平移
    scale=0.5,        # 缩放
    flipud=0.0,       # 上下翻转概率
    fliplr=0.5,       # 左右翻转概率
    mosaic=1.0,       # Mosaic 增强概率
    mixup=0.1         # MixUp 增强概率
)
```

**3. 多GPU训练：**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 使用多个GPU训练
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    device=[0, 1, 2, 3]  # 使用4个GPU
)
```

**4. 模型集成（Ensemble）：**

```python
from ultralytics import YOLO

# 加载多个模型
model1 = YOLO("best_model1.pt")
model2 = YOLO("best_model2.pt")
model3 = YOLO("best_model3.pt")

# 对同一张图片进行预测
results1 = model1("image.jpg")
results2 = model2("image.jpg")
results3 = model3("image.jpg")

# 可以合并多个模型的结果以提高准确率
```

##### 6.7.9 性能优化技巧

1. **选择合适的模型大小**：

   - `yolo11n.pt`: 最小最快，适合实时应用
   - `yolo11s.pt`: 平衡速度和精度
   - `yolo11m.pt`: 中等精度
   - `yolo11l.pt`: 较高精度
   - `yolo11x.pt`: 最高精度，但速度较慢
2. **调整推理参数**：

   ```python
   # 提高速度（降低精度）
   results = model.predict(source="image.jpg", imgsz=320, conf=0.5)

   # 提高精度（降低速度）
   results = model.predict(source="image.jpg", imgsz=1280, conf=0.1)
   ```
3. **使用 TensorRT 加速**：

   ```python
   # 导出为 TensorRT 格式
   model.export(format="engine", device=0)

   # 使用 TensorRT 模型推理（速度提升2-5倍）
   model_trt = YOLO("model.engine")
   results = model_trt("image.jpg")
   ```
4. **批量处理优化**：

   ```python
   # 批量处理多张图片（比循环处理更快）
   results = model.predict(source=["img1.jpg", "img2.jpg", "img3.jpg"], batch=8)
   ```

#### 6.8 学习资源

- **官方文档**：[https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **GitHub 仓库**：[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **模型库**：[https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)
- **社区论坛**：[https://community.ultralytics.com/](https://community.ultralytics.com/)

#### 6.9 版本兼容性

本项目使用 `ultralytics>=8.0.0`，支持以下功能：

- ✅ YOLOv5、YOLOv8、YOLOv9、YOLOv10、YOLO11 模型
- ✅ 完整的训练、验证、推理流程
- ✅ GPU 和 CPU 推理加速
- ✅ 模型导出和部署
- ✅ 实时视频流处理

### 7. 依赖说明

**Python 版本要求**：

- **最低版本**：Python 3.10（Streamlit Cloud 部署要求）
- **推荐版本**：Python 3.10 或更高版本
- 项目最初在 Python 3.9 环境下开发，现已完全兼容 Python 3.10+

**主要依赖包**：

| 依赖包                     | 版本要求        | 用途                    |
| -------------------------- | --------------- | ----------------------- |
| `streamlit`              | >=1.18.0        | Web 应用框架            |
| `streamlit-webrtc`       | >=0.44.0        | 摄像头模式 WebRTC 支持  |
| `av`                     | >=10.0.0        | WebRTC 视频处理         |
| `twilio`                 | >=8.0.0         | TURN 服务器支持（可选） |
| `ultralytics`            | >=8.0.0         | YOLO 模型库（核心依赖） |
| `numpy`                  | >=1.23.0,<2.0.0 | 数值计算库              |
| `pandas`                 | >=1.5.0         | 数据处理和分析          |
| `opencv-python-headless` | >=4.5.0,<5.0.0  | 图像处理（无 GUI 版本） |
| `Pillow`                 | >=9.0.0         | 图像处理库              |
| `imageio`                | >=2.25.0        | 视频处理                |
| `imageio-ffmpeg`         | >=0.4.8         | 视频编解码              |
| `matplotlib`             | >=3.5.0         | 绘图库                  |
| `altair`                 | >=4.2.0         | 交互式数据可视化        |
| `pyyaml`                 | >=6.0.0         | YAML 配置文件解析       |

> **注意**：
>
> - `ultralytics` 会自动安装其依赖项（如 `torch`、`torchvision` 等）。如果需要 GPU 支持，请确保安装了 CUDA 版本的 PyTorch。
> - `twilio` 是可选依赖，用于在 Streamlit Cloud 上配置 TURN 服务器以获得更稳定的 WebRTC 连接。如果未配置，会自动回退到免费的 Google STUN 服务器。
> - `numpy<2.0.0` 限制是为了确保与其他依赖库的兼容性。

#### 7.1 Streamlit Cloud 部署

本项目已配置支持 Streamlit Cloud 部署，包含以下配置文件：

- **`runtime.txt`**：指定 Python 3.10 版本（Streamlit Cloud 最低要求）
- **`packages.txt`**：系统级依赖包列表（用于安装 OpenCV 所需的系统库，如 `libGL.so.1`）
- **`.streamlit/config.toml`**：Streamlit 应用配置文件

部署步骤：

1. **将项目推送到 GitHub**：

   ```bash
   git add .
   git commit -m "升级到 Python 3.10，支持 Streamlit Cloud 部署"
   git push origin main
   ```
2. **在 Streamlit Cloud 上部署**：

   - 访问 [Streamlit Cloud](https://streamlit.io/cloud)
   - 使用 GitHub 账号登录
   - 点击 "New app"
   - 选择你的 GitHub 仓库
   - 设置主文件路径为 `app.py`
   - Streamlit Cloud 会自动检测 `runtime.txt` 并使用 Python 3.10
3. **部署注意事项**：

   - 确保 `model/` 目录中的模型文件（如 `best.pt`、`yolo11n.pt`）已提交到仓库
   - 如果模型文件过大，可以考虑使用 Git LFS 或外部存储
   - Streamlit Cloud 提供免费的 CPU 资源，GPU 推理可能较慢
   - 首次部署可能需要几分钟时间来安装依赖
   - **重要**：如果遇到 `ImportError: libGL.so.1` 错误，确保 `packages.txt` 文件存在并包含 `libgl1-mesa-glx` 和 `libglib2.0-0`

### 8. 技术特点

- **模型缓存**：使用 `st.cache_resource` 缓存模型，避免重复加载
- **设备选择**：支持 CPU 和 GPU（CUDA）推理
- **实时检测**：支持图片、视频和摄像头流的实时检测
- **结果可视化**：自动绘制检测框、类别标签和置信度
- **统计分析**：实时统计各类缺陷的数量，生成时间分布图表
- **多模式支持**：图片检测、视频检测（支持抽帧）、摄像头实时检测（支持时间间隔控制）
- **参数可调**：支持自定义置信度阈值（默认 0.15）、IoU 阈值（默认 0.25）等

### 9. 开发信息

- **项目作者**：Joyce Pan
- **联系邮箱**：[panjiao007@126.com](mailto:panjiao007@126.com)
- **Github**：[sharp-007/carbon_fiber_defect_detection_system_pyhton3.10](https://github.com/sharp-007/carbon_fiber_defect_detection_system_pyhton3.10)
- **在线演示**：[https://carbon-fiber-defect-detection-system.streamlit.app/](https://carbon-fiber-defect-detection-system.streamlit.app/)
- **版本控制**：使用 Git 进行版本管理

### 10. 常见问题

**Q: 模型加载失败怎么办？**
A: 请确保已安装 `ultralytics` 库：`pip install ultralytics`

**Q: 支持 GPU 加速吗？**
A: 支持。如果有 NVIDIA GPU 并已安装 CUDA 和 PyTorch GPU 版本，在侧边栏选择 `cuda` 设备即可。

**Q: 可以使用其他 YOLO 模型吗？**
A: 可以。只要模型是 `.pt` 格式且能被 `ultralytics` 库加载，都可以使用。

**Q: 如何训练自定义模型？**
A: 项目已提供完整的训练脚本 `train.py`。运行 `python train.py --help` 查看使用说明。训练完成后，使用生成的 `best.pt` 模型在 Streamlit 应用中进行检测。

**Q: 如何部署到 Streamlit Cloud？**
A: 项目已配置支持 Streamlit Cloud 部署。确保使用 Python 3.10+，将代码推送到 GitHub，然后在 Streamlit Cloud 上连接仓库即可。详细步骤请参考第 8.1 节。

**Q: Python 3.9 和 Python 3.10 有什么区别？**
A: 本项目代码完全兼容 Python 3.10，无需修改。主要区别是 Streamlit Cloud 平台要求最低 Python 3.10 版本。所有依赖包都已兼容 Python 3.10。

### 11. 许可证

本项目采用 **MIT 许可证** 开源。

**数据集许可证**：CC BY 4.0（来自 Roboflow）

**第三方依赖许可证**：

- Ultralytics (YOLO): AGPL-3.0
- Streamlit: Apache-2.0
- OpenCV: Apache-2.0
- PyTorch: BSD-3-Clause
