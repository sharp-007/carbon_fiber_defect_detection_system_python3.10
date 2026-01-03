"""
使用 YOLO11n 预训练模型进行微调训练

基于 dataset 目录中的 train 和 valid 数据集进行训练，
训练完成后会生成 best.pt 和 last.pt 模型文件。
"""

from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import argparse


def read_yolo_label(label_path: Path, img_width: int, img_height: int) -> List[Tuple[float, float, float, float, int]]:
    """
    读取 YOLO 格式的标签文件
    支持两种格式：
    1. 标准边界框格式: class_id x_center y_center width height
    2. 多边形格式: class_id x1 y1 x2 y2 x3 y3 ... (需要转换为边界框)
    
    Args:
        label_path: 标签文件路径
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        List of (x1, y1, x2, y2, class_id) in pixel coordinates
    """
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            values = [float(x) for x in parts[1:]]
            
            if len(values) == 4:
                # 标准 YOLO 边界框格式: x_center y_center width height
                x_center, y_center, width, height = values
                
                # 转换为像素坐标 (x1, y1, x2, y2)
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                
                boxes.append((x1, y1, x2, y2, class_id))
            else:
                # 多边形格式: x1 y1 x2 y2 x3 y3 ...
                # 计算多边形的边界框（最小外接矩形）
                x_coords = []
                y_coords = []
                
                # 交替提取 x 和 y 坐标
                for i in range(0, len(values), 2):
                    if i + 1 < len(values):
                        x_coords.append(values[i] * img_width)
                        y_coords.append(values[i + 1] * img_height)
                
                if x_coords and y_coords:
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)
                    
                    boxes.append((x1, y1, x2, y2, class_id))
    
    return boxes


def draw_boxes_on_image(
    img: np.ndarray,
    pred_boxes: List[Tuple[float, float, float, float]],
    pred_classes: List[int],
    pred_scores: List[float],
    gt_boxes: List[Tuple[float, float, float, float, int]],
    class_names: List[str],
) -> np.ndarray:
    """
    在图片上绘制预测框和真实框
    
    Args:
        img: BGR格式的图片 (numpy array)
        pred_boxes: 预测框列表 [(x1, y1, x2, y2), ...]
        pred_classes: 预测类别ID列表
        pred_scores: 预测置信度列表
        gt_boxes: 真实框列表 [(x1, y1, x2, y2, class_id), ...]
        class_names: 类别名称列表
    
    Returns:
        绘制了框的图片 (BGR格式)
    """
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # 绘制真实框 (绿色)
    for x1, y1, x2, y2, class_id in gt_boxes:
        x1_i = int(max(0, min(w - 1, x1)))
        y1_i = int(max(0, min(h - 1, y1)))
        x2_i = int(max(0, min(w - 1, x2)))
        y2_i = int(max(0, min(h - 1, y2)))
        
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        color = (0, 255, 0)  # BGR: 绿色表示真实框
        cv2.rectangle(img_copy, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        
        label = f"GT: {class_name}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        
        # 确保标签不超出图片边界
        label_y = max(th, y1_i)  # 如果框在上边缘，标签放在框内
        label_y1 = label_y - th
        if label_y1 < 0:
            label_y1 = y1_i
            label_y = y1_i + th
        
        cv2.rectangle(img_copy, (x1_i, label_y1), (x1_i + tw, label_y), color, -1)
        cv2.putText(
            img_copy,
            label,
            (x1_i, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    
    # 绘制预测框 (红色)
    for (x1, y1, x2, y2), class_id, score in zip(pred_boxes, pred_classes, pred_scores):
        x1_i = int(max(0, min(w - 1, x1)))
        y1_i = int(max(0, min(h - 1, y1)))
        x2_i = int(max(0, min(w - 1, x2)))
        y2_i = int(max(0, min(h - 1, y2)))
        
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        color = (0, 0, 255)  # BGR: 红色表示预测框
        cv2.rectangle(img_copy, (x1_i, y1_i), (x2_i, y2_i), color, 2)
        
        label = f"Pred: {class_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        th = th + baseline
        
        # 标签放在框的下方，如果空间不足则放在上方
        if y2_i + th < h:
            # 放在下方
            label_y1 = y2_i
            label_y2 = y2_i + th
            text_y = y2_i + th - baseline
        else:
            # 放在上方
            label_y1 = max(0, y1_i - th)
            label_y2 = y1_i
            text_y = y1_i - baseline
        
        cv2.rectangle(img_copy, (x1_i, label_y1), (x1_i + tw, label_y2), color, -1)
        cv2.putText(
            img_copy,
            label,
            (x1_i, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    
    return img_copy


def visualize_validation_results(
    model: YOLO,
    val_images_dir: Path,
    val_labels_dir: Path,
    class_names: List[str],
    output_dir: Path,
    device: str = "cpu",
    conf_threshold: float = 0.25,
    max_images: int = 20,
):
    """
    生成验证集的预测和真实标签对比可视化图片
    
    Args:
        model: YOLO模型
        val_images_dir: 验证集图片目录
        val_labels_dir: 验证集标签目录
        class_names: 类别名称列表
        output_dir: 输出目录
        device: 设备 ('cpu' 或 'cuda')
        conf_threshold: 置信度阈值
        max_images: 最多可视化的图片数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
    image_files = image_files[:max_images]  # 限制数量
    
    print(f"\n开始生成验证集可视化图片（共 {len(image_files)} 张）...")
    
    for img_idx, img_path in enumerate(image_files, 1):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 读取真实标签
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_label(label_path, img_width, img_height)
        
        # 进行预测
        results = model.predict(
            source=img,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        
        if not results:
            continue
        
        r = results[0]
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        
        if hasattr(r.boxes, "xyxy") and len(r.boxes.xyxy) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, cls_id, score in zip(boxes_xyxy, cls_ids, scores):
                pred_boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                pred_classes.append(int(cls_id))
                pred_scores.append(float(score))
        
        # 绘制对比图
        vis_img = draw_boxes_on_image(
            img,
            pred_boxes,
            pred_classes,
            pred_scores,
            gt_boxes,
            class_names,
        )
        
        # 保存图片
        output_path = output_dir / f"{img_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_path), vis_img)
        
        if img_idx % 5 == 0:
            print(f"  已处理 {img_idx}/{len(image_files)} 张图片...")
    
    print(f"可视化图片已保存到: {output_dir}")
    print(f"  - 绿色框: 真实标签 (Ground Truth)")
    print(f"  - 红色框: 预测结果 (Prediction)")


def train_model(epochs: int = 100):
    """
    训练 YOLO11n 模型
    
    Args:
        epochs: 训练轮数，默认100轮
    """
    
    # 项目根目录
    project_root = Path(__file__).parent
    
    # 预训练模型路径（yolo11n.pt）
    pretrained_model = project_root / "model" / "yolo11n.pt"
    
    # 数据集配置文件路径
    data_yaml = project_root / "dataset" / "data.yaml"
    
    # 检查文件是否存在
    if not pretrained_model.exists():
        raise FileNotFoundError(f"预训练模型文件不存在: {pretrained_model}")
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 读取数据集配置以获取类别名称
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [])
    
    # 自动检测设备（GPU或CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16 if device == 'cuda' else 4  # CPU使用较小的批次大小
    
    print(f"预训练模型: {pretrained_model}")
    print(f"数据集配置: {data_yaml}")
    print(f"训练数据集: {data_config.get('train', 'N/A')}")
    print(f"验证数据集: {data_config.get('val', 'N/A')}")
    print(f"类别数量: {data_config.get('nc', 'N/A')}")
    print(f"类别名称: {class_names}")
    print(f"训练设备: {device} ({'GPU' if device == 'cuda' else 'CPU'})")
    print(f"批次大小: {batch_size}")
    print("\n数据增强配置:")
    print(f"  - 颜色增强: HSV-H=0.015, HSV-S=0.7, HSV-V=0.4 (亮度调节)")
    print(f"  - 几何变换: 旋转±10.0°, 平移=0.1, 缩放=0.5, 左右翻转=0.5")
    print(f"  - 高级增强: Mosaic=1.0, MixUp=0.1")
    print("\n验证参数配置:")
    print(f"  - IoU阈值: 0.7 (用于NMS非极大值抑制，去除重叠检测框)")
    print(f"  - 置信度阈值: 0.001 (用于过滤低置信度检测框)")
    print("-" * 50)
    
    # 加载预训练模型
    model = YOLO(str(pretrained_model))
    
    # 自动生成运行文件夹名称（run1, run2, run3...）
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找已有的 run 文件夹，确定下一个编号
    existing_runs = []
    if runs_dir.exists():
        for item in runs_dir.iterdir():
            if item.is_dir() and item.name.startswith("run") and item.name[3:].isdigit():
                try:
                    run_num = int(item.name[3:])
                    existing_runs.append(run_num)
                except ValueError:
                    pass
    
    # 确定下一个可用的 run 编号
    if existing_runs:
        next_run_num = max(existing_runs) + 1
    else:
        next_run_num = 1
    
    run_name = f"run{next_run_num}"
    print(f"本次训练结果将保存到: runs/{run_name}")
    
    # 开始训练
    print("开始训练...")
    results = model.train(
        data=str(data_yaml),           # 数据集配置文件路径
        epochs=epochs,                  # 训练轮数
        imgsz=640,                      # 输入图像尺寸
        batch=batch_size,               # 批次大小（根据设备自动调整）
        device=device,                  # 自动检测设备
        project=str(project_root / "runs" / run_name),  # 训练结果保存到 run1/train/
        name="train",                   # 训练结果文件夹名称
        exist_ok=False,                 # 如果目录已存在则报错（不应该发生）
        pretrained=True,                # 使用预训练权重
        optimizer='auto',               # 优化器：'SGD', 'Adam', 'AdamW', 'RMSProp'
        verbose=True,                   # 显示详细输出
        seed=0,                         # 随机种子
        deterministic=True,             # 确保可复现性
        single_cls=False,               # 多类别检测
        rect=False,                     # 矩形训练
        cos_lr=False,                   # 余弦学习率调度
        close_mosaic=10,                # 最后N个epoch关闭mosaic增强
        resume=False,                   # 是否从上次中断的地方继续训练
        amp=True,                       # 自动混合精度训练
        fraction=1.0,                   # 使用数据集的百分比
        profile=False,                  # 性能分析
        freeze=None,                    # 冻结前N层（None表示不冻结）
        lr0=0.01,                       # 初始学习率
        lrf=0.01,                       # 最终学习率 (lr0 * lrf)
        momentum=0.937,                 # SGD动量
        weight_decay=0.0005,            # 权重衰减
        warmup_epochs=3.0,              # 预热轮数
        warmup_momentum=0.8,            # 预热动量
        warmup_bias_lr=0.1,             # 预热偏置学习率
        box=7.5,                        # 边界框损失权重
        cls=0.5,                        # 分类损失权重
        dfl=1.5,                        # DFL损失权重
        pose=12.0,                      # 姿态损失权重（用于姿态估计，此处不使用）
        kobj=1.0,                       # 关键点对象损失权重
        label_smoothing=0.0,            # 标签平滑
        nbs=64,                         # 标准批次大小
        overlap_mask=True,              # 训练时掩码重叠（用于分割任务）
        mask_ratio=4,                   # 掩码下采样比率
        dropout=0.0,                    # Dropout（正则化）
        val=True,                       # 训练期间进行验证
        plots=True,                     # 生成训练图表（包含混淆矩阵、PR曲线等评估指标）
        save=True,                     # 保存模型检查点（best.pt 和 last.pt），默认True
        save_period=-1,                 # 不保存中间检查点（只保存best和last），-1表示禁用
        # NMS和置信度阈值（用于验证阶段）
        iou=0.7,                        # IoU阈值，用于NMS（非极大值抑制），范围0-1，默认0.7
        conf=0.001,                     # 置信度阈值，用于验证时过滤低置信度检测框，范围0-1，默认0.001
        # 数据增强参数（针对小数据集优化）
        hsv_h=0.015,                    # 色调增强（Hue），范围0-0.5，增加颜色多样性
        hsv_s=0.7,                      # 饱和度增强（Saturation），范围0-1，增强颜色鲜艳度
        hsv_v=0.4,                      # 明度/亮度增强（Value），范围0-1，增强亮度变化
        degrees=10.0,                   # 旋转角度（±degrees），范围0-180，增加旋转鲁棒性
        translate=0.1,                  # 平移比例，范围0-1，增加位置变化
        scale=0.5,                      # 缩放比例，范围0-1，增加尺度变化
        shear=0.0,                      # 剪切角度，范围0-180，增加几何变形
        perspective=0.0,                # 透视变换，范围0-0.001，增加透视变化
        flipud=0.0,                     # 上下翻转概率，范围0-1
        fliplr=0.5,                     # 左右翻转概率，范围0-1，增加镜像变化
        mosaic=1.0,                     # Mosaic增强概率，范围0-1，将4张图片拼接成1张
        mixup=0.1,                      # MixUp增强概率，范围0-1，混合两张图片和标签
        copy_paste=0.0,                 # Copy-Paste增强概率，范围0-1，复制粘贴目标
    )
    
    print("\n训练完成！")
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    last_model_path = results.save_dir / 'weights' / 'last.pt'
    
    # 检查模型文件是否存在
    if not best_model_path.exists():
        if last_model_path.exists():
            print(f"警告: best.pt 不存在，使用 last.pt 进行验证")
            print(f"  可能原因: 训练轮数过少或训练被中断")
            best_model_path = last_model_path
        else:
            raise FileNotFoundError(
                f"模型文件不存在！\n"
                f"  尝试查找 best.pt: {best_model_path}\n"
                f"  尝试查找 last.pt: {last_model_path}\n"
                f"  可能原因: 训练未完成或训练过程中出现错误"
            )
    
    # 输出训练结果文件夹结构说明
    print("\n" + "=" * 70)
    print("训练结果文件夹结构说明")
    print("=" * 70)
    print(f"训练结果保存在: runs/{run_name}")
    print("\n文件夹结构:")
    print(f"  {run_name}/")
    print(f"    ├─ train/                      # 训练结果目录")
    print(f"    │   ├─ weights/                # 模型权重文件")
    print(f"    │   │   ├─ best.pt             # 最佳模型（验证集上表现最好，推荐使用）")
    print(f"    │   │   └─ last.pt             # 最后一个epoch的模型")
    print(f"    │   ├─ args.yaml               # 训练参数配置")
    print(f"    │   ├─ results.csv             # 训练指标CSV文件（每个epoch的loss和metrics）")
    print(f"    │   ├─ results.png             # 训练曲线图（loss和metrics变化）")
    print(f"    │   ├─ confusion_matrix.png    # 混淆矩阵（训练时验证集）")
    print(f"    │   ├─ confusion_matrix_normalized.png  # 归一化混淆矩阵")
    print(f"    │   ├─ BoxPR_curve.png        # PR曲线（Precision-Recall）")
    print(f"    │   ├─ BoxF1_curve.png        # F1曲线")
    print(f"    │   ├─ BoxP_curve.png         # Precision曲线")
    print(f"    │   └─ BoxR_curve.png         # Recall曲线")
    print(f"    └─ val/                        # 验证结果目录")
    print(f"        ├─ validation_summary.json # 验证结果摘要（JSON格式）")
    print(f"        ├─ confusion_matrix.png    # 验证集混淆矩阵")
    print(f"        ├─ confusion_matrix_normalized.png  # 归一化混淆矩阵")
    print(f"        ├─ BoxPR_curve.png        # PR曲线")
    print(f"        ├─ BoxF1_curve.png        # F1曲线")
    print(f"        ├─ BoxP_curve.png         # Precision曲线")
    print(f"        ├─ BoxR_curve.png         # Recall曲线")
    print(f"        └─ validation_visualizations/  # 验证集可视化对比图片")
    print(f"            └─ *.jpg               # 预测vs真实标签对比图（绿色框=真实，红色框=预测）")
    print("=" * 70)
    
    # 在 valid 数据集上进行验证
    best_model = YOLO(str(best_model_path))
    
    # 对验证集进行验证（会自动使用 data.yaml 中的 val 路径）
    # 验证结果保存到 run1/val/ 目录
    print("\n正在验证模型...")
    val_results = best_model.val(
        data=str(data_yaml),  # 使用 data.yaml 配置，会自动使用 val 路径
        imgsz=640,
        batch=batch_size,
        device=device,
        project=str(project_root / "runs" / run_name),  # 验证结果保存到 run1/val/
        name="val",     # 验证结果文件夹名称
        verbose=False,  # 关闭详细输出，避免重复
        plots=True,     # 生成验证图表（包含混淆矩阵、PR曲线等评估指标）
        save=False,     # 不保存验证批次图片（精简输出）
        iou=0.7,        # IoU阈值，用于NMS（非极大值抑制），范围0-1，默认0.7
        conf=0.001,     # 置信度阈值，用于过滤低置信度检测框，范围0-1，默认0.001
    )
    
    # 验证结果目录
    val_dir = project_root / "runs" / run_name / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出验证结果摘要
    print("\n" + "=" * 70)
    print("验证结果摘要")
    print("=" * 70)
    
    # 计算 F1-score
    precision = val_results.box.mp
    recall = val_results.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 整体指标
    print(f"整体性能: mAP50={val_results.box.map50:.4f}, mAP50-95={val_results.box.map:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
    
    # 每个类别的详细指标（简化输出）
    if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > 0:
        maps = val_results.box.maps
        map50s = None
        if hasattr(val_results.box, 'map50s') and len(val_results.box.map50s) > 0:
            map50s = val_results.box.map50s
        precisions = None
        recalls = None
        if hasattr(val_results.box, 'p') and len(val_results.box.p) > 0:
            precisions = val_results.box.p
        if hasattr(val_results.box, 'r') and len(val_results.box.r) > 0:
            recalls = val_results.box.r
        
        print("\n各类别指标:")
        for i, class_name in enumerate(class_names):
            if i < len(maps):
                metrics_str = f"  {class_name}: mAP50-95={maps[i]:.4f}"
                if map50s is not None and i < len(map50s):
                    metrics_str += f", mAP50={map50s[i]:.4f}"
                if precisions is not None and i < len(precisions):
                    metrics_str += f", P={precisions[i]:.4f}"
                if recalls is not None and i < len(recalls):
                    metrics_str += f", R={recalls[i]:.4f}"
                    if precisions is not None and i < len(precisions):
                        f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0.0
                        metrics_str += f", F1={f1:.4f}"
                print(metrics_str)
    
    # 保存验证结果到 JSON 文件
    val_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(best_model_path),
        "dataset": "valid (验证集)",
        "overall_metrics": {
            "map50": float(val_results.box.map50),
            "map50_95": float(val_results.box.map),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        },
        "per_class_metrics": {}
    }
    
    # 添加每个类别的指标
    if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > 0:
        maps = val_results.box.maps
        map50s = val_results.box.map50s if hasattr(val_results.box, 'map50s') and len(val_results.box.map50s) > 0 else None
        precisions = val_results.box.p if hasattr(val_results.box, 'p') and len(val_results.box.p) > 0 else None
        recalls = val_results.box.r if hasattr(val_results.box, 'r') and len(val_results.box.r) > 0 else None
        
        for i, class_name in enumerate(class_names):
            if i < len(maps):
                class_metrics = {
                    "map50_95": float(maps[i])
                }
                if map50s is not None and i < len(map50s):
                    class_metrics["map50"] = float(map50s[i])
                if precisions is not None and i < len(precisions):
                    class_metrics["precision"] = float(precisions[i])
                if recalls is not None and i < len(recalls):
                    class_metrics["recall"] = float(recalls[i])
                    if precisions is not None and i < len(precisions):
                        prec_val = precisions[i]
                        recall_val = recalls[i]
                        f1 = 2 * (prec_val * recall_val) / (prec_val + recall_val) if (prec_val + recall_val) > 0 else 0.0
                        class_metrics["f1_score"] = float(f1)
                
                val_summary["per_class_metrics"][class_name] = class_metrics
    
    # 保存到 JSON 文件（保存到验证结果目录）
    summary_file = val_dir / "validation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(val_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n验证结果摘要已保存: {summary_file}")
    
    # 生成验证集预测和真实标签对比可视化图片（精简：只生成少量示例）
    print("\n生成验证集可视化对比图片（示例）...")
    
    # 确定验证集的图片和标签目录（基于 data.yaml 的相对路径）
    dataset_dir = data_yaml.parent  # dataset 目录
    val_images_path_str = data_config.get('val', 'valid/images')
    val_images_dir = (dataset_dir / val_images_path_str).resolve()
    val_labels_dir = val_images_dir.parent / "labels"  # labels 通常在 images 的兄弟目录
    
    # 如果路径不存在，尝试其他可能的路径
    if not val_images_dir.exists():
        # 尝试直接从 dataset 目录查找
        val_images_dir = dataset_dir / "valid" / "images"
        val_labels_dir = dataset_dir / "valid" / "labels"
    
    if val_images_dir.exists() and val_labels_dir.exists():
        # 创建可视化输出目录（保存到验证结果目录）
        vis_output_dir = val_dir / "validation_visualizations"
        
        visualize_validation_results(
            model=best_model,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            class_names=class_names,
            output_dir=vis_output_dir,
            device=device,
            conf_threshold=0.25,  # 置信度阈值
            max_images=10,  # 精简：只生成10张示例图片
        )
        print(f"可视化图片已保存: {vis_output_dir} (共10张示例)")
    else:
        print(f"警告: 无法找到验证集目录 ({val_images_dir})")
    
    return results, val_results


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练 YOLO11n 模型进行缺陷检测')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数（默认: 100）')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式（1轮训练）')
    
    args = parser.parse_args()
    
    # 如果使用快速测试模式，设置为1轮
    epochs = 1 if args.quick_test else args.epochs
    
    try:
        train_results, val_results = train_model(epochs=epochs)
        print("\n训练和验证全部完成！")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

