"""
在测试集上评估训练好的 YOLO 模型

基于 dataset/test/ 测试集对训练好的模型进行评估，
生成详细的测试结果报告和可视化对比图片。
"""

from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
import json
from datetime import datetime
import cv2
import numpy as np
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
        label_y = max(th, y1_i)
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
            label_y1 = y2_i
            label_y2 = y2_i + th
            text_y = y2_i + th - baseline
        else:
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


def visualize_test_results(
    model: YOLO,
    test_images_dir: Path,
    test_labels_dir: Path,
    class_names: List[str],
    output_dir: Path,
    device: str = "cpu",
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.25,
):
    """
    生成测试集的预测和真实标签对比可视化图片
    
    Args:
        model: YOLO模型
        test_images_dir: 测试集图片目录
        test_labels_dir: 测试集标签目录
        class_names: 类别名称列表
        output_dir: 输出目录
        device: 设备 ('cpu' 或 'cuda')
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值，用于NMS
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    print(f"\n开始生成测试集可视化图片（共 {len(image_files)} 张）...")
    
    for img_idx, img_path in enumerate(image_files, 1):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 读取真实标签
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = read_yolo_label(label_path, img_width, img_height)
        
        # 进行预测
        results = model.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
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
        
        print(f"  已处理 {img_idx}/{len(image_files)} 张图片: {img_path.name}")
    
    print(f"\n可视化图片已保存到: {output_dir}")
    print("  - 绿色框: 真实标签 (Ground Truth)")
    print("  - 红色框: 预测结果 (Prediction)")


def test_model(model_path: str = None, model_type: str = "best", conf_threshold: float = 0.15, iou_threshold: float = 0.25):
    """
    在测试集上评估模型
    
    Args:
        model_path: 模型文件路径，如果为None则自动查找最新的模型
        model_type: 模型类型，'best' 或 'last'（仅在model_path为None时有效）
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值，用于NMS
    """
    
    # 项目根目录
    project_root = Path(__file__).parent
    
    # 数据集配置文件路径
    data_yaml = project_root / "dataset" / "data.yaml"
    
    # 检查文件是否存在
    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 读取数据集配置以获取类别名称
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [])
    
    # 自动检测设备（GPU或CPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16 if device == 'cuda' else 4
    
    # 确定模型路径
    if model_path is None:
        # 自动查找最新的模型（best.pt 或 last.pt）
        runs_dir = project_root / "runs"
        model_filename = f"{model_type}.pt"  # best.pt 或 last.pt
        found_model_path = None
        
        if runs_dir.exists():
            # 查找所有run文件夹
            run_dirs = []
            for item in runs_dir.iterdir():
                if item.is_dir() and item.name.startswith("run") and item.name[3:].isdigit():
                    try:
                        run_num = int(item.name[3:])
                        model_file = item / "train" / "weights" / model_filename
                        if model_file.exists():
                            run_dirs.append((run_num, model_file))
                    except ValueError:
                        pass
            
            if run_dirs:
                # 使用最新的run文件夹中的模型
                run_dirs.sort(key=lambda x: x[0], reverse=True)
                found_model_path = run_dirs[0][1]
                print(f"自动找到模型 ({model_type}.pt): {found_model_path}")
            else:
                # 尝试使用model目录下的模型
                found_model_path = project_root / "model" / model_filename
                if not found_model_path.exists():
                    raise FileNotFoundError(f"未找到训练好的模型 ({model_filename})，请先训练模型或指定模型路径")
        else:
            # 尝试使用model目录下的模型
            found_model_path = project_root / "model" / model_filename
            if not found_model_path.exists():
                raise FileNotFoundError(f"未找到训练好的模型 ({model_filename})，请先训练模型或指定模型路径")
        
        best_model_path = found_model_path
    else:
        best_model_path = Path(model_path)
        if not best_model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {best_model_path}")
    
    print("=" * 70)
    print("测试集模型评估")
    print("=" * 70)
    print(f"模型路径: {best_model_path}")
    print(f"数据集配置: {data_yaml}")
    print(f"测试设备: {device} ({'GPU' if device == 'cuda' else 'CPU'})")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IoU阈值: {iou_threshold}")
    print("-" * 70)
    
    # 加载模型
    print("正在加载模型...")
    model = YOLO(str(best_model_path))
    
    # 创建测试结果输出目录（根目录下的单独文件夹）
    test_output_dir = project_root / "test_results"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 在测试集上进行评估
    # 注意：需要临时修改data.yaml中的val路径为test路径，或者直接指定测试集路径
    print("\n正在测试集上评估模型...")
    
    # 创建临时的测试集配置文件
    test_data_yaml = test_output_dir / "test_data.yaml"
    test_data_config = data_config.copy()
    
    # 确定测试集路径（使用绝对路径，相对于项目根目录）
    dataset_dir = data_yaml.parent  # dataset 目录
    test_images_dir = dataset_dir / "test" / "images"
    
    # 将val路径替换为test路径（使用绝对路径）
    test_data_config['val'] = str(test_images_dir.resolve())
    
    # 同时更新 train 路径为绝对路径（如果需要）
    if 'train' in test_data_config:
        train_images_dir = dataset_dir / "train" / "images"
        test_data_config['train'] = str(train_images_dir.resolve())
    
    # 保存临时配置文件
    with open(test_data_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(test_data_config, f, allow_unicode=True, default_flow_style=False)
    
    test_results = model.val(
        data=str(test_data_yaml),  # 使用修改后的配置文件（val路径指向test）
        imgsz=640,
        batch=batch_size,
        device=device,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        plots=True,  # 生成测试图表（包含混淆矩阵、PR曲线等评估指标）
        save=False,  # 不保存测试批次图片（精简输出）
        project=str(test_output_dir),  # 测试结果保存目录（直接使用已创建的test_results目录）
        name="",  # 空名称，结果直接保存在project目录下
    )
    
    # 输出测试结果摘要
    print("\n" + "=" * 70)
    print("测试结果摘要")
    print("=" * 70)
    
    # 计算 F1-score
    precision = test_results.box.mp
    recall = test_results.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 整体指标
    print(f"整体性能: mAP50={test_results.box.map50:.4f}, mAP50-95={test_results.box.map:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
    
    # 每个类别的详细指标（简化输出）
    if hasattr(test_results.box, 'maps') and len(test_results.box.maps) > 0:
        maps = test_results.box.maps
        map50s = None
        if hasattr(test_results.box, 'map50s') and len(test_results.box.map50s) > 0:
            map50s = test_results.box.map50s
        precisions = None
        recalls = None
        if hasattr(test_results.box, 'p') and len(test_results.box.p) > 0:
            precisions = test_results.box.p
        if hasattr(test_results.box, 'r') and len(test_results.box.r) > 0:
            recalls = test_results.box.r
        
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
    
    # 保存测试结果到 JSON 文件
    test_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(best_model_path),
        "dataset": "test (测试集)",
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "overall_metrics": {
            "map50": float(test_results.box.map50),
            "map50_95": float(test_results.box.map),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        },
        "per_class_metrics": {}
    }
    
    # 添加每个类别的指标
    if hasattr(test_results.box, 'maps') and len(test_results.box.maps) > 0:
        maps = test_results.box.maps
        map50s = test_results.box.map50s if hasattr(test_results.box, 'map50s') and len(test_results.box.map50s) > 0 else None
        precisions = test_results.box.p if hasattr(test_results.box, 'p') and len(test_results.box.p) > 0 else None
        recalls = test_results.box.r if hasattr(test_results.box, 'r') and len(test_results.box.r) > 0 else None
        
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
                
                test_summary["per_class_metrics"][class_name] = class_metrics
    
    # 保存到 JSON 文件
    summary_file = test_output_dir / "test_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果摘要已保存: {summary_file}")
    print("=" * 70)
    
    # 生成测试集预测和真实标签对比可视化图片
    print("\n生成测试集可视化对比图片...")
    
    # 确定测试集的图片和标签目录
    dataset_dir = data_yaml.parent  # dataset 目录
    test_images_dir = dataset_dir / "test" / "images"
    test_labels_dir = dataset_dir / "test" / "labels"
    
    if test_images_dir.exists() and test_labels_dir.exists():
        # 创建可视化输出目录
        vis_output_dir = test_output_dir / "test_visualizations"
        
        visualize_test_results(
            model=model,
            test_images_dir=test_images_dir,
            test_labels_dir=test_labels_dir,
            class_names=class_names,
            output_dir=vis_output_dir,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        print(f"可视化图片已保存: {vis_output_dir}")
    else:
        print(f"警告: 无法找到测试集目录")
        print(f"  尝试的图片目录: {test_images_dir}")
        print(f"  尝试的标签目录: {test_labels_dir}")
    
    print("\n" + "=" * 70)
    print("测试结果文件夹结构说明")
    print("=" * 70)
    print(f"测试结果保存在: {test_output_dir}")
    print("\n文件夹结构:")
    print(f"  test_results/")
    print(f"    ├─ test_summary.json          # 测试结果摘要（JSON格式）")
    print(f"    ├─ confusion_matrix.png      # 测试集混淆矩阵")
    print(f"    ├─ confusion_matrix_normalized.png  # 归一化混淆矩阵")
    print(f"    ├─ BoxPR_curve.png          # PR曲线")
    print(f"    ├─ BoxF1_curve.png         # F1曲线")
    print(f"    ├─ BoxP_curve.png          # Precision曲线")
    print(f"    ├─ BoxR_curve.png          # Recall曲线")
    print(f"    └─ test_visualizations/     # 测试集可视化对比图片")
    print(f"        └─ *.jpg                # 预测vs真实标签对比图（绿色框=真实，红色框=预测）")
    print("=" * 70)
    
    return test_results, test_summary


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='在测试集上评估训练好的 YOLO 模型')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（默认：自动查找最新的模型）')
    parser.add_argument('--model-type', type=str, default='best', choices=['best', 'last'],
                       help='模型类型：best 或 last（仅在未指定--model时有效，默认: best）')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='置信度阈值（默认: 0.15）')
    parser.add_argument('--iou', type=float, default=0.25,
                       help='IoU阈值，用于NMS（默认: 0.25）')
    
    args = parser.parse_args()
    
    try:
        test_results, test_summary = test_model(
            model_path=args.model,
            model_type=args.model_type,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print("\n测试评估全部完成！")
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

