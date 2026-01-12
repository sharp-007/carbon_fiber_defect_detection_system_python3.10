import os
import warnings
import logging

# 彻底解决 missing ScriptRunContext 警告
# 这个警告在 streamlit-webrtc 的后台线程中是正常的，可以安全忽略
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")
# 忽略 use_container_width 废弃警告（功能正常，只是参数名即将变更）
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*will be removed after.*")
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state").setLevel(logging.ERROR)

# 在导入 OpenCV 之前设置环境变量，禁用 GUI 功能（适用于 headless 环境）
# 注意：这些设置不会影响 cv2.VideoCapture() 和视频文件读取功能
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
# 禁用 OpenCV 的 GUI 后端，强制使用 headless 模式
# 这些设置只影响 cv2.imshow() 等GUI显示功能，不影响视频文件读取和处理
# 在 Streamlit Cloud 等 headless 环境中，这些设置是必要的
if 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ''

import tempfile
import time
import zipfile
import io
import copy
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib
import altair as alt
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# OpenCV 导入（优先使用 headless 版本）
try:
    import cv2
    # 验证 OpenCV 版本并设置 headless 模式
    # 注意：setNumThreads(0) 禁用多线程，可能影响视频处理性能，但不影响功能
    # 如果视频处理速度慢，可以尝试设置为1或更高值
    # 但在某些系统上，多线程可能导致问题，所以保持为0更安全
    cv2.setNumThreads(0)  # 禁用多线程以避免某些系统库问题
except (ImportError, OSError) as e:
    error_msg = str(e)
    if "libGL.so.1" in error_msg or "libGL.so" in error_msg:
        st.error("""
        ❌ **OpenCV 系统库缺失错误**
        
        检测到缺少系统库 `libGL.so.1`，这通常发生在 Linux 环境（如 Streamlit Cloud）中。
        
        **解决方案：**
        1. 确保项目根目录存在 `packages.txt` 文件，内容包含：
           ```
           libgl1-mesa-glx
           libglib2.0-0
           ```
        2. 提交并推送到 GitHub 后，Streamlit Cloud 会自动安装这些系统依赖。
        3. 如果问题仍然存在，请检查 Streamlit Cloud 的部署日志。
        """)
    else:
        st.error(f"""
        ❌ **OpenCV 导入错误**
        
        错误信息：{error_msg}
        
        **可能的原因：**
        1. 未安装 OpenCV：请在 requirements.txt 中确保包含 `opencv-python-headless>=4.5.0`
        2. 系统库缺失：在 Linux 环境中可能需要安装系统库（见上方 libGL.so.1 错误处理）
        3. Python 版本不兼容：请确保使用 Python 3.10 或更高版本
        """)
    st.stop()

try:
    from ultralytics import YOLO
    import torch
except Exception:  # pragma: no cover - ultralytics may not be installed in lint env
    YOLO = None  # type: ignore
    torch = None  # type: ignore

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtc_streamer = None  # type: ignore
    WebRtcMode = None  # type: ignore
    av = None  # type: ignore

# 全局锁和数据容器（用于线程间共享数据）
# 参考: https://github.com/whitphx/streamlit-webrtc#pull-values-from-the-callback
camera_lock = threading.Lock()
camera_result_container = {
    "objects": [],           # 当前帧检测到的对象列表
    "current_defect_count": 0, # 当前帧检测到的缺陷数量
    "frame_count": 0,        # 处理的帧数
    "detection_count": 0,    # 检测到缺陷的次数
    "last_detect_time": 0.0, # 上次检测时间
    "start_time": None,      # 开始时间
    "annotated_frame": None, # 带标注的帧（用于显示）
    "frames": [],            # 保存的帧图片列表（用于下载）
    "records": [],           # 保存的 DataFrame 记录列表（用于下载）
}

# 导入 TURN 服务器配置
try:
    from turn import get_ice_servers
    TURN_AVAILABLE = True
except ImportError:
    TURN_AVAILABLE = False
    def get_ice_servers():
        """回退到免费的 Google STUN 服务器"""
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


def load_model(model_file: Path, device: str = "cpu"):
    """加载 YOLO 模型，使用 Streamlit 缓存避免重复加载。"""
    @st.cache_resource(show_spinner=False)
    def _load(path_str: str, device_str: str):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，请先在环境中执行: pip install ultralytics")
        model = YOLO(path_str)
        return model
    return _load(str(model_file), device)


def pil_to_ndarray(img: Image.Image) -> np.ndarray:
    """PIL Image -> numpy ndarray (BGR for OpenCV 绘制)."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def ndarray_to_pil(bgr: np.ndarray) -> Image.Image:
    """BGR ndarray -> PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_to_16_9(frame_bgr: np.ndarray) -> np.ndarray:
    """
    将图像调整为16:9比例，不符合的部分用黑色填充（letterboxing/pillarboxing）。
    
    Args:
        frame_bgr: BGR格式的numpy数组
        
    Returns:
        调整为16:9比例的BGR图像
    """
    h, w = frame_bgr.shape[:2]
    target_aspect = 16.0 / 9.0
    current_aspect = w / h
    
    if abs(current_aspect - target_aspect) < 0.001:  # 已经是16:9，直接返回
        return frame_bgr
    
    # 计算目标尺寸（以较大边为准）
    if current_aspect > target_aspect:
        # 当前更宽，以宽度为准
        new_w = w
        new_h = int(w / target_aspect)
        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top
        pad_left = 0
        pad_right = 0
    else:
        # 当前更高，以高度为准
        new_h = h
        new_w = int(h * target_aspect)
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        pad_top = 0
        pad_bottom = 0
    
    # 添加黑色边框
    if len(frame_bgr.shape) == 3:
        # 彩色图像
        padded = cv2.copyMakeBorder(
            frame_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # 黑色（BGR）
        )
    else:
        # 灰度图像
        padded = cv2.copyMakeBorder(
            frame_bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  # 黑色
        )
    
    return padded


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    cls_names: List[str],
    scores: List[float],
    line_thickness: int = 2,
) -> np.ndarray:
    """在 BGR 图像上绘制检测框和标签。"""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for (x1, y1, x2, y2), name, score in zip(boxes, cls_names, scores):
        # 坐标转为 int 且限制在图像范围内
        x1_i = int(max(0, min(w - 1, x1)))
        y1_i = int(max(0, min(h - 1, y1)))
        x2_i = int(max(0, min(w - 1, x2)))
        y2_i = int(max(0, min(h - 1, y2)))

        color = (0, 0, 255)  # BGR: 红色
        cv2.rectangle(out, (x1_i, y1_i), (x2_i, y2_i), color, line_thickness)

        label = f"{name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        th = th + baseline
        
        # 确保标签不超出图片边界
        if y1_i - th < 0:
            # 如果上方空间不足，标签放在框内
            label_y = y1_i + th
            cv2.rectangle(out, (x1_i, y1_i), (x1_i + tw, label_y), color, -1)
            cv2.putText(
                out,
                label,
                (x1_i, y1_i + th - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # 标签放在框的上方
            cv2.rectangle(out, (x1_i, y1_i - th), (x1_i + tw, y1_i), color, -1)
            cv2.putText(
                out,
                label,
                (x1_i, y1_i - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out


def run_inference_image(
    model, 
    img: Image.Image, 
    conf: float, 
    iou: float = 0.45,
    device: str = "cpu"
) -> Tuple[Image.Image, pd.DataFrame]:
    """对单张图片进行检测，返回绘制好框的图片以及结果表。"""
    # 在原始图片上进行检测（不调整16:9，避免检测到黑边）
    frame_bgr = pil_to_ndarray(img)

    results = model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    if not results:
        # 即使没有检测结果，也要调整为16:9用于显示
        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
        return ndarray_to_pil(frame_bgr_16_9), pd.DataFrame()

    r = results[0]
    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") and len(r.boxes.xyxy) > 0 else np.empty((0, 4))
    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") and len(r.boxes.conf) > 0 else np.array([])
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") and len(r.boxes.cls) > 0 else np.array([], dtype=int)

    names = r.names if hasattr(r, "names") else {}
    cls_names = [names.get(int(c), f"class_{int(c)}") for c in cls_ids] if len(cls_ids) > 0 else []

    if len(boxes_xyxy) > 0:
        # 在原始图片上绘制检测框
        drawn = draw_boxes(
            frame_bgr,
            boxes_xyxy.tolist(),
            cls_names,
            scores.tolist(),
        )

        # 将绘制了检测框的图片调整为16:9用于显示
        drawn_16_9 = resize_to_16_9(drawn)

        widths = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).astype(int)
        heights = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).astype(int)
        df = pd.DataFrame({
            "缺陷类别": cls_names,
            "置信度": scores,
            "左上角X": boxes_xyxy[:, 0].astype(int),
            "左上角Y": boxes_xyxy[:, 1].astype(int),
            "右下角X": boxes_xyxy[:, 2].astype(int),
            "右下角Y": boxes_xyxy[:, 3].astype(int),
            "宽度": widths,
            "高度": heights,
            "面积": (widths * heights).astype(int),
        })
        
        return ndarray_to_pil(drawn_16_9), df
    else:
        # 没有检测到缺陷，也要调整为16:9用于显示
        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
        return ndarray_to_pil(frame_bgr_16_9), pd.DataFrame()


def convert_video_to_16_9(video_bytes: bytes, video_filename: Optional[str] = None) -> Optional[bytes]:
    """
    将视频转换为16:9格式，用于显示。
    
    Args:
        video_bytes: 原始视频字节数据
        video_filename: 视频文件名（可选），用于确定正确的文件扩展名
        
    Returns:
        16:9格式的视频字节数据，失败则返回None
    """
    tmp_path = None
    output_video_path = None
    cap = None
    
    try:
        # 检查并导入必要的库
        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401
        except ImportError as e:
            # 如果imageio不可用，记录错误但继续尝试其他方法
            import logging
            logging.warning(f"imageio或imageio-ffmpeg未安装: {e}")
            return None
        
        # 从文件名中提取扩展名，支持多种视频格式
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        if video_filename:
            file_ext = Path(video_filename).suffix.lower()
            if file_ext in supported_extensions:
                suffix = file_ext
            else:
                suffix = '.mp4'  # 默认使用 .mp4
        else:
            suffix = '.mp4'  # 如果没有文件名，默认使用 .mp4
        
        # 将上传的视频写入临时文件，使用正确的扩展名
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 检查视频尺寸是否有效
        if video_width <= 0 or video_height <= 0:
            return None
        
        # 计算16:9尺寸
        sample_frame_bgr = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        sample_frame_16_9 = resize_to_16_9(sample_frame_bgr)
        video_height_16_9, video_width_16_9 = sample_frame_16_9.shape[:2]
        
        # 生成16:9视频
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        
        frames_16_9 = []
        frame_count = 0
        max_frames = 10000  # 限制最大帧数，避免内存问题
        
        while frame_count < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_16_9 = resize_to_16_9(frame_bgr)
            frames_16_9.append(cv2.cvtColor(frame_16_9, cv2.COLOR_BGR2RGB))
            frame_count += 1
        
        if cap:
            cap.release()
            cap = None
        
        if len(frames_16_9) > 0:
            # 确保fps有效
            fps = original_fps if original_fps > 0 else 30.0
            
            # 确保视频尺寸是偶数（yuv420p要求）
            h, w = frames_16_9[0].shape[:2]
            if w % 2 != 0:
                w = w - 1
            if h % 2 != 0:
                h = h - 1
            if w != frames_16_9[0].shape[1] or h != frames_16_9[0].shape[0]:
                frames_16_9 = [cv2.resize(frame, (w, h)) for frame in frames_16_9]
            
            # 使用imageio创建视频，确保浏览器兼容性
            imageio.mimsave(
                output_video_path,
                frames_16_9,
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'  # yuv420p要求宽度和高度都是偶数
            )
            
            # 等待文件写入完成（在Streamlit Cloud上可能需要更长时间）
            time.sleep(1.0)  # 增加等待时间，确保文件完全写入
            
            # 读取视频文件并验证
            if Path(output_video_path).exists():
                file_size = Path(output_video_path).stat().st_size
                if file_size > 0:
                    with open(output_video_path, 'rb') as f:
                        output_video_bytes = f.read()
                    
                    # 验证视频字节是否有效（至少应该有一些数据）
                    if len(output_video_bytes) > 1000:  # 至少1KB
                        return output_video_bytes
        
        return None
        
    except Exception as e:
        import logging
        logging.error(f"convert_video_to_16_9错误: {e}")
        return None
    finally:
        # 清理资源
        if cap:
            try:
                cap.release()
            except Exception:
                pass
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
        if output_video_path and Path(output_video_path).exists():
            try:
                Path(output_video_path).unlink()
            except Exception:
                pass


def run_inference_video(
    model,
    video_bytes: bytes,
    conf: float,
    iou: float = 0.45,
    frame_step: int = 5,
    device: str = "cpu",
    process_all_frames: bool = True,
    stop_check_callback: Optional[Callable[[], bool]] = None,
    video_filename: Optional[str] = None,
) -> Tuple[List[Image.Image], pd.DataFrame, dict, Optional[bytes]]:
    """
    对视频进行检测，参考 app1.py.py 的逻辑：
    - 使用 frame_step 控制抽帧步长（用于预览）；
    - 输出视频使用原始视频的帧率；
    - 如果 process_all_frames=True，所有帧都会被处理并生成完整视频；
    - 如果 process_all_frames=False，只处理抽帧的帧。
    - 支持多种视频格式：mp4, avi, mov, mkv, flv

    返回：关键帧预览列表、统计 DataFrame、视频信息、完整检测视频字节。
    """
    # 从文件名中提取扩展名，支持多种视频格式
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    if video_filename:
        file_ext = Path(video_filename).suffix.lower()
        if file_ext in supported_extensions:
            suffix = file_ext
        else:
            suffix = '.mp4'  # 默认使用 .mp4
    else:
        suffix = '.mp4'  # 如果没有文件名，默认使用 .mp4
    
    # 将上传的视频写入临时文件，使用正确的扩展名
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    # 打开视频做一次遍历：同时完成抽帧检测、统计和输出视频帧收集
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return [], pd.DataFrame(), {}, None

    # 获取视频信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / original_fps if original_fps > 0 else 0

    # 收集所有处理后的帧（用于生成视频）
    processed_frames: List[np.ndarray] = []  # BGR格式的numpy数组
    frames_out: List[Image.Image] = []  # 用于预览的抽帧
    all_records: List[pd.DataFrame] = []

    idx = 0

    # 使用当前时间作为视频开始的绝对时间基准（用于"绝对时间戳"和"日期"列）
    video_start_timestamp = datetime.now().timestamp()

    while True:
        if stop_check_callback is not None and stop_check_callback():
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 计算时间点（秒）
        timestamp = idx / original_fps if original_fps > 0 else 0
        time_str = format_seconds_to_hhmmss_mmm(timestamp)

        # 决定是否检测这一帧
        should_detect = True if process_all_frames else (idx % frame_step == 0)
        # 决定是否记录到统计表（只有抽帧的帧才记录）
        should_record = (idx % frame_step == 0)

        if should_detect:
            if stop_check_callback is not None and stop_check_callback():
                break

            # 在原始帧上进行检测（不调整16:9，避免检测到黑边）
            results = model.predict(
                source=frame_bgr,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )

            current_timestamp = video_start_timestamp + timestamp
            date_str = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d")

            if results and len(results) > 0:
                r = results[0]
                boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") and len(r.boxes.xyxy) > 0 else np.empty((0, 4))
                scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") and len(r.boxes.conf) > 0 else np.array([])
                cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") and len(r.boxes.cls) > 0 else np.array([], dtype=int)
                names = r.names if hasattr(r, "names") else {}
                cls_names = [names.get(int(c), f"class_{int(c)}") for c in cls_ids] if len(cls_ids) > 0 else []

                if len(boxes_xyxy) > 0:
                    # 在原始帧上绘制检测框
                    drawn = draw_boxes(
                        frame_bgr,
                        boxes_xyxy.tolist(),
                        cls_names,
                        scores.tolist(),
                    )

                    # 将绘制了检测框的帧调整为16:9用于显示和保存
                    drawn_16_9 = resize_to_16_9(drawn)

                    # 如果是抽帧预览，保存PIL图片
                    if idx % frame_step == 0:
                        frames_out.append(ndarray_to_pil(drawn_16_9))

                    # 保存处理后的帧（用于生成视频）
                    if process_all_frames:
                        processed_frames.append(drawn_16_9)

                    # 记录检测结果（使用原始帧的坐标）- 只有抽帧的帧才记录
                    if should_record:
                        widths = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).astype(int)
                        heights = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).astype(int)
                        df_frame = pd.DataFrame({
                            "日期": [date_str] * len(cls_names),
                            "检测序号": [(idx // frame_step) + 1] * len(cls_names),
                            "帧序号": [idx] * len(cls_names),
                            "时间点(秒)": [round(timestamp, 3)] * len(cls_names),
                            "绝对时间戳": [round(current_timestamp, 2)] * len(cls_names),
                            "时间点(HH:MM:SS.mmm)": [time_str] * len(cls_names),
                            "缺陷类别": cls_names,
                            "置信度": scores,
                            "左上角X": boxes_xyxy[:, 0].astype(int),
                            "左上角Y": boxes_xyxy[:, 1].astype(int),
                            "右下角X": boxes_xyxy[:, 2].astype(int),
                            "右下角Y": boxes_xyxy[:, 3].astype(int),
                            "宽度": widths,
                            "高度": heights,
                            "面积": (widths * heights).astype(int),
                        })
                        all_records.append(df_frame)
                else:
                    # 没有检测到缺陷，但仍需要记录该帧信息（缺陷相关字段为空值）
                    # 保存原始帧（不标注框）到frames_out（如果是抽帧预览）
                    if should_record:
                        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                        frames_out.append(ndarray_to_pil(frame_bgr_16_9))
                    
                    # 保存处理后的帧（用于生成视频）
                    if process_all_frames:
                        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                        processed_frames.append(frame_bgr_16_9)
                    
                    # 记录检测结果（缺陷相关字段为空值），确保图表能显示所有抽帧时间点
                    if should_record:
                        df_frame = pd.DataFrame({
                            "日期": [date_str],
                            "检测序号": [(idx // frame_step) + 1],
                            "帧序号": [idx],
                            "时间点(秒)": [round(timestamp, 3)],
                            "绝对时间戳": [round(current_timestamp, 2)],
                            "时间点(HH:MM:SS.mmm)": [time_str],
                            "缺陷类别": [None],
                            "置信度": [None],
                            "左上角X": [None],
                            "左上角Y": [None],
                            "右下角X": [None],
                            "右下角Y": [None],
                            "宽度": [None],
                            "高度": [None],
                            "面积": [None],
                        })
                        all_records.append(df_frame)
            else:
                # 检测失败，将原始帧调整为16:9保存
                if process_all_frames:
                    frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                    processed_frames.append(frame_bgr_16_9)
                
                # 如果是抽帧预览，也需要记录时间点（即使检测失败）
                if should_record:
                    df_frame = pd.DataFrame({
                        "日期": [date_str],
                        "检测序号": [(idx // frame_step) + 1],
                        "帧序号": [idx],
                        "时间点(秒)": [round(timestamp, 3)],
                        "绝对时间戳": [round(current_timestamp, 2)],
                        "时间点(HH:MM:SS.mmm)": [time_str],
                        "缺陷类别": [None],
                        "置信度": [None],
                        "左上角X": [None],
                        "左上角Y": [None],
                        "右下角X": [None],
                        "右下角Y": [None],
                        "宽度": [None],
                        "高度": [None],
                        "面积": [None],
                    })
                    all_records.append(df_frame)
        else:
            # 跳过检测的帧，将原始帧调整为16:9保存
            if process_all_frames:
                frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                processed_frames.append(frame_bgr_16_9)

        idx += 1

    cap.release()

    # 汇总结果表
    summary_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    
    # 更新视频尺寸信息（如果处理了帧，使用第一帧的尺寸）
    if len(processed_frames) > 0:
        video_height_16_9, video_width_16_9 = processed_frames[0].shape[:2]
    else:
        # 如果没有处理帧，计算16:9尺寸
        frame_bgr_16_9_sample = resize_to_16_9(np.zeros((video_height, video_width, 3), dtype=np.uint8))
        video_height_16_9, video_width_16_9 = frame_bgr_16_9_sample.shape[:2]
    
    # 使用原始帧率作为输出帧率
    video_info = {
        "fps": original_fps,
        "total_frames": total_frames,
        "duration": video_duration,
        "width": video_width_16_9 if len(processed_frames) > 0 else video_width,
        "height": video_height_16_9 if len(processed_frames) > 0 else video_height,
    }

    # 使用imageio创建视频文件（更可靠的方法）
    processed_video_bytes: Optional[bytes] = None
    if process_all_frames and len(processed_frames) > 0:
        try:
            import imageio
            # 检查是否安装了imageio-ffmpeg插件（imageio.mimsave需要它）
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError:
                raise ImportError("需要安装imageio-ffmpeg插件: pip install imageio-ffmpeg")

            output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            # 将BGR转换为RGB（imageio需要RGB格式）
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames]
            # 使用原始帧率以确保视频长度一致
            video_fps = original_fps if original_fps > 0 else 30.0
            
            # 确保视频尺寸是偶数（yuv420p要求）
            if len(rgb_frames) > 0:
                h, w = rgb_frames[0].shape[:2]
                if w % 2 != 0 or h % 2 != 0:
                    # 调整到偶数尺寸
                    new_w = w if w % 2 == 0 else w - 1
                    new_h = h if h % 2 == 0 else h - 1
                    rgb_frames = [cv2.resize(frame, (new_w, new_h)) for frame in rgb_frames]
            
            # 先使用imageio生成临时视频文件
            temp_video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            imageio.mimsave(
                temp_video_path,
                rgb_frames,
                fps=video_fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'  # 确保浏览器兼容性，yuv420p要求宽度和高度都是偶数
            )
            
            # 等待文件写入完成
            time.sleep(1.0)
            
            # 使用ffmpeg转换视频为H.264编码（确保浏览器兼容性）
            # 命令格式: ffmpeg -y -i input.mp4 -vcodec libx264 output.mp4
            try:
                import subprocess
                # 尝试使用imageio-ffmpeg提供的ffmpeg路径
                ffmpeg_path = None
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                except (ImportError, Exception):
                    # 如果imageio-ffmpeg不可用，尝试使用系统ffmpeg
                    ffmpeg_path = 'ffmpeg'
                
                # 使用ffmpeg转换视频为H.264编码
                # -y: 覆盖输出文件
                # -i: 输入文件
                # -vcodec libx264: 使用H.264编码（浏览器兼容）
                ffmpeg_cmd = [
                    ffmpeg_path,
                    '-y',  # 覆盖输出文件
                    '-i', temp_video_path,  # 输入文件
                    '-vcodec', 'libx264',  # 使用H.264编码
                    output_video_path  # 输出文件
                ]
                result = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0 and Path(output_video_path).exists():
                    file_size = Path(output_video_path).stat().st_size
                    if file_size > 0:
                        time.sleep(0.5)  # 等待文件完全写入
                        with open(output_video_path, 'rb') as f:
                            processed_video_bytes = f.read()
                        # 验证视频字节是否有效
                        if len(processed_video_bytes) < 100:
                            processed_video_bytes = None
                    else:
                        processed_video_bytes = None
                else:
                    # ffmpeg转换失败，使用imageio生成的原始文件
                    if Path(temp_video_path).exists() and Path(temp_video_path).stat().st_size > 0:
                        with open(temp_video_path, 'rb') as f:
                            processed_video_bytes = f.read()
                    else:
                        processed_video_bytes = None
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                # ffmpeg不可用或转换失败，使用imageio生成的原始文件
                if Path(temp_video_path).exists() and Path(temp_video_path).stat().st_size > 0:
                    with open(temp_video_path, 'rb') as f:
                        processed_video_bytes = f.read()
                else:
                    processed_video_bytes = None
            
            # 清理临时文件
            try:
                if Path(temp_video_path).exists():
                    Path(temp_video_path).unlink()
            except Exception:
                pass
            try:
                if Path(output_video_path).exists():
                    Path(output_video_path).unlink()
            except Exception:
                pass
        except ImportError:
            # imageio是必需依赖，如果不可用应该报错而不是fallback
            raise ImportError("需要安装imageio和imageio-ffmpeg: pip install imageio imageio-ffmpeg")
        except Exception:
            processed_video_bytes = None

    # 清理上传视频临时文件
    try:
        Path(tmp_path).unlink()
    except Exception:
        pass

    return frames_out, summary_df, video_info, processed_video_bytes


def sidebar_controls() -> Tuple[Optional[Path], str, str, float, float, int, float, float, int, Optional[int]]:
    """侧边栏 UI：模型、模式、设备和参数设置。"""
    st.sidebar.header("⚙️ 模型配置")
    
    # 默认模型路径
    default_model_path = Path("model/best.pt")
    use_default_model = st.sidebar.checkbox("使用默认模型 (best.pt)", value=True if default_model_path.exists() else False)
    
    model_path: Optional[Path] = None
    
    if use_default_model and default_model_path.exists():
        model_path = default_model_path
        st.sidebar.success(f"✅ 使用默认模型: {default_model_path}")
    else:
        model_file = st.sidebar.file_uploader(
            "上传 YOLO 模型文件 (.pt)",
            type=["pt"],
            help="上传训练好的模型权重文件，例如 best.pt",
        )
        if model_file is not None:
            # 将上传的模型文件保存到临时目录
            tmp_dir = Path(tempfile.gettempdir()) / "defect_detect_models"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_model_path = tmp_dir / model_file.name
            with open(tmp_model_path, "wb") as f:
                f.write(model_file.read())
            model_path = tmp_model_path

    st.sidebar.header("📊 检测设置")
    mode = st.sidebar.selectbox("检测模式", ["图片检测", "视频检测", "摄像头实时检测"])
    
    # 检查 CUDA 是否可用，动态设置设备选项
    if torch is not None and torch.cuda.is_available():
        device_options = ["cpu", "cuda"]
        default_device_index = 1  # 默认选择 cuda
    else:
        device_options = ["cpu"]
        default_device_index = 0
    
    device = st.sidebar.selectbox("推理设备", device_options, index=default_device_index)

    st.sidebar.header("🎯 模型参数")
    conf = st.sidebar.slider(
        "置信度阈值 (Confidence)", 
        0.0, 1.0, 0.15, 0.01,
        help="置信度阈值：只显示置信度高于此值的检测结果"
    )
    iou = st.sidebar.slider(
        "IoU 阈值", 
        0.0, 1.0, 0.25, 0.01,
        help="IoU阈值：用于非极大值抑制(NMS)，过滤重叠的检测框"
    )

    frame_step = 5
    time_interval = 1.0  # 默认值，只有摄像头模式下使用
    camera_index = 0  # 默认摄像头索引
    max_frames = None  # 默认不限制帧数
    
    if mode == "视频检测":
        st.sidebar.header("🎬 视频参数")
        frame_step = st.sidebar.number_input(
            "抽帧步长",
            min_value=1,
            value=5,
            step=1,
            help="每隔多少帧抽取一帧进行检测和保存（用于预览）"
        )
    elif mode == "摄像头实时检测":
        st.sidebar.header("📹 摄像头参数")
        camera_index = st.sidebar.selectbox(
            "选择摄像头",
            options=[0, 1, 2],
            index=0,
            help="选择要使用的摄像头设备索引（0通常是默认摄像头）"
        )
        max_frames_input = st.sidebar.number_input(
            "最大检测帧数（留空表示不限制）",
            min_value=1,
            value=None,
            step=100,
            help="设置最大处理的帧数，达到后自动停止"
        )
        max_frames = int(max_frames_input) if max_frames_input else None
        # 根据时间间隔设置抽帧间隔（秒）
        time_interval = st.sidebar.slider(
            "抽帧时间间隔（秒）",
            0.1, 10.0, 1.0, 0.1,
            help="每隔多少秒抽取一帧进行检测和保存"
        )
        frame_step = 1  # 摄像头模式下不再使用frame_step，改为基于时间间隔

    return model_path, mode, device, conf, iou, frame_step, time_interval, camera_index, max_frames


def format_real_time(timestamp: float) -> str:
    """将时间戳格式化为实际时间 HH:MM:SS 格式（不带毫秒）"""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S")  # 只保留时分秒

def format_seconds_to_hhmmss_mmm(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS.x 格式，其中小数部分保留1位（0.1秒精度），不进行四舍五入。
    例如：00:00:00.0、00:00:00.1、00:00:00.2、00:00:00.3 ...
    所有精确到毫秒的时间点都不进行四舍五入处理，保持精确时间点。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    # 只保留 1 位小数（0.1 秒），使用向下取整避免进位导致秒数跳变
    tenth = int((seconds % 1) * 10)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{tenth:01d}"


def format_detection_index(x):
    """将检测序号格式化为6位数字（如000001）"""
    try:
        if pd.notna(x) and x != "":
            return f"{int(x):06d}"
        return x
    except (ValueError, TypeError):
        return x


def log_image_sidebar_parameters(params: dict, image_filename: str = ''):
    """
    记录图片模式左侧面板参数变化到CSV文件（检测一张图片记录一次）
    
    Args:
        params: 参数字典，包含所有左侧面板的参数
        image_filename: 图片文件名
    """
    try:
        # 确保log目录存在
        log_dir = Path("log/image")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径
        csv_file = log_dir / "sidebar_parameter.csv"
        
        # 添加时间戳
        timestamp = time.time()
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        # 定义统一的列顺序
        column_order = [
            '时间戳',
            'Unix时间戳',
            '图片文件名',
            '模型路径',
            '置信度阈值',
            'IoU阈值',
            '推理设备'
        ]
        
        # 准备记录数据
        record = {
            '时间戳': timestamp_str,
            'Unix时间戳': timestamp,
            '图片文件名': str(image_filename) if image_filename else '',
            '模型路径': str(params.get('model_path', '')),
            '置信度阈值': float(params.get('conf', 0)) if params.get('conf') is not None else '',
            'IoU阈值': float(params.get('iou', 0)) if params.get('iou') is not None else '',
            '推理设备': str(params.get('device', ''))
        }
        
        # 创建DataFrame，按照定义的列顺序
        df_record = pd.DataFrame([record], columns=column_order)
        
        # 如果文件存在，追加数据；否则创建新文件
        if csv_file.exists():
            df_record.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_record.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8-sig')
    except Exception:
        # 静默处理错误，避免影响主程序运行
        pass


def log_video_sidebar_parameters(params: dict, video_filename: str = ''):
    """
    记录视频模式左侧面板参数变化到CSV文件（检测一个视频记录一次）
    
    Args:
        params: 参数字典，包含所有左侧面板的参数
        video_filename: 视频文件名
    """
    try:
        # 确保log目录存在
        log_dir = Path("log/video")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径
        csv_file = log_dir / "sidebar_parameter.csv"
        
        # 添加时间戳
        timestamp = time.time()
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        # 定义统一的列顺序
        column_order = [
            '时间戳',
            'Unix时间戳',
            '视频文件名',
            '模型路径',
            '置信度阈值',
            'IoU阈值',
            '抽帧步长',
            '推理设备'
        ]
        
        # 准备记录数据
        record = {
            '时间戳': timestamp_str,
            'Unix时间戳': timestamp,
            '视频文件名': str(video_filename) if video_filename else '',
            '模型路径': str(params.get('model_path', '')),
            '置信度阈值': float(params.get('conf', 0)) if params.get('conf') is not None else '',
            'IoU阈值': float(params.get('iou', 0)) if params.get('iou') is not None else '',
            '抽帧步长': int(params.get('frame_step', 0)) if params.get('frame_step') is not None else '',
            '推理设备': str(params.get('device', ''))
        }
        
        # 创建DataFrame，按照定义的列顺序
        df_record = pd.DataFrame([record], columns=column_order)
        
        # 如果文件存在，追加数据；否则创建新文件
        if csv_file.exists():
            df_record.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_record.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8-sig')
    except Exception:
        # 静默处理错误，避免影响主程序运行
        pass


def log_camera_sidebar_parameters(params: dict):
    """
    记录摄像头模式左侧面板参数变化到CSV文件
    
    Args:
        params: 参数字典，包含所有左侧面板的参数
    """
    try:
        # 确保log目录存在
        log_dir = Path("log/camera")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV文件路径
        csv_file = log_dir / "sidebar_parameter.csv"
        
        # 添加时间戳
        timestamp = time.time()
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        # 定义统一的列顺序
        column_order = [
            '时间戳',
            'Unix时间戳',
            '模型路径',
            '摄像头索引',
            '置信度阈值',
            'IoU阈值',
            '时间间隔',
            '推理设备',
            '最大帧数'
        ]
        
        # 准备记录数据
        record = {
            '时间戳': timestamp_str,
            'Unix时间戳': timestamp,
            '模型路径': str(params.get('model_path', '')),
            '摄像头索引': int(params.get('camera_index', 0)) if params.get('camera_index') is not None else '',
            '置信度阈值': float(params.get('conf', 0)) if params.get('conf') is not None else '',
            'IoU阈值': float(params.get('iou', 0)) if params.get('iou') is not None else '',
            '时间间隔': float(params.get('time_interval', 0)) if params.get('time_interval') is not None else '',
            '推理设备': str(params.get('device', '')),
            '最大帧数': int(params.get('max_frames', 0)) if params.get('max_frames') is not None else ''
        }
        
        # 创建DataFrame，按照定义的列顺序
        df_record = pd.DataFrame([record], columns=column_order)
        
        # 如果文件存在，追加数据；否则创建新文件
        if csv_file.exists():
            df_record.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_record.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8-sig')
    except Exception:
        # 静默处理错误，避免影响主程序运行
        pass


def create_defect_detection_callback(model, conf: float, iou: float, device: str, time_interval: float):
    """
    创建视频帧回调函数（参考 yolo_streamlit_cloud_mini_project_test 的实现）
    使用闭包传递模型和参数，使用全局锁共享数据
    
    Args:
        model: YOLO 模型
        conf: 置信度阈值
        iou: IoU 阈值
        device: 推理设备
        time_interval: 检测时间间隔（秒）
    
    Returns:
        video_frame_callback: 视频帧回调函数
    """
    # 使用闭包内的局部变量跟踪状态
    callback_state = {
        "last_detect_time": 0.0,
        "frame_count": 0,
        "detection_count": 0,
        "start_time": None,
        "frames": [],       # 保存帧图片
        "records": [],      # 保存 DataFrame 记录
        "last_objects": [], # 保存最后一次检测到的对象（持续显示）
        "current_defect_count": 0,  # 当前检测的缺陷数（每次检测更新）
    }
    
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        """处理视频帧的回调函数"""
        # 将 VideoFrame 转换为 numpy 数组
        image = frame.to_ndarray(format="bgr24")
        
        if model is None:
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        
        current_time = time.time()
        
        # 初始化开始时间
        if callback_state["start_time"] is None:
            callback_state["start_time"] = current_time
            callback_state["last_detect_time"] = current_time - time_interval - 0.1
        
        callback_state["frame_count"] += 1
        
        # 基于时间间隔判断是否应该检测
        elapsed_since_last = current_time - callback_state["last_detect_time"]
        should_detect = (elapsed_since_last >= time_interval)
        
        detected_objects = []
        annotated_image = image
        
        if should_detect:
            callback_state["last_detect_time"] = current_time
            callback_state["detection_count"] += 1
            
            # 计算时间信息（无论是否有缺陷都需要）
            start_time_val = callback_state["start_time"]
            relative_time = current_time - start_time_val
            dt = datetime.fromtimestamp(current_time)
            seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0
            time_str_real_precise = format_seconds_to_hhmmss_mmm(seconds_of_day)
            date_str = dt.strftime("%Y-%m-%d")
            
            try:
                # 使用 YOLO 进行检测
                results = model(image, conf=conf, iou=iou, device=device, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    boxes_xyxy = []
                    scores_list = []
                    cls_names = []
                    
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        confidence = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        detected_objects.append({
                            "class": class_name,
                            "confidence": confidence,
                            "xyxy": xyxy.tolist(),
                        })
                        boxes_xyxy.append(xyxy)
                        scores_list.append(confidence)
                        cls_names.append(class_name)
                    
                    # 在图像上绘制检测结果
                    annotated_image = results[0].plot()
                    
                    # 更新最后检测到的对象和缺陷数（用于持续显示）
                    callback_state["last_objects"] = detected_objects.copy()
                    callback_state["current_defect_count"] = len(detected_objects)
                    
                    try:
                        # 保存带标注的帧
                        frame_16_9 = resize_to_16_9(annotated_image)
                        callback_state["frames"].append(ndarray_to_pil(frame_16_9))
                        
                        # 创建 DataFrame 记录（有缺陷）
                        boxes_array = np.array(boxes_xyxy)
                        widths = (boxes_array[:, 2] - boxes_array[:, 0]).astype(int)
                        heights = (boxes_array[:, 3] - boxes_array[:, 1]).astype(int)
                        
                        df_frame = pd.DataFrame({
                            "日期": [date_str] * len(cls_names),
                            "检测序号": [callback_state["detection_count"]] * len(cls_names),
                            "帧序号": [callback_state["frame_count"]] * len(cls_names),
                            "时间点(秒)": [round(relative_time, 2)] * len(cls_names),
                            "绝对时间戳": [round(current_time, 2)] * len(cls_names),
                            "时间点(HH:MM:SS.mmm)": [time_str_real_precise] * len(cls_names),
                            "缺陷类别": cls_names,
                            "置信度": scores_list,
                            "左上角X": boxes_array[:, 0].astype(int),
                            "左上角Y": boxes_array[:, 1].astype(int),
                            "右下角X": boxes_array[:, 2].astype(int),
                            "右下角Y": boxes_array[:, 3].astype(int),
                            "宽度": widths,
                            "高度": heights,
                            "面积": (widths * heights).astype(int),
                            "缺陷数": [len(cls_names)] * len(cls_names),
                        })
                        callback_state["records"].append(df_frame)
                    except Exception:
                        pass
                else:
                    # 未检测到缺陷，记录一条空记录（缺陷信息为 None）
                    callback_state["last_objects"] = []  # 清空当前检测对象
                    callback_state["current_defect_count"] = 0  # 缺陷数为0
                    try:
                        # 保存无缺陷帧
                        frame_16_9 = resize_to_16_9(image)
                        callback_state["frames"].append(ndarray_to_pil(frame_16_9))
                        
                        # 创建 DataFrame 记录（无缺陷，用 None 表示）
                        df_frame = pd.DataFrame({
                            "日期": [date_str],
                            "检测序号": [callback_state["detection_count"]],
                            "帧序号": [callback_state["frame_count"]],
                            "时间点(秒)": [round(relative_time, 2)],
                            "绝对时间戳": [round(current_time, 2)],
                            "时间点(HH:MM:SS.mmm)": [time_str_real_precise],
                            "缺陷类别": [None],
                            "置信度": [None],
                            "左上角X": [None],
                            "左上角Y": [None],
                            "右下角X": [None],
                            "右下角Y": [None],
                            "宽度": [None],
                            "高度": [None],
                            "面积": [None],
                            "缺陷数": [0],
                        })
                        callback_state["records"].append(df_frame)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # 更新共享容器（线程安全）
        with camera_lock:
            # 使用 callback_state 中保存的值（保持最近一次检测的结果）
            camera_result_container["objects"] = callback_state["last_objects"].copy()
            camera_result_container["current_defect_count"] = callback_state["current_defect_count"]
            camera_result_container["frame_count"] = callback_state["frame_count"]
            camera_result_container["detection_count"] = callback_state["detection_count"]
            camera_result_container["last_detect_time"] = callback_state["last_detect_time"]
            camera_result_container["start_time"] = callback_state["start_time"]
            camera_result_container["frames"] = callback_state["frames"].copy()
            camera_result_container["records"] = callback_state["records"].copy()
            # 保存带标注的帧（用于显示）
            if len(detected_objects) > 0:
                camera_result_container["annotated_frame"] = annotated_image.copy()
        
        # 使用 PIL 处理以避免内存泄漏（参考官方文档）
        result_image = Image.fromarray(annotated_image)
        output_array = np.asarray(result_image)
        
        return av.VideoFrame.from_ndarray(output_array, format="bgr24")
    
    return video_frame_callback


def reset_camera_result_container():
    """重置摄像头结果容器"""
    global camera_result_container
    with camera_lock:
        camera_result_container["objects"] = []
        camera_result_container["current_defect_count"] = 0
        camera_result_container["frame_count"] = 0
        camera_result_container["detection_count"] = 0
        camera_result_container["last_detect_time"] = 0.0
        camera_result_container["start_time"] = None
        camera_result_container["annotated_frame"] = None
        camera_result_container["frames"] = []
        camera_result_container["records"] = []


def run_camera_detection(
    model,
    camera_index: int,
    conf: float,
    iou: float = 0.25,
    time_interval: float = 1.0,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> Tuple[List[Image.Image], pd.DataFrame]:
    """使用摄像头进行实时检测，支持实时调整参数
    
    Args:
        time_interval: 抽帧时间间隔（秒），每隔多少秒检测和保存一帧
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return [], pd.DataFrame()
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 初始化或获取检测结果存储
    if 'camera_detection_results' not in st.session_state:
        st.session_state.camera_detection_results = {
            'frames': [],
            'records': [],
            'frame_count': 0,
            'detection_count': 0,  # 检测到缺陷的帧数
            'actual_detection_count': 0,  # 实际进行检测的帧数（无论是否检测到缺陷）
            'start_time': time.time(),  # 记录检测开始时间
            'last_detect_time': time.time()  # 记录上次检测时间
        }
    
    # 获取已有的检测结果（不清空）
    frames_out = st.session_state.camera_detection_results.get('frames', [])
    all_records = st.session_state.camera_detection_results.get('records', [])
    frame_count = st.session_state.camera_detection_results.get('frame_count', 0)
    detection_count = st.session_state.camera_detection_results.get('detection_count', 0)
    actual_detection_count = st.session_state.camera_detection_results.get('actual_detection_count', 0)
    
    # 获取开始时间和上次检测时间
    if len(all_records) == 0 and frame_count == 0:
        # 这是新的检测，重置开始时间和上次检测时间
        start_time = time.time()
        last_detect_time = time.time()
        st.session_state.camera_detection_results['start_time'] = start_time
        st.session_state.camera_detection_results['last_detect_time'] = last_detect_time
    else:
        # 继续之前的检测，使用已有的开始时间和上次检测时间
        start_time = st.session_state.camera_detection_results.get('start_time', time.time())
        last_detect_time = st.session_state.camera_detection_results.get('last_detect_time', time.time())
        if 'start_time' not in st.session_state.camera_detection_results:
            st.session_state.camera_detection_results['start_time'] = start_time
        if 'last_detect_time' not in st.session_state.camera_detection_results:
            st.session_state.camera_detection_results['last_detect_time'] = last_detect_time
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    table_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # 使用session state来跟踪停止状态
    if 'stop_camera' not in st.session_state:
        st.session_state.stop_camera = False
    
    
    try:
        while not st.session_state.stop_camera:
            ret, frame_bgr = cap.read()
            if not ret:
                st.warning("无法从摄像头读取帧")
                break
            
            # 基于时间间隔判断是否应该检测
            current_time = time.time()
            elapsed_since_last = current_time - last_detect_time
            should_detect = (elapsed_since_last >= time_interval)
            
            if should_detect:
                # 更新上次检测时间
                last_detect_time = current_time
                st.session_state.camera_detection_results['last_detect_time'] = last_detect_time
                # 增加实际检测帧数计数
                actual_detection_count += 1
                # 从session_state读取最新的参数值（支持实时调整）
                current_conf = st.session_state.get('camera_conf', conf)
                current_iou = st.session_state.get('camera_iou', iou)
                
                # 进行检测
                results = model.predict(
                    source=frame_bgr,
                    conf=current_conf,
                    iou=current_iou,
                    device=device,
                    verbose=False,
                )
                
                if results and len(results) > 0:
                    r = results[0]
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") and len(r.boxes.xyxy) > 0 else np.empty((0, 4))
                    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") and len(r.boxes.conf) > 0 else np.array([])
                    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") and len(r.boxes.cls) > 0 else np.array([], dtype=int)
                    names = r.names if hasattr(r, "names") else {}
                    cls_names = [names.get(int(c), f"class_{int(c)}") for c in cls_ids] if len(cls_ids) > 0 else []
                    
                    if len(boxes_xyxy) > 0:
                        # 绘制检测框
                        drawn = draw_boxes(
                            frame_bgr,
                            boxes_xyxy.tolist(),
                            cls_names,
                            scores.tolist(),
                        )
                        
                        # 将检测结果调整为16:9用于显示
                        drawn_16_9 = resize_to_16_9(drawn)
                        
                        # 保存抽帧照片（16:9格式）- 基于时间间隔，每次检测都保存
                        frames_out.append(ndarray_to_pil(drawn_16_9))
                        detection_count += 1
                        
                        # 记录检测结果：
                        # - 相对时间（从检测开始算起，秒），保留在"时间点(秒)"列
                        # - 实际时间（当天时刻，精确到 0.1 秒），保留在"时间点(HH:MM:SS.mmm)"列，用于表格和统计图
                        relative_time = current_time - start_time  # 相对时间（秒）
                        dt = datetime.fromtimestamp(current_time)
                        # 使用摄像头实际时间点（当天秒数），不进行四舍五入，格式化为 HH:MM:SS.m（0.1 秒精度）
                        seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0
                        time_str_real_precise = format_seconds_to_hhmmss_mmm(seconds_of_day)
                        date_str = dt.strftime("%Y-%m-%d")  # 日期字符串
                        widths = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).astype(int)
                        heights = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).astype(int)
                        # 检测序号：从0开始，每次检测到缺陷时递增（使用detection_count-1，因为detection_count会在后面+1）
                        detection_index = detection_count
                        df_frame = pd.DataFrame({
                            "日期": [date_str] * len(cls_names),  # 日期列
                            "检测序号": [detection_index] * len(cls_names),  # 检测序号（从0开始）
                            "帧序号": [frame_count] * len(cls_names),  # 摄像头帧序号（保留用于其他用途）
                            "时间点(秒)": [round(relative_time, 2)] * len(cls_names),  # 相对时间（秒）
                            "绝对时间戳": [round(current_time, 2)] * len(cls_names),  # 绝对时间戳
                            "时间点(HH:MM:SS.mmm)": [time_str_real_precise] * len(cls_names),  # 实际时间（精确到 0.1 秒）
                            "缺陷类别": cls_names,
                            "置信度": scores,
                            "左上角X": boxes_xyxy[:, 0].astype(int),
                            "左上角Y": boxes_xyxy[:, 1].astype(int),
                            "右下角X": boxes_xyxy[:, 2].astype(int),
                            "右下角Y": boxes_xyxy[:, 3].astype(int),
                            "宽度": widths,
                            "高度": heights,
                            "面积": (widths * heights).astype(int),
                        })
                        all_records.append(df_frame)
                        
                        # 更新session_state中的检测结果
                        st.session_state.camera_detection_results['frames'] = frames_out
                        st.session_state.camera_detection_results['records'] = all_records
                        st.session_state.camera_detection_results['frame_count'] = frame_count
                        st.session_state.camera_detection_results['detection_count'] = detection_count
                        st.session_state.camera_detection_results['actual_detection_count'] = actual_detection_count
                        
                        # 显示带检测框的帧（16:9格式，只在有缺陷时更新，减少闪烁）
                        # 使用列布局限制视频框宽度，从而缩小显示尺寸
                        with frame_placeholder.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(ndarray_to_pil(drawn_16_9), use_container_width=True)
                    else:
                        # 没有检测到缺陷，但仍需要记录该帧信息（缺陷相关字段为空值）
                        detection_count += 1
                        # 保存原始帧（不标注框）到frames_out
                        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                        frames_out.append(ndarray_to_pil(frame_bgr_16_9))
                        
                        # 记录检测结果（缺陷相关字段为空值）
                        relative_time = current_time - start_time  # 相对时间（秒）
                        dt = datetime.fromtimestamp(current_time)
                        seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1_000_000.0
                        time_str_real_precise = format_seconds_to_hhmmss_mmm(seconds_of_day)  # 实际时间（精确到 0.1 秒，不四舍五入）
                        date_str = dt.strftime("%Y-%m-%d")  # 日期字符串
                        # 检测序号：从0开始，每次检测时递增
                        detection_index = detection_count
                        # 创建一条记录，缺陷相关字段为空值
                        df_frame = pd.DataFrame({
                            "日期": [date_str],
                            "检测序号": [detection_index],
                            "帧序号": [frame_count],
                            "时间点(秒)": [round(relative_time, 2)],
                            "绝对时间戳": [round(current_time, 2)],
                            "时间点(HH:MM:SS.mmm)": [time_str_real_precise],
                            "缺陷类别": [None],  # 空值
                            "置信度": [None],  # 空值
                            "左上角X": [None],  # 空值
                            "左上角Y": [None],  # 空值
                            "右下角X": [None],  # 空值
                            "右下角Y": [None],  # 空值
                            "宽度": [None],  # 空值
                            "高度": [None],  # 空值
                            "面积": [None],  # 空值
                        })
                        all_records.append(df_frame)
                        
                        # 更新session_state中的检测结果
                        st.session_state.camera_detection_results['frames'] = frames_out
                        st.session_state.camera_detection_results['records'] = all_records
                        st.session_state.camera_detection_results['frame_count'] = frame_count
                        st.session_state.camera_detection_results['detection_count'] = detection_count
                        st.session_state.camera_detection_results['actual_detection_count'] = actual_detection_count
                        
                        # 显示原始帧（调整为16:9，不频繁更新以减少提示）
                        if frame_count % 5 == 0:  # 每5帧更新一次，减少提示
                            with frame_placeholder.container():
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(ndarray_to_pil(frame_bgr_16_9), use_container_width=True)
                else:
                    # 检测失败，显示原始帧（调整为16:9，不频繁更新）
                    if frame_count % 5 == 0:
                        frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                        with frame_placeholder.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(ndarray_to_pil(frame_bgr_16_9), use_container_width=True)
                    # 更新session_state中的实际检测帧数（即使检测失败）
                    st.session_state.camera_detection_results['actual_detection_count'] = actual_detection_count
            else:
                # 跳过检测的帧，显示原始帧（调整为16:9，不频繁更新）
                if frame_count % 5 == 0:
                    frame_bgr_16_9 = resize_to_16_9(frame_bgr)
                    with frame_placeholder.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(ndarray_to_pil(frame_bgr_16_9), use_container_width=True)
            
            # 显示统计信息和实时报表（每10帧更新一次，减少提示）
            if frame_count % 10 == 0:
                # 只统计有缺陷的记录（缺陷类别不为空的记录）
                total_defects = sum(len(rec[rec["缺陷类别"].notna()]) for rec in all_records)
                # 计算总检测时长
                elapsed_time = time.time() - start_time
                # 格式化时长为 MM:SS 或 SS 秒
                if elapsed_time >= 60:
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)
                    time_str = f"{minutes}分{seconds}秒"
                else:
                    time_str = f"{int(elapsed_time)}秒"
                
                stats_placeholder.markdown(
                    f"""
                    **实时统计：**
                    - 总检测时长: {time_str}
                    - 检测帧数: {actual_detection_count}
                    - 总缺陷数量: {total_defects}
                    """
                )
                
                # 实时显示检测结果表格和图表
                if all_records:
                    current_df = pd.concat(all_records, ignore_index=True)
                    with table_placeholder.container():
                        st.markdown("#### 📊 实时检测结果报表")
                        # 去掉绝对时间戳列，格式化检测序号
                        current_df_display = current_df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in current_df.columns else current_df.copy()
                        # 将检测序号格式化为6位数字（如000001）
                        if "检测序号" in current_df_display.columns:
                            current_df_display = current_df_display.copy()  # 避免SettingWithCopyWarning
                            current_df_display["检测序号"] = current_df_display["检测序号"].apply(format_detection_index)
                        st.dataframe(current_df_display, use_container_width=True, height=300)
                    
                    # 实时显示统计图表（使用检测序号）
                    with chart_placeholder.container():
                        st.markdown("#### ⏱️ 缺陷出现时间分布")
                        # 按时间点统计缺陷数量（使用摄像头实际时间 HH:MM:SS.mmm 作为横轴）
                        if "时间点(HH:MM:SS.mmm)" in current_df.columns:
                            all_time_points = (
                                current_df["时间点(HH:MM:SS.mmm)"]
                                .astype(str)
                                .dropna()
                                .sort_values()
                                .reset_index(drop=True)
                                .unique()
                            )
                            if len(all_time_points) > 0:
                                defect_df = current_df[current_df["缺陷类别"].notna()].copy()
                                if not defect_df.empty:
                                    defect_counts = (
                                        defect_df.groupby("时间点(HH:MM:SS.mmm)")
                                        .size()
                                        .reindex(all_time_points, fill_value=0)
                                    )
                                else:
                                    defect_counts = pd.Series(
                                        [0] * len(all_time_points),
                                        index=all_time_points,
                                    )
                                # 为绘图单独建一列简单字段名“时间点”
                                time_counts = pd.DataFrame({
                                    "时间点": all_time_points,
                                    "缺陷数量": defect_counts.values,
                                })
                                # 使用 Altair 显式指定 X/Y
                                chart = (
                                    alt.Chart(time_counts)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X(field="时间点", type="nominal", sort=None, title="时间点"),
                                        y=alt.Y(field="缺陷数量", type="quantitative", title="缺陷数量"),
                                    )
                                    .properties(height=300)
                                )
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                st.info("等待检测数据...")
                        else:
                            st.info("等待检测数据...")
            
            frame_count += 1
            st.session_state.camera_detection_results['frame_count'] = frame_count
            
            if max_frames is not None and frame_count >= max_frames:
                break
                
    except Exception as e:
        st.error(f"摄像头检测出错: {e}")
    finally:
        cap.release()
        frame_placeholder.empty()
        stats_placeholder.empty()
        table_placeholder.empty()
        chart_placeholder.empty()
        st.session_state.stop_camera = False  # 重置停止状态
    
    summary_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    return frames_out, summary_df


def main():
    st.set_page_config(
        page_title="碳纤维缺陷检测系统", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 添加自定义CSS，优化界面显示
    st.markdown("""
    <style>
    .main .block-container { padding-top: -2rem !important; padding-bottom: 1rem; }
    .main div[data-testid="stMarkdownContainer"]:first-of-type { margin-top: -2rem !important; margin-bottom: -5rem !important; padding: 0 !important; position: relative; top: -2rem; }
    .main h1[title] { margin: 0 !important; padding: 0 !important; font-size: 1.8rem !important; line-height: 1.2 !important; }
    .main .element-container:nth-of-type(2) { margin-top: -4.5rem !important; margin-bottom: 0 !important; }
    .main .element-container:nth-of-type(2) [data-testid="stMarkdownContainer"], .main .element-container:nth-of-type(2) [data-baseweb="notification"], .main .element-container:nth-of-type(2) > div { margin: 0 !important; padding-top: 0 !important; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    .element-container { margin-bottom: 0.5rem; }
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stText { font-size: 0.85rem; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stNumberInput label { font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-size: 1rem !important; }
    div[data-testid="stVideo"] { aspect-ratio: 16/9; width: 100%; max-width: 100%; }
    div[data-testid="stVideo"] video, div[data-testid="stVideo"] iframe { width: 100%; height: 100%; aspect-ratio: 16/9; object-fit: contain; }
    div[data-testid="stImage"] { aspect-ratio: 16/9; width: 100%; max-width: 100%; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; overflow: hidden; }
    div[data-testid="stImage"][id*="camera"] { max-height: 50vh !important; }
    div[data-testid="stImage"] img { max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; }
    </style>
    """, unsafe_allow_html=True)
    
    # 基于 YOLO11 + Streamlit 实现碳纤维缺陷检测
    tooltip_text = (
        "✅ 基于 YOLO11 + Streamlit 实现碳纤维缺陷检测\n"
        "✅ 支持图片、视频和摄像头实时缺陷检测，自动生成检测框标注缺陷位置，生成详细的检测结果统计图表\n"
        "✅ 视频检测支持带检测框视频的播放与下载\n"
        "✅ 摄像头实时检测支持实时检测框生成、关键帧图片展示与批量下载\n"
        "✅ 支持自定义模型加载，支持自定义置信度、IoU阈值、检测频率、输出帧率、最大检测帧数，支持CPU和GPU推理加速"
    ).replace('"', '&quot;').replace("'", "&#39;")
    st.markdown(
        f'<div style="margin-top:-2rem!important;margin-bottom:-5rem!important;padding:0!important;position:relative;top:-2rem"><h1 title="{tooltip_text}" style="margin:0!important;padding:0!important;font-size:1.8rem!important;line-height:1.2!important">🔍 碳纤维缺陷检测系统</h1></div>',
        unsafe_allow_html=True
    )

    model_path, mode, device, conf, iou, frame_step, time_interval, camera_index, max_frames = sidebar_controls()
    
    # 记录当前模式（不清空视频检测结果，允许在不同模式间切换时保留数据）
    st.session_state.last_mode = mode

    if model_path is None or not model_path.exists():
        st.warning("⚠️ 请先配置模型：在左侧勾选使用默认模型或上传模型文件")
        st.info("💡 提示：如果已训练模型，默认模型路径为 `model/best.pt`")
        return

    try:
        with st.spinner("🔄 正在加载模型，请稍候..."):
            model = load_model(model_path, device)
        st.success(f"✅ 模型加载成功: {model_path.name}")
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.info("💡 请检查模型文件是否正确，或确保已安装 ultralytics 库：`pip install ultralytics`")
        return

    if mode == "图片检测":
        st.subheader("📷 图片检测")
        
        # 当前检测参数
        model_path_str = str(model_path) if model_path else "none"
        current_image_params = {
            'model_path': model_path_str,
            'conf': conf,
            'iou': iou,
            'device': device
        }
        
        # 初始化图片检测结果存储
        if 'image_detection_results' not in st.session_state:
            st.session_state.image_detection_results = {}
        
        # 初始化文件上传器重置计数器
        if 'image_uploader_reset_counter' not in st.session_state:
            st.session_state.image_uploader_reset_counter = 0
        
        img_file = st.file_uploader(
            "上传待检测图片", 
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="支持 JPG、PNG、BMP、WEBP 格式",
            key=f"image_file_uploader_{st.session_state.image_uploader_reset_counter}"
        )
        
        # 如果上传了新文件，立即清空选中的测试图片
        if img_file is not None:
            if 'selected_test_image' in st.session_state and st.session_state.selected_test_image is not None:
                st.session_state.selected_test_image = None
        
        # 显示测试图片缩略图
        test_images_dir = Path("test_image_video/images")
        test_image_files = []
        if test_images_dir.exists():
            test_image_files = sorted([f for f in test_images_dir.iterdir() 
                                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']])
            
            if test_image_files:
                st.markdown("### 📸 图片示例")
                st.caption("点击下方缩略图选择测试图片进行检测")
                
                # 初始化选中的测试图片
                if 'selected_test_image' not in st.session_state:
                    st.session_state.selected_test_image = None
                
                # 创建缩略图网格（每行5个）
                cols_per_row = 5
                for i in range(0, len(test_image_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, img_path in enumerate(test_image_files[i:i+cols_per_row]):
                        with cols[j]:
                            try:
                                # 加载并显示缩略图
                                test_img = Image.open(img_path)
                                # 创建缩略图（最大宽度150px）
                                test_img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                                st.image(test_img, use_container_width=True)
                                
                                # 创建选择按钮
                                button_key = f"select_test_img_{img_path.name}"
                                # 高亮显示当前选中的图片
                                is_selected = (st.session_state.get('selected_test_image') == str(img_path))
                                button_label = "✓ 已选择" if is_selected else "选择"
                                button_type = "primary" if is_selected else "secondary"
                                
                                if st.button(button_label, key=button_key, use_container_width=True, type=button_type):
                                    st.session_state.selected_test_image = str(img_path)
                                    # 清空文件上传器状态，确保使用测试图片
                                    uploader_key = f"image_file_uploader_{st.session_state.image_uploader_reset_counter}"
                                    if uploader_key in st.session_state:
                                        del st.session_state[uploader_key]
                                    st.rerun()
                            except Exception:
                                st.error(f"无法加载图片: {img_path.name}")

        # 处理图片检测
        result = None
        img_file_key = None
        selected_test_image_path = st.session_state.get('selected_test_image')
        
        # 优先使用上传的文件，如果没有上传文件则使用选中的测试图片
        if img_file is not None:
            # 如果上传了新文件，清空选中的测试图片
            if 'selected_test_image' in st.session_state:
                st.session_state.selected_test_image = None
            use_test_image = False
        elif selected_test_image_path and Path(selected_test_image_path).exists():
            # 使用测试图片
            use_test_image = True
            test_img = Image.open(selected_test_image_path)
            # 创建一个模拟的文件名用于生成key
            test_img_name = Path(selected_test_image_path).name
        else:
            use_test_image = False
        
        if img_file is not None or (use_test_image and selected_test_image_path):
            # 确定使用的图片和文件名
            if use_test_image:
                img = test_img
                img_name = test_img_name
            else:
                img = Image.open(img_file)
                img_name = img_file.name
            
            # 生成包含所有影响检测结果的参数的缓存key
            # 包括：图片名称、模型路径、推理设备、置信度阈值、IoU阈值
            model_path_str = str(model_path) if model_path else "none"
            model_path_hash = hash(model_path_str)
            img_file_key = f"image_{img_name}_{hash(img_name)}_model{model_path_hash}_dev{device}_conf{conf:.3f}_iou{iou:.3f}"
            
            # 如果上传了新图片、选择了新的测试图片，或任何影响检测结果的参数改变（模型路径、设备、阈值等），进行检测
            if img_file_key not in st.session_state.image_detection_results:
                with st.spinner("🔍 正在检测缺陷..."):
                    out_img, df = run_inference_image(model, img, conf, iou, device)
                
                # 将原始图片调整为16:9用于存储
                img_bgr = pil_to_ndarray(img)
                img_16_9 = resize_to_16_9(img_bgr)
                img_16_9_pil = ndarray_to_pil(img_16_9)
                
                # 将PIL图片转换为字节存储
                img_bytes_io = io.BytesIO()
                img_16_9_pil.save(img_bytes_io, format='PNG')
                img_bytes_io.seek(0)
                original_img_bytes = img_bytes_io.getvalue()
                
                out_img_bytes_io = io.BytesIO()
                out_img.save(out_img_bytes_io, format='PNG')
                out_img_bytes_io.seek(0)
                detected_img_bytes = out_img_bytes_io.getvalue()
                
                # 保存检测结果到session_state
                st.session_state.image_detection_results[img_file_key] = {
                    'original_img_bytes': original_img_bytes,
                    'detected_img_bytes': detected_img_bytes,
                    'df': df,
                    'img_file_name': img_name,
                }
                result = st.session_state.image_detection_results[img_file_key]
                
                # 检测完成后记录参数到CSV文件
                log_image_sidebar_parameters(current_image_params, img_name)
            else:
                # 使用已有的检测结果
                result = st.session_state.image_detection_results[img_file_key]
        elif len(st.session_state.image_detection_results) > 0:
            # 没有上传新图片，但session_state中有结果，显示最后一个结果
            last_key = list(st.session_state.image_detection_results.keys())[-1]
            result = st.session_state.image_detection_results[last_key]
            img_file_key = last_key
        
        # 显示检测结果
        if result:
                # 从字节恢复图片
                original_img = Image.open(io.BytesIO(result['original_img_bytes']))
                detected_img = Image.open(io.BytesIO(result['detected_img_bytes']))
                df = result['df']
                img_file_name = result.get('img_file_name', 'image')
                
                # 左右对照显示原始图片和检测结果
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📸 原始图片")
                    st.image(original_img, use_container_width=True)

                with col2:
                    st.markdown("### ✅ 检测结果")
                    st.image(detected_img, use_container_width=True)
                
                # 清空检测结果按钮
                if st.button("🗑️ 清空检测结果", type="secondary", use_container_width=True, key="clear_image_results"):
                    st.session_state.image_detection_results = {}
                    # 清空选中的测试图片
                    if 'selected_test_image' in st.session_state:
                        st.session_state.selected_test_image = None
                    # 增加重置计数器，强制文件上传器重新渲染
                    st.session_state.image_uploader_reset_counter += 1
                    # 清空所有可能的文件上传器相关状态
                    uploader_keys = [k for k in st.session_state.keys() if k.startswith("image_file_uploader_")]
                    for key in uploader_keys:
                        del st.session_state[key]
                    st.rerun()

                if not df.empty:
                    # 统计信息 - 紧凑显示
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("检测到缺陷数量", len(df))
                    with col_stat2:
                        st.metric("缺陷类别数", df["缺陷类别"].nunique())
                    with col_stat3:
                        st.metric("平均置信度", f"{df['置信度'].mean():.3f}")
                    
                    # 使用expander折叠详细信息，节省空间
                    with st.expander("📊 查看详细检测结果", expanded=False):
                        st.dataframe(df, use_container_width=True, height=250)
                        # 下载按钮
                        csv = df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载检测结果 (CSV)",
                            data=csv,
                            file_name=f"detection_results_{img_file_name}.csv",
                            mime="text/csv",
                        )
                    
                else:
                    st.info("ℹ️ 未检测到缺陷。可以尝试降低置信度阈值。")

    elif mode == "视频检测":
        st.subheader("🎬 视频检测")
        
        # 当前检测参数
        model_path_str = str(model_path) if model_path else "none"
        current_video_params = {
            'model_path': model_path_str,
            'conf': conf,
            'iou': iou,
            'frame_step': frame_step,
            'device': device
        }
        
        video_file = st.file_uploader(
            "上传待检测视频", 
            type=["mp4", "avi", "mov", "mkv", "flv"],
            help="支持 MP4、AVI、MOV、MKV、FLV 格式"
        )
        
        # 如果上传了新文件，立即清空选中的测试视频
        if video_file is not None:
            if 'selected_test_video' in st.session_state and st.session_state.selected_test_video is not None:
                st.session_state.selected_test_video = None
        
        # 显示测试视频缩略图
        test_videos_dir = Path("test_image_video/videos")
        test_video_files = []
        if test_videos_dir.exists():
            test_video_files = sorted([f for f in test_videos_dir.iterdir() 
                                      if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']])
            
            if test_video_files:
                st.markdown("### 🎬 视频示例")
                st.caption("点击下方缩略图选择测试视频进行检测")
                
                # 初始化选中的测试视频
                if 'selected_test_video' not in st.session_state:
                    st.session_state.selected_test_video = None
                
                # 创建缩略图网格（每行3个，因为视频缩略图较大）
                cols_per_row = 3
                for i in range(0, len(test_video_files), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, video_path in enumerate(test_video_files[i:i+cols_per_row]):
                        with cols[j]:
                            try:
                                # 提取视频第一帧作为缩略图
                                cap = cv2.VideoCapture(str(video_path))
                                if cap.isOpened():
                                    ret, frame = cap.read()
                                    cap.release()
                                    
                                    if ret:
                                        # 转换为RGB并调整大小
                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_pil = Image.fromarray(frame_rgb)
                                        # 创建缩略图（最大宽度200px）
                                        frame_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
                                        st.image(frame_pil, use_container_width=True)
                                        
                                        # 显示视频文件名
                                        st.caption(video_path.name)
                                        
                                        # 创建选择按钮
                                        button_key = f"select_test_video_{video_path.name}"
                                        # 高亮显示当前选中的视频
                                        is_selected = (st.session_state.get('selected_test_video') == str(video_path))
                                        button_label = "✓ 已选择" if is_selected else "选择"
                                        button_type = "primary" if is_selected else "secondary"
                                        
                                        if st.button(button_label, key=button_key, use_container_width=True, type=button_type):
                                            st.session_state.selected_test_video = str(video_path)
                                            # 清空文件上传器状态，确保使用测试视频
                                            if "video_file_uploader" in st.session_state:
                                                del st.session_state["video_file_uploader"]
                                            st.rerun()
                                    else:
                                        st.error(f"无法读取视频: {video_path.name}")
                                else:
                                    st.error(f"无法打开视频: {video_path.name}")
                            except Exception:
                                st.error(f"无法加载视频: {video_path.name}")

        # 处理选中的测试视频
        selected_test_video_path = st.session_state.get('selected_test_video')
        if selected_test_video_path and Path(selected_test_video_path).exists():
            # 如果选择了测试视频，将其作为上传的文件处理
            if video_file is None:  # 只有在没有上传新文件时才使用测试视频
                # 读取测试视频文件
                with open(selected_test_video_path, 'rb') as f:
                    test_video_bytes = f.read()
                
                # 创建一个模拟的 UploadedFile 对象
                class MockUploadedFile:
                    def __init__(self, name, bytes_data):
                        self.name = name
                        self.read = lambda: bytes_data
                        self.seek = lambda pos: None
                        self._is_mock = True  # 标记这是模拟文件
                
                video_file = MockUploadedFile(Path(selected_test_video_path).name, test_video_bytes)

        if video_file is not None:
            # 如果上传了新文件（不是MockUploadedFile），清空选中的测试视频
            if not hasattr(video_file, '_is_mock'):
                # 这是真实上传的文件，清空选中的测试视频
                if 'selected_test_video' in st.session_state:
                    st.session_state.selected_test_video = None
            
            # 初始化session_state
            # 生成包含所有影响检测结果的参数的缓存key
            # 包括：视频文件名、模型路径、推理设备、置信度阈值、IoU阈值、抽帧步长、输出帧率
            model_path_str = str(model_path) if model_path else "none"
            model_path_hash = hash(model_path_str)
            video_file_key = f"video_{video_file.name}_{hash(video_file.name)}_model{model_path_hash}_dev{device}_conf{conf:.3f}_iou{iou:.3f}_step{frame_step}"
            if 'video_detection_results' not in st.session_state:
                st.session_state.video_detection_results = {}
            
            # 初始化视频检测参数跟踪（保留用于兼容性，但主要依赖key中的参数）
            if 'video_detection_params' not in st.session_state:
                st.session_state.video_detection_params = {}
            
            # 跟踪检测参数（用于识别参数变化，不在变化时立即清空）
            current_params_key = f"{video_file_key}_params"
            current_params = {
                'video_file_name': video_file.name,  # 保存视频文件名用于比较
                'conf': conf,
                'iou': iou,
                'frame_step': frame_step,
                'device': device,
                'model_path': model_path_str
            }
            
            # 初始化停止检测状态
            stop_video_key = f"stop_video_{video_file_key}"
            if stop_video_key not in st.session_state:
                st.session_state[stop_video_key] = False
            
            # 显示视频信息
            video_bytes = video_file.read()
            
            # 视频信息 - 紧凑显示
            video_info_cols = st.columns(4)
            with video_info_cols[0]:
                st.caption("📁 文件名")
                st.caption(video_file.name)
            with video_info_cols[1]:
                st.caption("📏 文件大小")
                file_size_mb = len(video_bytes) / (1024 * 1024)
                st.caption(f"{file_size_mb:.2f} MB")

            # 原始视频 & 检测结果视频预览（加载后立即显示，并进行 16:9 适配）
            # 使用与结果区域一致的两列布局，检测完成后右侧自动覆盖为检测视频
            # 将原始视频转换为16:9格式用于显示
            video_display_bytes = convert_video_to_16_9(video_bytes, video_file.name)
            # 如果转换失败，使用原始视频（可能不是16:9，但至少可以显示）
            if video_display_bytes is None:
                video_display_bytes = video_bytes
            
            # 如果已经有检测结果，则从 session_state 中取出检测视频字节
            processed_video_bytes_preview = None
            if (
                "video_detection_results" in st.session_state
                and video_file_key in st.session_state.video_detection_results
            ):
                result = st.session_state.video_detection_results[video_file_key]
                processed_video_bytes_preview = result.get("processed_video_bytes")
                # 验证视频字节是否有效（至少应该有一些数据）
                if processed_video_bytes_preview is not None and len(processed_video_bytes_preview) < 100:
                    processed_video_bytes_preview = None

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("##### 🎞️ 原始视频")
                # 尝试显示视频，如果失败则显示提示
                video_display_success = False
                try:
                    st.video(video_display_bytes, format="video/mp4")
                    video_display_success = True
                except Exception:
                    try:
                        st.video(video_display_bytes)
                        video_display_success = True
                    except Exception:
                        video_display_success = False
                
                if not video_display_success:
                    st.warning("⚠️ 视频无法在浏览器中播放，请下载后使用本地播放器查看")
                    st.download_button(
                        label="📥 下载原始视频",
                        data=video_display_bytes,
                        file_name=video_file.name,
                        mime="video/mp4",
                        key="download_original_video"
                    )
            
            with col_v2:
                st.markdown("##### ✅ 检测结果视频")
                if processed_video_bytes_preview and len(processed_video_bytes_preview) > 0:
                    # 尝试显示视频
                    result_video_display_success = False
                    try:
                        st.video(processed_video_bytes_preview, format="video/mp4")
                        result_video_display_success = True
                    except Exception:
                        try:
                            st.video(processed_video_bytes_preview)
                            result_video_display_success = True
                        except Exception:
                            result_video_display_success = False
                    
                    # 如果视频无法播放，显示关键帧序列作为备选方案
                    if not result_video_display_success:
                        st.warning("⚠️ 视频无法在浏览器中播放")
                        # 从检测结果中获取关键帧
                        if (
                            "video_detection_results" in st.session_state
                            and video_file_key in st.session_state.video_detection_results
                        ):
                            result = st.session_state.video_detection_results[video_file_key]
                            preview_frames = result.get('frames', [])
                            if preview_frames and len(preview_frames) > 0:
                                # 使用可展开的容器显示关键帧预览
                                with st.expander(f"📸 查看关键帧预览（共 {len(preview_frames)} 帧）", expanded=False):
                                    display_frames = preview_frames[:12]  # 最多显示12帧
                                    # 使用3列布局显示关键帧，更紧凑
                                    cols_per_row = 3
                                    for i in range(0, len(display_frames), cols_per_row):
                                        cols = st.columns(cols_per_row)
                                        for j, col in enumerate(cols):
                                            if i + j < len(display_frames):
                                                with col:
                                                    st.image(display_frames[i + j], use_container_width=True, caption=f"帧 {i + j + 1}")
                                    if len(preview_frames) > 12:
                                        st.caption("* 仅显示前12帧，完整视频请下载查看")
                            else:
                                st.info("💡 提示：视频文件已生成，请下载后使用本地播放器查看")
                        else:
                            st.info("💡 提示：视频文件已生成，请下载后使用本地播放器查看")
                    
                    # 始终提供下载按钮
                    st.download_button(
                        label="📥 下载检测结果视频",
                        data=processed_video_bytes_preview,
                        file_name=f"video_detected_{video_file.name.rsplit('.', 1)[0]}.mp4",
                        mime="video/mp4",
                        key="download_result_video"
                    )
                else:
                    st.info("检测完成后将在此显示检测结果视频")
            
            # 停止检测和清空检测结果按钮（放在开始检测按钮上方）
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                if st.button("⏹️ 停止检测", type="secondary", use_container_width=True, key=f"stop_video_button_{video_file_key}"):
                    # 设置停止标志
                    st.session_state[stop_video_key] = True
                    # 清空检测结果，回到初始状态
                    if video_file_key in st.session_state.video_detection_results:
                        del st.session_state.video_detection_results[video_file_key]
                    # 清空参数记录
                    if current_params_key in st.session_state.video_detection_params:
                        del st.session_state.video_detection_params[current_params_key]
                    st.rerun()
            with button_col2:
                if st.button("🗑️ 清空检测结果", type="secondary", use_container_width=True, key=f"clear_video_results_{video_file_key}"):
                    if video_file_key in st.session_state.video_detection_results:
                        del st.session_state.video_detection_results[video_file_key]
                    # 同时清空参数记录，这样下次点击开始检测时会重新检测
                    if current_params_key in st.session_state.video_detection_params:
                        del st.session_state.video_detection_params[current_params_key]
                    # 清空选中的测试视频
                    if 'selected_test_video' in st.session_state:
                        st.session_state.selected_test_video = None
                    st.session_state[stop_video_key] = False  # 重置停止状态
                    st.rerun()
            
            # 检测按钮放在按钮下方
            if st.button("🚀 开始检测", type="primary", use_container_width=True, key="start_video_detection"):
                # 重置停止状态
                st.session_state[stop_video_key] = False
                
                # 检查是否需要重新检测：由于key已包含所有参数，如果key存在且结果有效则复用，否则重新检测
                need_re_detect = True
                if video_file_key in st.session_state.video_detection_results:
                    old_result = st.session_state.video_detection_results[video_file_key]
                    old_video_bytes = old_result.get('processed_video_bytes')
                    # 只有当旧结果中有有效的检测视频字节时才复用
                    if old_video_bytes is not None and len(old_video_bytes) > 0:
                        need_re_detect = False
                        st.info("ℹ️ 检测设置与上次一致，使用已有检测结果")
                    else:
                        # 旧结果中没有有效的检测视频，需要重新检测
                        del st.session_state.video_detection_results[video_file_key]
                
                # 如果需要重新检测
                if need_re_detect:
                    
                    # 更新参数记录
                    st.session_state.video_detection_params[current_params_key] = current_params
                    
                    with st.spinner("🔄 正在对视频进行缺陷检测，请稍候..."):
                        try:
                            # 创建停止检查回调函数
                            def check_stop():
                                return st.session_state.get(stop_video_key, False)
                            
                            frames, df, video_info, processed_video_bytes = run_inference_video(
                                model, video_bytes, conf, iou, frame_step, device, process_all_frames=True, stop_check_callback=check_stop, video_filename=video_file.name
                            )
                            # 检测完成后，检查停止状态
                            if not st.session_state.get(stop_video_key, False):
                                # 保存检测结果到session_state
                                st.session_state.video_detection_results[video_file_key] = {
                                    'frames': frames,
                                    'df': df,
                                    'video_info': video_info,
                                    'video_file_name': video_file.name,
                                    'processed_video_bytes': processed_video_bytes,
                                }
                                # 检测完成后记录参数到CSV文件
                                log_video_sidebar_parameters(current_video_params, video_file.name)
                                # 强制刷新页面，确保右侧视频显示
                                st.rerun()
                            else:
                                # 如果检测过程中被停止，不保存结果
                                st.info("ℹ️ 检测已停止，结果未保存")
                        except Exception as e:
                            st.error(f"❌ 视频检测出错：{e}")
                            import traceback
                            st.error(f"详细错误：{traceback.format_exc()}")
                            frames, df, video_info = [], pd.DataFrame(), {}
                else:
                    # 不需要重新检测，但需要更新参数记录（以防万一）
                    st.session_state.video_detection_params[current_params_key] = current_params
            
            # 从session_state读取检测结果（如果存在）
            if video_file_key in st.session_state.video_detection_results:
                result = st.session_state.video_detection_results[video_file_key]
                frames = result['frames']
                df = result['df']
                video_info = result['video_info']
                processed_video_bytes = result.get('processed_video_bytes')
                has_results = True
            else:
                frames = []
                df = pd.DataFrame()
                video_info = {}
                processed_video_bytes = None
                has_results = False
            
            # 显示检测结果详情（如果有）
            if has_results:

                # 在上方预览区域已经展示了原始视频与检测结果视频，这里仅显示统计与表格
                if not df.empty:
                    # 统计信息 - 紧凑显示（只统计有缺陷的记录，与摄像头模式保持一致）
                    df_with_defects = df[df["缺陷类别"].notna()]  # 过滤掉缺陷类别为空的记录
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("总缺陷数", len(df_with_defects) if not df_with_defects.empty else 0)
                    with stat_col2:
                        if not df_with_defects.empty and df_with_defects["置信度"].notna().any():
                            st.metric("平均置信度", f"{df_with_defects['置信度'].mean():.3f}")
                        else:
                            st.metric("平均置信度", "0.000")
                    
                    # 使用expander折叠详细信息（与摄像头模式保持一致）
                    with st.expander("📋 查看详细检测结果", expanded=False):
                        # 显示详细表格（去掉绝对时间戳列，格式化检测序号）
                        # 注意：表格显示所有记录，包括无缺陷帧（缺陷类别为空的记录），与摄像头模式保持一致
                        df_display = df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in df.columns else df.copy()
                        # 将检测序号格式化为6位数字（如000001）
                        if "检测序号" in df_display.columns:
                            df_display = df_display.copy()  # 避免SettingWithCopyWarning
                            df_display["检测序号"] = df_display["检测序号"].apply(format_detection_index)
                        st.dataframe(df_display, use_container_width=True, height=250)
                        
                        # 下载按钮（下载时也去掉绝对时间戳列，格式化检测序号）
                        df_download = df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in df.columns else df.copy()
                        # 将检测序号格式化为6位数字（如000001）
                        if "检测序号" in df_download.columns:
                            df_download = df_download.copy()  # 避免SettingWithCopyWarning
                            df_download["检测序号"] = df_download["检测序号"].apply(format_detection_index)
                        csv = df_download.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载检测结果 (CSV)",
                            data=csv,
                            file_name=f"video_detection_results_{video_file.name}.csv",
                            mime="text/csv",
                        )
                    
                    # 统计图表 - 使用expander折叠，独立于详细检测结果（与摄像头模式保持一致）
                    with st.expander("📊 查看统计图表", expanded=False):
                        st.markdown("#### ⏱️ 缺陷出现时间分布")
                        # 使用精确到 0.1 秒的时间点(HH:MM:SS.mmm) 作为横轴
                        time_col_name = "时间点(HH:MM:SS.mmm)"
                        if time_col_name in df.columns:
                            # 1）获取所有检测时间点（包含无缺陷帧）
                            all_time_points = (
                                df[time_col_name]
                                .astype(str)
                                .dropna()
                                .sort_values()
                                .reset_index(drop=True)
                                .unique()
                            )

                            if len(all_time_points) > 0:
                                # 2）只用有缺陷记录统计每个时间点的缺陷数量
                                defect_df = df[df["缺陷类别"].notna()].copy()
                                if not defect_df.empty:
                                    defect_counts = (
                                        defect_df.groupby(time_col_name)
                                        .size()
                                        .reindex(all_time_points, fill_value=0)
                                    )
                                else:
                                    # 全部是无缺陷帧，缺陷数量全为 0
                                    defect_counts = pd.Series(
                                        [0] * len(all_time_points),
                                        index=all_time_points,
                                    )

                                # 组装为 DataFrame，为绘图单独建一列简单字段名“时间点”
                                time_counts = pd.DataFrame({
                                    "时间点": all_time_points,
                                    "缺陷数量": defect_counts.values,
                                })

                                # 使用 Altair 绘制，明确指定 X/Y，避免 Streamlit 的索引聚合问题
                                chart = (
                                    alt.Chart(time_counts)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X(field="时间点", type="nominal", sort=None, title="时间点"),
                                        y=alt.Y(field="缺陷数量", type="quantitative", title="缺陷数量"),
                                    )
                                    .properties(height=250)
                                )
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                st.info("暂无检测时间点，无法绘制时间分布图")

                            # 同时使用 matplotlib 生成一份高分辨率图像，用于导出下载
                            # 将高度适当减小，并收紧上下边距，使图表更紧凑、美观
                            fig, ax = plt.subplots(figsize=(12, 4))
                            # 使用数字索引作为X轴位置
                            x_positions = range(len(time_counts))
                            ax.plot(x_positions, time_counts["缺陷数量"], marker='o', linewidth=2, markersize=6)
                            ax.set_xlabel("时间点", fontsize=12)
                            ax.set_ylabel("缺陷数量", fontsize=12)
                            ax.set_title("缺陷出现时间分布", fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            # 收紧图像上下左右空白区域
                            fig.subplots_adjust(top=0.85, bottom=0.28, left=0.08, right=0.98)
                            
                            # 根据time_interval调整X轴标签显示，确保标签间隔至少为time_interval
                            num_time_points = len(time_counts)
                            
                            # 简化刻度选择：按顺序均匀抽取部分时间点作为刻度，避免依赖检测时的 time_interval
                            max_ticks = 15
                            if num_time_points <= max_ticks:
                                tick_indices = list(range(num_time_points))
                            else:
                                step = max(1, num_time_points // max_ticks)
                                tick_indices = list(range(0, num_time_points, step))
                                if tick_indices[-1] != num_time_points - 1:
                                    tick_indices.append(num_time_points - 1)
                            
                            # 确保最后一个时间点也被显示
                            if num_time_points > 0 and tick_indices[-1] != num_time_points - 1:
                                tick_indices.append(num_time_points - 1)
                            
                            # 使用 all_time_points 作为时间标签来源（精确到0.1秒，不四舍五入）
                            # all_time_points 来自 DataFrame 的"时间点(HH:MM:SS.mmm)"列，由 format_seconds_to_hhmmss_mmm 函数生成
                            tick_labels = [all_time_points[i] for i in tick_indices]
                            tick_positions = tick_indices
                            
                            ax.set_xticks(tick_positions)
                            # X轴标签倾斜45度显示，使用精确到毫秒的不四舍五入时间点
                            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                            plt.tight_layout()
                            
                            # 将图表保存到内存中的字节流用于下载
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            plt.close(fig)
                            
                            # 添加下载按钮
                            st.download_button(
                                label="📥 下载图表图片",
                                data=buf.getvalue(),
                                file_name="缺陷出现时间分布.png",
                                mime="image/png"
                            )
                        else:
                            st.info("暂无检测数据")
                    
                    # 检测关键帧 - 使用expander折叠（与摄像头模式保持一致）
                    if frames:
                        with st.expander("📸 查看检测关键帧", expanded=False):
                            # 获取所有唯一的帧序号及其第一个记录的时间点（用于批量下载和显示）
                            unique_frames = df.drop_duplicates(subset=["帧序号"], keep="first")
                            
                            # 分页设置
                            frames_per_page = 6  # 每页显示6张图片（2行3列）
                            total_pages = (len(frames) + frames_per_page - 1) // frames_per_page
                            # 顶部操作栏：批量下载按钮和分页选择器
                            if len(frames) > 0:
                                # 创建zip文件
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for idx, frame in enumerate(frames):
                                        # 获取文件名信息（使用与图片下方注释相同的逻辑）
                                        frame_row = None
                                        # 方法1：直接从unique_frames按索引获取
                                        if idx < len(unique_frames):
                                            frame_row = unique_frames.iloc[idx]
                                        # 方法2：如果超出unique_frames范围，从df中按顺序获取唯一帧
                                        elif not df.empty:
                                            all_unique_frame_nums = df["帧序号"].drop_duplicates().reset_index(drop=True)
                                            if idx < len(all_unique_frame_nums):
                                                frame_num = all_unique_frame_nums.iloc[idx]
                                                frame_data = df[df["帧序号"] == frame_num]
                                                if not frame_data.empty:
                                                    frame_row = frame_data.iloc[0]
                                        
                                        # 生成文件名（使用与注释相同的格式）
                                        if frame_row is not None:
                                            detection_index = frame_row.get("检测序号", None)
                                            # 将检测序号格式化为6位数字（如000001）
                                            if detection_index is not None and detection_index != "":
                                                try:
                                                    detection_index_str = f"{int(detection_index):06d}"
                                                except (ValueError, TypeError):
                                                    detection_index_str = str(detection_index)
                                            else:
                                                detection_index_str = ""
                                            # 支持新的毫秒格式列名
                                            timestamp = frame_row.get("时间点(HH:MM:SS.mmm)", "") or frame_row.get("时间点(HH:MM:SS)", "")
                                            if not timestamp:
                                                if "绝对时间戳" in frame_row:
                                                    timestamp = format_real_time(frame_row["绝对时间戳"])
                                                else:
                                                    timestamp = ""
                                            frame_num = frame_row["帧序号"]
                                            # 统计该帧的缺陷数量（只统计有缺陷的记录）
                                            defect_count = len(df[(df["帧序号"] == frame_num) & (df["缺陷类别"].notna())])
                                            
                                            # 根据注释格式生成文件名
                                            if detection_index_str and timestamp:
                                                filename_base = f"检测序号_{detection_index_str}_时间点_{timestamp}_{defect_count}个缺陷"
                                            elif detection_index_str:
                                                filename_base = f"检测序号_{detection_index_str}_{defect_count}个缺陷"
                                            elif timestamp:
                                                filename_base = f"时间点_{timestamp}_{defect_count}个缺陷"
                                            else:
                                                filename_base = f"帧序号_{frame_num}_{defect_count}个缺陷"
                                            
                                            # 清理文件名中的非法字符
                                            filename = filename_base.replace(":", "-").replace("/", "-").replace(" ", "_").replace("|", "_") + ".jpg"
                                        else:
                                            # 如果找不到对应记录，使用索引
                                            filename = f"frame_{idx:05d}.jpg"
                                        
                                        # 将PIL图片转换为字节
                                        img_bytes = io.BytesIO()
                                        frame.save(img_bytes, format='JPEG')
                                        img_bytes.seek(0)
                                        zip_file.writestr(filename, img_bytes.read())
                                
                                zip_buffer.seek(0)
                                
                                # 批量下载按钮（左对齐）
                                st.download_button(
                                    label=f"📥 批量下载全部关键帧照片 ({len(frames)} 张)",
                                    data=zip_buffer.getvalue(),
                                    file_name="video_keyframes.zip",
                                    mime="application/zip"
                                )
                                
                                # 分页选择器（在下载按钮下方，左对齐）
                                if total_pages > 1:
                                    # 使用列布局，让选择器左对齐且宽度较小
                                    col_page, _ = st.columns([1, 7])
                                    with col_page:
                                        # selectbox在选项较多时会自动使用滚动模式
                                        page = st.selectbox(
                                            "选择页面",
                                            range(1, total_pages + 1),
                                            key="video_keyframes_page",
                                            format_func=lambda x: f"第 {x} 页（共 {total_pages} 页）"
                                        )
                                else:
                                    page = 1
                                
                                st.markdown("---")  # 添加分隔线
                            
                            # 计算当前页的帧范围
                            start_idx = (page - 1) * frames_per_page
                            end_idx = min(start_idx + frames_per_page, len(frames))
                            current_frames = frames[start_idx:end_idx]
                            
                            # 显示当前页的帧（3列布局）
                            num_cols = 3
                            for i in range(0, len(current_frames), num_cols):
                                cols = st.columns(num_cols)
                                for j, col in enumerate(cols):
                                    if i + j < len(current_frames):
                                        # current_frames中的相对索引
                                        frame_idx_local = i + j
                                        # frames中的全局索引
                                        frame_idx_global = start_idx + frame_idx_local
                                        with col:
                                            st.image(current_frames[frame_idx_local], use_container_width=True)
                                            # 从unique_frames或df中获取对应的时间点信息
                                            # frames列表和unique_frames应该是一一对应的，但为了安全起见，使用多种方法查找
                                            frame_row = None
                                            
                                            # 方法1：直接从unique_frames按索引获取（最直接的方法）
                                            if frame_idx_global < len(unique_frames):
                                                frame_row = unique_frames.iloc[frame_idx_global]
                                            # 方法2：如果超出unique_frames范围，从df中按顺序获取唯一帧
                                            elif not df.empty:
                                                # 获取所有唯一的帧序号，按顺序排列（与frames列表对应）
                                                all_unique_frame_nums = df["帧序号"].drop_duplicates().reset_index(drop=True)
                                                # 确保索引在有效范围内
                                                if frame_idx_global < len(all_unique_frame_nums):
                                                    frame_num = all_unique_frame_nums.iloc[frame_idx_global]
                                                    frame_data = df[df["帧序号"] == frame_num]
                                                    if not frame_data.empty:
                                                        frame_row = frame_data.iloc[0]
                                            
                                            # 显示图片下方的注释信息（与摄像头模式保持一致）
                                            if frame_row is not None:
                                                # 获取检测序号、时间点和缺陷数量
                                                detection_index = frame_row.get("检测序号", None)
                                                # 将检测序号格式化为6位数字（如000001）
                                                if detection_index is not None and detection_index != "":
                                                    try:
                                                        detection_index_str = f"{int(detection_index):06d}"
                                                    except (ValueError, TypeError):
                                                        detection_index_str = str(detection_index)
                                                else:
                                                    detection_index_str = ""
                                                # 支持新的毫秒格式列名
                                                timestamp = frame_row.get("时间点(HH:MM:SS.mmm)", "") or frame_row.get("时间点(HH:MM:SS)", "")
                                                if not timestamp:
                                                    # 如果没有时间点格式，尝试从绝对时间戳生成
                                                    if "绝对时间戳" in frame_row:
                                                        timestamp = format_real_time(frame_row["绝对时间戳"])
                                                    else:
                                                        timestamp = ""
                                                frame_num = frame_row.get("帧序号", "")
                                                # 统计该帧的缺陷数量（只统计有缺陷的记录）
                                                defect_count = len(df[(df["帧序号"] == frame_num) & (df["缺陷类别"].notna())]) if frame_num != "" else 0
                                                
                                                # 显示检测序号、时间点和缺陷数量（与摄像头模式格式一致）
                                                if detection_index_str and timestamp:
                                                    st.caption(f"检测序号: {detection_index_str} | 时间点: {timestamp} | {defect_count} 个缺陷")
                                                elif detection_index_str:
                                                    st.caption(f"检测序号: {detection_index_str} | {defect_count} 个缺陷")
                                                elif timestamp:
                                                    st.caption(f"时间点: {timestamp} | {defect_count} 个缺陷")
                                                else:
                                                    st.caption(f"帧序号: {frame_num} | {defect_count} 个缺陷")
                                            else:
                                                # 如果找不到对应记录，至少显示帧索引和缺陷数量
                                                defect_count = len(df) if not df.empty else 0
                                                st.caption(f"帧索引: {frame_idx_global} | {defect_count} 个缺陷")
                else:
                    # 只有在检测完成后（有video_info）且确实没有检测到缺陷时才显示
                    if video_info:
                        st.info("ℹ️ 视频中未检测到缺陷。可以尝试降低置信度阈值。")
    
    else:  # 摄像头实时检测
        st.subheader("📹 摄像头实时检测")
        
        if not WEBRTC_AVAILABLE:
            st.error("""
            ❌ **缺少 streamlit-webrtc 组件**
            
            请安装 streamlit-webrtc 以启用摄像头实时检测功能：
            ```bash
            pip install streamlit-webrtc av
            ```
            
            安装完成后，请重启 Streamlit 应用。
            """)
        else:
            st.info("💡 请确保摄像头已连接并授权访问权限。点击下方视频组件上的 START 按钮开始实时检测。")
        
        # 当前检测参数
        model_path_str = str(model_path) if model_path else "none"
        current_camera_params = {
            'model_path': model_path_str,
            'camera_index': camera_index,
            'conf': conf,
            'iou': iou,
            'time_interval': time_interval,
            'device': device,
            'max_frames': max_frames
        }
        
        # 初始化检测结果存储
        if 'camera_detection_results' not in st.session_state:
            st.session_state.camera_detection_results = {
                'frames': [],
                'records': [],
                'frame_count': 0,
                'detection_count': 0,
                'actual_detection_count': 0,
                'start_time': time.time(),
                'last_detect_time': time.time()
            }
        
        # 将当前参数保存到session_state，供视频处理器实时读取
        st.session_state.camera_conf = conf
        st.session_state.camera_iou = iou
        st.session_state.camera_time_interval = time_interval
        st.session_state.camera_device = device
        
        # 初始化停止状态
        if 'stop_camera' not in st.session_state:
            st.session_state.stop_camera = False
        
        # 初始化摄像头播放状态跟踪（用于检测状态变化并记录日志）
        if 'camera_was_playing' not in st.session_state:
            st.session_state.camera_was_playing = False
        
        # 注意：停止检测按钮已移除，使用 webrtc_streamer 自带的 STOP 按钮
        # 清空检测结果按钮放在 webrtc_streamer 组件下方
        
        # 初始化摄像头检测历史（参考 yolo_streamlit_cloud_mini_project_test）
        if 'camera_history' not in st.session_state:
            st.session_state.camera_history = {
                'current_objects': [],
                'all_detections': [],
                'frame_count': 0,
                'start_time': None,
                'end_time': None,
                'class_counts': Counter(),
            }
        
        if WEBRTC_AVAILABLE:
            # 创建视频帧回调函数
            video_callback = create_defect_detection_callback(
                model=model,
                conf=conf,
                iou=iou,
                device=device,
                time_interval=time_interval
            )
            
            # 获取 ICE 服务器配置
            ice_servers = get_ice_servers()
            
            # 主布局：视频和统计信息
            col_video, col_stats = st.columns([3, 2])
            
            with col_video:
                st.markdown("##### 📹 实时视频流")
                
                # WebRTC 配置 - 使用 video_frame_callback 参数（官方推荐方式）
                ctx = webrtc_streamer(
                    key="defect-detection",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_callback,
                    rtc_configuration={"iceServers": ice_servers},
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # 清空检测结果按钮（放在 START 按钮下方，样式一致）
                if st.button("🗑️ 清空检测结果", type="primary", use_container_width=True, key="clear_camera_results_main"):
                    # 重置全局结果容器
                    reset_camera_result_container()
                    # 重置 session_state
                    if 'camera_detection_results' in st.session_state:
                        st.session_state.camera_detection_results = {
                            'frames': [],
                            'records': [],
                            'frame_count': 0,
                            'detection_count': 0,
                            'actual_detection_count': 0,
                            'start_time': None,
                            'end_time': None,
                            'class_counts': Counter(),
                            'all_detections': [],
                        }
                    if 'camera_history' in st.session_state:
                        st.session_state.camera_history = {
                            'current_objects': [],
                            'all_detections': [],
                            'frame_count': 0,
                            'start_time': None,
                            'end_time': None,
                            'class_counts': Counter(),
                        }
                    st.session_state.stop_camera = False  # 重置停止标志
                    st.rerun()
            
            with col_stats:
                st.markdown("##### 📊 实时检测统计")
                
                # 创建占位符用于动态更新
                status_placeholder = st.empty()
                stats_placeholder = st.empty()
            
            # 实时详细结果区域（在视频和统计列下方，全宽显示）
            realtime_table_placeholder = st.empty()
            realtime_chart_placeholder = st.empty()
            
            # 当视频正在播放时，使用循环持续更新统计
            if ctx.state.playing:
                # 标记摄像头正在播放
                st.session_state.camera_was_playing = True
                
                with col_stats:
                    status_placeholder.success("🟢 摄像头已连接，正在检测...")
                
                # 记录开始时间
                if st.session_state.camera_history['start_time'] is None:
                    st.session_state.camera_history['start_time'] = datetime.now()
                
                # 使用循环持续更新统计信息（参考 yolo_streamlit_cloud_mini_project_test）
                while ctx.state.playing:
                    # 从全局容器读取数据（线程安全）
                    with camera_lock:
                        objects = camera_result_container["objects"].copy()
                        current_defect_count = camera_result_container.get("current_defect_count", 0)
                        frame_count = camera_result_container["frame_count"]
                        detection_count = camera_result_container["detection_count"]
                        start_time_container = camera_result_container["start_time"]
                        frames_container = camera_result_container.get("frames", []).copy()
                        records_container = camera_result_container.get("records", []).copy()
                    
                    # 累积保存检测结果
                    if objects:
                        st.session_state.camera_history['current_objects'] = objects
                        st.session_state.camera_history['frame_count'] = frame_count
                        
                        # 累积所有检测结果
                        st.session_state.camera_history['all_detections'].extend(objects)
                        
                        # 更新类别计数
                        for obj in objects:
                            st.session_state.camera_history['class_counts'][obj["class"]] += 1
                    
                    # 同步数据到 camera_detection_results（供后面的详细结果展示使用）
                    st.session_state.camera_detection_results = {
                        'frames': frames_container,
                        'records': records_container,
                        'frame_count': frame_count,
                        'detection_count': detection_count,
                        'actual_detection_count': detection_count,
                        'start_time': start_time_container if start_time_container else time.time(),
                        'last_detect_time': time.time()
                    }
                    
                    # 渲染实时统计（右侧列）- 只显示检测时长、当前帧缺陷数及置信度（表格形式）
                    with stats_placeholder.container():
                        # 计算检测时长
                        if start_time_container:
                            elapsed_time = time.time() - start_time_container
                            if elapsed_time >= 60:
                                minutes = int(elapsed_time // 60)
                                seconds = int(elapsed_time % 60)
                                time_str = f"{minutes}分{seconds}秒"
                            else:
                                time_str = f"{int(elapsed_time)}秒"
                        else:
                            time_str = "0秒"
                        
                        # 检测时长
                        st.metric("⏱️ 检测时长", time_str)
                        
                        # 当前帧缺陷数及置信度（表格形式，带序号）
                        st.markdown("**🎯 当前帧检测结果**")
                        if current_defect_count > 0 and objects:
                            df_current = pd.DataFrame([
                                {"序号": i + 1, "缺陷类别": obj["class"], "置信度": f"{obj['confidence']:.2%}"}
                                for i, obj in enumerate(objects)
                            ])
                            st.dataframe(df_current, use_container_width=True, hide_index=True)
                        else:
                            st.info("无缺陷")
                    
                    # 渲染实时详细结果表格（下方全宽）
                    with realtime_table_placeholder.container():
                        if records_container and len(records_container) > 0:
                            st.markdown("#### 📋 实时检测结果详情")
                            try:
                                current_df = pd.concat(records_container, ignore_index=True)
                                # 显示表格（去掉绝对时间戳列，格式化检测序号）
                                df_display = current_df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in current_df.columns else current_df.copy()
                                if "检测序号" in df_display.columns:
                                    df_display = df_display.copy()
                                    df_display["检测序号"] = df_display["检测序号"].apply(format_detection_index)
                                st.dataframe(df_display, use_container_width=True, height=250)
                            except Exception:
                                st.info("等待检测数据...")
                        else:
                            st.info("📋 等待检测数据...")
                    
                    # 渲染实时统计图表（下方全宽）- 显示缺陷时间分布（包括缺陷数为0的时间点）
                    with realtime_chart_placeholder.container():
                        if records_container and len(records_container) > 0:
                            st.markdown("#### 📊 缺陷时间分布")
                            try:
                                current_df = pd.concat(records_container, ignore_index=True)
                                
                                if "时间点(HH:MM:SS.mmm)" in current_df.columns and "缺陷数" in current_df.columns:
                                    # 按时间点和检测序号分组，取每次检测的缺陷数（使用最大值，因为同一检测序号的缺陷数相同）
                                    time_defect_df = current_df.groupby(["时间点(HH:MM:SS.mmm)", "检测序号"]).agg({
                                        "缺陷数": "max"
                                    }).reset_index()
                                    
                                    # 按时间点排序
                                    time_defect_df = time_defect_df.sort_values("检测序号")
                                    
                                    if len(time_defect_df) > 0:
                                        time_counts = pd.DataFrame({
                                            "时间点": time_defect_df["时间点(HH:MM:SS.mmm)"].values,
                                            "缺陷数量": time_defect_df["缺陷数"].values,
                                        })
                                        chart = (
                                            alt.Chart(time_counts)
                                            .mark_line(point=True)
                                            .encode(
                                                x=alt.X(field="时间点", type="nominal", sort=None, title="时间点"),
                                                y=alt.Y(field="缺陷数量", type="quantitative", title="缺陷数量"),
                                            )
                                            .properties(height=250)
                                        )
                                        st.altair_chart(chart, use_container_width=True)
                                    else:
                                        st.info("等待更多数据...")
                                else:
                                    st.info("等待时间数据...")
                            except Exception:
                                st.info("等待检测数据...")
                        else:
                            st.info("📊 等待检测数据生成图表...")
                    
                    # 短暂休眠，避免过于频繁的更新
                    time.sleep(0.5)
                
                # 循环结束，记录结束时间
                st.session_state.camera_history['end_time'] = datetime.now()
                
                # 最终同步数据
                with camera_lock:
                    frames_final = camera_result_container.get("frames", []).copy()
                    records_final = camera_result_container.get("records", []).copy()
                    frame_count_final = camera_result_container["frame_count"]
                    detection_count_final = camera_result_container["detection_count"]
                    start_time_final = camera_result_container["start_time"]
                
                st.session_state.camera_detection_results = {
                    'frames': frames_final,
                    'records': records_final,
                    'frame_count': frame_count_final,
                    'detection_count': detection_count_final,
                    'actual_detection_count': detection_count_final,
                    'start_time': start_time_final if start_time_final else time.time(),
                    'last_detect_time': time.time()
                }
                
            else:
                # 检测状态变化：从播放变为停止时记录日志
                if st.session_state.camera_was_playing:
                    st.session_state.camera_was_playing = False
                    # 记录参数到日志
                    log_camera_sidebar_parameters(current_camera_params)
                
                with col_stats:
                    # 从 session_state 读取保存的历史结果
                    history = st.session_state.camera_history
                    
                    with stats_placeholder.container():
                        if history['all_detections']:
                            # 显示汇总统计：只显示检测时长、总缺陷数、平均置信度
                            all_detections = history['all_detections']
                            
                            # 计算检测时长 - 从 camera_detection_results 获取时间信息
                            detection_results = st.session_state.get('camera_detection_results', {})
                            start_time_ts = detection_results.get('start_time')
                            end_time_ts = detection_results.get('last_detect_time')
                            
                            if start_time_ts and end_time_ts:
                                duration = end_time_ts - start_time_ts
                                if duration >= 60:
                                    minutes = int(duration // 60)
                                    seconds = int(duration % 60)
                                    duration_str = f"{minutes}分{seconds}秒"
                                else:
                                    duration_str = f"{duration:.1f}秒"
                            else:
                                duration_str = "0秒"
                            
                            # 计算平均置信度
                            if all_detections:
                                confidences = [obj.get('confidence', 0) for obj in all_detections]
                                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                            else:
                                avg_confidence = 0
                            
                            # 检测时长
                            st.metric("⏱️ 检测时长", duration_str)
                            # 总缺陷数
                            st.metric("🎯 总缺陷数", len(all_detections))
                            # 平均置信度
                            st.metric("📊 平均置信度", f"{avg_confidence:.2%}")
        
        # 从session_state获取完整的检测结果（如果检测还在进行中或已完成，显示当前结果）
        df = pd.DataFrame()
        frames = []
        if 'camera_detection_results' in st.session_state and st.session_state.camera_detection_results.get('records'):
            current_records = st.session_state.camera_detection_results['records']
            if current_records:
                df_all = pd.concat(current_records, ignore_index=True)
                # 由于检测已经基于time_interval进行，所有记录都符合时间间隔，直接使用所有记录
                df = df_all.copy()
                frames = st.session_state.camera_detection_results.get('frames', [])
        
        if not df.empty:
            # 使用expander折叠详细信息
            with st.expander("📋 查看详细检测结果", expanded=False):
                # 显示详细表格（去掉绝对时间戳列，格式化检测序号）
                df_display = df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in df.columns else df.copy()
                # 将检测序号格式化为6位数字（如000001）
                if "检测序号" in df_display.columns:
                    df_display = df_display.copy()  # 避免SettingWithCopyWarning
                    df_display["检测序号"] = df_display["检测序号"].apply(format_detection_index)
                st.dataframe(df_display, use_container_width=True, height=250)
                
                # 下载按钮（下载时也去掉绝对时间戳列，格式化检测序号）
                df_download = df.drop(columns=["绝对时间戳"]) if "绝对时间戳" in df.columns else df.copy()
                # 将检测序号格式化为6位数字（如000001）
                if "检测序号" in df_download.columns:
                    df_download = df_download.copy()  # 避免SettingWithCopyWarning
                    df_download["检测序号"] = df_download["检测序号"].apply(format_detection_index)
                csv = df_download.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载检测结果 (CSV)",
                    data=csv,
                    file_name="camera_detection_results.csv",
                    mime="text/csv",
                )
                    
            # 统计图表 - 使用expander折叠，独立于详细检测结果（与视频检测保持一致）
            with st.expander("📊 查看统计图表", expanded=False):
                st.markdown("#### ⏱️ 缺陷出现时间分布")
                # 按时间点统计缺陷数量（使用摄像头实际时间 HH:MM:SS.mmm 作为X轴）
                # 需要包含所有检测时间点，即使缺陷数为0
                if "时间点(HH:MM:SS.mmm)" in df.columns:
                    all_time_points = (
                        df["时间点(HH:MM:SS.mmm)"]
                        .astype(str)
                        .dropna()
                        .sort_values()
                        .reset_index(drop=True)
                        .unique()
                    )

                    if len(all_time_points) > 0:
                        defect_df = df[df["缺陷类别"].notna()].copy()
                        if not defect_df.empty:
                            defect_counts = (
                                defect_df.groupby("时间点(HH:MM:SS.mmm)")
                                .size()
                                .reindex(all_time_points, fill_value=0)
                            )
                        else:
                            defect_counts = pd.Series(
                                [0] * len(all_time_points),
                                index=all_time_points,
                            )

                        # 为绘图单独建一列简单字段名“时间点”
                        time_counts = pd.DataFrame({
                            "时间点": all_time_points,
                            "缺陷数量": defect_counts.values,
                        })

                        # 使用 Altair 绘制，明确指定 X/Y，避免 Streamlit 的索引聚合问题
                        chart = (
                            alt.Chart(time_counts)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X(field="时间点", type="nominal", sort=None, title="时间点"),
                                y=alt.Y(field="缺陷数量", type="quantitative", title="缺陷数量"),
                            )
                            .properties(height=250)
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("暂无检测时间点，无法绘制时间分布图")

                    # 同时使用 matplotlib 生成一份高分辨率图像，用于导出下载
                    # 将高度适当减小，并收紧上下边距，使图表更紧凑、美观
                    fig, ax = plt.subplots(figsize=(12, 4))
                    # 使用数字索引作为X轴位置
                    x_positions = range(len(time_counts))
                    ax.plot(x_positions, time_counts["缺陷数量"], marker='o', linewidth=2, markersize=6)
                    ax.set_xlabel("时间点", fontsize=12)
                    ax.set_ylabel("缺陷数量", fontsize=12)
                    ax.set_title("缺陷出现时间分布", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    # 收紧图像上下左右空白区域
                    fig.subplots_adjust(top=0.85, bottom=0.28, left=0.08, right=0.98)
                    
                    # 智能调整X轴标签显示，避免重叠
                    num_time_points = len(time_counts)
                    # 如果时间点较多，只显示部分标签（最多显示15个）
                    if num_time_points > 15:
                        step = max(1, num_time_points // 15)
                        tick_indices = list(range(0, num_time_points, step))
                        # 确保最后一个时间点也被显示
                        if tick_indices[-1] != num_time_points - 1:
                            tick_indices.append(num_time_points - 1)
                        # 使用 all_time_points 作为时间标签来源（精确到0.1秒，不四舍五入）
                        # all_time_points 来自 DataFrame 的"时间点(HH:MM:SS.mmm)"列，由 format_seconds_to_hhmmss_mmm 函数生成
                        tick_labels = [all_time_points[i] for i in tick_indices]
                        tick_positions = tick_indices
                    else:
                        # 如果时间点不多，显示所有标签（精确到0.1秒，不四舍五入）
                        tick_labels = list(all_time_points)
                        tick_positions = range(num_time_points)
                    
                    ax.set_xticks(tick_positions)
                    # X轴标签倾斜45度显示，使用精确到毫秒的不四舍五入时间点
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
                    plt.tight_layout()
                    
                    # 将图表保存到内存中的字节流用于下载
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    
                    # 添加下载按钮
                    st.download_button(
                        label="📥 下载图表图片",
                        data=buf.getvalue(),
                        file_name="缺陷出现时间分布.png",
                        mime="image/png"
                    )
                else:
                    st.info("暂无检测数据")
            
            # 检测关键帧 - 使用expander折叠
            if frames:
                with st.expander("📸 查看检测关键帧", expanded=False):
                    # 获取所有唯一的帧序号及其第一个记录的时间点（用于批量下载和显示）
                    unique_frames = df.drop_duplicates(subset=["帧序号"], keep="first")
                    
                    # 分页设置
                    frames_per_page = 6  # 每页显示6张图片（2行3列）
                    total_pages = (len(frames) + frames_per_page - 1) // frames_per_page
                    
                    # 顶部操作栏：批量下载按钮和分页选择器
                    if len(frames) > 0:
                        # 创建zip文件
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, frame in enumerate(frames):
                                # 获取文件名信息（使用与图片下方注释相同的逻辑）
                                frame_row = None
                                # 方法1：直接从unique_frames按索引获取
                                if idx < len(unique_frames):
                                    frame_row = unique_frames.iloc[idx]
                                # 方法2：如果超出unique_frames范围，从df中按顺序获取唯一帧
                                elif not df.empty:
                                    all_unique_frame_nums = df["帧序号"].drop_duplicates().reset_index(drop=True)
                                    if idx < len(all_unique_frame_nums):
                                        frame_num = all_unique_frame_nums.iloc[idx]
                                        frame_data = df[df["帧序号"] == frame_num]
                                        if not frame_data.empty:
                                            frame_row = frame_data.iloc[0]
                                
                                # 生成文件名（使用与注释相同的格式）
                                if frame_row is not None:
                                    detection_index = frame_row.get("检测序号", None)
                                    # 将检测序号格式化为6位数字（如000001）
                                    if detection_index is not None and detection_index != "":
                                        try:
                                            detection_index_str = f"{int(detection_index):06d}"
                                        except (ValueError, TypeError):
                                            detection_index_str = str(detection_index)
                                    else:
                                        detection_index_str = ""
                                    timestamp = frame_row.get("时间点(HH:MM:SS)", "")
                                    if not timestamp:
                                        if "绝对时间戳" in frame_row:
                                            timestamp = format_real_time(frame_row["绝对时间戳"])
                                        else:
                                            timestamp = ""
                                    frame_num = frame_row["帧序号"]
                                    # 统计该帧的缺陷数量（只统计有缺陷的记录）
                                    defect_count = len(df[(df["帧序号"] == frame_num) & (df["缺陷类别"].notna())])
                                    
                                    # 根据注释格式生成文件名
                                    if detection_index_str and timestamp:
                                        filename_base = f"检测序号_{detection_index_str}_时间点_{timestamp}_{defect_count}个缺陷"
                                    elif detection_index_str:
                                        filename_base = f"检测序号_{detection_index_str}_{defect_count}个缺陷"
                                    elif timestamp:
                                        filename_base = f"时间点_{timestamp}_{defect_count}个缺陷"
                                    else:
                                        filename_base = f"帧序号_{frame_num}_{defect_count}个缺陷"
                                    
                                    # 清理文件名中的非法字符
                                    filename = filename_base.replace(":", "-").replace("/", "-").replace(" ", "_").replace("|", "_") + ".jpg"
                                else:
                                    # 如果找不到对应记录，使用索引
                                    filename = f"frame_{idx:05d}.jpg"
                                
                                # 将PIL图片转换为字节
                                img_bytes = io.BytesIO()
                                frame.save(img_bytes, format='JPEG')
                                img_bytes.seek(0)
                                zip_file.writestr(filename, img_bytes.read())
                        
                        zip_buffer.seek(0)
                        
                        # 批量下载按钮（左对齐）
                        st.download_button(
                            label=f"📥 批量下载全部关键帧照片 ({len(frames)} 张)",
                            data=zip_buffer.getvalue(),
                            file_name="camera_keyframes.zip",
                            mime="application/zip"
                        )
                        
                        # 分页选择器（在下载按钮下方，左对齐）
                        if total_pages > 1:
                            # 使用列布局，让选择器左对齐且宽度较小
                            col_page, _ = st.columns([1, 7])
                            with col_page:
                                # selectbox在选项较多时会自动使用滚动模式
                                page = st.selectbox(
                                    "选择页面",
                                    range(1, total_pages + 1),
                                    key="camera_keyframes_page",
                                    format_func=lambda x: f"第 {x} 页（共 {total_pages} 页）"
                                )
                        else:
                            page = 1
                        
                        st.markdown("---")  # 添加分隔线
                    
                    # 计算当前页的帧范围
                    start_idx = (page - 1) * frames_per_page
                    end_idx = min(start_idx + frames_per_page, len(frames))
                    current_frames = frames[start_idx:end_idx]
                    
                    # 显示当前页的帧（3列布局）
                    num_cols = 3
                    for i in range(0, len(current_frames), num_cols):
                        cols = st.columns(num_cols)
                        for j, col in enumerate(cols):
                            if i + j < len(current_frames):
                                # current_frames中的相对索引
                                frame_idx_local = i + j
                                # frames中的全局索引
                                frame_idx_global = start_idx + frame_idx_local
                                with col:
                                    st.image(current_frames[frame_idx_local], use_container_width=True)
                                    # 从unique_frames或df中获取对应的时间点信息
                                    # frames列表和unique_frames应该是一一对应的，但为了安全起见，使用多种方法查找
                                    frame_row = None
                                    
                                    # 方法1：直接从unique_frames按索引获取（最直接的方法）
                                    if frame_idx_global < len(unique_frames):
                                        frame_row = unique_frames.iloc[frame_idx_global]
                                    # 方法2：如果超出unique_frames范围，从df中按顺序获取唯一帧
                                    elif not df.empty:
                                        # 获取所有唯一的帧序号，按顺序排列（与frames列表对应）
                                        all_unique_frame_nums = df["帧序号"].drop_duplicates().reset_index(drop=True)
                                        # 确保索引在有效范围内
                                        if frame_idx_global < len(all_unique_frame_nums):
                                            frame_num = all_unique_frame_nums.iloc[frame_idx_global]
                                        elif len(all_unique_frame_nums) > 0:
                                            # 如果超出范围，使用最后一个唯一帧序号（可能是最后一帧）
                                            frame_num = all_unique_frame_nums.iloc[-1]
                                        else:
                                            frame_num = None
                                        
                                        if frame_num is not None:
                                            frame_data = df[df["帧序号"] == frame_num]
                                            if not frame_data.empty:
                                                frame_row = frame_data.iloc[0]
                                    
                                    if frame_row is not None:
                                        # 获取检测序号、时间点和缺陷数量
                                        detection_index = frame_row.get("检测序号", None)
                                        # 将检测序号格式化为6位数字（如000001）
                                        if detection_index is not None and detection_index != "":
                                            try:
                                                detection_index_str = f"{int(detection_index):06d}"
                                            except (ValueError, TypeError):
                                                detection_index_str = str(detection_index)
                                        else:
                                            detection_index_str = ""
                                        # 优先使用精确到毫秒的时间点(HH:MM:SS.mmm)
                                        timestamp = frame_row.get("时间点(HH:MM:SS.mmm)", "")
                                        if not timestamp:
                                            # 回退到 HH:MM:SS 或绝对时间戳
                                            timestamp = frame_row.get("时间点(HH:MM:SS)", "")
                                            if not timestamp and "绝对时间戳" in frame_row:
                                                timestamp = format_real_time(frame_row["绝对时间戳"])
                                        frame_num = frame_row["帧序号"]
                                        # 统计该帧的缺陷数量（只统计有缺陷的记录）
                                        defect_count = len(df[(df["帧序号"] == frame_num) & (df["缺陷类别"].notna())])
                                        
                                        # 显示检测序号、时间点和缺陷数量
                                        if detection_index_str and timestamp:
                                            st.caption(f"检测序号: {detection_index_str} | 时间点: {timestamp} | {defect_count} 个缺陷")
                                        elif detection_index_str:
                                            st.caption(f"检测序号: {detection_index_str} | {defect_count} 个缺陷")
                                        elif timestamp:
                                            st.caption(f"时间点: {timestamp} | {defect_count} 个缺陷")
                                        else:
                                            st.caption(f"帧序号: {frame_num} | {defect_count} 个缺陷")
                                    else:
                                        # 如果找不到对应记录，至少显示帧索引和缺陷数量
                                        defect_count = len(df) if not df.empty else 0
                                        st.caption(f"帧索引: {frame_idx_global} | {defect_count} 个缺陷")
            else:
                if 'camera_detection_results' not in st.session_state or not st.session_state.camera_detection_results.get('records'):
                    st.info("ℹ️ 未检测到缺陷。")


if __name__ == "__main__":
    main()
    
