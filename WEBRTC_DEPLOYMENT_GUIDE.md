# WebRTC 在 Streamlit Cloud 部署指南

本文档详细说明该项目中 WebRTC 在 Streamlit Cloud 部署的应用方法。

## 目录
1. [核心组件](#核心组件)
2. [WebRTC 配置](#webrtc-配置)
3. [TURN 服务器设置](#turn-服务器设置)
4. [视频流处理](#视频流处理)
5. [性能优化](#性能优化)
6. [部署步骤](#部署步骤)

---

## 核心组件

### 1. streamlit-webrtc 库
项目使用 `streamlit-webrtc` 库来实现浏览器中的实时视频流处理。

**主要功能：**
- 从用户摄像头捕获视频流
- 实时处理视频帧
- 将处理后的视频流返回给浏览器

### 2. 关键文件说明

#### `input.py` - WebRTC 实现
```python
from streamlit_webrtc import webrtc_streamer
from turn import get_ice_servers
from streamlit_session_memo import st_session_memo
```

#### `turn.py` - TURN 服务器配置
负责获取 ICE 服务器配置，确保 WebRTC 连接在 Streamlit Cloud 上正常工作。

---

## WebRTC 配置

### 1. webrtc_streamer 配置

在 `input.py` 的 `webcam_input()` 函数中：

```python
ctx = webrtc_streamer(
    key="neural-style-transfer",  # 唯一标识符
    video_frame_callback=video_frame_callback,  # 视频帧处理回调函数
    rtc_configuration={"iceServers": get_ice_servers()},  # ICE 服务器配置
    media_stream_constraints={"video": True, "audio": False},  # 媒体流约束
)
```

**参数说明：**
- `key`: 唯一标识符，用于区分不同的 WebRTC 流
- `video_frame_callback`: 处理每一帧视频的回调函数
- `rtc_configuration`: RTC 配置，包含 ICE 服务器信息
- `media_stream_constraints`: 指定需要视频流，不需要音频流

### 2. ICE 服务器配置

ICE (Interactive Connectivity Establishment) 服务器用于建立 WebRTC 连接。

**为什么需要 TURN 服务器？**
- Streamlit Community Cloud 的基础设施变化
- 某些网络环境下，STUN 服务器无法建立直接连接
- TURN 服务器作为中继，确保连接成功

---

## TURN 服务器设置

### 1. Twilio TURN 服务器（推荐）

项目使用 Twilio 的 TURN 服务器，因为：
- 稳定可靠
- 专门为 WebRTC 设计
- 在 Streamlit Cloud 上经过验证

### 2. 配置方法

#### 步骤 1: 获取 Twilio 凭证

1. 注册 Twilio 账号：https://www.twilio.com/
2. 获取 Account SID 和 Auth Token
3. 在 Streamlit Cloud 设置环境变量：
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`

#### 步骤 2: 代码实现

`turn.py` 中的实现：

```python
@st.cache_data
def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        # 如果没有设置 Twilio 凭证，回退到免费的 Google STUN 服务器
        logger.warning("Twilio credentials are not set. Fallback to a free STUN server from Google.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers
```

**关键点：**
- 使用 `@st.cache_data` 装饰器缓存 ICE 服务器配置
- 如果没有 Twilio 凭证，回退到免费的 Google STUN 服务器
- 使用 Twilio API 动态获取 ICE 服务器列表

### 3. 免费替代方案

如果不想使用 Twilio，可以使用：
- Google STUN 服务器：`stun:stun.l.google.com:19302`
- 注意：STUN 服务器在某些网络环境下可能无法工作

---

## 视频流处理

### 1. 视频帧回调函数

```python
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 1. 将 VideoFrame 转换为 numpy 数组
    image = frame.to_ndarray(format="bgr24")
    
    # 2. 检查模型是否加载
    if model is None:
        return image
    
    # 3. 获取原始尺寸
    orig_h, orig_w = image.shape[0:2]
    
    # 4. 调整图像大小（使用 PIL 避免内存泄漏）
    input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))
    
    # 5. 应用风格迁移
    transferred = style_transfer(input, model)
    
    # 6. 转换回 VideoFrame
    result = Image.fromarray((transferred * 255).astype(np.uint8))
    image = np.asarray(result.resize((orig_w, orig_h)))
    return av.VideoFrame.from_ndarray(image, format="bgr24")
```

**重要注意事项：**
- 使用 `Image.fromarray()` 和 `PIL.resize()` 而不是 `cv2.resize()`，避免在分叉线程中导致内存泄漏
- 保持原始宽高比
- 确保返回的格式与输入格式一致（bgr24）

### 2. 视频格式

- **输入格式**: `bgr24` (OpenCV 默认格式)
- **输出格式**: `bgr24`
- **音频**: 禁用 (`audio: False`)

---

## 性能优化

### 1. 模型缓存

使用 `streamlit-session-memo` 缓存模型加载：

```python
from streamlit_session_memo import st_session_memo

@st_session_memo
def load_model(model_name, width):
    return get_model_from_path(model_name)

model = load_model(style_models_dict[style_model_name], width)
```

**优势：**
- 避免重复加载模型
- 提高响应速度
- 减少内存使用

### 2. 图像质量调整

允许用户调整处理质量：

```python
WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', 
                                  list(range(150, 501, 50)))
```

**权衡：**
- 更高的质量 = 更慢的处理速度
- 更低的质量 = 更快的处理速度

### 3. 缓存 ICE 服务器配置

使用 `@st.cache_data` 缓存 ICE 服务器配置，避免重复 API 调用。

---

## 部署步骤

### 1. 准备文件

确保以下文件存在：
- `app.py` - 主应用文件
- `input.py` - WebRTC 输入处理
- `turn.py` - TURN 服务器配置
- `requirements.txt` - 依赖列表
- `neural_style_transfer.py` - 风格迁移实现
- `data.py` - 数据配置
- `models/` - 模型文件目录
- `images/` - 示例图片目录

### 2. 配置 requirements.txt

```txt
imutils==0.5.3
numpy<2.0
streamlit~=1.21.0
opencv-python-headless==4.6.0.66
streamlit-webrtc~=0.45.0
twilio~=8.1.0
streamlit-session-memo~=0.3.1
```

### 3. 设置环境变量

在 Streamlit Cloud 中设置：
- `TWILIO_ACCOUNT_SID`: 你的 Twilio Account SID
- `TWILIO_AUTH_TOKEN`: 你的 Twilio Auth Token

### 4. 部署到 Streamlit Cloud

1. 将代码推送到 GitHub 仓库
2. 在 Streamlit Cloud 中连接仓库
3. 设置环境变量
4. 部署应用

### 5. 验证部署

部署后检查：
- WebRTC 连接是否成功建立
- 视频流是否正常显示
- 风格迁移是否正常工作

---

## 常见问题

### Q1: WebRTC 连接失败怎么办？

**A:** 检查以下几点：
1. 是否设置了 Twilio 环境变量
2. 网络是否允许 WebRTC 连接
3. 浏览器是否支持 WebRTC

### Q2: 视频处理很慢怎么办？

**A:** 
1. 降低图像质量（WIDTH 参数）
2. 使用更轻量的模型
3. 优化 `video_frame_callback` 函数

### Q3: 内存泄漏问题？

**A:** 
- 使用 `PIL.Image.resize()` 而不是 `cv2.resize()`
- 确保及时释放不需要的资源

### Q4: 在本地可以运行，但部署后不行？

**A:**
- 检查环境变量是否正确设置
- 确认所有依赖都已安装
- 检查 Streamlit Cloud 的日志

---

## 参考资源

- [streamlit-webrtc 文档](https://github.com/whitphx/streamlit-webrtc)
- [Twilio TURN API 文档](https://www.twilio.com/docs/stun-turn/api)
- [WebRTC 官方文档](https://webrtc.org/)
- [Streamlit Cloud 文档](https://docs.streamlit.io/streamlit-community-cloud)

---

## 总结

该项目展示了如何在 Streamlit Cloud 上部署使用 WebRTC 的实时视频处理应用：

1. **使用 streamlit-webrtc** 实现浏览器视频流
2. **配置 Twilio TURN 服务器** 确保连接稳定
3. **优化视频处理** 避免内存泄漏和性能问题
4. **缓存模型和配置** 提高响应速度

通过这些方法，可以在 Streamlit Cloud 上成功部署实时视频处理应用。

