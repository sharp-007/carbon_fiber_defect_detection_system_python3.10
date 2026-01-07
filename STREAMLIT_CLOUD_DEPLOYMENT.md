# Streamlit Cloud 部署指南

本文档说明如何在 Streamlit Cloud 上部署碳纤维缺陷检测系统，并确保 WebRTC 摄像头模式正常工作。

## 📋 部署前准备

### 1. 必需文件

确保以下文件存在于项目根目录：

- ✅ `app.py` - 主应用文件
- ✅ `requirements.txt` - Python 依赖
- ✅ `packages.txt` - 系统依赖（Linux）
- ✅ `turn.py` - TURN 服务器配置模块
- ✅ `model/best.pt` - YOLO 模型文件（或您训练的模型）

### 2. 环境变量配置（可选但推荐）

为了在 Streamlit Cloud 上获得更好的 WebRTC 连接稳定性，建议配置 Twilio TURN 服务器：

#### 步骤 1: 注册 Twilio 账号

1. 访问 https://www.twilio.com/
2. 注册账号并登录
3. 在控制台获取：
   - **Account SID**
   - **Auth Token**

#### 步骤 2: 在 Streamlit Cloud 设置环境变量

1. 登录 Streamlit Cloud
2. 进入您的应用设置
3. 添加以下环境变量：
   - `TWILIO_ACCOUNT_SID`: 您的 Twilio Account SID
   - `TWILIO_AUTH_TOKEN`: 您的 Twilio Auth Token

**注意**：如果不配置 Twilio，系统会自动使用免费的 Google STUN 服务器，但在某些网络环境下可能无法建立连接。

## 🚀 部署步骤

### 1. 推送代码到 GitHub

```bash
git add .
git commit -m "准备 Streamlit Cloud 部署"
git push origin main
```

### 2. 在 Streamlit Cloud 创建应用

1. 访问 https://share.streamlit.io/
2. 点击 "New app"
3. 连接您的 GitHub 仓库
4. 选择分支（通常是 `main` 或 `master`）
5. 设置主文件路径：`app.py`
6. 配置环境变量（如已设置 Twilio）

### 3. 等待部署完成

部署过程可能需要几分钟，Streamlit Cloud 会：
- 安装系统依赖（`packages.txt`）
- 安装 Python 依赖（`requirements.txt`）
- 启动应用

## 🔧 关键配置说明

### WebRTC 配置

项目已配置支持：

1. **TURN 服务器自动选择**：
   - 如果配置了 Twilio 环境变量 → 使用 Twilio TURN 服务器（更稳定）
   - 否则 → 使用免费的 Google STUN 服务器

2. **数据同步机制**：
   - 后台线程通过 `video_processor.detection_results` 存储检测结果
   - 主线程定期同步数据到 `session_state`
   - UI 每 0.5 秒刷新一次，确保实时显示检测结果

### 摄像头模式工作流程

1. **初始化阶段**：
   - 用户选择摄像头模式
   - 加载 YOLO 模型
   - 创建 `DefectDetectionVideoProcessor` 实例

2. **视频流启动**：
   - 用户点击视频组件上的 START 按钮
   - WebRTC 连接建立
   - 开始接收视频帧

3. **实时检测**：
   - 后台线程处理视频帧
   - 根据时间间隔进行检测
   - 检测结果保存到 `video_processor.detection_results`

4. **UI 更新**：
   - 主线程定期同步数据到 `session_state`
   - 显示实时统计信息
   - 显示检测结果表格和图表

## ✅ 验证部署

部署完成后，检查以下功能：

### 1. 基本功能

- [ ] 应用正常启动
- [ ] 可以上传图片进行检测
- [ ] 可以上传视频进行检测

### 2. 摄像头模式（WebRTC）

- [ ] 点击 "摄像头实时检测" 模式
- [ ] 点击视频组件上的 START 按钮
- [ ] 浏览器请求摄像头权限（允许）
- [ ] 视频流正常显示
- [ ] 检测结果实时更新：
  - [ ] 实时统计信息显示
  - [ ] 检测结果表格显示
  - [ ] 缺陷时间分布图表显示

### 3. 检测结果

- [ ] 检测到的缺陷正确标注在视频帧上
- [ ] 统计信息准确（检测帧数、缺陷数量等）
- [ ] 可以下载检测结果 CSV 文件

## 🐛 常见问题排查

### 问题 1: WebRTC 连接失败

**症状**：点击 START 后无法看到视频流

**解决方案**：
1. 检查浏览器是否支持 WebRTC（现代浏览器都支持）
2. 检查是否允许了摄像头权限
3. 如果使用免费 STUN 服务器，尝试配置 Twilio TURN 服务器
4. 检查 Streamlit Cloud 日志中的错误信息

### 问题 2: 检测结果不显示

**症状**：视频流正常，但没有检测结果

**解决方案**：
1. 检查模型文件是否正确上传到 `model/` 目录
2. 检查 Streamlit Cloud 日志，查看是否有模型加载错误
3. 确认检测参数设置合理（置信度阈值等）
4. 等待几秒钟，数据同步可能需要时间

### 问题 3: 页面刷新过于频繁

**症状**：页面不断刷新，影响使用

**解决方案**：
- 这是正常的，系统每 0.5 秒刷新一次以显示最新检测结果
- 如果觉得太频繁，可以在 `app.py` 中调整刷新间隔（搜索 `elapsed_since_refresh >= 0.5`）

### 问题 4: 内存不足

**症状**：应用崩溃或响应变慢

**解决方案**：
1. Streamlit Cloud 免费版有内存限制
2. 减少同时保存的检测帧数
3. 定期清空检测结果
4. 考虑升级到 Streamlit Cloud 付费版

## 📊 性能优化建议

1. **模型优化**：
   - 使用较小的模型（如 YOLOv8n 而不是 YOLOv8m）
   - 使用量化模型（INT8）

2. **检测间隔**：
   - 增加 `time_interval` 参数（默认 1.0 秒）
   - 减少检测频率可以降低 CPU 使用率

3. **数据管理**：
   - 定期清空旧的检测结果
   - 限制保存的帧数

## 📝 部署检查清单

部署前确认：

- [ ] 所有必需文件已提交到 GitHub
- [ ] `requirements.txt` 包含所有依赖
- [ ] `packages.txt` 包含系统依赖
- [ ] 模型文件已上传（`model/best.pt`）
- [ ] Twilio 环境变量已配置（可选但推荐）
- [ ] 代码已测试通过

部署后验证：

- [ ] 应用正常启动
- [ ] 图片检测功能正常
- [ ] 视频检测功能正常
- [ ] 摄像头模式正常（WebRTC 连接成功）
- [ ] 检测结果实时显示
- [ ] 可以下载检测结果

## 🔗 相关文档

- [WEBRTC_DEPLOYMENT_GUIDE.md](./WEBRTC_DEPLOYMENT_GUIDE.md) - WebRTC 详细技术文档
- [CAMERA_MODE_FLOW_ANALYSIS.md](./CAMERA_MODE_FLOW_ANALYSIS.md) - 摄像头模式流程分析
- [Streamlit Cloud 官方文档](https://docs.streamlit.io/streamlit-community-cloud)
- [streamlit-webrtc 文档](https://github.com/whitphx/streamlit-webrtc)

## 💡 技术支持

如果遇到问题：

1. 查看 Streamlit Cloud 的部署日志
2. 检查浏览器控制台错误信息
3. 参考上述常见问题排查
4. 查看项目 GitHub Issues

---

**最后更新**：2024年

