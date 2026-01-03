# Streamlit Cloud 部署检查清单

## ✅ 部署前检查

### 1. Python 版本要求
- [x] 已创建 `runtime.txt` 文件，指定 Python 3.10
- [x] 所有代码兼容 Python 3.10（无需修改代码）
- [x] 依赖包版本已更新，确保兼容 Python 3.10

### 2. 必需文件检查
- [x] `app.py` - Streamlit 主应用文件
- [x] `requirements.txt` - Python 依赖包列表
- [x] `runtime.txt` - Python 版本指定文件（Python 3.10）
- [x] `packages.txt` - 系统级依赖包列表（用于安装 libGL.so.1 等系统库）
- [x] `.streamlit/config.toml` - Streamlit 配置文件（可选但推荐）

### 3. 模型文件检查
- [ ] `model/best.pt` - 训练后的最佳模型（如果存在）
- [ ] `model/yolo11n.pt` - YOLO11n 预训练模型（如果存在）
- ⚠️ **注意**：如果模型文件过大（>100MB），考虑使用 Git LFS 或外部存储

### 4. 数据集文件（可选）
- [ ] `dataset/data.yaml` - 数据集配置文件（如果需要）
- ⚠️ **注意**：训练数据通常不需要部署到 Streamlit Cloud

### 5. Git 仓库准备
- [ ] 所有文件已提交到 Git
- [ ] 已推送到 GitHub 仓库
- [ ] 仓库是公开的（Streamlit Cloud 免费版要求）或已配置访问权限

## 🚀 部署步骤

### 步骤 1：准备 GitHub 仓库
```bash
# 确保所有更改已提交
git add .
git commit -m "配置 Python 3.10，准备 Streamlit Cloud 部署"
git push origin main
```

### 步骤 2：在 Streamlit Cloud 上部署
1. 访问 [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. 使用 GitHub 账号登录
3. 点击 "New app" 按钮
4. 选择你的 GitHub 仓库
5. 设置主文件路径：`app.py`
6. 设置 Python 版本：Streamlit Cloud 会自动从 `runtime.txt` 读取（Python 3.10）
7. 点击 "Deploy!"

### 步骤 3：等待部署完成
- 首次部署通常需要 3-5 分钟
- Streamlit Cloud 会自动安装 `requirements.txt` 中的所有依赖
- 如果部署失败，检查日志中的错误信息

## 🔍 常见问题排查

### 问题 1：部署失败 - Python 版本错误
**解决方案**：
- 确保 `runtime.txt` 文件存在且内容为 `python-3.10`
- 检查文件路径是否正确（应在项目根目录）

### 问题 2：依赖安装失败
**解决方案**：
- 检查 `requirements.txt` 格式是否正确
- 确保所有依赖包都支持 Python 3.10
- 查看部署日志中的具体错误信息

### 问题 2.1：ImportError: libGL.so.1: cannot open shared object file
**错误原因**：
- OpenCV 在 Linux 环境中需要系统库 `libGL.so.1`
- Streamlit Cloud 运行在 Linux 环境中，需要额外安装系统依赖

**解决方案**：
1. **确保使用 `opencv-python-headless`**：
   - 在 `requirements.txt` 中使用 `opencv-python-headless>=4.5.0,<5.0.0`
   - 不要使用 `opencv-python`（它会依赖 GUI 库）

2. **创建 `packages.txt` 文件**：
   - 确保项目根目录存在 `packages.txt` 文件
   - `packages.txt` 内容应包含：
     ```
     libgl1-mesa-glx
     libglib2.0-0
     libsm6
     libxext6
     libxrender1
     libgomp1
     ```
   - 每行一个包名，不要有空行

3. **环境变量设置**：
   - `app.py` 中已在导入 OpenCV 之前设置了环境变量来禁用 GUI 功能
   - 这有助于确保 OpenCV 在 headless 模式下运行

4. **验证部署**：
   - 提交并推送到 GitHub 后，Streamlit Cloud 会自动安装这些系统依赖
   - 检查部署日志确认系统包是否成功安装
   - 如果问题仍然存在，查看完整的错误日志

### 问题 3：模型文件过大
**解决方案**：
- 使用 Git LFS 管理大文件：
  ```bash
  git lfs install
  git lfs track "*.pt"
  git add .gitattributes
  git add model/*.pt
  git commit -m "使用 Git LFS 管理模型文件"
  ```
- 或使用外部存储（如 AWS S3、Google Cloud Storage）并在运行时下载

### 问题 4：应用运行缓慢
**原因**：
- Streamlit Cloud 免费版只提供 CPU 资源
- 模型推理在 CPU 上较慢
- 考虑使用更小的模型（如 yolo11n.pt）

### 问题 5：找不到模型文件
**解决方案**：
- 确保模型文件已提交到 Git 仓库
- 检查 `app.py` 中的模型路径是否正确
- 如果使用默认模型，确保 `model/best.pt` 存在

## 📝 部署后验证

部署成功后，验证以下功能：

- [ ] 应用可以正常访问
- [ ] 侧边栏配置正常显示
- [ ] 可以上传图片进行检测
- [ ] 模型加载成功（检查日志）
- [ ] 检测结果正常显示
- [ ] 统计图表正常生成

## 🔄 更新应用

如果需要更新应用：

1. 在本地修改代码
2. 提交并推送到 GitHub：
   ```bash
   git add .
   git commit -m "更新应用功能"
   git push origin main
   ```
3. Streamlit Cloud 会自动检测更改并重新部署

## 📚 相关资源

- [Streamlit Cloud 文档](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit 部署指南](https://docs.streamlit.io/deploy)
- [Python 3.10 新特性](https://docs.python.org/3/whatsnew/3.10.html)

## ✨ 提示

- Streamlit Cloud 免费版提供足够的资源用于演示和小型应用
- 如果需要 GPU 加速，考虑使用付费版本或其他云平台
- 定期检查应用日志，确保应用正常运行
- 建议在本地 Python 3.10 环境中测试后再部署

