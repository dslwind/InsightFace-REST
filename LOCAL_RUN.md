# 本地运行指南 (Local Run Guide)

本项目现在支持在不依赖 Docker 的情况下本地直接运行。

## 前置要求

1. **Python 3.10+** - 确保已安装 Python 3.10 或更高版本
2. **系统依赖** (Linux/Mac):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 libgomp1
   
   # macOS (使用 Homebrew)
   brew install libomp
   ```

## 启动步骤

手动启动服务，按以下步骤操作：

1. **创建虚拟环境** (首次运行):
   
   可以选择使用 venv 或 conda 创建虚拟环境：
   
   **方式 1: 使用 venv**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   **方式 2: 使用 conda (推荐，可指定 Python 3.10)**
   ```bash
   # 创建 conda 环境，指定 Python 3.10
   conda create -n ifr python=3.10 -y
   
   # 激活环境
   # Windows
   conda activate ifr
   
   # Linux/Mac
   conda activate ifr
   ```

2. **安装依赖**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **配置环境变量** (可选):
   - 编辑 `local.env` 文件并根据需要修改配置
   - 或直接设置环境变量

4. **运行服务**:
   ```bash
   python run_local.py
   ```

## 配置文件

编辑 `local.env` 文件来配置服务：

### 重要配置项

- **MODELS_DIR**: 模型存储目录（默认: `models`，相对于项目根目录）
- **PORT**: 服务端口（默认: `18080`）
- **NUM_WORKERS**: 工作进程数（默认: `4`）
- **LOG_LEVEL**: 日志级别（默认: `INFO`）
- **INFERENCE_BACKEND**: 推理后端（`onnx` 用于 CPU，`trt` 用于 GPU）
- **DET_NAME**: 人脸检测模型
- **REC_NAME**: 人脸识别模型

### 模型目录

默认情况下，模型会下载到项目根目录下的 `models/` 文件夹。你可以通过修改 `local.env` 中的 `MODELS_DIR` 来更改：

```env
MODELS_DIR=/path/to/your/models
```

或者使用相对路径：

```env
MODELS_DIR=models
```

## 访问服务

启动成功后，你可以通过以下方式访问：

- **API 文档**: http://localhost:18080/docs
- **健康检查**: http://localhost:18080/health

## 与 Docker 版本的差异

1. **模型目录**: 本地版本默认使用 `models/` (相对于项目根目录)，而不是 Docker 中的 `/models`
2. **端口**: 可以通过 `local.env` 中的 `PORT` 配置，默认为 `18080`
3. **Windows 平台**: 
   - Windows 上使用 `uvicorn` 直接运行（因为 `gunicorn` 不支持 Windows）
   - Windows 上不支持多工作进程（单进程运行）
   - Linux/Mac 上使用 `gunicorn` + `uvicorn` 支持多进程
4. **GPU 支持**: 如需使用 GPU，需要：
   - 安装 CUDA 和 cuDNN
   - 安装 TensorRT (如果使用 `trt` 后端)
   - 安装相应的 GPU 版本的依赖

## 故障排除

### 端口已被占用

修改 `local.env` 中的 `PORT` 配置为其他端口。

### 内存不足

- 减少 `NUM_WORKERS`
- 减少 `REC_BATCH_SIZE` 和 `DET_BATCH_SIZE`
- 使用更轻量的模型

### 依赖安装失败

- 确保 Python 版本 >= 3.10
- 确保系统依赖已安装
- 尝试使用虚拟环境

## 开发说明

本项目尽量不修改源文件。仅做了以下最小化修改：

1. `if_rest/prepare_models.py`: 在 `__main__` 中支持从环境变量读取模型目录
2. `if_rest/core/processing.py`: 在 `get_processing()` 中支持从环境变量读取模型目录

这些修改保持了向后兼容性，如果未设置 `MODELS_DIR` 环境变量，仍然使用默认的 `/models` 路径。
