import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- 自定义验证器和类型 ---

def empty_str_to_none(v: str) -> Optional[str]:
    """将空字符串或'none'转换为 None。"""
    if v is None or v.lower() in ('', 'none', 'null'):
        return None
    return v

def str_to_int_list(v: Union[str, List[int]]) -> List[int]:
    """将逗号分隔的字符串转换为整数列表。"""
    if isinstance(v, str):
        if not v:
            return []
        return [int(item.strip()) for item in v.split(',')]
    return v

# --- 配置模型 ---

class DefaultAPISettings(BaseSettings):
    """API 请求的默认参数值。"""
    return_face_data: bool = Field(False, description="是否返回base64编码的人脸裁剪图")
    return_landmarks: bool = Field(False, description="是否返回人脸关键点")
    extract_embedding: bool = Field(True, description="是否提取人脸特征向量")
    extract_ga: bool = Field(False, description="是否提取性别年龄信息")
    detect_masks: bool = Field(False, description="是否检测口罩")
    det_thresh: float = Field(0.6, description="人脸检测置信度阈值")
    img_req_headers: Dict[str, str] = Field(
        default_factory=lambda: {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"},
        description="获取远程图片时使用的HTTP头"
    )
    sslv3_hack: bool = Field(False, description="是否为旧版SSLv3服务器启用兼容模式")

    model_config = SettingsConfigDict(env_prefix='DEF_')


class ModelSettings(BaseSettings):
    """模型相关的配置。"""
    inference_backend: str = Field("onnx", description="推理后端 (e.g., 'onnx', 'trt')")
    det_name: str = Field("scrfd_10g_gnkps", description="人脸检测模型名称")
    rec_name: str = Field("glintr100", description="人脸识别模型名称")
    max_size: List[int] = Field([640, 640], description="输入图像的最大尺寸 [宽, 高]")
    ga_name: Optional[str] = Field(None, description="性别年龄识别模型名称")
    mask_detector: Optional[str] = Field(None, description="口罩检测模型名称")
    rec_batch_size: int = Field(1, description="识别模型的最大批处理大小")
    det_batch_size: int = Field(1, description="检测模型的最大批处理大小")
    force_fp16: bool = Field(False, description="是否强制使用FP16精度 (仅TensorRT)")
    triton_uri: Optional[str] = Field(None, description="Triton推理服务器的URI (e.g., 'localhost:8001')")
    
    # 自定义类型验证器
    _validate_max_size = field_validator('max_size', mode='before')(str_to_int_list)
    _validate_optional_models = field_validator('ga_name', 'mask_detector', 'triton_uri', mode='before')(empty_str_to_none)


class AppSettings(BaseSettings):
    """应用的主配置。"""
    log_level: str = Field("INFO", description="日志级别 (e.g., 'DEBUG', 'INFO')")
    port: int = Field(18080, description="应用监听的端口号")
    models_dir: str = Field("/models", description="存放模型的根目录")
    root_images_dir: str = Field("/images", description="用于从相对路径加载图片的根目录")

    # 嵌套的配置模型
    models: ModelSettings = Field(default_factory=ModelSettings)
    defaults: DefaultAPISettings = Field(default_factory=DefaultAPISettings)

    # Pydantic Settings 配置
    model_config = SettingsConfigDict(
        env_nested_delimiter='__',  # 使用双下划线来区分嵌套环境变量, e.g., MODELS__DET_NAME
        env_file='.env',             # 从 .env 文件加载配置
        env_file_encoding='utf-8'
    )

    @property
    def model_configs_path(self) -> Path:
        """返回模型元数据 `models.json` 的路径。"""
        return Path(self.models_dir)

    def get_model_meta(self) -> Dict[str, Any]:
        """
        加载并返回模型的元数据。
        
        首先尝试加载 `models.override.json`，如果不存在则加载 `models.json`。
        """
        base_path = self.model_configs_path
        override_path = base_path / 'models.override.json'
        default_path = base_path / 'models.json'

        config_path = override_path if override_path.exists() else default_path

        if not config_path.exists():
            raise FileNotFoundError(f"模型元数据文件未找到: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

# 创建一个全局的 settings 实例，供整个应用使用
# Pydantic 会自动从环境变量和 .env 文件中读取配置来填充这个实例
settings = AppSettings()
