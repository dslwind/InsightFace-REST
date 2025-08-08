import os

import tensorrt as trt
from fastapi import APIRouter, HTTPException

from if_rest.core.processing import ProcessingDep
from if_rest.schemas import Images
from if_rest.settings import settings

router = APIRouter()

# 从环境变量获取版本号，如果不存在则使用默认值
__version__ = os.getenv("IFR_VERSION", "0.9.5.0")


@router.get("/info", tags=["Utility"])
def get_service_info():
    """
    返回当前服务的配置信息。
    """
    try:
        # 使用 Pydantic V2 的 model_dump 方法来序列化配置
        # exclude 用于移除不需要在 API 中暴露的敏感或内部字段
        models_info = settings.models.model_dump(
            exclude={"ga_ignore", "rec_ignore", "mask_ignore", "device"}
        )
        defaults_info = settings.defaults.model_dump()

        service_info = {
            "version": __version__,
            "tensorrt_version": trt.__version__,
            "log_level": settings.log_level,
            "models": models_info,
            "defaults": defaults_info,
        }
        return service_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["Utility"])
async def check_health(processing: ProcessingDep):
    """
    执行一次简单的识别请求，以验证服务是否正常工作。
    """
    # 使用位于服务端的测试图片进行健康检查
    health_check_image = Images(urls=["test_images/Stallone.jpg"])

    try:
        result = await processing.extract(images=health_check_image)
        # 确认返回结果中至少检测到了一个人脸
        faces = result.get("data", [{}])[0].get("faces", [])
        if not faces:
            raise ValueError(
                "Health check failed: No faces detected in the test image."
            )
        return {"status": "ok"}
    except Exception as e:
        # 如果健康检查失败，返回 500 错误
        raise HTTPException(status_code=500, detail=f"Self-check failed: {e}")
