from typing import Dict, List, Optional, Union

import pydantic
from pydantic import BaseModel, Field

# 导入统一的 settings 单例
from if_rest.settings import settings

# --- 示例数据 ---
# 这些示例数据用于 API 文档的自动生成
example_img_url = "test_images/Stallone.jpg"
dummy_base64_string_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
dummy_base64_string_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFhAJ/wlseKgAAAABJRU5ErkJggg=="

face_verification_example = {
    "images": {
        "data": [dummy_base64_string_1, dummy_base64_string_2],
        "urls": ["test_images/TH.png", "test_images/TH1.jpg"],
    },
    "threshold": 0.3,
    "limit_faces": 1,
    "min_face_size": 0,
    "verbose_timings": False,
    "msgpack": False,
}

# --- 请求体模型 ---


class Images(BaseModel):
    """定义了图像输入的两种方式：Base64 数据或 URL 列表。"""

    data: Optional[Union[List[str], List[bytes]]] = Field(
        default=None, example=None, description="List of base64 encoded images"
    )
    urls: Optional[List[str]] = Field(
        default=None, example=[example_img_url], description="List of images urls"
    )


class BodyExtract(BaseModel):
    """`/extract` 端点的请求体模型。"""

    images: Images

    threshold: Optional[float] = Field(
        default=settings.defaults.det_thresh, description="Detector threshold"
    )
    embed_only: Optional[bool] = Field(
        default=False,
        description="Treat input images as face crops and omit detection step",
    )
    return_face_data: Optional[bool] = Field(
        default=settings.defaults.return_face_data,
        description="Return face crops encoded in base64",
    )
    return_landmarks: Optional[bool] = Field(
        default=settings.defaults.return_landmarks, description="Return face landmarks"
    )
    extract_embedding: Optional[bool] = Field(
        default=settings.defaults.extract_embedding,
        description="Extract face embeddings (otherwise only detect faces)",
    )
    extract_ga: Optional[bool] = Field(
        default=settings.defaults.extract_ga, description="Extract gender/age"
    )
    detect_masks: Optional[bool] = Field(
        default=settings.defaults.detect_masks, description="Detect medical masks"
    )
    limit_faces: Optional[int] = Field(
        default=0,
        description="Maximum number of faces to be processed (0 for no limit)",
    )
    min_face_size: Optional[int] = Field(
        default=0,
        description="Ignore faces smaller than this size in pixels (0 for no limit)",
    )
    verbose_timings: Optional[bool] = Field(
        default=False, description="Return all timings for debugging."
    )
    msgpack: Optional[bool] = Field(
        default=False, description="Use MSGPACK for response serialization"
    )
    img_req_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom headers for retrieving images from remote servers",
    )


class BodyDraw(BaseModel):
    """`/draw_detections` 端点的请求体模型。"""

    images: Images

    threshold: Optional[float] = Field(
        default=settings.defaults.det_thresh, description="Detector threshold"
    )
    draw_landmarks: Optional[bool] = Field(
        default=True, description="Draw face landmarks"
    )
    draw_scores: Optional[bool] = Field(
        default=True, description="Draw detection scores on bounding boxes"
    )
    draw_sizes: Optional[bool] = Field(
        default=True, description="Draw face sizes on bounding boxes"
    )
    limit_faces: Optional[int] = Field(
        default=0,
        description="Maximum number of faces to be processed (0 for no limit)",
    )
    min_face_size: Optional[int] = Field(
        default=0,
        description="Ignore faces smaller than this size in pixels (0 for no limit)",
    )
    detect_masks: Optional[bool] = Field(
        default=settings.defaults.detect_masks, description="Detect medical masks"
    )


class FaceVerification(BaseModel):
    """`/verify` 端点的请求体模型。"""

    images: Images = Field(example=face_verification_example["images"])
    threshold: Optional[float] = Field(
        default=0.3, description="The threshold for face verification."
    )
    limit_faces: Optional[int] = Field(
        default=0,
        description="Maximum number of faces to be processed (0 for no limit)",
    )
    min_face_size: Optional[int] = Field(
        default=0,
        description="Ignore faces smaller than this size in pixels (0 for no limit)",
    )
    verbose_timings: Optional[bool] = Field(
        default=False, description="Return all timings for debugging."
    )
    msgpack: Optional[bool] = Field(
        default=False, description="Use MSGPACK for response serialization"
    )

    class Config:
        json_schema_extra = {"example": face_verification_example}
