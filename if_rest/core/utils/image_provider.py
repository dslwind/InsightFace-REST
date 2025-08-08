import asyncio
import base64
import io
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import aiohttp
import cv2
import exifread
import imageio
import numpy as np
from tenacity import (before_sleep_log, retry, retry_if_not_exception_type,
                      stop_after_attempt, wait_exponential)
from turbojpeg import TurboJPEG

from if_rest.core.utils.helpers import tobool
from if_rest.logger import logger
from if_rest.schemas import Images

# 根据环境变量决定是否使用 nvJPEG
if tobool(os.getenv("USE_NVJPEG", "False")):
    try:
        from nvjpeg import NvJpeg
        jpeg_decoder = NvJpeg()
        logger.info("使用 nvJPEG 进行 JPEG 解码。")
    except ImportError:
        logger.warning("无法导入 nvJPEG，将回退到 TurboJPEG。")
        jpeg_decoder = TurboJPEG()
else:
    logger.info("使用 TurboJPEG 进行 JPEG 解码。")
    jpeg_decoder = TurboJPEG()


def resize_image(image: np.ndarray, max_size: list = None) -> Tuple[np.ndarray, float]:
    """
    根据给定的最大尺寸，按比例缩放图像。

    Args:
        image (np.ndarray): 输入的图像 (H, W, C 格式)。
        max_size (list): [宽度, 高度] 格式的最大尺寸。

    Returns:
        Tuple[np.ndarray, float]: 缩放后的图像和所使用的缩放因子。
    """
    if max_size is None:
        max_size = [640, 640]

    target_width, target_height = max_size
    height, width, _ = image.shape

    scale_factor = min(target_width / width, target_height / height)
    
    # 如果图像过小，可能会影响检测精度，适当减小缩放因子
    if scale_factor > 2:
        scale_factor *= 0.7

    interpolation = cv2.INTER_AREA if scale_factor <= 1.0 else cv2.INTER_LINEAR

    if scale_factor == 1.0:
        transformed_image = image
    else:
        transformed_image = cv2.resize(
            image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interpolation
        )

    # 如果缩放后尺寸小于目标尺寸，用黑边填充
    h, w, _ = transformed_image.shape
    pad_right = target_width - w
    pad_bottom = target_height - h
    
    if pad_right > 0 or pad_bottom > 0:
        transformed_image = cv2.copyMakeBorder(
            transformed_image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return transformed_image, scale_factor


def sniff_and_process_gif(data: bytes) -> bytes:
    """
    检查数据是否为 GIF 格式，如果是，则提取第一帧并转为 JPEG 格式。

    Args:
        data (bytes): 输入的图像二进制数据。

    Returns:
        bytes: 如果是 GIF，返回第一帧的 JPEG 数据；否则返回原始数据。
    """
    try:
        # GIF 文件通常以 'GIF' 开头
        if data.startswith(b"GIF"):
            sequence = imageio.get_reader(data, ".gif")
            # 只处理第一帧
            first_frame = sequence.get_data(0)
            
            output_buffer = io.BytesIO()
            imageio.imwrite(output_buffer, first_frame, format="jpeg")
            output_buffer.seek(0)
            return output_buffer.read()
        return data
    except Exception:
        # 如果处理失败，返回原始数据
        return data


def transpose_image_by_exif(image: np.ndarray, orientation_tag) -> np.ndarray:
    """
    根据 EXIF 方向标签旋转或翻转图像。

    Args:
        image (np.ndarray): 输入图像。
        orientation_tag: 从 exifread 库获取的方向标签对象。

    Returns:
        np.ndarray: 调整方向后的图像。
    """
    TRANSPOSE_MAP = {
        2: np.fliplr,
        3: lambda img: np.rot90(img, 2),
        4: np.flipud,
        5: lambda img: np.rot90(np.flipud(img), -1),
        6: lambda img: np.rot90(img, -1),
        7: lambda img: np.rot90(np.flipud(img)),
        8: lambda img: np.rot90(img),
    }

    if orientation_tag is None:
        return image

    orientation_value = orientation_tag.values[0]
    transform_func = TRANSPOSE_MAP.get(orientation_value, lambda img: img)
    return transform_func(image)


async def read_local_file_as_bytes(path: Path) -> np.ndarray:
    """异步读取本地文件并返回 numpy 字节数组。"""
    async with aiofiles.open(path, mode="rb") as f:
        data = await f.read()
        processed_data = sniff_and_process_gif(data)
        return np.frombuffer(processed_data, dtype="uint8")


def base64_to_bytes(b64_encoded: str, b64_decode: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """将 Base64 编码的字符串解码为 numpy 字节数组。"""
    try:
        data_bytes = b64_encoded.split(",")[-1] if b64_decode else b64_encoded
        decoded_bytes = base64.b64decode(data_bytes)
        processed_data = sniff_and_process_gif(decoded_bytes)
        return np.frombuffer(processed_data, dtype="uint8"), None
    except Exception:
        error_message = traceback.format_exc()
        logger.warning(error_message)
        return None, error_message


def decode_image_bytes(image_bytes: np.ndarray) -> np.ndarray:
    """
    使用高性能解码器将字节数组解码为图像。
    首先尝试使用 TurboJPEG/nvJPEG，失败则回退到 OpenCV。
    """
    decoding_start_time = time.perf_counter()
    try:
        # 从字节流中读取 EXIF 信息
        orientation_tag = exifread.process_file(
            io.BytesIO(image_bytes), stop_tag="Image Orientation"
        ).get("Image Orientation")
        
        decoded_image = jpeg_decoder.decode(image_bytes)
        decoded_image = transpose_image_by_exif(decoded_image, orientation_tag)
    except Exception:
        logger.debug("JPEG 解码失败，回退到 cv2.imdecode。")
        decoded_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    
    decoding_duration_ms = (time.perf_counter() - decoding_start_time) * 1000
    logger.debug(f"图像解码耗时: {decoding_duration_ms:.3f} ms.")
    return decoded_image


@retry(
    wait=wait_exponential(min=0.5, max=5),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_not_exception_type(ValueError),
)
async def make_request(url: str, session: aiohttp.ClientSession, headers: dict) -> aiohttp.ClientResponse:
    """带重试机制的异步 GET 请求。"""
    resp = await session.get(url, allow_redirects=True, headers=headers)
    if resp.status in [401, 403, 404]:
        raise ValueError(f"获取数据失败 {url}。状态码: {resp.status}")
    resp.raise_for_status() # 对其他 >= 400 的状态码抛出异常
    return resp


async def download_image(
    path_or_url: str,
    session: aiohttp.ClientSession,
    headers: dict,
    root_images_dir: str
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """从 URL 或本地路径获取图像字节。"""
    try:
        if path_or_url.startswith("http"):
            resp = await make_request(path_or_url, session, headers=headers)
            content = await resp.read()
            processed_data = sniff_and_process_gif(content)
            data_bytes = np.frombuffer(processed_data, dtype="uint8")
        else:
            # 处理本地文件路径
            full_path = Path(root_images_dir) / path_or_url
            if not full_path.exists():
                error_message = f"文件未找到: '{full_path}'"
                return None, error_message
            data_bytes = await read_local_file_as_bytes(full_path)
        return data_bytes, None
    except Exception:
        error_message = traceback.format_exc()
        logger.warning(error_message)
        return None, error_message


def create_image_data_dict(data_bytes: Optional[np.ndarray], error_message: Optional[str], decode: bool) -> dict:
    """根据字节和错误信息创建最终的图像数据字典。"""
    if error_message:
        return {"data": None, "traceback": error_message}

    if decode:
        try:
            decoded_image = decode_image_bytes(data_bytes)
            if decoded_image is None:
                return {"data": None, "traceback": "无法解码文件，可能不是有效的图像格式。"}
            return {"data": decoded_image, "traceback": None}
        except Exception:
            return {"data": None, "traceback": traceback.format_exc()}
    else:
        return {"data": data_bytes, "traceback": None}


async def get_images(
    images_input: Images,
    decode: bool,
    session: aiohttp.ClientSession,
    b64_decode: bool,
    headers: dict,
    root_images_dir: str
) -> List[Dict]:
    """
    根据输入的 URL 或 Base64 数据，异步获取并处理图像。
    """
    image_data_list = []

    if images_input.urls:
        tasks = [
            asyncio.ensure_future(download_image(url, session, headers, root_images_dir))
            for url in images_input.urls
        ]
        results = await asyncio.gather(*tasks)
        for data_bytes, error_message in results:
            image_data = create_image_data_dict(data_bytes, error_message, decode)
            image_data_list.append(image_data)

    elif images_input.data:
        for b64_img in images_input.data:
            data_bytes, error_message = base64_to_bytes(b64_img, b64_decode=b64_decode)
            image_data = create_image_data_dict(data_bytes, error_message, decode)
            image_data_list.append(image_data)

    return image_data_list
