import importlib
import io
import os
import sys
import time
from typing import Annotated, List

import aiohttp
import cv2
import numpy as np
from fastapi import Depends

from if_rest.core.utils.image_provider import get_images
from if_rest.logger import logger
from if_rest.schemas import Images
from if_rest.settings import AppSettings, settings

# 动态导入 face_model 模块
face_model_class_name = os.getenv("FACE_MODEL_CLASS", "face_model")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
face_model_module = importlib.import_module(face_model_class_name, package=None)
FaceAnalysis = face_model_module.FaceAnalysis


class Processing:
    def __init__(self, app_settings: AppSettings):
        """
        处理人脸检测、特征提取和图像绘制的核心类。
        """
        self.settings = app_settings
        self.download_client: aiohttp.ClientSession | None = None
        self.face_analysis_model: FaceAnalysis | None = None
        self.model_metadata = None

    async def start(self, download_client: aiohttp.ClientSession = None):
        self.download_client = download_client

        # 从 settings 加载模型元数据
        try:
            self.model_metadata = self.settings.get_model_meta()
        except FileNotFoundError as e:
            logger.critical(f"无法加载模型元数据: {e}。程序退出。")
            sys.exit(1)

        # 使用 settings 初始化 FaceAnalysis
        self.face_analysis_model = FaceAnalysis(
            det_name=self.settings.models.det_name,
            rec_name=self.settings.models.rec_name,
            ga_name=self.settings.models.ga_name,
            mask_detector=self.settings.models.mask_detector,
            max_size=self.settings.models.max_size,
            max_rec_batch_size=self.settings.models.rec_batch_size,
            max_det_batch_size=self.settings.models.det_batch_size,
            backend_name=self.settings.models.inference_backend,
            force_fp16=self.settings.models.force_fp16,
            triton_uri=self.settings.models.triton_uri,
            root_dir=self.settings.models_dir,
            # 传递 settings 和 model_meta 到 FaceAnalysis
            settings=self.settings,
            model_meta=self.model_metadata,
        )

    async def extract(
        self,
        images: Images,
        max_size: List[int] = None,
        threshold: float = 0.6,
        limit_faces: int = 0,
        min_face_size: int = 0,
        embed_only: bool = False,
        return_face_data: bool = False,
        extract_embedding: bool = True,
        extract_ga: bool = True,
        return_landmarks: bool = False,
        detect_masks: bool = False,
        verbose_timings=True,
        b64_decode=True,
        img_req_headers=None,
        **kwargs,
    ):
        """
        从图像中提取人脸信息。
        """
        request_headers = img_req_headers or self.settings.defaults.img_req_headers
        target_max_size = max_size or self.settings.models.max_size
        total_start_time = time.time()

        # 获取和解码图像
        image_load_start_time = time.time()
        processed_images = await get_images(
            images,
            decode=self.face_analysis_model.decode_required,
            session=self.download_client,
            b64_decode=b64_decode,
            headers=request_headers,
            root_images_dir=self.settings.root_images_dir,
        )
        image_loading_duration_ms = (time.time() - image_load_start_time) * 1000
        logger.debug(f"读取图像耗时: {image_loading_duration_ms:.3f} ms.")

        # 根据模式（仅嵌入或完整流程）处理图像
        if embed_only:
            faces_result = await self.face_analysis_model.embed_crops(
                processed_images,
                extract_embedding=extract_embedding,
                extract_ga=extract_ga,
                detect_masks=detect_masks,
            )
            return faces_result
        else:
            embed_start_time = time.time()
            output_data = await self.face_analysis_model.embed(
                processed_images,
                max_size=target_max_size,
                return_face_data=return_face_data,
                threshold=threshold,
                limit_faces=limit_faces,
                min_face_size=min_face_size,
                extract_embedding=extract_embedding,
                extract_ga=extract_ga,
                return_landmarks=return_landmarks,
                detect_masks=detect_masks,
            )
            embed_duration_ms = (time.time() - embed_start_time) * 1000

            # 记录详细耗时
            total_duration_ms = (time.time() - total_start_time) * 1000
            output_data["took"] = {"total_ms": total_duration_ms}
            if verbose_timings:
                output_data["took"]["read_imgs_ms"] = image_loading_duration_ms
                output_data["took"]["embed_all_ms"] = embed_duration_ms

            return output_data

    async def draw(
        self,
        images: Images,
        threshold: float = 0.6,
        draw_landmarks: bool = True,
        draw_scores: bool = True,
        draw_sizes: bool = True,
        limit_faces=0,
        min_face_size: int = 0,
        detect_masks: bool = False,
        multipart=False,
        **kwargs,
    ):
        """
        在图像上绘制检测到的人脸框和关键点。
        """
        if not multipart:
            processed_images = await get_images(
                images,
                session=self.download_client,
                root_images_dir=self.settings.root_images_dir,
            )
            image_cv2 = processed_images[0].get("data")
        else:
            image_binary_data = np.fromstring(images, np.uint8)
            image_cv2 = cv2.imdecode(image_binary_data, cv2.IMREAD_COLOR)

        faces_list = await self.face_analysis_model.get(
            [image_cv2],
            threshold=threshold,
            return_face_data=False,
            extract_embedding=False,
            extract_ga=False,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
            detect_masks=detect_masks,
        )

        image_cv2 = np.ascontiguousarray(image_cv2)
        drawn_image = self.face_analysis_model.draw_faces(
            image_cv2,
            faces_list[0],
            draw_landmarks=draw_landmarks,
            draw_scores=draw_scores,
            draw_sizes=draw_sizes,
        )

        is_success, buffer = cv2.imencode(".jpg", drawn_image)
        image_buffer = io.BytesIO(buffer)
        return image_buffer

    async def verify(
        self,
        images: Images,
        threshold: float = 0.3,
        limit_faces: int = 0,
        min_face_size: int = 0,
        **kwargs,
    ):
        """
        验证两张图片中的人脸是否为同一个人。
        """
        verification_result = {
            "is_same_person": False,
            "similarity_score": -1.0,
            "error_message": "",
        }

        processed_images = await get_images(
            images,
            decode=self.face_analysis_model.decode_required,
            session=self.download_client,
            root_images_dir=self.settings.root_images_dir,
        )

        if len(processed_images) < 2:
            verification_result["error_message"] = "验证需要至少两张图片。"
            return verification_result

        image_one_cv2 = processed_images[0].get("data")
        image_two_cv2 = processed_images[1].get("data")

        if image_one_cv2 is None or image_two_cv2 is None:
            verification_result["error_message"] = "加载一张或两张图片失败。"
            return verification_result

        # 分别获取两张图片的人脸信息
        faces_list_one = await self.face_analysis_model.get(
            [image_one_cv2],
            threshold=0.5,
            extract_embedding=True,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
        )
        faces_list_two = await self.face_analysis_model.get(
            [image_two_cv2],
            threshold=0.5,
            extract_embedding=True,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
        )

        faces_one = faces_list_one[0] if faces_list_one else []
        faces_two = faces_list_two[0] if faces_list_two else []

        if not faces_one:
            verification_result["error_message"] = "在第一张图片中未检测到人脸。"
            return verification_result
        if not faces_two:
            verification_result["error_message"] = "在第二张图片中未检测到人脸。"
            return verification_result

        # 选择置信度最高的人脸进行比较
        best_face_one = sorted(faces_one, key=lambda x: x["prob"], reverse=True)[0]
        best_face_two = sorted(faces_two, key=lambda x: x["prob"], reverse=True)[0]

        embedding_one = best_face_one["vec"]
        embedding_two = best_face_two["vec"]

        try:
            # 计算余弦相似度
            similarity = np.dot(embedding_one, embedding_two)

            verification_result["similarity_score"] = float(similarity)
            verification_result["is_same_person"] = bool(similarity > threshold)
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}")
            verification_result["error_message"] = f"比较过程中发生意外错误: {e}"

        return verification_result


# 全局单例处理对象
processing: Processing | None = None


async def get_processing() -> Processing:
    """
    FastAPI 依赖注入函数，用于获取 Processing 的单例。
    """
    global processing
    if not processing:
        processing = Processing(app_settings=settings)
    return processing


# 为 FastAPI 依赖注入创建类型注解
ProcessingDep = Annotated[Processing, Depends(get_processing)]
