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

f_model = os.getenv("FACE_MODEL_CLASS", "face_model")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
face_model = importlib.import_module(f_model, package=None)
FaceAnalysis = face_model.FaceAnalysis


class Processing:
    def __init__(self, app_settings: AppSettings):
        """
        Processing class for detecting faces, extracting embeddings and drawing faces from images.
        """
        self.settings = app_settings
        self.dl_client = None
        self.model: FaceAnalysis = None
        self.model_meta = None

    async def start(self, dl_client: aiohttp.ClientSession = None):
        self.dl_client = dl_client

        # 从 settings 加载模型元数据
        try:
            self.model_meta = self.settings.get_model_meta()
        except FileNotFoundError as e:
            logger.critical(f"无法加载模型元数据: {e}。程序退出。")
            sys.exit(1)

        # 使用 settings 初始化 FaceAnalysis
        self.model = FaceAnalysis(
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
            model_meta=self.model_meta,
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
        Extracts faces from images.
        """

        if img_req_headers is None:
            img_req_headers = self.settings.defaults.img_req_headers

        if not max_size:
            max_size = self.settings.models.max_size

        t0 = time.time()

        tl0 = time.time()
        images = await get_images(
            images,
            decode=self.model.decode_required,
            session=self.dl_client,
            b64_decode=b64_decode,
            headers=img_req_headers,
            root_images_dir=self.settings.root_images_dir,
        )

        tl1 = time.time()
        took_loading = tl1 - tl0
        logger.debug(f"Reading images took: {took_loading * 1000:.3f} ms.")

        if embed_only:
            _faces_dict = await self.model.embed_crops(
                images,
                extract_embedding=extract_embedding,
                extract_ga=extract_ga,
                detect_masks=detect_masks,
            )
            return _faces_dict

        else:
            te0 = time.time()
            output = await self.model.embed(
                images,
                max_size=max_size,
                return_face_data=return_face_data,
                threshold=threshold,
                limit_faces=limit_faces,
                min_face_size=min_face_size,
                extract_embedding=extract_embedding,
                extract_ga=extract_ga,
                return_landmarks=return_landmarks,
                detect_masks=detect_masks,
            )
            took_embed = time.time() - te0
            took = time.time() - t0
            output["took"]["total_ms"] = took * 1000
            if verbose_timings:
                output["took"]["read_imgs_ms"] = took_loading * 1000
                output["took"]["embed_all_ms"] = took_embed * 1000

            return output

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
        Draws faces on images.
        """

        if not multipart:
            images = await get_images(
                images,
                session=self.dl_client,
                root_images_dir=self.settings.root_images_dir,
            )
            image = images[0].get("data")
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)

        faces = await self.model.get(
            [image],
            threshold=threshold,
            return_face_data=False,
            extract_embedding=False,
            extract_ga=False,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
            detect_masks=detect_masks,
        )

        image = np.ascontiguousarray(image)
        image = self.model.draw_faces(
            image,
            faces[0],
            draw_landmarks=draw_landmarks,
            draw_scores=draw_scores,
            draw_sizes=draw_sizes,
        )

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf

    async def verify(
        self,
        images: Images,
        threshold: float = 0.3,
        limit_faces: int = 0,
        min_face_size: int = 0,
        **kwargs,
    ):
        """
        Verifies if two images contain the same person.
        """
        t0 = time.time()

        result = {
            "is_same_person": False,
            "similarity_score": -1.0,
            "error_message": "",
        }

        processed_images = await get_images(
            images,
            decode=self.model.decode_required,
            session=self.dl_client,
            root_images_dir=self.settings.root_images_dir,
        )

        if len(processed_images) < 2:
            result["error_message"] = "Verification requires at least two images."
            return result

        img1_cv2 = processed_images[0].get("data")
        img2_cv2 = processed_images[1].get("data")

        if img1_cv2 is None or img2_cv2 is None:
            result["error_message"] = "Failed to load one or both images."
            return result

        faces1_list = await self.model.get(
            [img1_cv2],
            threshold=0.5,
            extract_embedding=True,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
        )

        faces2_list = await self.model.get(
            [img2_cv2],
            threshold=0.5,
            extract_embedding=True,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
        )

        faces1 = faces1_list[0] if faces1_list else []
        faces2 = faces2_list[0] if faces2_list else []

        if not faces1:
            result["error_message"] = "No face detected in the first image."
            return result
        if not faces2:
            result["error_message"] = "No face detected in the second image."
            return result

        face1 = sorted(faces1, key=lambda x: x["prob"], reverse=True)[0]
        face2 = sorted(faces2, key=lambda x: x["prob"], reverse=True)[0]

        embedding1 = face1["vec"]
        embedding2 = face2["vec"]

        try:
            similarity = np.dot(embedding1, embedding2)

            result["similarity_score"] = float(similarity)
            result["is_same_person"] = bool(similarity > threshold)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            result["error_message"] = (
                f"An unexpected error occurred during comparison: {e}"
            )

        return result


processing: Processing | None = None


async def get_processing() -> Processing:
    global processing
    if not processing:
        # 现在 Processing 直接从全局 settings 对象初始化
        processing = Processing(app_settings=settings)
    return processing


ProcessingDep = Annotated[Processing, Depends(get_processing)]
