import asyncio
import importlib
import io
import os
import sys
import time
from typing import Annotated, Any, Dict, List

import aiohttp
import cv2
import numpy as np
from fastapi import Depends

from if_rest.core.utils.image_provider import get_images
from if_rest.logger import logger
from if_rest.schemas import Images
from if_rest.settings import Settings

f_model = os.getenv("FACE_MODEL_CLASS", "face_model")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
face_model = importlib.import_module(f_model, package=None)
FaceAnalysis = face_model.FaceAnalysis


class Processing:

    def __init__(
        self,
        det_name: str = "retinaface_r50_v1",
        rec_name: str = "arcface_r100_v1",
        ga_name: str = "genderage_v1",
        mask_detector: str = "mask_detector",
        max_size: List[int] = None,
        backend_name: str = "trt",
        max_rec_batch_size: int = 1,
        max_det_batch_size: int = 1,
        force_fp16: bool = False,
        triton_uri=None,
        root_dir: str = "/models",
        **kwargs,
    ):
        """
        Processing class for detecting faces, extracting embeddings and drawing faces from images.

        Args:
            det_name (str): The name of the detection model to use. Defaults to 'retinaface_r50_v1'.
            rec_name (str): The name of the recognition model to use. Defaults to 'arcface_r100_v1'.
            ga_name (str): The name of the gender and age model to use. Defaults to 'genderage_v1'.
            mask_detector (str): The name of the mask detector model to use. Defaults to 'mask_detector'.
            max_size (List[int]): The maximum size for images. Defaults to [640, 480].
            backend_name (str): The backend name to use. Defaults to 'trt'.
            max_rec_batch_size (int): The maximum batch size for recognition. Defaults to 1.
            max_det_batch_size (int): The maximum batch size for detection. Defaults to 1.
            force_fp16 (bool): Whether to force FP16 mode. Defaults to False.
            triton_uri (str): The URI for Triton server. Defaults to None.
            root_dir (str): The root directory for models. Defaults to '/models'.
            dl_client (aiohttp.ClientSession): An asynchronous HTTP client session. Defaults to None.
        """

        if max_size is None:
            max_size = [640, 480]

        self.max_rec_batch_size = max_rec_batch_size
        self.max_det_batch_size = max_det_batch_size
        self.det_name = det_name
        self.rec_name = rec_name
        self.ga_name = ga_name
        self.max_size = max_size
        self.mask_detector = mask_detector
        self.force_fp16 = force_fp16
        self.backend_name = backend_name
        self.triton_uri = triton_uri
        self.root_dir = root_dir
        self.dl_client = None
        self.model: FaceAnalysis = None

    async def start(self, dl_client: aiohttp.ClientSession = None):
        self.dl_client = dl_client
        self.model = FaceAnalysis(
            det_name=self.det_name,
            rec_name=self.rec_name,
            ga_name=self.ga_name,
            mask_detector=self.mask_detector,
            max_size=self.max_size,
            max_rec_batch_size=self.max_rec_batch_size,
            max_det_batch_size=self.max_det_batch_size,
            backend_name=self.backend_name,
            force_fp16=self.force_fp16,
            triton_uri=self.triton_uri,
            root_dir=self.root_dir,
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

        Args:
            images (Dict[str, list]): A dictionary containing image data.
            max_size (List[int]): The maximum size for images. Defaults to None.
            threshold (float): The threshold for face detection. Defaults to 0.6.
            limit_faces (int): The maximum number of faces to detect. Defaults to 0.
            min_face_size (int): The minimum size of a face to detect. Defaults to 0.
            embed_only (bool): Whether to only extract embeddings. Defaults to False.
            return_face_data (bool): Whether to return face data. Defaults to False.
            extract_embedding (bool): Whether to extract embeddings. Defaults to True.
            extract_ga (bool): Whether to extract gender and age. Defaults to True.
            return_landmarks (bool): Whether to return landmarks. Defaults to False.
            detect_masks (bool): Whether to detect masks. Defaults to False.
            verbose_timings (bool): Whether to print verbose timings. Defaults to True.

        Returns:
            Dict[str, Union[List[Dict], bytes]]: A dictionary containing extracted faces and timing information.
        """

        if img_req_headers is None:
            img_req_headers = {}

        if not max_size:
            max_size = self.max_size

        t0 = time.time()

        tl0 = time.time()
        images = await get_images(
            images,
            decode=self.model.decode_required,
            session=self.dl_client,
            b64_decode=b64_decode,
            headers=img_req_headers,
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
        dl_client: aiohttp.ClientSession = None,
        **kwargs,
    ):
        """
        Draws faces on images.

        Args:
            images (Union[Dict[str, list], bytes]): The input image data.
            threshold (float): The threshold for face detection. Defaults to 0.6.
            draw_landmarks (bool): Whether to draw landmarks. Defaults to True.
            draw_scores (bool): Whether to draw scores. Defaults to True.
            draw_sizes (bool): Whether to draw sizes. Defaults to True.
            limit_faces (int): The maximum number of faces to detect. Defaults to 0.
            min_face_size (int): The minimum size of a face to detect. Defaults to 0.
            detect_masks (bool): Whether to detect masks. Defaults to False.
            multipart (bool): Whether the input is multipart data. Defaults to False.
            dl_client (aiohttp.ClientSession): An asynchronous HTTP client session. Defaults to None.

        Returns:
            bytes: The image with drawn faces.
        """

        if not multipart:
            images = await get_images(images, session=self.dl_client)
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

    async def _get_single_image_faces(
        self,
        image_b64: str,
        limit_faces: int,
        min_face_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Helper function to decode an image and get face embeddings for it.
        Encapsulates the logic for processing a single image.
        """
        images_obj = Images(data=[image_b64])
        processed_images = await get_images(
            images_obj, decode=self.model.decode_required, session=self.dl_client
        )

        if not processed_images:
            return []  # Return empty list if image is invalid

        cv2_image = processed_images[0].get("data")
        if cv2_image is None or cv2_image.size == 0:
            return []

        # model.get expects a list of images, so we wrap it
        faces = await self.model.get(
            [cv2_image],
            threshold=0.6,
            extract_embedding=True,
            limit_faces=limit_faces,
            min_face_size=min_face_size,
        )
        return faces[0] if faces else []

    async def verify(
        self,
        image1: str,
        image2: str,
        threshold: float = 0.3,
        limit_faces: int = 0,
        min_face_size: int = 0,
        **kwargs,
    ):
        """
        Verifies if two images contain the same person using parallel processing.

        Args:
            image1 (str): Base64 encoded first image.
            image2 (str): Base64 encoded second image.
            threshold (float): Similarity threshold (0.0-1.0). Defaults to 0.3.
            limit_faces (int): The maximum number of faces to detect per image. Defaults to 0.
            min_face_size (int): The minimum size of a face to detect. Defaults to 0.

        Returns:
            Dict: A dictionary containing the verification result.
        """
        try:
            # 1. Create and run face detection tasks for both images in parallel
            task1 = self._get_single_image_faces(image1, limit_faces, min_face_size)
            task2 = self._get_single_image_faces(image2, limit_faces, min_face_size)

            faces1, faces2 = await asyncio.gather(task1, task2)

            # 2. Validate that faces were found in both images
            if not faces1:
                return {
                    "is_same_person": False,
                    "similarity_score": -1.0,
                    "error_message": "No face detected in the first image.",
                }
            if not faces2:
                return {
                    "is_same_person": False,
                    "similarity_score": -1.0,
                    "error_message": "No face detected in the second image.",
                }

            # 3. Select the most prominent face from each image
            best_face1 = sorted(faces1, key=lambda x: x["prob"], reverse=True)[0]
            best_face2 = sorted(faces2, key=lambda x: x["prob"], reverse=True)[0]

            # 4. Compare embeddings
            embedding1 = best_face1.get("vec")
            embedding2 = best_face2.get("vec")

            if embedding1 is None or embedding2 is None:
                return {
                    "is_same_person": False,
                    "similarity_score": -1.0,
                    "error_message": "Could not extract embedding vectors for one or both faces.",
                }

            similarity = np.dot(embedding1, embedding2)
            logger.debug(f"Similarity score: {similarity}")

            return {
                "is_same_person": bool(similarity > threshold),
                "similarity_score": float(similarity),
                "error_message": "",
            }

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {
                "is_same_person": False,
                "similarity_score": -1.0,
                "error_message": f"An unexpected error occurred during verification: {e}",
            }


processing: Processing | None = None


async def get_processing() -> Processing:
    global processing
    settings = Settings()
    if not processing:
        processing = Processing(
            det_name=settings.models.det_name,
            rec_name=settings.models.rec_name,
            ga_name=settings.models.ga_name,
            mask_detector=settings.models.mask_detector,
            max_size=settings.models.max_size,
            max_rec_batch_size=settings.models.rec_batch_size,
            max_det_batch_size=settings.models.det_batch_size,
            backend_name=settings.models.inference_backend,
            force_fp16=settings.models.force_fp16,
            triton_uri=settings.models.triton_uri,
            root_dir="/models",
        )
    return processing


ProcessingDep = Annotated[Processing, Depends(get_processing)]
