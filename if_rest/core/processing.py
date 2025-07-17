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
from if_rest.settings import Settings

f_model = os.getenv("FACE_MODEL_CLASS", "face_model")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
face_model = importlib.import_module(f_model, package=None)
FaceAnalysis = face_model.FaceAnalysis


class Processing:

    def __init__(self,
                 det_name: str = 'retinaface_r50_v1',
                 rec_name: str = 'arcface_r100_v1',
                 ga_name: str = 'genderage_v1',
                 mask_detector: str = 'mask_detector',
                 max_size: List[int] = None,
                 backend_name: str = 'trt',
                 max_rec_batch_size: int = 1,
                 max_det_batch_size: int = 1,
                 force_fp16: bool = False,
                 triton_uri=None,
                 root_dir: str = '/models',
                 **kwargs):
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
        self.model = FaceAnalysis(det_name=self.det_name,
                                  rec_name=self.rec_name,
                                  ga_name=self.ga_name,
                                  mask_detector=self.mask_detector,
                                  max_size=self.max_size,
                                  max_rec_batch_size=self.max_rec_batch_size,
                                  max_det_batch_size=self.max_det_batch_size,
                                  backend_name=self.backend_name,
                                  force_fp16=self.force_fp16,
                                  triton_uri=self.triton_uri,
                                  root_dir=self.root_dir
                                  )

    async def extract(self,
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
                      **kwargs):
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
        images = await get_images(images, decode=self.model.decode_required, session=self.dl_client,
                                  b64_decode=b64_decode, headers=img_req_headers)

        tl1 = time.time()
        took_loading = tl1 - tl0
        logger.debug(f'Reading images took: {took_loading * 1000:.3f} ms.')

        if embed_only:
            _faces_dict = await self.model.embed_crops(images,
                                                       extract_embedding=extract_embedding,
                                                       extract_ga=extract_ga,
                                                       detect_masks=detect_masks)
            return _faces_dict

        else:
            te0 = time.time()
            output = await self.model.embed(images,
                                            max_size=max_size,
                                            return_face_data=return_face_data,
                                            threshold=threshold,
                                            limit_faces=limit_faces,
                                            min_face_size=min_face_size,
                                            extract_embedding=extract_embedding,
                                            extract_ga=extract_ga,
                                            return_landmarks=return_landmarks,
                                            detect_masks=detect_masks
                                            )
            took_embed = time.time() - te0
            took = time.time() - t0
            output['took']['total_ms'] = took * 1000
            if verbose_timings:
                output['took']['read_imgs_ms'] = took_loading * 1000
                output['took']['embed_all_ms'] = took_embed * 1000

            return output

    async def draw(self,
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
                   **kwargs):

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
            image = images[0].get('data')
        else:
            __bin = np.fromstring(images, np.uint8)
            image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)

        faces = await self.model.get([image],
                                     threshold=threshold,
                                     return_face_data=False,
                                     extract_embedding=False,
                                     extract_ga=False,
                                     limit_faces=limit_faces,
                                     min_face_size=min_face_size,
                                     detect_masks=detect_masks)

        image = np.ascontiguousarray(image)
        image = self.model.draw_faces(image, faces[0],
                                      draw_landmarks=draw_landmarks,
                                      draw_scores=draw_scores,
                                      draw_sizes=draw_sizes)

        is_success, buffer = cv2.imencode(".jpg", image)
        io_buf = io.BytesIO(buffer)
        return io_buf


    async def verify(self,
                     images: Images,
                     threshold: float = 0.3,
                     limit_faces: int = 0,
                     min_face_size: int = 0,
                     **kwargs):
        """
        Verifies if two images contain the same person.

        Args:
            images (Images): An Images object containing a list of image data or links.
            threshold (float): The threshold for face verification. Defaults to 0.3.
            limit_faces (int): The maximum number of faces to detect. Defaults to 0.
            min_face_size (int): The minimum size of a face to detect. Defaults to 0.

        Returns:
            Dict: A dictionary containing verification result.
        """
        t0 = time.time()
        
        result = {
            "is_same_person": False,
            "similarity_score": -1.0,
            "error_message": "",
            "took": {
                "total_ms": 0
            }
        }

        # Process the incoming images object
        processed_images = await get_images(images, decode=self.model.decode_required, session=self.dl_client)

        if len(processed_images) < 2:
            result["error_message"] = "Verification requires at least two images."
            return result

        # Take the first two images for comparison
        img1_cv2 = processed_images[0].get('data')
        img2_cv2 = processed_images[1].get('data')

        if img1_cv2 is None or img2_cv2 is None:
            result["error_message"] = "Failed to load one or both images."
            return result

        # Get faces from both images
        t_det_start = time.time()
        faces1_list = await self.model.get([img1_cv2],
                                      threshold=0.5,
                                      extract_embedding=True,
                                      limit_faces=limit_faces,
                                      min_face_size=min_face_size)

        faces2_list = await self.model.get([img2_cv2],
                                      threshold=0.5,
                                      extract_embedding=True,
                                      limit_faces=limit_faces,
                                      min_face_size=min_face_size)
        
        took_detection = time.time() - t_det_start
        
        faces1 = faces1_list[0] if faces1_list else []
        faces2 = faces2_list[0] if faces2_list else []

        # Check if faces were found
        if not faces1:
            result["error_message"] = "No face detected in the first image."
            return result
        if not faces2:
            result["error_message"] = "No face detected in the second image."
            return result

        # Use the face with the highest detection score
        face1 = sorted(faces1, key=lambda x: x["prob"], reverse=True)[0]
        face2 = sorted(faces2, key=lambda x: x["prob"], reverse=True)[0]

        # Compare embeddings
        embedding1 = face1['vec']
        embedding2 = face2['vec']

        try:
            similarity = np.dot(embedding1, embedding2)
            
            result["similarity_score"] = float(similarity)
            result["is_same_person"] = bool(similarity > threshold)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            result["error_message"] = f"An unexpected error occurred during comparison: {e}"

        # took_total = time.time() - t0
        # result["took"]["total_ms"] = took_total * 1000
        # result["took"]["detection_ms"] = took_detection * 1000
        return result

processing: Processing | None = None


async def get_processing() -> Processing:
    global processing
    settings = Settings()
    if not processing:
        processing = Processing(det_name=settings.models.det_name, rec_name=settings.models.rec_name,
                                ga_name=settings.models.ga_name,
                                mask_detector=settings.models.mask_detector,
                                max_size=settings.models.max_size,
                                max_rec_batch_size=settings.models.rec_batch_size,
                                max_det_batch_size=settings.models.det_batch_size,
                                backend_name=settings.models.inference_backend,
                                force_fp16=settings.models.force_fp16,
                                triton_uri=settings.models.triton_uri,
                                root_dir='/models'
                                )
    return processing


ProcessingDep = Annotated[Processing, Depends(get_processing)]
