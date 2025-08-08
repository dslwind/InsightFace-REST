import asyncio
import base64
import collections
import time
import traceback
from typing import Any, Dict, List

import cv2
import numpy as np
from numpy.linalg import norm

from if_rest.core.model_zoo.getter import get_model
from if_rest.core.utils import fast_face_align as face_align
from if_rest.core.utils.helpers import colorize_log, to_chunks
from if_rest.core.utils.image_provider import resize_image
from if_rest.logger import logger
from if_rest.settings import AppSettings

# 使用 collections.namedtuple 定义一个标准的人脸数据结构
Face = collections.namedtuple(
    "Face",
    [
        "bbox", "landmark", "det_score", "embedding", "gender", "age",
        "embedding_norm", "normed_embedding", "facedata", "scale",
        "num_det", "mask", "mask_probs",
    ],
    defaults=(None,) * 13
)


def serialize_face(face_dict: dict, return_face_data: bool, return_landmarks: bool = False) -> dict:
    """
    序列化人脸字典，以便作为 API 响应返回。
    """
    # 规范化数值类型
    if face_dict.get("norm") is not None:
        face_dict.update(
            vec=face_dict["vec"].tolist(),
            norm=float(face_dict["norm"]),
        )
    if face_dict.get("prob") is not None:
        face_dict.update(
            prob=float(face_dict["prob"]),
            bbox=face_dict["bbox"].astype(int).tolist(),
            size=int(face_dict["bbox"][2] - face_dict["bbox"][0]),
        )

    # 根据请求参数决定是否返回关键点
    if return_landmarks and face_dict.get("landmarks") is not None:
        face_dict["landmarks"] = face_dict["landmarks"].astype(int).tolist()
    else:
        face_dict.pop("landmarks", None)

    # 根据请求参数决定是否返回 Base64 编码的人脸图像
    if return_face_data and face_dict.get("facedata") is not None:
        face_dict["facedata"] = base64.b64encode(
            cv2.imencode(".jpg", face_dict["facedata"])[1].tobytes()
        ).decode("ascii")
    else:
        face_dict.pop("facedata", None)

    return face_dict


class Detector:
    """
    一个封装了人脸检测模型的包装类。
    """
    def __init__(self, settings: AppSettings, model_meta: Dict[str, Any]):
        """
        初始化检测器。

        Args:
            settings (AppSettings): 应用配置。
            model_meta (Dict[str, Any]): 模型元数据。
        """
        self.detector_model = get_model(
            model_name=settings.models.det_name,
            settings=settings,
            model_meta=model_meta,
            im_size=settings.models.max_size,
            max_batch_size=settings.models.det_batch_size,
            force_fp16=settings.models.force_fp16,
            download_model=False,
        )
        self.detector_model.prepare(nms=0.35)

    def detect(self, data, threshold=0.3):
        """
        对输入图像进行人脸检测，并统一输出格式。

        Args:
            data (np.ndarray): 批处理的图像数据。
            threshold (float): 检测置信度阈值。

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: 
            返回边界框、置信度和关键点列表。
        """
        all_bboxes, all_landmarks = self.detector_model.detect(data, threshold=threshold)

        # 将 bboxes 分解为 boxes 和 probs
        boxes = [e[:, 0:4] for e in all_bboxes]
        probs = [e[:, 4] for e in all_bboxes]

        return boxes, probs, all_landmarks


class FaceAnalysis:
    def __init__(self, settings: AppSettings, model_meta: Dict[str, Any], **kwargs):
        """
        人脸分析的核心类。

        Args:
            settings (AppSettings): 统一的应用配置对象。
            model_meta (Dict[str, Any]): 从 models.json 加载的模型元数据。
        """
        self.settings = settings
        self.model_meta = model_meta
        self.decode_required = True

        # 初始化检测器
        self.det_model = Detector(settings=settings, model_meta=model_meta)

        # 初始化识别模型
        self.rec_model = None
        if settings.models.rec_name:
            self.rec_model = get_model(
                model_name=settings.models.rec_name,
                settings=settings,
                model_meta=model_meta,
                max_batch_size=settings.models.rec_batch_size,
                force_fp16=settings.models.force_fp16,
                download_model=False,
            )
            self.rec_model.prepare()

        # 初始化性别年龄模型
        self.ga_model = None
        if settings.models.ga_name:
            self.ga_model = get_model(
                model_name=settings.models.ga_name,
                settings=settings,
                model_meta=model_meta,
                max_batch_size=settings.models.rec_batch_size,
                force_fp16=settings.models.force_fp16,
                download_model=False,
            )
            self.ga_model.prepare()

        # 初始化口罩检测模型
        self.mask_model = None
        if settings.models.mask_detector:
            self.mask_model = get_model(
                model_name=settings.models.mask_detector,
                settings=settings,
                model_meta=model_meta,
                max_batch_size=settings.models.rec_batch_size,
                force_fp16=settings.models.force_fp16,
                download_model=False,
            )
            self.mask_model.prepare()

    def _sort_faces_by_centrality(self, boxes, probs, landmarks, image_shape, max_num=0):
        """按人脸在图像中心的程度和大小对检测结果进行排序。"""
        if max_num > 0 and boxes.shape[0] > max_num:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            indices = np.argsort(areas)[::-1][:max_num]
            return boxes[indices, :], probs[indices], landmarks[indices, :]
        return boxes, probs, landmarks

    def _run_feature_extraction_batch(self, crops, extract_embedding, extract_ga, detect_masks, mask_thresh):
        """在一个批次上同步运行所有特征提取模型。"""
        batch_size = len(crops)
        embeddings, gender_age_results, mask_results = [None] * batch_size, [[None, None]] * batch_size, [None] * batch_size

        if extract_embedding and self.rec_model:
            embeddings = self.rec_model.get_embedding(crops)
        if extract_ga and self.ga_model:
            gender_age_results = self.ga_model.get(crops)
        if detect_masks and self.mask_model:
            mask_predictions = self.mask_model.get(crops)
            mask_results = []
            for mask_pred in mask_predictions:
                mask_prob, no_mask_prob = float(mask_pred[0]), float(mask_pred[1])
                is_mask = mask_prob > no_mask_prob and mask_prob >= mask_thresh
                mask_results.append({"is_mask": is_mask, "probs": {"mask": mask_prob, "no_mask": no_mask_prob}})
        return embeddings, gender_age_results, mask_results

    async def _process_faces_in_batches(self, faces, **kwargs):
        """异步处理所有检测到的人脸，分批次进行特征提取。"""
        processed_faces = []
        face_chunks = to_chunks(faces, self.settings.models.rec_batch_size)
        mask_thresh = kwargs.get("mask_thresh", 0.89)
        
        for chunk in face_chunks:
            chunk = list(chunk)
            crops = [face["facedata"] for face in chunk]
            
            embeddings, ga_results, mask_results = await asyncio.to_thread(
                self._run_feature_extraction_batch, 
                crops, 
                kwargs.get("extract_embedding", True),
                kwargs.get("extract_ga", True),
                kwargs.get("detect_masks", True),
                mask_thresh
            )
            
            for i, face_data in enumerate(chunk):
                embedding, (gender, age), mask_info = embeddings[i], ga_results[i], mask_results[i]

                if embedding is not None:
                    embedding_norm = norm(embedding)
                    face_data.update(norm=embedding_norm, vec=embedding / embedding_norm)
                if gender is not None:
                    face_data.update(gender=int(gender), age=age)
                if mask_info is not None:
                    face_data.update(mask=mask_info["is_mask"], mask_probs=mask_info["probs"])
                if not kwargs.get("return_face_data", False):
                    face_data["facedata"] = None
                processed_faces.append(face_data)
        
        return processed_faces

    def _run_detection_batch(self, images_batch, threshold):
        """在一个批次上同步运行检测模型。"""
        return self.det_model.detect(images_batch, threshold=threshold)

    async def get(self, images, **kwargs):
        """
        异步处理一系列图像，执行人脸检测和特征提取。
        """
        total_start_time = time.perf_counter()
        
        max_size = kwargs.get("max_size") or self.settings.models.max_size
        threshold = kwargs.get("threshold", 0.6)
        limit_faces = kwargs.get("limit_faces", 0)
        min_face_size = kwargs.get("min_face_size", 0)

        resized_images_with_scales = [resize_image(img, max_size) for img in images]
        resized_images, scales = zip(*resized_images_with_scales)
        
        all_faces, faces_per_image = [], {}
        image_batches = to_chunks(resized_images, self.settings.models.det_batch_size)
        
        for batch_index, image_batch in enumerate(image_batches):
            image_batch = list(image_batch)
            detection_results = await asyncio.to_thread(self._run_detection_batch, image_batch, threshold)
            
            for i, (boxes, probs, landmarks) in enumerate(zip(*detection_results)):
                original_image_index = batch_index * self.settings.models.det_batch_size + i
                
                if boxes is None or len(boxes) == 0:
                    faces_per_image[original_image_index] = 0
                    continue

                if limit_faces > 0:
                    boxes, probs, landmarks = self._sort_faces_by_centrality(
                        boxes, probs, landmarks, image_batch[i].shape, limit_faces
                    )
                
                scale = scales[original_image_index]
                if scale != 1.0:
                    boxes /= scale
                    landmarks /= scale
                
                crops = face_align.norm_crop_batched(images[original_image_index], landmarks)
                
                num_valid_faces = 0
                for j, crop in enumerate(crops):
                    if min_face_size > 0 and (boxes[j][2] - boxes[j][0]) < min_face_size:
                        continue
                    
                    all_faces.append({
                        "bbox": boxes[j], "landmarks": landmarks[j], "prob": probs[j],
                        "num_det": j, "scale": scale, "facedata": crop
                    })
                    num_valid_faces += 1
                faces_per_image[original_image_index] = num_valid_faces

        if all_faces:
            all_faces = await self._process_faces_in_batches(all_faces, **kwargs)
        
        output_by_image, current_pos = [], 0
        for i in range(len(images)):
            num_faces = faces_per_image.get(i, 0)
            output_by_image.append(all_faces[current_pos : current_pos + num_faces])
            current_pos += num_faces
            
        total_duration_ms = (time.perf_counter() - total_start_time) * 1000
        logger.debug(colorize_log(f"完整处理流程耗时: {total_duration_ms:.3f} ms.", "red"))
        
        return output_by_image

    async def embed_crops(self, images: List[Dict], **kwargs) -> Dict:
        """
        为已裁剪的人脸图像（112x112）提取特征。
        """
        total_start_time = time.time()
        output = {"took_ms": None, "data": [], "status": "ok"}

        valid_crops = [{"facedata": img.get("data")} for img in images if img.get("traceback") is None]
        processed_faces = await self._process_faces_in_batches(valid_crops, **kwargs) if valid_crops else []
        
        processed_iter = iter(processed_faces)
        for image_data in images:
            if image_data.get("traceback"):
                face_dict = {"status": "failed", "traceback": image_data.get("traceback")}
            else:
                try:
                    face_dict = next(processed_iter)
                    face_dict.update({"bbox": [0, 0, 112, 112], "size": 112, "prob": 1.0, "status": "ok"})
                    face_dict = serialize_face(
                        face_dict,
                        return_face_data=kwargs.get("return_face_data", False),
                        return_landmarks=kwargs.get("return_landmarks", False)
                    )
                except StopIteration:
                     face_dict = {"status": "failed", "traceback": "Processing error: not enough results."}
            
            output["data"].append({"status": "ok", "took_ms": 0, "faces": [face_dict]})

        output["took_ms"] = (time.time() - total_start_time) * 1000
        return output

    async def embed(self, images: List[Dict], **kwargs) -> Dict:
        """
        处理完整图像，执行检测和特征提取，并格式化为最终 API 响应。
        """
        output = {"took": {}, "data": []}
        
        valid_images = [img["data"] for img in images if img.get("traceback") is None]
        faces_by_image = await self.get(valid_images, **kwargs) if valid_images else []

        faces_iter = iter(faces_by_image)
        for image_data in images:
            response_item = {"status": "failed", "took_ms": 0.0, "faces": []}
            if image_data.get("traceback"):
                response_item["traceback"] = image_data.get("traceback")
            else:
                try:
                    start_time = time.perf_counter()
                    faces = next(faces_iter)
                    # *** BUG FIX START ***
                    # 显式地从 kwargs 中提取 serialize_face 需要的参数
                    # 而不是传递整个 kwargs 字典
                    response_item["faces"] = [
                        serialize_face(
                            face,
                            return_face_data=kwargs.get("return_face_data", False),
                            return_landmarks=kwargs.get("return_landmarks", False)
                        ) for face in faces
                    ]
                    # *** BUG FIX END ***
                    response_item["status"] = "ok"
                    response_item["took_ms"] = (time.perf_counter() - start_time) * 1000
                except StopIteration:
                    response_item["traceback"] = "Processing error: not enough results."
                except Exception:
                    response_item["traceback"] = traceback.format_exc()
            
            output["data"].append(response_item)
            
        return output

    def draw_faces(self, image, faces, draw_landmarks=True, draw_scores=True, draw_sizes=True):
        """
        在图像上绘制检测到的人脸框、关键点等信息。
        """
        for face in faces:
            bbox = face["bbox"].astype(int)
            pt1, pt2 = tuple(bbox[0:2]), tuple(bbox[2:4])
            color = (0, 0, 255) if face.get("mask") is False else (0, 255, 0)
            x, y = pt1
            w = bbox[2] - bbox[0]
            cv2.rectangle(image, pt1, pt2, color, 1)

            if draw_landmarks and face.get("landmarks") is not None:
                lms = face["landmarks"].astype(int)
                pt_size = max(int(w * 0.05), 1)
                colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
                for i in range(5):
                    cv2.circle(image, (lms[i][0], lms[i][1]), 1, colors[i], pt_size)

            if draw_scores and face.get("prob") is not None:
                text, pos = f"{face['prob']:.3f}", (x + 3, y - 5)
                cv2.rectangle(image, (x - 1, y - 21, w + 2, 21), color, -1)
                cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, (255, 255, 255), 1, 16)
            
            if draw_sizes:
                text, pos = f"w:{w}", (x + 3, bbox[3] - 5)
                cv2.putText(image, text, pos, 0, 0.5, (0, 0, 0), 3, 16)
                cv2.putText(image, text, pos, 0, 0.5, (0, 255, 0), 1, 16)

        total_text = f"faces: {len(faces)} ({self.settings.models.det_name})"
        bottom = image.shape[0]
        cv2.putText(image, total_text, (5, bottom - 5), 0, 1, (0, 0, 0), 3, 16)
        cv2.putText(image, total_text, (5, bottom - 5), 0, 1, (0, 255, 0), 1, 16)

        return image
