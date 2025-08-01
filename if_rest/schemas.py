from typing import List, Optional

import pydantic
from pydantic import BaseModel

from if_rest.settings import Settings

example_img = 'test_images/Stallone.jpg'
# Read runtime settings from environment variables
settings = Settings()

dummy_base64_string_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
dummy_base64_string_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFhAJ/wlseKgAAAABJRU5ErkJggg=="

face_verification_example = {
    "images": {
        "data": [
            dummy_base64_string_1,
            dummy_base64_string_2
        ],
        "urls": [
            'test_images/TH.png',
            'test_images/TH1.jpg'
        ]
    },
    "threshold": 0.3,
    "limit_faces": 1,
    "min_face_size": 0,
    "verbose_timings": False,
    "msgpack": False
}

class Images(BaseModel):
    data: Optional[List[str] | List[bytes]] = pydantic.Field(default=None, example=None,
                                                             description='List of base64 encoded images')
    urls: Optional[List[str]] = pydantic.Field(default=None,
                                               example=[example_img],
                                               description='List of images urls')


class BodyExtract(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(default=settings.defaults.det_thresh,
                                                example=settings.defaults.det_thresh,
                                                description='Detector threshold')

    embed_only: Optional[bool] = pydantic.Field(default=False,
                                                example=False,
                                                description='Treat input images as face crops and omit detection step')

    return_face_data: Optional[bool] = pydantic.Field(default=settings.defaults.return_face_data,
                                                      example=settings.defaults.return_face_data,
                                                      description='Return face crops encoded in base64')

    return_landmarks: Optional[bool] = pydantic.Field(default=settings.defaults.return_landmarks,
                                                      example=settings.defaults.return_landmarks,
                                                      description='Return face landmarks')

    extract_embedding: Optional[bool] = pydantic.Field(default=settings.defaults.extract_embedding,
                                                       example=settings.defaults.extract_embedding,
                                                       description='Extract face embeddings (otherwise only detect \
                                                       faces)')

    extract_ga: Optional[bool] = pydantic.Field(default=settings.defaults.extract_ga,
                                                example=settings.defaults.extract_ga,
                                                description='Extract gender/age')

    detect_masks: Optional[bool] = pydantic.Field(default=settings.defaults.detect_masks,
                                                  example=settings.defaults.detect_masks,
                                                  description='Detect medical masks')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = pydantic.Field(default=False,
                                                     example=True,
                                                     description='Return all timings.')

    msgpack: Optional[bool] = pydantic.Field(default=False,
                                             example=False,
                                             description='Use MSGPACK for response serialization')
    img_req_headers: Optional[dict] = pydantic.Field(default={},
                                                     example={},
                                                     description='Custom headers for image retrieving from remote servers')


class BodyDraw(BaseModel):
    images: Images

    threshold: Optional[float] = pydantic.Field(default=settings.defaults.det_thresh,
                                                example=settings.defaults.det_thresh,
                                                description='Detector threshold')

    draw_landmarks: Optional[bool] = pydantic.Field(default=True,
                                                    example=True,
                                                    description='Return face landmarks')

    draw_scores: Optional[bool] = pydantic.Field(default=True,
                                                 example=True,
                                                 description='Draw detection scores')

    draw_sizes: Optional[bool] = pydantic.Field(default=True,
                                                example=True,
                                                description='Draw face sizes')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')

    detect_masks: Optional[bool] = pydantic.Field(default=settings.defaults.detect_masks,
                                                  example=settings.defaults.detect_masks,
                                                  description='Detect medical masks')

class FaceVerification(BaseModel):
    images: Images = pydantic.Field(
        example=face_verification_example["images"]
    )

    threshold: Optional[float] = pydantic.Field(default=0.3,
                                                example=0.3,
                                                description='The threshold for face verification.')

    limit_faces: Optional[int] = pydantic.Field(default=0,
                                                example=0,
                                                description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = pydantic.Field(default=0,
                                                  example=0,
                                                  description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = pydantic.Field(default=False,
                                                     example=True,
                                                     description='Return all timings.')

    msgpack: Optional[bool] = pydantic.Field(default=False,
                                             example=False,
                                             description='Use MSGPACK for response serialization')
    
    class Config:
        json_schema_extra = {
            "example": face_verification_example
        }