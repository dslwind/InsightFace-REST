from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

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
        ]
    },
    "threshold": 0.3,
    "det_threshold": 0.6,
    "limit_faces": 1,
    "min_face_size": 0,
    "verbose_timings": False,
    "msgpack": False
}


class Images(BaseModel):
    data: Optional[List[str] | List[bytes]] = Field(
        default=None,
        example=None,
        description='List of base64 encoded images',
    )
    urls: Optional[List[str]] = Field(
        default=None,
        example=[example_img],
        description='List of images urls',
    )

    @model_validator(mode='after')
    def validate_sources(self):
        has_data = bool(self.data)
        has_urls = bool(self.urls)
        if has_data == has_urls:
            raise ValueError("Provide exactly one of 'data' or 'urls', with at least one image.")
        return self


class BodyExtract(BaseModel):
    images: Images

    threshold: Optional[float] = Field(default=settings.defaults.det_thresh,
                                       ge=0.0,
                                       le=1.0,
                                       example=settings.defaults.det_thresh,
                                       description='Detector threshold')

    embed_only: Optional[bool] = Field(default=False,
                                       example=False,
                                       description='Treat input images as face crops and omit detection step')

    return_face_data: Optional[bool] = Field(default=settings.defaults.return_face_data,
                                             example=settings.defaults.return_face_data,
                                             description='Return face crops encoded in base64')

    return_landmarks: Optional[bool] = Field(default=settings.defaults.return_landmarks,
                                             example=settings.defaults.return_landmarks,
                                             description='Return face landmarks')

    extract_embedding: Optional[bool] = Field(default=settings.defaults.extract_embedding,
                                              example=settings.defaults.extract_embedding,
                                              description='Extract face embeddings (otherwise only detect \
                                              faces)')

    extract_ga: Optional[bool] = Field(default=settings.defaults.extract_ga,
                                       example=settings.defaults.extract_ga,
                                       description='Extract gender/age')

    detect_masks: Optional[bool] = Field(default=settings.defaults.detect_masks,
                                         example=settings.defaults.detect_masks,
                                         description='Detect medical masks')

    limit_faces: Optional[int] = Field(default=0,
                                       ge=0,
                                       example=0,
                                       description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = Field(default=0,
                                         ge=0,
                                         example=0,
                                         description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = Field(default=False,
                                            example=True,
                                            description='Return all timings.')

    msgpack: Optional[bool] = Field(default=False,
                                    example=False,
                                    description='Use MSGPACK for response serialization')
    img_req_headers: Optional[dict] = Field(default_factory=dict,
                                            example={},
                                            description='Custom headers for image retrieving from remote servers')


class BodyDraw(BaseModel):
    images: Images

    threshold: Optional[float] = Field(default=settings.defaults.det_thresh,
                                       ge=0.0,
                                       le=1.0,
                                       example=settings.defaults.det_thresh,
                                       description='Detector threshold')

    draw_landmarks: Optional[bool] = Field(default=True,
                                           example=True,
                                           description='Return face landmarks')

    draw_scores: Optional[bool] = Field(default=True,
                                        example=True,
                                        description='Draw detection scores')

    draw_sizes: Optional[bool] = Field(default=True,
                                       example=True,
                                       description='Draw face sizes')

    limit_faces: Optional[int] = Field(default=0,
                                       ge=0,
                                       example=0,
                                       description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = Field(default=0,
                                         ge=0,
                                         example=0,
                                         description='Ignore faces smaller than this size')

    detect_masks: Optional[bool] = Field(default=settings.defaults.detect_masks,
                                         example=settings.defaults.detect_masks,
                                         description='Detect medical masks')

class FaceVerification(BaseModel):
    images: Images = Field(
        example=face_verification_example["images"]
    )

    threshold: Optional[float] = Field(default=0.3,
                                       ge=-1.0,
                                       le=1.0,
                                       example=0.3,
                                       description='The threshold for face verification.')

    det_threshold: Optional[float] = Field(default=settings.defaults.det_thresh,
                                           ge=0.0,
                                           le=1.0,
                                           example=settings.defaults.det_thresh,
                                           description='Detector threshold used before verification.')

    limit_faces: Optional[int] = Field(default=0,
                                       ge=0,
                                       example=0,
                                       description='Maximum number of faces to be processed')

    min_face_size: Optional[int] = Field(default=0,
                                         ge=0,
                                         example=0,
                                         description='Ignore faces smaller than this size')

    verbose_timings: Optional[bool] = Field(default=False,
                                            example=True,
                                            description='Return all timings.')

    msgpack: Optional[bool] = Field(default=False,
                                    example=False,
                                    description='Use MSGPACK for response serialization')

    @model_validator(mode='after')
    def validate_num_images(self):
        image_items = self.images.data if self.images.data is not None else self.images.urls
        if len(image_items) < 2:
            raise ValueError("Verification requires at least two images.")
        return self

    model_config = {
        "json_schema_extra": {
            "example": face_verification_example
        }
    }
