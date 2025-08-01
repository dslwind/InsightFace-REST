from typing import Annotated, Callable, List, Optional

import msgpack
from fastapi import (APIRouter, File, Form, Header, HTTPException, Request,
                     Response)
from fastapi.responses import UJSONResponse
from fastapi.routing import APIRoute
from starlette.responses import PlainTextResponse, StreamingResponse

from if_rest.core.processing import ProcessingDep
from if_rest.schemas import BodyDraw, BodyExtract, FaceVerification


class MsgPackRequest(Request):
    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            if "application/msgpack" in self.headers.getlist("Content-Type"):
                body = msgpack.unpackb(body)
            self._body = body
        return self._body


class MsgpackRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = MsgPackRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


router = APIRouter(route_class=MsgpackRoute)


@router.post('/extract', tags=['Detection & recognition'])
async def extract(data: BodyExtract,
                  processing: ProcessingDep,
                  accept: Optional[List[str]] = Header(None),
                  content_type: Annotated[str | None, Header()] = None):
    """
    Face extraction/embeddings endpoint accept json with
    parameters in following format:

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **embed_only**: Treat input images as face crops (112x112 crops required), omit detection step. Default: False (*optional*)
       - **return_face_data**: Return face crops encoded in base64. Default: False (*optional*)
       - **return_landmarks**: Return face landmarks. Default: False (*optional*)
       - **extract_embedding**: Extract face embeddings (otherwise only detect faces). Default: True (*optional*)
       - **extract_ga**: Extract gender/age. Default: False (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       - **verbose_timings**: Return all timings. Default: False (*optional*)
       - **msgpack**: Serialize output to msgpack format for transfer. Default: False (*optional*)
       \f

       :return:
       List[List[dict]]
    """
    try:
        b64_decode = True
        if content_type == 'application/msgpack':
            b64_decode = False
        output = await processing.extract(data.images, return_face_data=data.return_face_data,
                                          embed_only=data.embed_only, extract_embedding=data.extract_embedding,
                                          threshold=data.threshold, extract_ga=data.extract_ga,
                                          limit_faces=data.limit_faces, min_face_size=data.min_face_size,
                                          return_landmarks=data.return_landmarks,
                                          detect_masks=data.detect_masks,
                                          verbose_timings=data.verbose_timings, b64_decode=b64_decode,
                                          img_req_headers=data.img_req_headers)

        if data.msgpack or 'application/x-msgpack' in accept:
            return PlainTextResponse(msgpack.dumps(output, use_single_float=True), media_type='application/x-msgpack')
        else:
            return UJSONResponse(output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/draw_detections', tags=['Detection & recognition'])
async def draw(data: BodyDraw,
               processing: ProcessingDep):
    """
    Return image with drawn faces for testing purposes.

       - **images**: dict containing either links or data lists. (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """
    try:
        output = await processing.draw(data.images, threshold=data.threshold,
                                       draw_landmarks=data.draw_landmarks, draw_scores=data.draw_scores,
                                       limit_faces=data.limit_faces, min_face_size=data.min_face_size,
                                       draw_sizes=data.draw_sizes,
                                       detect_masks=data.detect_masks)
        output.seek(0)
        return StreamingResponse(output, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/multipart/draw_detections', tags=['Detection & recognition'])
async def draw_upl(processing: ProcessingDep, file: bytes = File(...), threshold: float = Form(0.6),
                   draw_landmarks: bool = Form(True),
                   draw_scores: bool = Form(True), draw_sizes: bool = Form(True), limit_faces: int = Form(0),
                   use_rotation: bool = Form(False)):
    """
    Return image with drawn faces for testing purposes.

       - **file**: Image file (*required*)
       - **threshold**: Detection threshold. Default: 0.6 (*optional*)
       - **draw_landmarks**: Draw faces landmarks Default: True (*optional*)
       - **draw_scores**: Draw detection scores Default: True (*optional*)
       - **draw_sizes**: Draw face sizes Default: True (*optional*)
       - **limit_faces**: Maximum number of faces to be processed.  0 for unlimited number. Default: 0 (*optional*)
       \f
    """
    try:
        output = await processing.draw(file, threshold=threshold,
                                       draw_landmarks=draw_landmarks, draw_scores=draw_scores, draw_sizes=draw_sizes,
                                       limit_faces=limit_faces,
                                       multipart=True)
        output.seek(0)
        return StreamingResponse(output, media_type='image/jpg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/verify', tags=['Detection & recognition'])
async def verify(
    data: FaceVerification,
    processing: ProcessingDep
):
    """
    Compare two faces from the first two images provided and verify if they are from the same person.

       - **images**: dict containing either links or data lists with at least two images. (*required*)
       - **threshold**: Similarity threshold. Faces with a score above this value are considered the same person. Default: 0.3 (*optional*)
       \f
    """
    try:
        # Use the threshold from the request if provided, otherwise default to 0.3 for verification
        threshold = data.threshold if data.threshold is not None else 0.3
        result = await processing.verify(
            images=data.images,
            threshold=threshold
        )
        # logger.info(result)
        return UJSONResponse(dict(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))