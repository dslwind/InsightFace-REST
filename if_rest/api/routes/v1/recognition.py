from typing import Annotated, Callable, Optional

import msgpack
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import UJSONResponse
from fastapi.routing import APIRoute
from starlette.responses import StreamingResponse

from if_rest.core.processing import ProcessingDep
from if_rest.schemas import BodyDraw, BodyExtract, FaceVerification


class MsgPackRequest(Request):
    """自定义 Request 类以处理 msgpack 格式的请求体。"""

    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            body = await super().body()
            if "application/msgpack" in self.headers.get("Content-Type", ""):
                body = msgpack.unpackb(body)
            self._body = body
        return self._body


class MsgpackRoute(APIRoute):
    """自定义路由类，以使用 MsgPackRequest。"""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            request = MsgPackRequest(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler


router = APIRouter(route_class=MsgpackRoute)


@router.post("/extract", tags=["Detection & recognition"])
async def extract(
    data: BodyExtract,
    processing: ProcessingDep,
    accept: Optional[str] = Header(None),
    content_type: Annotated[str | None, Header()] = None,
):
    """
    人脸提取/特征向量端点。
    """
    try:
        # 如果请求体是 msgpack，则图像数据不需要进行 base64 解码
        b64_decode = content_type != "application/msgpack"

        output = await processing.extract(
            images=data.images,
            return_face_data=data.return_face_data,
            embed_only=data.embed_only,
            extract_embedding=data.extract_embedding,
            threshold=data.threshold,
            extract_ga=data.extract_ga,
            limit_faces=data.limit_faces,
            min_face_size=data.min_face_size,
            return_landmarks=data.return_landmarks,
            detect_masks=data.detect_masks,
            verbose_timings=data.verbose_timings,
            b64_decode=b64_decode,
            img_req_headers=data.img_req_headers,
        )

        # 根据客户端请求的 Accept 头或请求参数决定响应格式
        if data.msgpack or (accept and "application/x-msgpack" in accept):
            return Response(
                content=msgpack.dumps(output, use_single_float=True),
                media_type="application/x-msgpack",
            )
        else:
            return UJSONResponse(output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/draw_detections", tags=["Detection & recognition"])
async def draw(data: BodyDraw, processing: ProcessingDep):
    """
    返回带有检测框的图像，用于测试目的。
    """
    try:
        output_buffer = await processing.draw(
            images=data.images,
            threshold=data.threshold,
            draw_landmarks=data.draw_landmarks,
            draw_scores=data.draw_scores,
            limit_faces=data.limit_faces,
            min_face_size=data.min_face_size,
            draw_sizes=data.draw_sizes,
            detect_masks=data.detect_masks,
        )
        output_buffer.seek(0)
        return StreamingResponse(output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", tags=["Detection & recognition"])
async def verify(data: FaceVerification, processing: ProcessingDep):
    """
    比较两张图片中的人脸，验证是否为同一个人。
    """
    try:
        result = await processing.verify(
            images=data.images,
            threshold=data.threshold,
            limit_faces=data.limit_faces,
            min_face_size=data.min_face_size,
        )
        return UJSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
