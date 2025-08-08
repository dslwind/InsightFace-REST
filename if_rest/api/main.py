import os
import ssl
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline

from if_rest.api.routes.v1 import v1_router
from if_rest.core.processing import get_processing
from if_rest.logger import logger
from if_rest.settings import settings

# 从环境变量获取版本号
__version__ = os.getenv("IFR_VERSION", "0.9.5.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理函数。
    在应用启动时初始化资源，在关闭时清理资源。
    """
    logger.info("应用启动，开始初始化资源...")

    # 初始化 aiohttp.ClientSession 用于异步下载图片
    timeout = ClientTimeout(total=60.0)
    connector_args = {}
    if settings.defaults.sslv3_hack:
        ssl_context = ssl.create_default_context()
        ssl_context.minimum_version = ssl.TLSVersion.SSLv3
        connector_args["ssl"] = ssl_context
    else:
        # 推荐在生产环境中验证 SSL 证书
        # connector_args['ssl'] = True
        connector_args["ssl"] = False  # 为方便起见，暂时禁用

    download_client = aiohttp.ClientSession(
        timeout=timeout, connector=TCPConnector(**connector_args)
    )

    # 初始化核心处理模块
    try:
        processing_instance = await get_processing()
        await processing_instance.start(download_client=download_client)
        logger.info("核心处理模块已成功初始化！")
    except Exception as e:
        logger.critical(f"核心处理模块初始化失败: {e}", exc_info=True)
        # 在关键组件失败时，可以选择退出应用
        exit(1)

    yield  # 应用在此处运行

    # 应用关闭时执行清理操作
    logger.info("应用关闭，开始清理资源...")
    await download_client.close()
    logger.info("资源已成功清理。")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例。
    """
    application = FastAPIOffline(
        title="InsightFace-REST",
        description="高性能人脸识别 REST API",
        version=__version__,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(v1_router)

    return application


app = create_app()
