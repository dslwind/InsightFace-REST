import os

from fastapi import APIRouter, HTTPException

from if_rest.core.processing import ProcessingDep
from if_rest.logger import logger
from if_rest.settings import Settings

settings = Settings()
router = APIRouter()

__version__ = os.getenv('IFR_VERSION', '0.9.5.0')


@router.get('/info', tags=['Utility'])
def info():
    """
    Enlist container configuration.

    """
    return dict(
        version=__version__,
        tensorrt_version=os.getenv('TRT_VERSION', os.getenv('TENSORRT_VERSION')),
        log_level=settings.log_level,
        models=settings.models.dict(),
        defaults=settings.defaults.dict(),
    )


@router.get('/health', tags=['Utility'])
async def check_health(processing: ProcessingDep):
    """
    Execute recognition request with default parameters to verify recognition is actually working

    """
    try:
        await processing.healthcheck()
        return {'status': 'ok'}
    except Exception:
        logger.exception("Self check failed")
        raise HTTPException(503, detail='self check failed')


# @router.get('/', include_in_schema=False)
# async def redirect_to_docs():
#     return RedirectResponse(url="/docs")
