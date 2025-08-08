import logging
import sys

from if_rest.core.model_zoo.getter import prepare_backend
from if_rest.core.utils.helpers import validate_max_size
from if_rest.logger import logger
from if_rest.settings import AppSettings, settings


def prepare_all_models(app_settings: AppSettings):
    """
    根据提供的配置，准备所有必要的模型。
    """
    logger.info("开始模型准备流程...")

    # 从配置中获取需要准备的模型列表
    models_to_prepare = [
        model
        for model in [
            app_settings.models.det_name,
            app_settings.models.rec_name,
            app_settings.models.ga_name,
            app_settings.models.mask_detector,
        ]
        if model is not None
    ]

    if not models_to_prepare:
        logger.warning("在配置中没有找到需要准备的模型。")
        return

    logger.info(f"将要准备以下模型: {', '.join(models_to_prepare)}")

    # 加载模型元数据
    try:
        model_meta = app_settings.get_model_meta()
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"无法加载模型元数据: {e}。程序退出。")
        sys.exit(1)

    # 逐个准备模型
    for model_name in models_to_prepare:
        try:
            logger.info(f"开始准备 '{model_name}' 模型...")

            model_info = model_meta.get(model_name, {})
            max_size = validate_max_size(app_settings.models.max_size)
            batch_size = 1
            if model_info.get("allow_batching"):
                if model_name == app_settings.models.det_name:
                    batch_size = app_settings.models.det_batch_size
                else:
                    batch_size = app_settings.models.rec_batch_size

            prepare_backend(
                model_name=model_name,
                backend_name=app_settings.models.inference_backend,
                model_meta=model_meta,
                settings=app_settings,
                im_size=max_size,
                force_fp16=app_settings.models.force_fp16,
                max_batch_size=batch_size,
            )
            logger.info(f"'{model_name}' 模型已成功准备就绪！")
        except Exception as e:
            logger.critical(
                f"准备 '{model_name}' 模型时发生严重错误: {e}", exc_info=True
            )
            sys.exit(1)

    logger.info("所有模型均已成功准备就绪！")


if __name__ == "__main__":
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="[%H:%M:%S]",
    )
    prepare_all_models(settings)
