import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import onnx
from tenacity import (before_sleep_log, retry, retry_if_not_exception_type,
                      stop_after_attempt, wait_exponential)

from if_rest.core.converters.remove_initializer_from_input import \
    remove_initializer_from_input
from if_rest.core.converters.reshape_onnx import reshape
from if_rest.core.model_zoo.exec_backends import onnxrt_backend as onnx_backend
from if_rest.core.model_zoo.face_detectors import *
from if_rest.core.model_zoo.face_processors import *
from if_rest.core.utils.download import download
from if_rest.core.utils.download_google import check_hash, download_from_gdrive
from if_rest.core.utils.helpers import prepare_folders
from if_rest.logger import logger
from if_rest.settings import AppSettings

# Since TensorRT, TritonClient and PyCUDA are optional dependencies it might be not available
try:
    from if_rest.core.converters.onnx_to_trt import check_fp16, convert_onnx
    from if_rest.core.model_zoo.exec_backends import trt_backend
except Exception as e:
    logger.error(e)
    trt_backend = None
    convert_onnx = None

# Map function names to corresponding functions
func_map = {
    "genderage_v1": genderage_v1,
    "retinaface_r50_v1": retinaface_r50_v1,
    "retinaface_mnet025_v1": retinaface_mnet025_v1,
    "retinaface_mnet025_v2": retinaface_mnet025_v2,
    "mnet_cov2": mnet_cov2,
    "centerface": centerface,
    "dbface": dbface,
    "scrfd": scrfd,
    "scrfd_v2": scrfd_v2,
    "arcface_mxnet": arcface_mxnet,
    "arcface_torch": arcface_torch,
    "adaface": adaface,
    "mask_detector": mask_detector,
    "yolov5_face": yolov5_face,
}


def sniff_output_order(model_path: Path, save_dir: Path) -> List[str]:
    """
    Sniffs the output order of a model.

    Args:
        model_path (Path): The path to the ONNX model.
        save_dir (Path): The directory where the output order will be saved.

    Returns:
        List[str]: A list of output names in the correct order.
    """
    outputs_file = save_dir / "output_order.json"
    if not outputs_file.exists():
        model = onnx.load(str(model_path))
        output = [node.name for node in model.graph.output]
        with open(outputs_file, mode="w") as fl:
            fl.write(json.dumps(output))
    else:
        output = read_outputs_order(save_dir)
    return output


def read_outputs_order(trt_dir: Path) -> List[str]:
    """
    Reads the output order from a TRT directory.

    Args:
       trt_dir (Path): The directory containing the TRT model.

    Returns:
       List[str]: A list of output names in the correct order.
    """
    outputs = None
    outputs_file = trt_dir / "output_order.json"
    if outputs_file.exists():
        with open(outputs_file, mode="r") as fl:
            outputs = json.loads(fl.read())
    return outputs


@retry(
    wait=wait_exponential(min=0.5, max=5),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_not_exception_type(ValueError),
)
def download_onnx(src: str, dst: Path, dl_type="google", md5=None):
    if src not in [None, ""]:
        if dl_type == "google":
            download_from_gdrive(src, str(dst))
        else:
            download(src, str(dst))
        if md5:
            hashes_match = check_hash(str(dst), md5, algo="md5")
            if not hashes_match:
                logger.error(f"ONNX model hash mismatch after download for {dst}")
                raise AssertionError("Hash mismatch")
        return dst
    else:
        logger.error(f"No download link provided for {dst}")
        raise ValueError("No download link")


def build_model_paths(settings: AppSettings, model_name: str, ext: str) -> (Path, Path):
    """Helper to build model paths based on settings."""
    base_dir = Path(settings.models_dir)
    subdirs = {"onnx": "onnx", "plan": "trt-engines", "engine": "trt-engines"}
    if ext not in subdirs:
        raise ValueError(f"Unknown extension type: {ext}")

    parent_dir = base_dir / subdirs[ext] / model_name
    file_path = parent_dir / f"{model_name}.{ext}"
    return parent_dir, file_path


def prepare_backend(
    model_name: str,
    backend_name: str,
    model_meta: Dict[str, Any],
    settings: AppSettings,
    im_size: List[int] = None,
    max_batch_size: int = 1,
    force_fp16: bool = False,
    download_model: bool = True,
):
    """
    Prepares the backend for a model.
    """
    model_info = model_meta.get(model_name, {})
    if not model_info:
        raise ValueError(f"Metadata for model '{model_name}' not found in models.json")

    models_root = Path(settings.models_dir)
    prepare_folders([models_root / "onnx", models_root / "trt-engines"])

    reshape_allowed = model_info.get("reshape")
    shape = model_info.get("shape")

    if reshape_allowed and im_size:
        shape = (1, 3) + tuple(im_size)[::-1]

    onnx_dir, onnx_path = build_model_paths(settings, model_name, "onnx")
    trt_dir, trt_path = build_model_paths(settings, model_name, "plan")

    onnx_exists = onnx_path.exists()
    onnx_hash = model_info.get("md5")
    trt_rebuild_required = False

    if onnx_exists and onnx_hash:
        logger.info(f"Checking model hash for {model_name}...")
        if not check_hash(str(onnx_path), onnx_hash, algo="md5"):
            logger.warning("ONNX model hash mismatch, re-downloading.")
            onnx_exists = False
            trt_rebuild_required = True

    if not onnx_exists and download_model:
        prepare_folders([onnx_dir])
        dl_link = model_info.get("link")
        dl_type = model_info.get("dl_type")
        if dl_link:
            try:
                download_onnx(dl_link, onnx_path, dl_type, onnx_hash)
                remove_initializer_from_input(str(onnx_path), str(onnx_path))
            except Exception as e:
                logger.critical(
                    f"Model download failed for {model_name}: {e}. Exiting."
                )
                exit(1)
        else:
            logger.critical(
                f"No download link for '{model_name}' and model not found locally. Exiting."
            )
            exit(1)

    if backend_name == "triton":
        return model_name

    if backend_name == "onnx":
        model = onnx.load(str(onnx_path))
        if reshape_allowed and im_size:
            logger.info(f"Reshaping ONNX inputs to: {shape}")
            model = reshape(model, h=im_size[1], w=im_size[0])
        return model.SerializeToString()

    if backend_name == "trt":
        if not convert_onnx:
            raise ImportError(
                "TensorRT backend is not available. Please install dependencies."
            )

        has_fp16 = check_fp16()

        if reshape_allowed:
            trt_path = trt_path.with_name(
                f"{trt_path.stem}_{shape[3]}_{shape[2]}{trt_path.suffix}"
            )
        if max_batch_size > 1:
            trt_path = trt_path.with_name(
                f"{trt_path.stem}_batch{max_batch_size}{trt_path.suffix}"
            )
        if force_fp16 or has_fp16:
            trt_path = trt_path.with_name(f"{trt_path.stem}_fp16{trt_path.suffix}")

        prepare_folders([trt_dir])

        if not model_info.get("outputs"):
            logger.debug(
                f"No output order provided for {model_name}, sniffing from ONNX model."
            )
            sniff_output_order(onnx_path, trt_dir)

        if not trt_path.exists() or trt_rebuild_required:
            temp_onnx_model_bytes = None
            if reshape_allowed or max_batch_size != 1:
                logger.info(f"Reshaping ONNX model '{model_name}' for TRT conversion.")
                model = onnx.load(str(onnx_path))
                onnx_batch_size = -1 if max_batch_size > 1 else 1
                reshaped = reshape(model, n=onnx_batch_size, h=shape[2], w=shape[3])
                temp_onnx_model_bytes = reshaped.SerializeToString()
            else:
                with open(onnx_path, "rb") as f:
                    temp_onnx_model_bytes = f.read()

            logger.info(f"Building TRT engine for {model_name}...")
            convert_onnx(
                temp_onnx_model_bytes,
                engine_file_path=str(trt_path),
                max_batch_size=max_batch_size,
                force_fp16=force_fp16,
            )
            logger.info("Building TRT engine complete!")
        return str(trt_path)

    raise ValueError(f"Unsupported backend: {backend_name}")


def get_model(
    model_name: str,
    settings: AppSettings,
    model_meta: Dict[str, Any],
    im_size: List[int] = None,
    max_batch_size: int = 1,
    force_fp16: bool = False,
    download_model: bool = True,
):
    """
    Returns an inference backend instance with a loaded model.
    """
    backend_name = settings.models.inference_backend
    triton_uri = settings.models.triton_uri

    backends = {"onnx": onnx_backend, "trt": trt_backend}

    if backend_name not in backends:
        logger.critical(f"Unknown backend '{backend_name}' specified. Exiting.")
        exit(1)

    if model_name not in model_meta:
        logger.critical(f"Unknown model '{model_name}' specified. Exiting.")
        exit(1)

    backend = backends[backend_name]

    model_path = prepare_backend(
        model_name,
        backend_name,
        model_meta=model_meta,
        settings=settings,
        im_size=im_size,
        max_batch_size=max_batch_size,
        force_fp16=force_fp16,
        download_model=download_model,
    )

    model_info = model_meta[model_name]
    outputs = model_info.get("outputs")
    if not outputs and backend_name == "trt":
        logger.debug(f"No output order in meta, reading from file for '{model_name}'")
        _, trt_path = build_model_paths(settings, model_name, "plan")
        outputs = read_outputs_order(trt_path.parent)

    func = func_map[model_info.get("function")]
    model = func(
        model_path=model_path,
        backend=backend,
        outputs=outputs,
        triton_uri=triton_uri,
    )
    return model
