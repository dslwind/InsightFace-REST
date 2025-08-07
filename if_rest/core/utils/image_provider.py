import asyncio
import base64
import io
import logging
import os
import time
import traceback
from typing import Dict

import aiofiles
import aiohttp
import cv2
import exifread
import imageio
import numpy as np
from tenacity import (before_sleep_log, retry, retry_if_not_exception_type,
                      stop_after_attempt, wait_exponential)
from turbojpeg import TurboJPEG

from if_rest.core.utils.helpers import tobool
from if_rest.logger import logger
from if_rest.schemas import Images
from if_rest.settings import Settings

if tobool(os.getenv("USE_NVJPEG", False)):
    try:
        from nvjpeg import NvJpeg

        jpeg = NvJpeg()
        logger.info("Using nvJPEG for JPEG decoding")
    except:
        logger.info("Using TurboJPEG for JPEG decoding")
        jpeg = TurboJPEG()
else:
    jpeg = TurboJPEG()

settings = Settings()
def_headers = settings.defaults.img_req_headers


def resize_image(image, max_size: list = None):
    if max_size is None:
        max_size = [640, 640]

    cw = max_size[0]
    ch = max_size[1]
    h, w, _ = image.shape

    scale_factor = min(cw / w, ch / h)
    # If image is too small, it may contain only single face, which leads to decreased detection accuracy,
    # so we reduce scale factor by some factor
    if scale_factor > 2:
        scale_factor = scale_factor * 0.7

    if scale_factor <= 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    if scale_factor == 1.0:
        transformed_image = image
    else:
        transformed_image = cv2.resize(
            image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interp
        )

    h, w, _ = transformed_image.shape

    if w < cw:
        transformed_image = cv2.copyMakeBorder(
            transformed_image, 0, 0, 0, cw - w, cv2.BORDER_CONSTANT
        )
    if h < ch:
        transformed_image = cv2.copyMakeBorder(
            transformed_image, 0, ch - h, 0, 0, cv2.BORDER_CONSTANT
        )

    return transformed_image, scale_factor


def sniff_gif(data):
    """
    Sniffs first 32 bytes of data and decodes first frame if it's GIF.

    Args:
        data (bytes): The input data to be processed.

    Returns:
        bytes: The binary image if it's a GIF, otherwise the original data.
    """
    try:
        if b"GIF" in data[:32]:
            sequence = imageio.get_reader(data, ".gif")
            binary = None
            for frame in sequence:
                outp = io.BytesIO()
                imageio.imwrite(outp, frame, format="jpeg")
                outp.seek(0)
                binary = outp.read()
                break
            return binary
        else:
            return data
    except:
        return data


def transpose_image(image, orientation):
    """
    Transposes the image based on the given orientation.
    See Orientation in https://www.exif.org/Exif2-2.PDF for details.
    Args:
        image (np.ndarray): The input image.
        orientation (exifread.Image): The orientation of the image.

    Returns:
        np.ndarray: The transposed image.
    """
    TRANSPOSE_MAP = {
        2: np.fliplr,
        3: lambda img: np.rot90(img, 2),
        4: np.flipud,
        5: lambda img: np.rot90(np.flipud(img), -1),
        6: lambda img: np.rot90(img, -1),
        7: lambda img: np.rot90(np.flipud(img)),
        8: lambda img: np.rot90(img),
    }

    if orientation is None:
        return image

    val = orientation.values[0]

    # If the value is not in the map (e.g., for value 1 or an invalid value),
    # returns the default value: an "identity function" (lambda img: img).
    transform_func = TRANSPOSE_MAP.get(val, lambda img: img)

    return transform_func(image)


async def read_as_bytes(path, **kwargs):
    """
    Asynchronously reads the file at the given path and returns it as a bytes object.

    Args:
        path (str): The path of the file to be read.

    Returns:
        np.ndarray: The file contents.
    """
    async with aiofiles.open(path, mode="rb") as fl:
        data = await fl.read()
        data = sniff_gif(data)
        _bytes = np.frombuffer(data, dtype="uint8")
        return _bytes


def base64_to_bytes(b64encoded, b64_decode=True, **kwargs):
    """
    Decodes the base64 encoded string and returns it as a bytes object.

    Args:
        b64encoded (str): The base64 encoded string to be decoded.

    Returns:
        tuple: A tuple containing the decoded bytes and any error message.
    """
    data_bytes = None
    try:
        if b64_decode:
            data_bytes = b64encoded.split(",")[-1]
            data_bytes = base64.b64decode(data_bytes)
        else:
            data_bytes = b64encoded
        data_bytes = sniff_gif(data_bytes)
        data_bytes = np.frombuffer(data_bytes, dtype="uint8")
    except Exception:
        error_message = traceback.format_exc()
        logger.warning(error_message)
        return data_bytes, error_message
    return data_bytes, None


def decode_img_bytes(image_bytes, **kwargs):
    """
    Decodes the image bytes and returns it as a numpy array.
    Tries JPEG decoding with TurboJPEG or NVJpeg first.

    Args:
        image_bytes (bytes): The image bytes to be decoded.

    Returns:
        np.ndarray: The decoded image.
    """
    t0 = time.perf_counter()
    try:
        orientation_tag = exifread.process_file(
            io.BytesIO(image_bytes), stop_tag="Image Orientation"
        ).get("Image Orientation")
        decoded_image = jpeg.decode(image_bytes)
        decoded_image = transpose_image(decoded_image, orientation=orientation_tag)
    except:
        logger.debug("JPEG decoder failed, fallback to cv2.imdecode")
        decoded_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    logger.debug(f"Decoding took: {(t1 - t0) * 1000:.3f} ms.")
    return decoded_image


@retry(
    wait=wait_exponential(min=0.5, max=5),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_not_exception_type(ValueError),
)
async def make_request(url, session, headers: dict = None):
    """
    Makes a GET request to the given URL and returns the response. Retries on failure.

    Args:
        url (str): The URL of the request.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        aiohttp.ClientResponse: The response from the server.
    """
    if not headers:
        headers = def_headers
    resp = await session.get(url, allow_redirects=True, headers=headers)
    # Here we make an assumption that 404 and 403 codes shouldn't require retries.
    # Any other exception might be retried again.
    if resp.status in [404, 403, 401]:
        raise ValueError(f"Failed to get data from {url}. Status code: {resp.status}")
    if resp.status >= 400:
        raise aiohttp.ClientResponseError(
            resp.request_info, status=resp.status, history=()
        )
    return resp


async def dl_image(path, session: aiohttp.ClientSession = None, headers=None, **kwargs):
    """
    Downloads the image from the given path and returns it as a bytes object.

    Args:
        path (str): The path of the file to be downloaded.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        tuple: A tuple containing the downloaded bytes and any error message.
    """
    data_bytes = None
    try:
        if path.startswith("http"):
            resp = await make_request(path, session, headers=headers)
            content = await resp.content.read()
            data = sniff_gif(content)
            data_bytes = np.frombuffer(data, dtype="uint8")
        else:
            path = os.path.join(settings.root_images_dir, path)
            if not os.path.exists(path):
                error_message = f"File: '{path}' not found"
                return data_bytes, error_message
            data_bytes = await read_as_bytes(path)
    except Exception:
        error_message = traceback.format_exc()
        logger.warning(error_message)
        return data_bytes, error_message
    return data_bytes, None


def make_im_data(data_bytes, error_message, decode=True):
    """
    Creates a dictionary containing the image data and any error message occurred.

    Args:
        data_bytes (np.ndarray): The image bytes.
        error_message (str): The error message.
        decode (bool): Whether to decode the image or not.

    Returns:
        dict: A dictionary containing the image data and any error message.
    """
    traceback_msg = None
    if error_message is None:
        if decode:
            try:
                data = decode_img_bytes(data_bytes)
            except Exception:
                data = None
                error_message = traceback.format_exc()
                logger.warning(error_message)
        else:
            data = data_bytes

        if isinstance(data, type(None)):
            if error_message is None:
                error_message = "Can't decode file, possibly not an image"

    if error_message:
        data = None
        traceback_msg = error_message
        logger.warning(error_message)

    im_data = dict(data=data, traceback=traceback_msg)
    return im_data


async def get_images(
    data: Images,
    decode=True,
    session: aiohttp.ClientSession = None,
    b64_decode=True,
    headers: dict = None,
    **kwargs,
):
    """
    Downloads and decodes the images from the given data.

    Args:
        data (Dict[str, list]): The input data containing URLs or base64 encoded strings.
        decode (bool): Whether to decode the images or not. Defaults to True.
        session (aiohttp.ClientSession): The client session for making requests.

    Returns:
        list: A list of dictionaries containing the image data and any error message.
    """

    images = []

    if data.urls is not None:
        urls = data.urls
        tasks = []
        for url in urls:
            tasks.append(
                asyncio.ensure_future(dl_image(url, session=session, headers=headers))
            )

        results = await asyncio.gather(*tasks)
        for res in results:
            date_bytes, error_message = res
            im_data = make_im_data(date_bytes, error_message, decode=decode)
            images.append(im_data)

    elif data.data is not None:
        base64_images = data.data
        images = []
        for b64_img in base64_images:
            date_bytes, error_message = base64_to_bytes(b64_img, b64_decode=b64_decode)
            im_data = make_im_data(date_bytes, error_message, decode=decode)
            images.append(im_data)

    return images
