#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2024/9/5 21:21
@File: human_matting.py
@IDE: pycharm
@Description:
    人像抠图
"""
import numpy as np
from PIL import Image
import onnxruntime
from .tensor2numpy import NNormalize, NTo_Tensor, NUnsqueeze
from .context import Context
import cv2
import os
from time import time
import logging


WEIGHTS = {
    "hivision_modnet": os.path.join(
        os.path.dirname(__file__), "weights", "hivision_modnet.onnx"
    ),
    "modnet_photographic_portrait_matting": os.path.join(
        os.path.dirname(__file__),
        "weights",
        "modnet_photographic_portrait_matting.onnx",
    ),
    "mnn_hivision_modnet": os.path.join(
        os.path.dirname(__file__),
        "weights",
        "mnn_hivision_modnet.mnn",
    ),
    "rmbg-1.4": os.path.join(os.path.dirname(__file__), "weights", "rmbg-1.4.onnx"),
    "birefnet-v1-lite": os.path.join(
        os.path.dirname(__file__), "weights", "birefnet-v1-lite.onnx"
    ),
}

ONNX_DEVICE = onnxruntime.get_device()
ONNX_PROVIDER = (
    "CUDAExecutionProvider" if ONNX_DEVICE == "GPU" else "CPUExecutionProvider"
)

# Model cache dictionary
MODEL_CACHE = {}

HIVISION_MODNET_SESS = None
MODNET_PHOTOGRAPHIC_PORTRAIT_MATTING_SESS = None
RMBG_SESS = None
BIREFNET_V1_LITE_SESS = None


def load_onnx_model(checkpoint_path, set_cpu=False):
    """
    Load ONNX model with caching
    """
    # Check if model is already in cache
    if checkpoint_path in MODEL_CACHE:
        logging.info(f"Using cached model: {checkpoint_path}")
        return MODEL_CACHE[checkpoint_path]

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ONNX_PROVIDER == "CUDAExecutionProvider"
        else ["CPUExecutionProvider"]
    )

    try:
        if set_cpu:
            sess = onnxruntime.InferenceSession(
                checkpoint_path, providers=["CPUExecutionProvider"]
            )
        else:
            try:
                sess = onnxruntime.InferenceSession(checkpoint_path, providers=providers)
            except Exception as e:
                if ONNX_DEVICE == "CUDAExecutionProvider":
                    logging.warning(f"Failed to load model with CUDAExecutionProvider: {e}")
                    logging.info("Falling back to CPUExecutionProvider")
                    sess = onnxruntime.InferenceSession(
                        checkpoint_path, providers=["CPUExecutionProvider"]
                    )
                else:
                    raise e

        # Cache the loaded model
        MODEL_CACHE[checkpoint_path] = sess
        logging.info(f"Cached new model: {checkpoint_path}")
        return sess

    except Exception as e:
        logging.error(f"Error loading model {checkpoint_path}: {str(e)}")
        raise


def clear_model_cache():
    """
    Clear the model cache
    """
    global MODEL_CACHE
    MODEL_CACHE.clear()
    logging.info("Model cache cleared")


def extract_human(ctx: Context):
    """
    人像抠图
    :param ctx: 上下文
    """
    # 抠图
    matting_image = get_modnet_matting(ctx.processing_image, WEIGHTS["hivision_modnet"])
    # 修复抠图
    ctx.processing_image = hollow_out_fix(matting_image)
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_modnet_photographic_portrait_matting(ctx: Context):
    """
    人像抠图
    :param ctx: 上下文
    """
    # 抠图
    matting_image = get_modnet_matting_photographic_portrait_matting(
        ctx.processing_image, WEIGHTS["modnet_photographic_portrait_matting"]
    )
    # 修复抠图
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_mnn_modnet(ctx: Context):
    matting_image = get_mnn_modnet_matting(
        ctx.processing_image, WEIGHTS["mnn_hivision_modnet"]
    )
    ctx.processing_image = hollow_out_fix(matting_image)
    ctx.matting_image = ctx.processing_image.copy()


def extract_human_rmbg(ctx: Context):
    matting_image = get_rmbg_matting(ctx.processing_image, WEIGHTS["rmbg-1.4"])
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


# def extract_human_birefnet_portrait(ctx: Context):
#     matting_image = get_birefnet_portrait_matting(
#         ctx.processing_image, WEIGHTS["birefnet-portrait"]
#     )
#     ctx.processing_image = matting_image
#     ctx.matting_image = ctx.processing_image.copy()


def extract_human_birefnet_lite(ctx: Context):
    matting_image = get_birefnet_portrait_matting(
        ctx.processing_image, WEIGHTS["birefnet-v1-lite"]
    )
    ctx.processing_image = matting_image
    ctx.matting_image = ctx.processing_image.copy()


def hollow_out_fix(src: np.ndarray) -> np.ndarray:
    """
    修补抠图区域，作为抠图模型精度不够的补充
    :param src:
    :return:
    """
    b, g, r, a = cv2.split(src)
    src_bgr = cv2.merge((b, g, r))
    # -----------padding---------- #
    add_area = np.zeros((10, a.shape[1]), np.uint8)
    a = np.vstack((add_area, a, add_area))
    add_area = np.zeros((a.shape[0], 10), np.uint8)
    a = np.hstack((add_area, a, add_area))
    # -------------end------------ #
    _, a_threshold = cv2.threshold(a, 127, 255, 0)
    a_erode = cv2.erode(
        a_threshold,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=3,
    )
    contours, hierarchy = cv2.findContours(
        a_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [x for x in contours]
    # contours = np.squeeze(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    a_contour = cv2.drawContours(np.zeros(a.shape, np.uint8), contours[0], -1, 255, 2)
    # a_base = a_contour[1:-1, 1:-1]
    h, w = a.shape[:2]
    mask = np.zeros(
        [h + 2, w + 2], np.uint8
    )  # mask 必须行和列都加 2，且必须为 uint8 单通道阵列
    cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
    a = cv2.add(a, 255 - a_contour)
    return cv2.merge((src_bgr, a[10:-10, 10:-10]))


def image2bgr(input_image):
    if len(input_image.shape) == 2:
        input_image = input_image[:, :, None]
    if input_image.shape[2] == 1:
        result_image = np.repeat(input_image, 3, axis=2)
    elif input_image.shape[2] == 4:
        result_image = input_image[:, :, 0:3]
    else:
        result_image = input_image

    return result_image


def read_modnet_image(input_image, ref_size=512):
    im = Image.fromarray(np.uint8(input_image))
    width, length = im.size[0], im.size[1]
    im = np.asarray(im)
    im = image2bgr(im)
    im = cv2.resize(im, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    im = NNormalize(im, mean=np.array([0.5, 0.5, 0.5]), std=np.array([0.5, 0.5, 0.5]))
    im = NUnsqueeze(NTo_Tensor(im))

    return im, width, length


def get_modnet_matting(input_image, checkpoint_path, ref_size=512):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    # Load model with caching
    sess = load_onnx_model(checkpoint_path)
    
    # Rest of the function remains the same
    im, width, length = read_modnet_image(input_image, ref_size)
    im = im.astype(np.float32)
    im = np.transpose(im, (0, 2, 3, 1))
    im = np.squeeze(im)
    im = (im * 255).astype(np.uint8)
    im = cv2.resize(im, (width, length), interpolation=cv2.INTER_AREA)
    return im


def get_modnet_matting_photographic_portrait_matting(
    input_image, checkpoint_path, ref_size=512
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    # Load model with caching
    sess = load_onnx_model(checkpoint_path)
    
    # Rest of the function remains the same
    im, width, length = read_modnet_image(input_image, ref_size)
    im = im.astype(np.float32)
    im = np.transpose(im, (0, 2, 3, 1))
    im = np.squeeze(im)
    im = (im * 255).astype(np.uint8)
    im = cv2.resize(im, (width, length), interpolation=cv2.INTER_AREA)
    return im


def get_rmbg_matting(input_image: np.ndarray, checkpoint_path, ref_size=1024):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    # Load model with caching
    sess = load_onnx_model(checkpoint_path)
    
    # Rest of the function remains the same
    def resize_rmbg_image(image):
        h, w = image.shape[:2]
        scale = ref_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Process image
    image = resize_rmbg_image(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)

    # Run inference
    outputs = sess.run(None, {"input": image})
    alpha = outputs[0][0]
    alpha = np.transpose(alpha, (1, 2, 0))
    alpha = cv2.resize(alpha, (input_image.shape[1], input_image.shape[0]))
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    # Create RGBA output
    rgba = cv2.cvtColor(input_image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def get_mnn_modnet_matting(input_image, checkpoint_path, ref_size=512):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None

    try:
        import MNN.expr as expr
        import MNN.nn as nn
    except ImportError as e:
        raise ImportError(
            "The MNN module is not installed or there was an import error. Please ensure that the MNN library is installed by using the command 'pip install mnn'."
        ) from e

    config = {}
    config["precision"] = "low"  # 当硬件支持（armv8.2）时使用fp16推理
    config["backend"] = 0  # CPU
    config["numThread"] = 4  # 线程数
    im, width, length = read_modnet_image(input_image, ref_size=512)
    rt = nn.create_runtime_manager((config,))
    net = nn.load_module_from_file(
        checkpoint_path, ["input1"], ["output1"], runtime_manager=rt
    )
    input_var = expr.convert(im, expr.NCHW)
    output_var = net.forward(input_var)
    matte = expr.convert(output_var, expr.NCHW)
    matte = matte.read()  # var转换为np
    matte = (matte * 255).astype("uint8")
    matte = np.squeeze(matte)
    mask = cv2.resize(matte, (width, length), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(np.uint8(input_image))

    output_image = cv2.merge((b, g, r, mask))

    return output_image


def get_birefnet_portrait_matting(input_image, checkpoint_path, ref_size=1024):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    # Load model with caching
    sess = load_onnx_model(checkpoint_path)
    
    # Get the input name from the model
    input_name = sess.get_inputs()[0].name
    logging.info(f"Model input name: {input_name}")
    
    try:
        def transform_image(image):
            # Resize to 1024x1024 as required by the model
            image = cv2.resize(image, (1024, 1024))
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            # Change from (H, W, C) to (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            # Add batch dimension
            image = np.expand_dims(image, 0)
            return image

        # Process image
        image = transform_image(input_image)
        
        # Use the correct input name from the model
        outputs = sess.run(None, {input_name: image})
        
        # Process outputs and create alpha mask
        alpha = outputs[0][0]
        alpha = np.transpose(alpha, (1, 2, 0))
        alpha = cv2.resize(alpha, (input_image.shape[1], input_image.shape[0]))
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # Create RGBA output
        rgba = cv2.cvtColor(input_image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha

        # Clean up intermediate variables
        del image, outputs, alpha
        import gc
        gc.collect()

        return rgba

    except Exception as e:
        logging.error(f"Error during birefnet matting: {str(e)}")
        raise
    finally:
        # Clean up any remaining variables
        if 'image' in locals():
            del image
        if 'outputs' in locals():
            del outputs
        if 'alpha' in locals():
            del alpha
        import gc
        gc.collect()
