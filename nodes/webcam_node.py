import base64
import binascii
import hashlib
import io

import numpy as np
import torch
from PIL import Image, ImageOps

_NO_CAPTURE_MSG = (
    "Webkamera nepořídila žádný snímek. Povolte prohlížeči přístup ke kameře "
    "a ujistěte se, že náhled kamery pořídil snímek před spuštěním workflow."
)


def _decode_capture(image):
    """Dekóduje base64 řetězec (data URL) se snímkem přímo v paměti, bez ukládání na disk."""
    data = image.split(",", 1)[1] if "," in image else image
    try:
        return base64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(_NO_CAPTURE_MSG) from exc


class WebcamCaptureNode:
    """Pořídí snímek z webkamery při spuštění workflow a vrátí jej jako IMAGE."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "image" nese snímek jako base64 data URL (dočasný řetězec v paměti) -
                # náhled kamery a tlačítko pro pořízení snímku zajišťuje frontend JS (web/webcam_capture.js).
                "image": ("STRING", {"default": "", "multiline": False}),
                "width": ("INT", {"default": 640, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 480, "min": 0, "max": 8192, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "capture"

    def capture(self, image, width, height, capture_on_queue=True):
        if not image:
            raise RuntimeError(_NO_CAPTURE_MSG)

        image_bytes = _decode_capture(image)

        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, :, :, :]

        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, image, width, height, capture_on_queue):
        if not image:
            return float("nan")
        return hashlib.sha256(image.encode("utf-8")).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image:
            return _NO_CAPTURE_MSG
        return True


NODE_CLASS_MAPPINGS = {
    "FAL_WebcamCapture": WebcamCaptureNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_WebcamCapture": "Webcam Capture (fal.ai)",
}
