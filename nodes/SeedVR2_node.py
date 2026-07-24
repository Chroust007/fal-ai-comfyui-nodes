import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO

from fal_client.client import FalClientHTTPError
from .fal_utils import ApiHandler, ImageUtils, extract_fal_error_message


class FalSeedVR2SeamlessNode:
    """Image upscaling using fal.ai's SeedVR2 Seamless model."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_mode": (["factor", "target"], {"default": "factor"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "target_resolution": (["720p", "1080p", "1440p", "2160p"], {"default": "1080p"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_image"

    def upscale_image(
        self,
        image,
        upscale_mode="factor",
        upscale_factor=2.0,
        target_resolution="1080p",
        noise_scale=0.1,
        output_format="png",
        seed=0,
        private_mode=False,
    ):
        # 1. Nahrání vstupního tensoru obrázku do fal.ai storage
        image_url = ImageUtils.upload_image(image)

        # 2. Příprava argumentů podle schématu API SeedVR2
        arguments = {
            "image_url": image_url,
            "upscale_mode": upscale_mode,
            "upscale_factor": upscale_factor,
            "target_resolution": target_resolution,
            "noise_scale": noise_scale,
            "output_format": output_format,
        }

        if seed > 0:
            arguments["seed"] = seed

        # 3. Odeslání požadavku na fal.ai endpoint
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/seedvr/upscale/image/seamless",
                arguments,
                disable_storage=private_mode,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 4. Získání URL nebo Base64 zvětšeného obrázku
        output_image_obj = result.get("image", {})
        output_image_url = output_image_obj.get("url")

        if not output_image_url:
            raise ValueError(
                f"SeedVR2 API nevrátilo žádný obrázek. Odpověď: {result}"
            )

        # 5. Bezpečné stažení / dekódování obrázku
        if output_image_url.startswith("data:image"):
            base64_data = output_image_url.split(",")[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))
        else:
            response = requests.get(output_image_url)
            img = Image.open(BytesIO(response.content))

        # Převedení do RGB/RGBA podle potřeby
        img = img.convert("RGB")

        # Převedení na numpy array a tensor pro ComfyUI
        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, :, :, :]

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "FAL_SeedVR2Seamless": FalSeedVR2SeamlessNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_SeedVR2Seamless": "SeedVR2 Seamless Upscale (fal.ai)",
}