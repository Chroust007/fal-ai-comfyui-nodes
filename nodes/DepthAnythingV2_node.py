import base64
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO

from fal_client.client import FalClientHTTPError
from .fal_utils import ApiHandler, ImageUtils, extract_fal_error_message

class FalDepthAnythingV2Node:
    """Depth map generation using fal.ai's Depth Anything V2 model."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "generate_depth_map"

    def generate_depth_map(self, image, private_mode=False):
        # 1. Nahrání vstupního obrázku (tensoru) do fal.ai storage
        image_url = ImageUtils.upload_image(image)

        # 2. Příprava argumentů pro API (Depth Anything v2 přijímá image_url)
        arguments = {
            "image_url": image_url,
        }

        # 3. Volání správného endpointu pro Depth Anything V2
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/image-preprocessors/depth-anything/v2",
                arguments,
                disable_storage=private_mode,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 4. Získání URL nebo Base64 vygenerované depth mapy
        output_image_obj = result.get("image", {})
        output_image_url = output_image_obj.get("url")

        if not output_image_url:
            raise ValueError(
                f"Depth Anything API nevrátilo žádný obrázek. Odpověď: {result}"
            )

        # 5. Bezpečné stažení a dekódování obrázku (Base64 vs HTTP)
        if output_image_url.startswith('data:image'):
            base64_data = output_image_url.split(',')[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))
        else:
            response = requests.get(output_image_url)
            img = Image.open(BytesIO(response.content))

        # Převedeme do RGB (depth mapy jsou často jen černobílé, ale ComfyUI preferuje 3 kanály)
        img = img.convert("RGB")

        # Převedení na numpy array, normalizace a formát pro ComfyUI
        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, :, :, :]

        return (image_tensor,)

# Nutné pro registraci uzlu v ComfyUI
NODE_CLASS_MAPPINGS = {
    "FAL_DepthAnythingV2": FalDepthAnythingV2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_DepthAnythingV2": "Depth Anything V2 (fal.ai)",
}