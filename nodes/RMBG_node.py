import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64 # Přidán import pro dekódování Base64

from fal_client.client import FalClientHTTPError

from .fal_utils import ApiHandler, ImageUtils, ResultProcessor, extract_fal_error_message


class BriaRmbg2Node:
    """Background removal using fal.ai's BRIA RMBG-2.0 model."""

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
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_background"

    def remove_background(self, image, private_mode=False):
        # 1. Nahrání vstupního tensoru do fal.ai storage
        image_url = ImageUtils.upload_image(image)

        # 2. Příprava argumentů podle schématu (image_url)
        arguments = {
            "image_url": image_url,
        }

        # 3. Volání přesného endpointu z dokumentace
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/bria/background/remove",
                arguments,
                disable_storage=private_mode,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 4. Získání URL nebo Base64 oříznutého obrázku z odpovědi
        output_image_obj = result.get("image", {})
        output_image_url = output_image_obj.get("url")

        if not output_image_url:
            raise ValueError(
                f"RMBG API nevrátilo žádný obrázek. Odpověď: {result}"
            )

        # 5. Zpracování výsledku: Rozhodnutí mezi Base64 a klasickým URL (HTTP/HTTPS)
        if output_image_url.startswith('data:image'):
            # Jde o Base64 řetězec, odstraníme hlavičku (např. 'data:image/png;base64,') a dekódujeme
            base64_data = output_image_url.split(',')[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(BytesIO(image_data))
        else:
             # Jde o klasickou URL, stáhneme ji pomocí requests
            response = requests.get(output_image_url)
            img = Image.open(BytesIO(response.content))

        # Vynucení RGBA formátu pro zachování průhlednosti pozadí
        img = img.convert("RGBA")

        # Převedení na numpy array a normalizace (0.0 až 1.0)
        image_np = np.array(img).astype(np.float32) / 255.0

        # Přidání batch dimenze [B, H, W, C] požadované systémem ComfyUI
        image_tensor = torch.from_numpy(image_np)[None, :, :, :]

        return (image_tensor,)

NODE_CLASS_MAPPINGS = {
    "FAL_RMBG2": BriaRmbg2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_RMBG2": "BriaRmbg2 (fal.ai)",
}