"""Image upscaling using fal.ai's SeedVR2 Seamless model."""

import base64
import os

import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests

from fal_client.client import FalClientHTTPError
from .fal_utils import ApiHandler, ImageUtils, FalConfig, extract_fal_error_message
import requests as _requests_for_usage


def _fetch_balance(admin_key):
    """Fetch current credit balance from fal.ai billing API."""
    if not admin_key:
        return ("N/A", "USD")

    url = "https://api.fal.ai/v1/account/billing"
    headers = {"Authorization": f"Key {admin_key}"}

    try:
        response = _requests_for_usage.get(
            url, headers=headers, params={"expand": "credits"}, timeout=5
        )
        response.raise_for_status()
        credits_dict = response.json().get("credits", {})
        return (
            credits_dict.get("current_balance", "N/A"),
            credits_dict.get("currency", "USD"),
        )
    except Exception as e:
        return (f"Chyba při zjišťování ({e})", "USD")


def _build_usage_stats(fal_auth, result, image_size):
    """Sestaví textový výstup se statistikami generování."""
    account_name = (fal_auth.get("account_name", "Neznámý") if fal_auth else "Neznámý")
    admin_key = fal_auth.get("admin_key") if fal_auth else None

    images_count = 1 if result else 0
    seed_used = result.get("seed", "N/A") if isinstance(result, dict) else "N/A"

    # Timings
    timings = (result.get("timings") if isinstance(result, dict) else None) or {}
    inference_time = timings.get("inference")
    time_str = f"{inference_time:.2f} s" if isinstance(inference_time, (int, float)) else "N/A"

    # NSFW flags
    nsfw_flags = result.get("has_nsfw_concepts") if isinstance(result, dict) else None
    if nsfw_flags:
        nsfw_str = f"{sum(1 for f in nsfw_flags if f)}/{len(nsfw_flags)}"
    else:
        nsfw_str = "N/A"

    # Expanded prompt (only if it differs from original)
    expanded_prompt_line = ""
    if isinstance(result, dict):
        actual_prompt = result.get("actual_prompt")
        original_prompt = result.get("prompt")
        if actual_prompt and actual_prompt != original_prompt:
            expanded_prompt_line = f"• Rozšířený prompt: {actual_prompt}\n"

    # Balance
    balance, currency = _fetch_balance(admin_key)

    return (
        f"📊 Statistiky generování (Účet: {account_name}):\n"
        f"• Vygenerováno obrázků: {images_count}\n"
        f"• Velikost / mód upscale: {image_size}\n"
        f"• Použitý seed: {seed_used}\n"
        f"• Doba generování: {time_str}\n"
        f"• NSFW obsah (obrázky): {nsfw_str}\n"
        f"{expanded_prompt_line}"
        f"• Zbývá kreditu: {balance} {currency}"
    )


def _download_image(output_image_url):
    """Download an image from a URL or base64 data URI and return as RGB PIL Image."""
    if output_image_url.startswith("data:image"):
        base64_data = output_image_url.split(",", 1)[1]
        image_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_data))
    else:
        response = requests.get(output_image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    return img.convert("RGB")


def _image_to_tensor(img):
    """Convert a PIL Image to a ComfyUI-compatible tensor batch [1, H, W, C]."""
    image_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None, :, :, :]


class FalSeedVR2SeamlessNode:
    """Image upscaling using fal.ai's SeedVR2 Seamless model."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs for the ComfyUI interface."""
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "upscale_mode": (["factor", "target"], {"default": "factor"}),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "target_resolution": (
                    ["720p", "1080p", "1440p", "2160p"],
                    {"default": "1080p"},
                ),
                "noise_scale": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "upscale_image"

    def upscale_image(
        self,
        fal_auth: dict,
        image,
        upscale_mode="factor",
        upscale_factor=2.0,
        target_resolution="1080p",
        noise_scale=0.1,
        output_format="png",
        seed=0,
        private_mode=False,
    ):
        """Upscale an image via the SeedVR2 Seamless endpoint on fal.ai."""
        # 1. Autorizace – použijeme FalConfig jako ostatní uzly
        gen_key = fal_auth.get("generation_key")
        FalConfig._client = None  # Force refresh if key changes
        os.environ["FAL_KEY"] = gen_key
        FalConfig.get_client()

        # 2. Upload vstupního tensoru do fal.ai storage
        image_url = ImageUtils.upload_image(image)

        # 3. Příprava argumentů
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

        # 4. Požadavek na fal.ai endpoint
        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/seedvr/upscale/image/seamless",
                arguments,
                disable_storage=private_mode,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 5. Získání URL zvětšeného obrázku
        output_image_obj = result.get("image", {})
        output_image_url = output_image_obj.get("url") if output_image_obj else None

        if not output_image_url:
            raise ValueError(
                f"SeedVR2 API nevrátilo žádný obrázek. Odpověď: {result}"
            )

        # 6. Bezpečné stažení / dekódování obrázku
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

        # Sestavení řetězce statistik pro výstupní konektor
        size_desc = f"{upscale_mode} {upscale_factor}x {target_resolution}"
        usage_stats = _build_usage_stats(fal_auth, result, size_desc)

        return (image_tensor, usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_SeedVR2Seamless": FalSeedVR2SeamlessNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_SeedVR2Seamless": "SeedVR2 Seamless Upscale (fal.ai)",
}