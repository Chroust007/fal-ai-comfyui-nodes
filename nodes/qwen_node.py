import os

import fal_client
import requests
from fal_client.client import FalClientHTTPError

from .fal_utils import ImageUtils, ResultProcessor, extract_fal_error_message

IMAGE_SIZE_PRESETS = [
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]


def _build_usage_stats(fal_auth, result, image_size):
    """Sestaví textový výstup se statistikami generování, stejně jako
    u uzlu FAL_OpenRouterLLMAPI (počet obrázků, velikost, zůstatek kreditu)."""
    account_name = fal_auth.get("account_name", "Neznámý") if fal_auth else "Neznámý"
    admin_key = fal_auth.get("admin_key") if fal_auth else None

    images_count = len(result.get("images", []))

    balance = "N/A"
    currency = "USD"
    if admin_key:
        url = "https://api.fal.ai/v1/account/billing"
        headers = {"Authorization": f"Key {admin_key}"}
        try:
            response = requests.get(
                url,
                headers=headers,
                params={"expand": "credits"},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            credits_dict = data.get("credits", {})
            balance = credits_dict.get("current_balance", "N/A")
            currency = credits_dict.get("currency", "USD")
        except Exception as e:
            balance = f"Chyba při zjišťování ({e})"

    return (
        f"📊 Statistiky generování (Účet: {account_name}):\n"
        f"• Vygenerováno obrázků: {images_count}\n"
        f"• Velikost obrázku: {image_size}\n"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class QwenImageNode:
    """Text-to-image generation using fal.ai's Qwen Image."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "landscape_4_3"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 14142}),
                "custom_height": ("INT", {"default": 768, "min": 64, "max": 14142}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 250}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
                "use_turbo": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth: dict,
        prompt,
        num_images=1,
        image_size="landscape_4_3",
        custom_width=1024,
        custom_height=768,
        num_inference_steps=30,
        guidance_scale=2.5,
        seed=-1,
        negative_prompt="",
        output_format="png",
        acceleration="none",
        use_turbo=False,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "image_size": size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": output_format,
            "acceleration": acceleration,
            "use_turbo": use_turbo,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImageToImageNode:
    """Image-to-image generation using fal.ai's Qwen Image."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "landscape_4_3"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 14142}),
                "custom_height": ("INT", {"default": 768, "min": 64, "max": 14142}),
                "num_inference_steps": ("INT", {"default": 30, "min": 2, "max": 250}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "none"}),
                "use_turbo": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth: dict,
        image,
        prompt,
        strength=0.6,
        num_images=1,
        image_size="landscape_4_3",
        custom_width=1024,
        custom_height=768,
        num_inference_steps=30,
        guidance_scale=2.5,
        seed=-1,
        negative_prompt="",
        output_format="png",
        acceleration="none",
        use_turbo=False,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        image_url = ImageUtils.upload_image(image)

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "strength": strength,
            "num_images": num_images,
            "image_size": size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": output_format,
            "acceleration": acceleration,
            "use_turbo": use_turbo,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image/image-to-image", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImageEditNode:
    """Image editing using fal.ai's Qwen Image Edit 2511."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "landscape_4_3"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 14142}),
                "custom_height": ("INT", {"default": 768, "min": 64, "max": 14142}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "edit"

    def edit(
        self,
        fal_auth: dict,
        image,
        prompt,
        image2=None,
        image3=None,
        num_images=1,
        image_size="landscape_4_3",
        custom_width=1024,
        custom_height=768,
        num_inference_steps=28,
        guidance_scale=4.5,
        seed=-1,
        negative_prompt="",
        output_format="png",
        acceleration="regular",
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(image, image2, image3)

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "image_size": size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": output_format,
            "acceleration": acceleration,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-edit-2511", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImage2Node:
    """Text-to-image generation using fal.ai's Qwen Image 2."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "square_hd"}),
                "custom_width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "custom_height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth: dict,
        prompt,
        num_images=1,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        seed=-1,
        negative_prompt="",
        output_format="png",
        enable_prompt_expansion=True,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "image_size": size,
            "output_format": output_format,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-2/text-to-image", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImage2EditNode:
    """Image editing using fal.ai's Qwen Image 2 Edit."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "square_hd"}),
                "custom_width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "custom_height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "edit"

    def edit(
        self,
        fal_auth: dict,
        image,
        prompt,
        image2=None,
        image3=None,
        num_images=1,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        seed=-1,
        negative_prompt="",
        output_format="png",
        enable_prompt_expansion=True,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(image, image2, image3)

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "image_size": size,
            "output_format": output_format,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-2/edit", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImage2ProNode:
    """Text-to-image generation using fal.ai's Qwen Image 2 Pro."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "square_hd"}),
                "custom_width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "custom_height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth: dict,
        prompt,
        num_images=1,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        seed=-1,
        negative_prompt="",
        output_format="png",
        enable_prompt_expansion=True,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "image_size": size,
            "output_format": output_format,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-2/pro/text-to-image", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class QwenImage2ProEditNode:
    """Image editing using fal.ai's Qwen Image 2 Pro Edit."""

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "square_hd"}),
                "custom_width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "custom_height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "edit"

    def edit(
        self,
        fal_auth: dict,
        image,
        prompt,
        image2=None,
        image3=None,
        num_images=1,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        seed=-1,
        negative_prompt="",
        output_format="png",
        enable_prompt_expansion=True,
        enable_safety_checker=True,
        private_mode=False,
    ):
        gen_key = fal_auth.get("generation_key") if fal_auth else None
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key)."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(image, image2, image3)

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "image_size": size,
            "output_format": output_format,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed != -1:
            arguments["seed"] = seed
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/qwen-image-2/pro/edit", arguments=arguments, headers=headers
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_QwenImage": QwenImageNode,
    "FAL_QwenImageToImage": QwenImageToImageNode,
    "FAL_QwenImageEdit": QwenImageEditNode,
    "FAL_QwenImage2": QwenImage2Node,
    "FAL_QwenImage2Edit": QwenImage2EditNode,
    "FAL_QwenImage2Pro": QwenImage2ProNode,
    "FAL_QwenImage2ProEdit": QwenImage2ProEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_QwenImage": "Qwen Image (fal.ai)",
    "FAL_QwenImageToImage": "Qwen Image-to-Image (fal.ai)",
    "FAL_QwenImageEdit": "Qwen Image Edit (fal.ai)",
    "FAL_QwenImage2": "Qwen Image 2 (fal.ai)",
    "FAL_QwenImage2Edit": "Qwen Image 2 Edit (fal.ai)",
    "FAL_QwenImage2Pro": "Qwen Image 2 Pro (fal.ai)",
    "FAL_QwenImage2ProEdit": "Qwen Image 2 Pro Edit (fal.ai)",
}
