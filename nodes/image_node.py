import os
import requests
import fal_client
from fal_client.client import FalClientHTTPError

from .fal_utils import ImageUtils, ResultProcessor, extract_fal_error_message
from .Fal_Auth_node import FalAuthVault

ASPECT_RATIOS_STANDARD = [
    "auto", "21:9", "16:9", "3:2", "4:3", "5:4",
    "1:1", "4:5", "3:4", "2:3", "9:16",
]

ASPECT_RATIOS_EXTENDED = ASPECT_RATIOS_STANDARD + ["4:1", "1:4", "8:1", "1:8"]


def _fetch_balance(admin_key):
    """Získá aktuální zůstatek kreditu z fal.ai billing API."""
    if not admin_key:
        return ("N/A", "USD")

    url = "https://api.fal.ai/v1/account/billing"
    headers = {"Authorization": f"Key {admin_key}"}

    try:
        response = requests.get(
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


def _build_usage_stats(fal_auth, result, resolution, num_images):
    """Sestaví textový výstup se statistikami generování."""
    creds = FalAuthVault.get_credentials(fal_auth)
    account_name = creds.get("account_name", "Neznámý")
    admin_key = creds.get("admin_key")

    images_count = (
        len(result.get("images", [])) if isinstance(result, dict) else num_images
    )
    seed_used = result.get("seed", "N/A") if isinstance(result, dict) else "N/A"

    timings = (result.get("timings") if isinstance(result, dict) else None) or {}
    inference_time = timings.get("inference")
    time_str = (
        f"{inference_time:.2f} s" if isinstance(inference_time, (int, float)) else "N/A"
    )

    nsfw_flags = result.get("has_nsfw_concepts") if isinstance(result, dict) else None
    if nsfw_flags:
        nsfw_str = f"{sum(1 for f in nsfw_flags if f)}/{len(nsfw_flags)}"
    else:
        nsfw_str = "N/A"

    expanded_prompt_line = ""
    if isinstance(result, dict):
        actual_prompt = result.get("actual_prompt")
        original_prompt = result.get("prompt")
        if actual_prompt and actual_prompt != original_prompt:
            expanded_prompt_line = f"• Rozšířený prompt: {actual_prompt}\n"

    balance, currency = _fetch_balance(admin_key)

    return (
        f"📊 Statistiky generování (Účet: {account_name}):\n"
        f"• Vygenerováno obrázků: {images_count}\n"
        f"• Rozlišení / velikost: {resolution}\n"
        f"• Použitý seed: {seed_used}\n"
        f"• Doba generování: {time_str}\n"
        f"• NSFW obsah (obrázky): {nsfw_str}\n"
        f"{expanded_prompt_line}"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class NanoBanana2Node:
    """Text-to-image generation using fal.ai's Nano Banana 2 (Gemini Flash)."""

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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "aspect_ratio": (ASPECT_RATIOS_EXTENDED, {"default": "auto"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "thinking_level": (["none", "minimal", "high"], {"default": "none"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth,
        prompt,
        num_images=1,
        seed=-1,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        safety_tolerance=4,
        thinking_level="none",
        enable_web_search=False,
        private_mode=False,
    ):
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "enable_web_search": enable_web_search,
        }

        if seed != -1:
            arguments["seed"] = seed
        if thinking_level != "none":
            arguments["thinking_level"] = thinking_level

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/nano-banana-2",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, resolution, num_images)
        return (ResultProcessor.process_image_result(result), usage_stats)


class NanoBanana2EditNode:
    """Image editing using fal.ai's Nano Banana 2 Edit (Gemini Flash)."""

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
                "image4": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "aspect_ratio": (ASPECT_RATIOS_EXTENDED, {"default": "auto"}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "thinking_level": (["none", "minimal", "high"], {"default": "none"}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "edit"

    def edit(
        self,
        fal_auth,
        image,
        prompt,
        image2=None,
        image3=None,
        image4=None,
        num_images=1,
        seed=-1,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        safety_tolerance=4,
        thinking_level="none",
        enable_web_search=False,
        private_mode=False,
    ):
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(image, image2, image3, image4)

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "enable_web_search": enable_web_search,
        }

        if seed != -1:
            arguments["seed"] = seed
        if thinking_level != "none":
            arguments["thinking_level"] = thinking_level

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/nano-banana-2/edit",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, resolution, num_images)
        return (ResultProcessor.process_image_result(result), usage_stats)


class NanoBananaProNode:
    """Text-to-image generation using fal.ai's Nano Banana Pro."""

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
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "aspect_ratio": (ASPECT_RATIOS_STANDARD, {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "generate"

    def generate(
        self,
        fal_auth,
        prompt,
        num_images=1,
        seed=-1,
        aspect_ratio="1:1",
        resolution="1K",
        output_format="png",
        safety_tolerance=4,
        enable_web_search=False,
        private_mode=False,
    ):
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "enable_web_search": enable_web_search,
        }

        if seed != -1:
            arguments["seed"] = seed

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/nano-banana-pro",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, resolution, num_images)
        return (ResultProcessor.process_image_result(result), usage_stats)


class NanoBananaProEditNode:
    """Image editing using fal.ai's Nano Banana Pro Edit."""

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
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "aspect_ratio": (ASPECT_RATIOS_STANDARD, {"default": "auto"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "safety_tolerance": ("INT", {"default": 4, "min": 1, "max": 6}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "usage_stats")
    FUNCTION = "edit"

    def edit(
        self,
        fal_auth,
        image,
        prompt,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        image11=None,
        image12=None,
        image13=None,
        image14=None,
        num_images=1,
        seed=-1,
        aspect_ratio="auto",
        resolution="1K",
        output_format="png",
        safety_tolerance=4,
        enable_web_search=False,
        private_mode=False,
    ):
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(
            image,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
            image11,
            image12,
            image13,
            image14,
        )

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
            "safety_tolerance": safety_tolerance,
            "enable_web_search": enable_web_search,
        }

        if seed != -1:
            arguments["seed"] = seed

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/nano-banana-pro/edit",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, resolution, num_images)
        return (ResultProcessor.process_image_result(result), usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_NanoBanana2": NanoBanana2Node,
    "FAL_NanoBanana2Edit": NanoBanana2EditNode,
    "FAL_NanoBananaPro": NanoBananaProNode,
    "FAL_NanoBananaProEdit": NanoBananaProEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_NanoBanana2": "Nano Banana 2 (fal.ai)",
    "FAL_NanoBanana2Edit": "Nano Banana 2 Edit (fal.ai)",
    "FAL_NanoBananaPro": "Nano Banana Pro (fal.ai)",
    "FAL_NanoBananaProEdit": "Nano Banana Pro Edit (fal.ai)",
}
