import os

import fal_client
import requests
from fal_client.client import FalClientHTTPError

from .fal_utils import ImageUtils, ResultProcessor, extract_fal_error_message
from .Fal_Auth_node import FalAuthVault

IMAGE_SIZE_PRESETS = [
    "auto_2K",
    "auto_3K",
    "auto_4K",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]

# fal.ai vyžaduje, aby celkový počet pixelů (width * height) byl v rozsahu
# 2560x1440 (3 686 400 px) až 4096x4096 (16 777 216 px). Mimo tento rozsah
# si hodnotu API samo přeškáluje, což vede k neshodě s požadovaným rozlišením.
MIN_TOTAL_PIXELS = 2560 * 1440
MAX_TOTAL_PIXELS = 4096 * 4096

# Varianta Pro Edit má odlišný povolený rozsah: 1024x1024 (1 048 576 px)
# až 2048x2048 (4 194 304 px).
PRO_MIN_TOTAL_PIXELS = 1024 * 1024
PRO_MAX_TOTAL_PIXELS = 2048 * 2048

IMAGE_SIZE_PRESETS_PRO = [
    "auto_1K",
    "auto_2K",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]


def _clamp_custom_size(width, height, min_total_pixels=MIN_TOTAL_PIXELS, max_total_pixels=MAX_TOTAL_PIXELS):
    """Přeškáluje width/height (při zachování poměru stran) tak, aby celkový
    počet pixelů odpovídal limitům fal.ai API. Vrací (width, height, adjusted)."""
    total = width * height
    if total < min_total_pixels:
        scale = (min_total_pixels / total) ** 0.5
    elif total > max_total_pixels:
        scale = (max_total_pixels / total) ** 0.5
    else:
        return width, height, False
    return max(1, round(width * scale)), max(1, round(height * scale)), True


def _build_usage_stats(fal_auth, result, image_size):
    """Sestaví textový výstup se statistikami generování, stejně jako
    u uzlu FAL_OpenRouterLLMAPI (počet obrázků, velikost, zůstatek kreditu)."""
    creds = FalAuthVault.get_credentials(fal_auth)
    account_name = creds.get("account_name", "Neznámý")
    admin_key = creds.get("admin_key")

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


class FAL_SeedreamV5LiteEdit:
    """Editace obrázků pomocí fal.ai Bytedance SeedDream v5 Lite Edit.

    Autorizace je řešena přes propojený uzel Fal_Auth (FAL_AUTH),
    ne přes globální config.ini/FAL_KEY.
    """

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
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "auto_2K"}),
                "custom_width": ("INT", {"default": 2560, "min": 1, "max": 14142}),
                "custom_height": ("INT", {"default": 1440, "min": 1, "max": 14142}),
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
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        num_images=1,
        max_images=1,
        image_size="auto_2K",
        custom_width=2560,
        custom_height=1440,
        enable_safety_checker=True,
        private_mode=False,
    ):
        # Autorizace přes propojený Fal_Auth uzel (In-Memory Vault)
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(
            image, image2, image3, image4, image5, image6, image7, image8, image9, image10
        )

        if image_size == "custom":
            width, height, adjusted = _clamp_custom_size(custom_width, custom_height)
            if adjusted:
                print(
                    f"[FAL_SeedreamV5LiteEdit] custom_width/custom_height "
                    f"({custom_width}x{custom_height} = {custom_width * custom_height} px) je mimo "
                    f"povoleny rozsah fal.ai (2560x1440 az 4096x4096 px). "
                    f"Upraveno na {width}x{height}, aby vystup odpovidal zadani."
                )
            size = {"width": width, "height": height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "max_images": max_images,
            "image_size": size,
            "enable_safety_checker": enable_safety_checker,
        }

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/bytedance/seedream/v5/lite/edit",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


class FAL_SeedreamV5ProEdit:
    """Editace obrázků pomocí fal.ai Bytedance SeedDream v5 Pro Edit.

    Autorizace je řešena přes propojený uzel Fal_Auth (FAL_AUTH),
    ne přes globální config.ini/FAL_KEY.
    """

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
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "image_size": (IMAGE_SIZE_PRESETS_PRO, {"default": "auto_2K"}),
                "custom_width": ("INT", {"default": 2048, "min": 1, "max": 14142}),
                "custom_height": ("INT", {"default": 2048, "min": 1, "max": 14142}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
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
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        num_images=1,
        image_size="auto_2K",
        custom_width=2048,
        custom_height=2048,
        output_format="jpeg",
        enable_safety_checker=True,
        private_mode=False,
    ):
        # Autorizace přes propojený Fal_Auth uzel (In-Memory Vault)
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key")
        if not gen_key:
            raise RuntimeError(
                "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
            )
        os.environ["FAL_KEY"] = gen_key

        image_urls = ImageUtils.collect_image_urls(
            image, image2, image3, image4, image5, image6, image7, image8, image9, image10
        )

        if image_size == "custom":
            width, height, adjusted = _clamp_custom_size(
                custom_width, custom_height, PRO_MIN_TOTAL_PIXELS, PRO_MAX_TOTAL_PIXELS
            )
            if adjusted:
                print(
                    f"[FAL_SeedreamV5ProEdit] custom_width/custom_height "
                    f"({custom_width}x{custom_height} = {custom_width * custom_height} px) je mimo "
                    f"povoleny rozsah fal.ai (1024x1024 az 2048x2048 px). "
                    f"Upraveno na {width}x{height}, aby vystup odpovidal zadani."
                )
            size = {"width": width, "height": height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "image_size": size,
            "output_format": output_format,
            "enable_safety_checker": enable_safety_checker,
        }

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "bytedance/seedream/v5/pro/edit",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_SeedreamV5LiteEdit": FAL_SeedreamV5LiteEdit,
    "FAL_SeedreamV5ProEdit": FAL_SeedreamV5ProEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_SeedreamV5LiteEdit": "SeedDream v5 Lite Edit (fal.ai)",
    "FAL_SeedreamV5ProEdit": "SeedDream v5 Pro Edit (fal.ai)",
}
