import os

import fal_client
import requests
from fal_client.client import FalClientHTTPError

from .fal_utils import ResultProcessor, extract_fal_error_message
from .Fal_Auth_node import FalAuthVault

IMAGE_SIZE_PRESETS = [
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
    "custom",
]

ACCELERATION_OPTIONS = ["none", "regular"]
OUTPUT_FORMATS = ["jpeg", "png"]


def _build_usage_stats(fal_auth, result, image_size):
    """Sestaví textový výstup se statistikami generování, stejně jako
    u uzlu FAL_OpenRouterLLMAPI (počet obrázků, velikost, seed, čas generování,
    NSFW detekce, rozšířený prompt, zůstatek kreditu)."""
    creds = FalAuthVault.get_credentials(fal_auth)
    account_name = creds.get("account_name", "Neznámý")
    admin_key = creds.get("admin_key")

    images_count = len(result.get("images", []))
    seed_used = result.get("seed", "N/A")

    timings = result.get("timings") or {}
    inference_time = timings.get("inference")
    time_str = f"{inference_time:.2f} s" if isinstance(inference_time, (int, float)) else "N/A"

    nsfw_flags = result.get("has_nsfw_concepts") or []
    nsfw_str = f"{sum(1 for f in nsfw_flags if f)}/{len(nsfw_flags)}" if nsfw_flags else "N/A"

    actual_prompt = result.get("actual_prompt")
    original_prompt = result.get("prompt")
    expanded_prompt_line = ""
    if actual_prompt and actual_prompt != original_prompt:
        expanded_prompt_line = f"• Rozšířený prompt: {actual_prompt}\n"

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
        f"• Použitý seed: {seed_used}\n"
        f"• Doba generování: {time_str}\n"
        f"• NSFW obsah (obrázky): {nsfw_str}\n"
        f"{expanded_prompt_line}"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class FAL_Krea2TurboLoRA:
    """Generování obrázků z textu pomocí fal.ai Krea 2 Turbo LoRA.

    Autorizace je řešena přes propojený uzel Fal_Auth (FAL_AUTH),
    ne přes globální config.ini/FAL_KEY.
    """

    CATEGORY = "fal.ai/Image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fal_auth": ("FAL_AUTH",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": ""}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "lora_path2": ("STRING", {"default": ""}),
                "lora_scale2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "lora_path3": ("STRING", {"default": ""}),
                "lora_scale3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "image_size": (IMAGE_SIZE_PRESETS, {"default": "square_hd"}),
                "custom_width": ("INT", {"default": 1024, "min": 1, "max": 14142}),
                "custom_height": ("INT", {"default": 1024, "min": 1, "max": 14142}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "acceleration": (ACCELERATION_OPTIONS, {"default": "none"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
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
        lora_path="",
        lora_scale=1.0,
        lora_path2="",
        lora_scale2=1.0,
        lora_path3="",
        lora_scale3=1.0,
        seed=-1,
        image_size="square_hd",
        custom_width=1024,
        custom_height=1024,
        num_images=1,
        acceleration="none",
        enable_prompt_expansion=False,
        enable_safety_checker=True,
        output_format="png",
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

        loras = []
        for path, scale in (
            (lora_path, lora_scale),
            (lora_path2, lora_scale2),
            (lora_path3, lora_scale3),
        ):
            if path and path.strip():
                loras.append({"path": path.strip(), "scale": scale})

        if image_size == "custom":
            size = {"width": custom_width, "height": custom_height}
        else:
            size = image_size

        arguments = {
            "prompt": prompt,
            "image_size": size,
            "num_images": num_images,
            "acceleration": acceleration,
            "enable_prompt_expansion": enable_prompt_expansion,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }

        if loras:
            arguments["loras"] = loras

        if seed is not None and seed >= 0:
            arguments["seed"] = seed

        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        try:
            result = fal_client.subscribe(
                "fal-ai/krea-2/turbo/lora",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        usage_stats = _build_usage_stats(fal_auth, result, size)
        return (ResultProcessor.process_image_result(result), usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_Krea2TurboLoRA": FAL_Krea2TurboLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_Krea2TurboLoRA": "Krea 2 Turbo LoRA (fal.ai)",
}
