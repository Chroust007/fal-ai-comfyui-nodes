import os
import requests
from fal_client.client import FalClientHTTPError

import fal_client
from .Fal_Auth_node import FalAuthVault
from .fal_utils import FalConfig, ImageUtils, ResultProcessor, extract_fal_error_message

RESOLUTION_OPTIONS = ["768P", "480P"]
PROMPT_EXPANSION_MODES = ["balanced", "quality"]


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


def _build_usage_stats(
    fal_auth,
    result,
    resolution,
    duration,
    prompt_expansion_mode,
    has_image,
    has_end_image,
    seed_used,
):
    """Sestaví textový výstup se statistikami generování videa."""
    account_name = "Neznámý"
    admin_key = None
    if fal_auth:
        creds = FalAuthVault.get_credentials(fal_auth)
        account_name = creds.get("account_name", "Neznámý")
        admin_key = creds.get("admin_key")

    timings = (result.get("timings") if isinstance(result, dict) else None) or {}
    inference_time = timings.get("inference")
    time_str = (
        f"{inference_time:.2f} s" if isinstance(inference_time, (int, float)) else "N/A"
    )

    video_obj = result.get("video", {}) if isinstance(result, dict) else {}
    video_url = (
        video_obj.get("url", "N/A") if isinstance(video_obj, dict) else str(video_obj)
    )

    expanded_prompt = (
        result.get("expanded_prompt") if isinstance(result, dict) else None
    )
    expanded_prompt_line = ""
    if expanded_prompt:
        expanded_prompt_line = f"• Rozšířený prompt: {expanded_prompt}\n"

    mode_desc = "Image-to-Video" if has_image else "Text-to-Video"
    if has_end_image:
        mode_desc += " (First-to-Last Keyframe)"

    balance, currency = _fetch_balance(admin_key)

    return (
        f"📊 Statistiky generování videa (Účet: {account_name}):\n"
        f"• Model: minimax/h3-max-turbo/image-to-video\n"
        f"• Režim: {mode_desc}\n"
        f"• Rozlišení: {resolution}\n"
        f"• Délka videa: {duration} s\n"
        f"• Expanze promptu: {prompt_expansion_mode}\n"
        f"• Použitý seed: {seed_used}\n"
        f"• Doba generování: {time_str}\n"
        f"• Video URL: {video_url}\n"
        f"{expanded_prompt_line}"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class FAL_MinimaxH3MaxTurboImageToVideo:
    """Generování videa z obrázku nebo textu pomocí fal.ai MiniMax H3 Max Turbo.

    Model: minimax/h3-max-turbo/image-to-video
    Podporuje text-to-video, image-to-video i first-to-last frame animaci (end_image).
    Výstupem jsou jednotlivé snímky videa ve formátu ComfyUI IMAGE tensor batch,
    přímá URL videa, rozšířený prompt, použitý seed, FPS a detailní statistiky.
    """

    CATEGORY = "fal.ai/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "duration": (
                    "INT",
                    {"default": 5, "min": 5, "max": 15, "step": 1},
                ),
                "resolution": (RESOLUTION_OPTIONS, {"default": "768P"}),
                "prompt_expansion_mode": (
                    PROMPT_EXPANSION_MODES,
                    {"default": "balanced"},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "custom_image_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná přímá URL prvního snímku (image_url)",
                    },
                ),
                "custom_end_image_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná přímá URL koncového snímku (end_image_url)",
                    },
                ),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "frames",
        "video_url",
        "expanded_prompt",
        "seed",
        "fps",
        "usage_stats",
    )
    FUNCTION = "generate_video"

    def generate_video(
        self,
        prompt: str,
        fal_auth=None,
        image=None,
        end_image=None,
        duration: int = 5,
        resolution: str = "768P",
        prompt_expansion_mode: str = "balanced",
        seed: int = -1,
        enable_safety_checker: bool = True,
        custom_image_url: str = "",
        custom_end_image_url: str = "",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        # 1. Autorizace přes Fal_Auth uzel nebo fallback na FalConfig / FAL_KEY
        if fal_auth is not None:
            creds = FalAuthVault.get_credentials(fal_auth)
            gen_key = creds.get("generation_key")
            if not gen_key:
                raise RuntimeError(
                    "Uzel Fal_Auth nevrátil platný API klíč (generation_key). "
                    "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován."
                )
            os.environ["FAL_KEY"] = gen_key
        else:
            FalConfig.get_client()

        # 2. Zpracování vstupních obrázků (první a koncový snímek)
        resolved_image_url = None
        if custom_image_url and custom_image_url.strip():
            resolved_image_url = custom_image_url.strip()
        elif image is not None:
            resolved_image_url = ImageUtils.upload_image(image)

        resolved_end_image_url = None
        if custom_end_image_url and custom_end_image_url.strip():
            resolved_end_image_url = custom_end_image_url.strip()
        elif end_image is not None:
            resolved_end_image_url = ImageUtils.upload_image(end_image)

        # 3. Sestavení argumentů pro API
        arguments = {
            "prompt": prompt.strip(),
            "duration": int(duration),
            "resolution": resolution,
            "prompt_expansion_mode": prompt_expansion_mode,
            "enable_safety_checker": enable_safety_checker,
        }

        if resolved_image_url:
            arguments["image_url"] = resolved_image_url

        if resolved_end_image_url:
            arguments["end_image_url"] = resolved_end_image_url

        if seed is not None and seed >= 0:
            arguments["seed"] = seed

        # 4. Hlavičky (private_mode)
        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        # 5. Volání API fal.ai
        try:
            result = fal_client.subscribe(
                "minimax/h3-max-turbo/image-to-video",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 6. Zpracování výstupu (dekódování videa na frames + fps)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        expanded_prompt_out = (
            result.get("expanded_prompt") or prompt.strip()
            if isinstance(result, dict)
            else prompt.strip()
        )
        seed_out = (
            int(result.get("seed", seed if seed >= 0 else 0))
            if isinstance(result, dict)
            else (seed if seed >= 0 else 0)
        )

        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            resolution=resolution,
            duration=duration,
            prompt_expansion_mode=prompt_expansion_mode,
            has_image=(resolved_image_url is not None),
            has_end_image=(resolved_end_image_url is not None),
            seed_used=seed_out,
        )

        return (
            frames_tensor,
            video_url,
            expanded_prompt_out,
            seed_out,
            fps,
            usage_stats,
        )


NODE_CLASS_MAPPINGS = {
    "FAL_MinimaxH3MaxTurboImageToVideo": FAL_MinimaxH3MaxTurboImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_MinimaxH3MaxTurboImageToVideo": "MiniMax H3 Max Turbo Image to Video (fal.ai)",
}
