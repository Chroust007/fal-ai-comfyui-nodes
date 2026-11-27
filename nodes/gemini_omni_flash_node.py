import os
import requests
import fal_client
from fal_client.client import FalClientHTTPError

from .Fal_Auth_node import FalAuthVault
from .fal_utils import (
    FalConfig,
    ImageUtils,
    ResultProcessor,
    VideoUtils,
    extract_fal_error_message,
)

REFERENCE_TO_VIDEO_ENDPOINT = "google/gemini-omni-flash/v1.2/reference-to-video"
IMAGE_TO_VIDEO_ENDPOINT = "google/gemini-omni-flash/v1.2/image-to-video"
ASPECT_RATIO_OPTIONS = ["16:9", "9:16"]
RESOLUTION_OPTIONS = ["360p", "720p", "1080p", "4k"]
MAX_REFERENCE_IMAGES = 10
MAX_REFERENCE_VIDEOS = 3
MAX_PROMPT_LENGTH = 20000
MAX_DURATION_WITH_END_IMAGE = 10


def _configure_fal_auth(fal_auth):
    """Nastaví API klíč přes Fal_Auth uzel nebo fallback na FalConfig / FAL_KEY."""
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
    endpoint,
    aspect_ratio,
    resolution,
    duration,
    extra_lines=None,
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

    balance, currency = _fetch_balance(admin_key)

    lines = [
        f"📊 Statistiky generování videa (Účet: {account_name}):",
        f"• Model: {endpoint}",
        f"• Poměr stran: {aspect_ratio}",
        f"• Rozlišení: {resolution}",
        f"• Délka videa: {duration} s",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend(
        [
            f"• Doba generování: {time_str}",
            f"• Video URL: {video_url}",
            f"• Zbývá kreditu: {balance} {currency}",
        ]
    )
    return "\n".join(lines)


class FAL_GeminiOmniFlashReferenceToVideo:
    """Generování videa pomocí Google Gemini OmniFlash v1.2 Reference-to-Video na fal.ai.

    Model: google/gemini-omni-flash/v1.2/reference-to-video
    Podporuje až 10 referenčních obrázků (<IMAGE_REF_0> .. <IMAGE_REF_9>)
    a až 3 referenční videa (<VIDEO_REF_0> .. <VIDEO_REF_2>, každé max 15 s).
    Výstupem jsou snímky videa jako ComfyUI IMAGE tensor batch, URL videa, FPS a statistiky.
    """

    CATEGORY = "fal.ai/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Např.: A cat inspired by <IMAGE_REF_0> walks through the setting in <VIDEO_REF_0>.",
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "reference_video_1": ("VIDEO",),
                "reference_video_2": ("VIDEO",),
                "reference_video_3": ("VIDEO",),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "duration": (
                    "INT",
                    {"default": 8, "min": 3, "max": 15, "step": 1},
                ),
                "custom_image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Volitelné URL obrázků (1 URL na řádek, max 10 celkem)",
                    },
                ),
                "reference_video_url_1": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná URL nebo cesta k referenčnímu videu 1 (max 15 s)",
                    },
                ),
                "reference_video_url_2": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná URL nebo cesta k referenčnímu videu 2 (max 15 s)",
                    },
                ),
                "reference_video_url_3": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná URL nebo cesta k referenčnímu videu 3 (max 15 s)",
                    },
                ),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "video_url", "fps", "usage_stats")
    FUNCTION = "generate_video"

    def generate_video(
        self,
        prompt: str,
        fal_auth=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
        reference_video_1=None,
        reference_video_2=None,
        reference_video_3=None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration: int = 8,
        custom_image_urls: str = "",
        reference_video_url_1: str = "",
        reference_video_url_2: str = "",
        reference_video_url_3: str = "",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        cleaned_prompt = prompt.strip()
        if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt přesahuje maximální povolenou délku {MAX_PROMPT_LENGTH} znaků "
                f"(aktuálně {len(cleaned_prompt)})."
            )

        # 1. Autorizace přes Fal_Auth uzel nebo fallback na FalConfig / FAL_KEY
        _configure_fal_auth(fal_auth)

        # 2. Zpracování referenčních obrázků (max 10)
        image_urls = ImageUtils.collect_image_urls(
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        )
        if custom_image_urls and custom_image_urls.strip():
            for line in custom_image_urls.splitlines():
                url = line.strip()
                if url:
                    image_urls.append(url)

        if len(image_urls) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Maximální počet referenčních obrázků je {MAX_REFERENCE_IMAGES} "
                f"(zadáno {len(image_urls)})."
            )

        # 3. Zpracování referenčních videí (max 3, pořadí odpovídá <VIDEO_REF_0..2>)
        reference_video_urls = []
        for video_url_str, video_obj in (
            (reference_video_url_1, reference_video_1),
            (reference_video_url_2, reference_video_2),
            (reference_video_url_3, reference_video_3),
        ):
            resolved_url = None
            if video_url_str and video_url_str.strip():
                resolved_url = VideoUtils.upload_video(video_url_str)
            elif video_obj is not None:
                resolved_url = VideoUtils.upload_video(video_obj)
            if resolved_url:
                reference_video_urls.append(resolved_url)

        # 4. Sestavení argumentů pro API
        arguments = {
            "prompt": cleaned_prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": int(duration),
        }

        if image_urls:
            arguments["image_urls"] = image_urls

        if reference_video_urls:
            arguments["reference_video_urls"] = reference_video_urls

        # 5. Hlavičky (private_mode)
        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        # 6. Volání API fal.ai
        try:
            result = fal_client.subscribe(
                REFERENCE_TO_VIDEO_ENDPOINT,
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 7. Zpracování výstupu (dekódování videa na frames + fps)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            endpoint=REFERENCE_TO_VIDEO_ENDPOINT,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=int(duration),
            extra_lines=[
                f"• Referenční obrázky: {len(image_urls)}",
                f"• Referenční videa: {len(reference_video_urls)}",
            ],
        )

        return (
            frames_tensor,
            video_url,
            fps,
            usage_stats,
        )


class FAL_GeminiOmniFlashImageToVideo:
    """Generování videa z obrázku pomocí Google Gemini OmniFlash v1.2 Image-to-Video na fal.ai.

    Model: google/gemini-omni-flash/v1.2/image-to-video
    Animuje vstupní obrázek (first frame) do videa s nativním zvukem, s volitelným
    koncovým obrázkem (end_image) pro interpolaci prvního a posledního snímku (max 10 s).
    Výstupem jsou snímky videa jako ComfyUI IMAGE tensor batch, URL videa, FPS a statistiky.
    """

    CATEGORY = "fal.ai/Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Např.: The dog turns its head and wags its tail in warm sunlight.",
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "duration": (
                    "INT",
                    {"default": 8, "min": 3, "max": 15, "step": 1},
                ),
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
                        "placeholder": "Volitelná přímá URL koncového snímku (end_image_url, max délka 10 s)",
                    },
                ),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "video_url", "fps", "usage_stats")
    FUNCTION = "generate_video"

    def generate_video(
        self,
        prompt: str,
        fal_auth=None,
        image=None,
        end_image=None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration: int = 8,
        custom_image_url: str = "",
        custom_end_image_url: str = "",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        cleaned_prompt = prompt.strip()
        if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt přesahuje maximální povolenou délku {MAX_PROMPT_LENGTH} znaků "
                f"(aktuálně {len(cleaned_prompt)})."
            )

        has_first_image = bool(
            (custom_image_url and custom_image_url.strip()) or image is not None
        )
        if not has_first_image:
            raise ValueError(
                "Je nutné zadat vstupní obrázek (image) nebo URL prvního snímku (custom_image_url)."
            )

        has_end_image = bool(
            (custom_end_image_url and custom_end_image_url.strip())
            or end_image is not None
        )
        duration_int = int(duration)
        if has_end_image and duration_int > MAX_DURATION_WITH_END_IMAGE:
            raise ValueError(
                f"Při zadání koncového snímku (end_image) je maximální délka videa "
                f"{MAX_DURATION_WITH_END_IMAGE} sekund (zadáno {duration_int} s)."
            )

        # 1. Autorizace přes Fal_Auth uzel nebo fallback na FalConfig / FAL_KEY
        _configure_fal_auth(fal_auth)

        # 2. Zpracování vstupních obrázků (první a volitelný koncový snímek)
        if custom_image_url and custom_image_url.strip():
            resolved_image_url = custom_image_url.strip()
        else:
            resolved_image_url = ImageUtils.upload_image(image)

        resolved_end_image_url = None
        if custom_end_image_url and custom_end_image_url.strip():
            resolved_end_image_url = custom_end_image_url.strip()
        elif end_image is not None:
            resolved_end_image_url = ImageUtils.upload_image(end_image)

        # 3. Sestavení argumentů pro API
        arguments = {
            "prompt": cleaned_prompt,
            "image_url": resolved_image_url,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration_int,
        }

        if resolved_end_image_url:
            arguments["end_image_url"] = resolved_end_image_url

        # 4. Hlavičky (private_mode)
        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        # 5. Volání API fal.ai
        try:
            result = fal_client.subscribe(
                IMAGE_TO_VIDEO_ENDPOINT,
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 6. Zpracování výstupu (dekódování videa na frames + fps)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        mode_desc = (
            "Image-to-Video (First-to-Last Keyframe)"
            if resolved_end_image_url
            else "Image-to-Video"
        )
        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            endpoint=IMAGE_TO_VIDEO_ENDPOINT,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration_int,
            extra_lines=[f"• Režim: {mode_desc}"],
        )

        return (
            frames_tensor,
            video_url,
            fps,
            usage_stats,
        )


NODE_CLASS_MAPPINGS = {
    "FAL_GeminiOmniFlashReferenceToVideo": FAL_GeminiOmniFlashReferenceToVideo,
    "FAL_GeminiOmniFlashImageToVideo": FAL_GeminiOmniFlashImageToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_GeminiOmniFlashReferenceToVideo": "Gemini OmniFlash v1.2 Reference to Video (fal.ai)",
    "FAL_GeminiOmniFlashImageToVideo": "Gemini OmniFlash v1.2 Image to Video (fal.ai)",
}
