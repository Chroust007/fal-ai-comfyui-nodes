import os
import requests
import fal_client
from fal_client.client import FalClientHTTPError

from .Fal_Auth_node import FalAuthVault
from .fal_utils import (
    FalConfig,
    ImageUtils,
    ResultProcessor,
    extract_fal_error_message,
)

VEO31_T2V_ENDPOINT = "fal-ai/veo3.1"
VEO31_FAST_T2V_ENDPOINT = "fal-ai/veo3.1/fast"
VEO31_FLF2V_ENDPOINT = "fal-ai/veo3.1/first-last-frame-to-video"
VEO31_FAST_FLF2V_ENDPOINT = "fal-ai/veo3.1/fast/first-last-frame-to-video"
VEO31_R2V_ENDPOINT = "fal-ai/veo3.1/reference-to-video"

ASPECT_RATIO_OPTIONS = ["16:9", "9:16"]
ASPECT_RATIO_AUTO_OPTIONS = ["auto", "16:9", "9:16"]
DURATION_OPTIONS = ["4s", "6s", "8s"]
RESOLUTION_OPTIONS = ["720p", "1080p", "4k"]
SAFETY_TOLERANCE_OPTIONS = ["1", "2", "3", "4", "5", "6"]


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
    generate_audio,
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
        f"• Délka videa: {duration}",
        f"• Generovat zvuk: {'Ano' if generate_audio else 'Ne'}",
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


def _subscribe_veo(endpoint, arguments, private_mode):
    """Odešle požadavek na fal.ai endpoint a zpracuje případnou HTTP chybu."""
    headers = {}
    if private_mode:
        headers["X-Fal-Store-IO"] = "0"
        arguments["sync_mode"] = True

    try:
        return fal_client.subscribe(
            endpoint,
            arguments=arguments,
            headers=headers,
        )
    except FalClientHTTPError as e:
        msg = extract_fal_error_message(e)
        raise RuntimeError(f"fal.ai error: {msg}") from None


class _FAL_Veo31TextToVideoBase:
    """Sdílená základna pro Google Veo 3.1 Text-to-Video nody (Standard a Fast)."""

    CATEGORY = "fal.ai/Video"
    ENDPOINT = VEO31_T2V_ENDPOINT

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Popis videa: subjekt, prostředí, akce, styl, pohyb kamery...",
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "duration": (DURATION_OPTIONS, {"default": "8s"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Volitelný negativní prompt",
                    },
                ),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "auto_fix": ("BOOLEAN", {"default": True}),
                "safety_tolerance": (SAFETY_TOLERANCE_OPTIONS, {"default": "4"}),
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
        aspect_ratio: str = "16:9",
        duration: str = "8s",
        resolution: str = "720p",
        negative_prompt: str = "",
        generate_audio: bool = True,
        seed: int = -1,
        auto_fix: bool = True,
        safety_tolerance: str = "4",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        _configure_fal_auth(fal_auth)

        arguments = {
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "auto_fix": auto_fix,
            "safety_tolerance": safety_tolerance,
        }

        if negative_prompt and negative_prompt.strip():
            arguments["negative_prompt"] = negative_prompt.strip()

        if seed is not None and seed >= 0:
            arguments["seed"] = int(seed)

        result = _subscribe_veo(self.ENDPOINT, arguments, private_mode)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        extra_lines = []
        if seed is not None and seed >= 0:
            extra_lines.append(f"• Seed: {int(seed)}")

        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            endpoint=self.ENDPOINT,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            generate_audio=generate_audio,
            extra_lines=extra_lines,
        )

        return (frames_tensor, video_url, fps, usage_stats)


class FAL_Veo31TextToVideo(_FAL_Veo31TextToVideoBase):
    """Generování videa z textu pomocí Google Veo 3.1 na fal.ai.

    Model: fal-ai/veo3.1
    """

    ENDPOINT = VEO31_T2V_ENDPOINT


class FAL_Veo31FastTextToVideo(_FAL_Veo31TextToVideoBase):
    """Rychlé generování videa z textu pomocí Google Veo 3.1 Fast na fal.ai.

    Model: fal-ai/veo3.1/fast
    """

    ENDPOINT = VEO31_FAST_T2V_ENDPOINT


class _FAL_Veo31FirstLastFrameBase:
    """Sdílená základna pro Google Veo 3.1 First-Last-Frame-to-Video nody (Standard a Fast)."""

    CATEGORY = "fal.ai/Video"
    ENDPOINT = VEO31_FLF2V_ENDPOINT

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Popis přechodu a animace mezi prvním a posledním snímkem...",
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "aspect_ratio": (ASPECT_RATIO_AUTO_OPTIONS, {"default": "auto"}),
                "duration": (DURATION_OPTIONS, {"default": "8s"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Volitelný negativní prompt",
                    },
                ),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "auto_fix": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (SAFETY_TOLERANCE_OPTIONS, {"default": "4"}),
                "custom_first_frame_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná přímá URL prvního snímku (first_frame_url)",
                    },
                ),
                "custom_last_frame_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Volitelná přímá URL posledního snímku (last_frame_url)",
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
        first_frame=None,
        last_frame=None,
        aspect_ratio: str = "auto",
        duration: str = "8s",
        resolution: str = "720p",
        negative_prompt: str = "",
        generate_audio: bool = True,
        seed: int = -1,
        auto_fix: bool = False,
        safety_tolerance: str = "4",
        custom_first_frame_url: str = "",
        custom_last_frame_url: str = "",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        has_first = bool(
            (custom_first_frame_url and custom_first_frame_url.strip())
            or first_frame is not None
        )
        if not has_first:
            raise ValueError(
                "Je nutné zadat první snímek (first_frame) nebo jeho URL (custom_first_frame_url)."
            )

        has_last = bool(
            (custom_last_frame_url and custom_last_frame_url.strip())
            or last_frame is not None
        )
        if not has_last:
            raise ValueError(
                "Je nutné zadat poslední snímek (last_frame) nebo jeho URL (custom_last_frame_url)."
            )

        _configure_fal_auth(fal_auth)

        if custom_first_frame_url and custom_first_frame_url.strip():
            first_frame_url = custom_first_frame_url.strip()
        else:
            first_frame_url = ImageUtils.upload_image(first_frame)

        if custom_last_frame_url and custom_last_frame_url.strip():
            last_frame_url = custom_last_frame_url.strip()
        else:
            last_frame_url = ImageUtils.upload_image(last_frame)

        arguments = {
            "prompt": prompt.strip(),
            "first_frame_url": first_frame_url,
            "last_frame_url": last_frame_url,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "auto_fix": auto_fix,
            "safety_tolerance": safety_tolerance,
        }

        if negative_prompt and negative_prompt.strip():
            arguments["negative_prompt"] = negative_prompt.strip()

        if seed is not None and seed >= 0:
            arguments["seed"] = int(seed)

        result = _subscribe_veo(self.ENDPOINT, arguments, private_mode)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        extra_lines = ["• Režim: First-Last Frame to Video"]
        if seed is not None and seed >= 0:
            extra_lines.append(f"• Seed: {int(seed)}")

        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            endpoint=self.ENDPOINT,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            generate_audio=generate_audio,
            extra_lines=extra_lines,
        )

        return (frames_tensor, video_url, fps, usage_stats)


class FAL_Veo31FirstLastFrameToVideo(_FAL_Veo31FirstLastFrameBase):
    """Generování videa mezi prvním a posledním snímkem pomocí Google Veo 3.1 na fal.ai.

    Model: fal-ai/veo3.1/first-last-frame-to-video
    """

    ENDPOINT = VEO31_FLF2V_ENDPOINT


class FAL_Veo31FastFirstLastFrameToVideo(_FAL_Veo31FirstLastFrameBase):
    """Rychlé generování videa mezi prvním a posledním snímkem pomocí Google Veo 3.1 Fast na fal.ai.

    Model: fal-ai/veo3.1/fast/first-last-frame-to-video
    """

    ENDPOINT = VEO31_FAST_FLF2V_ENDPOINT


class FAL_Veo31ReferenceToVideo:
    """Generování videa z referenčních obrázků a textu pomocí Google Veo 3.1 na fal.ai.

    Model: fal-ai/veo3.1/reference-to-video
    """

    CATEGORY = "fal.ai/Video"
    ENDPOINT = VEO31_R2V_ENDPOINT

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Popis videa s využitím referenčních obrázků...",
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "16:9"}),
                "duration": (DURATION_OPTIONS, {"default": "8s"}),
                "resolution": (RESOLUTION_OPTIONS, {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "auto_fix": ("BOOLEAN", {"default": False}),
                "safety_tolerance": (SAFETY_TOLERANCE_OPTIONS, {"default": "4"}),
                "custom_image_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Volitelné URL referenčních obrázků (1 URL na řádek)",
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
        aspect_ratio: str = "16:9",
        duration: str = "8s",
        resolution: str = "720p",
        generate_audio: bool = True,
        auto_fix: bool = False,
        safety_tolerance: str = "4",
        custom_image_urls: str = "",
        private_mode: bool = False,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("Vstupní prompt pro generování videa nesmí být prázdný.")

        _configure_fal_auth(fal_auth)

        image_urls = ImageUtils.collect_image_urls(image1, image2, image3)
        if custom_image_urls and custom_image_urls.strip():
            for line in custom_image_urls.splitlines():
                url = line.strip()
                if url:
                    image_urls.append(url)

        if not image_urls:
            raise ValueError(
                "Je nutné zadat alespoň jeden referenční obrázek (image1–image3 nebo custom_image_urls)."
            )

        arguments = {
            "prompt": prompt.strip(),
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "auto_fix": auto_fix,
            "safety_tolerance": safety_tolerance,
        }

        result = _subscribe_veo(self.ENDPOINT, arguments, private_mode)
        frames_tensor, fps, video_url = ResultProcessor.process_video_result(result)

        usage_stats = _build_usage_stats(
            fal_auth=fal_auth,
            result=result,
            endpoint=self.ENDPOINT,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            generate_audio=generate_audio,
            extra_lines=[f"• Referenční obrázky: {len(image_urls)}"],
        )

        return (frames_tensor, video_url, fps, usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_Veo31TextToVideo": FAL_Veo31TextToVideo,
    "FAL_Veo31FastTextToVideo": FAL_Veo31FastTextToVideo,
    "FAL_Veo31FirstLastFrameToVideo": FAL_Veo31FirstLastFrameToVideo,
    "FAL_Veo31FastFirstLastFrameToVideo": FAL_Veo31FastFirstLastFrameToVideo,
    "FAL_Veo31ReferenceToVideo": FAL_Veo31ReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_Veo31TextToVideo": "Veo 3.1 Text to Video (fal.ai)",
    "FAL_Veo31FastTextToVideo": "Veo 3.1 Fast Text to Video (fal.ai)",
    "FAL_Veo31FirstLastFrameToVideo": "Veo 3.1 First-Last Frame to Video (fal.ai)",
    "FAL_Veo31FastFirstLastFrameToVideo": "Veo 3.1 Fast First-Last Frame to Video (fal.ai)",
    "FAL_Veo31ReferenceToVideo": "Veo 3.1 Reference to Video (fal.ai)",
}
