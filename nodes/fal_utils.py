import base64
import configparser
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import fal_client as fal

CONFIG_PATH = Path(__file__).parent.parent / "config.ini"


class FalConfig:
    """Singleton managing fal.ai API credentials."""

    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            api_key = os.getenv("FAL_KEY", "")
            if not api_key:
                api_key = cls._read_config_key()
            if not api_key:
                raise ValueError(
                    "FAL_KEY not set. Set the FAL_KEY environment variable "
                    "or add it to config.ini."
                )
            os.environ["FAL_KEY"] = api_key
            cls._client = fal
        return cls._client

    @classmethod
    def _read_config_key(cls):
        if not CONFIG_PATH.exists():
            return ""
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        key = config.get("API", "FAL_KEY", fallback="")
        if key.startswith("<") and key.endswith(">"):
            return ""
        return key


class ImageUtils:
    @staticmethod
    def tensor_to_pil(image_tensor):
        """Convert a single HxWxC float tensor (0-1) to a PIL Image."""
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        arr = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def upload_image(image_tensor):
        """Upload an image tensor to fal and return the URL."""
        FalConfig.get_client()
        pil_image = ImageUtils.tensor_to_pil(image_tensor)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_image.save(f, format="PNG")
            tmp_path = f.name
        try:
            url = fal.upload_file(tmp_path)
        finally:
            os.unlink(tmp_path)
        return url

    @staticmethod
    def prepare_image_urls(image_tensor):
        """Upload batch of images and return list of URLs."""
        if image_tensor.dim() == 3:
            return [ImageUtils.upload_image(image_tensor)]
        urls = []
        for i in range(image_tensor.shape[0]):
            urls.append(ImageUtils.upload_image(image_tensor[i]))
        return urls

    @staticmethod
    def collect_image_urls(*image_tensors):
        """Upload multiple optional image tensors and return a flat list of URLs."""
        urls = []
        for t in image_tensors:
            if t is not None:
                urls.append(ImageUtils.upload_image(t))
        return urls


class ResultProcessor:
    @staticmethod
    def process_image_result(result):
        """Download images from result URLs or data URIs and return as tensor batch."""
        import urllib.request

        images = result.get("images", [])
        if not images:
            raise ValueError("No images returned from API")

        tensors = []
        for img_data in images:
            url = img_data["url"]
            if url.startswith("data:"):
                # data URI: data:image/png;base64,...
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req) as resp:
                    data = resp.read()

            pil_image = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(pil_image).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

        return torch.stack(tensors)

    @staticmethod
    def decode_audio_bytes(data: bytes):
        """Decode raw audio bytes (mp3, wav, flac, etc.) to ComfyUI AUDIO dict: {'waveform': tensor [batch, channels, samples], 'sample_rate': int}."""
        # 1. Try PyAV (av)
        try:
            import av

            with av.open(io.BytesIO(data)) as af:
                if af.streams.audio:
                    stream = af.streams.audio[0]
                    sr = stream.codec_context.sample_rate or 44100
                    n_channels = stream.channels or 1
                    frames = []
                    for frame in af.decode(streams=stream.index):
                        buf = torch.from_numpy(frame.to_ndarray())
                        if buf.shape[0] != n_channels:
                            buf = buf.view(-1, n_channels).t()
                        frames.append(buf)
                    if frames:
                        wav = torch.cat(frames, dim=1)
                        if not wav.dtype.is_floating_point:
                            if wav.dtype == torch.int16:
                                wav = wav.float() / 32768.0
                            elif wav.dtype == torch.int32:
                                wav = wav.float() / 2147483648.0
                            elif wav.dtype == torch.uint8:
                                wav = (wav.float() - 128.0) / 128.0
                            else:
                                wav = wav.float()
                        if wav.dim() == 2:
                            wav = wav.unsqueeze(0)
                        return {"waveform": wav, "sample_rate": int(sr)}
        except Exception:
            pass

        # 2. Try torchaudio
        try:
            import torchaudio

            waveform, sr = torchaudio.load(io.BytesIO(data))
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            return {"waveform": waveform, "sample_rate": int(sr)}
        except Exception:
            pass

        raise RuntimeError("Failed to decode audio bytes with both PyAV (av) and torchaudio.")

    @staticmethod
    def process_audio_result(result):
        """Download audio from result URL or data URI and return ComfyUI AUDIO dict."""
        import urllib.request

        audio_obj = result.get("audio")
        if not audio_obj:
            raise ValueError(f"No audio found in API response: {result}")

        url = audio_obj.get("url") if isinstance(audio_obj, dict) else str(audio_obj)
        if not url:
            raise ValueError(f"No audio URL in API response: {result}")

        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()

        return ResultProcessor.decode_audio_bytes(data)

    @staticmethod
    def decode_video_bytes(data: bytes):
        """Decode raw video bytes (mp4, webm, etc.) to ComfyUI IMAGE tensor batch [frames, H, W, 3] and fps (float)."""
        # 1. Try PyAV (av)
        try:
            import av

            with av.open(io.BytesIO(data)) as container:
                if container.streams.video:
                    stream = container.streams.video[0]
                    fps = 24.0
                    if stream.average_rate:
                        fps = float(stream.average_rate)
                    elif stream.r_frame_rate:
                        fps = float(stream.r_frame_rate)

                    frames = []
                    for frame in container.decode(stream):
                        img = frame.to_image().convert("RGB")
                        arr = np.array(img).astype(np.float32) / 255.0
                        frames.append(torch.from_numpy(arr))
                    if frames:
                        return torch.stack(frames), float(fps)
        except Exception:
            pass

        # 2. Try OpenCV (cv2)
        try:
            import cv2

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(data)
                tmp_path = f.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    arr = frame_rgb.astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(arr))
                cap.release()
                if frames:
                    return torch.stack(frames), fps
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception:
            pass

        # 3. Try imageio
        try:
            import imageio.v3 as iio

            frames_np = iio.imread(io.BytesIO(data), index=None)
            if len(frames_np) > 0:
                arr = frames_np.astype(np.float32) / 255.0
                if arr.ndim == 4:
                    return torch.from_numpy(arr), 24.0
        except Exception:
            pass

        raise RuntimeError("Failed to decode video bytes with PyAV (av), OpenCV (cv2), and imageio.")

    @staticmethod
    def process_video_result(result):
        """Download video from result URL or data URI and return (IMAGE tensor [frames, H, W, 3], fps (float), video_url (str))."""
        import urllib.request

        video_obj = result.get("video")
        if not video_obj:
            raise ValueError(f"No video found in API response: {result}")

        url = video_obj.get("url") if isinstance(video_obj, dict) else str(video_obj)
        if not url:
            raise ValueError(f"No video URL in API response: {result}")

        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()

        frames_tensor, fps = ResultProcessor.decode_video_bytes(data)
        return frames_tensor, fps, url


def extract_fal_error_message(e):
    """Pull a human-readable message from a FalClientHTTPError."""
    body = e.args[0] if e.args else None
    if isinstance(body, list) and body:
        entry = body[0]
        if isinstance(entry, dict):
            return entry.get("msg", str(e))
    return str(e)


class ApiHandler:
    @staticmethod
    def submit_and_get_result(endpoint, arguments, disable_storage=False):
        """Submit a request to fal.ai and return the result."""
        client = FalConfig.get_client()
        headers = {}
        if disable_storage:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True
        result = client.subscribe(endpoint, arguments=arguments, headers=headers)
        return result

    @staticmethod
    def submit_multiple_and_get_results(endpoint, variations, disable_storage=False):
        """Submit multiple requests concurrently."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    ApiHandler.submit_and_get_result, endpoint, args, disable_storage
                )
                for args in variations
            ]
            return [f.result() for f in futures]
