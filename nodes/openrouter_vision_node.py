import os
import configparser
import tempfile
import torch
import numpy as np
from PIL import Image
import fal_client
import requests
import json

# Omezený seznam modelů přesně podle zadání
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4-fast"
]

def load_fal_key_from_config() -> str:
    """Prohledá složku uzlu a načte FAL_KEY ze souboru config.ini."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Hledá config.ini v aktuální složce nebo o úroveň výš
    possible_paths = [
        os.path.join(current_dir, "config.ini"),
        os.path.join(os.path.dirname(current_dir), "config.ini"),
    ]

    for config_path in possible_paths:
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path, encoding="utf-8")
            if "API" in config and "FAL_KEY" in config["API"]:
                key = config["API"]["FAL_KEY"].strip()
                if key:
                  return key

    return os.environ.get("FAL_KEY", "").strip()

def load_fal_key_admin_from_config() -> str:
    """Prohledá složku uzlu a načte FAL_KEY ze souboru config.ini."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Hledá config.ini v aktuální složce nebo o úroveň výš
    possible_paths = [
        os.path.join(current_dir, "config.ini"),
        os.path.join(os.path.dirname(current_dir), "config.ini"),
    ]

    for config_path in possible_paths:
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path, encoding="utf-8")
            if "API" in config and "FAL_KEY_ADMIN" in config["API"]:
                key = config["API"]["FAL_KEY_ADMIN"].strip()
                if key:
                    return key

    return os.environ.get("FAL_KEY", "").strip()

class FAL_OpenRouterVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Popiš detailně tento obrázek.",
                    "dynamicPrompts": False
                }),
                "model": (OPENROUTER_MODELS, {
                    "default": "google/gemini-2.5-flash"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "enable_reasoning": ("BOOLEAN", {
                    "default": False
                }),
                "streaming": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    # Přidán druhý výstup pro statistiky
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "usage_stats")
    FUNCTION = "generate_vision_response"
    CATEGORY = "fal.ai/LLM"

    def _ensure_auth(self):
        """Zajistí, že je FAL_KEY správně nastaven v os.environ pro fal_client."""
        fal_key = load_fal_key_from_config()
        fal_key_admin = load_fal_key_admin_from_config()

        if not fal_key:
            raise RuntimeError(
                "Nenalezen FAL_KEY! Ujistěte se, že máte v kořenu custom nodu soubor config.ini "
                "obsahující sekci [API] s položkou FAL_KEY."
            )
        os.environ["FAL_KEY"] = fal_key
        if not fal_key_admin:
            raise RuntimeError(
                "Nenalezen FAL_KEY_ADMIN! Ujistěte se, že máte v kořenu custom nodu soubor config.ini "
                "obsahující sekci [API] s položkou FAL_KEY."
            )
        os.environ["FAL_KEY_ADMIN"] = fal_key_admin

    def _tensor_to_fal_url(self, image_tensor: torch.Tensor) -> str:
        """Převede ComfyUI IMAGE tensor na dočasný PNG soubor a nahraje ho na CDN fal.ai."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        img_np = (255.0 * image_tensor.cpu().numpy()).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        try:
            image_url = fal_client.upload_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return image_url

    def generate_vision_response(
        self,
        image: torch.Tensor,
        prompt: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        enable_reasoning: bool = False,
        streaming: bool = False
    ):
        self._ensure_auth()
        image_url = self._tensor_to_fal_url(image)

        arguments = {
            "image_urls": [image_url],
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_reasoning": enable_reasoning,
            "reasoning": enable_reasoning,
            "thinking": enable_reasoning,
        }
        
        if streaming:
            arguments["stream"] = True

        if system_prompt and system_prompt.strip():
            arguments["system_prompt"] = system_prompt.strip()

        result = fal_client.subscribe(
            "openrouter/router/vision",
            arguments=arguments
        )

        # Hlavní textový výstup
        text_output = result.get("output", "")

        # Extrakce metadat (struktura závisí na tom, co fal.ai předává z OpenRouteru)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", "N/A")
        completion_tokens = usage.get("completion_tokens", "N/A")
        total_tokens = usage.get("total_tokens", "N/A")
        cost = usage.get("cost", "N/A")



        # Připojení k API pro zjištění kreditu
        url = "https://api.fal.ai/v1/account/billing"

        # Bezpečné použití načteného klíče s prefixem 'Key'
        headers = {"Authorization": "Key " + load_fal_key_admin_from_config()}

        try:
            # 'expand=credits' je nutné, jinak API pole 'credits' vůbec nevrátí
            response = requests.get(
                url,
                headers=headers,
                params={"expand": "credits"},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            # Bezpečné čtení - zabrání pádu uzlu, pokud by API mělo výpadek
            credits_dict = data.get("credits", {})
            balance = credits_dict.get("current_balance", "N/A")
            currency = credits_dict.get("currency", "USD")

        except Exception as e:
            balance = f"Chyba při zjišťování ({e})"
            currency = "USD"

        # Formátování statistik do jednoho přehledného řetězce
        usage_stats = (
            f"📊 Statistiky generování:\n"
            f"• Tokeny promptu: {prompt_tokens}\n"
            f"• Tokeny odpovědi: {completion_tokens}\n"
            f"• Tokeny celkem: {total_tokens}\n"
            f"• Cena za požadavek: {cost}\n"
            f"• Zbývá kreditu: {balance} {currency}"
            # Z výpisu jsem odstranil {headers}, aby se tvůj klíč neukazoval na obrazovce!
        )

        # Návrat obou hodnot
        return (text_output, usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_OpenRouterVision": FAL_OpenRouterVision
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_OpenRouterVision": "fal.ai OpenRouter Vision"
}