import os
import configparser
import tempfile
import torch
import numpy as np
from PIL import Image
import fal_client
import requests
import json

# Seznam vision modelů dostupných přes fal.ai OpenRouter model selector
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4-fast",
]



# ====================================================================
# UZEL 2: VISION UZEL
# ====================================================================

class FAL_OpenRouterVisionAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Nový povinný vstup - propojení z Auth uzlu
                "fal_auth": ("FAL_AUTH",),
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "usage_stats")
    FUNCTION = "generate_vision_response"
    CATEGORY = "fal.ai/LLM"

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
        fal_auth: dict,
        image: torch.Tensor,
        prompt: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        enable_reasoning: bool = False,
        streaming: bool = False
    ):
        # 1. NASTAVENÍ AUTORIZACE Z PROPOJENÉHO UZLU
        gen_key = fal_auth.get("generation_key")
        admin_key = fal_auth.get("admin_key")
        account_name = fal_auth.get("account_name", "Neznámý")
        
        # Fal_client interně čte os.environ["FAL_KEY"]
        os.environ["FAL_KEY"] = gen_key

        # 2. NAHRÁNÍ OBRÁZKU
        image_url = self._tensor_to_fal_url(image)

        # 3. SESTAVENÍ ARGUMENTŮ
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

        # 4. VOLÁNÍ API (Generování)
        result = fal_client.subscribe(
            "openrouter/router/vision",
            arguments=arguments
        )

        # 5. ZPRACOVÁNÍ VÝSTUPU
        # Zde může být nutné upravit tahání dat podle toho, jakou přesně strukturu 
        # vrací OpenRouter přes Fal.ai. Zatím nechávám tvůj původní kód.
        text_output = result.get("output", "")

        # Extrakce metadat
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", "N/A")
        completion_tokens = usage.get("completion_tokens", "N/A")
        total_tokens = usage.get("total_tokens", "N/A")
        cost = usage.get("cost", "N/A")

        # 6. KONTROLA KREDITU (Vždy přes admin_key)
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
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()

                credits_dict = data.get("credits", {})
                balance = credits_dict.get("current_balance", "N/A")
                currency = credits_dict.get("currency", "USD")

            except Exception as e:
                balance = f"Chyba při zjišťování ({e})"

        usage_stats = (
            f"📊 Statistiky generování (Účet: {account_name}):\n"
            f"• Tokeny promptu: {prompt_tokens}\n"
            f"• Tokeny odpovědi: {completion_tokens}\n"
            f"• Tokeny celkem: {total_tokens}\n"
            f"• Cena za požadavek: {cost}\n"
            f"• Zbývá kreditu: {balance} {currency}"
        )

        return (text_output, usage_stats)

# ====================================================================
# REGISTRACE UZLŮ PRO COMFYUI
# ====================================================================

NODE_CLASS_MAPPINGS = {
    "FAL_OpenRouterVisionAPI": FAL_OpenRouterVisionAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_OpenRouterVisionAPI": "fal.ai OpenRouter Vision API"
}