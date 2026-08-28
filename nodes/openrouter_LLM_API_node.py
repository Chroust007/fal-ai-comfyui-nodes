import os
import configparser
from typing import Any, Dict, Tuple
import fal_client
import requests
import json

from .Fal_Auth_node import FalAuthVault

# Seznam modelů dostupných přes fal.ai OpenRouter model selector
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4.1",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick",
    "moonshotai/kimi-k2.5",
]


# ====================================================================
# UZEL: TEXTOVÝ LLM UZEL (bez obrázku)
# https://fal.ai/models/openrouter/router/api
# ====================================================================

class FAL_OpenRouterLLMAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Povinný vstup - propojení z Auth uzlu
                "fal_auth": ("FAL_AUTH",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Napiš krátkou odpověď na tento dotaz.",
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
    FUNCTION = "generate_text_response"
    CATEGORY = "fal.ai/LLM"

    def generate_text_response(
        self,
        fal_auth: dict,
        prompt: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 1.0,
        max_tokens: int = 1000,
        enable_reasoning: bool = False,
        streaming: bool = False
    ) -> Tuple[str, str]:
        # 1. NASTAVENÍ AUTORIZACE Z PROPOJENÉHO UZLU PŘES IN-MEMORY VAULT
        creds = FalAuthVault.get_credentials(fal_auth)
        gen_key = creds.get("generation_key", "")
        admin_key = creds.get("admin_key", "")
        account_name = creds.get("account_name", "Neznámý")

        if not gen_key:
            raise RuntimeError(
                "Nepodařilo se získat platný API klíč z Fal_Auth uzlu (In-Memory Vault). "
                "Ujistěte se, že uzel Fal_Auth je správně připojen a nakonfigurován v config.ini."
            )

        # Fal_client interně čte os.environ["FAL_KEY"]
        os.environ["FAL_KEY"] = gen_key

        # 2. SESTAVENÍ ARGUMENTŮ (pouze text, žádný obrázek)
        arguments = {
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

        # 3. VOLÁNÍ API (Generování) - textový endpoint openrouter/router
        result = fal_client.subscribe(
            "openrouter/router",
            arguments=arguments
        )

        # 4. ZPRACOVÁNÍ VÝSTUPU
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
    "FAL_OpenRouterLLMAPI": FAL_OpenRouterLLMAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_OpenRouterLLMAPI": "fal.ai OpenRouter LLM API (Text)"
}