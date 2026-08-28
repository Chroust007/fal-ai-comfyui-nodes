import os
import configparser
from typing import Any, Dict, Tuple
import fal_client
import requests
import json

from .Fal_Auth_node import FalAuthVault

# Seznam modelů dostupných přes fal.ai OpenRouter Enterprise
OPENROUTER_MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash-preview-09-2025",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "deepseek/deepseek-v3.2",
    "deepcogito/cogito-v2.1-671b",
    "deepcogito/cogito-v2-preview-llama-405b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-30b-a3b",
    "amazon/nova-premier-v1",
    "amazon/nova-2-lite-v1",
    "moonshotai/kimi-k2.5",
]


# ====================================================================
# UZEL: TEXTOVÝ LLM UZEL (bez obrázku)
# https://fal.ai/models/openrouter/router/enterprise/api
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
                    "default": "anthropic/claude-sonnet-5"
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
                "enable_web_search": ("BOOLEAN", {
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
        enable_web_search: bool = False
    ) -> Tuple[str, str]:
        # 1. NASTAVENÍ AUTORIZACE Z PROPOJENÉHO UZLU PŘES IN-MEMORY VAULT
        # Klíče se vyzvednou z bezpečné paměti Pythonu těsně před odesláním požadavku
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
            "reasoning": enable_reasoning,
            "enable_web_search": enable_web_search,
        }

        if system_prompt and system_prompt.strip():
            arguments["system_prompt"] = system_prompt.strip()

        # 3. VOLÁNÍ API (Generování) - Enterprise endpoint
        result = fal_client.subscribe(
            "openrouter/router/enterprise",
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
    "FAL_OpenRouterEnterpriseLLMAPI": FAL_OpenRouterLLMAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_OpenRouterEnterpriseLLMAPI": "fal.ai OpenRouter Enterprise LLM API (Text)"
}