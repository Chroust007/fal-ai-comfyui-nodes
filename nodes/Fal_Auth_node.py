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
# (Mimochodem: doporučuji zkontrolovat přesné ID modelů na OpenRouteru, 
# např. Claude 4.5/4.6 a Grok 4 ještě oficiálně neexistují, OpenRouter by mohl vracet chybu).
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-3-5-sonnet", # Opravený název pro aktuální Sonnet
    "openai/gpt-4o",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-2-vision" # Grok 4 zatím není, dal jsem nejnovější dostupný vision od xAI
]


class FAL_Auth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "account": (["Chroust", "Seznam"], {"default": "Chroust"}),
                # Přepínač Role určí, který klíč se použije pro SAMOTNÉ GENEROVÁNÍ.
                # Pro zjištění zůstatku (statistik) se VŽDY na pozadí použije Admin klíč.
                "role": (["User", "Admin"], {"default": "User"}),
            }
        }
    
    # Výstupem je náš vlastní datový typ, přes který si uzly předají klíče
    RETURN_TYPES = ("FAL_AUTH",)
    RETURN_NAMES = ("fal_auth",)
    FUNCTION = "get_auth"
    CATEGORY = "fal.ai/Config"

    def get_auth(self, account, role):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "config.ini"),
            os.path.join(os.path.dirname(current_dir), "config.ini"),
        ]
        
        config = configparser.ConfigParser()
        config_found = False
        for config_path in possible_paths:
            if os.path.exists(config_path):
                config.read(config_path, encoding="utf-8")
                config_found = True
                break
                
        if not config_found or "FAL_API" not in config:
            raise RuntimeError("Nenalezen config.ini se sekcí [FAL_API]! Ujistěte se, že soubor existuje.")
            
        # Získání prefixu podle vybraného účtu
        prefix = account.upper()
        
        user_key = config["FAL_API"].get(f"{prefix}_USER", "").strip()
        admin_key = config["FAL_API"].get(f"{prefix}_ADMIN", "").strip()
        
        if not admin_key:
            print(f"⚠️ Varování: Admin klíč pro účet {account} nenalezen v config.ini!")
            
        # Zvolíme klíč pro generování na základě přepínače Role
        generation_key = admin_key if role == "Admin" else user_key
        
        if not generation_key:
            raise RuntimeError(f"API klíč pro účet {account} (Role: {role}) nebyl v config.ini nalezen!")
            
        # Uzel vrátí slovník s oběma klíči (pro generování i pro kontrolu kreditu)
        auth_data = {
            "account_name": account,
            "generation_key": generation_key,
            "admin_key": admin_key
        }
        
        return (auth_data,)

NODE_CLASS_MAPPINGS = {
    "FAL_Auth": FAL_Auth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_Auth": "fal.ai Authorization",
}