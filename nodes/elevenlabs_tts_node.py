import json
import os
import requests
from fal_client.client import FalClientHTTPError

import fal_client
from .Fal_Auth_node import FalAuthVault
from .fal_utils import FalConfig, ResultProcessor, extract_fal_error_message

# Seznam ověřených vestavěných hlasů ElevenLabs podporovaných na fal.ai
ELEVENLABS_VOICES = [
    "Rachel",
    "Aria",
    "Sarah",
    "Laura",
    "Charlie",
    "George",
    "Callum",
    "Liam",
    "Charlotte",
    "Alice",
    "Matilda",
    "Will",
    "Jessica",
    "Eric",
    "Chris",
    "Brian",
    "Daniel",
    "Lily",
    "Bill",
    "Adam",
    "River",
    "Roger",
    "custom",
]

# Podporované jazyky (ISO 639-1) pro Multilingual v2
LANGUAGE_CODES = [
    "auto",
    "en",
    "cs",
    "sk",
    "de",
    "es",
    "fr",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "ar",
    "zh",
    "ja",
    "ko",
    "hi",
    "sv",
    "id",
    "fil",
    "uk",
    "el",
    "ro",
    "hu",
    "da",
    "fi",
    "vi",
    "ms",
    "hr",
    "bg",
    "ta",
]


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


def _build_usage_stats(fal_auth, result, voice_used, text_len):
    """Sestaví textový výstup se statistikami generování."""
    account_name = "Neznámý"
    admin_key = None
    if fal_auth:
        creds = FalAuthVault.get_credentials(fal_auth)
        account_name = creds.get("account_name", "Neznámý")
        admin_key = creds.get("admin_key")

    timings = (result.get("timings") if isinstance(result, dict) else None) or {}
    inference_time = timings.get("inference")
    time_str = f"{inference_time:.2f} s" if isinstance(inference_time, (int, float)) else "N/A"

    audio_obj = result.get("audio", {}) if isinstance(result, dict) else {}
    audio_url = audio_obj.get("url", "N/A") if isinstance(audio_obj, dict) else str(audio_obj)

    balance, currency = _fetch_balance(admin_key)

    return (
        f"📊 Statistiky generování (Účet: {account_name}):\n"
        f"• Model: fal-ai/elevenlabs/tts/multilingual-v2\n"
        f"• Hlas: {voice_used}\n"
        f"• Délka textu: {text_len} znaků\n"
        f"• Doba generování: {time_str}\n"
        f"• Audio URL: {audio_url}\n"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class FAL_ElevenLabsTTSMultilingualV2:
    """Text-to-Speech generation using fal.ai's ElevenLabs Multilingual v2 model.

    Supports 29 languages with high quality, natural-sounding voices,
    custom voice IDs, word timestamps, and voice style/speed adjustments.
    """

    CATEGORY = "fal.ai/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "voice": (ELEVENLABS_VOICES, {"default": "Rachel"}),
                "custom_voice": ("STRING", {"default": "", "multiline": False}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "similarity_boost": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.7, "max": 1.2, "step": 0.01}),
                "language_code": (LANGUAGE_CODES, {"default": "auto"}),
                "apply_text_normalization": (["auto", "on", "off"], {"default": "auto"}),
                "timestamps": ("BOOLEAN", {"default": False}),
                "previous_text": ("STRING", {"default": "", "multiline": True}),
                "next_text": ("STRING", {"default": "", "multiline": True}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_url", "timestamps", "usage_stats")
    FUNCTION = "generate_speech"

    def generate_speech(
        self,
        text: str,
        fal_auth=None,
        voice: str = "Rachel",
        custom_voice: str = "",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        speed: float = 1.0,
        language_code: str = "auto",
        apply_text_normalization: str = "auto",
        timestamps: bool = False,
        previous_text: str = "",
        next_text: str = "",
        private_mode: bool = False,
    ):
        if not text or not text.strip():
            raise ValueError("Vstupní text pro převod na řeč nesmí být prázdný.")

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

        # 2. Výběr hlasu (custom_voice má přednost)
        selected_voice = custom_voice.strip() if custom_voice and custom_voice.strip() else voice
        if selected_voice == "custom":
            raise ValueError(
                "Byla vybrána možnost 'custom', ale pole 'custom_voice' je prázdné. "
                "Zadejte název nebo ID hlasu do 'custom_voice'."
            )

        # 3. Příprava argumentů pro fal-ai/elevenlabs/tts/multilingual-v2
        arguments = {
            "text": text.strip(),
            "voice": selected_voice,
            "stability": stability,
            "similarity_boost": similarity_boost,
            "speed": speed,
            "apply_text_normalization": apply_text_normalization,
            "timestamps": timestamps,
        }

        if style > 0.0:
            arguments["style"] = style

        if language_code and language_code != "auto":
            arguments["language_code"] = language_code

        if previous_text and previous_text.strip():
            arguments["previous_text"] = previous_text.strip()

        if next_text and next_text.strip():
            arguments["next_text"] = next_text.strip()

        # 4. Hlavičky (private_mode)
        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        # 5. Volání API
        try:
            result = fal_client.subscribe(
                "fal-ai/elevenlabs/tts/multilingual-v2",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 6. Zpracování výstupu
        audio_output = ResultProcessor.process_audio_result(result)

        audio_obj = result.get("audio", {}) if isinstance(result, dict) else {}
        audio_url = audio_obj.get("url", "") if isinstance(audio_obj, dict) else str(audio_obj)

        timestamps_res = result.get("timestamps")
        timestamps_str = ""
        if timestamps_res is not None:
            timestamps_str = json.dumps(timestamps_res, indent=2, ensure_ascii=False)

        usage_stats = _build_usage_stats(fal_auth, result, selected_voice, len(text.strip()))

        return (audio_output, audio_url, timestamps_str, usage_stats)


def _parse_dialogue_inputs(
    text: str,
    voice_aliases_str: str = "",
    default_v1: str = "George",
    default_v2: str = "Rachel",
    valid_voices: list = None,
):
    """Parsuje vstupní text dialogu do struktury očekávané modelem eleven-v3.

    Podporuje:
    - JSON formát: [{"voice": "George", "text": "..."}, ...]
    - Skriptový formát: 'George: Ahoj!\nRachel: Čau!'
    - Formát se závorkami: '[George]: Ahoj!\n[Rachel]: Čau!'
    - Přemapování mluvčích přes voice_aliases (např. Host=George, Guest=Rachel, Narrator=CwhRBWXzGAHq8TQ4Fs17)
    - Střídavé přiřazení pro řádky bez explicitního mluvčího
    """
    import re

    text = text.strip()
    if not text:
        raise ValueError("Vstupní text dialogu nesmí být prázdný.")

    # 1. Zkouška JSON pole
    if text.startswith("[") and text.endswith("]"):
        try:
            items = json.loads(text)
            if (
                isinstance(items, list)
                and len(items) > 0
                and all(
                    isinstance(x, dict) and "text" in x and "voice" in x for x in items
                )
            ):
                return [
                    {"text": str(x["text"]).strip(), "voice": str(x["voice"]).strip()}
                    for x in items
                ]
        except Exception:
            pass

    # 2. Parsování aliasů hlasů
    aliases = {}
    if voice_aliases_str:
        for line in voice_aliases_str.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                aliases[k.strip().lower()] = v.strip()
            elif ":" in line:
                k, v = line.split(":", 1)
                aliases[k.strip().lower()] = v.strip()

    # 3. Parsování řádek po řádku
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    dialogue_blocks = []
    current_default_voice = default_v1

    for line in lines:
        if line.startswith("#") or line.startswith("//"):
            continue

        matched_voice = None
        dialogue_content = None

        # Hledáme vzor Speaker: Text nebo [Speaker]: Text
        colon_match = re.match(r"^\[?([a-zA-Z0-9_\-\s]+?)\]?:\s*(.+)$", line)
        if colon_match:
            speaker_candidate = colon_match.group(1).strip()
            msg = colon_match.group(2).strip()
            spk_lower = speaker_candidate.lower()

            if spk_lower in aliases:
                matched_voice = aliases[spk_lower]
                dialogue_content = msg
            elif valid_voices and any(v.lower() == spk_lower for v in valid_voices):
                matched_voice = next(v for v in valid_voices if v.lower() == spk_lower)
                dialogue_content = msg
            elif spk_lower in [
                "speaker 1",
                "speaker1",
                "speaker_1",
                "1",
                "person 1",
                "host",
                "moderator",
                "male",
            ]:
                matched_voice = aliases.get(spk_lower, default_v1)
                dialogue_content = msg
            elif spk_lower in [
                "speaker 2",
                "speaker2",
                "speaker_2",
                "2",
                "person 2",
                "guest",
                "female",
            ]:
                matched_voice = aliases.get(spk_lower, default_v2)
                dialogue_content = msg
            else:
                matched_voice = speaker_candidate
                dialogue_content = msg

        if matched_voice and dialogue_content:
            dialogue_blocks.append({"voice": matched_voice, "text": dialogue_content})
            current_default_voice = (
                default_v2 if matched_voice == default_v1 else default_v1
            )
        else:
            # Řádek bez explicitního mluvčího použije střídavý výchozí hlas
            dialogue_blocks.append({"voice": current_default_voice, "text": line})
            current_default_voice = (
                default_v2 if current_default_voice == default_v1 else default_v1
            )

    if not dialogue_blocks:
        raise ValueError("Nepodařilo se extrahovat žádné repliky dialogu.")

    return dialogue_blocks


def _build_dialogue_usage_stats(fal_auth, result, dialogue_blocks, seed_used):
    """Sestaví textový výstup se statistikami generování dialogu."""
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

    audio_obj = result.get("audio", {}) if isinstance(result, dict) else {}
    audio_url = (
        audio_obj.get("url", "N/A") if isinstance(audio_obj, dict) else str(audio_obj)
    )

    total_chars = sum(len(b.get("text", "")) for b in dialogue_blocks)
    speakers_used = list(dict.fromkeys(b.get("voice", "") for b in dialogue_blocks))

    balance, currency = _fetch_balance(admin_key)

    return (
        f"📊 Statistiky generování dialogu (Účet: {account_name}):\n"
        f"• Model: fal-ai/elevenlabs/text-to-dialogue/eleven-v3\n"
        f"• Počet replik: {len(dialogue_blocks)}\n"
        f"• Použité hlasy: {', '.join(speakers_used)}\n"
        f"• Celkem znaků: {total_chars}\n"
        f"• Použitý seed: {seed_used}\n"
        f"• Doba generování: {time_str}\n"
        f"• Audio URL: {audio_url}\n"
        f"• Zbývá kreditu: {balance} {currency}"
    )


class FAL_ElevenLabsDialogueV3:
    """Multi-speaker dialogue text-to-speech generation using fal.ai ElevenLabs Text to Dialogue (eleven-v3).

    Supports script format (e.g. 'George: Hello\nRachel: Hi!'), emotion cues
    ([whispering], [laughter], [applause], [strong canadian accent]),
    custom voice mappings, speaker boosting, and seed control.
    """

    CATEGORY = "fal.ai/Audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_text": (
                    "STRING",
                    {
                        "default": "George: [applause] Vítejte u dnešního rozhovoru!\nRachel: [excited] Děkuji za pozvání, těšila jsem se sem.",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "fal_auth": ("FAL_AUTH",),
                "speaker_1_voice": (ELEVENLABS_VOICES, {"default": "George"}),
                "speaker_2_voice": (ELEVENLABS_VOICES, {"default": "Rachel"}),
                "custom_voice_aliases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "SpeakerName = VoiceName or VoiceID (one per line)",
                    },
                ),
                "stability": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "use_speaker_boost": ("BOOLEAN", {"default": True}),
                "language_code": (LANGUAGE_CODES, {"default": "auto"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
                "private_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "INT", "STRING")
    RETURN_NAMES = ("audio", "audio_url", "seed", "usage_stats")
    FUNCTION = "generate_dialogue"

    def generate_dialogue(
        self,
        dialogue_text: str,
        fal_auth=None,
        speaker_1_voice: str = "George",
        speaker_2_voice: str = "Rachel",
        custom_voice_aliases: str = "",
        stability: float = 0.5,
        use_speaker_boost: bool = True,
        language_code: str = "auto",
        seed: int = -1,
        private_mode: bool = False,
    ):
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

        # 2. Parsování dialogových bloků
        dialogue_blocks = _parse_dialogue_inputs(
            dialogue_text,
            voice_aliases_str=custom_voice_aliases,
            default_v1=speaker_1_voice,
            default_v2=speaker_2_voice,
            valid_voices=ELEVENLABS_VOICES,
        )

        # 3. Příprava argumentů pro fal-ai/elevenlabs/text-to-dialogue/eleven-v3
        arguments = {
            "inputs": dialogue_blocks,
            "stability": stability,
            "use_speaker_boost": use_speaker_boost,
        }

        if seed is not None and seed >= 0:
            arguments["seed"] = seed

        if language_code and language_code != "auto":
            arguments["language_code"] = language_code

        # 4. Hlavičky (private_mode)
        headers = {}
        if private_mode:
            headers["X-Fal-Store-IO"] = "0"
            arguments["sync_mode"] = True

        # 5. Volání API
        try:
            result = fal_client.subscribe(
                "fal-ai/elevenlabs/text-to-dialogue/eleven-v3",
                arguments=arguments,
                headers=headers,
            )
        except FalClientHTTPError as e:
            msg = extract_fal_error_message(e)
            raise RuntimeError(f"fal.ai error: {msg}") from None

        # 6. Zpracování výstupu
        audio_output = ResultProcessor.process_audio_result(result)

        audio_obj = result.get("audio", {}) if isinstance(result, dict) else {}
        audio_url = (
            audio_obj.get("url", "") if isinstance(audio_obj, dict) else str(audio_obj)
        )
        seed_out = int(result.get("seed", seed if seed >= 0 else 0))

        usage_stats = _build_dialogue_usage_stats(
            fal_auth, result, dialogue_blocks, seed_out
        )

        return (audio_output, audio_url, seed_out, usage_stats)


NODE_CLASS_MAPPINGS = {
    "FAL_ElevenLabsTTSMultilingualV2": FAL_ElevenLabsTTSMultilingualV2,
    "FAL_ElevenLabsDialogueV3": FAL_ElevenLabsDialogueV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_ElevenLabsTTSMultilingualV2": "ElevenLabs TTS Multilingual v2 (fal.ai)",
    "FAL_ElevenLabsDialogueV3": "ElevenLabs Text to Dialogue Eleven v3 (fal.ai)",
}
