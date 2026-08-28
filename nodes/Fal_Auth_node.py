"""
Modul pro správu autorizace a API klíčů pro fal.ai v ComfyUI.

Implementuje vzor In-Memory Vault:
- API klíče jsou bezpečně uloženy pouze v paměti procesu (FalAuthVault).
- Mezi uzly se předává pouze tokenizovaný objekt (FalAuthToken), který neobsahuje otevřený text klíčů.
- Zabraňuje úniku API klíčů při exportu workflow do JSONu, logování nebo bypassu uzlu.
"""

import configparser
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_config_path() -> Path | None:
    """
    Vyhledá cestu ke konfiguračnímu souboru `config.ini`.

    Prohledává aktuální složku uzlů i kořenovou složku balíčku.

    Returns:
        Path | None: Cesta k existujícímu souboru config.ini, nebo None pokud nebyl nalezen.
    """
    current_dir = Path(__file__).resolve().parent
    search_paths = [
        current_dir / "config.ini",
        current_dir.parent / "config.ini",
    ]
    for path in search_paths:
        if path.is_file():
            return path
    return None


def _load_available_accounts() -> List[str]:
    """
    Dynamicky detekuje dostupné účty z config.ini, nebo vrátí výchozí seznam.

    Returns:
        List[str]: Seznam názvů účtů pro výběr v UI.
    """
    config_path = _find_config_path()
    if config_path:
        config = configparser.ConfigParser()
        try:
            config.read(config_path, encoding="utf-8")
            if "FAL_API" in config:
                accounts = set()
                for key in config["FAL_API"]:
                    # Hledáme klíče ve tvaru <ACCOUNT>_USER nebo <ACCOUNT>_ADMIN
                    parts = key.upper().split("_")
                    if len(parts) >= 2 and parts[-1] in ("USER", "ADMIN"):
                        account_name = "_".join(parts[:-1]).capitalize()
                        accounts.add(account_name)
                if accounts:
                    # Seřadíme účty s preferencí pro 'Chroust'
                    sorted_accs = sorted(list(accounts))
                    if "Chroust" in sorted_accs:
                        sorted_accs.remove("Chroust")
                        sorted_accs.insert(0, "Chroust")
                    return sorted_accs
        except Exception:
            pass
    return ["Chroust", "Seznam"]


class FalAuthVault:
    """
    In-Memory Vault (bezpečné úložiště API klíčů v operační paměti procesu).

    Zajišťuje, že citlivé API klíče:
    - Nikdy necestují jako otevřený text po propojovacích drátech (edges) v ComfyUI grafu.
    - Neukládají se do workflow JSONů ani do prompt metadat obrázků.
    - Jsou asociovány s jedinečným náhodným tokenem v paměti Pythonu.
    - Jsou vyzvednuty až cílovým uzlem těsně před provedením HTTP požadavku.
    """

    _vault: Dict[str, Dict[str, str]] = {}

    @classmethod
    def register(cls, account_name: str, generation_key: str, admin_key: str) -> str:
        """
        Zaregistruje novou sadu API klíčů do privátního trezoru a vygeneruje bezpečný token.

        Args:
            account_name: Název účtu (např. "Chroust").
            generation_key: API klíč pro generování.
            admin_key: Administrátorský klíč pro dotazy na zůstatek kreditu.

        Returns:
            str: Bezpečný jedinečný token (např. "fal_vault_a1b2c3d4...").
        """
        token = f"fal_vault_{secrets.token_hex(16)}"
        cls._vault[token] = {
            "account_name": account_name,
            "generation_key": generation_key,
            "admin_key": admin_key,
        }
        return token

    @classmethod
    def get_credentials(cls, auth_payload: Any) -> Dict[str, str]:
        """
        Vyzvedne přihlašovací údaje z trezoru na základě předaného tokenu nebo objektu.

        Poskytuje plnou zpětnou kompatibilitu jak pro nový tokenový systém,
        tak pro starší přímé slovníky.

        Args:
            auth_payload: Token (str), FalAuthToken nebo běžný dict.

        Returns:
            Dict[str, str]: Slovník s klíči:
                - `account_name` (str)
                - `generation_key` (str)
                - `admin_key` (str)
        """
        if not auth_payload:
            return {"account_name": "Neznámý", "generation_key": "", "admin_key": ""}

        # 1. Přímý string token
        if isinstance(auth_payload, str) and auth_payload in cls._vault:
            return cls._vault[auth_payload]

        # 2. Slovník / FalAuthToken obsahující 'auth_token'
        if isinstance(auth_payload, dict):
            token = auth_payload.get("auth_token")
            if token and token in cls._vault:
                return cls._vault[token]

            # 3. Zpětná kompatibilita pro staré uzly / slovníky s přímými klíči
            if "generation_key" in auth_payload or "admin_key" in auth_payload:
                return {
                    "account_name": auth_payload.get("account_name", "Neznámý"),
                    "generation_key": auth_payload.get("generation_key", ""),
                    "admin_key": auth_payload.get("admin_key", ""),
                }

        return {"account_name": "Neznámý", "generation_key": "", "admin_key": ""}

    @classmethod
    def clear(cls) -> None:
        """Vymaže obsah paměťového trezoru."""
        cls._vault.clear()


class FalAuthToken(dict):
    """
    Tokenový objekt předávaný v ComfyUI grafu.

    Dědí z `dict`, ale v samotném těle slovníku obsahuje POUZE bezpečný token a název účtu,
    nikoliv samotné API klíče.

    Bezpečnostní funkce:
    - `str()` a `repr()` vracejí prázdný řetězec `""` (ochrana při bypassu i logování).
    - Metoda `.get(key)` dynamicky dotazuje `FalAuthVault`, takže i neaktualizované uzly
      volající `fal_auth.get("generation_key")` budou bez problémů fungovat.
    - Serializace do JSON/workflow neuloží žádný API klíč.
    """

    def __init__(self, auth_token: str, account_name: str):
        super().__init__({
            "auth_token": auth_token,
            "account_name": account_name,
            "empty_string": "",
        })
        self._auth_token = auth_token

    def get(self, key: str, default: Any = None) -> Any:
        """
        Vrací hodnotu klíče. Pro citlivé klíče dotazuje in-memory vault.
        """
        if key in ("generation_key", "admin_key", "account_name"):
            creds = FalAuthVault.get_credentials(self._auth_token)
            return creds.get(key, default)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key in ("generation_key", "admin_key", "account_name"):
            creds = FalAuthVault.get_credentials(self._auth_token)
            if key in creds and creds[key]:
                return creds[key]
        return super().__getitem__(key)

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return ""


# Alias pro zpětnou kompatibilitu
FalAuth = FalAuthToken


class FAL_Auth:
    """
    ComfyUI uzel pro autorizaci služeb fal.ai s In-Memory Vaultem.

    Vlastnosti a chování:
    - Načítá uživatelské (_USER) i administrátorské (_ADMIN) API klíče z `config.ini`.
    - Uloží klíče do bezpečného paměťového trezoru `FalAuthVault`.
    - Do grafu ComfyUI odešle pouze tokenizovaný objekt `FalAuthToken`.
    - Podporuje přepínání rolí (User / Admin).
    - Výstup se při převodu na řetězec (str/repr/bypass) chová jako prázdný string `""`.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Definuje vstupní parametry uzlu pro ComfyUI rozhraní.

        Returns:
            Dict[str, Any]: Slovník s definicí povinných vstupů:
                - `account`: Výběr účtu (např. Chroust, Seznam).
                - `role`: Role pro generování (User / Admin).
        """
        accounts = _load_available_accounts()
        default_account = "Chroust" if "Chroust" in accounts else accounts[0]

        return {
            "required": {
                "account": (accounts, {"default": default_account}),
                # Přepínač Role určí, který klíč se použije pro SAMOTNÉ GENEROVÁNÍ.
                # Pro zjištění zůstatku (statistik) se VŽDY na pozadí použije Admin klíč.
                "role": (["User", "Admin"], {"default": "User"}),
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("FAL_AUTH",)
    RETURN_NAMES: Tuple[str, ...] = ("fal_auth",)
    FUNCTION: str = "get_auth"
    CATEGORY: str = "fal.ai/Config"

    def get_auth(self, account: str, role: str) -> Tuple[FalAuthToken]:
        """
        Načte a ověří API klíče, zaregistruje je do trezoru a vrátí bezpečný token.

        Args:
            account: Název zvoleného účtu (např. "Chroust").
            role: Zvolená role pro generování ("User" nebo "Admin").

        Returns:
            Tuple[FalAuthToken]: Jednoprvková n-tice s instancí `FalAuthToken`.

        Raises:
            RuntimeError: Pokud chybí konfigurační soubor, sekce [FAL_API] nebo potřebný API klíč.
        """
        config_path = _find_config_path()
        if not config_path:
            raise RuntimeError(
                "Nenalezen soubor config.ini! Ujistěte se, že existuje v kořenovém adresáři balíčku."
            )

        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")

        if "FAL_API" not in config:
            raise RuntimeError(
                f"Sekce [FAL_API] nebyla v {config_path.name} nalezena! "
                "Zkontrolujte formát konfiguračního souboru."
            )

        # Prefix pro vyhledání klíčů (např. CHROUST_USER, CHROUST_ADMIN)
        prefix = account.strip().upper()
        api_section = config["FAL_API"]

        user_key = api_section.get(f"{prefix}_USER", "").strip()
        admin_key = api_section.get(f"{prefix}_ADMIN", "").strip()

        if not admin_key:
            print(f"⚠️ Varování: Admin klíč pro účet '{account}' nenalezen v {config_path.name}!")

        # Volba klíče pro generování podle zvolené role
        generation_key = admin_key if role == "Admin" else user_key

        if not generation_key:
            raise RuntimeError(
                f"API klíč pro účet '{account}' (Role: {role}) nebyl v config.ini nalezen! "
                f"Ověřte existenci položky '{prefix}_{role.upper()}' v sekci [FAL_API]."
            )

        # Uložíme klíče do privátního in-memory vaultu a získáme bezpečný token
        auth_token = FalAuthVault.register(
            account_name=account,
            generation_key=generation_key,
            admin_key=admin_key,
        )

        # Výstupem je tokenizovaný kontejner (žádné otevřené klíče v grafu)
        return (FalAuthToken(auth_token=auth_token, account_name=account),)


NODE_CLASS_MAPPINGS = {
    "FAL_Auth": FAL_Auth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_Auth": "fal.ai Authorization",
}


NODE_CLASS_MAPPINGS = {
    "FAL_Auth": FAL_Auth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAL_Auth": "fal.ai Authorization",
}