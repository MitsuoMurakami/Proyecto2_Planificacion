"""Configuración central de PaisajIA.

Todo lo que define el comportamiento del sistema (ejes, variantes, umbrales,
rutas y modelos) vive aquí para que los experimentos sean reproducibles.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Carga un archivo .env simple (CLAVE=valor  # comentario) sin dependencias extra."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = (x.strip() for x in line.split("=", 1))
            if k and v and k not in os.environ:
                os.environ[k] = v


_load_dotenv()
DATA_DIR = Path(os.getenv("PAISAJIA_DATA_DIR", ROOT / "data"))
SESSIONS_DIR = DATA_DIR / "sessions"
DB_PATH = DATA_DIR / "paisajia.sqlite"

# Número de variaciones por ronda (requisito del enunciado: exactamente cinco).
N_VARIATIONS = 5
IMAGE_SIZE = 1024

# ---------------------------------------------------------------------------
# Ejes y variantes (la primera variante de cada eje es la base de la ronda 1)
# ---------------------------------------------------------------------------
AXES: dict[str, dict] = {
    "hora": {
        "label": "Hora del día",
        "variants": {
            "atardecer": "Change the lighting to a warm golden sunset, low sun near the horizon",
            "amanecer": "Change the lighting to a soft pink and orange sunrise",
            "hora azul": "Change the lighting to blue hour, cool twilight tones just after sunset",
            "noche estrellada": "Turn the scene into a clear night with a starry sky and moonlight",
        },
    },
    "estilo": {
        "label": "Estilo artístico",
        "variants": {
            "óleo impresionista": "Render the scene as an impressionist oil painting with visible brush strokes",
            "acuarela": "Render the scene as a delicate watercolor painting on paper",
            "grabado": "Render the scene as a detailed black ink engraving with fine hatching",
            "pastel": "Render the scene as a soft pastel chalk drawing",
        },
    },
    "estacion": {
        "label": "Estación",
        "variants": {
            "invierno": "Make it winter, with snow covering the ground and trees",
            "otoño": "Make it autumn, with orange, red and yellow foliage",
            "primavera": "Make it spring, with fresh green vegetation and blooming flowers",
            "verano seco": "Make it a dry summer, with golden dry grass and harsh sunlight",
        },
    },
    "clima": {
        "label": "Clima",
        "variants": {
            "niebla": "Add a light mist and fog over the landscape",
            "lluvia ligera": "Add light rain and an overcast sky",
            "tormenta": "Add dramatic storm clouds over the landscape, no lightning damage",
            "cielo despejado": "Make the sky completely clear and bright blue",
        },
    },
    "epoca": {
        "label": "Época",
        "variants": {
            "años 70": "Make it look like a 1970s film photograph with warm faded colors and grain",
            "años 20 (B/N)": "Make it look like a 1920s black and white photograph with grain and vignette",
            "años 50": "Make it look like a 1950s Kodachrome color photograph",
            "años 90": "Make it look like a 1990s disposable camera photo with flash tones",
        },
    },
}
CUSTOM_AXIS = "personalizada"
AXIS_NAMES = list(AXES.keys())

# Sufijo añadido a todas las instrucciones: preservar la composición y no
# insertar elementos nuevos (restricción del proyecto).
PRESERVE_SUFFIX = (
    " Keep the exact same composition, terrain, horizon and camera angle. "
    "Do not add people, animals, text, signs, buildings or any new objects."
)
NEGATIVE_PROMPT = "people, person, human, face, text, letters, watermark, signature, building, logo, deformed"


def base_variant(axis: str) -> str:
    return next(iter(AXES[axis]["variants"]))


def prompt_for(axis: str, variant: str, custom_text: str | None = None) -> str:
    if axis == CUSTOM_AXIS:
        return (custom_text or "").strip() + PRESERVE_SUFFIX
    return AXES[axis]["variants"][variant] + PRESERVE_SUFFIX


# ---------------------------------------------------------------------------
# Umbrales y modelos
# ---------------------------------------------------------------------------
SCENES = ["montaña", "costa", "bosque", "desierto", "nieve"]
SCENE_PROMPTS = {
    "montaña": "a photo of a mountain landscape",
    "costa": "a photo of a coast, beach or sea",
    "bosque": "a photo of a forest",
    "desierto": "a photo of a desert",
    "nieve": "a photo of a snowy landscape",
}


@dataclass
class Thresholds:
    dino_min: float = 0.60          # O1: similitud estructural mínima
    lpips_target: float = 0.30      # O1: diversidad media por ronda
    round_seconds: float = 60.0     # O1: tiempo por ronda
    blur_laplacian_min: float = 60.0  # varianza del laplaciano en 512px
    max_retries: int = 2            # reintentos de generación por imagen
    clip_text_max: float = 0.26     # CLIP "text/people" -> rechazo
    llm_max_retries: int = 2        # reintentos del agente antes de la plantilla


@dataclass
class Settings:
    generator: str = os.getenv("PAISAJIA_GENERATOR", "mock")  # mock | flux | sd
    device: str = os.getenv("PAISAJIA_DEVICE", "auto")
    llm: str = os.getenv("PAISAJIA_LLM", "auto")  # auto | claude | ollama | simulado
    llm_model: str = os.getenv("PAISAJIA_LLM_MODEL", "claude-opus-5-5")
    ollama_model: str = os.getenv("PAISAJIA_OLLAMA_MODEL", "qwen2.5:3b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Capas de Qwen en GPU (0 = solo CPU, útil en una T4 compartida con FLUX); vacío = automático.
    ollama_num_gpu: str = os.getenv("PAISAJIA_OLLAMA_NUM_GPU", "")
    # Resolución a la que genera FLUX (la salida siempre se normaliza a IMAGE_SIZE). 768 acelera en GPUs pequeñas.
    gen_size: int = int(os.getenv("PAISAJIA_GEN_SIZE", "1024"))
    # Precisión de FLUX: auto = bf16 en GPUs Ampere o más nuevas (A100, A6000), fp16 en T4/V100.
    gen_dtype: str = os.getenv("PAISAJIA_DTYPE", "auto")
    # Cuantización del codificador de texto de FLUX (Qwen3 4B) con bitsandbytes: auto | 4bit | 8bit | off.
    # auto = 4 bits en GPUs < 20 GB (T4 de Colab): así FLUX cabe entero en la GPU sin descarga a CPU,
    # que en Colab gratis no es viable (12.7 GB de RAM).
    te_quant: str = os.getenv("PAISAJIA_TE_QUANT", "auto")
    use_vision_models: bool = os.getenv("PAISAJIA_VISION", "auto") != "off"
    flux_model_id: str = os.getenv("PAISAJIA_FLUX_ID", "black-forest-labs/FLUX.2-klein-4B")
    sd_model_id: str = os.getenv("PAISAJIA_SD_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    controlnet_id: str = os.getenv("PAISAJIA_CONTROLNET_ID", "lllyasviel/control_v11f1p_sd15_depth")
    dino_id: str = "facebook/dinov2-small"
    clip_id: str = "openai/clip-vit-base-patch32"
    thresholds: Thresholds = field(default_factory=Thresholds)


SETTINGS = Settings()

AI_LABEL = "Generado con IA"
