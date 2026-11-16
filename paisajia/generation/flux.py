"""Método principal: FLUX.2 [klein] 4B (Apache 2.0), edición por instrucción.

Cuatro pasos de inferencia (modelo destilado), ~13 GB de VRAM en bf16. Si la
GPU no alcanza, se habilita `enable_model_cpu_offload()` (descarga a CPU).
En GPUs sin bf16 acelerado (T4) usa fp16; la resolución de trabajo se fija con
PAISAJIA_GEN_SIZE (la salida siempre se normaliza a 1024×1024).

En GPUs de menos de 20 GB (T4 de Colab o de Khipu) el codificador de texto se
cuantiza a 4 bits con bitsandbytes (PAISAJIA_TE_QUANT): transformer fp16 (~8 GB)
+ codificador 4 bits (~2.5 GB) caben en los 15 GB de la T4 sin descarga a CPU.
La descarga a CPU necesita ~16 GB de RAM, más de lo que da Colab gratis.
"""
from __future__ import annotations

import logging

from PIL import Image

from ..config import IMAGE_SIZE, SETTINGS
from .base import GenRequest, Generator

log = logging.getLogger(__name__)


class FluxKleinGenerator(Generator):
    name = "flux2-klein-4b"

    def __init__(self, model_id: str | None = None, steps: int = 4, guidance: float = 1.0, offload: bool | None = None):
        import torch
        import diffusers

        self.model_id = model_id or SETTINGS.flux_model_id
        self.steps, self.guidance = steps, guidance
        # El nombre de la clase depende de la versión de diffusers; probamos en orden.
        cls = None
        for cname in ("Flux2KleinPipeline", "Flux2Pipeline"):
            cls = getattr(diffusers, cname, None)
            if cls is not None:
                break
        cls = cls or diffusers.DiffusionPipeline
        has_cuda = torch.cuda.is_available()
        self.dtype = self._pick_dtype(torch, has_cuda)
        self.size = SETTINGS.gen_size
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_cuda else 0
        self.te_quant = self._pick_te_quant(has_cuda, vram_gb)
        log.info("Cargando %s con %s (%s, %d px, codificador de texto: %s)",
                 self.model_id, cls.__name__, self.dtype, self.size, self.te_quant)
        kwargs = {"torch_dtype": getattr(torch, self.dtype)}
        if self.te_quant != "off":
            try:
                kwargs["quantization_config"] = self._quant_config(torch, self.te_quant, self.dtype)
                self.pipe = cls.from_pretrained(self.model_id, **kwargs)
            except Exception as exc:  # bitsandbytes ausente o diffusers sin PipelineQuantizationConfig
                log.warning("No se pudo cuantizar el codificador de texto (%s); se carga sin cuantizar", exc)
                kwargs.pop("quantization_config", None)
                self.te_quant = "off"
        if self.te_quant == "off":
            self.pipe = cls.from_pretrained(self.model_id, **kwargs)

        if offload is None:
            offload = has_cuda and vram_gb < 16 and self.te_quant == "off"
        if has_cuda and offload:
            self.pipe.enable_model_cpu_offload()
        elif has_cuda:
            self.pipe.to("cuda")
        self.offload = offload
        self.device = "cuda" if has_cuda else "cpu"
        self.pipe.set_progress_bar_config(disable=True)

    @staticmethod
    def _pick_te_quant(has_cuda: bool, vram_gb: float) -> str:
        choice = SETTINGS.te_quant.lower()
        if choice in ("4bit", "8bit", "off"):
            return choice if has_cuda or choice == "off" else "off"
        if not has_cuda or vram_gb >= 20:
            return "off"
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            log.warning("GPU de %.0f GB sin bitsandbytes: FLUX usará descarga a CPU (lento, ~16 GB de RAM)", vram_gb)
            return "off"
        return "4bit"

    @staticmethod
    def _quant_config(torch, mode: str, dtype: str):
        from diffusers.quantizers import PipelineQuantizationConfig

        if mode == "8bit":
            return PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", quant_kwargs={"load_in_8bit": True},
                                              components_to_quantize=["text_encoder"])
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": getattr(torch, dtype)},
            components_to_quantize=["text_encoder"],
        )

    @staticmethod
    def _pick_dtype(torch, has_cuda: bool) -> str:
        """bf16 solo donde la GPU lo acelera (capacidad ≥ 8: A100, A6000); en T4 (7.5) se usa fp16."""
        choice = SETTINGS.gen_dtype.lower()
        if choice in ("bf16", "bfloat16"):
            return "bfloat16"
        if choice in ("fp16", "float16"):
            return "float16"
        if has_cuda and torch.cuda.get_device_capability(0)[0] < 8:
            return "float16"
        return "bfloat16"

    def generate(self, image: Image.Image, req: GenRequest) -> Image.Image:
        import torch

        g = torch.Generator(device="cpu").manual_seed(req.seed)
        src = image.convert("RGB")
        if src.size != (self.size, self.size):
            src = src.resize((self.size, self.size), Image.LANCZOS)
        out = self.pipe(
            image=src,
            prompt=req.prompt,
            height=self.size,
            width=self.size,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=g,
        ).images[0]
        if out.convert("L").getextrema()[1] < 3:  # imagen negra: típico desborde (NaN) en fp16
            log.error("FLUX devolvió una imagen negra (posible NaN en %s). Prueba otra semilla o PAISAJIA_GEN_SIZE=640.",
                      self.dtype)
        return out.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    def params(self) -> dict:
        return {"method": self.name, "model_id": self.model_id, "steps": self.steps,
                "guidance": self.guidance, "cpu_offload": self.offload, "dtype": self.dtype, "gen_size": self.size,
                "text_encoder_quant": self.te_quant}
