"""Línea base: Stable Diffusion 1.5 + ControlNet Depth (img2img).

Un mapa de profundidad (DPT/MiDaS) fija la estructura del paisaje; img2img
parte de la foto original para conservar colores y composición.
"""
from __future__ import annotations

from PIL import Image

from ..config import IMAGE_SIZE, NEGATIVE_PROMPT, SETTINGS
from .base import GenRequest, Generator

SD_RES = 512  # resolución nativa de SD 1.5


class SDControlNetGenerator(Generator):
    name = "sd15-controlnet-depth"

    def __init__(self, strength: float = 0.75, steps: int = 25, guidance: float = 7.5, control_scale: float = 0.9):
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
        from transformers import pipeline as hf_pipeline

        self.strength, self.steps, self.guidance, self.control_scale = strength, steps, guidance, control_scale
        has_cuda = torch.cuda.is_available()
        dtype = torch.float16 if has_cuda else torch.float32
        controlnet = ControlNetModel.from_pretrained(SETTINGS.controlnet_id, torch_dtype=dtype)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SETTINGS.sd_model_id, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_cuda else 0
        self.offload = has_cuda and vram_gb < 8  # SD 1.5 + ControlNet en fp16 ocupa ~4 GB: en una T4 va entero en GPU
        if self.offload:
            self.pipe.enable_model_cpu_offload()
        elif has_cuda:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.depth = hf_pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0 if has_cuda else -1)
        self._depth_cache: dict[int, Image.Image] = {}

    def depth_map(self, image: Image.Image) -> Image.Image:
        key = hash(image.tobytes()[:4096])
        if key not in self._depth_cache:
            d = self.depth(image.convert("RGB"))["depth"].convert("RGB").resize((SD_RES, SD_RES))
            self._depth_cache = {key: d}
        return self._depth_cache[key]

    def generate(self, image: Image.Image, req: GenRequest) -> Image.Image:
        import torch

        src = image.convert("RGB").resize((SD_RES, SD_RES), Image.LANCZOS)
        g = torch.Generator(device="cpu").manual_seed(req.seed)
        out = self.pipe(
            prompt=req.prompt + ", landscape photography, highly detailed",
            negative_prompt=NEGATIVE_PROMPT,
            image=src,
            control_image=self.depth_map(image),
            strength=self.strength,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            controlnet_conditioning_scale=self.control_scale,
            generator=g,
        ).images[0]
        return out.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    def params(self) -> dict:
        return {"method": self.name, "model_id": SETTINGS.sd_model_id, "controlnet": SETTINGS.controlnet_id,
                "strength": self.strength, "steps": self.steps, "guidance": self.guidance,
                "controlnet_scale": self.control_scale, "cpu_offload": self.offload}
