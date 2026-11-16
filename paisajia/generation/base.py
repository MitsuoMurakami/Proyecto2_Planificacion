from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class GenRequest:
    axis: str
    variant: str
    prompt: str
    seed: int


class Generator(ABC):
    """Interfaz común de los métodos de generación."""

    name: str = "base"

    @abstractmethod
    def generate(self, image: Image.Image, req: GenRequest) -> Image.Image:
        """Devuelve una variación de `image` (1024×1024) según `req`."""

    def params(self) -> dict:
        """Parámetros que se guardan en el registro para reproducibilidad."""
        return {"method": self.name}
