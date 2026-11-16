import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from paisajia.config import SETTINGS  # noqa: E402

# Las pruebas no descargan modelos: usan los proxies deterministas.
SETTINGS.use_vision_models = False


@pytest.fixture
def landscape():
    from make_demo_image import synthetic_landscape

    return synthetic_landscape(512)


@pytest.fixture
def registry(tmp_path):
    from paisajia.registry import Registry

    return Registry(tmp_path / "db.sqlite", tmp_path / "sessions")
