"""Ses modeli motorları. Her motor kendi sentez parametrelerini tanımlar;
arayüz bu şemadan otomatik olarak kontrol üretir. Yeni model eklemek için
`Engine` alt sınıfı yazıp `ENGINES` sözlüğüne eklemek yeterli."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class Param:
    key: str
    label: str
    kind: str  # "float" | "int" | "bool" | "choice"
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = ()
    help: str = ""


SEED_PARAM = Param("seed", "Seed (-1 = rastgele, kitap boyunca sabit)", "int", -1, -1, 2**31 - 1, 1,
                   help="Aynı seed + aynı ayarlar = aynı ses. Rastgele seçilirse iş boyunca aynı kalır.")


class Engine:
    id = ""
    label = ""
    description = ""
    params: list[Param] = []
    sample_rates: tuple[int, ...] = (48000, 24000)

    def __init__(self):
        self.device = "cpu"

    def load(self, on_status: Callable[[str], None] = lambda m: None) -> None: ...

    def synth(self, text: str, params: dict, seed: int) -> Iterator[np.ndarray]:
        """float32 mono parçalar üretir, örnekleme hızı params['sample_rate']."""
        raise NotImplementedError

    def defaults(self) -> dict:
        d = {p.key: p.default for p in self.params}
        d["sample_rate"] = self.sample_rates[0]
        return d


def configure_cache(root: str | os.PathLike | None = None) -> Path:
    """Modelleri uygulama klasöründe tutar (internet yalnız ilk indirmede gerekir)."""
    root = Path(root or os.environ.get("KITAPSESI_MODELS_DIR") or Path.cwd() / "models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "huggingface" / "hub"))
    os.environ.setdefault("ANTALIA_MINI_CACHE", str(root / "antalia"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return root


def _device_name() -> str:
    import torch
    return f"CUDA / {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"


class EmaEngine(Engine):
    id = "ema"
    label = "EMA Lightning"
    description = "8.6M parametre, çok hızlı Türkçe TTS (canberkkkkkk/ema-lightning)."
    sample_rates = (48000, 24000, 16000, 8000)
    params = [
        Param("speed", "Hız", "float", 1.0, 0.25, 4.0, 0.05),
        SEED_PARAM,
    ]

    def load(self, on_status=lambda m: None):
        configure_cache()
        on_status("EMA Lightning yükleniyor (ilk seferde model indirilir)…")
        from ema_lightning import EMA
        self._tts = EMA()
        self.device = _device_name()

    def synth(self, text, params, seed):
        yield from self._tts.stream(text, speed=float(params["speed"]), seed=int(seed),
                                    sample_rate=int(params["sample_rate"]))


class AntaliaEngine(Engine):
    id = "antalia"
    label = "Antalia-2 Mini"
    description = "Ayarlanabilir adım/CFG/EQ’lu Türkçe TTS (cloud0day3/antalia-mini)."
    sample_rates = (48000, 24000, 16000, 8000)
    params = [
        Param("speed", "Hız", "float", 0.95, 0.25, 4.0, 0.05),
        Param("steps", "Adım sayısı", "int", 8, 1, 64, 1, help="Yüksek = daha kaliteli, daha yavaş."),
        Param("cfg", "CFG", "float", 2.0, 0.5, 10.0, 0.1),
        Param("eq", "EQ (hafif bas kısma)", "bool", True),
        SEED_PARAM,
    ]

    def load(self, on_status=lambda m: None):
        configure_cache()
        on_status("Antalia-2 Mini yükleniyor (ilk seferde model indirilir)…")
        from antalia_mini import Antalia
        self._tts = Antalia(model="cloud0day3/antalia-mini")
        self.device = _device_name()

    def synth(self, text, params, seed):
        yield from self._tts.stream(text, seed=int(seed), speed=float(params["speed"]),
                                    steps=int(params["steps"]), cfg=float(params["cfg"]),
                                    sample_rate=int(params["sample_rate"]), eq=bool(params["eq"]))


class DemoEngine(Engine):
    """Model gerektirmez: metin uzunluğuna orantılı sinüs tonu üretir. Boru hattını
    ve MP3 bölmeyi denemek / test etmek içindir."""
    id = "demo"
    label = "Deneme (model yok)"
    description = "Gerçek ses üretmez; akışı ve dosya bölmeyi hızlıca denemek içindir."
    sample_rates = (24000, 48000, 16000, 8000)
    params = [Param("speed", "Hız", "float", 1.0, 0.25, 4.0, 0.05), SEED_PARAM]

    def load(self, on_status=lambda m: None):
        self.device = "CPU"

    def synth(self, text, params, seed):
        sr = int(params["sample_rate"])
        secs = max(0.2, len(text) / 15.0 / float(params["speed"]))  # ~15 karakter/sn
        t = np.arange(int(secs * sr)) / sr
        freq = 180 + (seed % 7) * 10
        yield (0.15 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


ENGINES: dict[str, type[Engine]] = {e.id: e for e in (EmaEngine, AntaliaEngine, DemoEngine)}


def create_engine(engine_id: str) -> Engine:
    return ENGINES[engine_id]()
