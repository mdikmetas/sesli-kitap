"""Akışlı MP3 yazıcı (lameenc, ffmpeg gerekmez) + ID3 etiketleri."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Mp3Writer:
    def __init__(self, path: str | Path, sample_rate: int, bitrate: int = 64, quality: int = 2):
        import lameenc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(self.path.suffix + ".part")
        self.sample_rate, self.bitrate = sample_rate, bitrate
        self.samples = 0
        self._f = open(self.tmp, "wb")
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)  # TTS çıktısı mono
        enc.set_quality(quality)
        self._enc = enc

    @property
    def seconds(self) -> float:
        return self.samples / self.sample_rate

    @property
    def est_bytes(self) -> int:
        return int(self.seconds * self.bitrate * 1000 / 8)

    def write(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        self.samples += len(pcm)
        self._f.write(self._enc.encode(pcm.tobytes()))

    def close(self, tags: dict | None = None) -> Path:
        self._f.write(self._enc.flush())
        self._f.close()
        self.tmp.replace(self.path)
        if tags:
            try:
                _tag(self.path, tags)
            except Exception:
                pass
        return self.path

    def abort(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        self.tmp.unlink(missing_ok=True)


def _tag(path: Path, tags: dict) -> None:
    from mutagen.id3 import ID3, TALB, TIT2, TPE1, TRCK, ID3NoHeaderError

    try:
        id3 = ID3(path)
    except ID3NoHeaderError:
        id3 = ID3()
    if tags.get("title"):
        id3.add(TIT2(encoding=3, text=tags["title"]))
    if tags.get("album"):
        id3.add(TALB(encoding=3, text=tags["album"]))
    if tags.get("artist"):
        id3.add(TPE1(encoding=3, text=tags["artist"]))
    if tags.get("track"):
        id3.add(TRCK(encoding=3, text=str(tags["track"])))
    id3.save(path)
