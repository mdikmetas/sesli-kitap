"""Kitap -> MP3 hattı: bölümleme, sentez, parçalama, kaldığı yerden devam."""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .encoder import Mp3Writer, _tag
from .engines import Engine
from .extract import Book, read_book
from .textprep import Chunk, clean_text, make_chunks

CHARS_PER_SECOND = 15.0  # Türkçe konuşma ≈ 15 karakter/sn (hız 1.0) — yalnız tahmin için


@dataclass
class JobSettings:
    engine: str = "ema"
    params: dict = field(default_factory=dict)  # sample_rate, speed, seed, ...
    output_dir: str = "outputs"
    merge_inputs: bool = False        # tüm dosyaları tek kitap gibi birleştir
    split_mode: str = "none"          # none | chapter | minutes | size
    split_minutes: int = 60
    split_mb: int = 100
    bitrate: int = 64                 # kbps (konuşma için 48–96 yeterli)
    pdf_pages_per_section: int = 20
    chunk_chars: int = 900
    pause_sentence_ms: int = 150
    pause_paragraph_ms: int = 600
    pause_section_ms: int = 1500
    announce_titles: bool = True
    strip_headers: bool = True
    checkpoint_minutes: int = 20      # kesintiye karşı ara kayıt aralığı
    resume: bool = True
    id3_artist: str = ""


@dataclass
class Progress:
    unit: int
    units: int
    unit_name: str
    chars_done: int
    chars_total: int
    audio_seconds: float
    elapsed: float
    message: str = ""

    @property
    def fraction(self) -> float:
        return self.chars_done / self.chars_total if self.chars_total else 0.0

    @property
    def eta(self) -> float | None:
        f = self.fraction
        return self.elapsed * (1 - f) / f if f > 0.01 and self.elapsed > 3 else None


class Cancelled(Exception):
    pass


def slug(s: str, n: int = 60) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.translate(tr)).strip("_")
    return s[:n] or "kitap"


def estimate(books: list[Book], speed: float = 1.0) -> dict:
    chars = sum(b.chars for b in books)
    return {"chars": chars, "audio_hours": chars / CHARS_PER_SECOND / max(speed, 0.1) / 3600}


# --------------------------------------------------------------------------------------
class Unit:
    """Bir çıktı birimi: tek kitap veya birleştirilmiş tüm girdiler."""

    def __init__(self, name: str, books: list[Book]):
        self.name, self.books = name, books

    def sections(self, st: JobSettings) -> list[tuple[str, str]]:
        out = []
        for b in self.books:
            for s in b.sections:
                title = s.title if len(self.books) == 1 else f"{b.title}. {s.title}"
                out.append((title, clean_text(s.text, st.strip_headers)))
        return [(t, x) for t, x in out if x.strip()]


def build_units(paths: list[str], st: JobSettings) -> list[Unit]:
    books = [read_book(p, st.pdf_pages_per_section) for p in paths]
    if st.merge_inputs and len(books) > 1:
        return [Unit(slug(books[0].title) + "_birlesik", books)]
    units, used = [], set()
    for b in books:
        name = slug(b.path.stem)
        while name in used:
            name += "_2"
        used.add(name)
        units.append(Unit(name, [b]))
    return units


class JobRunner:
    def __init__(self, engine: Engine, st: JobSettings,
                 on_progress: Callable[[Progress], None] = lambda p: None,
                 on_log: Callable[[str], None] = lambda m: None):
        self.engine, self.st = engine, st
        self.on_progress, self.on_log = on_progress, on_log
        self.cancel_event = threading.Event()
        self.outputs: list[Path] = []

    def cancel(self):
        self.cancel_event.set()

    # ------------------------------------------------------------------ giriş
    def run(self, paths: list[str]) -> list[Path]:
        st = self.st
        units = build_units(paths, st)
        prepared = []
        for u in units:
            secs = u.sections(st)
            chunks = make_chunks(secs, st.chunk_chars, st.announce_titles)
            if not chunks:
                self.on_log(f"Atlandı (okunabilir metin yok): {u.name}")
                continue
            prepared.append((u, secs, chunks))
        if not prepared:
            raise ValueError("Seslendirilecek metin bulunamadı (taranmış/görsel PDF olabilir; OCR gerekir).")
        total = sum(len(c.text) for _, _, cs in prepared for c in cs)
        done_before = 0
        t0 = time.time()
        for idx, (u, secs, chunks) in enumerate(prepared, 1):
            if self.cancel_event.is_set():
                break
            try:
                self._run_unit(u, secs, chunks, idx, len(prepared), done_before, total, t0)
            except Cancelled:
                self.on_log("Durduruldu. Aynı ayarlarla tekrar başlatırsanız kaldığı yerden devam eder.")
                break
            done_before += sum(len(c.text) for c in chunks)
        return self.outputs

    # ------------------------------------------------------------------ birim
    def _fingerprint(self, u: Unit, chunks: list[Chunk]) -> str:
        st = self.st
        keys = asdict(st)
        for k in ("output_dir", "resume", "checkpoint_minutes", "id3_artist"):
            keys.pop(k)
        files = [(str(b.path), b.path.stat().st_size, int(b.path.stat().st_mtime)) for b in u.books]
        raw = json.dumps([keys, files, len(chunks), sum(len(c.text) for c in chunks)], sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def _run_unit(self, u: Unit, secs, chunks: list[Chunk], idx: int, n_units: int,
                  done_before: int, total: int, t0: float):
        st = self.st
        out_dir = Path(st.output_dir)
        work = out_dir / ".kitapsesi" / u.name
        work.mkdir(parents=True, exist_ok=True)
        state_path = work / "state.json"
        fp = self._fingerprint(u, chunks)

        state = None
        if st.resume and state_path.exists():
            try:
                s = json.loads(state_path.read_text(encoding="utf-8"))
                if s.get("fingerprint") == fp:
                    state = s
            except (OSError, ValueError):
                pass
        if state is None:
            for f in work.glob("*"):
                f.unlink()
            seed = int(st.params.get("seed", -1))
            state = {"fingerprint": fp, "seed": seed if seed >= 0 else random.randrange(2**31),
                     "next_chunk": 0, "parts": [], "segs": [], "done": False}
        if state["done"]:
            self.on_log(f"Zaten tamamlanmış, atlandı: {u.name}")
            self.outputs += [Path(p) for p in state["parts"]]
            return
        if state["next_chunk"]:
            self.on_log(f"{u.name}: parça {state['next_chunk']}/{len(chunks)} noktasından devam ediliyor.")

        sr = int(st.params["sample_rate"])
        pauses = {k: np.zeros(int(sr * ms / 1000), dtype=np.float32) for k, ms in
                  (("s", st.pause_sentence_ms), ("p", st.pause_paragraph_ms), ("c", st.pause_section_ms))}
        chars_done_unit = sum(len(c.text) for c in chunks[:state["next_chunk"]])
        part_secs = sum(sg["seconds"] for sg in state["segs"])
        part_bytes = sum(sg["bytes"] for sg in state["segs"])
        finished_secs = state.get("finished_secs", 0.0)
        part_first_title = state.get("part_title") or ""
        pending_split = False
        writer: Mp3Writer | None = None

        def save():
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

        def close_segment(next_chunk: int):
            """Açık ara kaydı kapatır; next_chunk'a kadar olan her şey tamam kabul edilir."""
            nonlocal writer, part_secs, part_bytes
            if writer is None:
                return
            w, writer = writer, None
            path = w.close()
            size = path.stat().st_size
            part_secs += w.seconds
            part_bytes += size
            state["segs"].append({"file": path.name, "seconds": w.seconds, "bytes": size})
            state["next_chunk"] = next_chunk
            state["part_title"] = part_first_title
            save()

        def finalize_part(final: bool):
            nonlocal part_secs, part_bytes, part_first_title, pending_split, finished_secs
            if not state["segs"]:
                return
            n = len(state["parts"]) + 1
            multi = (not final) or n > 1
            if multi and st.split_mode == "chapter":
                fname = f"{u.name}_{n:02d}_{slug(part_first_title, 40)}.mp3"
            else:
                fname = f"{u.name}_{n:02d}.mp3"
            dest = (out_dir / u.name / fname) if multi else out_dir / f"{u.name}.mp3"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".mp3.tmp")
            with open(tmp, "wb") as o:  # MP3 kareleri doğrudan birleştirilebilir
                for sg in state["segs"]:
                    o.write((work / sg["file"]).read_bytes())
            tmp.replace(dest)
            try:
                _tag(dest, {"title": part_first_title or u.name, "album": u.books[0].title,
                            "artist": st.id3_artist, "track": n})
            except Exception:
                pass
            for sg in state["segs"]:
                (work / sg["file"]).unlink(missing_ok=True)
            finished_secs += part_secs
            state["finished_secs"] = finished_secs
            state["parts"].append(str(dest))
            state["segs"], part_secs, part_bytes, part_first_title = [], 0.0, 0, ""
            state["part_title"] = ""
            pending_split = False
            save()
            self.on_log(f"Yazıldı: {dest}")

        def progress(msg=""):
            audio = finished_secs + part_secs + (writer.seconds if writer else 0.0)
            self.on_progress(Progress(idx, n_units, u.name, done_before + chars_done_unit, total,
                                      audio, time.time() - t0, msg))

        try:
            progress("Başlıyor")
            for i in range(state["next_chunk"], len(chunks)):
                ch = chunks[i]
                if self.cancel_event.is_set():  # yalnızca parçalar arasında: yarım parça asla kaydedilmez
                    close_segment(i)
                    raise Cancelled()

                if (pending_split or (st.split_mode == "chapter" and ch.section_start)) and (state["segs"] or writer):
                    close_segment(i)
                    finalize_part(final=False)
                if writer is None:
                    k = len(state["segs"]) + 1
                    writer = Mp3Writer(work / f"seg_{len(state['parts']) + 1:03d}_{k:03d}.mp3", sr, st.bitrate)
                if not part_first_title:
                    part_first_title = secs[ch.section][0]

                for audio in self.engine.synth(ch.text, st.params, state["seed"]):
                    writer.write(audio)
                nxt = chunks[i + 1] if i + 1 < len(chunks) else None
                kind = "c" if (nxt and nxt.section_start) else "p" if ch.paragraph_end else "s"
                writer.write(pauses[kind])
                chars_done_unit += len(ch.text)

                if ch.paragraph_end:
                    if st.split_mode == "minutes" and part_secs + writer.seconds >= st.split_minutes * 60:
                        pending_split = True
                    elif st.split_mode == "size" and part_bytes + writer.est_bytes >= st.split_mb * 1024 * 1024:
                        pending_split = True
                if writer.seconds >= st.checkpoint_minutes * 60:
                    close_segment(i + 1)
                progress()

            close_segment(len(chunks))
            finalize_part(final=True)
            state["done"] = True
            save()
            self._write_playlist(u, state)
            self.outputs += [Path(p) for p in state["parts"]]
            progress("Tamamlandı")
        except Cancelled:
            raise
        except BaseException:
            if writer is not None:  # hata: yarım ara kaydı at, öncekiler durur
                writer.abort()
            raise

    def _write_playlist(self, u: Unit, state: dict):
        parts = [Path(p) for p in state["parts"]]
        if len(parts) > 1:
            base = Path(self.st.output_dir)
            lines = ["#EXTM3U"] + [str(p.relative_to(base)).replace("\\", "/") for p in parts]
            (base / f"{u.name}.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")
