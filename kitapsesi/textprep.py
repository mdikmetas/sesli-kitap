"""Metin temizleme ve seslendirme parçalarına bölme."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_SENT = re.compile(r"(?<=[.!?…])\s+(?=[\"'“‘(]?[A-ZÇĞİÖŞÜ0-9])")


def clean_text(text: str, strip_headers: bool = True) -> str:
    """PDF/EPUB kalıntılarını temizler: satır sonu tireleri, sayfa numaraları,
    tekrarlayan üst/alt bilgiler, tek satırlık kırılmalar."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "").replace("​", "")
    text = text.replace(" ", " ").replace("﻿", "")

    lines = [l.strip() for l in text.split("\n")]
    if strip_headers:
        # Çok sayfada tekrar eden kısa satırlar (üst/alt bilgi) ve yalnız sayfa numaraları.
        counts = Counter(l for l in lines if 3 <= len(l) <= 80)
        repeated = {l for l, c in counts.items() if c >= 5 and not l.endswith((".", "!", "?"))}
        lines = [l for l in lines
                 if l not in repeated and not re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", l)
                 and not re.fullmatch(r"(?i)(sayfa|page)\s*\d+(\s*/\s*\d+)?", l)]

    # Paragrafları satır boşluklarından kur, aradaki tek satır sonlarını boşluğa çevir.
    paras, buf = [], []
    for l in lines:
        if not l:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            continue
        if buf and buf[-1].endswith("-") and l[:1].islower():
            buf[-1] = buf[-1][:-1] + l  # satır sonu tire birleştirme
        else:
            buf.append(l)
    if buf:
        paras.append(" ".join(buf))
    out = "\n\n".join(re.sub(r"[ \t]+", " ", p).strip() for p in paras if p.strip())
    return out


def split_sentences(par: str) -> list[str]:
    return [s.strip() for s in _SENT.split(par) if s.strip()]


@dataclass
class Chunk:
    text: str
    section: int
    paragraph_end: bool  # paragraf sonunda daha uzun duraklama
    section_start: bool = False


def make_chunks(sections: list[tuple[str, str]], max_chars: int = 900,
                announce_titles: bool = True) -> list[Chunk]:
    """Bölümleri en çok `max_chars` uzunluğunda, cümle sınırlarına saygılı parçalara böler."""
    chunks: list[Chunk] = []
    for si, (title, body) in enumerate(sections):
        first = len(chunks)
        if announce_titles and title:
            t = title.strip()
            chunks.append(Chunk(t if t[-1:] in ".!?:" else t + ".", si, True))
        for par in body.split("\n\n"):
            par = par.strip()
            if not par:
                continue
            cur = ""
            for sent in split_sentences(par):
                while len(sent) > max_chars:  # aşırı uzun cümle: virgülden/boşluktan kes
                    cut = max(sent.rfind(", ", 0, max_chars), sent.rfind(" ", 0, max_chars))
                    cut = cut if cut > max_chars // 3 else max_chars
                    piece, sent = sent[:cut + 1].strip(), sent[cut + 1:].strip()
                    if cur:
                        chunks.append(Chunk(cur, si, False)); cur = ""
                    chunks.append(Chunk(piece, si, False))
                if cur and len(cur) + 1 + len(sent) > max_chars:
                    chunks.append(Chunk(cur, si, False)); cur = sent
                else:
                    cur = f"{cur} {sent}" if cur else sent
            if cur:
                chunks.append(Chunk(cur, si, False))
            if chunks and chunks[-1].section == si:
                chunks[-1].paragraph_end = True
        if len(chunks) > first:
            chunks[first].section_start = True
    return chunks
