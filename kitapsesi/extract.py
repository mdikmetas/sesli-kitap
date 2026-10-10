"""Belge okuyucular: EPUB, PDF, TXT/MD, HTML -> bölümler (başlık + metin)."""
from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

SUPPORTED = {".epub", ".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass
class Section:
    title: str
    text: str
    page_start: int | None = None  # PDF için


@dataclass
class Book:
    path: Path
    title: str
    sections: list[Section]

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


# ----------------------------------------------------------------------------- HTML
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
          "blockquote", "pre", "dt", "dd", "table", "ul", "ol"}
_SKIP = {"script", "style", "head", "nav", "svg", "math"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.heading: str | None = None
        self._in_heading = False
        self._hbuf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")
        if tag in ("h1", "h2", "h3") and self.heading is None:
            self._in_heading, self._hbuf = True, []

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        elif tag in _BLOCK:
            self.parts.append("\n\n")
        if tag in ("h1", "h2", "h3") and self._in_heading:
            self._in_heading = False
            self.heading = " ".join("".join(self._hbuf).split()) or None
            self.parts.append(".")  # başlık sonrası doğal duraklama

    def handle_data(self, data):
        if self.skip:
            return
        self.parts.append(data)
        if self._in_heading:
            self._hbuf.append(data)


def html_to_text(html: str) -> tuple[str, str | None]:
    p = _Text()
    p.feed(html)
    return "".join(p.parts), p.heading


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1254", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", "replace")


# ----------------------------------------------------------------------------- EPUB
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_epub(path: Path) -> Book:
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in container.iter() if _local(e.tag) == "rootfile")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)

        manifest, spine, title = {}, [], path.stem
        for el in opf.iter():
            t = _local(el.tag)
            if t == "item":
                manifest[el.get("id")] = (el.get("href"), el.get("media-type", ""), el.get("properties", ""))
            elif t == "itemref" and el.get("linear", "yes") != "no":
                spine.append(el.get("idref"))
            elif t == "title" and el.text and title == path.stem:
                title = el.text.strip()

        # içindekiler tablosundan başlıklar (varsa)
        toc = _epub_toc(z, manifest, base)

        sections: list[Section] = []
        for idref in spine:
            if idref not in manifest:
                continue
            href, mtype, _ = manifest[idref]
            if "html" not in mtype and "xml" not in mtype:
                continue
            full = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))
            try:
                text, heading = html_to_text(_decode(z.read(full)))
            except KeyError:
                continue
            if len(text.strip()) < 20:
                continue
            name = toc.get(full) or heading or f"Bölüm {len(sections) + 1}"
            sections.append(Section(name, text))
    return Book(path, title, sections)


def _epub_toc(z: zipfile.ZipFile, manifest: dict, base: str) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        nav = next((h for h, _, p in manifest.values() if "nav" in p.split()), None)
        ncx = next((h for h, m, _ in manifest.values() if m == "application/x-dtbncx+xml"), None)
        if ncx:
            root = ET.fromstring(z.read(posixpath.normpath(posixpath.join(base, ncx))))
            ncx_base = posixpath.dirname(posixpath.normpath(posixpath.join(base, ncx)))
            for np_ in root.iter():
                if _local(np_.tag) != "navPoint":
                    continue
                label = next((t.text for t in np_.iter() if _local(t.tag) == "text" and t.text), None)
                src = next((c.get("src") for c in np_ if _local(c.tag) == "content"), None)
                if label and src:
                    key = posixpath.normpath(posixpath.join(ncx_base, unquote(src.split("#")[0])))
                    out.setdefault(key, " ".join(label.split()))
        elif nav:
            nav_full = posixpath.normpath(posixpath.join(base, nav))
            nb = posixpath.dirname(nav_full)
            for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', _decode(z.read(nav_full)), re.S):
                label = " ".join(re.sub(r"<[^>]+>", "", m.group(2)).split())
                if label:
                    key = posixpath.normpath(posixpath.join(nb, unquote(m.group(1).split("#")[0])))
                    out.setdefault(key, label)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- PDF
def read_pdf(path: Path, pages_per_section: int = 20) -> Book:
    import pymupdf

    doc = pymupdf.open(path)
    try:
        title = (doc.metadata or {}).get("title") or path.stem
        # Önce PDF yer imlerini (outline) dene: gerçek bölüm sınırları.
        toc = [(lvl, t, p) for lvl, t, p in doc.get_toc() if p >= 1]
        top = min((l for l, _, _ in toc), default=1)
        marks = [(t.strip(), p - 1) for l, t, p in toc if l == top]
        # Aynı sayfaya düşenleri tekilleştir
        seen, uniq = set(), []
        for t, p in marks:
            if p not in seen:
                seen.add(p)
                uniq.append((t, p))
        if len(uniq) < 2:
            uniq = [(f"Sayfa {s + 1}–{min(s + pages_per_section, len(doc))}", s)
                    for s in range(0, len(doc), pages_per_section)]
        elif uniq[0][1] > 0:
            uniq.insert(0, ("Giriş", 0))
        uniq.sort(key=lambda x: x[1])

        sections = []
        for i, (name, start) in enumerate(uniq):
            end = uniq[i + 1][1] if i + 1 < len(uniq) else len(doc)
            pages = [doc[n].get_text("text", sort=True) for n in range(start, end)]
            sections.append(Section(name, "\n\n".join(pages), page_start=start + 1))
    finally:
        doc.close()
    return Book(path, str(title), [s for s in sections if s.text.strip()])


# ----------------------------------------------------------------------------- düz metin
def read_text(path: Path) -> Book:
    raw = _decode(path.read_bytes())
    if path.suffix.lower() in (".html", ".htm"):
        raw, _ = html_to_text(raw)
    elif path.suffix.lower() == ".md":
        raw = re.sub(r"^#{1,6}\s*", "", raw, flags=re.M)
        raw = re.sub(r"[*_`>]+", "", raw)
    return Book(path, path.stem, [Section(path.stem, raw)])


def read_book(path: str | Path, pdf_pages_per_section: int = 20) -> Book:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".epub":
        return read_epub(p)
    if ext == ".pdf":
        return read_pdf(p, pdf_pages_per_section)
    if ext in SUPPORTED:
        return read_text(p)
    raise ValueError(f"Desteklenmeyen dosya türü: {ext}")
