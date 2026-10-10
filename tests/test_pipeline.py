import json
import zipfile
from pathlib import Path

import pymupdf
import pytest
from mutagen.mp3 import MP3

from kitapsesi.engines import DemoEngine
from kitapsesi.extract import read_book
from kitapsesi.pipeline import JobRunner, JobSettings
from kitapsesi.textprep import clean_text, make_chunks

PARA = ("Bu bir deneme cümlesidir. Kitap sesi uygulaması metni parçalara böler. "
        "Her parça seslendirilir ve MP3 olarak yazılır. ") * 6


def make_epub(path: Path, n=4):
    items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(n))
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(n))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   f'<package xmlns="http://www.idpf.org/2007/opf"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:title>Deneme Kitabı</dc:title></metadata><manifest>{items}</manifest><spine>{spine}</spine></package>')
        for i in range(n):
            z.writestr(f"OEBPS/c{i}.xhtml",
                       f"<html><body><h1>Bölüm {i + 1}</h1><p>{PARA}</p><p>{PARA}</p>"
                       f"<script>alert(1)</script></body></html>")
    return path


def make_pdf(path: Path, pages=12, bookmarks=False):
    doc = pymupdf.open()
    for i in range(pages):
        pg = doc.new_page()
        pg.insert_text((72, 60), "Kitap Başlığı", fontsize=9)  # tekrarlayan üst bilgi
        pg.insert_textbox(pymupdf.Rect(72, 90, 520, 760), f"Sayfa içeriği {i}. " + PARA, fontsize=11)
        pg.insert_text((290, 800), str(i + 1), fontsize=9)  # sayfa numarası
    if bookmarks:
        doc.set_toc([[1, "Birinci", 1], [1, "İkinci", 5], [1, "Üçüncü", 9]])
    doc.save(path)
    return path


def run(engine, st, files):
    return JobRunner(engine, st).run([str(f) for f in files])


def settings(tmp, **kw):
    return JobSettings(engine="demo", params={"sample_rate": 24000, "speed": 1.0, "seed": 1},
                       output_dir=str(tmp / "out"), **kw)


def test_epub_reading(tmp_path):
    b = read_book(make_epub(tmp_path / "k.epub"))
    assert b.title == "Deneme Kitabı"
    assert [s.title for s in b.sections][:2] == ["Bölüm 1", "Bölüm 2"]
    assert "alert" not in b.sections[0].text


def test_pdf_reading_bookmarks_and_cleaning(tmp_path):
    b = read_book(make_pdf(tmp_path / "k.pdf", bookmarks=True))
    assert [s.title for s in b.sections] == ["Birinci", "İkinci", "Üçüncü"]
    text = clean_text("\n\n".join(s.text for s in b.sections))
    assert "Kitap Başlığı" not in text  # tekrarlayan üst bilgi gitti
    assert "\n7\n" not in text


def test_hyphen_join_and_chunks():
    t = clean_text("Bu uzun bir kel-\nime örneğidir ve\ndevam eder.")
    assert "kelime" in t and "ve devam" in t
    chunks = make_chunks([("Başlık", "A. " * 400)], max_chars=200)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text == "Başlık." and chunks[0].section_start


def test_single_mp3(tmp_path):
    f = make_epub(tmp_path / "k.epub")
    outs = run(DemoEngine(), settings(tmp_path), [f])
    assert len(outs) == 1 and outs[0].name == "k.mp3"
    m = MP3(outs[0])
    assert m.info.length > 30 and m.info.sample_rate == 24000
    assert m.tags["TALB"].text[0] == "Deneme Kitabı"


def test_split_by_chapter(tmp_path):
    outs = run(DemoEngine(), settings(tmp_path, split_mode="chapter"), [make_epub(tmp_path / "k.epub")])
    assert len(outs) == 4
    assert outs[0].parent.name == "k" and "Bolum_1" in outs[0].name
    assert (tmp_path / "out" / "k.m3u8").exists()


def test_split_by_minutes_and_size(tmp_path):
    f = make_epub(tmp_path / "k.epub", n=6)
    total = MP3(run(DemoEngine(), settings(tmp_path / "a"), [f])[0]).info.length
    outs = run(DemoEngine(), settings(tmp_path / "b", split_mode="minutes", split_minutes=1), [f])
    assert len(outs) >= 2
    got = sum(MP3(o).info.length for o in outs)
    assert got == pytest.approx(total, rel=0.03)  # bölmek ses kaybettirmez
    outs = run(DemoEngine(), settings(tmp_path / "c", split_mode="size", split_mb=1), [f])
    assert len(outs) >= 1
    for o in outs[:-1]:
        assert o.stat().st_size < 1.3 * 1024 * 1024


def test_merge_inputs_and_pdf(tmp_path):
    a, b = make_epub(tmp_path / "a.epub", 1), make_pdf(tmp_path / "b.pdf")
    outs = run(DemoEngine(), settings(tmp_path / "x"), [a, b])
    assert sorted(o.name for o in outs) == ["a.mp3", "b.mp3"]
    outs = run(DemoEngine(), settings(tmp_path / "y", merge_inputs=True), [a, b])
    assert len(outs) == 1 and outs[0].name.endswith("_birlesik.mp3")


def test_cancel_and_resume_matches_uninterrupted(tmp_path):
    f = make_epub(tmp_path / "k.epub", n=6)
    full = MP3(run(DemoEngine(), settings(tmp_path / "ref", checkpoint_minutes=0.2), [f])[0]).info.length

    st = settings(tmp_path / "res", checkpoint_minutes=0.2)
    r = JobRunner(DemoEngine(), st)
    n = {"c": 0}

    def prog(p):
        n["c"] += 1
        if n["c"] == 10:
            r.cancel()

    r.on_progress = prog
    r.run([str(f)])
    assert not (tmp_path / "res" / "out" / "k.mp3").exists()  # yarım iş çıktı üretmedi
    state = json.loads((tmp_path / "res" / "out" / ".kitapsesi" / "k" / "state.json").read_text())
    assert 0 < state["next_chunk"] and not state["done"]

    outs = run(DemoEngine(), st, [f])  # devam
    assert MP3(outs[0]).info.length == pytest.approx(full, rel=0.02)
    again = run(DemoEngine(), st, [f])  # tamamlanmış iş atlanır
    assert again == outs


def test_resume_invalidated_when_settings_change(tmp_path):
    f = make_epub(tmp_path / "k.epub", n=2)
    first = MP3(run(DemoEngine(), settings(tmp_path), [f])[0]).info.length
    st2 = settings(tmp_path)
    st2.params["speed"] = 2.0
    outs = run(DemoEngine(), st2, [f])
    full = MP3(outs[0]).info.length
    assert full < 0.7 * first  # yeni hızla baştan üretildi (eski sonuç atlanmadı)


def test_engine_error_leaves_no_partial_part(tmp_path):
    class Boom(DemoEngine):
        n = 0

        def synth(self, text, params, seed):
            Boom.n += 1
            if Boom.n == 8:
                raise RuntimeError("CUDA OOM")
            yield from super().synth(text, params, seed)

    f = make_epub(tmp_path / "k.epub", n=3)
    st = settings(tmp_path)
    with pytest.raises(RuntimeError):
        run(Boom(), st, [f])
    assert not list((tmp_path / "out").glob("*.mp3"))
    assert not list((tmp_path / "out").rglob("*.part"))
    outs = run(DemoEngine(), st, [f])  # sonraki çalıştırma sorunsuz tamamlar
    assert outs[0].exists()
