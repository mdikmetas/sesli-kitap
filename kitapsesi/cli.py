"""Komut satırı: python -m kitapsesi cli --engine ema kitap.epub ..."""
from __future__ import annotations

import argparse
import sys

from .engines import ENGINES, create_engine
from .extract import SUPPORTED
from .pipeline import JobRunner, JobSettings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kitapsesi cli", description="EPUB/PDF/TXT -> MP3 (yerel Türkçe TTS)")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-e", "--engine", choices=sorted(ENGINES), default="ema")
    ap.add_argument("-o", "--output", default="outputs")
    ap.add_argument("--sample-rate", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="ANAHTAR=DEĞER",
                    help="motor parametresi (speed, steps, cfg, eq, seed ...) — birden çok kez verilebilir")
    ap.add_argument("--split", choices=["none", "chapter", "minutes", "size"], default="none")
    ap.add_argument("--split-minutes", type=int, default=60)
    ap.add_argument("--split-mb", type=int, default=100)
    ap.add_argument("--bitrate", type=int, default=64)
    ap.add_argument("--merge", action="store_true", help="tüm dosyaları tek kitap olarak birleştir")
    ap.add_argument("--pdf-pages", type=int, default=20, help="bookmark yoksa PDF'yi kaç sayfalık bölümlere ayır")
    ap.add_argument("--no-titles", action="store_true", help="bölüm başlıklarını okuma")
    ap.add_argument("--fresh", action="store_true", help="kaldığı yerden devam etme, baştan başla")
    a = ap.parse_args(argv)

    eng = create_engine(a.engine)
    params = eng.defaults()
    if a.sample_rate:
        params["sample_rate"] = a.sample_rate
    types = {p.key: p for p in eng.params}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in types:
            ap.error(f"{eng.label} için bilinmeyen parametre: {k} (var olanlar: {', '.join(types)})")
        p = types[k]
        params[k] = (v.lower() in ("1", "true", "evet", "açık", "on")) if p.kind == "bool" \
            else int(v) if p.kind == "int" else float(v) if p.kind == "float" else v
    st = JobSettings(engine=a.engine, params=params, output_dir=a.output, split_mode=a.split,
                     split_minutes=a.split_minutes, split_mb=a.split_mb, bitrate=a.bitrate,
                     merge_inputs=a.merge, pdf_pages_per_section=a.pdf_pages,
                     announce_titles=not a.no_titles, resume=not a.fresh)

    eng.load(lambda m: print(m, file=sys.stderr))
    print(f"Cihaz: {eng.device}", file=sys.stderr)

    def prog(p):
        eta = f"{int(p.eta // 60)} dk kaldı" if p.eta else ""
        print(f"\r[{p.unit}/{p.units}] {p.unit_name}  %{p.fraction * 100:5.1f}  "
              f"{p.audio_seconds / 60:7.1f} dk ses  {eta}      ", end="", file=sys.stderr, flush=True)

    runner = JobRunner(eng, st, prog, lambda m: print("\n" + m, file=sys.stderr))
    try:
        outs = runner.run(a.files)
    except KeyboardInterrupt:
        runner.cancel()
        print("\nDurduruldu; aynı komutla devam edebilirsiniz.", file=sys.stderr)
        return 130
    print()
    for o in outs:
        print(o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
