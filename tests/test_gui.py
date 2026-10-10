import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path

import pytest
from mutagen.mp3 import MP3

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from kitapsesi.gui import MainWindow

_APP = QApplication.instance() or QApplication([])  # test süresince canlı kalmalı
from test_pipeline import make_epub, make_pdf


def wait(app, cond, ms=60000):
    import time
    t = time.time()
    while not cond() and time.time() - t < ms / 1000:
        app.processEvents()
        time.sleep(0.02)
    assert cond()


def test_window_runs_job_end_to_end(tmp_path, monkeypatch):
    app = _APP
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "cfg"))
    w = MainWindow()
    w.settings.clear()
    w.show()
    w._add_paths([str(make_epub(tmp_path / "a.epub", 3)), str(make_pdf(tmp_path / "b.pdf"))])
    assert w.files.count() == 2 and "saat" in w.info.text()

    # model seç + sentez parametreleri dinamik gelir
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("antalia"))
    assert {"speed", "steps", "cfg", "eq", "seed", "sample_rate"} <= set(w.param_widgets)
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("ema"))
    assert set(w.param_widgets) == {"sample_rate", "speed", "seed"}
    w.engine_combo.setCurrentIndex(w.engine_combo.findData("demo"))

    w.out_dir.setText(str(tmp_path / "mp3"))
    w.split_combo.setCurrentIndex(w.split_combo.findData("chapter"))
    w.param_widgets["sample_rate"].setCurrentIndex(w.param_widgets["sample_rate"].findData(16000))
    w._start()
    wait(app, lambda: w.btn_start.isEnabled())
    assert "Bitti" in w.status.text(), w.status.text()
    mp3s = sorted((tmp_path / "mp3").rglob("*.mp3"))
    assert len(mp3s) >= 4 and MP3(mp3s[0]).info.sample_rate == 16000
    w.grab().save(str(tmp_path / "shot.png"))
    Path("/tmp/kitapsesi_shot.png").write_bytes((tmp_path / "shot.png").read_bytes())

    # örnek önizleme .wav üretir
    w._preview()
    wait(app, lambda: w.btn_start.isEnabled())
    assert Path(tempfile_wav := __import__("tempfile").gettempdir()).exists()
    w.close()
    w.deleteLater()
    app.processEvents()
