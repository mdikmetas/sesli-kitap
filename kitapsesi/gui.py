"""PySide6 arayüzü: model seç, sentez ayarlarını yap, dosyaları ekle, MP3 al."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from .engines import ENGINES, Engine, create_engine
from .extract import SUPPORTED, read_book
from .pipeline import CHARS_PER_SECOND, JobRunner, JobSettings, Progress

SPLIT_MODES = [("Bölme yok (kitap başına tek MP3)", "none"),
               ("Bölüm/başlık başına", "chapter"),
               ("Süreye göre (dakika)", "minutes"),
               ("Boyuta göre (MB)", "size")]
SAMPLE_TEXT = ("Merhaba, bu bir deneme okumasıdır. Seçtiğiniz model ve ayarlarla kitabınız "
               "bu şekilde seslendirilecek. 1923 yılında, yüzde yirmi beş artışla, saat 14.30'da.")


def fmt_time(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h} sa {m:02d} dk" if h else f"{m} dk {s:02d} sn"


class Worker(QObject):
    progress = Signal(object)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(list, str)  # çıktılar, hata

    def __init__(self, engines: dict[str, Engine], engine_id: str, st: JobSettings, files: list[str],
                 preview: bool = False):
        super().__init__()
        self.engines, self.engine_id, self.st, self.files, self.preview = engines, engine_id, st, files, preview
        self.runner: JobRunner | None = None
        self._cancel = False

    def cancel(self):
        self._cancel = True
        if self.runner:
            self.runner.cancel()

    def run(self):
        try:
            eng = self.engines.get(self.engine_id)
            if eng is None:
                eng = create_engine(self.engine_id)
                eng.load(self.status.emit)
                self.engines[self.engine_id] = eng
            self.status.emit(f"Cihaz: {eng.device}")
            if self.preview:
                self.finished.emit([self._preview(eng)], "")
                return
            self.runner = JobRunner(eng, self.st, self.progress.emit, self.log.emit)
            if self._cancel:
                self.runner.cancel()
            outs = self.runner.run(self.files)
            self.finished.emit([str(o) for o in outs], "")
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit([], f"{type(exc).__name__}: {exc}")

    def _preview(self, eng: Engine) -> str:
        text = self.files[0] if self.files else SAMPLE_TEXT
        seed = int(self.st.params.get("seed", -1))
        seed = seed if seed >= 0 else 12345
        audio = np.concatenate(list(eng.synth(text, self.st.params, seed)) or [np.zeros(1, np.float32)])
        path = Path(tempfile.gettempdir()) / "kitapsesi_onizleme.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.st.params["sample_rate"]))
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        return str(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kitap Sesi — EPUB / PDF → MP3")
        self.resize(1180, 760)
        self.engines: dict[str, Engine] = {}  # yüklenmiş modeller bellekte kalır
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.player = None
        self.settings = QSettings("KitapSesi", "KitapSesi")
        self.param_widgets: dict[str, QWidget] = {}
        self.setAcceptDrops(True)
        self._build()
        self._load_settings()
        self._on_engine_changed()
        self._on_split_changed()

    # ------------------------------------------------------------------ arayüz
    def _build(self):
        split = QSplitter(Qt.Horizontal)
        self.setCentralWidget(split)

        # Sol: dosyalar
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>1. Dosyalar</b> (EPUB, PDF, TXT, MD, HTML — sürükleyip bırakabilirsiniz)"))
        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lv.addWidget(self.files, 1)
        row = QHBoxLayout()
        for text, fn in (("Dosya ekle…", self._add_files), ("Seçileni çıkar", self._remove_files),
                         ("Temizle", self.files.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lv.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lv.addWidget(self.info)
        split.addWidget(left)

        # Sağ: ayarlar + çalıştırma
        right = QWidget()
        rv = QVBoxLayout(right)
        tabs = QTabWidget()
        rv.addWidget(tabs)
        tabs.addTab(self._model_tab(), "2. Model ve ses")
        tabs.addTab(self._output_tab(), "3. MP3 çıktısı")
        tabs.addTab(self._text_tab(), "Metin ayarları")

        row = QHBoxLayout()
        self.btn_preview = QPushButton("▶ Örnek dinle")
        self.btn_preview.clicked.connect(self._preview)
        self.btn_start = QPushButton("Başlat")
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("Durdur")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_open = QPushButton("Çıktı klasörünü aç")
        self.btn_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.out_dir.text())))
        for b in (self.btn_preview, self.btn_start, self.btn_stop, self.btn_open):
            row.addWidget(b)
        rv.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        rv.addWidget(self.bar)
        self.status = QLabel("Hazır.")
        rv.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        rv.addWidget(self.log, 1)
        split.addWidget(right)
        split.setSizes([380, 800])

    def _model_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.engine_combo = QComboBox()
        for e in ENGINES.values():
            self.engine_combo.addItem(e.label, e.id)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.engine_desc = QLabel()
        self.engine_desc.setWordWrap(True)
        v.addWidget(QLabel("Model:"))
        v.addWidget(self.engine_combo)
        v.addWidget(self.engine_desc)
        box = QGroupBox("Sentez parametreleri")
        self.params_form = QFormLayout(box)
        v.addWidget(box)
        v.addStretch(1)
        return w

    def _output_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        row = QHBoxLayout()
        self.out_dir = QLineEdit(str(Path.home() / "KitapSesi"))
        b = QPushButton("Seç…")
        b.clicked.connect(self._pick_dir)
        row.addWidget(self.out_dir, 1)
        row.addWidget(b)
        f.addRow("Çıktı klasörü:", row)
        self.bitrate = QComboBox()
        for k in (32, 48, 64, 96, 128, 192):
            self.bitrate.addItem(f"{k} kbps", k)
        self.bitrate.setCurrentIndex(2)
        f.addRow("MP3 kalitesi:", self.bitrate)
        self.split_combo = QComboBox()
        for label, key in SPLIT_MODES:
            self.split_combo.addItem(label, key)
        self.split_combo.currentIndexChanged.connect(self._on_split_changed)
        f.addRow("Dosya bölme:", self.split_combo)
        self.split_minutes = QSpinBox()
        self.split_minutes.setRange(1, 1440)
        self.split_minutes.setValue(60)
        self.split_minutes.setSuffix(" dk")
        self.split_mb = QSpinBox()
        self.split_mb.setRange(1, 4096)
        self.split_mb.setValue(100)
        self.split_mb.setSuffix(" MB")
        f.addRow("Parça süresi:", self.split_minutes)
        f.addRow("Parça boyutu:", self.split_mb)
        self.merge = QCheckBox("Birden çok dosyayı tek kitap gibi birleştir")
        f.addRow("", self.merge)
        self.artist = QLineEdit()
        self.artist.setPlaceholderText("isteğe bağlı (ID3 sanatçı etiketi)")
        f.addRow("Yazar etiketi:", self.artist)
        self.resume = QCheckBox("Kesilen işe kaldığı yerden devam et")
        self.resume.setChecked(True)
        f.addRow("", self.resume)
        note = QLabel("Uzun kitaplar 20 dakikalık ara kayıtlarla işlenir; bilgisayar kapansa bile "
                      "aynı ayarlarla tekrar başlatınca kalınan yerden devam edilir.")
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _text_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.pdf_pages = QSpinBox()
        self.pdf_pages.setRange(1, 500)
        self.pdf_pages.setValue(20)
        self.pdf_pages.setSuffix(" sayfa")
        f.addRow("PDF bölüm boyu (yer imi yoksa):", self.pdf_pages)
        self.announce = QCheckBox("Bölüm başlıklarını da oku")
        self.announce.setChecked(True)
        f.addRow("", self.announce)
        self.strip = QCheckBox("Sayfa numaralarını ve tekrarlayan üst/alt bilgileri at")
        self.strip.setChecked(True)
        f.addRow("", self.strip)
        self.chunk = QSpinBox()
        self.chunk.setRange(200, 3000)
        self.chunk.setSingleStep(100)
        self.chunk.setValue(900)
        self.chunk.setSuffix(" karakter")
        f.addRow("Okuma parçası uzunluğu:", self.chunk)
        self.p_sent, self.p_par, self.p_sec = QSpinBox(), QSpinBox(), QSpinBox()
        for sp, val in ((self.p_sent, 150), (self.p_par, 600), (self.p_sec, 1500)):
            sp.setRange(0, 10000)
            sp.setSingleStep(50)
            sp.setValue(val)
            sp.setSuffix(" ms")
        f.addRow("Parçalar arası duraklama:", self.p_sent)
        f.addRow("Paragraf sonu duraklaması:", self.p_par)
        f.addRow("Bölüm arası duraklama:", self.p_sec)
        return w

    # ------------------------------------------------------------------ dinamik parametreler
    def _engine(self) -> type[Engine]:
        return ENGINES[self.engine_combo.currentData()]

    def _on_engine_changed(self):
        eng = self._engine()
        self.engine_desc.setText(eng.description)
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.param_widgets.clear()
        saved = json.loads(self.settings.value(f"params/{eng.id}", "{}"))

        sr = QComboBox()
        for r in eng.sample_rates:
            sr.addItem(f"{r} Hz", r)
        i = sr.findData(saved.get("sample_rate", eng.sample_rates[0]))
        sr.setCurrentIndex(max(i, 0))
        self.params_form.addRow("Örnekleme hızı:", sr)
        self.param_widgets["sample_rate"] = sr

        for p in eng.params:
            val = saved.get(p.key, p.default)
            if p.kind == "float":
                w = QDoubleSpinBox()
                w.setRange(p.min, p.max)
                w.setSingleStep(p.step or 0.1)
                w.setDecimals(2)
                w.setValue(float(val))
            elif p.kind == "int":
                w = QSpinBox()
                w.setRange(int(p.min), int(p.max))
                w.setValue(int(val))
            elif p.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
            else:
                w = QComboBox()
                for c in p.choices:
                    w.addItem(str(c), c)
                w.setCurrentIndex(max(w.findData(val), 0))
            if p.help:
                w.setToolTip(p.help)
            self.params_form.addRow(p.label + ":", w)
            self.param_widgets[p.key] = w

    def _params(self) -> dict:
        out = {}
        for k, w in self.param_widgets.items():
            out[k] = (w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.isChecked()
                      if isinstance(w, QCheckBox) else w.currentData())
        return out

    def _on_split_changed(self):
        mode = self.split_combo.currentData()
        self.split_minutes.setEnabled(mode == "minutes")
        self.split_mb.setEnabled(mode == "size")

    # ------------------------------------------------------------------ dosyalar
    def _add_paths(self, paths):
        have = {self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())}
        for p in paths:
            if Path(p).suffix.lower() in SUPPORTED and p not in have:
                self.files.addItem(Path(p).name)
                self.files.item(self.files.count() - 1).setData(Qt.UserRole, p)
        self._update_info()

    def _paths(self) -> list[str]:
        return [self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())]

    def _add_files(self):
        exts = " ".join("*" + e for e in sorted(SUPPORTED))
        paths, _ = QFileDialog.getOpenFileNames(self, "Dosya seç", str(Path.home()), f"Belgeler ({exts})")
        self._add_paths(paths)

    def _remove_files(self):
        for it in self.files.selectedItems():
            self.files.takeItem(self.files.row(it))
        self._update_info()

    def _update_info(self):
        """Hızlı tahmin: dosya boyutundan değil, gerçekten okunan karakter sayısından."""
        paths = self._paths()
        if not paths:
            self.info.setText("")
            return
        try:
            chars, secs = 0, 0.0
            for p in paths:
                chars += read_book(p).chars
            speed = float(self.param_widgets.get("speed").value()) if "speed" in self.param_widgets else 1.0
            secs = chars / CHARS_PER_SECOND / max(speed, 0.1)
            self.info.setText(f"{len(paths)} dosya · yaklaşık {chars:,} karakter · tahmini ses ≈ {secs / 3600:.1f} saat"
                              .replace(",", "."))
        except Exception as exc:  # noqa: BLE001
            self.info.setText(f"Okunamadı: {exc}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add_paths([u.toLocalFile() for u in e.mimeData().urls()])

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Çıktı klasörü", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    # ------------------------------------------------------------------ çalıştırma
    def job_settings(self) -> JobSettings:
        return JobSettings(
            engine=self.engine_combo.currentData(), params=self._params(), output_dir=self.out_dir.text(),
            merge_inputs=self.merge.isChecked(), split_mode=self.split_combo.currentData(),
            split_minutes=self.split_minutes.value(), split_mb=self.split_mb.value(),
            bitrate=self.bitrate.currentData(), pdf_pages_per_section=self.pdf_pages.value(),
            chunk_chars=self.chunk.value(), pause_sentence_ms=self.p_sent.value(),
            pause_paragraph_ms=self.p_par.value(), pause_section_ms=self.p_sec.value(),
            announce_titles=self.announce.isChecked(), strip_headers=self.strip.isChecked(),
            resume=self.resume.isChecked(), id3_artist=self.artist.text().strip())

    def _set_busy(self, busy: bool):
        self.btn_start.setEnabled(not busy)
        self.btn_preview.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.engine_combo.setEnabled(not busy)

    def _launch(self, worker: Worker):
        self._save_settings()
        self.worker = worker
        self.thread = QThread()
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self.log.appendPlainText)
        worker.status.connect(self.status.setText)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(self.thread.quit)
        self._set_busy(True)
        self.thread.start()

    def _start(self):
        if not self._paths():
            QMessageBox.information(self, "Kitap Sesi", "Önce en az bir dosya ekleyin.")
            return
        Path(self.out_dir.text()).mkdir(parents=True, exist_ok=True)
        self.bar.setValue(0)
        self.log.clear()
        self.status.setText("Model yükleniyor…")
        self._launch(Worker(self.engines, self.engine_combo.currentData(), self.job_settings(), self._paths()))

    def _preview(self):
        self.status.setText("Örnek hazırlanıyor…")
        self._launch(Worker(self.engines, self.engine_combo.currentData(), self.job_settings(), [SAMPLE_TEXT],
                            preview=True))

    def _stop(self):
        if self.worker:
            self.status.setText("Durduruluyor (geçerli parça bitince)…")
            self.worker.cancel()

    def _on_progress(self, p: Progress):
        self.bar.setValue(int(p.fraction * 1000))
        rtf = f" · {p.audio_seconds / p.elapsed:.0f}× gerçek zamanlı" if p.elapsed > 3 and p.audio_seconds else ""
        self.status.setText(f"[{p.unit}/{p.units}] {p.unit_name} — %{p.fraction * 100:.1f} · "
                            f"{fmt_time(p.audio_seconds)} ses üretildi · kalan ≈ {fmt_time(p.eta)}{rtf}")

    def _on_finished(self, outs: list, error: str):
        if self.thread:  # iş parçacığı güvenle sonlansın, sonra referanslar bırakılsın
            self.thread.quit()
            self.thread.wait(5000)
        self._set_busy(False)
        self.worker = None
        self.thread = None
        if error:
            self.status.setText("Hata.")
            QMessageBox.critical(self, "Kitap Sesi", error)
            return
        if outs and outs[0].endswith(".wav"):
            self._play(outs[0])
            self.status.setText("Örnek çalınıyor.")
            return
        self.bar.setValue(1000 if outs else self.bar.value())
        self.status.setText(f"Bitti: {len(outs)} MP3 dosyası." if outs else "Durduruldu.")
        if outs:
            self.log.appendPlainText("\n".join(outs))

    def _play(self, path: str):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":  # başsız ortam: ses aygıtı yok
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self.player = QMediaPlayer()
            self._audio_out = QAudioOutput()
            self.player.setAudioOutput(self._audio_out)
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
        except Exception:  # noqa: BLE001
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ------------------------------------------------------------------ ayar kaydı
    def _save_settings(self):
        eng = self.engine_combo.currentData()
        s = self.settings
        s.setValue(f"params/{eng}", json.dumps(self._params()))
        s.setValue("engine", eng)
        st = asdict(self.job_settings())
        st.pop("params")
        s.setValue("job", json.dumps(st))

    def _load_settings(self):
        s = self.settings
        i = self.engine_combo.findData(s.value("engine", "ema"))
        self.engine_combo.setCurrentIndex(max(i, 0))
        try:
            st = json.loads(s.value("job", "{}"))
        except ValueError:
            return
        self.out_dir.setText(st.get("output_dir", self.out_dir.text()))
        self.split_combo.setCurrentIndex(max(self.split_combo.findData(st.get("split_mode", "none")), 0))
        self.split_minutes.setValue(st.get("split_minutes", 60))
        self.split_mb.setValue(st.get("split_mb", 100))
        self.bitrate.setCurrentIndex(max(self.bitrate.findData(st.get("bitrate", 64)), 0))
        self.pdf_pages.setValue(st.get("pdf_pages_per_section", 20))
        self.chunk.setValue(st.get("chunk_chars", 900))
        self.p_sent.setValue(st.get("pause_sentence_ms", 150))
        self.p_par.setValue(st.get("pause_paragraph_ms", 600))
        self.p_sec.setValue(st.get("pause_section_ms", 1500))
        self.announce.setChecked(st.get("announce_titles", True))
        self.strip.setChecked(st.get("strip_headers", True))
        self.artist.setText(st.get("id3_artist", ""))

    def closeEvent(self, e):
        if self.worker:
            self.worker.cancel()
            if self.thread:
                self.thread.quit()
                self.thread.wait(15000)
        self._save_settings()
        e.accept()


def run() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    code = app.exec()
    w.deleteLater()
    app.processEvents()
    del w  # pencere, QApplication'dan önce yok edilmeli
    return code
