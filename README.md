# Kitap Sesi

EPUB / PDF / TXT dosyalarını yerel Türkçe TTS modelleriyle **MP3 sesli kitaba** çeviren masaüstü uygulaması.
[muhabbet-kusu](https://github.com/brknl28/muhabbet-kusu) projesinin kitap odaklı benzeridir.

- **Model seç:** EMA Lightning, Antalia-2 Mini (yeni model eklemek için `kitapsesi/engines.py`).
- **Sentez ayarları modele göre gelir:** hız, örnekleme hızı, seed; Antalia için adım, CFG, EQ.
- **Dosya türleri:** EPUB, PDF (yer imlerinden bölüm çıkarır; yoksa N sayfalık bölümler), TXT, MD, HTML. Birden çok dosya eklenebilir.
- **1000 sayfalık kitaplar:** tek MP3, bölüm başına, X dakikada bir veya X MB'ta bir bölünmüş çoklu MP3. Çoklu çıktıda `.m3u8` çalma listesi de yazılır.
- **Kesintiye dayanıklı:** 20 dakikalık ara kayıtlar tutulur; durdurulan, çöken ya da kapanan iş aynı ayarlarla başlatılınca kaldığı yerden sürer.
- **Metin temizliği:** satır sonu tireleri, sayfa numaraları, tekrarlayan üst/alt bilgiler atılır; bölüm başlıkları okunur (kapatılabilir).
- ID3 etiketleri (başlık, albüm, parça no) yazılır. FFmpeg gerekmez.

## Kurulum

Python 3.11+ gerekir.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU için; CPU için düz "pip install torch"
pip install -r requirements.txt
python -m kitapsesi
```

Modeller ilk kullanımda Hugging Face'ten indirilir ve `models/` klasöründe saklanır (`KITAPSESI_MODELS_DIR` ile değişir); sonrasında internet gerekmez.

## Kullanım (arayüz)

1. Dosyaları ekleyin veya pencereye sürükleyin; tahmini ses süresi görünür.
2. **Model ve ses** sekmesinde modeli ve parametreleri seçin, **▶ Örnek dinle** ile deneyin.
3. **MP3 çıktısı** sekmesinde klasör, kalite ve bölme düzenini seçin → **Başlat**.

Konuşma için 48–64 kbps genellikle yeterlidir (1 saat ≈ 22–29 MB).

## Komut satırı

```powershell
python -m kitapsesi cli -e antalia --set steps=12 --set cfg=2.5 --set speed=0.95 `
    --split minutes --split-minutes 120 --bitrate 64 -o cikti kitap.epub
python -m kitapsesi cli -e ema --split chapter --pdf-pages 25 buyuk_kitap.pdf
```

Yarıda kesilen iş için aynı komutu tekrar çalıştırın; baştan başlamak için `--fresh`.

## Çıktı düzeni

- Bölme yok: `cikti/kitap.mp3`
- Bölünmüş: `cikti/kitap/kitap_01.mp3`, `kitap_02.mp3`, … ve `cikti/kitap.m3u8`
- `cikti/.kitapsesi/` devam bilgisini tutar (silinebilir; silerseniz iş baştan başlar).

## Sınırlar

- Taranmış (görüntü) PDF'lerde metin yoktur; önce OCR gerekir.
- İlerleme/ETA tahmini karakter sayısına göredir; PDF'de tablo ve çok sütunlu sayfalar okuma sırasını bozabilir.
- Testler `DemoEngine` ile çalışır (`pytest`); gerçek modellerle GPU'nuzda bir kısa deneme yapmanızı öneririm.
