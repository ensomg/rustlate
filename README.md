# rustlate

**Canlı altyazı çevirici.** Seçtiğin uygulamanın sesini dinler, ekranda çevrilmiş altyazı gösterir. Windows için, Rust ile yazıldı. Konuşma tanıma ve çeviri bilgisayarında çalışır, ses buluta gönderilmez.

*Live subtitle translator for Windows. Captures the audio of one chosen app, transcribes and translates it on your machine, and shows subtitles in a click-through overlay.*

Site: <https://rustlate.vercel.app>

## İndir

[Releases](https://github.com/ensomg/rustlate/releases/latest) sayfasından `rustlate-windows-x64.exe` dosyasını indir ve çalıştır. Kurulum gerekmez (11,6 MB).

> Windows SmartScreen "bilinmeyen yayıncı" uyarısı gösterebilir, çünkü exe imzalı değil. "Ek bilgi → Yine de çalıştır" ile açılır. Kaynak kod burada, istersen kendin derleyebilirsin.

## Kullanım

1. **Dilleri seç** (konuşulan ve altyazı dili). Uygun çeviri modeli otomatik seçilir.
2. **Eksikleri indir.** Uygulama ekran kartını tanır ve ★ önerilen bileşenleri gösterir. Modeller ve konuşma motoru yalnızca ihtiyaç olunca, bir kez iner.
3. **Çevrilecek uygulamayı seç** ve başlat. Altyazı ekranın altında, tıklamaları engellemeden belirir. Kapatmak için **Ctrl + Shift + Q**.

Gereksinimler: Windows 10 (2004) veya üzeri. NVIDIA ekran kartı önerilir; olmadan da hafif modellerle işlemcide çalışır.

## Özellikler

- Yalnızca seçilen uygulamanın sesi dinlenir (WASAPI process loopback).
- Konuşma tanıma: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) sunucusu (CPU veya CUDA), ayrı bileşen olarak iner.
- Çeviri: İngilizce → Türkçe için Opus-MT, 56 dil için NLLB-200 (CTranslate2, int8).
- 6 altyazı stili (Klasik, Şarkı sözü, Karaoke, Daktilo, Pop, Kayan), renk, boyut, konum, animasyon hızı, özgün metin. Görünüm ayarları çalışan altyazıya anında uygulanır.
- Hız ve doğruluk ayarları: ara sonuç sıklığı, cümle sonu sessizliği, konuşma algılama eşiği.
- Ayarlar `%APPDATA%\rustlate\settings.toml`, bileşenler `%LOCALAPPDATA%\rustlate\components` altında.

## Kaynaktan derleme

Gerekenler: Rust (stable), Visual Studio Build Tools (C++), [CMake](https://cmake.org) (PATH'te).

```bash
cargo build --release
```

Çıktı: `target/release/rustlate.exe`. Komut satırı: `rustlate overlay <pid>`, `rustlate live <pid>`, `rustlate list`, `rustlate sim <wav>`, `rustlate mt <metin>`.

## Notlar ve lisanslar

- Bu depoda henüz bir lisans dosyası yok.
- İndirilen modellerin kendi lisansları geçerlidir. Özellikle NLLB-200 modeli **CC-BY-NC 4.0** (ticari olmayan kullanım). Whisper modelleri ve whisper.cpp MIT lisanslıdır.
- Gecikme (cümle bitince yaklaşık 1–2 sn) NVIDIA RTX 3060 Ti, Whisper small, İngilizce → Türkçe ile ölçüldü; kendi bilgisayarında değişebilir.
