//! The one window of the app. Simple by default: languages, setup, pick an app, start.
//! Everything else (models, speed, looks) is one tab away, with recommended choices starred.

use crate::audio::windows_list::{self, WindowInfo};
use crate::components::{self, Component, Kind, Progress, State};
use crate::config::{Backend, Position, Settings, Style};
use crate::{mt, style::Subtitles};
use eframe::egui::{self, Color32, RichText};
use std::collections::HashMap;
use std::os::windows::process::CommandExt;
use std::process::Child;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Start,
    Display,
    Models,
    Advanced,
}

const DEMO: [&str; 5] = [
    "Bir şey tespit ettim, bu bizim ilk tespitimiz ve altımda bir ses var.",
    "Bu modül aslında onu atlıyor, ki bu çılgınca.",
    "Açalım ve bu çiftliğin ne kadar para kazandığını görelim.",
    "Geçen hafta beni iki dakikadan kısa sürede yasakladı.",
    "Tamam, beş dakikadır çalışıyor ve zaten dört tam kutu eşyamız var.",
];

const OK: Color32 = Color32::from_rgb(90, 200, 120);
const WARN: Color32 = Color32::from_rgb(240, 170, 60);
const BAD: Color32 = Color32::from_rgb(230, 90, 90);
const STAR: &str = "★";

struct App {
    settings: Settings,
    saved: Settings,
    tab: Tab,
    windows: Vec<WindowInfo>,
    filter: String,
    selected: Option<WindowInfo>,
    overlay: Option<(Child, Settings)>, // running overlay + the settings it started with
    jobs: HashMap<&'static str, Arc<Progress>>,
    installed: HashMap<&'static str, bool>,
    refreshed: Instant,
    /// 0 = still checking, 1 = NVIDIA GPU found, 2 = none.
    gpu: Arc<AtomicU8>,
    preview: Subtitles,
    preview_at: Instant,
    preview_i: usize,
    message: String,
}

pub fn run() -> anyhow::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([880.0, 800.0]).with_min_inner_size([700.0, 560.0]).with_title("rustlate"),
        ..Default::default()
    };
    eframe::run_native(
        "rustlate",
        opts,
        Box::new(|cc| {
            crate::fonts::install(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = egui::vec2(8.0, 7.0);
            style.spacing.button_padding = egui::vec2(10.0, 5.0);
            cc.egui_ctx.set_style(style);
            Ok(Box::new(App::new()))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn list_windows() -> Vec<WindowInfo> {
    let me = std::process::id();
    const HIDDEN: [&str; 4] = ["Program Manager", "Windows Input Experience", "rustlate", "rustlate overlay"];
    let mut v: Vec<_> = windows_list::list_windows()
        .into_iter()
        .filter(|w| w.pid != me && !HIDDEN.contains(&w.title.as_str()) && !w.title.ends_with("-siw"))
        .collect();
    v.sort_by_key(|w| w.title.to_lowercase());
    v
}

/// Looks for an NVIDIA GPU in the background (`nvidia-smi -L`), result goes into `out`.
fn detect_gpu(out: Arc<AtomicU8>) {
    std::thread::spawn(move || {
        let ok = std::process::Command::new("nvidia-smi")
            .arg("-L")
            .creation_flags(0x0800_0000)
            .output()
            .map_or(false, |o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GPU"));
        out.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
    });
}

fn gb(mb: u32) -> String {
    if mb >= 1000 { format!("{:.1} GB", mb as f32 / 1000.0) } else { format!("{mb} MB") }
}

/// Turkish name for catalog entries (the catalog itself is English).
fn name_tr(c: &Component) -> String {
    c.name
        .replace("(English, small file)", "(İngilizce, küçük dosya)")
        .replace("(multilingual, small file)", "(çok dilli, küçük dosya)")
        .replace("(English)", "(İngilizce)")
        .replace("(multilingual)", "(çok dilli)")
        .replace("(small file)", "(küçük dosya)")
        .replace("Speech engine (CPU)", "Konuşma motoru (işlemci)")
        .replace("Speech engine (NVIDIA GPU, CUDA 12)", "Konuşma motoru (NVIDIA ekran kartı)")
        .replace("English to Turkish", "İngilizce → Türkçe")
}

fn desc_tr(c: &Component) -> String {
    let n = c.id;
    let small_file = if n.contains("q5") || n.contains("q8") { " Sıkıştırılmış sürüm: aynı kalite, daha küçük dosya." } else { "" };
    let base = match c.kind {
        Kind::SttServer if n.ends_with("cuda") => "NVIDIA ekran kartları için. Büyük indirme ama çok daha hızlı ve doğru model kullanılabilir.",
        Kind::SttServer => "Her bilgisayarda çalışır. Küçük indirme.",
        Kind::MtModel if n.starts_with("nllb") => "200 dil arasında çeviri. İşlemcide uzun cümlelerde birkaç saniye sürebilir.",
        Kind::MtModel => "Hızlı ve kaliteli İngilizce → Türkçe çeviri (yalnızca bu dil çifti).",
        Kind::SttModel if n.contains("tiny") => "En hızlı, en düşük doğruluk. Sadece deneme için.",
        Kind::SttModel if n.contains("base") => "Hızlı, orta doğruluk. İşlemcide canlı kullanılabilir.",
        Kind::SttModel if n.contains("small") => "İyi doğruluk. Ekran kartında anlık, çoğu işlemcide yavaş.",
        Kind::SttModel if n.contains("medium") => "Çok iyi doğruluk. Ekran kartı gerekir.",
        Kind::SttModel => "En iyi doğruluk, ekran kartında hızlı. İşlemci için uygun değil.",
    };
    format!("{base}{small_file}")
}

fn lang_label(iso: &str) -> String {
    mt::languages().iter().find(|l| l.iso == iso).map_or(iso.to_string(), |l| format!("{} ({})", l.name, l.iso))
}

fn lang_combo(ui: &mut egui::Ui, id: &str, value: &mut String, with_auto: bool) -> bool {
    let before = value.clone();
    let shown = if value == "auto" { "Otomatik algıla".to_string() } else { lang_label(value) };
    egui::ComboBox::from_id_salt(id).width(230.0).selected_text(shown).show_ui(ui, |ui| {
        if with_auto {
            ui.selectable_value(value, "auto".to_string(), "Otomatik algıla");
        }
        for l in mt::languages() {
            ui.selectable_value(value, l.iso.to_string(), format!("{} ({})", l.name, l.iso));
        }
    });
    *value != before
}

fn style_name(s: Style) -> &'static str {
    match s {
        Style::Classic => "Klasik",
        Style::Lyrics => "Şarkı sözü",
        Style::Karaoke => "Karaoke",
        Style::Typewriter => "Daktilo",
        Style::Pop => "Pop",
        Style::Slide => "Kayan",
    }
}

fn card(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style()).inner_margin(12.0).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(title).strong().size(16.0));
        ui.add_space(4.0);
        body(ui);
    });
    ui.add_space(8.0);
}

impl App {
    fn new() -> Self {
        let settings = Settings::load();
        let gpu = Arc::new(AtomicU8::new(0));
        detect_gpu(gpu.clone());
        let mut app = Self {
            preview: Subtitles::new(&settings.display, &settings.speed),
            saved: settings.clone(),
            settings,
            // RUSTLATE_TAB=display|models|advanced opens that tab (for screenshots).
            tab: match std::env::var("RUSTLATE_TAB").as_deref() {
                Ok("display") => Tab::Display,
                Ok("models") => Tab::Models,
                Ok("advanced") => Tab::Advanced,
                _ => Tab::Start,
            },
            windows: list_windows(),
            filter: String::new(),
            selected: None,
            overlay: None,
            jobs: HashMap::new(),
            installed: HashMap::new(),
            refreshed: Instant::now() - Duration::from_secs(10),
            gpu,
            preview_at: Instant::now(),
            preview_i: 0,
            message: String::new(),
        };
        // Adopt files that already sit in the working folder (developer setups).
        let data = app.settings.data_dir();
        if let Ok(cwd) = std::env::current_dir() {
            let done = components::import_existing(&cwd, &data);
            if !done.is_empty() {
                app.message = format!("{} bileşen mevcut dosyalardan alındı.", done.len());
            }
        }
        app.refresh_installed();
        app
    }

    fn has_gpu(&self) -> bool {
        self.gpu.load(Ordering::Relaxed) == 1
    }

    fn english_source(&self) -> bool {
        self.settings.translation.source == "en"
    }

    // ----- recommendations -----

    fn rec_server(&self) -> &'static str {
        if self.has_gpu() { "whisper-server-cuda" } else { "whisper-server-cpu" }
    }

    fn rec_stt(&self) -> &'static str {
        match (self.has_gpu(), self.english_source()) {
            (true, true) => "whisper-small.en",
            (true, false) => "whisper-small",
            (false, true) => "whisper-base.en",
            (false, false) => "whisper-base",
        }
    }

    fn rec_mt(&self) -> &'static str {
        if self.settings.translation.source == "en" && self.settings.translation.target == "tr" { "opus-mt-en-tr" } else { "nllb-200-600m" }
    }

    fn is_recommended(&self, id: &str) -> bool {
        id == self.rec_server() || id == self.rec_stt() || id == self.rec_mt()
    }

    /// After the languages changed: keep models consistent with them.
    fn apply_languages(&mut self) {
        let src = self.settings.translation.source.clone();
        self.settings.speech.language = src.clone();
        self.settings.translation.model = self.rec_mt().to_string();
        // NLLB is several times slower than Opus on a CPU: beam 1 keeps live subtitles usable.
        self.settings.translation.beam_size = if self.settings.translation.model.starts_with("nllb") { 1 } else { 2 };
        if src != "en" && self.settings.speech.model.contains(".en") {
            self.settings.speech.model = self.rec_stt().to_string();
        }
    }

    // ----- components -----

    fn refresh_installed(&mut self) {
        let data = self.settings.data_dir();
        for c in components::catalog() {
            self.installed.insert(c.id, components::is_installed(c, &data));
        }
        self.refreshed = Instant::now();
    }

    fn is_installed(&self, id: &str) -> bool {
        self.installed.get(id).copied().unwrap_or(false)
    }

    fn start_download(&mut self, id: &'static str) {
        let Some(c) = components::find(id) else { return };
        let (p, data) = (Progress::new(), self.settings.data_dir());
        let p2 = p.clone();
        std::thread::spawn(move || {
            let _ = components::download(c, &data, p2);
        });
        self.jobs.insert(id, p);
    }

    /// What this setup still needs: (component id, ready?) for engine, speech model, translation model.
    fn needs(&self) -> Vec<(&'static str, bool)> {
        let data = self.settings.data_dir();
        let engine_id = match self.settings.speech.backend {
            Backend::Cuda => "whisper-server-cuda",
            Backend::Cpu => "whisper-server-cpu",
            Backend::Auto => self.rec_server(),
        };
        let engine_ok = components::stt_server_exe(self.settings.speech.backend, &data).is_some();
        let mut v = vec![(engine_id, engine_ok)];
        if let Some(c) = components::find(&self.settings.speech.model) {
            v.push((c.id, self.is_installed(c.id)));
        }
        let t = &self.settings.translation;
        if t.source != t.target {
            if let Some(c) = components::find(&t.model) {
                v.push((c.id, self.is_installed(c.id)));
            }
        }
        v
    }

    // ----- overlay process -----

    fn start_overlay(&mut self) {
        let Some(w) = self.selected.clone() else { return };
        self.stop_overlay();
        let _ = self.settings.save();
        self.saved = self.settings.clone();
        match std::env::current_exe().and_then(|exe| std::process::Command::new(exe).arg("overlay").arg(w.pid.to_string()).spawn()) {
            Ok(child) => {
                self.overlay = Some((child, self.settings.clone()));
                self.message = format!("Çeviri başladı: {}", w.title);
            }
            Err(e) => self.message = format!("Başlatılamadı: {e}"),
        }
    }

    fn stop_overlay(&mut self) {
        if let Some((mut child, _)) = self.overlay.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Speech, translation or speed changed after the overlay started.
    fn needs_restart(&self) -> bool {
        self.overlay.as_ref().map_or(false, |(_, s)| {
            s.speech != self.settings.speech || s.translation != self.settings.translation || s.speed != self.settings.speed
        })
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if self.refreshed.elapsed() > Duration::from_millis(800) {
            self.refresh_installed();
        }
        if let Some((child, _)) = &mut self.overlay {
            if matches!(child.try_wait(), Ok(Some(_))) {
                self.overlay = None;
                self.message = "Çeviri durdu.".into();
            }
        }
        self.jobs.retain(|_, p| !matches!(p.state(), State::Done));
        if !self.jobs.is_empty() || self.overlay.is_some() || self.gpu.load(Ordering::Relaxed) == 0 {
            ctx.request_repaint_after(Duration::from_millis(250));
        }

        // Save when something changed and no slider is being dragged.
        if self.settings != self.saved && !ctx.input(|i| i.pointer.any_down()) {
            if let Err(e) = self.settings.save() {
                self.message = format!("Ayarlar kaydedilemedi: {e}");
            }
            self.preview.set_settings(&self.settings.display, &self.settings.speed);
            self.saved = self.settings.clone();
        }

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (t, name) in [(Tab::Start, "▶  Başlat"), (Tab::Display, "🎨  Görünüm"), (Tab::Models, "⬇  Modeller"), (Tab::Advanced, "⚙  Gelişmiş")] {
                    ui.selectable_value(&mut self.tab, t, RichText::new(name).size(16.0));
                }
            });
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                match &self.overlay {
                    Some(_) => ui.colored_label(OK, "● Çalışıyor  (durdurmak için Ctrl+Shift+Q)"),
                    None => ui.weak("○ Durdu"),
                };
                ui.separator();
                ui.label(&self.message);
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| match self.tab {
                Tab::Start => self.tab_start(ui),
                Tab::Display => self.tab_display(ui, ctx),
                Tab::Models => self.tab_models(ui),
                Tab::Advanced => self.tab_advanced(ui),
            });
        });
    }
}

impl App {
    fn tab_start(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Canlı altyazı çevirici").size(22.0).strong());
        ui.weak("Seçtiğin uygulamanın sesini dinler, ekranda çevrilmiş altyazı gösterir.");
        ui.add_space(10.0);

        // 1. Languages
        card(ui, "1 · Diller", |ui| {
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Konuşulan:");
                changed |= lang_combo(ui, "src", &mut self.settings.translation.source, false);
                if ui.button("⇄").on_hover_text("Dilleri değiştir").clicked() {
                    std::mem::swap(&mut self.settings.translation.source, &mut self.settings.translation.target);
                    changed = true;
                }
                ui.label("Altyazı:");
                changed |= lang_combo(ui, "dst", &mut self.settings.translation.target, false);
            });
            if changed {
                if self.settings.translation.source == self.settings.translation.target {
                    self.message = "Kaynak ve hedef dil aynı: çeviri yapılmaz.".into();
                }
                self.apply_languages();
            }
            ui.weak(format!(
                "Önerilen çeviri modeli otomatik seçildi: {}",
                components::find(self.rec_mt()).map_or("-".to_string(), name_tr)
            ));
        });

        // 2. Setup
        let needs = self.needs();
        let missing: Vec<&(&'static str, bool)> = needs.iter().filter(|(_, ok)| !*ok).collect();
        card(ui, "2 · Kurulum", |ui| {
            match self.gpu.load(Ordering::Relaxed) {
                0 => ui.weak("Ekran kartı aranıyor…"),
                1 => ui.colored_label(OK, "NVIDIA ekran kartı bulundu: hızlı GPU motoru önerilir."),
                _ => ui.colored_label(WARN, "NVIDIA ekran kartı bulunamadı: hafif modeller ve işlemci motoru önerilir."),
            };
            ui.add_space(4.0);
            for (id, ok) in &needs {
                let Some(c) = components::find(id) else { continue };
                ui.horizontal(|ui| {
                    ui.label(if *ok { "✔" } else { "•" });
                    ui.label(name_tr(c));
                    ui.weak(gb(c.size_mb));
                    if self.is_recommended(id) {
                        ui.colored_label(WARN, format!("{STAR} önerilen"));
                    }
                    if let Some(p) = self.jobs.get(id) {
                        match p.state() {
                            State::Failed(e) => {
                                ui.colored_label(BAD, format!("hata: {e}"));
                            }
                            State::Cancelled => {
                                ui.weak("durduruldu");
                            }
                            s => {
                                let txt = if s == State::Extracting { "açılıyor…".to_string() } else { format!("%{:.0}", p.fraction() * 100.0) };
                                ui.add(egui::ProgressBar::new(p.fraction()).desired_width(150.0).text(txt));
                                if ui.small_button("İptal").clicked() {
                                    p.cancel();
                                }
                            }
                        }
                    } else if *ok {
                        ui.colored_label(OK, "kurulu");
                    } else {
                        ui.colored_label(WARN, "indirilmedi");
                    }
                });
            }
            if !missing.is_empty() {
                ui.add_space(6.0);
                let total: u32 = missing.iter().filter_map(|(id, _)| components::find(id)).map(|c| c.size_mb).sum();
                if ui.button(RichText::new(format!("⬇  Eksikleri indir  ({})", gb(total))).size(16.0)).clicked() {
                    for (id, _) in &missing {
                        if !self.jobs.contains_key(id) {
                            self.start_download(id);
                        }
                    }
                }
                ui.weak("Dosyalar yalnızca bir kez indirilir. Başka modeller için Modeller sekmesine bak.");
            }
        });

        // 3. App
        card(ui, "3 · Çevrilecek uygulama", |ui| {
            ui.weak("Ses çalan uygulamayı seç (tarayıcı, oyun, oynatıcı…). Yalnızca onun sesi dinlenir.");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("ara…").desired_width(220.0));
                if ui.button("⟳ Yenile").clicked() {
                    self.windows = list_windows();
                }
            });
            let f = self.filter.to_lowercase();
            egui::ScrollArea::vertical().max_height(130.0).id_salt("wins").show(ui, |ui| {
                for w in self.windows.iter().filter(|w| f.is_empty() || w.title.to_lowercase().contains(&f)) {
                    let sel = self.selected.as_ref().map_or(false, |s| s.pid == w.pid && s.title == w.title);
                    let title: String = w.title.chars().take(90).collect();
                    if ui.selectable_label(sel, title).clicked() {
                        self.selected = Some(w.clone());
                    }
                }
            });
        });

        // Go
        if self.needs_restart() {
            ui.colored_label(WARN, "Ayarlar değişti: yeni ayarlar için yeniden başlat.");
        }
        ui.horizontal(|ui| {
            let can = self.selected.is_some() && missing.is_empty();
            let label = if self.overlay.is_some() { "↻  Yeniden başlat" } else { "▶  Çeviriyi başlat" };
            if ui.add_enabled(can, egui::Button::new(RichText::new(label).size(20.0)).min_size(egui::vec2(240.0, 44.0))).clicked() {
                self.start_overlay();
            }
            if self.overlay.is_some() && ui.add(egui::Button::new(RichText::new("■  Durdur").size(18.0)).min_size(egui::vec2(120.0, 44.0))).clicked() {
                self.stop_overlay();
                self.message = "Durduruldu.".into();
            }
        });
        if self.selected.is_none() {
            ui.weak("Önce bir uygulama seç.");
        } else if !missing.is_empty() {
            ui.weak("Önce eksik bileşenleri indir.");
        }
    }

    fn tab_models(&mut self, ui: &mut egui::Ui) {
        ui.heading("Modeller ve motorlar");
        ui.weak(format!("{STAR} = bilgisayarın ve seçtiğin dil için önerilen. Uygulamanın kendisi küçüktür, bunlar ihtiyaç olunca iner."));
        let data = self.settings.data_dir();
        ui.horizontal(|ui| {
            ui.weak(format!("Klasör: {}", data.display()));
            if ui.small_button("Aç").clicked() {
                let _ = std::fs::create_dir_all(&data);
                let _ = std::process::Command::new("explorer").arg(&data).spawn();
            }
        });
        for (kind, title) in [
            (Kind::SttServer, "Konuşma motoru"),
            (Kind::SttModel, "Konuşma tanıma modelleri"),
            (Kind::MtModel, "Çeviri modelleri"),
        ] {
            ui.add_space(8.0);
            ui.label(RichText::new(title).strong().size(16.0));
            ui.separator();
            for c in components::catalog().iter().filter(|c| c.kind == kind) {
                let installed = self.is_installed(c.id);
                let job = self.jobs.get(c.id).cloned();
                let rec = self.is_recommended(c.id);
                ui.horizontal(|ui| {
                    let name = if rec { format!("{STAR} {}", name_tr(c)) } else { name_tr(c) };
                    ui.label(if rec { RichText::new(name).strong().color(WARN) } else { RichText::new(name).strong() });
                    ui.weak(gb(c.size_mb));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match &job {
                        Some(p) => match p.state() {
                            State::Failed(e) => {
                                if ui.button("Tekrar dene").clicked() {
                                    self.start_download(c.id);
                                }
                                ui.colored_label(BAD, "hata").on_hover_text(&e);
                            }
                            State::Cancelled => {
                                if ui.button("Devam et").clicked() {
                                    self.start_download(c.id);
                                }
                            }
                            s => {
                                if ui.button("İptal").clicked() {
                                    p.cancel();
                                }
                                let txt = if s == State::Extracting { "açılıyor…".to_string() } else { format!("%{:.0}", p.fraction() * 100.0) };
                                ui.add(egui::ProgressBar::new(p.fraction()).desired_width(150.0).text(txt));
                            }
                        },
                        None if installed => {
                            if ui.button("Sil").clicked() {
                                let _ = components::remove(c, &data);
                                self.refresh_installed();
                            }
                            ui.colored_label(OK, "✔ kurulu");
                        }
                        None => {
                            if ui.button("⬇ İndir").clicked() {
                                self.start_download(c.id);
                            }
                        }
                    });
                });
                ui.weak(desc_tr(c));
            }
        }
    }

    fn tab_advanced(&mut self, ui: &mut egui::Ui) {
        ui.heading("Gelişmiş ayarlar");
        ui.weak("Çoğu kişi için varsayılanlar yeterli. Değişince çeviriyi yeniden başlat.");
        ui.add_space(6.0);

        egui::CollapsingHeader::new(RichText::new("Hız ve doğruluk").size(16.0)).default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                let s = &mut self.settings.speed;
                if ui.button("⚡ En hızlı").clicked() {
                    (s.partial_every_ms, s.end_silence_ms, s.max_utterance_s, s.read_cps) = (400, 300, 6, 30.0);
                }
                if ui.button(format!("Dengeli {STAR}")).clicked() {
                    *s = Default::default();
                }
                if ui.button("🎯 En doğru").clicked() {
                    (s.partial_every_ms, s.end_silence_ms, s.max_utterance_s, s.read_cps) = (900, 700, 12, 20.0);
                }
            });
            let s = &mut self.settings.speed;
            egui::Grid::new("speed").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                ui.label("Ara sonuç sıklığı");
                ui.add(egui::Slider::new(&mut s.partial_every_ms, 200..=2000).suffix(" ms"));
                ui.end_row();
                ui.label("Cümle sonu sessizliği");
                ui.add(egui::Slider::new(&mut s.end_silence_ms, 100..=1500).suffix(" ms"));
                ui.end_row();
                ui.label("En uzun parça");
                ui.add(egui::Slider::new(&mut s.max_utterance_s, 3..=30).suffix(" sn"));
                ui.end_row();
                ui.label("Konuşma algılama eşiği");
                ui.add(egui::Slider::new(&mut s.vad_threshold, 0.001..=0.05).logarithmic(true));
                ui.end_row();
                ui.label("Altyazı okuma hızı");
                ui.add(egui::Slider::new(&mut s.read_cps, 10.0..=50.0).suffix(" kar/sn"));
                ui.end_row();
            });
            ui.weak("Eşik: yüksek = müziği/sessiz sesleri yok say, düşük = kısık konuşmayı da yakala.");
        });

        egui::CollapsingHeader::new(RichText::new("Konuşma tanıma").size(16.0)).show(ui, |ui| {
            egui::Grid::new("speech").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                ui.label("Model");
                let cur = components::find(&self.settings.speech.model).map_or(self.settings.speech.model.clone(), name_tr);
                egui::ComboBox::from_id_salt("stt_model").width(340.0).selected_text(cur).show_ui(ui, |ui| {
                    for c in components::catalog().iter().filter(|c| c.kind == Kind::SttModel) {
                        let mark = if self.installed.get(c.id).copied().unwrap_or(false) { "✔" } else { "⬇" };
                        let star = if self.is_recommended(c.id) { STAR } else { " " };
                        ui.selectable_value(&mut self.settings.speech.model, c.id.to_string(), format!("{mark} {star} {} ({})", name_tr(c), gb(c.size_mb)));
                    }
                });
                ui.end_row();
                ui.label("Çalışma yeri");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.settings.speech.backend, Backend::Auto, format!("Otomatik {STAR}"));
                    ui.selectable_value(&mut self.settings.speech.backend, Backend::Cuda, "NVIDIA GPU");
                    ui.selectable_value(&mut self.settings.speech.backend, Backend::Cpu, "İşlemci");
                });
                ui.end_row();
                ui.label("İşlemci iş parçacığı");
                ui.add(egui::Slider::new(&mut self.settings.speech.threads, 0..=16).custom_formatter(|v, _| if v == 0.0 { "otomatik".into() } else { format!("{v}") }));
                ui.end_row();
            });
            if let Some(c) = components::find(&self.settings.speech.model) {
                ui.weak(desc_tr(c));
            }
            if self.settings.speech.model.contains(".en") && !self.english_source() {
                ui.colored_label(WARN, "Bu model yalnızca İngilizce bilir; başka dil için 'çok dilli' bir model seç.");
            }
        });

        egui::CollapsingHeader::new(RichText::new("Çeviri").size(16.0)).show(ui, |ui| {
            egui::Grid::new("tr").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                ui.label("Model");
                let cur = components::find(&self.settings.translation.model).map_or(self.settings.translation.model.clone(), name_tr);
                egui::ComboBox::from_id_salt("mt_model").width(340.0).selected_text(cur).show_ui(ui, |ui| {
                    for c in components::catalog().iter().filter(|c| c.kind == Kind::MtModel) {
                        let mark = if self.installed.get(c.id).copied().unwrap_or(false) { "✔" } else { "⬇" };
                        let star = if self.is_recommended(c.id) { STAR } else { " " };
                        ui.selectable_value(&mut self.settings.translation.model, c.id.to_string(), format!("{mark} {star} {} ({})", name_tr(c), gb(c.size_mb)));
                    }
                });
                ui.end_row();
                ui.label("Arama genişliği (beam)");
                ui.add(egui::Slider::new(&mut self.settings.translation.beam_size, 1..=6));
                ui.end_row();
                ui.label("İşlemci iş parçacığı");
                ui.add(egui::Slider::new(&mut self.settings.translation.threads, 0..=16).custom_formatter(|v, _| if v == 0.0 { "otomatik".into() } else { format!("{v}") }));
                ui.end_row();
            });
            ui.weak("Beam 1 en hızlı, 4-6 en doğru (2 önerilir).");
            if self.settings.translation.model == "opus-mt-en-tr" && !(self.english_source() && self.settings.translation.target == "tr") {
                ui.colored_label(WARN, "Bu model yalnızca İngilizce → Türkçe çevirir; başka diller için NLLB seç.");
            }
        });

        ui.add_space(10.0);
        if ui.button("Tüm ayarları varsayılana döndür").clicked() {
            self.settings = Settings::default();
            self.message = "Ayarlar sıfırlandı.".into();
        }
    }

    fn tab_display(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Live preview on a fake screen.
        let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 180.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 8.0, Color32::from_rgb(38, 52, 74));
        ui.painter().rect_filled(egui::Rect::from_min_max(rect.min, egui::pos2(rect.max.x, rect.min.y + rect.height() * 0.55)), 8.0, Color32::from_rgb(70, 96, 128));
        if self.preview.is_idle() && self.preview_at.elapsed() > Duration::from_millis(900) {
            self.preview.push(DEMO[self.preview_i % DEMO.len()], Some("Original sentence shown in small type."));
            self.preview_i += 1;
            self.preview_at = Instant::now();
        }
        self.preview.show(ui, rect.shrink2(egui::vec2(rect.width() * 0.06, 6.0)));
        ctx.request_repaint_after(self.preview.repaint_after());
        ui.add_space(8.0);

        let d = &mut self.settings.display;
        card(ui, "Stil", |ui| {
            ui.horizontal_wrapped(|ui| {
                for s in [Style::Classic, Style::Lyrics, Style::Karaoke, Style::Typewriter, Style::Pop, Style::Slide] {
                    let label = if s == Style::Classic { format!("{} {STAR}", style_name(s)) } else { style_name(s).to_string() };
                    ui.selectable_value(&mut d.style, s, RichText::new(label).size(15.0));
                }
            });
        });
        card(ui, "Yazı ve konum", |ui| {
            egui::Grid::new("display_basic").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                ui.label("Yazı boyutu");
                ui.add(egui::Slider::new(&mut d.font_size, 16.0..=96.0));
                ui.end_row();
                ui.label("Konum");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut d.position, Position::Top, "Üst");
                    ui.selectable_value(&mut d.position, Position::Center, "Orta");
                    ui.selectable_value(&mut d.position, Position::Bottom, format!("Alt {STAR}"));
                });
                ui.end_row();
            });
        });
        egui::CollapsingHeader::new(RichText::new("Gelişmiş görünüm").size(16.0)).show(ui, |ui| {
            egui::Grid::new("display_adv").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                ui.label("Renkler");
                ui.horizontal(|ui| {
                    ui.color_edit_button_srgba_unmultiplied(&mut d.text_color);
                    ui.label("yazı");
                    ui.color_edit_button_srgba_unmultiplied(&mut d.accent_color);
                    ui.label("vurgu");
                    ui.color_edit_button_srgba_unmultiplied(&mut d.bg_color);
                    ui.label("arka plan (alfa 0 = kutu yok)");
                });
                ui.end_row();
                ui.label("Yazı çerçevesi");
                ui.checkbox(&mut d.outline, "koyu çerçeve çiz");
                ui.end_row();
                ui.label("Genişlik");
                ui.add(egui::Slider::new(&mut d.width_pct, 20.0..=100.0).suffix(" %"));
                ui.end_row();
                ui.label("Kenar boşluğu");
                ui.add(egui::Slider::new(&mut d.margin_px, 0.0..=400.0).suffix(" px"));
                ui.end_row();
                ui.label("Bir seferde en çok");
                ui.add(egui::Slider::new(&mut d.max_chars, 20..=140).suffix(" karakter"));
                ui.end_row();
                ui.label("Animasyon hızı");
                ui.add(egui::Slider::new(&mut d.anim_speed, 0.25..=3.0).suffix("×"));
                ui.end_row();
                ui.label("Özgün metin");
                ui.checkbox(&mut d.show_original, "konuşulan dili küçük puntoyla da göster");
                ui.end_row();
            });
            if ui.button("Varsayılan görünüm").clicked() {
                *d = Default::default();
            }
        });
        ui.add_space(4.0);
        ui.weak("Görünüm değişiklikleri çalışan altyazıya anında uygulanır.");
    }
}
