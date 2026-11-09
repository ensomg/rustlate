//! Transparent, click-through subtitle overlay window (egui).

use crate::config::{self, Settings};
use crate::pipeline;
use crate::style::{self, Subtitles};
use eframe::egui;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_CONTROL, VK_SHIFT};
use windows::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

// ---------- Overlay ----------

#[derive(Default)]
struct Shared {
    /// New sentences from the pipeline (translated, original), drained by the UI every frame.
    incoming: Vec<(String, String)>,
    status: String,
}

struct Overlay {
    shared: Arc<Mutex<Shared>>,
    styled: bool,
    geometry_applied: bool,
    subs: Subtitles,
    settings: Settings,
    settings_mtime: Option<SystemTime>,
    last_check: Instant,
    demo: Option<Demo>,
}

/// Demo mode (`RUSTLATE_DEMO=1`): no audio, a loop of sample sentences every ~3 s.
struct Demo {
    next: usize,
    last: Option<Instant>,
}

const DEMO_LINES: &[(&str, &str)] = &[
    ("Tamam, bu gerçekten çılgınca.", "Okay, this is actually crazy."),
    (
        "Bu modül aslında onu atlıyor, ki bu delilik, çünkü geçen hafta beni iki dakikadan kısa sürede yasakladı.",
        "This module actually bypasses it, which is crazy, because last week it got me banned in under two minutes.",
    ),
    ("Açalım ve bu çiftliğin ne kadar para kazandığını görelim.", "Let's turn it on and see how much money this farm generates."),
    ("Evet!", "Yes!"),
    (
        "Açıkçası ilk denemede bunun işe yarayacağını beklemiyordum, ama işte buradayız.",
        "Honestly I did not expect this to work on the first try, but here we are.",
    ),
];

/// Make the window click-through. A layered window needs an explicit alpha or it is invisible.
fn make_click_through() -> bool {
    use windows::core::w;
    use windows::Win32::UI::WindowsAndMessaging::*;
    unsafe {
        let Ok(hwnd) = FindWindowW(None, w!("rustlate overlay")) else { return false };
        let ex = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        let ex = ex | WS_EX_LAYERED.0 | WS_EX_TRANSPARENT.0 | WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0;
        SetWindowLongW(hwnd, GWL_EXSTYLE, ex as i32);
        let _ = SetLayeredWindowAttributes(hwnd, windows::Win32::Foundation::COLORREF(0), 255, LWA_ALPHA);
        true
    }
}

fn screen_px() -> (f32, f32) {
    unsafe { (GetSystemMetrics(SM_CXSCREEN) as f32, GetSystemMetrics(SM_CYSCREEN) as f32) }
}

/// Moves/resizes the overlay window for the current display settings. Screen metrics are
/// physical pixels, egui viewport commands are logical points, hence the division.
fn apply_geometry(ctx: &egui::Context, d: &config::Display) {
    let (sw, sh) = screen_px();
    let (w, h, x, y) = style::window_geometry(d, sw, sh);
    let ppp = ctx.pixels_per_point().max(0.5);
    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w / ppp, h / ppp)));
    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x / ppp, y / ppp)));
}

fn settings_mtime() -> Option<SystemTime> {
    std::fs::metadata(Settings::path()).and_then(|m| m.modified()).ok()
}

impl eframe::App for Overlay {
    fn clear_color(&self, _: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if !self.styled {
            self.styled = make_click_through();
        }
        let down = |vk: windows::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY| unsafe { GetAsyncKeyState(vk.0 as i32) < 0 };
        if down(VK_CONTROL) && down(VK_SHIFT) && down(windows::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY(b'Q' as u16)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if !self.geometry_applied {
            apply_geometry(ctx, &self.settings.display);
            self.geometry_applied = true;
        }

        // Hot reload: the settings window (another process) writes settings.toml.
        if self.last_check.elapsed() >= Duration::from_secs(1) {
            self.last_check = Instant::now();
            let m = settings_mtime();
            if m != self.settings_mtime {
                self.settings_mtime = m;
                if m.is_some() {
                    self.settings = Settings::load();
                    self.subs.set_settings(&self.settings.display, &self.settings.speed);
                    apply_geometry(ctx, &self.settings.display);
                }
            }
        }

        if let Some(demo) = &mut self.demo {
            if demo.last.map_or(true, |t| t.elapsed() >= Duration::from_secs(3)) {
                let (tr, en) = DEMO_LINES[demo.next % DEMO_LINES.len()];
                self.subs.push(tr, Some(en));
                demo.next += 1;
                demo.last = Some(Instant::now());
            }
        }

        let status = {
            let mut s = self.shared.lock().unwrap();
            let show_original = self.settings.display.show_original;
            for (tr, orig) in s.incoming.drain(..) {
                self.subs.push(&tr, show_original.then_some(orig.as_str()));
            }
            s.status.clone()
        };
        self.subs.set_status(if status.is_empty() { None } else { Some(status) });
        ctx.request_repaint_after(self.subs.repaint_after());

        let subs = &mut self.subs;
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
            let rect = ui.max_rect();
            subs.show(ui, rect);
        });
    }
}

/// Runs the overlay with the settings currently on disk.
pub fn run_overlay(pid: u32) -> anyhow::Result<()> {
    run_overlay_with(pid, Settings::load())
}

pub fn run_overlay_with(pid: u32, settings: Settings) -> anyhow::Result<()> {
    let demo = std::env::var("RUSTLATE_DEMO").map_or(false, |v| v == "1");
    let (sw, sh) = screen_px();
    let (w, h, x, y) = style::window_geometry(&settings.display, sw, sh);
    let status = if demo { String::new() } else { "Dinleniyor… (modeller yükleniyor)".to_string() };
    let shared = Arc::new(Mutex::new(Shared { status, ..Default::default() }));

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_transparent(std::env::var("RUSTLATE_OPAQUE").is_err())
            .with_decorations(false)
            .with_always_on_top()
            .with_taskbar(false)
            .with_inner_size([w, h])
            .with_position([x, y]),
        ..Default::default()
    };
    let sh2 = shared.clone();
    eframe::run_native(
        "rustlate overlay",
        opts,
        Box::new(move |cc| {
            crate::fonts::install(&cc.egui_ctx);
            let ctx = cc.egui_ctx.clone();
            if !demo {
                let pipeline_settings = settings.clone();
                std::thread::spawn(move || {
                    let (s_ready, c_ready) = (sh2.clone(), ctx.clone());
                    let (s3, c2) = (sh2.clone(), ctx.clone());
                    sh2.lock().unwrap().status = "Modeller yükleniyor…".into();
                    ctx.request_repaint();
                    let r = pipeline::run_with(
                        pid,
                        &pipeline_settings,
                        move || {
                            s_ready.lock().unwrap().status = "Dinleniyor…".into();
                            c_ready.request_repaint();
                        },
                        move |tr, orig, _final, _secs| {
                            s3.lock().unwrap().incoming.push((tr.to_string(), orig.to_string()));
                            c2.request_repaint();
                        },
                    );
                    if let Err(e) = r {
                        sh2.lock().unwrap().status = format!("Hata: {e}");
                        ctx.request_repaint();
                    }
                });
            }
            Ok(Box::new(Overlay {
                shared,
                styled: false,
                geometry_applied: false,
                subs: Subtitles::new(&settings.display, &settings.speed),
                settings_mtime: settings_mtime(),
                settings,
                last_check: Instant::now(),
                demo: demo.then_some(Demo { next: 0, last: None }),
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
