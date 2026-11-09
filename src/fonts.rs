//! Fallback fonts from the Windows font folder, so subtitles and menus can show Arabic, Hebrew,
//! Chinese, Japanese, Korean, Thai and Indic scripts. Nothing is bundled (keeps the exe small);
//! fonts that are not installed are skipped.

use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;

/// (key, file name in %WINDIR%\Fonts), most useful first.
const CANDIDATES: [(&str, &str); 9] = [
    ("segoeui", "segoeui.ttf"),     // Latin ext, Cyrillic, Greek, Arabic, Hebrew, Armenian, Georgian
    ("msyh", "msyh.ttc"),           // Chinese (also most kanji)
    ("yugoth", "YuGothR.ttc"),      // Japanese kana
    ("malgun", "malgun.ttf"),       // Korean
    ("nirmala", "Nirmala.ttf"),     // Hindi, Bengali, Tamil, Telugu, Urdu ...
    ("leelawui", "LeelawUI.ttf"),   // Thai
    ("seguisym", "seguisym.ttf"),   // symbols
    ("msjh", "msjh.ttc"),           // Traditional Chinese
    ("mingliub", "mingliu.ttc"),    // fallback CJK
];

pub fn install(ctx: &egui::Context) {
    let dir = std::env::var_os("WINDIR").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("C:/Windows")).join("Fonts");
    let mut fonts = egui::FontDefinitions::default();
    let mut added = 0;
    for (key, file) in CANDIDATES {
        let Ok(bytes) = std::fs::read(dir.join(file)) else { continue };
        fonts.font_data.insert(key.to_string(), Arc::new(egui::FontData::from_owned(bytes)));
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            if let Some(list) = fonts.families.get_mut(&family) {
                list.push(key.to_string());
            }
        }
        added += 1;
    }
    if added > 0 {
        ctx.set_fonts(fonts);
    }
}
