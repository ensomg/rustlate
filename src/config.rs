//! User settings, stored as TOML in %APPDATA%\rustlate\settings.toml.
//! Every tunable of the app lives here so the settings window can edit all of it.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct Settings {
    pub general: General,
    pub speech: Speech,
    pub translation: Translation,
    pub speed: Speed,
    pub display: Display,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct General {
    /// Where downloaded components (models, GPU server) live. None = default data dir.
    pub data_dir: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Backend {
    /// CUDA server if installed, otherwise CPU server.
    #[default]
    Auto,
    Cuda,
    Cpu,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Speech {
    /// Component id of the whisper model, e.g. "whisper-small.en" (see components.rs).
    pub model: String,
    pub backend: Backend,
    /// CPU threads for the speech server. 0 = automatic.
    pub threads: u32,
    /// Spoken language, ISO 639-1 ("en", "de", ...). "auto" = detect (multilingual models only).
    pub language: String,
}

impl Default for Speech {
    fn default() -> Self {
        Self { model: "whisper-small.en".into(), backend: Backend::Auto, threads: 0, language: "en".into() }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Translation {
    /// Component id of the translation model, e.g. "nllb-200-600m" or "opus-mt-en-tr".
    pub model: String,
    /// Source / target language, ISO 639-1.
    pub source: String,
    pub target: String,
    pub beam_size: u32,
    /// CPU threads for translation. 0 = automatic.
    pub threads: u32,
}

impl Default for Translation {
    fn default() -> Self {
        Self { model: "opus-mt-en-tr".into(), source: "en".into(), target: "tr".into(), beam_size: 2, threads: 0 }
    }
}

/// Everything that trades latency against accuracy.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Speed {
    /// How often a partial transcript is produced while someone is speaking.
    pub partial_every_ms: u32,
    /// Silence that ends an utterance.
    pub end_silence_ms: u32,
    /// Longest utterance before a forced split at the quietest point.
    pub max_utterance_s: u32,
    /// Speech detection threshold (RMS, 0.001 - 0.05).
    pub vad_threshold: f32,
    /// Subtitle reading speed in characters per second.
    pub read_cps: f32,
}

impl Default for Speed {
    fn default() -> Self {
        Self { partial_every_ms: 600, end_silence_ms: 440, max_utterance_s: 8, vad_threshold: 0.008, read_cps: 24.0 }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Style {
    /// Rounded dark box, fade + slide.
    #[default]
    Classic,
    /// Large centered lyric lines, current line bright, previous line dimmed above it.
    Lyrics,
    /// Words light up one by one across the line.
    Karaoke,
    /// Characters appear one by one.
    Typewriter,
    /// Scale-in pop with overshoot.
    Pop,
    /// Slides in from the side.
    Slide,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Position {
    #[default]
    Bottom,
    Center,
    Top,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Display {
    pub style: Style,
    pub font_size: f32,
    /// RGBA, 0-255.
    pub text_color: [u8; 4],
    /// Colour used for the active word / current lyric line (Karaoke, Lyrics).
    pub accent_color: [u8; 4],
    /// RGBA of the background box. Alpha 0 = no box.
    pub bg_color: [u8; 4],
    /// Dark outline around the text (helps when there is no box).
    pub outline: bool,
    /// Overlay width as a percentage of the screen width.
    pub width_pct: f32,
    pub position: Position,
    /// Distance from the screen edge in pixels.
    pub margin_px: f32,
    /// Longest piece of text shown at once, in characters.
    pub max_chars: u32,
    /// Animation speed multiplier (0.5 slow ... 2.0 fast).
    pub anim_speed: f32,
    /// Also show the original (source language) text in small type.
    pub show_original: bool,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            style: Style::Classic,
            font_size: 32.0,
            text_color: [255, 255, 255, 255],
            accent_color: [255, 214, 64, 255],
            bg_color: [0, 0, 0, 185],
            outline: false,
            width_pct: 70.0,
            position: Position::Bottom,
            margin_px: 80.0,
            max_chars: 56,
            anim_speed: 1.0,
            show_original: false,
        }
    }
}

fn app_dir(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(fallback)).join("rustlate")
}

impl Settings {
    pub fn path() -> PathBuf {
        app_dir("APPDATA", ".").join("settings.toml")
    }

    /// Loads settings; a missing or broken file gives defaults (a broken one is left untouched).
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path()).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let p = Self::path();
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(p, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Folder for downloaded components.
    pub fn data_dir(&self) -> PathBuf {
        self.general.data_dir.clone().unwrap_or_else(|| app_dir("LOCALAPPDATA", ".").join("components"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_partial_files() {
        let s = Settings::default();
        let text = toml::to_string_pretty(&s).unwrap();
        assert_eq!(toml::from_str::<Settings>(&text).unwrap(), s);
        // Old/partial files must still load, missing keys fall back to defaults.
        let p: Settings = toml::from_str("[display]\nfont_size = 40.0\n").unwrap();
        assert_eq!(p.display.font_size, 40.0);
        assert_eq!(p.speed, Speed::default());
    }
}
