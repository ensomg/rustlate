//! capture -> chunker -> speech server -> sentence-cached translation -> callback.
//! Everything tunable comes from `config::Settings`; models and the speech engine come from the
//! downloaded components (components.rs).

use crate::config::Settings;
use crate::mt::{Mt, MtKind, MtSpec};
use crate::{audio, components, live, stt_server::SttServer};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

// Default model folder of the developer CLI (`rustlate mt`).
pub const MT_MODEL: &str = "models/opus-en-tr";

/// Sentence end marks that need no following space (CJK, Arabic, Devanagari).
const WIDE_ENDS: [char; 5] = ['。', '！', '？', '؟', '।'];

/// Splits on . ? ! followed by whitespace/end (so "2.5" stays whole) and on wide-script marks.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        cur.push(c);
        let ascii_end = matches!(c, '.' | '?' | '!') && it.peek().map_or(true, |n| n.is_whitespace());
        if ascii_end || WIDE_ENDS.contains(&c) {
            out.push(std::mem::take(&mut cur).trim().to_string());
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Ends with a sentence mark but is not an ellipsis (whisper writes "..." for trailing-off audio).
fn terminal(s: &str) -> bool {
    let s = s.trim_end();
    let marked = s.ends_with(['.', '?', '!']) || s.ends_with(WIDE_ENDS);
    marked && !s.ends_with("..") && !s.ends_with('…')
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// If `e` (already shown) is the beginning of the longer sentence `s`, returns the rest of `s`.
fn remainder_after(e: &str, s: &str) -> Option<String> {
    let (we, ws) = (words(e), words(s));
    if we.len() < 3 || ws.len() <= we.len() {
        return None;
    }
    let same = we.iter().zip(&ws).filter(|(a, b)| a == b).count();
    if (same as f32) / (we.len() as f32) < 0.85 {
        return None;
    }
    // Byte offset in `s` just after its `we.len()`-th word.
    let (mut count, mut in_word, mut end) = (0usize, false, s.len());
    for (i, c) in s.char_indices() {
        if c.is_alphanumeric() {
            if !in_word {
                in_word = true;
                count += 1;
            }
        } else if in_word {
            in_word = false;
            if count == we.len() {
                end = i;
                break;
            }
        }
    }
    let rest = s[end..].trim_start_matches(|c: char| !c.is_alphanumeric()).trim();
    Some(rest.to_string())
}

/// What still needs to be shown for sentence `s`: None = nothing new, Some(text) = this text.
fn novelty(emitted: &[String], s: &str) -> Option<String> {
    for e in emitted {
        if let Some(rest) = remainder_after(e, s) {
            return if rest.is_empty() { None } else { Some(rest) };
        }
    }
    if emitted.iter().any(|e| similar(e, s)) {
        return None;
    }
    Some(s.to_string())
}

/// Same sentence up to small whisper differences (word-set overlap >= 60 %).
fn similar(a: &str, b: &str) -> bool {
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() || wb.is_empty() {
        return false;
    }
    let common = wa.iter().filter(|w| wb.contains(w)).count() as f32;
    common / wa.len().max(wb.len()) as f32 >= 0.6
}

/// Translator with a per-sentence cache; `None` mt = source and target language are the same.
struct CachedMt {
    mt: Option<Mt>,
    cache: HashMap<String, String>,
}

impl CachedMt {
    fn open(s: &Settings) -> Result<Self> {
        if s.translation.source == s.translation.target {
            return Ok(Self { mt: None, cache: HashMap::new() });
        }
        let id = &s.translation.model;
        let dir = components::mt_model_dir(id, &s.data_dir())
            .ok_or_else(|| anyhow!("Çeviri modeli kurulu değil: {id}. Ayarlar > Bileşenler bölümünden indirin."))?;
        let kind = if id.starts_with("nllb") { MtKind::Nllb } else { MtKind::Opus };
        let mt = Mt::with_spec(&MtSpec {
            dir,
            kind,
            source: s.translation.source.clone(),
            target: s.translation.target.clone(),
            beam_size: s.translation.beam_size.max(1) as usize,
            threads: s.translation.threads as usize,
        })?;
        Ok(Self { mt: Some(mt), cache: HashMap::new() })
    }

    /// Only sentences not seen before hit the translator.
    fn translate_one(&mut self, s: &str) -> Result<String> {
        let Some(mt) = &self.mt else { return Ok(s.to_string()) };
        if let Some(t) = self.cache.get(s) {
            return Ok(t.clone());
        }
        let t = mt.translate(s)?;
        if self.cache.len() > 500 {
            self.cache.clear();
        }
        self.cache.insert(s.to_string(), t.clone());
        Ok(t)
    }
}

/// The speech engine: a whisper.cpp server child process (CPU or CUDA, per settings).
pub struct Engine(SttServer);

impl Engine {
    pub fn new(s: &Settings) -> Result<Self> {
        let data = s.data_dir();
        let exe = components::stt_server_exe(s.speech.backend, &data)
            .ok_or_else(|| anyhow!("Konuşma motoru kurulu değil. Ayarlar > Bileşenler bölümünden indirin."))?;
        let model = components::stt_model_path(&s.speech.model, &data)
            .ok_or_else(|| anyhow!("Konuşma modeli kurulu değil: {}. Ayarlar > Bileşenler bölümünden indirin.", s.speech.model))?;
        // English-only models cannot do anything else; "auto" only works with multilingual ones.
        let english_only = s.speech.model.contains(".en");
        let lang = if english_only { "en" } else { s.speech.language.as_str() };
        Ok(Self(SttServer::start(&exe, &model, s.speech.threads, lang)?))
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        Ok(clean(&self.0.transcribe(samples)?))
    }
}

/// Drops whisper's non-speech tags such as [BLANK_AUDIO] or (music).
fn clean(text: &str) -> String {
    let t = text.trim();
    let tagged = (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')'));
    if tagged { String::new() } else { t.to_string() }
}

/// Old signature, settings read from disk: `on_text(one_sentence, is_final, seconds_taken)`.
pub fn run(pid: u32, mut on_text: impl FnMut(&str, bool, f32)) -> Result<()> {
    run_with(pid, &Settings::load(), || {}, |tr, _orig, fin, secs| on_text(tr, fin, secs))
}

/// Blocks forever. `on_ready` fires once models are loaded and listening has started;
/// `on_sentence(translated, original, is_final, seconds_taken)` gets each finished sentence once.
pub fn run_with(
    pid: u32,
    settings: &Settings,
    on_ready: impl FnOnce(),
    mut on_sentence: impl FnMut(&str, &str, bool, f32),
) -> Result<()> {
    let stt = Engine::new(settings)?;
    let mut mt = CachedMt::open(settings)?;
    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let _h = audio::capture::start(pid, tx, stop)?;
    on_ready();
    let mut chunker = live::Chunker::new(&settings.speed);
    let mut emitted: Vec<String> = Vec::new(); // source sentences already shown this utterance
    let mut prev_last: Option<String> = None; // last sentence of the previous pass, if it looked finished

    while let Ok(chunk) = rx.recv() {
        let mut events = chunker.push(&chunk);
        while let Ok(more) = rx.try_recv() {
            events.extend(chunker.push(&more));
        }
        // Keep every Final; of the Partials keep only the newest one after the last Final.
        let last_final = events.iter().rposition(|e| matches!(e, live::Event::Final(_)));
        let last_partial = events.iter().rposition(|e| matches!(e, live::Event::Partial(_)));
        let events = events.into_iter().enumerate().filter(|(i, e)| match e {
            live::Event::Final(_) => true,
            live::Event::Partial(_) => Some(*i) == last_partial && last_final.map_or(true, |f| *i > f),
        });
        for (_, ev) in events {
            let (buf, is_final) = match ev {
                live::Event::Partial(b) => (b, false),
                live::Event::Final(b) => (b, true),
            };
            let t = Instant::now();
            let text = stt.transcribe(&buf)?;
            if text.is_empty() {
                continue;
            }
            // Whisper puts a period on the end of ANY audio, even a cut-off one, so the last
            // sentence of a partial is only trusted once it is unchanged in two passes in a row.
            // Sentences followed by more text are complete. At an utterance end all are flushed.
            let sentences = split_sentences(&text);
            let last_stable = !is_final
                && sentences.last().map_or(false, |l| terminal(l) && prev_last.as_deref().map_or(false, |p| words(p) == words(l)));
            let n_ready = if is_final || last_stable { sentences.len() } else { sentences.len().saturating_sub(1) };
            for s in &sentences[..n_ready] {
                let Some(new_text) = novelty(&emitted, s) else { continue };
                let tr = mt.translate_one(&new_text)?;
                on_sentence(&tr, &new_text, is_final, t.elapsed().as_secs_f32());
                // The full sentence replaces any fragment of it shown earlier.
                emitted.retain(|e| remainder_after(e, s).is_none());
                emitted.push(s.clone());
            }
            prev_last = sentences.last().filter(|l| terminal(l)).cloned();
            if is_final {
                emitted.clear();
                prev_last = None;
            }
        }
    }
    Ok(())
}

/// Offline simulation: feed a 16 kHz mono wav through chunker -> STT -> MT in 100 ms steps.
/// Prints each Final so dropped words are easy to spot.
pub fn run_wav(path: &str) -> Result<()> {
    let samples: Vec<f32> = hound::WavReader::open(path)?
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
        .collect::<Result<_, _>>()?;
    let settings = Settings::load();
    let stt = Engine::new(&settings)?;
    let mut mt = CachedMt::open(&settings)?;
    let mut chunker = live::Chunker::new(&settings.speed);
    let mut chunks: Vec<&[f32]> = samples.chunks(1600).collect();
    let silence = vec![0.0f32; 16_000];
    chunks.push(&silence);
    for c in chunks {
        for ev in chunker.push(c) {
            if let live::Event::Final(b) = ev {
                let en = stt.transcribe(&b)?;
                let tr = split_sentences(&en).iter().map(|s| mt.translate_one(s)).collect::<Result<Vec<_>>>()?.join(" ");
                println!("[{:.1}s] SRC: {en}\n        DST: {tr}", b.len() as f32 / 16000.0);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{novelty, similar, split_sentences, terminal};

    #[test]
    fn fragment_then_full_sentence_only_shows_the_rest() {
        let shown = vec!["This module actually bypasses it.".to_string()];
        assert_eq!(
            novelty(&shown, "This module actually bypasses it, which is crazy.").as_deref(),
            Some("which is crazy.")
        );
        assert_eq!(novelty(&shown, "This module actually bypasses it."), None);
        assert!(novelty(&shown, "Let's turn it on.").is_some());
    }

    #[test]
    fn ellipsis_is_not_terminal_and_similarity_works() {
        assert!(terminal("It works.") && terminal("Really?"));
        assert!(!terminal("So I just joined...") && !terminal("And then"));
        assert!(similar("So I just joined the server.", "So I just joined the server"));
        assert!(!similar("Let's turn it on.", "I did not expect this."));
    }

    #[test]
    fn splits_but_keeps_decimals() {
        assert_eq!(split_sentences("Hi there. Cost is 2.5 now! Ok"), vec!["Hi there.", "Cost is 2.5 now!", "Ok"]);
    }

    #[test]
    fn wide_script_marks_split_without_spaces() {
        assert_eq!(split_sentences("今日は晴れです。明日は雨？はい！"), vec!["今日は晴れです。", "明日は雨？", "はい！"]);
        assert!(terminal("これはテストです。") && terminal("هل هذا صحيح؟"));
    }
}
