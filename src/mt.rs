//! Machine translation through CTranslate2 (`ct2rs`): the fixed-pair Opus-MT model and the
//! multilingual NLLB-200 model (any language pair from [`languages`]).

use anyhow::{anyhow, bail, Result};
use ct2rs::tokenizers::{auto, hf};
use ct2rs::{Config, Device, TranslationOptions, Translator};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtKind {
    /// Single-pair Opus-MT model with its own SentencePiece tokenizer (language pair is baked in).
    Opus,
    /// NLLB-200: needs a source language token and a target language prefix.
    Nllb,
}

#[derive(Clone, Debug)]
pub struct MtSpec {
    pub dir: PathBuf,
    pub kind: MtKind,
    /// ISO 639-1 codes. Ignored for `Opus` (the pair is fixed by the model).
    pub source: String,
    pub target: String,
    pub beam_size: usize,
    /// 0 = automatic.
    pub threads: usize,
}

pub struct Lang {
    pub iso: &'static str,
    pub name: &'static str,
    /// NLLB-200 language token, e.g. "tur_Latn".
    pub nllb: &'static str,
}

macro_rules! langs {
    ($(($iso:expr, $name:expr, $nllb:expr)),* $(,)?) => {
        &[$(Lang { iso: $iso, name: $name, nllb: $nllb }),*]
    };
}

static LANGUAGES: &[Lang] = langs![
    ("en", "English", "eng_Latn"),
    ("tr", "Türkçe", "tur_Latn"),
    ("de", "Deutsch", "deu_Latn"),
    ("fr", "Français", "fra_Latn"),
    ("es", "Español", "spa_Latn"),
    ("it", "Italiano", "ita_Latn"),
    ("pt", "Português", "por_Latn"),
    ("ru", "Русский", "rus_Cyrl"),
    ("ar", "العربية", "arb_Arab"),
    ("zh", "中文 (简体)", "zho_Hans"),
    ("ja", "日本語", "jpn_Jpan"),
    ("ko", "한국어", "kor_Hang"),
    ("hi", "हिन्दी", "hin_Deva"),
    ("nl", "Nederlands", "nld_Latn"),
    ("pl", "Polski", "pol_Latn"),
    ("uk", "Українська", "ukr_Cyrl"),
    ("sv", "Svenska", "swe_Latn"),
    ("da", "Dansk", "dan_Latn"),
    ("fi", "Suomi", "fin_Latn"),
    ("no", "Norsk", "nob_Latn"),
    ("cs", "Čeština", "ces_Latn"),
    ("sk", "Slovenčina", "slk_Latn"),
    ("hu", "Magyar", "hun_Latn"),
    ("ro", "Română", "ron_Latn"),
    ("bg", "Български", "bul_Cyrl"),
    ("el", "Ελληνικά", "ell_Grek"),
    ("he", "עברית", "heb_Hebr"),
    ("fa", "فارسی", "pes_Arab"),
    ("id", "Bahasa Indonesia", "ind_Latn"),
    ("ms", "Bahasa Melayu", "zsm_Latn"),
    ("vi", "Tiếng Việt", "vie_Latn"),
    ("th", "ไทย", "tha_Thai"),
    ("bn", "বাংলা", "ben_Beng"),
    ("ur", "اردو", "urd_Arab"),
    ("ta", "தமிழ்", "tam_Taml"),
    ("te", "తెలుగు", "tel_Telu"),
    ("sr", "Српски", "srp_Cyrl"),
    ("hr", "Hrvatski", "hrv_Latn"),
    ("sl", "Slovenščina", "slv_Latn"),
    ("lt", "Lietuvių", "lit_Latn"),
    ("lv", "Latviešu", "lvs_Latn"),
    ("et", "Eesti", "est_Latn"),
    ("ca", "Català", "cat_Latn"),
    ("gl", "Galego", "glg_Latn"),
    ("eu", "Euskara", "eus_Latn"),
    ("af", "Afrikaans", "afr_Latn"),
    ("sw", "Kiswahili", "swh_Latn"),
    ("az", "Azərbaycanca", "azj_Latn"),
    ("kk", "Қазақша", "kaz_Cyrl"),
    ("uz", "Oʻzbekcha", "uzn_Latn"),
    ("ka", "ქართული", "kat_Geor"),
    ("hy", "Հայերեն", "hye_Armn"),
    ("sq", "Shqip", "als_Latn"),
    ("mk", "Македонски", "mkd_Cyrl"),
    ("bs", "Bosanski", "bos_Latn"),
    ("is", "Íslenska", "isl_Latn"),
    ("tl", "Filipino", "tgl_Latn"),
    ("cy", "Cymraeg", "cym_Latn"),
];

pub fn languages() -> &'static [Lang] {
    LANGUAGES
}

pub fn nllb_code(iso: &str) -> Option<&'static str> {
    LANGUAGES.iter().find(|l| l.iso.eq_ignore_ascii_case(iso)).map(|l| l.nllb)
}

/// NLLB tokenizer: `tokenizer.json` pieces, `[src_lang] tokens </s>` on the way in,
/// special tokens (`</s>`, language codes) skipped on the way out.
struct NllbTokenizer {
    hf: hf::Tokenizer,
    src_token: String,
}

impl ct2rs::Tokenizer for NllbTokenizer {
    fn encode(&self, input: &str) -> Result<Vec<String>> {
        // `(*self.hf)` = the inner `tokenizers::Tokenizer`; false = no template-added special tokens,
        // because tokenizer.json hard-codes the English source token.
        let enc = (*self.hf).encode(input, false).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut out = Vec::with_capacity(enc.get_tokens().len() + 2);
        out.push(self.src_token.clone());
        out.extend(enc.get_tokens().iter().cloned());
        out.push("</s>".to_string());
        Ok(out)
    }

    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let ids: Vec<u32> = tokens.iter().filter_map(|t| self.hf.token_to_id(t)).collect();
        (*self.hf).decode(&ids, true).map_err(|e| anyhow!("detokenizer: {e}"))
    }
}

enum Engine {
    Opus(Translator<auto::Tokenizer>),
    Nllb { tr: Translator<NllbTokenizer>, target_prefix: Vec<Vec<String>> },
}

pub struct Mt {
    engine: Engine,
    beam_size: usize,
    max_decoding_length: usize,
}

fn thread_count(requested: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism().map_or(4, |n| n.get()).min(8)
    }
}

impl Mt {
    /// Backwards compatible: the Opus-MT EN->TR model, beam 2, 8 threads.
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        Self::with_spec(&MtSpec {
            dir: model_dir.as_ref().to_path_buf(),
            kind: MtKind::Opus,
            source: "en".into(),
            target: "tr".into(),
            beam_size: 2,
            threads: 8,
        })
    }

    pub fn with_spec(spec: &MtSpec) -> Result<Self> {
        let cfg = Config {
            device: Device::CPU,
            num_threads_per_replica: thread_count(spec.threads),
            ..Default::default()
        };
        let beam_size = spec.beam_size.max(1);
        let engine = match spec.kind {
            MtKind::Opus => Engine::Opus(Translator::new(&spec.dir, &cfg)?),
            MtKind::Nllb => {
                let src = nllb_code(&spec.source).ok_or_else(|| anyhow!("unsupported source language {}", spec.source))?;
                let dst = nllb_code(&spec.target).ok_or_else(|| anyhow!("unsupported target language {}", spec.target))?;
                if src == dst {
                    bail!("source and target language are the same ({})", spec.source);
                }
                let tokenizer = NllbTokenizer { hf: hf::Tokenizer::new(&spec.dir)?, src_token: src.to_string() };
                Engine::Nllb {
                    tr: Translator::with_tokenizer(&spec.dir, tokenizer, &cfg)?,
                    target_prefix: vec![vec![dst.to_string()]],
                }
            }
        };
        Ok(Self { engine, beam_size, max_decoding_length: 128 })
    }

    pub fn translate(&self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let opts = TranslationOptions::<String, String> {
            beam_size: self.beam_size,
            max_decoding_length: self.max_decoding_length,
            ..Default::default()
        };
        let out = match &self.engine {
            Engine::Opus(tr) => tr.translate_batch(&[text], &opts, None)?,
            Engine::Nllb { tr, target_prefix } => {
                tr.translate_batch_with_target_prefix(&[text], target_prefix, &opts, None)?
            }
        };
        Ok(out.into_iter().next().map(|(s, _)| s.trim().to_string()).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn language_table_is_consistent() {
        let mut iso = HashSet::new();
        let mut codes = HashSet::new();
        for l in languages() {
            assert!(iso.insert(l.iso), "duplicate iso {}", l.iso);
            assert!(codes.insert(l.nllb), "duplicate nllb code {}", l.nllb);
            let (lang, script) = l.nllb.split_once('_').expect("xxx_Yyyy");
            assert_eq!(lang.len(), 3, "{}", l.nllb);
            assert_eq!(script.len(), 4, "{}", l.nllb);
            assert!(!l.name.is_empty());
        }
        assert!(languages().len() >= 40);
    }

    #[test]
    fn code_lookup() {
        assert_eq!(nllb_code("tr"), Some("tur_Latn"));
        assert_eq!(nllb_code("EN"), Some("eng_Latn"));
        assert_eq!(nllb_code("xx"), None);
    }

    /// Every language token must exist in the real NLLB vocabulary (skipped if not downloaded).
    #[test]
    fn language_codes_exist_in_nllb_vocab() {
        let dir = Path::new("models/nllb-200-600m");
        let Ok(tok) = hf::Tokenizer::new(dir) else { return };
        for l in languages() {
            assert!(tok.token_to_id(l.nllb).is_some(), "{} missing from NLLB vocab", l.nllb);
        }
    }

    #[test]
    fn same_language_is_rejected() {
        let spec = MtSpec {
            dir: PathBuf::from("models/nllb-200-600m"),
            kind: MtKind::Nllb,
            source: "tr".into(),
            target: "tr".into(),
            beam_size: 2,
            threads: 0,
        };
        assert!(Mt::with_spec(&spec).is_err());
    }
}
