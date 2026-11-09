//! Downloadable components: speech models, the whisper server (CPU / CUDA) and translation models.
//! The app itself stays small; everything big is fetched on demand into `Settings::data_dir()`.

#![allow(dead_code)] // the public API is wider than what the UI uses today

use crate::config::Backend;
use anyhow::{anyhow, bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    SttModel,
    MtModel,
    SttServer,
}

enum Source {
    /// One file, stored as `<data>/models/<file>`.
    File { url: &'static str, file: &'static str },
    /// A zip, extracted into `<data>/servers/<id>/`.
    Zip { url: &'static str },
    /// Several files of a Hugging Face repo, stored in `<data>/mt/<id>/`.
    HfRepo { repo: &'static str, files: &'static [&'static str] },
    /// Only available through `import_existing` (no public download).
    LocalOnly,
}

pub struct Component {
    pub id: &'static str,
    pub kind: Kind,
    pub name: &'static str,
    pub desc: &'static str,
    pub size_mb: u32,
    /// "en" = English only, "multi" = many languages. Servers/translation models: "multi" or "-".
    pub langs: &'static str,
    src: Source,
}

const HF: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
const GH: &str = "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/";

macro_rules! stt {
    ($id:expr, $file:expr, $mb:expr, $langs:expr, $name:expr, $desc:expr) => {
        Component {
            id: $id,
            kind: Kind::SttModel,
            name: $name,
            desc: $desc,
            size_mb: $mb,
            langs: $langs,
            src: Source::File { url: concat!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/", $file), file: $file },
        }
    };
}

static CATALOG: &[Component] = &[
    stt!("whisper-tiny.en", "ggml-tiny.en.bin", 74, "en", "Whisper tiny (English)", "Fastest, lowest accuracy. Fine on any CPU, only for testing or very weak PCs."),
    stt!("whisper-tiny.en-q5_1", "ggml-tiny.en-q5_1.bin", 31, "en", "Whisper tiny (English, small file)", "Compressed tiny. Smallest download."),
    stt!("whisper-base.en", "ggml-base.en.bin", 141, "en", "Whisper base (English)", "Fast, so-so accuracy. Usable live on a CPU (about 1 s delay)."),
    stt!("whisper-base.en-q5_1", "ggml-base.en-q5_1.bin", 57, "en", "Whisper base (English, small file)", "Compressed base: smaller and a bit faster."),
    stt!("whisper-small.en", "ggml-small.en.bin", 465, "en", "Whisper small (English)", "Good accuracy. Real-time on a GPU, too slow on most CPUs."),
    stt!("whisper-small.en-q5_1", "ggml-small.en-q5_1.bin", 181, "en", "Whisper small (English, small file)", "Compressed small: about the same accuracy, smaller download."),
    stt!("whisper-medium.en", "ggml-medium.en.bin", 1463, "en", "Whisper medium (English)", "Very good accuracy. Needs a GPU (about 2 GB VRAM)."),
    stt!("whisper-medium.en-q5_0", "ggml-medium.en-q5_0.bin", 514, "en", "Whisper medium (English, small file)", "Compressed medium, GPU recommended."),
    stt!("whisper-tiny", "ggml-tiny.bin", 74, "multi", "Whisper tiny (multilingual)", "Fastest, lowest accuracy, 99 languages."),
    stt!("whisper-base", "ggml-base.bin", 141, "multi", "Whisper base (multilingual)", "Fast, so-so accuracy, 99 languages."),
    stt!("whisper-small", "ggml-small.bin", 465, "multi", "Whisper small (multilingual)", "Good accuracy for many languages. GPU recommended."),
    stt!("whisper-small-q5_1", "ggml-small-q5_1.bin", 181, "multi", "Whisper small (multilingual, small file)", "Compressed small, 99 languages."),
    stt!("whisper-medium", "ggml-medium.bin", 1463, "multi", "Whisper medium (multilingual)", "Very good accuracy, 99 languages. Needs a GPU."),
    stt!("whisper-medium-q5_0", "ggml-medium-q5_0.bin", 514, "multi", "Whisper medium (multilingual, small file)", "Compressed medium, GPU recommended."),
    stt!("whisper-large-v3-turbo", "ggml-large-v3-turbo.bin", 1549, "multi", "Whisper large-v3 turbo", "Best accuracy that is still fast on a GPU (about 3 GB VRAM). Not for CPUs."),
    stt!("whisper-large-v3-turbo-q8_0", "ggml-large-v3-turbo-q8_0.bin", 834, "multi", "Whisper large-v3 turbo (q8)", "Near-identical accuracy to large-v3 turbo, about half the size."),
    stt!("whisper-large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin", 547, "multi", "Whisper large-v3 turbo (q5, small file)", "Smallest turbo, slight accuracy loss, GPU recommended."),
    Component {
        id: "whisper-server-cpu",
        kind: Kind::SttServer,
        name: "Speech engine (CPU)",
        desc: "whisper.cpp server for any PC. Small download.",
        size_mb: 9,
        langs: "multi",
        src: Source::Zip { url: concat!("https://github.com/ggml-org/whisper.cpp/releases/download/b5454/", "whisper-bin-x64.zip") },
    },
    Component {
        id: "whisper-server-cuda",
        kind: Kind::SttServer,
        name: "Speech engine (NVIDIA GPU, CUDA 12)",
        desc: "whisper.cpp server with CUDA for NVIDIA graphics cards. Big download, much faster.",
        size_mb: 653,
        langs: "multi",
        src: Source::Zip { url: concat!("https://github.com/ggml-org/whisper.cpp/releases/download/b5454/", "whisper-bin-win-cuda-12.4.0-x64.zip") },
    },
    Component {
        id: "nllb-200-600m",
        kind: Kind::MtModel,
        name: "NLLB-200 600M (int8)",
        desc: "Translates between about 200 languages. Needs a few seconds on CPU for long sentences.",
        size_mb: 620,
        langs: "multi",
        src: Source::HfRepo {
            repo: "JustFrederik/nllb-200-distilled-600M-ct2-int8",
            files: &[
                "config.json",
                "model.bin",
                "sentencepiece.bpe.model",
                "shared_vocabulary.txt",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        },
    },
    Component {
        id: "opus-mt-en-tr",
        kind: Kind::MtModel,
        name: "Opus-MT English to Turkish",
        desc: "Fast, good English to Turkish only (community float16 build of Helsinki-NLP opus-mt-tc-big-en-tr).",
        size_mb: 452,
        langs: "en-tr",
        src: Source::HfRepo {
            repo: "ooeoeo/opus-mt-tc-big-en-tr-ct2-float16",
            files: &["config.json", "model.bin", "shared_vocabulary.json", "source.spm", "target.spm", "tokenizer_config.json", "vocab.json"],
        },
    },
];

pub fn catalog() -> &'static [Component] {
    CATALOG
}

pub fn find(id: &str) -> Option<&'static Component> {
    CATALOG.iter().find(|c| c.id == id)
}

const MARKER: &str = ".installed";

fn file_name(c: &Component) -> Option<&'static str> {
    match c.src {
        Source::File { file, .. } => Some(file),
        _ => None,
    }
}

/// Where this component lives (a file for speech models, a folder otherwise).
pub fn install_dir(c: &Component, data_dir: &Path) -> PathBuf {
    match c.kind {
        Kind::SttModel => data_dir.join("models").join(file_name(c).unwrap_or(c.id)),
        Kind::SttServer => data_dir.join("servers").join(c.id),
        Kind::MtModel => data_dir.join("mt").join(c.id),
    }
}

pub fn is_installed(c: &Component, data_dir: &Path) -> bool {
    let p = install_dir(c, data_dir);
    match c.kind {
        Kind::SttModel => p.is_file(),
        _ => p.join(MARKER).is_file(),
    }
}

// ---------- progress ----------

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Idle,
    Downloading,
    Extracting,
    Done,
    Failed(String),
    Cancelled,
}

pub struct Progress {
    downloaded: AtomicU64,
    total: AtomicU64,
    state: Mutex<State>,
    cancel: AtomicBool,
}

impl Progress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            downloaded: AtomicU64::new(0),
            total: AtomicU64::new(0),
            state: Mutex::new(State::Idle),
            cancel: AtomicBool::new(false),
        })
    }
    /// 0.0 - 1.0 (0 while the size is not known yet).
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 {
            return 0.0;
        }
        (self.downloaded.load(Ordering::Relaxed) as f64 / t as f64).clamp(0.0, 1.0) as f32
    }
    pub fn bytes(&self) -> (u64, u64) {
        (self.downloaded.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed))
    }
    pub fn state(&self) -> State {
        self.state.lock().map(|s| s.clone()).unwrap_or(State::Idle)
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    fn set(&self, s: State) {
        if let Ok(mut g) = self.state.lock() {
            *g = s;
        }
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

// ---------- download ----------

const CANCELLED: &str = "cancelled";

/// Blocking. Run it on a thread and poll `Progress`. Resumes a previous partial download.
pub fn download(c: &Component, data_dir: &Path, p: Arc<Progress>) -> Result<()> {
    p.cancel.store(false, Ordering::Relaxed);
    p.downloaded.store(0, Ordering::Relaxed);
    p.total.store(0, Ordering::Relaxed);
    p.set(State::Downloading);
    let r = run_download(c, data_dir, &p);
    match &r {
        Ok(()) => p.set(State::Done),
        Err(e) if e.to_string() == CANCELLED => p.set(State::Cancelled),
        Err(e) => p.set(State::Failed(format!("{e:#}"))),
    }
    r
}

fn run_download(c: &Component, data_dir: &Path, p: &Progress) -> Result<()> {
    let dir = install_dir(c, data_dir);
    match &c.src {
        Source::LocalOnly => bail!("{} has no download; copy it into place or use import_existing", c.id),
        Source::File { url, .. } => {
            fs::create_dir_all(dir.parent().context("no parent dir")?)?;
            fetch(url, &dir, p, true)
        }
        Source::Zip { url } => {
            fs::create_dir_all(&dir)?;
            let zip_path = dir.join("download.zip");
            fetch(url, &zip_path, p, true)?;
            p.set(State::Extracting);
            extract_zip(&zip_path, &dir, p)?;
            let _ = fs::remove_file(&zip_path);
            fs::write(dir.join(MARKER), b"ok")?;
            Ok(())
        }
        Source::HfRepo { repo, files } => {
            fs::create_dir_all(&dir)?;
            // Learn the total size first so the bar is right from the start.
            let urls: Vec<String> = files.iter().map(|f| format!("https://huggingface.co/{repo}/resolve/main/{f}")).collect();
            let total: u64 = urls.iter().map(|u| remote_size(u).unwrap_or(0)).sum();
            p.total.store(total, Ordering::Relaxed);
            for (f, u) in files.iter().zip(&urls) {
                fetch(u, &dir.join(f), p, false)?;
            }
            fs::write(dir.join(MARKER), b"ok")?;
            Ok(())
        }
    }
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(8)
        .timeout_connect(std::time::Duration::from_secs(20))
        .timeout_read(std::time::Duration::from_secs(60))
        .build()
}

fn remote_size(url: &str) -> Option<u64> {
    let r = agent().head(url).call().ok()?;
    r.header("content-length")?.parse().ok()
}

/// Downloads `url` to `dest` through `dest.part` with resume. `own_total`: this file defines the
/// bar total (single-file components); otherwise only `downloaded` is advanced.
fn fetch(url: &str, dest: &Path, p: &Progress, own_total: bool) -> Result<()> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if dest.is_file() && !own_total {
        // multi-file component resumed after the earlier files finished
        let len = fs::metadata(dest)?.len();
        p.downloaded.fetch_add(len, Ordering::Relaxed);
        return Ok(());
    }

    let mut req = agent().get(url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // 416: the .part already holds the whole file
        Err(ureq::Error::Status(416, _)) => {
            fs::rename(&part, dest)?;
            if own_total {
                let len = fs::metadata(dest)?.len();
                p.total.store(len, Ordering::Relaxed);
                p.downloaded.store(len, Ordering::Relaxed);
            }
            return Ok(());
        }
        Err(e) => return Err(anyhow!("download failed for {url}: {e}")),
    };
    let resumed = resp.status() == 206;
    let body_len: u64 = resp.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let start = if resumed { have } else { 0 };
    let expected = if body_len > 0 { Some(start + body_len) } else { None };

    let mut file = if resumed {
        fs::OpenOptions::new().append(true).open(&part)?
    } else {
        File::create(&part)?
    };
    if own_total {
        p.total.store(expected.unwrap_or(0), Ordering::Relaxed);
        p.downloaded.store(start, Ordering::Relaxed);
    } else {
        p.downloaded.fetch_add(start, Ordering::Relaxed);
    }

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut got = start;
    loop {
        if p.cancelled() {
            file.flush().ok();
            bail!(CANCELLED); // .part stays for the next try
        }
        let n = reader.read(&mut buf).map_err(|e| anyhow!("network error: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        got += n as u64;
        p.downloaded.fetch_add(n as u64, Ordering::Relaxed);
    }
    file.flush()?;
    drop(file);
    if let Some(exp) = expected {
        if got != exp {
            let _ = fs::remove_file(&part);
            bail!("incomplete download of {url}: got {got} of {exp} bytes");
        }
    }
    fs::rename(&part, dest)?;
    Ok(())
}

fn extract_zip(zip_path: &Path, dir: &Path, p: &Progress) -> Result<()> {
    let mut zip = zip::ZipArchive::new(File::open(zip_path)?)?;
    for i in 0..zip.len() {
        if p.cancelled() {
            bail!(CANCELLED);
        }
        let mut entry = zip.by_index(i)?;
        // enclosed_name rejects absolute paths and `..` (zip-slip).
        let Some(rel) = entry.enclosed_name() else { bail!("unsafe path in zip: {}", entry.name()) };
        let out = dir.join(rel);
        if entry.is_dir() {
            fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut f = File::create(&out)?;
        std::io::copy(&mut entry, &mut f)?;
    }
    Ok(())
}

pub fn remove(c: &Component, data_dir: &Path) -> Result<()> {
    let p = install_dir(c, data_dir);
    if p.is_dir() {
        fs::remove_dir_all(&p)?;
    } else if p.is_file() {
        fs::remove_file(&p)?;
    }
    let part = PathBuf::from(format!("{}.part", p.display()));
    let _ = fs::remove_file(part);
    Ok(())
}

// ---------- resolvers ----------

pub fn stt_model_path(id: &str, data_dir: &Path) -> Option<PathBuf> {
    let c = find(id).filter(|c| c.kind == Kind::SttModel)?;
    is_installed(c, data_dir).then(|| install_dir(c, data_dir))
}

fn find_exe(dir: &Path, depth: u32) -> Option<PathBuf> {
    for e in fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_file() && p.file_name().map_or(false, |n| n.eq_ignore_ascii_case("whisper-server.exe")) {
            return Some(p);
        }
        if p.is_dir() && depth > 0 {
            if let Some(f) = find_exe(&p, depth - 1) {
                return Some(f);
            }
        }
    }
    None
}

fn server_in(id: &str, data_dir: &Path) -> Option<PathBuf> {
    let c = find(id)?;
    if !is_installed(c, data_dir) {
        return None;
    }
    find_exe(&install_dir(c, data_dir), 3)
}

pub fn stt_server_exe(backend: Backend, data_dir: &Path) -> Option<PathBuf> {
    match backend {
        Backend::Cuda => server_in("whisper-server-cuda", data_dir),
        Backend::Cpu => server_in("whisper-server-cpu", data_dir),
        Backend::Auto => server_in("whisper-server-cuda", data_dir).or_else(|| server_in("whisper-server-cpu", data_dir)),
    }
}

pub fn mt_model_dir(id: &str, data_dir: &Path) -> Option<PathBuf> {
    let c = find(id).filter(|c| c.kind == Kind::MtModel)?;
    is_installed(c, data_dir).then(|| install_dir(c, data_dir))
}

// ---------- import of files the developer already has ----------

fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::hard_link(from, to).is_err() {
        fs::copy(from, to)?;
    }
    Ok(())
}

fn link_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for e in fs::read_dir(from)?.flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            link_tree(&src, &dst)?;
        } else if !dst.exists() {
            link_or_copy(&src, &dst)?;
        }
    }
    Ok(())
}

/// Adopts `<project>/models/*` and `<project>/tools/whisper-cuda` (hard links when possible, so no
/// extra disk space). Returns a human-readable line per imported component.
pub fn import_existing(project_dir: &Path, data_dir: &Path) -> Vec<String> {
    let mut done = Vec::new();
    for c in CATALOG {
        if is_installed(c, data_dir) {
            continue;
        }
        let target = install_dir(c, data_dir);
        let local: Option<PathBuf> = match c.kind {
            Kind::SttModel => file_name(c).map(|f| project_dir.join("models").join(f)).filter(|p| p.is_file()),
            Kind::SttServer if c.id == "whisper-server-cuda" => {
                Some(project_dir.join("tools/whisper-cuda")).filter(|p| find_exe(p, 2).is_some())
            }
            Kind::MtModel => {
                let name = match c.id {
                    "nllb-200-600m" => "nllb-200-600m",
                    "opus-mt-en-tr" => "opus-en-tr",
                    _ => continue,
                };
                Some(project_dir.join("models").join(name)).filter(|p| p.join("model.bin").is_file())
            }
            _ => None,
        };
        let Some(local) = local else { continue };
        let r = if local.is_dir() {
            link_tree(&local, &target).and_then(|_| fs::write(target.join(MARKER), b"ok").map_err(Into::into))
        } else {
            link_or_copy(&local, &target)
        };
        match r {
            Ok(()) => done.push(format!("{} adopted from {}", c.id, local.display())),
            Err(e) => done.push(format!("{} could not be adopted: {e:#}", c.id)),
        }
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rustlate-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn ids_unique_and_findable() {
        let mut ids: Vec<_> = CATALOG.iter().map(|c| c.id).collect();
        let n = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), n);
        assert!(find("whisper-small.en").is_some() && find("nope").is_none());
        for c in CATALOG {
            assert!(c.size_mb > 0);
            if let Source::File { url, file } = &c.src {
                assert!(url.ends_with(file) && url.starts_with(HF), "{}", c.id);
            }
        }
        assert!(GH.contains("b5454"));
    }

    #[test]
    fn paths_and_installed_state() {
        let d = tmp("paths");
        let c = find("whisper-tiny.en").unwrap();
        assert!(!is_installed(c, &d) && stt_model_path(c.id, &d).is_none());
        let p = install_dir(c, &d);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, b"x").unwrap();
        assert_eq!(stt_model_path(c.id, &d), Some(p));

        // A server without the marker is not installed even if the exe exists.
        let s = find("whisper-server-cpu").unwrap();
        let sd = install_dir(s, &d).join("Release");
        fs::create_dir_all(&sd).unwrap();
        fs::write(sd.join("whisper-server.exe"), b"x").unwrap();
        assert!(stt_server_exe(Backend::Cpu, &d).is_none());
        fs::write(install_dir(s, &d).join(MARKER), b"ok").unwrap();
        assert_eq!(stt_server_exe(Backend::Cpu, &d), Some(sd.join("whisper-server.exe")));
        assert_eq!(stt_server_exe(Backend::Auto, &d), Some(sd.join("whisper-server.exe")));
        assert!(stt_server_exe(Backend::Cuda, &d).is_none());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn zip_slip_is_rejected() {
        let d = tmp("zipslip");
        let zp = d.join("evil.zip");
        {
            let mut w = zip::ZipWriter::new(File::create(&zp).unwrap());
            let o = zip::write::SimpleFileOptions::default();
            w.start_file("../escaped.txt", o).unwrap();
            w.write_all(b"x").unwrap();
            w.finish().unwrap();
        }
        let out = d.join("out");
        fs::create_dir_all(&out).unwrap();
        assert!(extract_zip(&zp, &out, &Progress::new()).is_err());
        assert!(!d.join("escaped.txt").exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn zip_extracts_nested_files() {
        let d = tmp("zipok");
        let zp = d.join("ok.zip");
        {
            let mut w = zip::ZipWriter::new(File::create(&zp).unwrap());
            let o = zip::write::SimpleFileOptions::default();
            w.start_file("Release/whisper-server.exe", o).unwrap();
            w.write_all(b"MZ").unwrap();
            w.finish().unwrap();
        }
        let out = d.join("out");
        fs::create_dir_all(&out).unwrap();
        extract_zip(&zp, &out, &Progress::new()).unwrap();
        assert_eq!(find_exe(&out, 3), Some(out.join("Release").join("whisper-server.exe")));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn progress_fraction_and_cancel() {
        let p = Progress::new();
        assert_eq!(p.fraction(), 0.0);
        p.total.store(200, Ordering::Relaxed);
        p.downloaded.store(50, Ordering::Relaxed);
        assert!((p.fraction() - 0.25).abs() < 1e-6);
        p.downloaded.store(500, Ordering::Relaxed);
        assert_eq!(p.fraction(), 1.0);
        assert_eq!(p.state(), State::Idle);
        p.cancel();
        assert!(p.cancelled());
    }

    #[test]
    fn import_links_existing_files() {
        let proj = tmp("proj");
        let data = tmp("data");
        fs::create_dir_all(proj.join("models")).unwrap();
        fs::write(proj.join("models/ggml-base.en.bin"), b"model").unwrap();
        fs::create_dir_all(proj.join("models/opus-en-tr")).unwrap();
        fs::write(proj.join("models/opus-en-tr/model.bin"), b"m").unwrap();
        let done = import_existing(&proj, &data);
        assert!(done.iter().any(|l| l.contains("whisper-base.en")), "{done:?}");
        assert!(stt_model_path("whisper-base.en", &data).is_some());
        assert!(mt_model_dir("opus-mt-en-tr", &data).is_some());
        // Second run adopts nothing new.
        assert!(import_existing(&proj, &data).is_empty());
        let _ = fs::remove_dir_all(&proj);
        let _ = fs::remove_dir_all(&data);
    }
}
