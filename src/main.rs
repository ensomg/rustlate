// Release builds are GUI-subsystem (no console window on double click). Subcommands re-attach to
// the parent console below so `rustlate live <pid>` etc. still print.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod audio;
mod components;
mod config;
mod fonts;
mod live;
mod mt;
mod pipeline;
mod stt_server;
mod style; // owned by the "overlay styles" session
mod ui;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Usage:
//   rustlate                 -> the app (pick a window, install components, settings)
//   rustlate overlay <pid>   -> subtitle overlay for that process directly
//   rustlate live <pid>      -> console subtitles
//   rustlate list            -> list windows
//   rustlate rec <pid> [s]   -> record s seconds of that process to out.wav
//   rustlate sim [wav]       -> run a wav through the whole pipeline
//   rustlate mt <text>       -> test translation (env RUSTLATE_MT_*)
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        unsafe {
            let _ = windows::Win32::System::Console::AttachConsole(windows::Win32::System::Console::ATTACH_PARENT_PROCESS);
        }
    }
    let pid_arg = || -> anyhow::Result<u32> {
        args.get(2).and_then(|a| a.parse().ok()).ok_or_else(|| anyhow::anyhow!("pid gerekli"))
    };
    match args.get(1).map(|a| a.as_str()) {
        None => app::run(),
        Some("overlay") => ui::run_overlay(pid_arg()?),
        Some("live") => {
            println!("listening (Ctrl+C to quit)");
            pipeline::run(pid_arg()?, move |tr, fin, secs| {
                let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
                println!("@{ms} {} {tr}  [{secs:.1}s]", if fin { "FINAL" } else { "…    " });
            })
        }
        Some("list") => {
            for w in audio::windows_list::list_windows() {
                println!("{:>6}  {}", w.pid, w.title);
            }
            Ok(())
        }
        Some("mt") => {
            // Env: RUSTLATE_MT_KIND=nllb|opus RUSTLATE_MT_DIR RUSTLATE_MT_SRC RUSTLATE_MT_DST
            //      RUSTLATE_MT_BEAM RUSTLATE_MT_THREADS. Each sentence arg is translated twice (cold, warm).
            let env = |k: &str| std::env::var(k).ok();
            let kind = if env("RUSTLATE_MT_KIND").as_deref() == Some("nllb") { mt::MtKind::Nllb } else { mt::MtKind::Opus };
            let spec = mt::MtSpec {
                dir: env("RUSTLATE_MT_DIR").map(Into::into).unwrap_or_else(|| pipeline::MT_MODEL.into()),
                kind,
                source: env("RUSTLATE_MT_SRC").unwrap_or_else(|| "en".into()),
                target: env("RUSTLATE_MT_DST").unwrap_or_else(|| "tr".into()),
                beam_size: env("RUSTLATE_MT_BEAM").and_then(|v| v.parse().ok()).unwrap_or(2),
                threads: env("RUSTLATE_MT_THREADS").and_then(|v| v.parse().ok()).unwrap_or(8),
            };
            let t = std::time::Instant::now();
            let mt = mt::Mt::with_spec(&spec)?;
            println!("loaded in {:.1}s ({:?} {}->{} beam {} threads {})", t.elapsed().as_secs_f32(), spec.kind, spec.source, spec.target, spec.beam_size, spec.threads);
            for text in &args[2..] {
                let t = std::time::Instant::now();
                let out = mt.translate(text)?;
                let cold = t.elapsed().as_millis();
                // Warm time = best of 3 runs (other processes on the machine make single runs noisy).
                let mut best = u128::MAX;
                for _ in 0..env("RUSTLATE_MT_REPEAT").and_then(|v| v.parse().ok()).unwrap_or(3) {
                    let t = std::time::Instant::now();
                    let _ = mt.translate(text)?;
                    best = best.min(t.elapsed().as_millis());
                }
                println!("[cold {cold} ms / warm best-of-3 {best} ms] {out}");
            }
            Ok(())
        }
        Some("sim") => pipeline::run_wav(args.get(2).map(|s| s.as_str()).unwrap_or("tts.wav")),
        Some("rec") => record(pid_arg()?, args.get(3).and_then(|a| a.parse().ok()).unwrap_or(10)),
        Some(other) => anyhow::bail!("bilinmeyen komut: {other}"),
    }
}

fn record(pid: u32, secs: u64) -> anyhow::Result<()> {
    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let h = audio::capture::start(pid, tx, stop.clone())?;
    println!("recording pid {pid} for {secs}s ...");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio::capture::OUT_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create("out.wav", spec)?;
    let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let (mut n, mut peak) = (0usize, 0f32);
    while std::time::Instant::now() < end {
        if let Ok(chunk) = rx.recv_timeout(std::time::Duration::from_millis(100)) {
            for s in chunk {
                peak = peak.max(s.abs());
                n += 1;
                w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
            }
        }
    }
    stop.store(true, Ordering::Relaxed);
    let _ = h.join();
    w.finalize()?;
    println!("wrote out.wav: {n} samples, peak {peak:.3}");
    Ok(())
}
