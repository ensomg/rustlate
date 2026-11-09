//! Speech-to-text through a local whisper.cpp `whisper-server.exe` child process (CPU or CUDA build,
//! both downloaded on demand, see components.rs).

use anyhow::{anyhow, bail, Context, Result};
use std::io::Cursor;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub struct SttServer {
    child: Child,
    port: u16,
}

/// Ties the child to this process: when we exit or are killed, Windows kills the server too.
fn kill_on_close(child: &Child) {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::*;
    unsafe {
        let Ok(job) = CreateJobObjectW(None, None) else { return };
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let _ = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        let _ = AssignProcessToJobObject(job, HANDLE(child.as_raw_handle() as _));
        // Intentionally never closed: it must live as long as this process.
    }
}

/// A free local port, so a leftover server or another instance never collides with us.
fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

impl SttServer {
    /// `threads` 0 = automatic. `language` = ISO code or "auto".
    pub fn start(exe: &Path, model: &Path, threads: u32, language: &str) -> Result<Self> {
        if !exe.is_file() {
            bail!("speech engine not found: {}", exe.display());
        }
        if !model.is_file() {
            bail!("speech model not found: {}", model.display());
        }
        let threads = if threads == 0 {
            (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(2, 8) as u32
        } else {
            threads
        };
        let port = free_port()?;
        let mut child = Command::new(exe)
            .arg("-m")
            .arg(model)
            .args(["--host", "127.0.0.1", "--port", &port.to_string()])
            .args(["-t", &threads.to_string(), "-l", language, "-nf", "-sns", "-nth", "0.8"])
            .current_dir(exe.parent().unwrap_or_else(|| Path::new(".")))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .context("could not start the speech engine")?;
        kill_on_close(&child);

        let addr: SocketAddr = ([127, 0, 0, 1], port).into();
        let start = Instant::now();
        loop {
            if let Some(st) = child.try_wait()? {
                bail!("speech engine exited at start ({st}); a CUDA engine needs an NVIDIA GPU and driver");
            }
            if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
                break;
            }
            if start.elapsed() > Duration::from_secs(90) {
                let _ = child.kill();
                bail!("speech engine did not start within 90 s");
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(Self { child, port })
    }

    /// 16 kHz mono f32 samples -> text in the configured language.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let mut wav = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::new(&mut wav, spec)?;
            for s in samples {
                w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
            }
            w.finalize()?;
        }
        let boundary = "----rustlate7d3f1c";
        let mut body: Vec<u8> = Vec::new();
        let mut field = |name: &str, value: &str| {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
            );
        };
        field("response_format", "json");
        field("temperature", "0.0");
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(wav.get_ref());
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let resp = ureq::post(&format!("http://127.0.0.1:{}/inference", self.port))
            .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .timeout(Duration::from_secs(30))
            .send_bytes(&body)
            .map_err(|e| anyhow!("speech engine request failed: {e}"))?
            .into_string()?;
        let v: serde_json::Value = serde_json::from_str(&resp)?;
        let text = v["text"].as_str().ok_or_else(|| anyhow!("unexpected reply: {resp}"))?;
        Ok(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }
}

impl Drop for SttServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}
