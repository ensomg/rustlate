//! Per-process audio capture via WASAPI process loopback (Windows 10 2004+).
//! Output: 16 kHz mono f32 samples sent over a channel.

use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Sender};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use windows::core::{implement, Interface, HRESULT, PROPVARIANT};
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};

const SRC_RATE: u32 = 48_000;
const CHANNELS: u16 = 2;
pub const OUT_RATE: u32 = 16_000;

#[implement(IActivateAudioInterfaceCompletionHandler)]
struct Handler(Sender<Result<IAudioClient, String>>);

impl IActivateAudioInterfaceCompletionHandler_Impl for Handler_Impl {
    fn ActivateCompleted(
        &self,
        op: Option<&IActivateAudioInterfaceAsyncOperation>,
    ) -> windows::core::Result<()> {
        let res = (|| -> Result<IAudioClient, String> {
            let op = op.ok_or("no operation")?;
            let mut hr = HRESULT(0);
            let mut unk: Option<windows::core::IUnknown> = None;
            unsafe { op.GetActivateResult(&mut hr, &mut unk) }.map_err(|e| e.to_string())?;
            hr.ok().map_err(|e| e.to_string())?;
            unk.ok_or("null interface")?.cast::<IAudioClient>().map_err(|e| e.to_string())
        })();
        let _ = self.0.send(res);
        Ok(())
    }
}

fn activate(pid: u32) -> Result<IAudioClient> {
    let mut params = AUDIOCLIENT_ACTIVATION_PARAMS {
        ActivationType: AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        Anonymous: AUDIOCLIENT_ACTIVATION_PARAMS_0 {
            ProcessLoopbackParams: AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS {
                TargetProcessId: pid,
                ProcessLoopbackMode: PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
            },
        },
    };
    // Hand-built VT_BLOB PROPVARIANT (x64 layout: 8-byte header + blob {u32 size, ptr}).
    #[repr(C)]
    struct RawPropVariant {
        vt: u16,
        r1: u16,
        r2: u16,
        r3: u16,
        cb_size: u32,
        p_blob: *mut u8,
    }
    const _: () = assert!(std::mem::size_of::<RawPropVariant>() == std::mem::size_of::<PROPVARIANT>());
    let raw = RawPropVariant {
        vt: 65, // VT_BLOB
        r1: 0,
        r2: 0,
        r3: 0,
        cb_size: std::mem::size_of::<AUDIOCLIENT_ACTIVATION_PARAMS>() as u32,
        p_blob: &mut params as *mut _ as *mut u8,
    };
    let prop: ManuallyDrop<PROPVARIANT> = ManuallyDrop::new(unsafe { std::mem::transmute_copy(&raw) });
    let (tx, rx) = bounded(1);
    let handler: IActivateAudioInterfaceCompletionHandler = Handler(tx).into();
    let _op = unsafe {
        ActivateAudioInterfaceAsync(
            VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            &IAudioClient::IID,
            Some(&*prop as *const PROPVARIANT),
            &handler,
        )?
    };
    rx.recv_timeout(Duration::from_secs(5))
        .map_err(|_| anyhow!("activation timed out"))?
        .map_err(|e| anyhow!("activation failed: {e}"))
}

/// Starts capture on a background thread. Samples are 16 kHz mono f32 in chunks.
/// Set `stop` to end the thread.
pub fn start(pid: u32, tx: Sender<Vec<f32>>, stop: Arc<AtomicBool>) -> Result<std::thread::JoinHandle<()>> {
    let (ready_tx, ready_rx) = bounded::<Result<(), String>>(1);
    let h = std::thread::spawn(move || {
        let r = run(pid, tx, stop, &ready_tx);
        if let Err(e) = r {
            let _ = ready_tx.send(Err(e.to_string()));
        }
    });
    ready_rx
        .recv_timeout(Duration::from_secs(8))
        .map_err(|_| anyhow!("capture did not start"))?
        .map_err(|e| anyhow!(e))?;
    Ok(h)
}

/// Windowed-sinc (Blackman) low-pass, `fc` as fraction of the sample rate, unity DC gain.
fn lowpass_taps(n: usize, fc: f32) -> Vec<f32> {
    let m = (n - 1) as f32 / 2.0;
    let mut h: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f32 - m;
            let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f32::consts::PI * fc * x).sin() / (std::f32::consts::PI * x) };
            let w = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
                + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
            sinc * w
        })
        .collect();
    let s: f32 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);
    h
}

fn run(pid: u32, tx: Sender<Vec<f32>>, stop: Arc<AtomicBool>, ready: &Sender<Result<(), String>>) -> Result<()> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let client = activate(pid)?;
        let block = CHANNELS * 4;
        let fmt = WAVEFORMATEX {
            wFormatTag: 3, // WAVE_FORMAT_IEEE_FLOAT
            nChannels: CHANNELS,
            nSamplesPerSec: SRC_RATE,
            nAvgBytesPerSec: SRC_RATE * block as u32,
            nBlockAlign: block,
            wBitsPerSample: 32,
            cbSize: 0,
        };
        client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
            2_000_000, // 200 ms
            0,
            &fmt,
            None,
        )?;
        let cap: IAudioCaptureClient = client.GetService()?;
        client.Start()?;
        let _ = ready.send(Ok(()));

        let ratio = (SRC_RATE / OUT_RATE) as usize; // 3
        let taps = lowpass_taps(129, 7200.0 / SRC_RATE as f32);
        let mut pos = 0usize;
        let mut carry: Vec<f32> = Vec::new();
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(20));
            loop {
                let n = cap.GetNextPacketSize()?;
                if n == 0 {
                    break;
                }
                let mut data: *mut u8 = std::ptr::null_mut();
                let mut frames = 0u32;
                let mut flags = 0u32;
                cap.GetBuffer(&mut data, &mut frames, &mut flags, None, None)?;
                let silent = flags & (AUDCLNT_BUFFERFLAGS_SILENT.0 as u32) != 0;
                let s = std::slice::from_raw_parts(data as *const f32, frames as usize * CHANNELS as usize);
                for f in s.chunks_exact(CHANNELS as usize) {
                    carry.push(if silent { 0.0 } else { (f[0] + f[1]) * 0.5 });
                }
                cap.ReleaseBuffer(frames)?;
            }
            // Low-pass FIR then decimate 48k -> 16k.
            let mut out = Vec::new();
            while pos + taps.len() <= carry.len() {
                let y: f32 = carry[pos..pos + taps.len()].iter().zip(&taps).map(|(a, b)| a * b).sum();
                out.push(y);
                pos += ratio;
            }
            carry.drain(..pos);
            pos = 0;
            if !out.is_empty() && tx.send(out).is_err() {
                break;
            }
        }
        let _ = client.Stop();
    }
    Ok(())
}
