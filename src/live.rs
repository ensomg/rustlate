//! Live pipeline: capture chunks -> energy VAD/utterance buffer -> STT -> events.

pub const RATE: usize = 16_000;
const FRAME: usize = 320; // 20 ms
const MIN_UTTERANCE: usize = RATE / 2;

/// The latency/accuracy knobs, converted from `config::Speed` into samples and frames.
#[derive(Clone, Copy)]
struct Cfg {
    speech_rms: f32,
    end_silence_frames: usize,
    partial_every: usize,
    max_utterance: usize,
}

impl Cfg {
    fn from(s: &crate::config::Speed) -> Self {
        Self {
            speech_rms: s.vad_threshold.clamp(0.0005, 0.2),
            end_silence_frames: (s.end_silence_ms as usize / 20).clamp(3, 200),
            partial_every: (s.partial_every_ms as usize * RATE / 1000).clamp(RATE / 5, RATE * 5),
            max_utterance: (s.max_utterance_s as usize).clamp(3, 30) * RATE,
        }
    }
}

pub enum Event {
    /// Utterance still in progress; transcribe `buf` as a partial.
    Partial(Vec<f32>),
    /// Utterance finished; transcribe `buf` as final.
    Final(Vec<f32>),
}

/// 300 ms kept from before speech starts so soft word onsets aren't clipped.
const PRE_ROLL: usize = FRAME * 15;
/// When forced to split a long utterance, look for the quietest frame in this tail.
const SPLIT_SEARCH_FRAMES: usize = 75; // 1.5 s

fn frame_rms(f: &[f32]) -> f32 {
    (f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32).sqrt()
}

pub struct Chunker {
    cfg: Cfg,
    buf: Vec<f32>,
    pre: Vec<f32>,
    pending: Vec<f32>,
    in_speech: bool,
    silence: usize,
    last_partial_len: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new(&crate::config::Speed::default())
    }
}

impl Chunker {
    pub fn new(speed: &crate::config::Speed) -> Self {
        Self {
            cfg: Cfg::from(speed),
            buf: Vec::new(),
            pre: Vec::new(),
            pending: Vec::new(),
            in_speech: false,
            silence: 0,
            last_partial_len: 0,
        }
    }

    /// Cut `buf` at its quietest frame near the end; the rest starts the next utterance.
    fn split_quiet(&mut self) -> Vec<f32> {
        let frames = self.buf.len() / FRAME;
        let from = frames.saturating_sub(SPLIT_SEARCH_FRAMES);
        let mut best = frames - 1;
        let mut best_rms = f32::MAX;
        for i in from..frames {
            let r = frame_rms(&self.buf[i * FRAME..(i + 1) * FRAME]);
            if r <= best_rms {
                best_rms = r;
                best = i;
            }
        }
        let cut = (best + 1) * FRAME;
        let rest = self.buf.split_off(cut);
        std::mem::replace(&mut self.buf, rest)
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<Event> {
        let mut events = Vec::new();
        self.pending.extend_from_slice(samples);
        let mut off = 0;
        while self.pending.len() - off >= FRAME {
            let f = self.pending[off..off + FRAME].to_vec();
            off += FRAME;
            let rms = frame_rms(&f);
            if rms > self.cfg.speech_rms {
                if !self.in_speech {
                    self.buf.append(&mut self.pre);
                }
                self.in_speech = true;
                self.silence = 0;
            } else if self.in_speech {
                self.silence += 1;
            }
            if !self.in_speech {
                self.pre.extend_from_slice(&f);
                if self.pre.len() > PRE_ROLL {
                    let extra = self.pre.len() - PRE_ROLL;
                    self.pre.drain(..extra);
                }
                continue;
            }
            {
                self.buf.extend_from_slice(&f);
                if self.silence >= self.cfg.end_silence_frames {
                    let b = std::mem::take(&mut self.buf);
                    self.in_speech = false;
                    self.silence = 0;
                    self.last_partial_len = 0;
                    if b.len() >= MIN_UTTERANCE {
                        events.push(Event::Final(b));
                    }
                } else if self.buf.len() >= self.cfg.max_utterance {
                    let b = self.split_quiet();
                    self.last_partial_len = 0;
                    events.push(Event::Final(b));
                } else if self.buf.len() - self.last_partial_len >= self.cfg.partial_every {
                    self.last_partial_len = self.buf.len();
                    events.push(Event::Partial(self.buf.clone()));
                }
            }
        }
        self.pending.drain(..off);
        // Only the newest event of each kind matters if STT is behind.
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.1).sin() * 0.3).collect()
    }

    #[test]
    fn speech_then_silence_gives_final() {
        let mut c = Chunker::default();
        let mut ev = c.push(&tone(RATE * 2));
        ev.extend(c.push(&vec![0.0; RATE]));
        assert!(ev.iter().any(|e| matches!(e, Event::Partial(_))));
        assert!(matches!(ev.last(), Some(Event::Final(b)) if b.len() >= RATE * 2));
    }

    #[test]
    fn silence_only_gives_nothing() {
        let mut c = Chunker::default();
        assert!(c.push(&vec![0.0; RATE * 3]).is_empty());
    }

    #[test]
    fn speed_settings_are_applied() {
        use crate::config::Speed;
        // Higher VAD threshold: a quiet tone is no longer speech.
        let quiet: Vec<f32> = (0..RATE).map(|i| (i as f32 * 0.1).sin() * 0.02).collect();
        let mut loud_only = Chunker::new(&Speed { vad_threshold: 0.05, ..Speed::default() });
        assert!(loud_only.push(&quiet).is_empty());
        let mut normal = Chunker::default();
        assert!(!normal.push(&quiet).is_empty());
        // Shorter max utterance splits earlier.
        let mut short = Chunker::new(&Speed { max_utterance_s: 3, ..Speed::default() });
        assert!(short.push(&tone(RATE * 4)).iter().any(|e| matches!(e, Event::Final(_))));
        // Partial interval is configurable.
        let mut slow = Chunker::new(&Speed { partial_every_ms: 2000, ..Speed::default() });
        let n_slow = slow.push(&tone(RATE * 3)).iter().filter(|e| matches!(e, Event::Partial(_))).count();
        let n_fast = Chunker::default().push(&tone(RATE * 3)).iter().filter(|e| matches!(e, Event::Partial(_))).count();
        assert!(n_fast > n_slow);
    }

    #[test]
    fn long_speech_is_split() {
        let mut c = Chunker::default();
        let ev = c.push(&tone(RATE * 9));
        assert!(ev.iter().any(|e| matches!(e, Event::Final(_))));
    }
}
