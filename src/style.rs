//! Subtitle presentation: queue, timing and the six animated styles (egui painter).
//!
//! `Subtitles` is a small state machine: finished sentences are pushed in, split into short
//! pieces, and shown one at a time. `show()` advances the queue and paints the current piece
//! with the configured style. The overlay window and the settings preview both use it.

use crate::config::{Display, Position, Speed, Style};
use eframe::egui::{
    self,
    text::{CCursor, LayoutJob, TextFormat},
    Align, Color32, FontId, Galley, Painter, Pos2, Rect, Vec2,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------
// Pure helpers (unit tested)
// ---------------------------------------------------------------------------------------------

/// Splits a sentence into short readable pieces of at most `max_chars`, preferring to break
/// after `,` `;` `:` once a piece is reasonably long.
pub fn chunks(sentence: &str, max_chars: usize) -> Vec<String> {
    let max = max_chars.max(8);
    let mut out = Vec::new();
    let mut cur = String::new();
    for w in sentence.split_whitespace() {
        let len = cur.chars().count();
        if len > 0 && len + 1 + w.chars().count() > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
        if cur.chars().count() >= max * 6 / 10 && w.ends_with([',', ';', ':']) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Seconds a piece stays on screen (including fades). Shorter when a backlog builds up.
pub fn hold_time(chars: usize, read_cps: f32, backlog: usize) -> f32 {
    let mut hold = (chars as f32 / read_cps.max(1.0) + 0.45).clamp(0.9, 3.2);
    hold *= match backlog {
        0 => 1.0,
        1 => 0.85,
        2 => 0.7,
        _ => 0.55,
    };
    hold.max(0.8)
}

/// Pieces beyond this many waiting are dropped (oldest first) so subtitles never lag far behind.
pub const MAX_BACKLOG: usize = 5;

pub fn ease_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

pub fn ease_in(t: f32) -> f32 {
    t.clamp(0.0, 1.0).powi(3)
}

/// Ease-out with overshoot (peaks around 1.1 before settling at 1.0).
pub fn ease_out_back(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let (c1, c3) = (1.70158_f32, 2.70158_f32);
    1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Fade duration for the current animation speed, never more than a third of the hold time.
pub fn fade_time(anim_speed: f32, hold: f32) -> f32 {
    (0.18 / anim_speed.clamp(0.25, 4.0)).clamp(0.04, 0.6).min(hold / 3.0)
}

/// Fade in at the start, fade out at the end: 0..1.
pub fn in_out_alpha(t: f32, hold: f32, fade: f32) -> f32 {
    let fin = ease_out(t / fade.max(0.001));
    let fout = ((hold - t) / fade.max(0.001)).clamp(0.0, 1.0);
    fin.min(fout)
}

/// Karaoke position in `[0, n_words]`: word `i` is fully lit when the position reaches `i + 1`.
/// Words light up between the end of the fade-in and ~85 % of the hold time.
pub fn karaoke_pos(n_words: usize, t: f32, hold: f32, fade: f32) -> f32 {
    let t0 = fade;
    let t1 = (hold * 0.85 - fade * 0.5).max(t0 + 0.25);
    ((t - t0) / (t1 - t0)).clamp(0.0, 1.0) * n_words as f32
}

/// Number of characters visible in the typewriter style at time `t`.
pub fn typed_chars(total: usize, t: f32, hold: f32, anim_speed: f32) -> usize {
    if total == 0 {
        return 0;
    }
    let rate = 45.0 * anim_speed.clamp(0.25, 4.0);
    let dur = (total as f32 / rate).min(hold * 0.75).max(0.05);
    ((t / dur).clamp(0.0, 1.0) * total as f32).floor() as usize
}

/// Overlay window geometry in screen pixels: `(width, height, x, y)`.
pub fn window_geometry(d: &Display, screen_w: f32, screen_h: f32) -> (f32, f32, f32, f32) {
    let w = (screen_w * d.width_pct.clamp(10.0, 100.0) / 100.0).clamp(200.0_f32.min(screen_w), screen_w);
    let h = d.font_size * if d.show_original { 5.6 } else { 4.2 };
    let x = (screen_w - w) / 2.0;
    let y = match d.position {
        Position::Bottom => screen_h - h - d.margin_px,
        Position::Top => d.margin_px,
        Position::Center => (screen_h - h) / 2.0,
    };
    (w, h, x, y.max(0.0))
}

// ---------------------------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------------------------

fn col(c: [u8; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3])
}

fn with_alpha(c: Color32, a: f32) -> Color32 {
    let [r, g, b, al] = c.to_srgba_unmultiplied();
    Color32::from_rgba_unmultiplied(r, g, b, (al as f32 * a.clamp(0.0, 1.0)).round() as u8)
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let (a, b) = (a.to_srgba_unmultiplied(), b.to_srgba_unmultiplied());
    let m = |i: usize| lerp(a[i] as f32, b[i] as f32, t.clamp(0.0, 1.0)).round() as u8;
    Color32::from_rgba_unmultiplied(m(0), m(1), m(2), m(3))
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Piece {
    text: String,
    original: Option<String>,
}

struct Active {
    piece: Piece,
    started: Instant,
    hold: f32,
}

type Parts = Vec<(String, Color32)>;

pub struct Subtitles {
    display: Display,
    speed: Speed,
    queue: VecDeque<Piece>,
    current: Option<Active>,
    /// Line shown just before `current` (Lyrics style scrolls it up and dims it).
    prev: Option<Piece>,
    prev_since: Instant,
    ever_shown: bool,
    status: Option<String>,
    wrap: f32,
}

impl Subtitles {
    pub fn new(display: &Display, speed: &Speed) -> Self {
        Self {
            display: display.clone(),
            speed: speed.clone(),
            queue: VecDeque::new(),
            current: None,
            prev: None,
            prev_since: Instant::now(),
            ever_shown: false,
            status: None,
            wrap: 800.0,
        }
    }

    /// Hot-applies new settings; queued and current pieces are kept.
    pub fn set_settings(&mut self, display: &Display, speed: &Speed) {
        self.display = display.clone();
        self.speed = speed.clone();
    }

    /// One finished translated sentence (optionally with its source-language text).
    pub fn push(&mut self, translated: &str, original: Option<&str>) {
        for (i, text) in chunks(translated, self.display.max_chars as usize).into_iter().enumerate() {
            let original = if i == 0 { original.map(str::to_string) } else { None };
            self.queue.push_back(Piece { text, original });
        }
    }

    /// Text shown (plain box) until the first subtitle appears, e.g. "Modeller yükleniyor…".
    pub fn set_status(&mut self, status: Option<String>) {
        self.status = status;
    }

    pub fn is_idle(&self) -> bool {
        self.current.is_none() && self.queue.is_empty()
    }

    /// 16 ms while animating, ~150 ms when idle.
    pub fn repaint_after(&self) -> Duration {
        Duration::from_millis(if self.is_idle() { 150 } else { 16 })
    }

    /// Moves to the next piece once the current one has been shown long enough.
    fn tick(&mut self, now: Instant) {
        let expired =
            self.current.as_ref().map_or(true, |c| now.duration_since(c.started).as_secs_f32() >= c.hold);
        if !expired {
            return;
        }
        let old = self.current.take();
        self.current = self.queue.pop_front().map(|piece| {
            while self.queue.len() > MAX_BACKLOG {
                self.queue.pop_front();
            }
            self.ever_shown = true;
            let hold = hold_time(piece.text.chars().count(), self.speed.read_cps, self.queue.len());
            Active { piece, started: now, hold }
        });
        // Only a continuous run of lines keeps a "previous line"; after a pause it starts fresh.
        self.prev = if self.current.is_some() { old.map(|a| a.piece) } else { None };
        self.prev_since = now;
    }

    /// Advances queue/animation and draws into `rect` (transparent background).
    pub fn show(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let now = Instant::now();
        self.tick(now);
        self.wrap = (rect.width() - 60.0).max(100.0);
        let painter = ui.painter().with_clip_rect(rect);
        let center = rect.center();

        let Some(a) = &self.current else {
            if !self.ever_shown {
                if let Some(s) = self.status.clone() {
                    let parts = vec![(s, col(self.display.text_color))];
                    self.draw_block(&painter, center, 1.0, 1.0, 1.0, &parts, None, true);
                }
            }
            return;
        };
        let t = now.duration_since(a.started).as_secs_f32();
        let hold = a.hold;
        let anim = self.display.anim_speed;
        let fade = fade_time(anim, hold);
        let text = a.piece.text.clone();
        let original = a.piece.original.clone();
        let orig = original.as_deref();
        let base = col(self.display.text_color);
        let accent = col(self.display.accent_color);
        let alpha = in_out_alpha(t, hold, fade);

        match self.display.style {
            Style::Classic => {
                let slide = (1.0 - ease_out(t / (fade * 1.4))) * 16.0;
                let parts = vec![(text, base)];
                self.draw_block(&painter, center + Vec2::new(0.0, slide), alpha, 1.0, 1.0, &parts, orig, true);
            }
            Style::Pop => {
                let tin = 0.32 / anim.clamp(0.25, 4.0);
                let s_in = lerp(0.6, 1.0, ease_out_back(t / tin));
                let fout = ((hold - t) / fade).clamp(0.0, 1.0);
                let scale = s_in * lerp(0.94, 1.0, fout);
                let a_in = (t / (0.12 / anim.clamp(0.25, 4.0))).clamp(0.0, 1.0);
                let parts = vec![(text, base)];
                self.draw_block(&painter, center, a_in.min(fout), scale, 1.0, &parts, orig, true);
            }
            Style::Slide => {
                let dist = rect.width() * 0.10;
                let tin = 0.35 / anim.clamp(0.25, 4.0);
                let fout = ((hold - t) / fade).clamp(0.0, 1.0);
                let x = -dist * (1.0 - ease_out(t / tin)) + dist * ease_in(1.0 - fout);
                let a_in = ease_out(t / (tin * 0.6));
                let parts = vec![(text, base)];
                self.draw_block(&painter, center + Vec2::new(x, 0.0), a_in.min(fout), 1.0, 1.0, &parts, orig, true);
            }
            Style::Typewriter => {
                let total = text.chars().count();
                let n = typed_chars(total, t, hold, anim);
                let split = text.char_indices().nth(n).map_or(text.len(), |(i, _)| i);
                let parts = vec![
                    (text[..split].to_string(), base),
                    (text[split..].to_string(), Color32::TRANSPARENT),
                ];
                let drawn = self.draw_block(&painter, center, alpha, 1.0, 1.0, &parts, orig, true);
                let typing = n < total;
                if let Some(d) = drawn {
                    if typing || (t * 2.2).fract() < 0.6 {
                        let r = d.galley.pos_from_cursor(&d.galley.from_ccursor(CCursor::new(n)));
                        let w = (self.display.font_size * 0.07).max(2.0);
                        let caret = Rect::from_min_size(
                            d.pos + r.min.to_vec2() + Vec2::new(2.0, 0.0),
                            Vec2::new(w, r.height()),
                        );
                        painter.rect_filled(caret, 1.0, with_alpha(accent, alpha));
                    }
                }
            }
            Style::Karaoke => {
                let words: Vec<&str> = text.split_whitespace().collect();
                let pos = karaoke_pos(words.len(), t, hold, fade);
                let dim = with_alpha(base, 0.5);
                let n = words.len();
                let parts: Parts = words
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let lit = (pos - i as f32).clamp(0.0, 1.0);
                        let s = if i + 1 < n { format!("{w} ") } else { w.to_string() };
                        (s, lerp_color(dim, accent, lit))
                    })
                    .collect();
                let rise = (1.0 - ease_out(t / (fade * 1.4))) * 10.0;
                self.draw_block(&painter, center + Vec2::new(0.0, rise), alpha, 1.0, 1.0, &parts, orig, true);
            }
            Style::Lyrics => {
                let fs = self.display.font_size;
                let (y_cur, y_prev) = (center.y + fs * 0.55, center.y - fs * 0.75);
                let since = now.duration_since(self.prev_since).as_secs_f32();
                let e = ease_out(since / (0.38 / anim.clamp(0.25, 4.0)));
                // Lines scroll on; only the last line of a run fades out.
                let out = if self.queue.is_empty() { ((hold - t) / fade).clamp(0.0, 1.0) } else { 1.0 };
                let has_prev = self.prev.is_some();
                if let Some(p) = &self.prev {
                    let size_mul = lerp(1.12, 0.72, e);
                    let color = lerp_color(accent, with_alpha(base, 0.55), e);
                    let parts = vec![(p.text.clone(), color)];
                    let c = Pos2::new(center.x, lerp(y_cur, y_prev, e));
                    self.draw_block(&painter, c, out, 1.0, size_mul, &parts, None, false);
                }
                let a_in = if has_prev { e } else { ease_out(t / fade) };
                let rise = if has_prev { (1.0 - e) * fs * 0.9 } else { (1.0 - ease_out(t / (fade * 1.6))) * 12.0 };
                let parts = vec![(text, accent)];
                let c = Pos2::new(center.x, y_cur + rise);
                self.draw_block(&painter, c, a_in.min(out), 1.0, 1.12, &parts, orig, false);
            }
        }
    }

    /// Draws one block of (optionally multi-coloured) text with its box/outline/shadow and the
    /// optional small original text below. Returns where the main galley was painted.
    #[allow(clippy::too_many_arguments)]
    fn draw_block(
        &self,
        painter: &Painter,
        center: Pos2,
        alpha: f32,
        scale: f32,
        size_mul: f32,
        parts: &Parts,
        original: Option<&str>,
        boxed: bool,
    ) -> Option<Drawn> {
        if alpha <= 0.003 {
            return None;
        }
        let d = &self.display;
        let size = (d.font_size * size_mul * scale).max(6.0);
        let faded: Parts = parts.iter().map(|(s, c)| (s.clone(), with_alpha(*c, alpha))).collect();
        let fill = layout(painter, &faded, size, self.wrap, false);
        let outl = layout(painter, &faded, size, self.wrap, true);

        let orig = original.filter(|_| d.show_original).map(|o| {
            let p: Parts = vec![(o.to_string(), with_alpha(col(d.text_color), 0.6 * alpha))];
            (layout(painter, &p, size * 0.5, self.wrap, false), layout(painter, &p, size * 0.5, self.wrap, true))
        });

        let gap = 4.0 * scale;
        let (ow, oh) = orig.as_ref().map_or((0.0, 0.0), |(g, _)| (g.size().x, g.size().y + gap));
        let total = Vec2::new(fill.size().x.max(ow), fill.size().y + oh);

        let has_box = boxed && d.bg_color[3] > 0;
        if has_box {
            let r = Rect::from_center_size(center, total + Vec2::new(44.0, 22.0) * scale);
            painter.rect_filled(r, 14.0 * scale, with_alpha(col(d.bg_color), alpha));
        }

        let top = center.y - total.y / 2.0;
        let main_center = Pos2::new(center.x, top + fill.size().y / 2.0);
        let main_pos = main_center - fill.rect.center().to_vec2();
        paint_text(painter, main_pos, &fill, &outl, size, d.outline, !has_box);
        if let Some((og, oo)) = orig {
            let oc = Pos2::new(center.x, top + fill.size().y + gap + og.size().y / 2.0);
            paint_text(painter, oc - og.rect.center().to_vec2(), &og, &oo, size * 0.5, d.outline, !has_box);
        }
        Some(Drawn { pos: main_pos, galley: fill })
    }
}

struct Drawn {
    pos: Pos2,
    galley: Arc<Galley>,
}

/// Lays out coloured parts as one centred, wrapping paragraph. `outline` swaps every colour for
/// a dark one with the same alpha (used for the outline/shadow pass).
fn layout(painter: &Painter, parts: &Parts, size: f32, wrap: f32, outline: bool) -> Arc<Galley> {
    let mut job = LayoutJob::default();
    job.wrap.max_width = wrap;
    job.halign = Align::Center;
    for (s, c) in parts {
        let color = if outline {
            Color32::from_rgba_unmultiplied(0, 0, 0, (c.a() as f32 * 0.9) as u8)
        } else {
            *c
        };
        job.append(s, 0.0, TextFormat { font_id: FontId::proportional(size), color, ..Default::default() });
    }
    painter.layout_job(job)
}

fn paint_text(painter: &Painter, pos: Pos2, fill: &Arc<Galley>, outl: &Arc<Galley>, size: f32, outline: bool, shadow: bool) {
    if outline {
        let r = (size * 0.045).max(1.0);
        for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)] {
            painter.galley(pos + Vec2::new(dx * r, dy * r), outl.clone(), Color32::TRANSPARENT);
        }
    } else if shadow {
        // No box and no outline: a soft drop shadow keeps the text readable.
        painter.galley(pos + Vec2::new(0.0, size * 0.06), outl.clone(), Color32::TRANSPARENT);
    }
    painter.galley(pos, fill.clone(), Color32::WHITE);
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_respects_limit_and_commas() {
        let s = "Bu modül aslında onu atlıyor, ki bu çılgınca, çünkü geçen hafta beni iki dakikadan az bir sürede yasakladı.";
        let c = chunks(s, 56);
        assert!(c.iter().all(|p| p.chars().count() <= 56), "{c:?}");
        assert_eq!(c.join(" "), s);
        assert!(c.len() >= 2);
        assert_eq!(chunks("Kısa cümle.", 56), vec!["Kısa cümle."]);
        assert!(chunks("", 56).is_empty());
        // Never panics on tiny limits, long words stay whole.
        assert!(!chunks("supercalifragilisticexpialidocious word", 3).is_empty());
    }

    #[test]
    fn hold_time_scales_with_length_and_backlog() {
        let short = hold_time(10, 24.0, 0);
        let long = hold_time(56, 24.0, 0);
        assert!(short >= 0.9 && long > short && long <= 3.2);
        assert!(hold_time(56, 24.0, 3) < hold_time(56, 24.0, 1));
        assert!(hold_time(1, 24.0, 9) >= 0.8);
        assert!(hold_time(40, 12.0, 0) > hold_time(40, 48.0, 0));
    }

    #[test]
    fn easing_functions() {
        for f in [ease_out, ease_in, ease_out_back] {
            assert!(f(0.0).abs() < 1e-6);
            assert!((f(1.0) - 1.0).abs() < 1e-5);
        }
        assert!(ease_out(0.5) > 0.5 && ease_in(0.5) < 0.5);
        let peak = (0..=100).map(|i| ease_out_back(i as f32 / 100.0)).fold(0.0_f32, f32::max);
        assert!(peak > 1.05 && peak < 1.2, "overshoot {peak}");
        assert_eq!(ease_out(-1.0), 0.0);
        assert_eq!(ease_out(2.0), 1.0);
    }

    #[test]
    fn fades() {
        let hold = 2.0;
        let fade = fade_time(1.0, hold);
        assert!((fade - 0.18).abs() < 1e-6);
        assert!(fade_time(2.0, hold) < fade && fade_time(0.5, hold) > fade);
        assert!(fade_time(0.25, 0.9) <= 0.3 + 1e-6); // never more than hold / 3
        assert!(in_out_alpha(0.0, hold, fade) < 0.01);
        assert!(in_out_alpha(1.0, hold, fade) > 0.99);
        assert!(in_out_alpha(hold - 0.01, hold, fade) < 0.1);
    }

    #[test]
    fn karaoke_words_light_up_in_order() {
        let (hold, fade) = (2.0, 0.18);
        assert_eq!(karaoke_pos(5, 0.0, hold, fade), 0.0);
        assert_eq!(karaoke_pos(5, hold, hold, fade), 5.0);
        let mut last = 0.0;
        for i in 0..=20 {
            let p = karaoke_pos(5, i as f32 * 0.1, hold, fade);
            assert!(p >= last);
            last = p;
        }
        assert!(karaoke_pos(5, hold * 0.4, hold, fade) < 5.0);
        assert_eq!(karaoke_pos(0, 1.0, hold, fade), 0.0);
    }

    #[test]
    fn typewriter_counts() {
        assert_eq!(typed_chars(0, 1.0, 2.0, 1.0), 0);
        assert_eq!(typed_chars(40, 0.0, 2.0, 1.0), 0);
        assert_eq!(typed_chars(40, 5.0, 2.0, 1.0), 40);
        assert!(typed_chars(40, 0.3, 2.0, 1.0) < typed_chars(40, 0.6, 2.0, 1.0));
        assert!(typed_chars(40, 0.3, 2.0, 2.0) > typed_chars(40, 0.3, 2.0, 1.0));
        // Always finished before the hold ends.
        assert_eq!(typed_chars(400, 2.0 * 0.75, 2.0, 0.5), 400);
    }

    fn subs() -> Subtitles {
        Subtitles::new(&Display::default(), &Speed::default())
    }

    #[test]
    fn queue_advances_and_keeps_previous_line() {
        let mut s = subs();
        assert!(s.is_idle());
        s.push("Birinci cümle burada.", Some("First sentence here."));
        s.push("İkinci cümle burada.", None);
        assert!(!s.is_idle());
        let t0 = Instant::now();
        s.tick(t0);
        assert_eq!(s.current.as_ref().unwrap().piece.text, "Birinci cümle burada.");
        assert!(s.prev.is_none());
        // Not expired yet: unchanged.
        s.tick(t0 + Duration::from_millis(300));
        assert_eq!(s.current.as_ref().unwrap().piece.text, "Birinci cümle burada.");
        // Expired: next piece, first one becomes `prev`.
        let hold = s.current.as_ref().unwrap().hold;
        s.tick(t0 + Duration::from_secs_f32(hold + 0.01));
        assert_eq!(s.current.as_ref().unwrap().piece.text, "İkinci cümle burada.");
        assert_eq!(s.prev.as_ref().unwrap().text, "Birinci cümle burada.");
        // Expired with nothing waiting: idle again, no stale previous line.
        let hold2 = s.current.as_ref().unwrap().hold;
        s.tick(t0 + Duration::from_secs_f32(hold + hold2 + 0.1));
        assert!(s.is_idle() && s.prev.is_none());
    }

    #[test]
    fn backlog_drops_oldest_and_speeds_up() {
        let mut s = subs();
        for i in 0..12 {
            s.push(&format!("Cümle {i}."), None);
        }
        s.tick(Instant::now());
        assert!(s.queue.len() <= MAX_BACKLOG);
        // The oldest were dropped, the newest are kept.
        assert_eq!(s.queue.back().unwrap().text, "Cümle 11.");
        let backlog_hold = s.current.as_ref().unwrap().hold;
        let mut fresh = subs();
        fresh.push("Cümle 0.", None);
        fresh.tick(Instant::now());
        assert!(backlog_hold < fresh.current.as_ref().unwrap().hold);
    }

    #[test]
    fn original_text_only_on_first_piece() {
        let mut s = subs();
        let long = "Bu oldukça uzun bir cümle, çünkü birkaç parçaya bölünmesi gerekiyor ve hepsini göstermek istiyoruz.";
        s.push(long, Some("original"));
        assert!(s.queue.len() >= 2);
        assert_eq!(s.queue[0].original.as_deref(), Some("original"));
        assert!(s.queue.iter().skip(1).all(|p| p.original.is_none()));
    }

    #[test]
    fn settings_apply_without_losing_queue() {
        let mut s = subs();
        s.push("Merhaba dünya.", None);
        let d = Display { max_chars: 20, style: Style::Lyrics, ..Display::default() };
        s.set_settings(&d, &Speed::default());
        assert_eq!(s.queue.len(), 1);
        assert_eq!(s.display.style, Style::Lyrics);
    }

    #[test]
    fn window_geometry_positions() {
        let d = Display::default();
        let (w, h, x, y) = window_geometry(&d, 1920.0, 1080.0);
        assert!((w - 1344.0).abs() < 0.5 && (x - 288.0).abs() < 0.5);
        assert!((h - 32.0 * 4.2).abs() < 0.01);
        assert!((y - (1080.0 - h - 80.0)).abs() < 0.01);
        let top = Display { position: Position::Top, margin_px: 30.0, ..Display::default() };
        assert_eq!(window_geometry(&top, 1920.0, 1080.0).3, 30.0);
        let center = Display { position: Position::Center, ..Display::default() };
        let (_, hc, _, yc) = window_geometry(&center, 1920.0, 1080.0);
        assert!((yc - (1080.0 - hc) / 2.0).abs() < 0.01);
        let orig = Display { show_original: true, ..Display::default() };
        assert!(window_geometry(&orig, 1920.0, 1080.0).1 > h);
        // Absurd settings stay on screen.
        let wild = Display { width_pct: 500.0, margin_px: 9999.0, ..Display::default() };
        let (w2, _, x2, y2) = window_geometry(&wild, 1920.0, 1080.0);
        assert!(w2 <= 1920.0 && x2 >= 0.0 && y2 >= 0.0);
    }
}
