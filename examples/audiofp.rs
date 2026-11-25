//! Fingerprint audio with the [`audiofp`](https://crates.io/crates/audiofp)
//! crate's Wang fingerprinter and search the fingerprints with this one.
//!
//! ```bash
//! cargo run --release --example audiofp
//! ```
//!
//! The two crates meet at one row type. `audiofp` turns 8 kHz mono audio
//! into `WangHash { hash, t_anchor }`, `t_anchor` being an STFT frame at
//! 62.5 frames per second, and each `(hash, t_anchor)` is a row here. A
//! candidate's Δ is then in frames too: where the clip starts in the
//! reference, when the clip's own frames start at 0.
//!
//! The audio is synthetic so the example runs anywhere: tones that change
//! every 125 ms over a little noise. Real audio must first be decoded,
//! mixed down to mono and resampled to 8 kHz, and references and queries
//! must be fingerprinted with the same `WangConfig`.

use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};
use hashtally::{Catalog, Query, Scratch, SearchOptions, vote_floor};

/// Samples per second that `Wang` takes.
const RATE: usize = 8_000;
/// Samples between STFT frames: 62.5 frames per second at [`RATE`].
const HOP: usize = 128;

/// The rows of 8 kHz mono audio: one per Wang hash.
fn rows(samples: &[f32]) -> Vec<(u32, u32)> {
    let fingerprint = Wang::default()
        .extract(samples, SampleRate::HZ_8000)
        .expect("at least a few seconds of audio");
    fingerprint
        .hashes
        .iter()
        .map(|h| (h.hash, h.t_anchor))
        .collect()
}

/// `secs` seconds of audio that depend only on `seed`: three tones that
/// change every 125 ms, over a little noise.
fn track(seed: u64, secs: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut tones = [0.0f64; 3];
    (0..secs * RATE)
        .map(|i| {
            if i % (RATE / 8) == 0 {
                for hz in &mut tones {
                    *hz = 200.0 + 3_000.0 * uniform();
                }
            }
            let t = i as f64 / RATE as f64;
            let tone = tones
                .iter()
                .map(|hz| (std::f64::consts::TAU * hz * t).sin())
                .sum::<f64>()
                / 3.0;
            (0.8 * tone + 0.05 * (uniform() - 0.5)) as f32
        })
        .collect()
}

fn main() {
    // Three one-minute tracks; the catalog numbers them in the order they
    // are added, so keep your own table from those ids to your tracks.
    let tracks: Vec<Vec<f32>> = (1..=3).map(|seed| track(seed, 60)).collect();
    let mut builder = Catalog::builder();
    let ids: Vec<u32> = tracks.iter().map(|t| builder.add(rows(t))).collect();
    let catalog = builder.build();
    println!(
        "catalog: {} references, {} postings",
        ids.len(),
        catalog.posting_count()
    );

    // The query: ten seconds of the second track from 20 s on, with
    // another track playing quietly underneath.
    let start = 20 * RATE;
    let clip: Vec<f32> = tracks[1][start..start + 10 * RATE]
        .iter()
        .zip(track(99, 10))
        .map(|(&s, other)| s + 0.25 * other)
        .collect();
    let pairs = rows(&clip);
    let query = Query::from_pairs(pairs.iter().copied(), Some(64));
    let mut options = SearchOptions::default();
    options.min_total_votes = vote_floor(pairs.len(), 16, 0.02);

    let found = catalog
        .search(&query, &options, &mut Scratch::new())
        .expect("valid options");
    let frames_per_sec = RATE as f64 / HOP as f64;
    for c in &found {
        println!(
            "reference {} at {:.2} s: {} votes in the best bin of {} total",
            c.reference,
            c.delta as f64 / frames_per_sec,
            c.votes,
            c.total_votes
        );
    }
    let best = found.iter().max_by_key(|c| c.votes).expect("a match");
    assert_eq!(best.reference, ids[1]);
    assert_eq!(best.delta, (start / HOP) as i64);
}
