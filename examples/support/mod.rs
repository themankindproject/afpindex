//! What the synthetic examples share: a seeded generator and the rows of a
//! synthetic reference, which are a pure function of its id, so an example
//! never keeps a second copy of the catalog it builds.

#![allow(dead_code, reason = "each example uses part of it")]

/// A xorshift generator: fast, seeded, and the same on every platform.
pub struct Rng(u64);

impl Rng {
    /// A generator for `seed`, past its first few outputs.
    pub fn seeded(seed: u64) -> Self {
        let mut r = Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }

    /// The next 64 random bits.
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value below `n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Frames in 45 minutes at 62.5 frames per second: a reference's times.
pub const MAX_TIME: u64 = 168_750;

/// The rows of synthetic reference `reference`: `per_ref` of them, hashes
/// uniform over `hashes` and times over 45 minutes.
pub fn rows_of(reference: u32, per_ref: usize, hashes: u64) -> Vec<(u32, u32)> {
    rows_in(reference, per_ref, hashes, MAX_TIME)
}

/// As [`rows_of`], with times below `max_time`.
pub fn rows_in(reference: u32, per_ref: usize, hashes: u64, max_time: u64) -> Vec<(u32, u32)> {
    let mut rng = Rng::seeded(u64::from(reference) + 1);
    (0..per_ref)
        .map(|_| (rng.below(hashes) as u32, rng.below(max_time) as u32))
        .collect()
}
