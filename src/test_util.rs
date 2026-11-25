//! What the unit tests share.

/// A small deterministic generator (xorshift64).
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value below `n`.
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A value of at most `bits` bits.
    pub(crate) fn bits(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            0
        } else {
            (self.next() & (u64::MAX >> (64 - bits))) as u32
        }
    }
}

/// A directory named after the test and the process, removed at the end.
#[cfg(feature = "mmap")]
pub(crate) struct TempDir(pub(crate) std::path::PathBuf);

#[cfg(feature = "mmap")]
impl TempDir {
    pub(crate) fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(std::format!("hashtally-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a fresh directory");
        Self(dir)
    }
}

#[cfg(feature = "mmap")]
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
