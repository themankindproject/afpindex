//! Snapshot files (`std`): little-endian arrays behind a small header,
//! every section 8-byte aligned, with a checksum trailer.
//!
//! Segment snapshot, version 2:
//!
//! ```text
//! 0   "AFIX"            4   version u32 = 2
//! 8   ref_bits u32      12  time_bits u32
//! 16  key_count u64     24  posting_count u64
//! 32  keys    u32 × key_count                             (zero-padded to 8)
//!     offsets u64 × (key_count + 1)
//!     refs    ceil(posting_count × ref_bits / 8) bytes    (zero-padded to 8)
//!     times   ceil(posting_count × time_bits / 8) bytes   (zero-padded to 8)
//!     checksum u64 over the header words (widths, key_count,
//!     posting_count), the keys and offsets as values, and the packed
//!     bytes as little-endian 64-bit words
//! ```
//!
//! References and times are packed at the width of the largest value in
//! the segment, little-endian bit order, as held in memory.
//!
//! Catalog snapshot, version 2:
//!
//! ```text
//! 0   "AFCT"            4   version u32 = 2
//! 8   flags u32 = 0     12  reserved u32 = 0
//! 16  reference_count u64   24  segment_count u64
//! 32  live bitmap, one bit per id, LSB first, ceil(n / 8) bytes (zero-padded to 8)
//!     checksum u64 over the header counts and the bitmap bytes
//!     segment_count segment snapshots, back to back
//! ```
//!
//! Flags and reserved fields are written as zero and ignored on read.

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread;
use std::vec::Vec;

use crate::packed::Packed;
use crate::segment::MAX_POSTINGS;
use crate::simd::{self, Decoder, LANES, Pass};
use crate::{Catalog, Segment, SnapshotError};

const SEGMENT_MAGIC: [u8; 4] = *b"AFIX";
const CATALOG_MAGIC: [u8; 4] = *b"AFCT";
const VERSION: u32 = 2;
const HEADER: usize = 32;
/// Bytes read or written per call while streaming an array.
const CHUNK: usize = 1 << 18;
/// Sanity bound on the segment count a catalog snapshot may declare; the
/// merge policy keeps real catalogs far below it.
const MAX_SEGMENTS: u64 = 1 << 16;

/// Order-sensitive 64-bit checksum over words: catches corruption, not
/// tampering.
struct Checksum(u64);

impl Checksum {
    const PRIME: u64 = 0x9E37_79B9_7F4A_7C15;

    fn new() -> Self {
        Self(0x4146_4958_0000_0001)
    }

    #[inline]
    fn word(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(29) ^ w).wrapping_mul(Self::PRIME);
    }

    /// Fold in `bytes` as little-endian 64-bit words, the last one
    /// zero-padded: how the packed arrays are covered. Bytes fed in several
    /// calls must be cut at multiples of 8.
    fn words(&mut self, bytes: &[u8]) {
        let mut state = Self(self.0);
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            state.word(u64::from_le_bytes(w.try_into().expect("8 bytes")));
        }
        let rest = words.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            state.word(u64::from_le_bytes(last));
        }
        *self = state;
    }

    fn finish(self) -> u64 {
        let mut h = self.0;
        h ^= h >> 32;
        h = h.wrapping_mul(Self::PRIME);
        h ^ (h >> 29)
    }
}

/// Zero bytes that bring `pos` to the next multiple of 8.
fn pad_len(pos: u64) -> usize {
    (pos.next_multiple_of(8) - pos) as usize
}

fn write_pad<W: Write>(w: &mut W, pos: usize) -> std::io::Result<usize> {
    let pad = pad_len(pos as u64);
    w.write_all(&[0u8; 8][..pad])?;
    Ok(pos + pad)
}

fn write_u32s<W: Write>(
    w: &mut W,
    values: impl Iterator<Item = u32>,
    ck: &mut Checksum,
    pos: usize,
) -> std::io::Result<usize> {
    let mut buf = Vec::with_capacity(CHUNK);
    let mut n = 0;
    for v in values {
        ck.word(u64::from(v));
        buf.extend_from_slice(&v.to_le_bytes());
        n += 1;
        if buf.len() >= CHUNK {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    Ok(pos + n * 4)
}

fn write_u64s<W: Write>(
    w: &mut W,
    values: impl Iterator<Item = u64>,
    ck: &mut Checksum,
    pos: usize,
) -> std::io::Result<usize> {
    let mut buf = Vec::with_capacity(CHUNK);
    let mut n = 0;
    for v in values {
        ck.word(v);
        buf.extend_from_slice(&v.to_le_bytes());
        n += 1;
        if buf.len() >= CHUNK {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    Ok(pos + n * 8)
}

/// Write a header: magic, version, two 32-bit fields, two counts.
fn write_header<W: Write>(
    w: &mut W,
    magic: [u8; 4],
    fields: (u32, u32),
    a: u64,
    b: u64,
) -> std::io::Result<()> {
    w.write_all(&magic)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&fields.0.to_le_bytes())?;
    w.write_all(&fields.1.to_le_bytes())?;
    w.write_all(&a.to_le_bytes())?;
    w.write_all(&b.to_le_bytes())
}

/// A header's two 32-bit fields and two counts.
type Header = ((u32, u32), u64, u64);

/// Read a header of `magic`.
fn read_header<R: Read>(r: &mut R, magic: [u8; 4]) -> Result<Header, SnapshotError> {
    let mut head = [0u8; HEADER];
    r.read_exact(&mut head)?;
    if head[..4] != magic {
        return Err(SnapshotError::BadMagic);
    }
    let version = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
    if version != VERSION {
        return Err(SnapshotError::UnsupportedVersion(version));
    }
    let f0 = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
    let f1 = u32::from_le_bytes([head[12], head[13], head[14], head[15]]);
    let a = u64::from_le_bytes(head[16..24].try_into().expect("8 bytes"));
    let b = u64::from_le_bytes(head[24..32].try_into().expect("8 bytes"));
    Ok(((f0, f1), a, b))
}

/// Skip the padding after `pos` bytes; positions are 64-bit so a crafted
/// count cannot wrap them on 32-bit targets.
fn read_pad<R: Read>(r: &mut R, pos: u64) -> Result<u64, SnapshotError> {
    let pad = pad_len(pos);
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf[..pad])?;
    Ok(pos + pad as u64)
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, SnapshotError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Grow `out` for at least `take` more items, doubling when possible but
/// never past `remaining`, so a corrupt count cannot force a huge
/// allocation and a correct one ends with no spare capacity.
fn grow<T>(out: &mut Vec<T>, take: usize, remaining: usize) {
    let spare = out.capacity() - out.len();
    if spare < take {
        out.reserve_exact(remaining.min(out.len().max(take)));
    }
}

fn read_u32s<R: Read>(r: &mut R, n: usize, ck: &mut Checksum) -> Result<Vec<u32>, SnapshotError> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; CHUNK];
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK / 4);
        r.read_exact(&mut buf[..take * 4])?;
        grow(&mut out, take, remaining);
        for c in buf[..take * 4].chunks_exact(4) {
            let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            ck.word(u64::from(v));
            out.push(v);
        }
        remaining -= take;
    }
    Ok(out)
}

fn read_u64s<R: Read>(r: &mut R, n: usize, ck: &mut Checksum) -> Result<Vec<u64>, SnapshotError> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; CHUNK];
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK / 8);
        r.read_exact(&mut buf[..take * 8])?;
        grow(&mut out, take, remaining);
        for c in buf[..take * 8].chunks_exact(8) {
            let v = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            ck.word(v);
            out.push(v);
        }
        remaining -= take;
    }
    Ok(out)
}

/// Append exactly `n` bytes from `r` to `out`, which has room for them:
/// the reader fills the spare capacity itself, with no buffer in between.
fn read_into<R: Read>(r: &mut R, out: &mut Vec<u8>, n: usize) -> Result<(), SnapshotError> {
    let got = r.by_ref().take(n as u64).read_to_end(out)?;
    if got == n {
        Ok(())
    } else {
        Err(SnapshotError::Truncated)
    }
}

/// Read `n` bytes, each checksummed as a word of its own (the liveness
/// bitmap).
fn read_bytes<R: Read>(r: &mut R, n: usize, ck: &mut Checksum) -> Result<Vec<u8>, SnapshotError> {
    let mut out = Vec::new();
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK);
        grow(&mut out, take, remaining);
        let at = out.len();
        read_into(r, &mut out, take)?;
        for &b in &out[at..] {
            ck.word(u64::from(b));
        }
        remaining -= take;
    }
    Ok(out)
}

/// Write packed bytes, checksummed as little-endian 64-bit words (the
/// last one zero-padded).
fn write_packed<W: Write>(
    w: &mut W,
    bytes: &[u8],
    ck: &mut Checksum,
    pos: usize,
) -> std::io::Result<usize> {
    ck.words(bytes);
    w.write_all(bytes)?;
    Ok(pos + bytes.len())
}

/// Read `n` values of `bits` packed, checksummed as [`write_packed`] does.
fn read_packed<R: Read>(
    r: &mut R,
    n: usize,
    bits: u32,
    ck: &mut Checksum,
) -> Result<Packed, SnapshotError> {
    if bits > u32::BITS {
        return Err(SnapshotError::Corrupt("a width exceeds 32 bits"));
    }
    let mut out = Vec::new();
    let mut remaining = Packed::data_len(n, bits);
    // `CHUNK` is a multiple of 8, so word boundaries match the writer's.
    while remaining > 0 {
        let take = remaining.min(CHUNK);
        grow(&mut out, take, remaining.saturating_add(Packed::SLACK));
        let at = out.len();
        read_into(r, &mut out, take)?;
        ck.words(&out[at..]);
        remaining -= take;
    }
    Packed::from_bytes(out, n, bits).ok_or(SnapshotError::Corrupt("packed array length"))
}

fn to_usize(n: u64) -> Result<usize, SnapshotError> {
    usize::try_from(n).map_err(|_| SnapshotError::Corrupt("a count exceeds the address space"))
}

/// Start a checksum with the header words it covers.
fn checksum_of(fields: (u32, u32), a: u64, b: u64) -> Checksum {
    let mut ck = Checksum::new();
    ck.word(u64::from(fields.0) | u64::from(fields.1) << 32);
    ck.word(a);
    ck.word(b);
    ck
}

fn write_segment<W: Write>(seg: &Segment, w: &mut W) -> std::io::Result<()> {
    let (n_keys, n_post) = (seg.key_count() as u64, seg.len() as u64);
    write_header(w, SEGMENT_MAGIC, seg.widths(), n_keys, n_post)?;
    let mut ck = checksum_of(seg.widths(), n_keys, n_post);
    let mut pos = HEADER;
    pos = write_u32s(w, seg.keys().iter().copied(), &mut ck, pos)?;
    pos = write_pad(w, pos)?;
    pos = write_u64s(w, seg.offsets(), &mut ck, pos)?;
    pos = write_packed(w, seg.refs().data(), &mut ck, pos)?;
    pos = write_pad(w, pos)?;
    pos = write_packed(w, seg.times().data(), &mut ck, pos)?;
    write_pad(w, pos)?;
    w.write_all(&ck.finish().to_le_bytes())
}

/// The key and posting counts of a segment header as sizes, refused when
/// implausible before anything is allocated.
fn segment_counts(n_keys: u64, n_post: u64) -> Result<(usize, usize), SnapshotError> {
    if n_post >= MAX_POSTINGS {
        return Err(SnapshotError::Corrupt("too many postings for a segment"));
    }
    // Distinct `u32` keys: at most 2^32 of them.
    if n_keys > 1 << 32 {
        return Err(SnapshotError::Corrupt("too many keys for a segment"));
    }
    let n_keys = to_usize(n_keys)?;
    let n_post = to_usize(n_post)?;
    if n_keys > n_post {
        return Err(SnapshotError::Corrupt("more keys than postings"));
    }
    Ok((n_keys, n_post))
}

/// Refuse a segment with a reference a catalog of `n_refs` ids cannot
/// hold.
fn check_id_space(ids: Ids, n_refs: usize) -> Result<(), SnapshotError> {
    if ids.is_some_and(|(_, max)| max as usize >= n_refs) {
        return Err(SnapshotError::Corrupt(
            "a posting references an id outside the id space",
        ));
    }
    Ok(())
}

fn read_segment<R: Read>(r: &mut R) -> Result<Segment, SnapshotError> {
    read_segment_ids(r).map(|(seg, _)| seg)
}

/// The smallest and the largest reference of a segment, if it has any.
type Ids = Option<(u32, u32)>;

/// Read a segment snapshot and report the range of its references.
fn read_segment_ids<R: Read>(r: &mut R) -> Result<(Segment, Ids), SnapshotError> {
    let (widths, n_keys, n_post) = read_header(r, SEGMENT_MAGIC)?;
    let (ref_bits, time_bits) = widths;
    let mut ck = checksum_of(widths, n_keys, n_post);
    let (n_keys, n_post) = segment_counts(n_keys, n_post)?;
    let mut pos = HEADER as u64;
    let keys = read_u32s(r, n_keys, &mut ck)?;
    pos = read_pad(r, pos + n_keys as u64 * 4)?;
    let offsets = read_u64s(r, n_keys + 1, &mut ck)?;
    pos += (n_keys as u64 + 1) * 8;
    let refs = read_packed(r, n_post, ref_bits, &mut ck)?;
    pos = read_pad(r, pos + refs.data().len() as u64)?;
    let times = read_packed(r, n_post, time_bits, &mut ck)?;
    read_pad(r, pos + times.data().len() as u64)?;
    if read_u64(r)? != ck.finish() {
        return Err(SnapshotError::Checksum);
    }
    // Checksummed as it was read.
    finish_segment(widths, n_post, keys, &offsets, refs, times, |_, _| true)
}

/// Segments with at least this many postings have their runs checked on
/// every core, and on the mapped path checksummed on a core of their own
/// at the same time.
#[cfg(not(test))]
const PARALLEL_MIN: usize = 1 << 22;
/// Small enough for the tests' segments to take the parallel path.
#[cfg(test)]
const PARALLEL_MIN: usize = 1 << 10;

/// Check that the arrays describe a valid segment (keys ascending, offsets
/// consistent, each run sorted by `(reference, time)` without duplicates)
/// and assemble it. The walk over the runs ([`CheckRuns`]) also records
/// each run's multiplicity for its directory slot and the range of the
/// references, so nothing reads the postings a second time at load.
///
/// `checksum` checks the packed arrays, unless the caller already has (it
/// then returns `true`); a mismatch is reported before anything else,
/// though the runs are checked while it runs.
fn finish_segment(
    (ref_bits, time_bits): (u32, u32),
    n_post: usize,
    keys: Vec<u32>,
    offsets: &[u64],
    refs: Packed,
    times: Packed,
    checksum: impl FnOnce(&Packed, &Packed) -> bool + Send,
) -> Result<(Segment, Ids), SnapshotError> {
    let index = check_index(n_post, &keys, offsets);
    let check = CheckRuns {
        // Distinct `(reference, time)` pairs bound a run's length by the
        // widths, which keeps the scan proportional to the bytes read.
        longest: 1u64 << (ref_bits + time_bits).min(63),
        offsets,
        refs: &refs,
        times: &times,
    };
    // The walk relies on the index, so it runs only once that holds.
    let (summed, walk) = if index.is_ok() {
        check_runs(check, || checksum(&refs, &times))
    } else {
        (checksum(&refs, &times), Ok((Vec::new(), None)))
    };
    if !summed {
        return Err(SnapshotError::Checksum);
    }
    index?;
    let (mults, ids) = walk?;
    Ok((Segment::from_parts(keys, offsets, refs, times, &mults), ids))
}

/// Keys strictly ascending, and offsets strictly ascending from 0 to the
/// posting count: what the walk over the runs relies on.
fn check_index(n_post: usize, keys: &[u32], offsets: &[u64]) -> Result<(), SnapshotError> {
    if !keys.windows(2).all(|w| w[0] < w[1]) {
        return Err(SnapshotError::Corrupt("keys not strictly ascending"));
    }
    if offsets.first() != Some(&0) || offsets.last().copied() != Some(n_post as u64) {
        return Err(SnapshotError::Corrupt("offsets do not span the postings"));
    }
    if !offsets.windows(2).all(|w| w[0] < w[1]) {
        return Err(SnapshotError::Corrupt("offsets not strictly ascending"));
    }
    Ok(())
}

/// Each run's multiplicity and the range of the references, or the error
/// of the first run in key order that fails the check.
type Walk = Result<(Vec<u8>, Ids), SnapshotError>;

/// Run `check` over every run, and `side` meanwhile; return what `side`
/// returned and the walk.
///
/// A large segment's runs are cut into pieces of about equal postings, a
/// few per core so that a slow piece does not hold the rest up, and every
/// thread takes the next piece until none are left; `side` is the first
/// piece taken. The calling thread works too, so a thread that cannot be
/// spawned costs nothing but parallelism. A small segment, or a single
/// core, does it all in order on the calling thread.
fn check_runs<T: Send>(check: CheckRuns<'_>, side: impl FnOnce() -> T + Send) -> (T, Walk) {
    let offsets = check.offsets;
    let n_keys = offsets.len() - 1;
    let n_post = offsets[n_keys];
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    if threads <= 1 || n_post < PARALLEL_MIN as u64 {
        let out = side();
        return (out, simd::run(check));
    }
    // Piece `i` is keys `bounds[i]..bounds[i + 1]`.
    let pieces = threads * 4;
    let mut bounds = vec![0];
    for piece in 1..pieces {
        let target = (u128::from(n_post) * piece as u128 / pieces as u128) as u64;
        let k = offsets.partition_point(|&o| o < target).min(n_keys);
        if k > bounds[bounds.len() - 1] {
            bounds.push(k);
        }
    }
    if bounds[bounds.len() - 1] < n_keys {
        bounds.push(n_keys);
    }
    let n = bounds.len() - 1;
    let next = AtomicUsize::new(0);
    let side = Mutex::new(Some(side));
    let side_out = Mutex::new(None);
    let walks: Vec<Mutex<Option<Walk>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let worker = || {
        loop {
            match next.fetch_add(1, Ordering::Relaxed) {
                0 => {
                    let side = side.lock().unwrap_or_else(PoisonError::into_inner).take();
                    if let Some(side) = side {
                        let out = side();
                        *side_out.lock().unwrap_or_else(PoisonError::into_inner) = Some(out);
                    }
                }
                i if i <= n => {
                    let piece = CheckRuns {
                        offsets: &offsets[bounds[i - 1]..=bounds[i]],
                        ..check
                    };
                    let walk = simd::run(piece);
                    *walks[i - 1].lock().unwrap_or_else(PoisonError::into_inner) = Some(walk);
                }
                _ => break,
            }
        }
    };
    thread::scope(|s| {
        for _ in 1..threads {
            // A failed spawn leaves its share to the others (the closure
            // captures only references, so it is `Copy`).
            let _ = thread::Builder::new().spawn_scoped(s, worker);
        }
        worker();
    });
    let out = side_out
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .expect("the first piece taken is `side`");
    let mut mults = Vec::with_capacity(n_keys);
    let mut ids: Ids = None;
    for walk in walks {
        let walk = walk
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .expect("every piece is taken");
        match walk {
            Ok((piece, piece_ids)) => {
                mults.extend_from_slice(&piece);
                ids = match (ids, piece_ids) {
                    (Some((lo, hi)), Some((a, b))) => Some((lo.min(a), hi.max(b))),
                    (a, b) => a.or(b),
                };
            }
            Err(e) => return (out, Err(e)),
        }
    }
    (out, Ok((mults, ids)))
}

/// The walk over every run of a segment being loaded: each run sorted by
/// `(reference, time)` without duplicates and no longer than `longest`.
/// Returns each run's multiplicity and the range of the references.
#[derive(Clone, Copy)]
struct CheckRuns<'a> {
    longest: u64,
    offsets: &'a [u64],
    refs: &'a Packed,
    times: &'a Packed,
}

const UNSORTED: SnapshotError = SnapshotError::Corrupt("a run is not sorted by (reference, time)");

impl Pass for CheckRuns<'_> {
    type Out = Result<(Vec<u8>, Ids), SnapshotError>;

    /// A posting is compared with the one before it as one word that
    /// orders as `(reference, time)` does. Where the decoder has a kernel,
    /// eight references and eight times are decoded and compared at a
    /// time, both arrays in the same loop; a block in which no reference
    /// repeats, which is most of them, needs no look at its streaks.
    #[inline(always)]
    fn run<D: Decoder>(self, decoder: D) -> Self::Out {
        let mut mults = Vec::with_capacity(self.offsets.len().saturating_sub(1));
        let mut ids: Ids = None;
        let (refs, times) = (self.refs.view(), self.times.view());
        let posting = |i: usize| u64::from(refs.get(i)) << 32 | u64::from(times.get(i));
        for w in self.offsets.windows(2) {
            if w[1] - w[0] > self.longest {
                return Err(SnapshotError::Corrupt(
                    "a run is longer than its widths allow",
                ));
            }
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let (mut mult, mut streak) = (1u32, 1u32);
            let first = posting(lo);
            let mut prev = first;
            let mut i = lo + 1;
            if hi - i >= LANES {
                let blocks = self
                    .refs
                    .blocks(i, decoder)
                    .zip(self.times.blocks(i, decoder));
                if let Some((mut refs, mut times)) = blocks {
                    while hi - i >= LANES {
                        let (Some(r), Some(t)) = (refs.next(), times.next()) else {
                            break;
                        };
                        // Each lane's predecessor: the block one lane
                        // later, behind the posting before the block.
                        let (mut before_r, mut before_t) =
                            ([(prev >> 32) as u32; LANES], [prev as u32; LANES]);
                        before_r[1..].copy_from_slice(&r[..LANES - 1]);
                        before_t[1..].copy_from_slice(&t[..LANES - 1]);
                        let (mut sorted, mut repeats) = (true, false);
                        for lane in 0..LANES {
                            let same = r[lane] == before_r[lane];
                            sorted &=
                                r[lane] > before_r[lane] || (same && t[lane] > before_t[lane]);
                            repeats |= same;
                        }
                        if !sorted {
                            return Err(UNSORTED);
                        }
                        if repeats {
                            for lane in 0..LANES {
                                streak = if r[lane] == before_r[lane] {
                                    streak.saturating_add(1)
                                } else {
                                    1
                                };
                                mult = mult.max(streak);
                            }
                        } else {
                            streak = 1;
                        }
                        prev = u64::from(r[LANES - 1]) << 32 | u64::from(t[LANES - 1]);
                        i += LANES;
                    }
                }
            }
            for i in i..hi {
                let cur = posting(i);
                if cur <= prev {
                    return Err(UNSORTED);
                }
                streak = if cur >> 32 == prev >> 32 {
                    streak.saturating_add(1)
                } else {
                    1
                };
                mult = mult.max(streak);
                prev = cur;
            }
            mults.push(u8::try_from(mult).unwrap_or(u8::MAX));
            // Sorted, so the run's first reference is its smallest and its
            // last one its largest.
            let (first, last) = ((first >> 32) as u32, (prev >> 32) as u32);
            ids = Some(ids.map_or((first, last), |(min, max)| (min.min(first), max.max(last))));
        }
        Ok((mults, ids))
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl Segment {
    /// Write the segment as a snapshot (format in the module docs).
    ///
    /// # Errors
    ///
    /// Any error of the writer.
    pub fn write_to<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        write_segment(self, &mut w)?;
        w.flush()
    }

    /// Read a snapshot written by [`Segment::write_to`], checking the
    /// checksum and that the arrays describe a valid segment: keys
    /// ascending, offsets consistent, each run sorted by
    /// `(reference, time)` without duplicates.
    ///
    /// # Errors
    ///
    /// [`SnapshotError`] on a read failure, a wrong magic or version, a
    /// checksum mismatch, or inconsistent arrays.
    pub fn read_from<R: Read>(mut r: R) -> Result<Segment, SnapshotError> {
        read_segment(&mut r)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl Catalog {
    /// Write the catalog as a snapshot: the id space, the liveness table
    /// and every segment as it stands (format in the module docs). Call
    /// [`Catalog::compact`] first to leave removed references' postings
    /// out.
    ///
    /// # Errors
    ///
    /// Any error of the writer.
    pub fn write_to<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        let live = self.live_table();
        let (n_refs, n_segments) = (live.len() as u64, self.segment_count() as u64);
        write_header(&mut w, CATALOG_MAGIC, (0, 0), n_refs, n_segments)?;
        let mut ck = checksum_of((0, 0), n_refs, n_segments);
        let mut bitmap = vec![0u8; live.len().div_ceil(8)];
        for (i, &l) in live.iter().enumerate() {
            if l {
                bitmap[i / 8] |= 1 << (i % 8);
            }
        }
        for &b in &bitmap {
            ck.word(u64::from(b));
        }
        w.write_all(&bitmap)?;
        write_pad(&mut w, HEADER + bitmap.len())?;
        w.write_all(&ck.finish().to_le_bytes())?;
        for seg in self.segments() {
            write_segment(seg, &mut w)?;
        }
        w.flush()
    }

    /// Read a snapshot written by [`Catalog::write_to`]. Ids, liveness and
    /// segments come back as written, so the next [`Catalog::add`] returns
    /// the id it would have returned before the round trip.
    ///
    /// # Errors
    ///
    /// [`SnapshotError`] on a read failure, a wrong magic or version, a
    /// checksum mismatch, or a segment that references an id outside the
    /// id space.
    pub fn read_from<R: Read>(mut r: R) -> Result<Catalog, SnapshotError> {
        let (live, n_segments, _) = read_catalog_head(&mut r)?;
        let mut segments = Vec::with_capacity(n_segments);
        for _ in 0..n_segments {
            let (seg, ids) = read_segment_ids(&mut r)?;
            check_id_space(ids, live.len())?;
            segments.push((seg, ids));
        }
        Ok(Catalog::from_parts(segments, live))
    }
}

/// The catalog header and liveness bitmap: the live table, the segment
/// count, and the bytes consumed (where the first segment starts).
fn read_catalog_head<R: Read>(r: &mut R) -> Result<(Vec<bool>, usize, usize), SnapshotError> {
    let (fields, n_refs, n_segments) = read_header(r, CATALOG_MAGIC)?;
    if n_refs > u64::from(u32::MAX) {
        return Err(SnapshotError::Corrupt("id space exceeds u32"));
    }
    if n_segments > MAX_SEGMENTS {
        return Err(SnapshotError::Corrupt("implausible segment count"));
    }
    let mut ck = checksum_of(fields, n_refs, n_segments);
    let n_refs = to_usize(n_refs)?;
    let bitmap = read_bytes(r, n_refs.div_ceil(8), &mut ck)?;
    let end = read_pad(r, (HEADER + bitmap.len()) as u64)?;
    if read_u64(r)? != ck.finish() {
        return Err(SnapshotError::Checksum);
    }
    let live = (0..n_refs)
        .map(|i| bitmap[i / 8] & (1 << (i % 8)) != 0)
        .collect();
    Ok((live, n_segments as usize, end as usize + 8))
}

#[cfg(feature = "mmap")]
mod mapped {
    use std::path::Path;
    use std::sync::Arc;

    use memmap2::Mmap;

    use super::{
        HEADER, Ids, SEGMENT_MAGIC, check_id_space, checksum_of, finish_segment, read_catalog_head,
        read_header, read_pad, read_u32s, read_u64, read_u64s, segment_counts,
    };
    use crate::packed::Packed;
    use crate::{Catalog, Segment, SnapshotError};

    /// Map `path` read-only with its pages faulted in.
    #[expect(
        unsafe_code,
        reason = "mapping a file is the one operation the crate cannot do safely"
    )]
    fn map_file(path: &Path) -> Result<Arc<Mmap>, SnapshotError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: a read-only mapping that this crate never writes through.
        // The contract on `map_from` is that nothing modifies or truncates
        // the file while the mapping lives; that is the one condition
        // `memmap2` cannot enforce.
        let map = unsafe { memmap2::MmapOptions::new().populate().map(&file)? };
        Ok(Arc::new(map))
    }

    /// Wrap the packed array whose data starts at `start` (where `r` also
    /// stands) and move `r` past it.
    fn map_packed(
        map: &Arc<Mmap>,
        start: usize,
        n: usize,
        bits: u32,
        r: &mut &[u8],
    ) -> Result<Packed, SnapshotError> {
        if bits > u32::BITS {
            return Err(SnapshotError::Corrupt("a width exceeds 32 bits"));
        }
        let packed =
            Packed::mapped(Arc::clone(map), start, n, bits).ok_or(SnapshotError::Truncated)?;
        *r = r
            .get(packed.data().len()..)
            .ok_or(SnapshotError::Truncated)?;
        Ok(packed)
    }

    /// The segment snapshot at `pos` in `map`: keys and offsets copied, the
    /// packed arrays left in the mapping. Returns the segment, the range of
    /// its references, and the position after it.
    fn read_segment_at(
        map: &Arc<Mmap>,
        pos: usize,
    ) -> Result<(Segment, Ids, usize), SnapshotError> {
        let mut r: &[u8] = map.get(pos..).ok_or(SnapshotError::Truncated)?;
        let (widths, n_keys, n_post) = read_header(&mut r, SEGMENT_MAGIC)?;
        let (ref_bits, time_bits) = widths;
        let mut ck = checksum_of(widths, n_keys, n_post);
        let (n_keys, n_post) = segment_counts(n_keys, n_post)?;
        let mut rel = HEADER as u64;
        let keys = read_u32s(&mut r, n_keys, &mut ck)?;
        rel = read_pad(&mut r, rel + n_keys as u64 * 4)?;
        let offsets = read_u64s(&mut r, n_keys + 1, &mut ck)?;
        rel += (n_keys as u64 + 1) * 8;
        let refs = map_packed(map, pos + rel as usize, n_post, ref_bits, &mut r)?;
        rel = read_pad(&mut r, rel + refs.data().len() as u64)?;
        let times = map_packed(map, pos + rel as usize, n_post, time_bits, &mut r)?;
        rel = read_pad(&mut r, rel + times.data().len() as u64)?;
        let stored = read_u64(&mut r)?;
        rel += 8;
        // The packed arrays are nearly all of the bytes: they are summed
        // on a core of their own while the runs are checked.
        let sum = move |refs: &Packed, times: &Packed| {
            ck.words(refs.data());
            ck.words(times.data());
            ck.finish() == stored
        };
        let (seg, ids) = finish_segment(widths, n_post, keys, &offsets, refs, times, sum)?;
        Ok((seg, ids, pos + rel as usize))
    }

    pub(super) fn catalog(path: &Path) -> Result<Catalog, SnapshotError> {
        let map = map_file(path)?;
        let mut head: &[u8] = &map;
        let (live, n_segments, mut pos) = read_catalog_head(&mut head)?;
        let mut segments = Vec::with_capacity(n_segments);
        for _ in 0..n_segments {
            let (seg, ids, next) = read_segment_at(&map, pos)?;
            check_id_space(ids, live.len())?;
            segments.push((seg, ids));
            pos = next;
        }
        Ok(Catalog::from_parts(segments, live))
    }

    pub(super) fn segment(path: &Path) -> Result<Segment, SnapshotError> {
        let map = map_file(path)?;
        read_segment_at(&map, 0).map(|(seg, _, _)| seg)
    }
}

#[cfg(feature = "mmap")]
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
impl Catalog {
    /// Map a snapshot written by [`Catalog::write_to`] instead of reading
    /// it. The keys and directories are built in memory; the packed
    /// postings stay in the file's pages, shared with every other process
    /// that maps the same file and reclaimable by the kernel without a
    /// write. The pages are faulted in before this returns, and the
    /// checksums and structural checks of [`Catalog::read_from`] run in
    /// full. [`Catalog::heap_bytes`] then counts the in-memory part only.
    ///
    /// The file must not be modified or truncated while this catalog, any
    /// clone of it, or any catalog that still shares a segment with it is
    /// alive: a write through another handle is undefined behaviour for
    /// the mapping. Write a new snapshot to a fresh name and rename it
    /// over the old one.
    ///
    /// # Errors
    ///
    /// As [`Catalog::read_from`], plus any error opening or mapping the
    /// file.
    pub fn map_from(path: impl AsRef<std::path::Path>) -> Result<Catalog, SnapshotError> {
        mapped::catalog(path.as_ref())
    }
}

#[cfg(feature = "mmap")]
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
impl Segment {
    /// Map a snapshot written by [`Segment::write_to`] instead of reading
    /// it, under the same contract as [`Catalog::map_from`].
    ///
    /// # Errors
    ///
    /// As [`Segment::read_from`], plus any error opening or mapping the
    /// file.
    pub fn map_from(path: impl AsRef<std::path::Path>) -> Result<Segment, SnapshotError> {
        mapped::segment(path.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Posting;

    /// Structural checks sit behind the checksum, so corrupt-but-consistent
    /// snapshots are made by writing invalid arrays through the writer.
    fn roundtrip(seg: &Segment) -> Result<Segment, SnapshotError> {
        let mut bytes = Vec::new();
        seg.write_to(&mut bytes).unwrap();
        Segment::read_from(bytes.as_slice())
    }

    #[test]
    fn structural_validation_rejects_bad_arrays() {
        let bad_keys = Segment::from_arrays(&[5, 3], &[0, 1, 2], &[0, 0], &[0, 1]);
        assert!(
            matches!(roundtrip(&bad_keys), Err(SnapshotError::Corrupt(m)) if m.contains("keys"))
        );
        let bad_offsets = Segment::from_arrays(&[1, 2], &[0, 2, 2], &[0, 0], &[0, 1]);
        assert!(
            matches!(roundtrip(&bad_offsets), Err(SnapshotError::Corrupt(m)) if m.contains("offsets"))
        );
        // (A span mismatch cannot come from the writer: the last offset is
        // always the posting count; the reader's check is defence only.)
        let bad_run = Segment::from_arrays(&[1], &[0, 2], &[1, 0], &[0, 0]);
        assert!(matches!(roundtrip(&bad_run), Err(SnapshotError::Corrupt(m)) if m.contains("run")));
        let dup_run = Segment::from_arrays(&[1], &[0, 2], &[0, 0], &[3, 3]);
        assert!(matches!(roundtrip(&dup_run), Err(SnapshotError::Corrupt(m)) if m.contains("run")));
        let good = Segment::from_postings(vec![Posting::new(1, 0, 3), Posting::new(1, 1, 0)]);
        assert_eq!(roundtrip(&good).unwrap(), good);
    }

    /// Runs long enough to be checked eight postings at a time agree with
    /// the check of one posting at a time: on their multiplicities, with
    /// streaks that cross blocks, and on a posting out of order or repeated
    /// at every position of a run.
    #[test]
    fn long_runs_are_checked_in_blocks() {
        // Two runs. The second starts below where the first ends, which
        // is valid: the order holds within a run only.
        let streaks = [1u32, 3, 1, 1, 12, 2, 1, 9, 1, 1, 1, 4, 20, 1, 3];
        let run: Vec<u32> = streaks
            .iter()
            .enumerate()
            .flat_map(|(r, &n)| (0..n).map(move |_| 7 * r as u32))
            .collect();
        let n = run.len();
        let refs: Vec<u32> = run.iter().chain(&run[5..]).copied().collect();
        let times: Vec<u32> = (0..refs.len() as u32).map(|i| 3 * i).collect();
        let offsets = [0, n as u64, refs.len() as u64];
        let good = Segment::from_arrays(&[4, 9], &offsets, &refs, &times);
        let back = roundtrip(&good).unwrap();
        assert_eq!(back, good);
        assert_eq!(back.range_mult(4), Some((0, n, 20)));
        assert_eq!(back.range_mult(9), Some((n, refs.len(), 20)));
        assert_eq!(good.range_mult(9), back.range_mult(9));
        for at in (1..refs.len()).filter(|&at| at != n) {
            // The posting at `at` moved before its predecessor, by its
            // reference and then by its time alone.
            let (mut r, mut t) = (refs.clone(), times.clone());
            r[at - 1] = r[at] + 1;
            let bad = Segment::from_arrays(&[4, 9], &offsets, &r, &times);
            assert!(
                matches!(roundtrip(&bad), Err(SnapshotError::Corrupt(m)) if m.contains("run")),
                "reference out of order at {at}"
            );
            if refs[at] == refs[at - 1] {
                t[at] = t[at - 1] - 1;
                let bad = Segment::from_arrays(&[4, 9], &offsets, &refs, &t);
                assert!(
                    matches!(roundtrip(&bad), Err(SnapshotError::Corrupt(m)) if m.contains("run")),
                    "time out of order at {at}"
                );
                // And repeated outright.
                t[at] = t[at - 1];
                let bad = Segment::from_arrays(&[4, 9], &offsets, &refs, &t);
                assert!(
                    matches!(roundtrip(&bad), Err(SnapshotError::Corrupt(m)) if m.contains("run")),
                    "posting repeated at {at}"
                );
            }
        }
    }

    /// A segment large enough to be checked on every core: the pieces'
    /// multiplicities and reference ranges join up to those of a check in
    /// order, and an unsorted run is found whichever piece it is in.
    #[test]
    fn large_segments_are_checked_in_parallel_pieces() {
        let (n_keys, per_key) = (600u32, 20u32);
        let mut refs = Vec::new();
        let mut offsets = vec![0u64];
        for k in 0..n_keys {
            // Key `k` repeats one reference `k % 5 + 1` times, so the
            // multiplicities differ from piece to piece.
            let first = k % 7;
            refs.extend((0..per_key).map(|i| first + i / (k % 5 + 1)));
            offsets.push(refs.len() as u64);
        }
        let times: Vec<u32> = (0..refs.len() as u32).collect();
        assert!(refs.len() >= 4 * PARALLEL_MIN);
        let keys: Vec<u32> = (0..n_keys).map(|k| 3 * k).collect();
        let good = Segment::from_arrays(&keys, &offsets, &refs, &times);
        let back = roundtrip(&good).unwrap();
        assert_eq!(back, good);
        for &key in &keys {
            assert_eq!(back.range_mult(key), good.range_mult(key), "key {key}");
        }
        for k in [0, 1, n_keys / 2, n_keys - 2, n_keys - 1] {
            let mut bad = refs.clone();
            let at = offsets[k as usize + 1] as usize - 1;
            bad[at] = 0;
            bad[at - 1] = 1;
            let bad = Segment::from_arrays(&keys, &offsets, &bad, &times);
            assert!(
                matches!(roundtrip(&bad), Err(SnapshotError::Corrupt(m)) if m.contains("run")),
                "unsorted run {k}"
            );
        }
    }

    /// The mapped path sums the packed arrays while the runs are checked:
    /// a flipped data byte is a checksum error, not a structural one, and
    /// an intact file maps to the segment it was written from.
    #[cfg(feature = "mmap")]
    #[test]
    fn mapped_large_segments_are_summed_alongside_the_check() {
        let refs: Vec<u32> = (0..8_000u32).map(|i| i % 50).collect();
        let offsets: Vec<u64> = (0..=400u64).map(|k| k * 20).collect();
        let times: Vec<u32> = (0..8_000u32).map(|i| i / 50).collect();
        let keys: Vec<u32> = (0..400).collect();
        let mut sorted = refs.clone();
        for run in sorted.chunks_mut(20) {
            run.sort_unstable();
        }
        let seg = Segment::from_arrays(&keys, &offsets, &sorted, &times);
        let mut bytes = Vec::new();
        seg.write_to(&mut bytes).unwrap();
        let path =
            std::env::temp_dir().join(format!("afpindex-parallel-{}.afx", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(Segment::map_from(&path).unwrap(), seg);
        // The last data byte before the checksum's padding: a reference or
        // time bit, so the arrays still describe a valid segment.
        let at = bytes.len() - 16;
        bytes[at] ^= 0x10;
        std::fs::write(&path, &bytes).unwrap();
        let got = Segment::map_from(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(got, Err(SnapshotError::Checksum)), "{got:?}");
    }

    #[test]
    fn padding_keeps_every_section_aligned() {
        for n in 0..9u32 {
            let seg = Segment::from_postings((0..n).map(|i| Posting::new(i, 0, i)).collect());
            let mut bytes = Vec::new();
            seg.write_to(&mut bytes).unwrap();
            assert_eq!(bytes.len() % 8, 0, "{n} keys");
            let keys_end = HEADER + 4 * n as usize;
            let offsets_at = keys_end.next_multiple_of(8);
            assert_eq!(offsets_at % 8, 0);
            assert_eq!(roundtrip(&seg).unwrap(), seg);
        }
    }

    #[test]
    fn catalog_reader_rejects_ids_outside_the_id_space() {
        let seg = Segment::from_postings(vec![Posting::new(1, 5, 0)]);
        let cat = Catalog::from_parts(vec![(seg, Some((5, 5)))], vec![true]);
        let mut bytes = Vec::new();
        cat.write_to(&mut bytes).unwrap();
        assert!(matches!(
            Catalog::read_from(bytes.as_slice()),
            Err(SnapshotError::Corrupt(m)) if m.contains("id space")
        ));
    }

    #[test]
    fn checksum_is_order_sensitive() {
        let mut a = Checksum::new();
        a.word(1);
        a.word(2);
        let mut b = Checksum::new();
        b.word(2);
        b.word(1);
        assert_ne!(a.finish(), b.finish());
        assert_ne!(Checksum::new().finish(), 0);
    }
}
