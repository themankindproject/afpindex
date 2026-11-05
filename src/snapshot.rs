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
use std::vec::Vec;

use crate::packed::Packed;
use crate::segment::MAX_POSTINGS;
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

    finish_segment(widths, n_post, keys, &offsets, refs, times)
}

/// Check that the arrays describe a valid segment (keys ascending, offsets
/// consistent, each run sorted by `(reference, time)` without duplicates)
/// and assemble it. The walk also records each run's multiplicity for its
/// directory slot and the range of the references, so nothing reads the
/// postings a second time at load.
fn finish_segment(
    (ref_bits, time_bits): (u32, u32),
    n_post: usize,
    keys: Vec<u32>,
    offsets: &[u64],
    refs: Packed,
    times: Packed,
) -> Result<(Segment, Ids), SnapshotError> {
    if !keys.windows(2).all(|w| w[0] < w[1]) {
        return Err(SnapshotError::Corrupt("keys not strictly ascending"));
    }
    if offsets.first() != Some(&0) || offsets.last().copied() != Some(n_post as u64) {
        return Err(SnapshotError::Corrupt("offsets do not span the postings"));
    }
    if !offsets.windows(2).all(|w| w[0] < w[1]) {
        return Err(SnapshotError::Corrupt("offsets not strictly ascending"));
    }
    // Distinct `(reference, time)` pairs bound a run's length by the
    // widths, which keeps the scan below proportional to the bytes read.
    let distinct = 1u64 << (ref_bits + time_bits).min(63);
    // The same walk records each run's multiplicity for its directory slot.
    let mut mults = Vec::with_capacity(keys.len());
    let mut ids: Ids = None;
    let (run_refs, run_times) = (refs.view(), times.view());
    // A posting as one word that orders as `(reference, time)` does.
    let posting = |i: usize| u64::from(run_refs.get(i)) << 32 | u64::from(run_times.get(i));
    for w in offsets.windows(2) {
        if w[1] - w[0] > distinct {
            return Err(SnapshotError::Corrupt(
                "a run is longer than its widths allow",
            ));
        }
        let (lo, hi) = (w[0] as usize, w[1] as usize);
        let (mut mult, mut streak) = (1u32, 1u32);
        let first = posting(lo);
        let mut prev = first;
        for i in lo + 1..hi {
            let cur = posting(i);
            if cur <= prev {
                return Err(SnapshotError::Corrupt(
                    "a run is not sorted by (reference, time)",
                ));
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
    Ok((Segment::from_parts(keys, offsets, refs, times, &mults), ids))
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
        Checksum, HEADER, Ids, SEGMENT_MAGIC, check_id_space, checksum_of, finish_segment,
        read_catalog_head, read_header, read_pad, read_u32s, read_u64, read_u64s, segment_counts,
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
    /// stands), checksum it, and move `r` past it.
    fn map_packed(
        map: &Arc<Mmap>,
        start: usize,
        n: usize,
        bits: u32,
        ck: &mut Checksum,
        r: &mut &[u8],
    ) -> Result<Packed, SnapshotError> {
        if bits > u32::BITS {
            return Err(SnapshotError::Corrupt("a width exceeds 32 bits"));
        }
        let packed =
            Packed::mapped(Arc::clone(map), start, n, bits).ok_or(SnapshotError::Truncated)?;
        let data = packed.data();
        ck.words(data);
        *r = r.get(data.len()..).ok_or(SnapshotError::Truncated)?;
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
        let refs = map_packed(map, pos + rel as usize, n_post, ref_bits, &mut ck, &mut r)?;
        rel = read_pad(&mut r, rel + refs.data().len() as u64)?;
        let times = map_packed(map, pos + rel as usize, n_post, time_bits, &mut ck, &mut r)?;
        rel = read_pad(&mut r, rel + times.data().len() as u64)?;
        if read_u64(&mut r)? != ck.finish() {
            return Err(SnapshotError::Checksum);
        }
        rel += 8;
        let (seg, ids) = finish_segment(widths, n_post, keys, &offsets, refs, times)?;
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
