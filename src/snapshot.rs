//! Snapshot files (`std`): little-endian arrays behind a small header,
//! every section 8-byte aligned, closed by checksums.
//!
//! Segment snapshot, version 4:
//!
//! ```text
//! 0   "AFIX"            4   version u32 = 4
//! 8   ref_bits u32      12  time_bits u32
//! 16  key_count u64     24  posting_count u64
//! 32  min_ref u32       36  max_ref u32      (both 0 without postings)
//! 40  refs    ceil(posting_count × ref_bits / 8) bytes    (zero-padded to 8)
//!     times   ceil(posting_count × time_bits / 8) bytes   (zero-padded to 8)
//!     keys    u32 × key_count                             (zero-padded to 8)
//!     offsets u64 × (key_count + 1)
//!     mults   u8  × key_count                             (zero-padded to 8)
//!     checksum u64 of refs, as little-endian 64-bit words
//!     checksum u64 of times, likewise
//!     checksum u64 of the keys and the offsets as values, then the
//!     multiplicities as little-endian 64-bit words
//!     checksum u64 of the header words (widths, key_count, posting_count,
//!     min_ref and max_ref) and the three checksums before it
//! ```
//!
//! References and times are packed at the width of the largest value in
//! the segment, little-endian bit order, as held in memory. `mults` holds
//! per key what a directory slot records of its run: the most postings any
//! one reference has in it, saturating at 255. `min_ref` and `max_ref` are
//! the smallest and the largest reference in the segment. A load would
//! otherwise recompute both from the runs one posting at a time; stored,
//! a trusted load can take them, and a full load checks them.
//!
//! The postings come first and every section has a checksum of its own so
//! that a merge can write the file in one pass
//! ([`crate::segment::to_file`]): the two packed arrays grow side by side,
//! each summed as it is written, and the counts in the header are known
//! only once the last posting is. A reader likewise sums the two arrays
//! apart, on two cores.
//!
//! Catalog snapshot, version 4 (the segments' format gave it the number):
//!
//! ```text
//! 0   "AFCT"            4   version u32 = 4
//! 8   flags u32 = 0     12  reserved u32 = 0
//! 16  reference_count u64   24  segment_count u64
//! 32  live bitmap, one bit per id, LSB first, ceil(n / 8) bytes (zero-padded to 8)
//!     checksum u64 over the header counts and the bitmap bytes
//!     segment_count segment snapshots, back to back
//! ```
//!
//! Flags and reserved fields are written as zero and ignored on read.

use std::io::{Read, Write};
use std::ops::RangeInclusive;
use std::sync::{Mutex, PoisonError};
use std::vec::Vec;

use crate::packed::Packed;
use crate::par::{lock, run_tasks, slots, threads};
use crate::segment::MAX_POSTINGS;
use crate::simd::{self, Decoder, LANES, Pass};
use crate::{Catalog, Segment, SnapshotError};

pub(crate) const SEGMENT_MAGIC: [u8; 4] = *b"AFIX";
const CATALOG_MAGIC: [u8; 4] = *b"AFCT";
const VERSION: u32 = 4;
/// Bytes before the first array of a segment snapshot.
pub(crate) const HEADER: usize = 40;
/// Bytes of the four checksums that close a segment snapshot.
pub(crate) const FOOTER: usize = 32;
/// Bytes before the first array of a catalog snapshot, which has no
/// reference range.
const CATALOG_HEADER: usize = 32;
/// Bytes read or written per call while streaming an array.
const CHUNK: usize = 1 << 18;
/// Sanity bound on the segment count a catalog snapshot may declare; the
/// merge policy keeps real catalogs far below it.
const MAX_SEGMENTS: u64 = 1 << 16;

/// Order-sensitive 64-bit checksum over words: catches corruption, not
/// tampering.
pub(crate) struct Checksum(u64);

impl Checksum {
    const PRIME: u64 = 0x9E37_79B9_7F4A_7C15;

    pub(crate) fn new() -> Self {
        Self(0x4146_4958_0000_0001)
    }

    #[inline]
    pub(crate) fn word(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(29) ^ w).wrapping_mul(Self::PRIME);
    }

    /// Fold in `bytes` as little-endian 64-bit words, the last one
    /// zero-padded: how the packed arrays are covered. Bytes fed in several
    /// calls must be cut at multiples of 8.
    pub(crate) fn words(&mut self, bytes: &[u8]) {
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

    pub(crate) fn finish(self) -> u64 {
        let mut h = self.0;
        h ^= h >> 32;
        h = h.wrapping_mul(Self::PRIME);
        h ^ (h >> 29)
    }
}

/// The checksum of a section whose bytes arrive in pieces cut anywhere:
/// what [`Checksum::words`] gives for all of them at once.
#[cfg(feature = "mmap")]
pub(crate) struct ByteSum {
    sum: Checksum,
    /// The bytes of a word that is not whole yet.
    carry: [u8; 8],
    held: usize,
}

#[cfg(feature = "mmap")]
impl ByteSum {
    pub(crate) fn new() -> Self {
        Self {
            sum: Checksum::new(),
            carry: [0; 8],
            held: 0,
        }
    }

    pub(crate) fn feed(&mut self, mut bytes: &[u8]) {
        if self.held > 0 {
            let take = (8 - self.held).min(bytes.len());
            self.carry[self.held..self.held + take].copy_from_slice(&bytes[..take]);
            self.held += take;
            bytes = &bytes[take..];
            if self.held < 8 {
                return;
            }
            self.sum.word(u64::from_le_bytes(self.carry));
            self.held = 0;
        }
        let whole = bytes.len() - bytes.len() % 8;
        self.sum.words(&bytes[..whole]);
        let rest = &bytes[whole..];
        self.carry[..rest.len()].copy_from_slice(rest);
        self.held = rest.len();
    }

    pub(crate) fn finish(mut self) -> u64 {
        if self.held > 0 {
            self.carry[self.held..].fill(0);
            self.sum.word(u64::from_le_bytes(self.carry));
        }
        self.sum.finish()
    }
}

/// Zero bytes that bring `pos` to the next multiple of 8.
pub(crate) fn pad_len(pos: u64) -> usize {
    (pos.next_multiple_of(8) - pos) as usize
}

fn write_pad<W: Write>(w: &mut W, pos: usize) -> std::io::Result<usize> {
    let pad = pad_len(pos as u64);
    w.write_all(&[0u8; 8][..pad])?;
    Ok(pos + pad)
}

/// Write `values` as little-endian bytes, a block at a time.
fn write_values<W: Write, T: Copy, const N: usize>(
    w: &mut W,
    values: &[T],
    bytes: impl Fn(T) -> [u8; N],
) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(CHUNK);
    for block in values.chunks(CHUNK / N) {
        buf.clear();
        buf.extend(block.iter().flat_map(|&v| bytes(v)));
        w.write_all(&buf)?;
    }
    Ok(())
}

/// Write the keys, the offsets and the multiplicities of a segment whose
/// arrays end at `pos`, each padded to 8 bytes.
pub(crate) fn write_index<W: Write>(
    w: &mut W,
    keys: &[u32],
    offsets: &[u64],
    mults: &[u8],
) -> std::io::Result<()> {
    write_values(w, keys, u32::to_le_bytes)?;
    w.write_all(&[0u8; 8][..pad_len(4 * keys.len() as u64)])?;
    write_values(w, offsets, u64::to_le_bytes)?;
    w.write_all(mults)?;
    w.write_all(&[0u8; 8][..pad_len(mults.len() as u64)])
}

/// The checksum of a segment's keys, offsets and multiplicities.
pub(crate) fn index_sum(keys: &[u32], offsets: &[u64], mults: &[u8]) -> u64 {
    let mut ck = Checksum::new();
    for &key in keys {
        ck.word(u64::from(key));
    }
    for &offset in offsets {
        ck.word(offset);
    }
    ck.words(mults);
    ck.finish()
}

/// The checksum of a packed array's bytes.
fn packed_sum(bytes: &[u8]) -> u64 {
    let mut ck = Checksum::new();
    ck.words(bytes);
    ck.finish()
}

/// The two header words of a segment's reference range.
pub(crate) fn range_word((min, max): (u32, u32)) -> u64 {
    u64::from(min) | u64::from(max) << 32
}

/// Write a header: magic, version, two 32-bit fields, two counts.
pub(crate) fn write_header<W: Write>(
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

/// Read a header of `magic`; a segment header's reference range follows
/// it.
fn read_header<R: Read>(r: &mut R, magic: [u8; 4]) -> Result<Header, SnapshotError> {
    let mut head = [0u8; CATALOG_HEADER];
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

/// Read the reference range of a segment header.
fn read_range<R: Read>(r: &mut R) -> Result<(u32, u32), SnapshotError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    let min = u32::from_le_bytes(buf[..4].try_into().expect("4 bytes"));
    let max = u32::from_le_bytes(buf[4..].try_into().expect("4 bytes"));
    Ok((min, max))
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

fn read_u32s<R: Read>(r: &mut R, n: usize) -> Result<Vec<u32>, SnapshotError> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; CHUNK];
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK / 4);
        r.read_exact(&mut buf[..take * 4])?;
        grow(&mut out, take, remaining);
        out.extend(
            buf[..take * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        remaining -= take;
    }
    Ok(out)
}

fn read_u64s<R: Read>(r: &mut R, n: usize) -> Result<Vec<u64>, SnapshotError> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; CHUNK];
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK / 8);
        r.read_exact(&mut buf[..take * 8])?;
        grow(&mut out, take, remaining);
        out.extend(
            buf[..take * 8]
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))),
        );
        remaining -= take;
    }
    Ok(out)
}

/// Read `n` single bytes (the multiplicities).
fn read_u8s<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, SnapshotError> {
    let mut out = Vec::new();
    let mut remaining = n;
    while remaining > 0 {
        let take = remaining.min(CHUNK);
        grow(&mut out, take, remaining);
        read_into(r, &mut out, take)?;
        remaining -= take;
    }
    Ok(out)
}

/// A segment's keys, offsets and multiplicities.
type Index = (Vec<u32>, Vec<u64>, Vec<u8>);

/// Read the keys, the offsets and the multiplicities of a segment of
/// `n_keys` keys, with the padding of each.
fn read_index<R: Read>(r: &mut R, n_keys: usize) -> Result<Index, SnapshotError> {
    let keys = read_u32s(r, n_keys)?;
    read_pad(r, 4 * n_keys as u64)?;
    let offsets = read_u64s(r, n_keys + 1)?;
    let mults = read_u8s(r, n_keys)?;
    read_pad(r, n_keys as u64)?;
    Ok((keys, offsets, mults))
}

/// Bytes of the keys, the offsets and the multiplicities of `n_keys` keys
/// in a snapshot.
#[cfg(any(feature = "mmap", test))]
fn index_len(n_keys: u64) -> u64 {
    (4 * n_keys).next_multiple_of(8) + 8 * (n_keys + 1) + n_keys.next_multiple_of(8)
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

/// Read `n` values of `bits` packed, summed as little-endian 64-bit
/// words (the last one zero-padded) if `ck` is given.
fn read_packed<R: Read>(
    r: &mut R,
    n: usize,
    bits: u32,
    mut ck: Option<&mut Checksum>,
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
        if let Some(ck) = ck.as_deref_mut() {
            ck.words(&out[at..]);
        }
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

/// What a segment snapshot's header says of the segment.
#[derive(Clone, Copy)]
pub(crate) struct Head {
    /// Bits of a reference and of a time.
    pub(crate) widths: (u32, u32),
    pub(crate) keys: u64,
    pub(crate) postings: u64,
    /// Smallest and largest reference, both zero without postings.
    pub(crate) range: (u32, u32),
}

impl Head {
    /// The header as it lies in the file.
    pub(crate) fn bytes(&self) -> [u8; HEADER] {
        let mut out = [0u8; HEADER];
        let mut w = out.as_mut_slice();
        write_header(&mut w, SEGMENT_MAGIC, self.widths, self.keys, self.postings)
            .and_then(|()| w.write_all(&range_word(self.range).to_le_bytes()))
            .expect("the header is 40 bytes");
        out
    }

    fn read<R: Read>(r: &mut R) -> Result<Self, SnapshotError> {
        let (widths, keys, postings) = read_header(r, SEGMENT_MAGIC)?;
        let range = read_range(r)?;
        Ok(Self {
            widths,
            keys,
            postings,
            range,
        })
    }

    /// The four checksums that close the snapshot, from those of its
    /// sections: the references', the times' and the index's, then one
    /// over the header words and those three.
    pub(crate) fn footer(&self, sums: [u64; 3]) -> [u8; FOOTER] {
        let mut ck = checksum_of(self.widths, self.keys, self.postings);
        ck.word(range_word(self.range));
        let mut out = [0u8; FOOTER];
        for (sum, bytes) in sums.iter().zip(out.chunks_exact_mut(8)) {
            ck.word(*sum);
            bytes.copy_from_slice(&sum.to_le_bytes());
        }
        out[24..].copy_from_slice(&ck.finish().to_le_bytes());
        out
    }
}

/// The checksums a snapshot stores for its references and its times, once
/// the footer is read and found to agree with the header and the index.
///
/// Those two are the checks that read no posting, so a trusted load makes
/// them too.
fn read_footer<R: Read>(
    r: &mut R,
    head: &Head,
    (keys, offsets, mults): &Index,
) -> Result<(u64, u64), SnapshotError> {
    let mut stored = [0u8; FOOTER];
    r.read_exact(&mut stored)?;
    let word = |i: usize| u64::from_le_bytes(stored[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    let (refs, times) = (word(0), word(1));
    if stored != head.footer([refs, times, index_sum(keys, offsets, mults)]) {
        return Err(SnapshotError::Checksum);
    }
    Ok((refs, times))
}

fn write_segment<W: Write>(seg: &Segment, w: &mut W) -> std::io::Result<()> {
    let n_post = seg.len() as u64;
    // One walk of the directory gives what the directory records of every
    // run, in key order: where it starts, which the offsets need, and its
    // multiplicity. Runs are sorted by reference, so the first and the
    // last posting of each run bound the segment's reference range.
    let mut offsets = Vec::with_capacity(seg.key_count() + 1);
    let mut mults = Vec::with_capacity(seg.key_count());
    for (start, mult) in seg.starts_and_mults() {
        offsets.push(start);
        mults.push(mult);
    }
    offsets.push(n_post);
    let refs = seg.refs();
    let mut range = (0u32, 0u32);
    let mut any = false;
    for run in offsets.windows(2) {
        let (start, end) = (run[0] as usize, run[1] as usize);
        if start < end {
            let (lo, hi) = (refs.get(start), refs.get(end - 1));
            range = if any {
                (range.0.min(lo), range.1.max(hi))
            } else {
                (lo, hi)
            };
            any = true;
        }
    }
    let head = Head {
        widths: seg.widths(),
        keys: seg.key_count() as u64,
        postings: n_post,
        range,
    };
    w.write_all(&head.bytes())?;
    let mut sums = [0u64; 3];
    for (packed, sum) in [seg.refs(), seg.times()].into_iter().zip(&mut sums) {
        let bytes = packed.data();
        *sum = packed_sum(bytes);
        w.write_all(bytes)?;
        w.write_all(&[0u8; 8][..pad_len(bytes.len() as u64)])?;
    }
    sums[2] = index_sum(seg.keys(), &offsets, &mults);
    write_index(w, seg.keys(), &offsets, &mults)?;
    w.write_all(&head.footer(sums))
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

/// How much of a snapshot a load checks.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    /// Everything: the checksums and the order of every run too.
    Full,
    /// Everything but the checksums and the order of the runs, the checks
    /// that read every posting; each run's multiplicity and the
    /// segment's reference range come from the header and read no posting.
    Trusted,
}

/// The smallest and the largest reference of a segment, if it has any.
type Ids = Option<(u32, u32)>;

/// Read a segment snapshot and report the range of its references.
fn read_segment<R: Read>(r: &mut R, check: Check) -> Result<(Segment, Ids), SnapshotError> {
    let head = Head::read(r)?;
    let (ref_bits, time_bits) = head.widths;
    let (n_keys, n_post) = segment_counts(head.keys, head.postings)?;
    let full = check == Check::Full;
    let (mut refs_sum, mut times_sum) = (Checksum::new(), Checksum::new());
    let refs = read_packed(r, n_post, ref_bits, full.then_some(&mut refs_sum))?;
    read_pad(r, refs.data().len() as u64)?;
    let times = read_packed(r, n_post, time_bits, full.then_some(&mut times_sum))?;
    read_pad(r, times.data().len() as u64)?;
    let index = read_index(r, n_keys)?;
    let stored = read_footer(r, &head, &index)?;
    if full && stored != (refs_sum.finish(), times_sum.finish()) {
        return Err(SnapshotError::Checksum);
    }
    // The packed arrays were summed as they were read, if at all.
    let (keys, offsets, mults) = index;
    finish_segment(
        head.widths,
        n_post,
        (keys, &offsets),
        (mults, head.range),
        (refs, times),
        check,
        |_, _| true,
    )
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
/// and assemble it, with each run's multiplicity for its directory slot
/// and the range of the references.
///
/// The file stores both, so a trusted load takes them as they are and
/// reads no posting: it checks only that the range could describe
/// references of the segment's width (see [`check_range`]). A full load
/// walks the runs ([`CheckRuns`]) and refuses a file whose stored
/// multiplicities or range differ from what the runs hold.
///
/// `checksum` checks the packed arrays, unless the caller already has (it
/// then returns `true`); a mismatch is reported before anything else,
/// though the runs are checked while it runs.
fn finish_segment(
    (ref_bits, time_bits): (u32, u32),
    n_post: usize,
    (keys, offsets): (Vec<u32>, &[u64]),
    (mults, range): (Vec<u8>, (u32, u32)),
    (refs, times): (Packed, Packed),
    check: Check,
    checksum: impl FnOnce(&Packed, &Packed) -> bool + Send,
) -> Result<(Segment, Ids), SnapshotError> {
    let index = check_index(n_post, &keys, offsets);
    let side = || checksum(&refs, &times);
    // A segment without postings has no reference range.
    let stored = (n_post > 0).then_some(range);
    // The walk relies on the index, so it runs only once that holds.
    let (summed, walk) = match (check, index.is_ok()) {
        (_, false) | (Check::Trusted, true) => (side(), None),
        (Check::Full, true) => {
            let (summed, walk) = walk_runs(
                offsets,
                |keys| CheckRuns {
                    // Distinct `(reference, time)` pairs bound a run's
                    // length by the widths, which keeps the scan
                    // proportional to the bytes read.
                    longest: 1u64 << (ref_bits + time_bits).min(63),
                    offsets: &offsets[keys],
                    refs: &refs,
                    times: &times,
                },
                side,
            );
            (summed, Some(walk))
        }
    };
    if !summed {
        return Err(SnapshotError::Checksum);
    }
    index?;
    let ids = match walk {
        None => {
            check_range(stored, ref_bits)?;
            stored
        }
        Some(walk) => {
            let (found_mults, found_ids) = walk?;
            if found_mults != mults || found_ids != stored {
                return Err(SnapshotError::Corrupt(
                    "the stored multiplicities or reference range are not the runs'",
                ));
            }
            found_ids
        }
    };
    Ok((Segment::from_parts(keys, offsets, refs, times, &mults), ids))
}

/// Check a stored reference range that no walk has confirmed: it must not
/// be inverted, its largest reference must fit the segment's reference
/// width (a value wider than the packed array could hold is a corrupt
/// file, and a range alone would let a segment claim references its
/// postings cannot have), and a segment without postings has none.
fn check_range(ids: Ids, ref_bits: u32) -> Result<(), SnapshotError> {
    let Some((min, max)) = ids else {
        return Ok(());
    };
    if min > max {
        return Err(SnapshotError::Corrupt("the reference range is inverted"));
    }
    if u64::from(max) >= 1u64 << ref_bits {
        return Err(SnapshotError::Corrupt(
            "a reference range exceeds the reference width",
        ));
    }
    Ok(())
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

/// Walk every run with the pass `piece` makes for a range of keys (the
/// range of their offsets, each key's start and the end of the last), and
/// run `side` meanwhile; return what `side` returned and the walk.
///
/// A large segment's runs are cut into pieces of about equal postings, a
/// few per core so that a slow piece does not hold the rest up, and every
/// thread takes the next task until none are left: task 0 is `side`, task
/// `i` is piece `i - 1`, so the checksum starts as soon as a thread is
/// free rather than waiting for a walk. The calling thread works too, so a
/// thread that cannot be spawned costs nothing but parallelism. A small
/// segment, or a single core, does it all in order on the calling thread.
fn walk_runs<T: Send, P: Pass<Out = Walk>>(
    offsets: &[u64],
    piece: impl Fn(RangeInclusive<usize>) -> P + Sync,
    side: impl FnOnce() -> T + Send,
) -> (T, Walk) {
    let n_keys = offsets.len() - 1;
    let n_post = offsets[n_keys];
    let threads = threads();
    if threads <= 1 || n_post < PARALLEL_MIN as u64 {
        let out = side();
        return (out, simd::run(piece(0..=n_keys)));
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
    let side = Mutex::new(Some(side));
    let side_out = Mutex::new(None);
    let walks = slots::<Walk>(n);
    run_tasks(threads, n + 1, |task| {
        if task == 0 {
            let side = lock(&side).take().expect("task 0 is `side`");
            *lock(&side_out) = Some(side());
        } else {
            *lock(&walks[task - 1]) = Some(simd::run(piece(bounds[task - 1]..=bounds[task])));
        }
    });
    let out = side_out
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .expect("task 0 ran `side`");
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
        read_segment(&mut r, Check::Full).map(|(seg, _)| seg)
    }

    /// Read a snapshot as [`Segment::read_from`] does, without the checks
    /// that read every posting, under the terms of
    /// [`Catalog::read_from_trusted`].
    ///
    /// # Errors
    ///
    /// As [`Segment::read_from`], except checksum mismatches and unsorted
    /// runs, which go unnoticed.
    pub fn read_from_trusted<R: Read>(mut r: R) -> Result<Segment, SnapshotError> {
        read_segment(&mut r, Check::Trusted).map(|(seg, _)| seg)
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
        write_pad(&mut w, CATALOG_HEADER + bitmap.len())?;
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
        read_catalog(&mut r, Check::Full)
    }

    /// Read a snapshot as [`Catalog::read_from`] does, without the checks
    /// that read every posting: the segments' checksums, and the order of
    /// each run. For a file this process or a pipeline it trusts wrote and
    /// has checked once, when the time to load it matters, as on a
    /// restart.
    ///
    /// The headers, the sizes, the keys, the offsets, the stored
    /// multiplicities and reference ranges and the id space are still
    /// checked, but no posting is read: the search takes each run's
    /// multiplicity from the file. A file damaged since it was checked can
    /// load without an error; searches and changes on it then return wrong
    /// candidates or panic, but do nothing undefined.
    ///
    /// # Errors
    ///
    /// As [`Catalog::read_from`], except checksum mismatches in the
    /// segments and unsorted runs, which go unnoticed.
    pub fn read_from_trusted<R: Read>(mut r: R) -> Result<Catalog, SnapshotError> {
        read_catalog(&mut r, Check::Trusted)
    }
}

fn read_catalog<R: Read>(r: &mut R, check: Check) -> Result<Catalog, SnapshotError> {
    let (live, n_segments, _) = read_catalog_head(r)?;
    let mut segments = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        let (seg, ids) = read_segment(r, check)?;
        check_id_space(ids, live.len())?;
        segments.push((seg, ids));
    }
    Ok(Catalog::from_parts(segments, live))
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
    let end = read_pad(r, (CATALOG_HEADER + bitmap.len()) as u64)?;
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
        Check, FOOTER, HEADER, Head, Ids, check_id_space, finish_segment, index_len, packed_sum,
        read_catalog_head, read_footer, read_index, read_pad, segment_counts,
    };
    use crate::packed::Packed;
    use crate::{Catalog, Segment, SnapshotError};

    /// Map `path` read-only, with its pages faulted in unless `populate`
    /// is false.
    #[expect(
        unsafe_code,
        reason = "mapping a file is the one operation the crate cannot do safely"
    )]
    fn map_file_with(path: &Path, populate: bool) -> Result<Arc<Mmap>, SnapshotError> {
        let file = std::fs::File::open(path)?;
        let mut options = memmap2::MmapOptions::new();
        if populate {
            options.populate();
        }
        // SAFETY: a read-only mapping that this crate never writes through.
        // The contract on `map_from` is that nothing modifies or truncates
        // the file while the mapping lives; that is the one condition
        // `memmap2` cannot enforce.
        let map = unsafe { options.map(&file)? };
        Ok(Arc::new(map))
    }

    /// Map `path` read-only with its pages faulted in.
    fn map_file(path: &Path) -> Result<Arc<Mmap>, SnapshotError> {
        map_file_with(path, true)
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

    /// The segment snapshot at `pos` in `map`: keys, offsets and
    /// multiplicities copied, the packed arrays left in the mapping.
    /// Returns the segment, the range of its references, and the position
    /// after it.
    fn read_segment_at(
        map: &Arc<Mmap>,
        pos: usize,
        check: Check,
    ) -> Result<(Segment, Ids, usize), SnapshotError> {
        let mut r: &[u8] = map.get(pos..).ok_or(SnapshotError::Truncated)?;
        let head = Head::read(&mut r)?;
        let (ref_bits, time_bits) = head.widths;
        let (n_keys, n_post) = segment_counts(head.keys, head.postings)?;
        let mut rel = HEADER as u64;
        let refs = map_packed(map, pos + rel as usize, n_post, ref_bits, &mut r)?;
        rel += read_pad(&mut r, refs.data().len() as u64)?;
        let times = map_packed(map, pos + rel as usize, n_post, time_bits, &mut r)?;
        rel += read_pad(&mut r, times.data().len() as u64)?;
        let index = read_index(&mut r, n_keys)?;
        let stored = read_footer(&mut r, &head, &index)?;
        rel += index_len(n_keys as u64) + FOOTER as u64;
        // The packed arrays are nearly all of the bytes: they are summed
        // on a core of their own while the runs are checked.
        let sum = move |refs: &Packed, times: &Packed| {
            check == Check::Trusted || stored == (packed_sum(refs.data()), packed_sum(times.data()))
        };
        let (keys, offsets, mults) = index;
        let (seg, ids) = finish_segment(
            head.widths,
            n_post,
            (keys, &offsets),
            (mults, head.range),
            (refs, times),
            check,
            sum,
        )?;
        Ok((seg, ids, pos + rel as usize))
    }

    pub(super) fn catalog(path: &Path, check: Check) -> Result<Catalog, SnapshotError> {
        let map = map_file(path)?;
        let mut head: &[u8] = &map;
        let (live, n_segments, mut pos) = read_catalog_head(&mut head)?;
        let mut segments = Vec::with_capacity(n_segments);
        for _ in 0..n_segments {
            let (seg, ids, next) = read_segment_at(&map, pos, check)?;
            check_id_space(ids, live.len())?;
            segments.push((seg, ids));
            pos = next;
        }
        Ok(Catalog::from_parts(segments, live))
    }

    pub(super) fn segment(path: &Path, check: Check) -> Result<Segment, SnapshotError> {
        let map = map_file(path)?;
        read_segment_at(&map, 0, check).map(|(seg, _, _)| seg)
    }

    /// The segment at `path` and the range of its references.
    pub(super) fn segment_with(
        path: &Path,
        check: Check,
        populate: bool,
    ) -> Result<(Segment, Ids), SnapshotError> {
        let map = map_file_with(path, populate)?;
        read_segment_at(&map, 0, check).map(|(seg, ids, _)| (seg, ids))
    }
}

#[cfg(feature = "mmap")]
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
impl Catalog {
    /// Map a snapshot written by [`Catalog::write_to`] instead of reading
    /// it. The keys and directories are built in memory; the packed
    /// postings stay in the file's pages, shared with every other process
    /// that maps the same file and reclaimable by the kernel without a
    /// write. On Linux and Android the pages are faulted in before this
    /// returns, where the kernel maps them populated (`MAP_POPULATE`);
    /// elsewhere they arrive as the checks reach them. The checksums and
    /// structural checks of [`Catalog::read_from`] run in full either way.
    /// [`Catalog::heap_bytes`] then counts the in-memory part only.
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
        mapped::catalog(path.as_ref(), Check::Full)
    }

    /// Map a snapshot as [`Catalog::map_from`] does, without the checks
    /// that read every posting, under the terms of
    /// [`Catalog::read_from_trusted`].
    ///
    /// # Errors
    ///
    /// As [`Catalog::map_from`], except checksum mismatches in the
    /// segments and unsorted runs, which go unnoticed.
    pub fn map_from_trusted(path: impl AsRef<std::path::Path>) -> Result<Catalog, SnapshotError> {
        mapped::catalog(path.as_ref(), Check::Trusted)
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
        mapped::segment(path.as_ref(), Check::Full)
    }

    /// Map a snapshot as [`Segment::map_from`] does, without the checks
    /// that read every posting, under the terms of
    /// [`Catalog::read_from_trusted`].
    ///
    /// # Errors
    ///
    /// As [`Segment::map_from`], except checksum mismatches and unsorted
    /// runs, which go unnoticed.
    pub fn map_from_trusted(path: impl AsRef<std::path::Path>) -> Result<Segment, SnapshotError> {
        mapped::segment(path.as_ref(), Check::Trusted)
    }
}

/// Map the segment snapshot at `path`. A trusted map takes the stored
/// multiplicities and range, and `populate` says whether the pages are
/// faulted in.
#[cfg(feature = "mmap")]
pub(crate) fn map_segment(
    path: &std::path::Path,
    trusted: bool,
    populate: bool,
) -> Result<(Segment, Option<(u32, u32)>), SnapshotError> {
    let check = if trusted { Check::Trusted } else { Check::Full };
    mapped::segment_with(path, check, populate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Posting;

    /// Write the last checksum of a segment snapshot again, over the header
    /// as it now stands and the section checksums the file stores: what an
    /// edit of the header needs to get past the checksum.
    fn reseal(bytes: &mut [u8]) {
        let head = Head::read(&mut &bytes[..]).unwrap();
        let at = bytes.len() - FOOTER;
        let sum =
            |i: usize| u64::from_le_bytes(bytes[at + 8 * i..at + 8 * i + 8].try_into().unwrap());
        let footer = head.footer([sum(0), sum(1), sum(2)]);
        bytes[at..].copy_from_slice(&footer);
    }

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
        // The first byte of the references, so the arrays still describe
        // a valid segment.
        bytes[HEADER] ^= 0x10;
        std::fs::write(&path, &bytes).unwrap();
        let got = Segment::map_from(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(got, Err(SnapshotError::Checksum)), "{got:?}");
    }

    /// A trusted load finds the multiplicities and reference ranges a
    /// full one does: streaks that cross blocks of eight, a saturated
    /// multiplicity, a segment walked in parallel pieces, and none at all.
    #[test]
    fn trusted_loads_agree_with_full_ones() {
        let streaks = [1u32, 3, 1, 1, 12, 2, 1, 9, 1, 1, 1, 4, 20, 1, 3, 300, 1];
        let run: Vec<u32> = streaks
            .iter()
            .enumerate()
            .flat_map(|(r, &n)| (0..n).map(move |_| 7 * r as u32 + 2))
            .collect();
        let times: Vec<u32> = (0..run.len() as u32).collect();
        let long = Segment::from_arrays(&[4], &[0, run.len() as u64], &run, &times);
        let (mut refs, mut offsets) = (Vec::new(), vec![0u64]);
        for k in 0..600u32 {
            refs.extend((0..20).map(|i| k % 7 + i / (k % 5 + 1)));
            offsets.push(refs.len() as u64);
        }
        assert!(refs.len() >= 4 * PARALLEL_MIN);
        let times: Vec<u32> = (0..refs.len() as u32).collect();
        let keys: Vec<u32> = (0..600).map(|k| 3 * k).collect();
        let large = Segment::from_arrays(&keys, &offsets, &refs, &times);
        let small = Segment::from_postings(vec![Posting::new(1, 9, 3), Posting::new(2, 4, 1)]);
        for seg in [long, large, small, Segment::from_postings(Vec::new())] {
            let mut bytes = Vec::new();
            seg.write_to(&mut bytes).unwrap();
            let (full, full_ids) = read_segment(&mut bytes.as_slice(), Check::Full).unwrap();
            let (trusted, ids) = read_segment(&mut bytes.as_slice(), Check::Trusted).unwrap();
            assert_eq!(trusted, full);
            assert_eq!(ids, full_ids);
            for &key in seg.keys() {
                assert_eq!(trusted.range_mult(key), full.range_mult(key), "key {key}");
            }
        }
    }

    /// A full load refuses a file whose stored multiplicities or reference
    /// range are not what its runs hold; a trusted load takes them.
    #[test]
    fn full_loads_check_the_values_the_file_stores() {
        let loaded = |seg: &Segment, check: Check| {
            let mut bytes = Vec::new();
            seg.write_to(&mut bytes).unwrap();
            read_segment(&mut bytes.as_slice(), check)
        };
        // A wrong multiplicity, as only a writer fed inconsistent arrays
        // can produce: the run holds a streak of two and the file says one.
        let keys = [1u32];
        let offsets = [0u64, 2];
        let refs = Packed::from_values(&[3, 3]);
        let times = Packed::from_values(&[0, 1]);
        let lying = Segment::from_parts(keys.to_vec(), &offsets, refs, times, &[1]);
        assert!(
            matches!(
                loaded(&lying, Check::Full),
                Err(SnapshotError::Corrupt(m)) if m.contains("stored")
            ),
            "a wrong multiplicity"
        );
        let (trusted, _) = loaded(&lying, Check::Trusted).unwrap();
        assert_eq!(trusted.range_mult(1), Some((0, 2, 1)), "the file's value");

        // A range that leaves a reference out: only a file edited behind
        // the writer's back holds one, so the test patching the header
        // has to patch the checksum over it too. A full load compares the
        // range with its walk; a trusted load takes it.
        let seg = Segment::from_postings(vec![
            Posting::new(1, 4, 0),
            Posting::new(2, 7, 0),
            Posting::new(3, 2, 0),
        ]);
        let mut bytes = Vec::new();
        seg.write_to(&mut bytes).unwrap();
        let range = (4, 7);
        bytes[32..40].copy_from_slice(&range_word(range).to_le_bytes());
        reseal(&mut bytes);
        assert!(
            matches!(
                Segment::read_from(bytes.as_slice()),
                Err(SnapshotError::Corrupt(m)) if m.contains("stored")
            ),
            "a range that misses a reference"
        );
        let (trusted, ids) = read_segment(&mut bytes.as_slice(), Check::Trusted).unwrap();
        assert_eq!(ids, Some(range), "the file's range");
        assert_eq!(trusted.len(), seg.len());

        // A range outside the reference width cannot describe the file.
        // The width is the largest reference's, so the file itself has to
        // be edited: this one claims a 2-bit reference for 1-bit postings.
        let mut bytes = Vec::new();
        Segment::from_postings(vec![Posting::new(1, 0, 0), Posting::new(1, 1, 0)])
            .write_to(&mut bytes)
            .unwrap();
        bytes[36..40].copy_from_slice(&3u32.to_le_bytes());
        // The header is summed, so an edit alone is a checksum error.
        assert!(matches!(
            Segment::read_from_trusted(bytes.as_slice()),
            Err(SnapshotError::Checksum)
        ));
        reseal(&mut bytes);
        assert!(matches!(
            Segment::read_from_trusted(bytes.as_slice()),
            Err(SnapshotError::Corrupt(m)) if m.contains("range")
        ));
        // And an inverted one.
        bytes[36..40].copy_from_slice(&1u32.to_le_bytes());
        bytes[32..36].copy_from_slice(&2u32.to_le_bytes());
        reseal(&mut bytes);
        assert!(matches!(
            Segment::read_from_trusted(bytes.as_slice()),
            Err(SnapshotError::Corrupt(m)) if m.contains("inverted")
        ));
    }

    /// A trusted load skips the checksum and the order of the runs, and
    /// nothing else.
    #[test]
    fn trusted_loads_skip_only_the_checks_that_read_every_posting() {
        let trusted = |seg: &Segment| {
            let mut bytes = Vec::new();
            seg.write_to(&mut bytes).unwrap();
            Segment::read_from_trusted(bytes.as_slice())
        };
        // Unsorted within a run, but its first reference is still below
        // its last, so the stored range is a range: only the order check
        // tells this file apart from a valid one.
        let unsorted = Segment::from_arrays(&[1], &[0, 3], &[0, 2, 1], &[0, 0, 0]);
        assert!(roundtrip(&unsorted).is_err());
        assert_eq!(trusted(&unsorted).unwrap(), unsorted);
        let bad_keys = Segment::from_arrays(&[5, 3], &[0, 1, 2], &[0, 0], &[0, 1]);
        assert!(matches!(trusted(&bad_keys), Err(SnapshotError::Corrupt(m)) if m.contains("keys")));
        let bad_offsets = Segment::from_arrays(&[1, 2], &[0, 2, 2], &[0, 0], &[0, 1]);
        assert!(
            matches!(trusted(&bad_offsets), Err(SnapshotError::Corrupt(m)) if m.contains("offsets"))
        );

        let good = Segment::from_postings((0..50).map(|i| Posting::new(i % 7, i, i)).collect());
        let mut bytes = Vec::new();
        good.write_to(&mut bytes).unwrap();
        // The first byte of the references.
        let mut flipped = bytes.clone();
        flipped[HEADER] ^= 0x10;
        assert!(matches!(
            Segment::read_from(flipped.as_slice()),
            Err(SnapshotError::Checksum)
        ));
        let loaded = Segment::read_from_trusted(flipped.as_slice()).unwrap();
        assert_eq!(loaded.len(), good.len());
        assert_ne!(loaded, good);
        // The keys, the offsets and the multiplicities are summed by a
        // trusted load too: they are read either way. The last word before
        // the checksums holds the seven multiplicities.
        let index_at = bytes.len() - FOOTER - index_len(good.key_count() as u64) as usize;
        for at in [index_at, bytes.len() - FOOTER - 8] {
            let mut flipped = bytes.clone();
            flipped[at] ^= 1;
            assert!(
                matches!(
                    Segment::read_from_trusted(flipped.as_slice()),
                    Err(SnapshotError::Checksum)
                ),
                "byte {at}"
            );
        }
        // And so are the checksums themselves.
        for at in bytes.len() - FOOTER..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[at] ^= 1;
            assert!(
                matches!(
                    Segment::read_from_trusted(flipped.as_slice()),
                    Err(SnapshotError::Checksum)
                ),
                "checksum byte {at}"
            );
        }
        assert!(matches!(
            Segment::read_from_trusted(&bytes[..bytes.len() - 1]),
            Err(SnapshotError::Truncated)
        ));
        let mut wrong = bytes.clone();
        wrong[4] = 9;
        assert!(matches!(
            Segment::read_from_trusted(wrong.as_slice()),
            Err(SnapshotError::UnsupportedVersion(9))
        ));

        #[cfg(feature = "mmap")]
        {
            let path =
                std::env::temp_dir().join(format!("afpindex-trusted-{}.afx", std::process::id()));
            std::fs::write(&path, &flipped).unwrap();
            let full = Segment::map_from(&path);
            let mapped = Segment::map_from_trusted(&path);
            std::fs::remove_file(&path).unwrap();
            assert!(matches!(full, Err(SnapshotError::Checksum)), "{full:?}");
            assert_eq!(mapped.unwrap(), loaded);
        }
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
        assert!(matches!(
            Catalog::read_from_trusted(bytes.as_slice()),
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
