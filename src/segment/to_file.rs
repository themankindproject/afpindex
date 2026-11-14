//! Merges that write straight into a snapshot file (`std`).
//!
//! [`merge_to_file`] joins segments the way [`Segment::merge_all`] does and
//! writes the result as a segment snapshot (format 3, see
//! [`crate::snapshot`]) without ever holding the result in memory. The
//! merge is cut into pieces by hash, as the merge on every core cuts it
//! ([`Plan`]): a counting pass over the pieces gives the key count, the
//! posting count and the reference range of the result, which fix the
//! header and the offset of every section of the file, and a second pass
//! over the same pieces appends each to the file in piece order. The bytes
//! are the ones [`Segment::write_to`] writes for the merged segment.
//!
//! The file is written by three handles: the header, the keys, the offsets
//! and the multiplicities through one, and each packed array through its
//! own, positioned at its section. A piece rarely ends on a whole byte, so
//! the byte two pieces share is put together in memory as the pieces
//! arrive and every other byte has a single writer.
//!
//! Memory is bounded by what a piece is worth and by what the result holds
//! per distinct hash (its keys, the start of each run and its
//! multiplicity, 13 bytes), never by the posting count. A piece holds at
//! most [`PIECE`] postings of input, and no piece is started while it is
//! more than [`WINDOW`] times the thread count ahead of the one the file
//! waits for, so pieces that merge early cannot pile up.

use alloc::vec::Vec;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Condvar, Mutex, PoisonError};

use super::parallel::{Parts, Plan};
use super::{MAX_POSTINGS, Segment, postings, widths};
use crate::packed::Packed;
use crate::par::{lock, run_tasks, slots};
use crate::snapshot::{
    Checksum, HEADER, SEGMENT_MAGIC, checksum_of_segment, pad_len, range_word, write_header,
};

/// Postings of input a piece holds at most: a piece in flight is tens of
/// megabytes however large the merge is.
const PIECE: usize = 1 << 22;

/// Pieces a thread may start beyond the one the file waits for, as a
/// multiple of the thread count.
const WINDOW: usize = 2;

/// Bytes gathered in a buffer before they go to a handle.
const BUFFER: usize = 1 << 18;

/// Bytes read back from the file per call while it is summed.
const CHUNK: usize = 1 << 18;

/// What a merge wrote.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Written {
    /// Smallest and largest reference, or `None` when every posting was
    /// dropped and the file holds an empty segment.
    pub(crate) ids: Option<(u32, u32)>,
    /// The most pieces merged and not yet in the file at one time: what a
    /// test holds against the window.
    #[cfg_attr(not(test), expect(dead_code))]
    pub(crate) held: usize,
}

/// How a merge to a file runs: the threads, the size of a piece, and a
/// limit on the bytes a handle takes before a write fails, which the tests
/// use to reach the error paths.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Settings {
    /// Threads the pieces merge on.
    pub(crate) threads: usize,
    /// Postings of input a piece holds at most.
    pub(crate) piece: usize,
    /// Bytes a handle writes before it fails, or `None` to write.
    pub(crate) fail_after: Option<u64>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            threads: crate::par::threads(),
            piece: PIECE,
            fail_after: None,
        }
    }
}

/// Merge `sides` as [`Segment::merge_all`] does and write the result to
/// `path` as a segment snapshot, holding no array of the result in memory.
///
/// The file is removed again if anything fails, so a caller never finds a
/// half-written snapshot where it asked for one. `settings` says how the
/// merge runs; [`Settings::default`] is the threads of the machine and the
/// default piece size.
///
/// # Errors
///
/// Any error of creating, writing, reading back or syncing the file, and an
/// error rather than the panic [`Segment::merge_all`] raises when the result
/// would hold 2<sup>40</sup> postings.
pub(crate) fn merge_to_file<F: Fn(u32) -> bool + Sync>(
    sides: &[(&Segment, Option<&F>)],
    path: &Path,
    settings: Settings,
) -> std::io::Result<Written> {
    let widths = widths(sides);
    let pieces = postings(sides).div_ceil(settings.piece.max(1)).max(1);
    let plan = Plan::new(sides, pieces);
    // The counting pass fixes the header and where every section starts,
    // and each piece is dropped as soon as it is counted: four numbers
    // survive the whole merge.
    let counts = merge_pieces(&plan, settings.threads, count_of);
    let mut total = Count {
        keys: counts.iter().map(|one| one.keys).sum(),
        postings: 0,
        ids: None,
    };
    for one in &counts {
        total.postings += one.postings;
        total.ids = ids_of(total.ids, one.ids);
    }
    if total.postings >= MAX_POSTINGS {
        return Err(std::io::Error::other(
            "the merged segment holds 2^40 postings or more",
        ));
    }
    // The section sizes are counts as the target's `usize` holds them: a
    // 32-bit target cannot lay out a file this large, mapped inputs or no.
    for count in [total.keys, total.postings] {
        if usize::try_from(count).is_err() {
            return Err(std::io::Error::other(
                "the merged segment is larger than this target can address",
            ));
        }
    }
    let layout = Layout::new(widths, total);
    let guard = RemoveOnFailure::new(path);
    let written = write_pieces(&plan, path, &layout, widths, settings)?;
    guard.keep();
    Ok(written)
}

/// The range that covers both ranges, if either has one.
fn ids_of(all: Option<(u32, u32)>, one: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (all, one) {
        (Some((lo, hi)), Some((a, b))) => Some((lo.min(a), hi.max(b))),
        (a, b) => a.or(b),
    }
}

/// What one piece adds to the result.
#[derive(Clone, Copy, Debug)]
struct Count {
    keys: u64,
    postings: u64,
    ids: Option<(u32, u32)>,
}

/// The keys, the postings and the reference range of one merged piece.
fn count_of((keys, starts, _, refs, _): &Parts) -> Count {
    let postings = refs.len();
    let mut ids: Option<(u32, u32)> = None;
    for (k, &start) in starts.iter().enumerate() {
        let lo = start as usize;
        let hi = starts.get(k + 1).map_or(postings, |&next| next as usize);
        if lo < hi {
            // A run's references ascend, so its first bounds it from below
            // and its last from above.
            ids = ids_of(ids, Some((refs.get(lo), refs.get(hi - 1))));
        }
    }
    Count {
        keys: keys.len() as u64,
        postings: postings as u64,
        ids,
    }
}

/// Deal the runs out to the pieces, merge every piece on up to `threads`
/// threads, and collect what `each` makes of each piece in piece order.
fn merge_pieces<F, T>(
    plan: &Plan<'_, F>,
    threads: usize,
    each: impl Fn(&Parts) -> T + Sync,
) -> Vec<T>
where
    F: Fn(u32) -> bool + Sync,
    T: Send,
{
    plan.deal(threads);
    let out = slots::<T>(plan.pieces);
    run_tasks(threads, plan.pieces, |piece| {
        let parts = plan.merge(piece);
        *lock(&out[piece]) = Some(each(&parts));
    });
    out.into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(PoisonError::into_inner)
                .expect("every piece is merged")
        })
        .collect()
}

/// Where every section of the file starts, from the widths and the counts
/// of the result: the layout [`Segment::write_to`] gives a segment of that
/// shape.
struct Layout {
    keys: u64,
    postings: u64,
    ids: Option<(u32, u32)>,
    keys_at: u64,
    mults_at: u64,
    refs_at: u64,
    refs_len: u64,
    times_at: u64,
    times_len: u64,
    trailer_at: u64,
    bytes: u64,
}

impl Layout {
    fn new((ref_bits, time_bits): (u32, u32), total: Count) -> Self {
        let postings = total.postings;
        let keys_at = HEADER as u64;
        let offsets_at = (keys_at + 4 * total.keys).next_multiple_of(8);
        let mults_at = offsets_at + 8 * (total.keys + 1);
        let refs_at = (mults_at + total.keys).next_multiple_of(8);
        // Both counts fit the address space: `merge_to_file` refuses them
        // otherwise.
        let n = usize::try_from(postings).unwrap_or(usize::MAX);
        let refs_len = Packed::data_len(n, ref_bits) as u64;
        let times_at = (refs_at + refs_len).next_multiple_of(8);
        let times_len = Packed::data_len(n, time_bits) as u64;
        let trailer_at = (times_at + times_len).next_multiple_of(8);
        Self {
            keys: total.keys,
            postings,
            ids: total.ids,
            keys_at,
            mults_at,
            refs_at,
            refs_len,
            times_at,
            times_len,
            trailer_at,
            bytes: trailer_at + 8,
        }
    }

    /// The header's reference range word, zero without postings.
    fn range(&self) -> (u32, u32) {
        self.ids.unwrap_or((0, 0))
    }

    /// The checksum of the header words: what the file's trailer starts
    /// from.
    fn checksum(&self, widths: (u32, u32)) -> Checksum {
        checksum_of_segment(widths, self.keys, self.postings, self.range())
    }
}

/// A path written here, removed again unless the merge finished.
struct RemoveOnFailure<'a> {
    path: &'a Path,
    keep: bool,
}

impl<'a> RemoveOnFailure<'a> {
    fn new(path: &'a Path) -> Self {
        Self { path, keep: false }
    }

    fn keep(mut self) {
        self.keep = true;
    }
}

impl Drop for RemoveOnFailure<'_> {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(self.path);
        }
    }
}

/// The writing pass: merge every piece again and append it to the file, in
/// piece order, then write the collected sections and the checksum.
fn write_pieces<F: Fn(u32) -> bool + Sync>(
    plan: &Plan<'_, F>,
    path: &Path,
    layout: &Layout,
    widths: (u32, u32),
    settings: Settings,
) -> std::io::Result<Written> {
    let mut head = Handle::create(path, settings.fail_after)?;
    // Each packed array is written through a handle of its own, positioned
    // at its section, so that both are appended in piece order at once.
    let refs = Handle::open_at(path, layout.refs_at, settings.fail_after)?;
    let times = Handle::open_at(path, layout.times_at, settings.fail_after)?;
    write_header(
        &mut head,
        SEGMENT_MAGIC,
        widths,
        layout.keys,
        layout.postings,
    )?;
    head.put(&range_word(layout.range()).to_le_bytes())?;
    let keys = usize::try_from(layout.keys)
        .map_err(|_| std::io::Error::other("the result has more keys than the address space"))?;
    let writer = Writer {
        next: 0,
        postings: 0,
        keys: Vec::with_capacity(keys),
        starts: Vec::with_capacity(keys.saturating_add(1)),
        mults: Vec::with_capacity(keys),
        refs: Sink::new(refs, widths.0),
        times: Sink::new(times, widths.1),
        held: 0,
        high: 0,
        stop: false,
    };
    let gate = Gate {
        wake: Condvar::new(),
        writer: Mutex::new(writer),
    };
    write_in_pieces(plan, &gate, settings)?;
    let Writer {
        mut starts,
        keys,
        mults,
        refs,
        times,
        postings,
        high,
        ..
    } = gate
        .writer
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    // The last start is where the postings end, which closes the offsets.
    starts.push(postings);
    head.seek_to(layout.keys_at)?;
    head.put_u32s(&keys)?;
    head.pad_to(layout.keys_at + 4 * keys.len() as u64)?;
    head.put_u64s(&starts)?;
    head.put(&mults)?;
    head.pad_to(layout.mults_at + mults.len() as u64)?;
    head.flush()?;
    refs.finish()?;
    times.finish()?;
    // The checksum covers the header words and then the arrays in file
    // order, which no pass over the pieces can sum while two arrays are
    // written side by side: one sequential read of the finished file sums
    // it, a block at a time and with no buffer in proportion to the file.
    let sum = sum_file(path, layout, widths)?;
    head.seek_to(layout.trailer_at)?;
    head.put(&sum.to_le_bytes())?;
    head.flush()?;
    head.sync()?;
    if std::fs::metadata(path)?.len() != layout.bytes {
        return Err(std::io::Error::other(
            "the merge to a file wrote a length of its own",
        ));
    }
    Ok(Written {
        ids: layout.ids,
        held: high,
    })
}

/// The file as the pieces are written into it, and how many of them are
/// merged but not yet written.
struct Writer {
    /// The next piece to write.
    next: usize,
    /// Postings written so far.
    postings: u64,
    keys: Vec<u32>,
    /// Where each key's run starts, counted from the file's first posting.
    starts: Vec<u64>,
    mults: Vec<u8>,
    refs: Sink,
    times: Sink,
    /// Pieces merged but not yet written.
    held: usize,
    /// The most pieces held at one time.
    high: usize,
    /// Set once a write has failed: no other piece is merged.
    stop: bool,
}

impl Writer {
    /// Append one merged piece after the pieces before it.
    fn place(&mut self, (keys, starts, mults, refs, times): Parts) -> std::io::Result<()> {
        self.keys.extend_from_slice(&keys);
        self.starts
            .extend(starts.iter().map(|&start| start + self.postings));
        self.mults.extend_from_slice(&mults);
        self.refs.append(&refs)?;
        self.times.append(&times)?;
        self.postings += refs.len() as u64;
        self.next += 1;
        Ok(())
    }
}

/// The writer and the threads waiting for room ahead of it.
struct Gate {
    writer: Mutex<Writer>,
    wake: Condvar,
}

/// Merge the pieces a thread at a time and write them in order.
///
/// A thread does not start a piece more than [`WINDOW`] times the thread
/// count ahead of the one the file waits for: a piece merged before the
/// pieces ahead of it would otherwise wait in memory while the threads run
/// on, and that wait is what the window bounds.
fn write_in_pieces<F: Fn(u32) -> bool + Sync>(
    plan: &Plan<'_, F>,
    gate: &Gate,
    settings: Settings,
) -> std::io::Result<()> {
    let threads = settings.threads.max(1);
    let window = WINDOW * threads;
    let failure: Mutex<Option<std::io::Error>> = Mutex::new(None);
    let merged = slots::<Parts>(plan.pieces);
    // The counting pass took the dealt runs as it merged them.
    plan.deal(threads);
    run_tasks(threads, plan.pieces, |piece| {
        // Wait for the piece to come within the window; the piece the file
        // waits for is always within it, so a thread holding a piece always
        // makes progress.
        {
            let mut w = lock(&gate.writer);
            while !w.stop && piece >= w.next + window {
                w = gate.wake.wait(w).unwrap_or_else(PoisonError::into_inner);
            }
            if w.stop {
                return;
            }
        }
        let parts = plan.merge(piece);
        {
            // The piece is published under the writer's lock, so that the
            // count of pieces held moves with it: the thread that writes
            // it takes it and gives the count back.
            let mut w = lock(&gate.writer);
            w.held += 1;
            w.high = w.high.max(w.held);
            *lock(&merged[piece]) = Some(parts);
        }
        // Write every piece that is next in line: this one, or the ones
        // before it that finished first.
        loop {
            let mut w = lock(&gate.writer);
            if w.stop {
                return;
            }
            let Some(parts) = merged.get(w.next).and_then(|slot| lock(slot).take()) else {
                return;
            };
            let placed = w.place(parts);
            w.held -= 1;
            gate.wake.notify_all();
            if let Err(e) = placed {
                w.stop = true;
                *lock(&failure) = Some(e);
                gate.wake.notify_all();
                return;
            }
        }
    });
    if let Some(e) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    let w = lock(&gate.writer);
    if w.next != plan.pieces || w.held != 0 {
        return Err(std::io::Error::other("a piece of the merge is missing"));
    }
    Ok(())
}

/// A handle on the file being written, buffered, with a byte count a test
/// can make fail.
struct Handle {
    out: BufWriter<File>,
    /// Bytes put through this handle.
    written: u64,
    /// Bytes after which a write fails, or `None` to write.
    fail_after: Option<u64>,
}

impl Handle {
    fn create(path: &Path, fail_after: Option<u64>) -> std::io::Result<Self> {
        Ok(Self::wrap(File::create(path)?, fail_after))
    }

    /// A handle on `path` positioned at byte `at`.
    fn open_at(path: &Path, at: u64, fail_after: Option<u64>) -> std::io::Result<Self> {
        let mut file = OpenOptions::new().write(true).open(path)?;
        file.seek(SeekFrom::Start(at))?;
        Ok(Self::wrap(file, fail_after))
    }

    fn wrap(file: File, fail_after: Option<u64>) -> Self {
        Self {
            out: BufWriter::with_capacity(BUFFER, file),
            written: 0,
            fail_after,
        }
    }

    fn put(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.written += bytes.len() as u64;
        if self.fail_after.is_some_and(|limit| self.written > limit) {
            return Err(std::io::Error::other("the merge to a file failed a write"));
        }
        self.out.write_all(bytes)
    }

    fn put_u32s(&mut self, values: &[u32]) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(BUFFER);
        for values in values.chunks(BUFFER / 4) {
            buf.clear();
            buf.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            self.put(&buf)?;
        }
        Ok(())
    }

    fn put_u64s(&mut self, values: &[u64]) -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(BUFFER);
        for values in values.chunks(BUFFER / 8) {
            buf.clear();
            buf.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            self.put(&buf)?;
        }
        Ok(())
    }

    /// Zero bytes that bring `pos` to the next multiple of 8.
    fn pad_to(&mut self, pos: u64) -> std::io::Result<()> {
        let pad = pad_len(pos);
        if pad > 0 {
            self.put(&[0u8; 8][..pad])?;
        }
        Ok(())
    }

    fn seek_to(&mut self, at: u64) -> std::io::Result<()> {
        self.out.flush()?;
        self.out.seek(SeekFrom::Start(at))?;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }

    fn sync(&mut self) -> std::io::Result<()> {
        self.out.get_ref().sync_all()
    }
}

impl Write for Handle {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.put(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Self::flush(self)
    }
}

/// One packed array's section: the values appended so far and the bits
/// already placed in the byte at bit `at` on, its low `at % 8` of them.
struct Sink {
    out: Handle,
    width: u32,
    /// Bits appended so far.
    at: u64,
    /// The placed bits of the byte `at` falls in, below `at % 8`.
    pending: u8,
    /// Bytes gathered between writes.
    buf: Vec<u8>,
}

impl Sink {
    fn new(out: Handle, width: u32) -> Self {
        Self {
            out,
            width,
            at: 0,
            pending: 0,
            buf: Vec::new(),
        }
    }

    /// Append `piece`'s values after the values appended so far.
    fn append(&mut self, piece: &Packed) -> std::io::Result<()> {
        let total = piece.len() as u64 * u64::from(self.width);
        if total == 0 {
            return Ok(());
        }
        let phase = self.at % 8;
        let mut from = 0;
        if phase != 0 {
            // The piece's first bits finish the byte the piece before it
            // ended in.
            let k = (8 - phase).min(total);
            self.pending |= (piece.bits_at(0, k as u32) << phase) as u8;
            if phase + k == 8 {
                self.out.put(&[self.pending])?;
                self.pending = 0;
            }
            from = k;
        }
        // From there the piece's bits start a byte of the file, so the
        // whole bytes among them are copied as they are.
        let own = (total - from) / 8;
        debug_assert!(
            own == 0 || from + 8 * own <= piece.len() as u64 * u64::from(self.width),
            "bits past the piece"
        );
        let mut buf = core::mem::take(&mut self.buf);
        let mut done = 0;
        while done < own {
            let n = (own - done).min(BUFFER as u64) as usize;
            buf.clear();
            buf.resize(n, 0);
            piece.copy_bits_into(&mut buf, from + 8 * done);
            self.out.put(&buf)?;
            done += n as u64;
        }
        let rest = (total - from) % 8;
        if rest > 0 {
            // The piece's last bits start the next shared byte.
            self.pending = piece.bits_at(total - rest, rest as u32) as u8;
        }
        self.at += total;
        self.buf = buf;
        Ok(())
    }

    /// Write what is left of the array and pad its section to 8 bytes.
    fn finish(mut self) -> std::io::Result<()> {
        if !self.at.is_multiple_of(8) {
            self.out.put(&[self.pending])?;
        }
        self.out.pad_to(self.at.div_ceil(8))?;
        self.out.flush()
    }
}

/// Sum the finished file as [`Segment::write_to`] sums what it writes: the
/// header words, then the keys, the offsets, the multiplicities and the two
/// packed arrays in file order, a block at a time.
fn sum_file(path: &Path, layout: &Layout, widths: (u32, u32)) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(HEADER as u64))?;
    let mut ck = layout.checksum(widths);
    let mut buf = alloc::vec![0u8; CHUNK];
    // The keys are summed as values, not as the bytes they lie in.
    read_section(&mut file, layout.keys * 4, &mut buf, |bytes| {
        for value in bytes.chunks_exact(4) {
            ck.word(u64::from(u32::from_le_bytes(
                value.try_into().expect("4 bytes"),
            )));
        }
    })?;
    skip(&mut file, layout.keys * 4)?;
    read_section(&mut file, (layout.keys + 1) * 8, &mut buf, |bytes| {
        for value in bytes.chunks_exact(8) {
            ck.word(u64::from_le_bytes(value.try_into().expect("8 bytes")));
        }
    })?;
    read_section(&mut file, layout.keys, &mut buf, |bytes| ck.words(bytes))?;
    skip(&mut file, layout.keys)?;
    read_section(&mut file, layout.refs_len, &mut buf, |bytes| {
        ck.words(bytes);
    })?;
    skip(&mut file, layout.refs_len)?;
    read_section(&mut file, layout.times_len, &mut buf, |bytes| {
        ck.words(bytes);
    })?;
    skip(&mut file, layout.times_len)?;
    debug_assert_eq!(file.stream_position()?, layout.trailer_at);
    Ok(ck.finish())
}

/// Read `bytes` of `file` in blocks of `buf` and fold each block into
/// `each`: one sequential read of the section, whatever its length.
fn read_section(
    file: &mut File,
    bytes: u64,
    buf: &mut [u8],
    mut each: impl FnMut(&[u8]),
) -> std::io::Result<()> {
    let mut remaining = bytes;
    while remaining > 0 {
        let take = remaining.min(buf.len() as u64) as usize;
        file.read_exact(&mut buf[..take])?;
        each(&buf[..take]);
        remaining -= take as u64;
    }
    Ok(())
}

/// Step over the padding after a section of `bytes`.
fn skip(file: &mut File, bytes: u64) -> std::io::Result<()> {
    let pad = pad_len(bytes) as i64;
    if pad > 0 {
        file.seek(SeekFrom::Current(pad))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Posting;

    /// A directory named after the test and the process, removed at the
    /// end.
    struct Scratch {
        dir: std::path::PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            // Tests run at the same time in one process: a counter keeps
            // their directories apart.
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "afpindex-merge-to-file-{name}-{}-{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("a fresh directory");
            Self { dir }
        }

        fn path(&self, name: &str) -> std::path::PathBuf {
            self.dir.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// `rows` postings over `hashes` hashes and `refs` references.
    fn segment(rng: &mut Rng, rows: usize, hashes: u64, refs: u32, max_time: u64) -> Segment {
        Segment::from_postings(
            (0..rows)
                .map(|_| {
                    Posting::new(
                        (rng.below(hashes) as u32).wrapping_mul(2_654_435_761),
                        rng.below(u64::from(refs)) as u32,
                        rng.below(max_time) as u32,
                    )
                })
                .collect(),
        )
    }

    type Keep<'a> = &'a (dyn Fn(u32) -> bool + Sync);

    /// What `Segment::merge_all` gives, written by `Segment::write_to`:
    /// the bytes every merge to a file must match.
    fn expected<F: Fn(u32) -> bool + Sync>(sides: &[(&Segment, Option<&F>)]) -> (Vec<u8>, Segment) {
        let merged = Segment::merge_all(sides);
        let mut bytes = Vec::new();
        merged.write_to(&mut bytes).expect("writing to memory");
        (bytes, merged)
    }

    /// Merge to a file and check it against the bytes of the same merge in
    /// memory.
    fn check<F: Fn(u32) -> bool + Sync>(
        sides: &[(&Segment, Option<&F>)],
        settings: Settings,
        what: &str,
    ) -> Segment {
        let (want, merged) = expected(sides);
        let scratch = Scratch::new(what);
        let path = scratch.path("out.afs");
        let written = merge_to_file(sides, &path, settings).expect("the merge writes");
        let got = std::fs::read(&path).expect("the file it wrote");
        assert_eq!(got, want, "{what}: bytes");
        assert_eq!(written.ids, merged_refs(&merged), "{what}: the range");
        let back = Segment::read_from(got.as_slice()).expect("a valid snapshot");
        assert_eq!(back, merged, "{what}: the file reads back");
        merged
    }

    /// The smallest and the largest reference of a segment, if any.
    fn merged_refs(seg: &Segment) -> Option<(u32, u32)> {
        let mut ids: Option<(u32, u32)> = None;
        for posting in seg.postings() {
            ids = ids_of(ids, Some((posting.reference, posting.reference)));
        }
        ids
    }

    /// The settings a test merges with: `threads` threads and pieces of
    /// `piece` postings.
    fn settings(threads: usize, piece: usize) -> Settings {
        Settings {
            threads,
            piece,
            fail_after: None,
        }
    }

    /// [`check`] for sources that really are cut into several pieces.
    fn check_cut<F: Fn(u32) -> bool + Sync>(
        sides: &[(&Segment, Option<&F>)],
        settings: Settings,
        what: &str,
    ) -> Segment {
        assert!(
            postings(sides) > settings.piece,
            "{what}: {} postings in pieces of {} is one piece",
            postings(sides),
            settings.piece
        );
        check(sides, settings, what)
    }

    /// Random sources, one to six of them, some with a `keep`: a keep that
    /// drops some references, one that leaves whole runs empty, and one
    /// that keeps nothing. Every thread count and piece size gives the
    /// bytes `Segment::write_to` writes for the same merge.
    #[test]
    fn random_sources_agree_with_a_merge_in_memory() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let a = segment(&mut rng, 9_000, 600, 400, 40_000);
        let b = segment(&mut rng, 3_000, 600, 300, 70);
        let c = segment(&mut rng, 1_500, 40, 900, 40_000);
        let one = Segment::from_postings(vec![Posting::new(5, 1, 1)]);
        let empty = Segment::default();
        let keeps: [Keep<'_>; 5] = [
            &|_| true,
            &|r| r % 3 != 0,
            &|r| r >= 290,
            &|r| r < 40,
            &|_| false,
        ];
        let sets: [&[&Segment]; 6] = [
            &[&a],
            &[&a, &b],
            &[&a, &b, &c],
            &[&c, &a],
            &[&one, &empty],
            &[&a, &empty, &b],
        ];
        for set in &sets {
            for keep in &keeps {
                let sides: Vec<(&Segment, Option<&Keep<'_>>)> =
                    set.iter().map(|&s| (s, Some(keep))).collect();
                // Pieces of 300 postings cut the 13 500 rows into 45 of
                // them, so a piece ends inside a byte at many phases.
                for (threads, piece) in [(1, 300), (3, 700), (4, 2_500), (2, 4_500)] {
                    let settings = settings(threads, piece);
                    if postings(&sides) > settings.piece {
                        check_cut(&sides, settings, "keep");
                    } else {
                        check(&sides, settings, "keep, one piece");
                    }
                }
            }
            // Four merges without a filter, two of them the same segment.
            let sides: Vec<(&Segment, Option<&Keep<'_>>)> =
                set.iter().map(|&s| (s, None)).collect();
            let settings = settings(1, 800);
            if postings(&sides) > settings.piece {
                check_cut(&sides, settings, "no keep");
            } else {
                check(&sides, settings, "no keep, one piece");
            }
        }
    }

    /// References and times of widths 0, 1, 17 and 32, and runs of one
    /// hash whose references interleave, so that pieces end inside bytes at
    /// every phase.
    #[test]
    fn every_width_and_interleaved_runs_agree_with_a_merge_in_memory() {
        // Width 0: one reference of time 0, the result of a single posting
        // of reference 0 at time 0.
        let zero = Segment::from_postings(vec![Posting::new(3, 0, 0)]);
        assert_eq!(zero.widths(), (0, 0));
        // Width 1: references 0 and 1, times 0 and 1.
        let one = Segment::from_postings(vec![
            Posting::new(3, 1, 1),
            Posting::new(4, 0, 0),
            Posting::new(5, 1, 0),
        ]);
        assert_eq!(one.widths(), (1, 1));
        // Width 17: a reference and a time above 2^16.
        let seventeen = Segment::from_postings(vec![
            Posting::new(7, 70_000, 90_000),
            Posting::new(7, 70_000, 31),
            Posting::new(8, 65_537, 65_536),
        ]);
        assert_eq!(seventeen.widths(), (17, 17));
        // Width 32: the widest reference and time.
        let full = Segment::from_postings(vec![
            Posting::new(9, u32::MAX, u32::MAX),
            Posting::new(9, u32::MAX - 1, 1),
        ]);
        assert_eq!(full.widths(), (32, 32));
        // Runs of one hash that interleave: every reference of each source
        // sits between two of the other's.
        let a = Segment::from_postings(vec![
            Posting::new(1, 0, 0),
            Posting::new(1, 2, 2),
            Posting::new(1, 4, 4),
        ]);
        let b = Segment::from_postings(vec![
            Posting::new(1, 1, 1),
            Posting::new(1, 3, 3),
            Posting::new(1, 5, 5),
        ]);
        let sets: [&[&Segment]; 5] = [
            &[&zero],
            &[&one, &zero],
            &[&seventeen, &one],
            &[&full, &seventeen, &zero],
            &[&a, &b],
        ];
        for set in sets {
            let sides: Vec<(&Segment, Option<&Keep<'_>>)> =
                set.iter().copied().map(|s| (s, None)).collect();
            for (threads, piece) in [(1, 1), (2, 2), (3, 3)] {
                let merged = check(&sides, settings(threads, piece), "widths");
                assert!(!merged.is_empty() || set.iter().all(|s| s.is_empty()));
            }
            let kept = |_| true;
            let keep: Keep<'_> = &kept;
            let filtered: Vec<(&Segment, Option<&Keep<'_>>)> =
                set.iter().copied().map(|s| (s, Some(&keep))).collect();
            check(&filtered, settings(2, 2), "widths, kept");
        }
    }

    /// A merge whose result is empty, and one a `keep` empties: the file is
    /// the one an empty segment writes, with no reference range.
    #[test]
    fn an_empty_result_writes_an_empty_segment() {
        let scratch = Scratch::new("empty");
        let nothing: [(&Segment, Option<&Keep<'_>>); 0] = [];
        for (sides, what) in [
            (nothing.as_slice(), "no source"),
            (&[(&Segment::default(), None)][..], "an empty source"),
        ] {
            let path = scratch.path("empty.afs");
            let written =
                merge_to_file(sides, &path, Settings::default()).expect("the merge writes");
            assert_eq!(written.ids, None, "{what}");
            let mut want = Vec::new();
            Segment::default()
                .write_to(&mut want)
                .expect("writing to memory");
            assert_eq!(std::fs::read(&path).expect("the file"), want, "{what}");
            let back = Segment::read_from(want.as_slice()).expect("a valid snapshot");
            assert!(back.is_empty());
        }
        // A keep that rejects every reference empties a segment of its
        // postings, at every piece size.
        let mut rng = Rng(0x4146);
        let source = segment(&mut rng, 2_000, 200, 100, 500);
        let none = |_: u32| false;
        let keep: Keep<'_> = &none;
        let sides: Vec<(&Segment, Option<&Keep<'_>>)> = vec![(&source, Some(&keep))];
        for (threads, piece) in [(1, 100), (3, 300), (2, 5_000)] {
            let settings = settings(threads, piece);
            let merged = check(&sides, settings, "everything rejected");
            assert!(merged.is_empty(), "{threads} threads, pieces of {piece}");
        }
    }

    /// A source mapped from a file rather than owned merges to the same
    /// bytes, and the result maps back equal to the merge in memory.
    #[cfg(feature = "mmap")]
    #[test]
    fn a_mapped_source_agrees_with_a_merge_in_memory() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let owned = segment(&mut rng, 12_000, 900, 300, 9_000);
        let other = segment(&mut rng, 4_000, 900, 150, 3_000);
        let scratch = Scratch::new("mapped-source");
        let source = scratch.path("source.afs");
        let mut bytes = Vec::new();
        owned.write_to(&mut bytes).expect("writing to memory");
        std::fs::write(&source, &bytes).expect("writing the source file");
        let mapped = Segment::map_from(&source).expect("a valid snapshot");
        assert_eq!(mapped, owned);

        let kept = |r: u32| !r.is_multiple_of(3);
        let keep: Keep<'_> = &kept;
        for (threads, piece) in [(1, 500), (3, 900), (4, 7_000)] {
            let sides: Vec<(&Segment, Option<&Keep<'_>>)> =
                vec![(&mapped, Some(&keep)), (&other, None)];
            let (want, merged) = expected(&sides);
            let path = scratch.path("merged-by-pieces.afs");
            let settings = settings(threads, piece);
            merge_to_file(&sides, &path, settings).expect("the merge writes");
            assert_eq!(
                std::fs::read(&path).expect("the file it wrote"),
                want,
                "{threads} threads, pieces of {piece}"
            );
            // The result maps back to what the merge in memory is, and its
            // pages are the file's.
            let back = Segment::map_from(&path).expect("a valid snapshot");
            assert_eq!(back, merged, "the file maps back");
            for &hash in merged.keys() {
                assert_eq!(
                    back.range_mult(hash),
                    merged.range_mult(hash),
                    "the run of {hash}"
                );
            }
        }
        // A source that is itself a file a merge wrote.
        let sides: Vec<(&Segment, Option<&Keep<'_>>)> = vec![(&mapped, None)];
        let path = scratch.path("twice.afs");
        merge_to_file(&sides, &path, settings(2, 300)).expect("the merge writes");
        let once = Segment::map_from(&path).expect("a valid snapshot");
        let again = scratch.path("thrice.afs");
        let sides: Vec<(&Segment, Option<&Keep<'_>>)> = vec![(&once, None)];
        merge_to_file(&sides, &again, settings(3, 400)).expect("the second merge writes");
        let (want, _) = expected(&sides);
        assert_eq!(std::fs::read(&again).expect("the second file"), want);
    }

    /// A merge of many small pieces on several threads holds no more than
    /// the window plus one piece per thread.
    #[test]
    fn the_pieces_held_stay_inside_the_window() {
        let mut rng = Rng(0x0DDB_A110_5EED_1234);
        let a = segment(&mut rng, 20_000, 2_000, 500, 5_000);
        let b = segment(&mut rng, 8_000, 2_000, 200, 5_000);
        let sides: Vec<(&Segment, Option<&Keep<'_>>)> = vec![(&a, None), (&b, None)];
        let scratch = Scratch::new("window");
        for threads in [1usize, 2, 4] {
            // Pieces of 32 postings: 875 of them, so threads run well
            // ahead of the file if the merge lets them.
            let written = merge_to_file(&sides, &scratch.path("window.afs"), settings(threads, 32))
                .expect("the merge writes");
            assert!(
                written.held <= WINDOW * threads + threads,
                "{threads} threads held {} pieces",
                written.held
            );
            assert!(
                written.held >= threads.min(2),
                "{threads} threads, held {}",
                written.held
            );
        }
    }

    /// A path in a directory that does not exist is an error and leaves no
    /// file, and a write that fails after the file was created removes it.
    #[test]
    fn a_failed_merge_removes_its_file() {
        let mut rng = Rng(0x4146_4958_0000_0001);
        let a = segment(&mut rng, 6_000, 800, 200, 3_000);
        let sides: Vec<(&Segment, Option<&Keep<'_>>)> = vec![(&a, None)];
        let scratch = Scratch::new("failed-merge");
        let missing = scratch.path("no-such-directory").join("out.afs");
        let err =
            merge_to_file(&sides, &missing, Settings::default()).expect_err("no such directory");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            !missing.exists(),
            "no file in a directory that is not there"
        );

        // A limit below the first byte that reaches a handle: the file is
        // created and then removed.
        // Every handle counts its own bytes, so these limits fail the
        // header, one of the packed arrays, and the keys and offsets
        // written after the pass.
        for limit in [1u64, 64, 1_000, 5_000] {
            let path = scratch.path("doomed.afs");
            let settings = Settings {
                threads: 2,
                piece: 500,
                fail_after: Some(limit),
            };
            let err = merge_to_file(&sides, &path, settings).expect_err("the write fails");
            assert_eq!(err.kind(), std::io::ErrorKind::Other, "{limit}");
            assert!(!path.exists(), "the file went with the error at {limit}");
        }
        // And the next merge without the limit writes the file.
        let path = scratch.path("fine.afs");
        merge_to_file(&sides, &path, Settings::default()).expect("the merge writes");
        assert!(path.exists());
    }
}
