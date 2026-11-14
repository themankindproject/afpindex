//! Merges and directories on every core (`std`).
//!
//! A merge is cut into pieces by hash: piece `i` holds the hashes from
//! `bounds[i]` up to `bounds[i + 1]`, chosen so that the pieces hold about
//! equal postings. Every piece is merged on its own by the same code as a
//! merge on one thread, and the pieces are then joined in order. The
//! result is the same segment posting for posting; only where the
//! directory puts each key may differ, which no lookup can tell.
//!
//! Three passes, each spread over every core:
//!
//! 1. Each source's directory is read in memory order, a stretch per task,
//!    and its runs are dealt out to the pieces by where they start. A run's
//!    start fixes its key's place, so a piece gets exactly its keys' runs.
//! 2. Each piece sorts its runs by start, as [`Segment::source`] does for a
//!    whole segment, and merges them into a [`Writer`] of its own.
//! 3. As soon as every piece before it is merged, a piece is copied into
//!    the joined arrays and freed, by whichever thread finds it next in
//!    line; meanwhile other threads merge and copy other pieces. A piece
//!    rarely ends on a whole byte: the bytes two pieces share are put
//!    together as the pieces are placed, and every other byte has a single
//!    writer.

use alloc::vec;
use alloc::vec::Vec;
use std::sync::{Mutex, PoisonError};

use super::{
    MAX_POSTINGS, REGION_SLOTS, Segment, Slot, Writer, build_table, merge_sources, place, postings,
    slot_of, widths,
};
use crate::packed::Packed;
use crate::par::{lock, run_tasks, slots, threads};

/// Postings below which a merge runs on the calling thread: below this,
/// starting threads costs more than it saves.
const MERGE_MIN: usize = 1 << 20;

/// Directory slots below which a directory is filled on the calling
/// thread.
const TABLE_MIN: usize = 1 << 20;

/// Directory slots per task when a source's runs are dealt out.
const SCAN_SLOTS: usize = 1 << 18;

/// The joined arrays are sized to the inputs, which no merge outgrows.
const ROOM: &str = "room for every piece";

/// [`Segment::merge_all`] on every core, or `None` for a merge too small
/// to be worth it or a single core.
pub(super) fn merge_all<F: Fn(u32) -> bool + Sync>(
    sides: &[(&Segment, Option<&F>)],
) -> Option<Segment> {
    let threads = threads();
    if threads <= 1 || postings(sides) < MERGE_MIN {
        return None;
    }
    // Several pieces per thread: a slow piece holds the rest up less, and
    // the pieces merged but not yet copied, the memory beyond the output,
    // are smaller. Each holds at least 2^16 postings, so that cutting
    // costs little against merging.
    let pieces = (threads * 8).min(postings(sides) >> 16).max(1);
    Some(merge_in_pieces(sides, threads, pieces))
}

/// Where the pieces start in one source: the index of each piece's first
/// key, and the entry its run starts at (the source's length past the
/// last key).
struct Cuts {
    keys: Vec<usize>,
    starts: Vec<usize>,
}

impl Segment {
    /// The entry key `k`'s run starts at, or the length for `k` past the
    /// last key.
    fn start_of_key(&self, k: usize) -> usize {
        self.keys
            .get(k)
            .and_then(|&key| self.range(key))
            .map_or(self.len(), |(lo, _)| lo)
    }

    /// Keys below hash `bound`.
    fn keys_below(&self, bound: u64) -> usize {
        self.keys.partition_point(|&k| u64::from(k) < bound)
    }
}

/// Hash bounds that cut `sides` into at most `pieces` pieces of about
/// equal postings: piece `i` is the hashes from `bounds[i]` up to
/// `bounds[i + 1]`, the first bound 0 and the last 2<sup>32</sup>.
fn split<F>(sides: &[(&Segment, Option<&F>)], pieces: usize) -> Vec<u64> {
    const END: u64 = 1 << 32;
    let below = |bound: u64| -> usize {
        sides
            .iter()
            .map(|&(s, _)| s.start_of_key(s.keys_below(bound)))
            .sum()
    };
    let total = postings(sides);
    let mut bounds = vec![0];
    for i in 1..pieces {
        let target = (total as u128 * i as u128 / pieces as u128) as usize;
        // The smallest bound with `target` postings below it.
        let (mut lo, mut hi) = (bounds[bounds.len() - 1], END);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if below(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > bounds[bounds.len() - 1] && lo < END {
            bounds.push(lo);
        }
    }
    bounds.push(END);
    bounds
}

/// One piece's merge: its keys, the start of each key's run counted from
/// the piece's first posting, the multiplicities, and the packed
/// references and times.
type Parts = (Vec<u32>, Vec<u64>, Vec<u8>, Packed, Packed);

/// [`Segment::merge_all`] in up to `pieces` pieces on up to `threads`
/// threads, whatever the size.
pub(super) fn merge_in_pieces<F: Fn(u32) -> bool + Sync>(
    sides: &[(&Segment, Option<&F>)],
    threads: usize,
    pieces: usize,
) -> Segment {
    let plan = Plan::new(sides, pieces);
    plan.deal(threads);
    // The second and third passes: merge every piece on its own, and copy
    // each into the joined arrays as soon as every piece before it is
    // merged, which is when its place is known. A thread that copies a
    // piece frees it before it takes the next, so the pieces in flight at
    // once are about all the memory beyond the joined arrays.
    let mut out = Output::with_room(sides, plan.ref_bits, plan.time_bits);
    let merged = slots::<Parts>(plan.pieces);
    let joined = Mutex::new(out.joined());
    run_tasks(threads, plan.pieces, |piece| {
        *lock(&merged[piece]) = Some(plan.merge(piece));
        // Copy every piece that is next in line, this one or later ones
        // that finished first.
        loop {
            let next = lock(&joined).place_next(&merged);
            match next {
                Some((place, parts)) => place.fill(parts),
                None => break,
            }
        }
    });
    drop(plan);
    let joined = joined.into_inner().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(joined.next, merged.len(), "every piece is joined");
    let done = joined.finish();
    out.into_segment(done)
}

/// A merge cut into pieces: which keys and runs of every source each
/// piece takes.
struct Plan<'a, F> {
    sides: &'a [(&'a Segment, Option<&'a F>)],
    ref_bits: u32,
    time_bits: u32,
    pieces: usize,
    cuts: Vec<Cuts>,
    /// Stretches of the sources' directories, one per task of the deal:
    /// the source, and the first slot and the end.
    scans: Vec<(usize, usize, usize)>,
    /// Per stretch, its runs dealt out to the pieces.
    dealt: Vec<Mutex<Option<Vec<Vec<u64>>>>>,
}

impl<'a, F: Fn(u32) -> bool + Sync> Plan<'a, F> {
    fn new(sides: &'a [(&'a Segment, Option<&'a F>)], pieces: usize) -> Self {
        let (ref_bits, time_bits) = widths(sides);
        let bounds = split(sides, pieces.max(1));
        let cuts = sides
            .iter()
            .map(|&(s, _)| {
                let keys: Vec<usize> = bounds.iter().map(|&b| s.keys_below(b)).collect();
                let starts = keys.iter().map(|&k| s.start_of_key(k)).collect();
                Cuts { keys, starts }
            })
            .collect();
        let mut scans = Vec::new();
        for (side, &(s, _)) in sides.iter().enumerate() {
            let n = s.table.len().div_ceil(SCAN_SLOTS).max(1);
            scans.extend((0..n).map(|c| {
                let lo = s.table.len() * c / n;
                (side, lo, s.table.len() * (c + 1) / n)
            }));
        }
        let dealt = slots(scans.len());
        Self {
            sides,
            ref_bits,
            time_bits,
            pieces: bounds.len() - 1,
            cuts,
            scans,
            dealt,
        }
    }

    /// The first pass: deal every source's runs out to the pieces, a
    /// stretch of its directory per task.
    fn deal(&self, threads: usize) {
        let pieces = self.pieces;
        run_tasks(threads, self.scans.len(), |task| {
            let (side, lo, hi) = self.scans[task];
            let starts = &self.cuts[side].starts;
            let mut out: Vec<Vec<u64>> = vec![Vec::new(); pieces];
            for slot in self.sides[side].0.table[lo..hi]
                .iter()
                .filter(|s| s.len() != 0)
            {
                let start = slot.start() as usize;
                let piece = starts.partition_point(|&b| b <= start) - 1;
                out[piece.min(pieces - 1)].push(slot.order());
            }
            *lock(&self.dealt[task]) = Some(out);
        });
    }

    /// The second pass, for one piece: merge it on its own.
    fn merge(&self, piece: usize) -> Parts {
        let mut sources = Vec::with_capacity(self.sides.len());
        let (mut capacity, mut keys) = (0, 0);
        for (side, &(s, keep)) in self.sides.iter().enumerate() {
            let cut = &self.cuts[side];
            let (k0, k1) = (cut.keys[piece], cut.keys[piece + 1]);
            // The piece's runs, taken out of the deal so that their memory
            // goes as the pieces are merged.
            let mut order: Vec<u64> = Vec::new();
            for (task, &(of, _, _)) in self.scans.iter().enumerate() {
                if of == side {
                    let mut runs = lock(&self.dealt[task]);
                    let runs = runs.as_mut().expect("every source was dealt");
                    let runs = core::mem::take(&mut runs[piece]);
                    if order.is_empty() {
                        order = runs;
                    } else {
                        order.extend_from_slice(&runs);
                    }
                }
            }
            debug_assert_eq!(order.len(), k1 - k0, "a piece gets its keys' runs");
            let end = cut.starts[piece + 1];
            capacity += end - cut.starts[piece];
            keys += k1 - k0;
            sources.push(s.source_of(keep, &s.keys[k0..k1], order, end));
        }
        // `keys` counts a hash in several sources once per source, so it is
        // an upper bound on the piece's own key count, as a whole merge's
        // is.
        let mut out = Writer::with_capacity(capacity, keys, self.ref_bits, self.time_bits);
        merge_sources(&mut out, &mut sources);
        drop(sources);
        out.into_parts()
    }
}

/// The joined arrays. They are zeroed at the size of the inputs, which no
/// merge outgrows, and cut to the output's at the end: a page is only
/// taken when a piece is copied into it.
struct Output {
    keys: Vec<u32>,
    starts: Vec<u64>,
    mults: Vec<u8>,
    refs: Vec<u8>,
    times: Vec<u8>,
    ref_bits: u32,
    time_bits: u32,
}

impl Output {
    fn with_room<F>(sides: &[(&Segment, Option<&F>)], ref_bits: u32, time_bits: u32) -> Self {
        let keys: usize = sides.iter().map(|&(s, _)| s.keys.len()).sum();
        let room = postings(sides);
        Self {
            keys: vec![0; keys],
            starts: vec![0; keys + 1],
            mults: vec![0; keys],
            refs: vec![0; Packed::data_len(room, ref_bits) + Packed::SLACK],
            times: vec![0; Packed::data_len(room, time_bits) + Packed::SLACK],
            ref_bits,
            time_bits,
        }
    }

    /// The arrays to place the pieces in.
    fn joined(&mut self) -> Joined<'_> {
        let keys = self.keys.len();
        Joined {
            keys: &mut self.keys,
            starts: &mut self.starts[..keys],
            mults: &mut self.mults,
            refs: Bits::new(&mut self.refs, self.ref_bits),
            times: Bits::new(&mut self.times, self.time_bits),
            next: 0,
            postings: 0,
            key_count: 0,
        }
    }

    /// The segment, once every piece is placed.
    fn into_segment(mut self, done: Done) -> Segment {
        let Done {
            key_count,
            postings,
            ref_shared,
            time_shared,
        } = done;
        for (at, value) in ref_shared {
            self.refs[at] = value;
        }
        for (at, value) in time_shared {
            self.times[at] = value;
        }
        assert!(
            (postings as u64) < MAX_POSTINGS,
            "a segment holds fewer than 2^40 postings, not {postings}"
        );
        self.keys.truncate(key_count);
        self.keys.shrink_to_fit();
        self.starts[key_count] = postings as u64;
        let table = build_table(
            &self.keys,
            &self.starts[..=key_count],
            &self.mults[..key_count],
        );
        // Past the output the arrays were never written: their first bytes
        // there are the slack, zero.
        let cut = |mut bytes: Vec<u8>, bits| {
            bytes.truncate(Packed::data_len(postings, bits) + Packed::SLACK);
            bytes.shrink_to_fit();
            Packed::from_padded(bytes, postings, bits)
        };
        Segment {
            keys: self.keys,
            refs: cut(self.refs, self.ref_bits),
            times: cut(self.times, self.time_bits),
            table,
        }
    }
}

/// What the join leaves to do once every piece is placed: how many keys
/// and postings there are, and the shared bytes of each packed array.
struct Done {
    key_count: usize,
    postings: usize,
    ref_shared: Vec<(usize, u8)>,
    time_shared: Vec<(usize, u8)>,
}

/// The joined arrays as the pieces are placed in them, in order: what is
/// left of each to fill.
struct Joined<'a> {
    keys: &'a mut [u32],
    starts: &'a mut [u64],
    mults: &'a mut [u8],
    refs: Bits<'a>,
    times: Bits<'a>,
    /// The next piece to place.
    next: usize,
    /// Postings and keys placed so far.
    postings: u64,
    key_count: usize,
}

impl<'a> Joined<'a> {
    /// The next piece's place and parts, if it is merged.
    fn place_next(&mut self, merged: &[Mutex<Option<Parts>>]) -> Option<(Place<'a>, Parts)> {
        let parts = lock(merged.get(self.next)?).take()?;
        self.next += 1;
        let n = parts.0.len();
        let (refs, refs_from) = self.refs.place(&parts.3);
        let (times, times_from) = self.times.place(&parts.4);
        let place = Place {
            keys: self.keys.split_off_mut(..n).expect(ROOM),
            starts: self.starts.split_off_mut(..n).expect(ROOM),
            mults: self.mults.split_off_mut(..n).expect(ROOM),
            refs,
            times,
            offset: self.postings,
            refs_from,
            times_from,
        };
        self.postings += parts.3.len() as u64;
        self.key_count += n;
        Some((place, parts))
    }

    fn finish(self) -> Done {
        Done {
            key_count: self.key_count,
            postings: self.postings as usize,
            ref_shared: self.refs.finish(),
            time_shared: self.times.finish(),
        }
    }
}

/// One packed array being joined: the bytes not yet given out, and the
/// byte the last piece placed ended part-way through.
///
/// A piece's own bytes are those whose bits are all its own; it fills
/// them. A byte a piece starts or ends inside is shared with its
/// neighbours: its bits are put together here as the pieces are placed,
/// and written once every piece is.
struct Bits<'a> {
    /// The bytes after the last one given out or shared.
    rest: &'a mut [u8],
    width: u32,
    /// Bits placed so far.
    at: u64,
    /// The placed bits of byte `at / 8`, the low `at % 8` of it.
    pending: u8,
    /// Shared bytes, with their values.
    shared: Vec<(usize, u8)>,
}

impl<'a> Bits<'a> {
    fn new(bytes: &'a mut [u8], width: u32) -> Self {
        Self {
            rest: bytes,
            width,
            at: 0,
            pending: 0,
            shared: Vec::new(),
        }
    }

    /// Place `piece` after the pieces placed so far: its own bytes, and
    /// the bit of `piece` they start at.
    fn place(&mut self, piece: &Packed) -> (&'a mut [u8], u64) {
        let total = piece.len() as u64 * u64::from(self.width);
        let (start, end) = (self.at, self.at + total);
        let mut from = 0;
        if start % 8 != 0 && total > 0 {
            // The first bits finish the byte the pieces before ended in.
            let k = (8 - start % 8).min(total);
            self.pending |= (piece.bits_at(0, k as u32) << (start % 8)) as u8;
            if start % 8 + k == 8 {
                self.shared.push(((start / 8) as usize, self.pending));
                self.pending = 0;
            }
            from = k;
        }
        let first = start.div_ceil(8);
        let own = (end / 8).saturating_sub(first) as usize;
        let bytes = self.rest.split_off_mut(..own).expect(ROOM);
        if end % 8 != 0 && end / 8 >= first {
            // The piece ends inside a byte of its own: its last bits start
            // the next shared byte.
            let r = end % 8;
            self.pending = piece.bits_at(total - r, r as u32) as u8;
            self.rest.split_off_mut(..1).expect(ROOM);
        }
        self.at = end;
        (bytes, from)
    }

    /// The shared bytes, the last one included if the array ends inside
    /// it.
    fn finish(mut self) -> Vec<(usize, u8)> {
        if !self.at.is_multiple_of(8) {
            self.shared.push(((self.at / 8) as usize, self.pending));
        }
        self.shared
    }
}

/// Where one piece goes in the joined arrays.
struct Place<'a> {
    keys: &'a mut [u32],
    starts: &'a mut [u64],
    mults: &'a mut [u8],
    refs: &'a mut [u8],
    times: &'a mut [u8],
    /// The piece's first posting in the joined arrays.
    offset: u64,
    /// The bit of the piece's references and times that `refs` and
    /// `times` start at.
    refs_from: u64,
    times_from: u64,
}

impl Place<'_> {
    /// Copy `parts` here and free them.
    fn fill(self, parts: Parts) {
        let (keys, starts, mults, refs, times) = parts;
        self.keys.copy_from_slice(&keys);
        for (out, start) in self.starts.iter_mut().zip(starts) {
            *out = start + self.offset;
        }
        self.mults.copy_from_slice(&mults);
        if !self.refs.is_empty() {
            refs.copy_bits_into(self.refs, self.refs_from);
        }
        if !self.times.is_empty() {
            times.copy_bits_into(self.times, self.times_from);
        }
    }
}

/// Place `by_region`, the slots of a directory sorted by the region their
/// key's home falls in (region `r`'s from `region_starts[r]`), into
/// `table` on every core; `false`, with nothing placed, for a directory
/// too small to be worth it or a single core.
///
/// Each task fills a stretch of whole regions on its own. A key whose
/// probe runs past the end of its stretch is placed after every task is
/// done: by then the slots it ran over are still taken, so it lands where
/// a lookup's probe will find it, and a probe never passes an empty slot.
pub(super) fn place_by_region(
    table: &mut [Slot],
    by_region: &[Slot],
    region_starts: &[usize],
) -> bool {
    let threads = threads();
    if threads <= 1 || table.len() < TABLE_MIN {
        return false;
    }
    place_in_stretches(table, by_region, region_starts, threads, threads * 4);
    true
}

/// [`place_by_region`] in up to `stretches` stretches on up to `threads`
/// threads, whatever the size; returns how many keys ran past their
/// stretch.
pub(super) fn place_in_stretches(
    table: &mut [Slot],
    by_region: &[Slot],
    region_starts: &[usize],
    threads: usize,
    stretches: usize,
) -> usize {
    let slots = table.len();
    let regions = region_starts.len() - 1;
    let stretches = stretches.clamp(1, regions.max(1));
    let mut jobs = Vec::with_capacity(stretches);
    {
        let mut rest = &mut table[..];
        let mut at = 0;
        for i in 0..stretches {
            let (r0, r1) = (regions * i / stretches, regions * (i + 1) / stretches);
            let end = (r1 * REGION_SLOTS).min(slots);
            let (chunk, tail) = rest.split_at_mut(end - at);
            rest = tail;
            let items = &by_region[region_starts[r0]..region_starts[r1]];
            jobs.push(Mutex::new(Some((chunk, at, items))));
            at = end;
        }
    }
    let spilled = slots_of_vecs(stretches);
    run_tasks(threads, stretches, |i| {
        let (chunk, base, items) = lock(&jobs[i]).take().expect("each stretch is filled once");
        let mut spill = Vec::new();
        for &slot in items {
            let mut j = slot_of(slot.key, slots) - base;
            while j < chunk.len() && chunk[j].len() != 0 {
                j += 1;
            }
            match chunk.get_mut(j) {
                Some(free) => *free = slot,
                None => spill.push(slot),
            }
        }
        *lock(&spilled[i]) = spill;
    });
    drop(jobs);
    let mut late = 0;
    for spill in spilled {
        for slot in spill.into_inner().unwrap_or_else(PoisonError::into_inner) {
            place(table, slot);
            late += 1;
        }
    }
    late
}

fn slots_of_vecs<T>(n: usize) -> Vec<Mutex<Vec<T>>> {
    (0..n).map(|_| Mutex::new(Vec::new())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Posting;
    use crate::segment::{build_table_with, dense_slots, keys, sparse_slots};

    type Keep<'a> = &'a (dyn Fn(u32) -> bool + Sync);

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// Every key's run, and a lookup of hashes the segment does not hold,
    /// agree between `got` and `want`, as do the postings in order.
    fn assert_same(got: &Segment, want: &Segment, what: &str) {
        assert_eq!(got, want, "{what}: postings");
        assert_eq!(got.keys(), want.keys(), "{what}: keys");
        for &k in want.keys() {
            assert_eq!(got.range_mult(k), want.range_mult(k), "{what}: run of {k}");
        }
        for h in [1u32, 7, 1 << 20, u32::MAX - 1] {
            if want.keys().binary_search(&h).is_err() {
                assert_eq!(got.range_mult(h), None, "{what}: absent {h}");
            }
        }
    }

    fn segment(rng: &mut Rng, rows: usize, hashes: u64, refs: (u64, u64), time: u64) -> Segment {
        Segment::from_postings(
            (0..rows)
                .map(|_| {
                    Posting::new(
                        // Spread over the whole hash space, so the cuts land
                        // everywhere.
                        (rng.below(hashes) as u32).wrapping_mul(2_654_435_761),
                        (refs.0 + rng.below(refs.1 - refs.0)) as u32,
                        rng.below(time) as u32,
                    )
                })
                .collect(),
        )
    }

    /// Pieces of every count, from one to more than there are hashes, give
    /// the merge of one thread: with and without filters, with sources
    /// that share hashes and references (runs that interleave and repeat),
    /// with sources of different widths, with references of width 0, and
    /// with pieces left empty by a filter.
    #[test]
    fn pieces_agree_with_one_thread() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let a = segment(&mut rng, 4_000, 700, (0, 300), 5_000);
        let b = segment(&mut rng, 900, 700, (300, 400), 70);
        let c = segment(&mut rng, 600, 50, (250, 350), 5_000);
        let zero = segment(&mut rng, 300, 200, (0, 1), 3);
        let one = Segment::from_postings(vec![Posting::new(5, 1, 1)]);
        let empty = Segment::default();
        let keeps: [Keep<'_>; 4] = [&|_| true, &|r| r % 3 != 0, &|r| r >= 290, &|_| false];
        let sets: [&[&Segment]; 6] = [
            &[&a, &b],
            &[&a, &b, &c],
            &[&c, &a],
            &[&zero, &one],
            &[&a, &empty],
            &[&one],
        ];
        for (n, set) in sets.iter().enumerate() {
            for (k, keep) in keeps.iter().enumerate() {
                let sides: Vec<(&Segment, Option<&Keep<'_>>)> =
                    set.iter().map(|&s| (s, Some(keep))).collect();
                if n == 0 {
                    // The cuts do cut: 700 hashes over the whole hash space
                    // make sixteen pieces of about equal postings.
                    let bounds = split(&sides, 16);
                    assert_eq!(bounds.len(), 17);
                    let below = |b: u64| -> usize {
                        sides
                            .iter()
                            .map(|&(s, _)| s.start_of_key(s.keys_below(b)))
                            .sum()
                    };
                    for w in bounds.windows(2) {
                        let piece = below(w[1]) - below(w[0]);
                        assert!((200..=500).contains(&piece), "a piece of {piece} postings");
                    }
                }
                let want = {
                    let (ref_bits, time_bits) = widths(&sides);
                    let mut out =
                        Writer::with_capacity(postings(&sides), keys(&sides), ref_bits, time_bits);
                    let mut sources: Vec<_> =
                        sides.iter().map(|&(s, keep)| s.source(keep)).collect();
                    merge_sources(&mut out, &mut sources);
                    drop(sources);
                    out.finish()
                };
                for (threads, pieces) in [(1, 1), (2, 2), (3, 7), (4, 16), (2, 200), (8, 800)] {
                    let got = merge_in_pieces(&sides, threads, pieces);
                    assert_same(
                        &got,
                        &want,
                        &alloc::format!("set {n}, keep {k}, {pieces} pieces"),
                    );
                }
            }
        }
    }

    /// A directory filled in stretches finds every key where a directory
    /// filled in order does, including keys whose probe crosses from one
    /// stretch into the next and wraps past the end of the table, at the
    /// sparse load a small segment gets and at the dense one a large
    /// segment uses.
    #[test]
    fn stretches_find_every_key() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let mut late = 0;
        for keys in [10_000usize, 70_000, 200_000] {
            // Crowd a few regions so that probes cross stretches, and the
            // last region so that some wrap.
            let mut hashes: Vec<u32> = (0..keys)
                .map(|i| match i % 4 {
                    0 => rng.below(1 << 32) as u32,
                    1 => (rng.below(1 << 8) as u32) << 24,
                    2 => u32::MAX - rng.below(1 << 12) as u32,
                    _ => rng.below(1 << 16) as u32,
                })
                .collect();
            hashes.sort_unstable();
            hashes.dedup();
            let n = hashes.len();
            let offsets: Vec<u64> = (0..=n as u64).map(|i| 3 * i).collect();
            let mults: Vec<u8> = (0..n).map(|i| (i % 7) as u8 + 1).collect();
            let find = |table: &[Slot], key: u32| {
                let mut i = slot_of(key, table.len());
                loop {
                    let s = table[i];
                    if s.len() == 0 {
                        return None;
                    }
                    if s.key == key {
                        return Some((s.start(), s.len(), s.mult()));
                    }
                    i = (i + 1) % table.len();
                }
            };
            // Both loads: the sparse one this key count gets, and the dense
            // one a segment above `SPARSE_KEYS` gets, which is a shorter
            // table whose probes run further.
            for slots in [sparse_slots(n), dense_slots(n)] {
                let want = build_table_with(&hashes, &offsets, &mults, slots);
                for (threads, stretches) in [(1, 1), (2, 3), (4, 16), (3, 1_000)] {
                    let regions = slots.div_ceil(REGION_SLOTS);
                    let mut by_region: Vec<Slot> = (0..n)
                        .map(|k| Slot::new(hashes[k], offsets[k], 3, mults[k]))
                        .collect();
                    by_region.sort_by_key(|s| slot_of(s.key, slots) / REGION_SLOTS);
                    let mut region_starts = vec![0usize; regions + 1];
                    for s in &by_region {
                        region_starts[slot_of(s.key, slots) / REGION_SLOTS + 1] += 1;
                    }
                    for r in 0..regions {
                        region_starts[r + 1] += region_starts[r];
                    }
                    let mut table = vec![Slot::EMPTY; slots];
                    late += place_in_stretches(
                        &mut table,
                        &by_region,
                        &region_starts,
                        threads,
                        stretches,
                    );
                    for &h in &hashes {
                        assert_eq!(
                            find(&table, h),
                            find(&want, h),
                            "{keys} keys, {slots} slots, {stretches} stretches, {h}"
                        );
                    }
                    assert_eq!(table.iter().filter(|s| s.len() != 0).count(), n);
                }
            }
        }
        // The crowded regions did send probes across stretches.
        assert!(late > 0);
    }
}
