use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;

use crate::Posting;
use crate::packed::{Packed, PackedBuilder, View};
use crate::refs::{self, Layout, RefCoding, Refs};

#[cfg(feature = "std")]
mod parallel;
// The merge that writes into a file is what a catalog directory spills
// with; no other build reaches it.
#[cfg(feature = "mmap")]
pub(crate) mod to_file;

/// One directory slot, as a lookup reads it: `key`'s run starts at `start` (40 bits),
/// holds `len` postings (16 bits), no reference has more than `mult`
/// postings in it (8 bits, saturating), and its coded references start
/// `refs` bits after the first run of its page (see [`PAGE_BITS`]).
///
/// `len == 0` marks an empty slot (a present key always has a posting), and
/// `len == LONG` sends the lookup to the sorted keys for the run's end, so
/// runs that long still resolve. `mult` lets the search skip the fanout
/// test on a run where no reference can reach the fanout.
#[derive(Clone, Copy, Debug)]
struct Slot {
    key: u32,
    start_lo: u32,
    /// `len << 16 | mult << 8 | start >> 32`.
    len_mult_hi: u32,
    /// Where the run's references start, in bits from the start of the
    /// first run of the page its first posting is in.
    refs: u32,
}

/// The postings of a page are `2^PAGE_BITS`: the segment records where
/// the coded references of the first run of each page start, and a slot
/// where its run's start counting from there. A page's runs hold fewer
/// than `2^PAGE_BITS` postings in fewer runs, at most
/// [`refs::max_run_bits`] each, so that offset fits 32 bits with room.
const PAGE_BITS: u32 = 20;

/// Run length a slot cannot hold exactly.
const LONG: u32 = (1 << 16) - 1;

impl Slot {
    const EMPTY: Slot = Slot {
        key: 0,
        start_lo: 0,
        len_mult_hi: 0,
        refs: 0,
    };

    #[inline]
    fn new(key: u32, start: u64, len: u64, mult: u8, refs: u32) -> Self {
        let len = u32::try_from(len).map_or(LONG, |l| l.min(LONG));
        Self {
            key,
            start_lo: start as u32,
            len_mult_hi: (len << 16) | (u32::from(mult) << 8) | ((start >> 32) as u32 & 0xFF),
            refs,
        }
    }

    #[inline]
    const fn len(self) -> u32 {
        self.len_mult_hi >> 16
    }

    /// Most postings any one reference has in the run, saturating at 255.
    #[inline]
    const fn mult(self) -> u8 {
        (self.len_mult_hi >> 8) as u8
    }

    #[inline]
    const fn start(self) -> u64 {
        self.start_lo as u64 | ((self.len_mult_hi & 0xFF) as u64) << 32
    }

    /// `start << 8 | mult`: a run as a merge walks it, ordered by where
    /// it starts.
    #[inline]
    const fn order(&self) -> u64 {
        self.start() << 8 | self.mult() as u64
    }
}

/// A directory slot as a segment stores it: whole, 16 bytes, for
/// references coded as gaps, or [`Short`], 12, for packed references,
/// where a run's references start at its first posting times the width.
trait Stored: Copy + Send + Sync {
    const EMPTY: Self;
    /// Whether the slot keeps [`Slot::refs`].
    const REFS: bool;
    fn of(slot: Slot) -> Self;
    fn slot(self) -> Slot;
    fn key(self) -> u32;
    /// The run's length, 0 in an empty slot.
    fn len(self) -> u32;
}

impl Stored for Slot {
    const EMPTY: Self = Slot::EMPTY;
    const REFS: bool = true;

    #[inline]
    fn of(slot: Slot) -> Self {
        slot
    }

    #[inline]
    fn slot(self) -> Slot {
        self
    }

    #[inline]
    fn key(self) -> u32 {
        self.key
    }

    #[inline]
    fn len(self) -> u32 {
        Slot::len(self)
    }
}

/// A [`Slot`] without its reference offset: 12 bytes.
#[derive(Clone, Copy, Debug)]
struct Short {
    key: u32,
    start_lo: u32,
    len_mult_hi: u32,
}

impl Stored for Short {
    const EMPTY: Self = Short {
        key: 0,
        start_lo: 0,
        len_mult_hi: 0,
    };
    const REFS: bool = false;

    #[inline]
    fn of(slot: Slot) -> Self {
        Short {
            key: slot.key,
            start_lo: slot.start_lo,
            len_mult_hi: slot.len_mult_hi,
        }
    }

    #[inline]
    fn slot(self) -> Slot {
        Slot {
            key: self.key,
            start_lo: self.start_lo,
            len_mult_hi: self.len_mult_hi,
            refs: 0,
        }
    }

    #[inline]
    fn key(self) -> u32 {
        self.key
    }

    #[inline]
    fn len(self) -> u32 {
        self.len_mult_hi >> 16
    }
}

/// A segment's directory: short slots over packed references, whole ones
/// over references coded as gaps.
#[derive(Clone)]
enum Table {
    Short(Vec<Short>),
    Whole(Vec<Slot>),
}

impl Table {
    /// The directory over `index` in a table of `slots`, for references
    /// held as `layout` says (`pages` as [`pages_of`] gives them for
    /// gaps; packed references need none).
    fn build(index: Index<'_>, pages: &[u64], slots: usize, layout: Layout) -> Self {
        match layout {
            Layout::Packed(_) => Self::Short(build_table_with(index, pages, slots)),
            Layout::Gaps => Self::Whole(build_table_with(index, pages, slots)),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Short(t) => t.len(),
            Self::Whole(t) => t.len(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Short(t) => t.capacity() * size_of::<Short>(),
            Self::Whole(t) => t.capacity() * size_of::<Slot>(),
        }
    }

    /// The slot of `hash`, if the table holds the hash.
    #[inline]
    fn find(&self, hash: u32) -> Option<Slot> {
        match self {
            Self::Short(t) => find_in(t, hash),
            Self::Whole(t) => find_in(t, hash),
        }
    }

    /// Load the home slots of `keys` and discard them.
    #[inline]
    fn touch(&self, keys: &[u32]) {
        match self {
            Self::Short(t) => touch_slots(t, keys),
            Self::Whole(t) => touch_slots(t, keys),
        }
    }

    /// Every taken slot of `lo..hi`, in table order.
    #[inline]
    fn for_each_in(&self, lo: usize, hi: usize, mut f: impl FnMut(Slot)) {
        fn each<T: Stored>(t: &[T], f: &mut impl FnMut(Slot)) {
            for &slot in t.iter().filter(|s| s.len() != 0) {
                f(slot.slot());
            }
        }
        match self {
            Self::Short(t) => each(&t[lo..hi], &mut f),
            Self::Whole(t) => each(&t[lo..hi], &mut f),
        }
    }
}

/// The slot of `hash` in `table`, if the table holds the hash.
#[inline]
fn find_in<T: Stored>(table: &[T], hash: u32) -> Option<Slot> {
    if table.is_empty() {
        return None;
    }
    let (before, from) = table.split_at(slot_of(hash, table.len()));
    for &slot in from.iter().chain(before) {
        if slot.len() == 0 {
            return None;
        }
        if slot.key() == hash {
            return Some(slot.slot());
        }
    }
    // The load bound leaves empty slots, so the probe above always ends.
    None
}

/// Keys per batch when the directory is built or walked in key order. A
/// key's slot is a cache miss of its own in a directory larger than the
/// cache, so the slots of the batch after the current one are loaded first
/// and the misses overlap instead of queueing: the same arrangement, and
/// the same size, as the lookups of a search.
const AHEAD: usize = 32;

/// Load the home slots of `keys` in `table` and discard them.
#[inline]
fn touch_slots<T: Stored>(table: &[T], keys: &[u32]) {
    let slots = table.len();
    if slots == 0 {
        return;
    }
    let acc = keys
        .iter()
        .fold(0u32, |acc, &key| acc ^ table[slot_of(key, slots)].len());
    core::hint::black_box(acc);
}

/// Directory slot of `key` in a table of `slots` entries.
///
/// Keys are already hashes, but may vary in few bits; the multiply spreads
/// every input bit into the high product bits that pick the slot, and the
/// range reduction needs no power-of-two table.
#[inline]
fn slot_of(key: u32, slots: usize) -> usize {
    let h = key.wrapping_mul(0x9E37_79B1);
    ((u128::from(h) * slots as u128) >> 32) as usize
}

/// An immutable set of postings sorted by `(hash, reference, time)`.
///
/// Every hash owns one contiguous run; inside a run each reference's
/// postings sit together with times ascending. The segment holds its
/// sorted distinct `keys`, the `refs` of every run (packed at the width
/// of the largest, or coded as the gaps between neighbours: see
/// [`RefCoding`]), the `times`
/// packed to the width of the largest, and an open-addressing directory
/// from hash to run. A lookup costs one directory read: a hash's slot depends
/// only on the hash, so the lookups of a query are independent and overlap
/// in the memory system. Iteration in key order looks each key up the same
/// way, so no offsets array is kept. The directory is rebuilt from the
/// keys whenever a segment is loaded and is never stored.
///
/// Exact duplicate postings are dropped when a segment is built. A segment
/// holds fewer than 2<sup>40</sup> postings.
#[derive(Clone)]
pub struct Segment {
    keys: Vec<u32>,
    refs: Refs,
    times: Packed,
    /// Linear-probing directory: at most three keys per five slots, or one
    /// per four below [`SPARSE_KEYS`] keys (see [`build_table`]).
    table: Table,
    /// Per page of `2^PAGE_BITS` postings, the bit the coded references
    /// of its first run start at (see [`Slot`]); empty for packed
    /// references, which need none.
    pages: Vec<u64>,
    /// The smallest and the largest reference, if there is a posting.
    ids: Option<(u32, u32)>,
    /// Bits of coded references per posting, rounded up: how far a
    /// prefetch of a run reaches.
    density: u32,
}

/// Postings a segment can hold: the reach of a directory slot.
pub(crate) const MAX_POSTINGS: u64 = 1 << 40;

/// One hash's postings inside a segment, grouped by reference with times
/// ascending.
#[derive(Clone, Copy)]
pub struct Run<'a> {
    segment: &'a Segment,
    lo: usize,
    hi: usize,
    /// Where the run's coded references start.
    at: u64,
}

impl<'a> Run<'a> {
    /// Number of postings in the run.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.hi - self.lo
    }

    /// `true` if the run holds no postings (never the case for a run a
    /// segment returns).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hi == self.lo
    }

    /// The postings as `(reference, time)`, grouped by reference with
    /// times ascending.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, u32)> + 'a {
        let times = self.segment.times.view();
        let refs = self.segment.refs.run(self.at, self.lo, self.len()).cursor();
        refs.zip(self.lo..self.hi)
            .map(move |(reference, i)| (reference, times.get(i)))
    }
}

impl PartialEq for Run<'_> {
    /// Same postings.
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for Run<'_> {}

impl fmt::Debug for Run<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Directories up to this many slots are filled in key order: they fit
/// the cache, so it does not matter that consecutive keys land far apart.
const DIRECT_SLOTS: usize = 1 << 16;
/// Slots per region when a larger directory is filled region by region.
const REGION_SLOTS: usize = 1 << 13;

/// Keys below which a directory gets four slots per key.
///
/// A miss walks to an empty slot, and a delta segment is mostly misses: a
/// search looks every query hash up in every segment, and most of them are
/// not in the small ones. Four slots per key make the table sparser, so a
/// miss finds an empty slot sooner; the price is 64 bytes per key instead
/// of 27, at most about 67 MB per segment.
const SPARSE_KEYS: usize = 1 << 20;

/// Slots in a directory over `keys` keys at the dense load of a large
/// segment: three keys per five slots.
fn dense_slots(keys: usize) -> usize {
    keys + keys / 3 * 2 + 1
}

/// Slots in a directory over `keys` keys at the sparse load of a small
/// segment: four per key.
fn sparse_slots(keys: usize) -> usize {
    4 * keys + 1
}

/// Slots in a directory over `keys` keys: the sparse load below
/// [`SPARSE_KEYS`], the dense one from there.
///
/// A lookup that misses probes on to an empty slot, and that walk is most
/// of what a lookup in a delta segment costs: at the denser load it is
/// five slots, and at the sparser one under two. A hit costs one or two
/// probes either way, and both stay inside one cache line.
fn slots_for(keys: usize) -> usize {
    if keys < SPARSE_KEYS {
        sparse_slots(keys)
    } else {
        dense_slots(keys)
    }
}

/// What a directory is built from: the sorted keys, where each key's run
/// starts (and the posting count after the last), where its coded
/// references start (and the bit count after the last), and its
/// multiplicity.
#[derive(Clone, Copy)]
pub(crate) struct Index<'a> {
    pub(crate) keys: &'a [u32],
    pub(crate) offsets: &'a [u64],
    pub(crate) ref_starts: &'a [u64],
    pub(crate) mults: &'a [u8],
}

/// Per page of `2^PAGE_BITS` postings, the bit the references of the
/// first run starting in it start at (the bit count where no run starts
/// in it, as past the last run).
fn pages_of(index: Index<'_>) -> Vec<u64> {
    let n = index.keys.len();
    let postings = index.offsets[n];
    let pages = (postings >> PAGE_BITS) as usize + 1;
    let mut out = Vec::with_capacity(pages);
    let mut k = 0;
    for page in 0..pages as u64 {
        while k < n && index.offsets[k] < page << PAGE_BITS {
            k += 1;
        }
        out.push(index.ref_starts[k]);
    }
    out
}

/// Directory over `index`, in a table of `slots`: [`slots_for`] at the
/// load a segment's size picks.
fn build_table_with<T: Stored>(index: Index<'_>, pages: &[u64], slots: usize) -> Vec<T> {
    let Index {
        keys,
        offsets,
        ref_starts,
        mults,
    } = index;
    debug_assert_eq!(mults.len(), keys.len());
    if keys.is_empty() {
        return Vec::new();
    }
    debug_assert!(slots > keys.len(), "a key needs a free slot");
    let n = keys.len();
    let mut table = alloc::vec![T::EMPTY; slots];
    // A key without postings (only arrays a reader has yet to reject) gets
    // no slot, which reads as "absent".
    let slot_of_key = |k: usize| {
        let (start, end) = (offsets[k], offsets[k + 1]);
        (end > start).then(|| {
            let refs = if T::REFS {
                let refs = ref_starts[k] - pages[(start >> PAGE_BITS) as usize];
                u32::try_from(refs).expect("a page's references span less than 2^32 bits")
            } else {
                0
            };
            T::of(Slot::new(keys[k], start, end - start, mults[k], refs))
        })
    };
    if slots <= DIRECT_SLOTS {
        for slot in (0..n).filter_map(slot_of_key) {
            place(&mut table, slot);
        }
        return table;
    }
    // Beyond the cache, placing keys in key order is a miss per key. Sort
    // the slots by the region of the table they land in (a counting sort:
    // two sequential passes over 12 bytes per key), then place them region
    // by region, each region cached while it fills.
    let region_of = |key: u32| slot_of(key, slots) / REGION_SLOTS;
    let regions = slots.div_ceil(REGION_SLOTS);
    let mut next = alloc::vec![0usize; regions + 1];
    for slot in (0..n).filter_map(slot_of_key) {
        next[region_of(slot.key()) + 1] += 1;
    }
    for region in 0..regions {
        next[region + 1] += next[region];
    }
    let mut by_region = alloc::vec![T::EMPTY; next[regions]];
    #[cfg(feature = "std")]
    let region_starts = next.clone();
    for slot in (0..n).filter_map(slot_of_key) {
        let at = &mut next[region_of(slot.key())];
        by_region[*at] = slot;
        *at += 1;
    }
    #[cfg(feature = "std")]
    if parallel::place_by_region(&mut table, &by_region, &region_starts) {
        return table;
    }
    for slot in by_region {
        place(&mut table, slot);
    }
    table
}

/// Directory over `index` in the table size [`slots_for`] gives its
/// keys, for references held as `layout` says, and the start of each
/// page's references the slots count from (none for packed references).
fn build_table(index: Index<'_>, layout: Layout) -> (Table, Vec<u64>) {
    let pages = match layout {
        Layout::Packed(_) => Vec::new(),
        Layout::Gaps => pages_of(index),
    };
    let slots = slots_for(index.keys.len());
    (Table::build(index, &pages, slots, layout), pages)
}

/// Put `slot` in the first free slot from its key's home.
#[inline]
fn place<T: Stored>(table: &mut [T], slot: T) {
    let mut i = slot_of(slot.key(), table.len());
    while table[i].len() != 0 {
        i += 1;
        if i == table.len() {
            i = 0;
        }
    }
    table[i] = slot;
}

/// Appends runs in hash order and finishes into a [`Segment`].
pub(crate) struct Writer {
    keys: Vec<u32>,
    /// Where each key's run starts.
    starts: Vec<u64>,
    /// Where each key's coded references start.
    ref_starts: Vec<u64>,
    /// Per key, the most postings any one reference has in its run.
    mults: Vec<u8>,
    /// The references, packed or coded as gaps as `layout` says (see
    /// [`crate::refs`]).
    refs: PackedBuilder,
    layout: Layout,
    times: PackedBuilder,
    /// The smallest and the largest reference coded here. A run copied
    /// as it is adds nothing: a merge adds its source's whole range
    /// instead ([`Writer::include`]).
    ids: Option<(u32, u32)>,
    /// The references of the run being written, kept between runs for
    /// its capacity: a run is coded once it is whole.
    run: Vec<u32>,
    /// The union of runs that interleave, kept between runs for its
    /// capacity.
    union: Vec<(u32, u32)>,
}

/// The range that covers both ranges, if either has one.
pub(crate) fn union_ids(a: Option<(u32, u32)>, b: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (a, b) {
        (Some((lo, hi)), Some((x, y))) => Some((lo.min(x), hi.max(y))),
        (a, b) => a.or(b),
    }
}

impl Writer {
    /// Room for `postings` rows whose references are laid out as `layout`
    /// says, in about `ref_bits` bits when they are coded as gaps, and
    /// whose times fit `time_bits`, and for `keys` distinct hashes. A merge passes the sum of its sources' key counts,
    /// an upper bound that keeps `keys`, `starts` and `mults` from
    /// growing by doubling as the runs arrive; [`Writer::finish`] cuts
    /// what the merge did not need.
    pub(crate) fn with_capacity(
        postings: usize,
        keys: usize,
        ref_bits: u64,
        time_bits: u32,
        layout: Layout,
    ) -> Self {
        let refs = match layout {
            Layout::Packed(width) => PackedBuilder::with_capacity(postings, width),
            Layout::Gaps => PackedBuilder::with_capacity(usize::try_from(ref_bits).unwrap_or(0), 1),
        };
        Self {
            keys: Vec::with_capacity(keys),
            // One past the keys: `finish` records where the last run ends.
            starts: Vec::with_capacity(keys.saturating_add(1)),
            ref_starts: Vec::with_capacity(keys.saturating_add(1)),
            mults: Vec::with_capacity(keys),
            refs,
            layout,
            times: PackedBuilder::with_capacity(postings, time_bits),
            ids: None,
            run: Vec::new(),
            union: Vec::new(),
        }
    }

    /// Postings appended so far.
    #[inline]
    fn postings(&self) -> usize {
        self.times.len()
    }

    /// Bits of references appended so far.
    #[inline]
    fn ref_len(&self) -> u64 {
        self.refs.len() as u64 * u64::from(self.layout.array_width())
    }

    /// Append the run of `hash`, which must be greater than the hash of
    /// every run before it: `(reference, time)` pairs in ascending order.
    /// A run without pairs leaves no trace.
    pub(crate) fn push_run(&mut self, hash: u32, pairs: impl Iterator<Item = (u32, u32)> + Clone) {
        let (start, at) = (self.postings(), self.ref_len());
        self.run.clear();
        let mut streaks = Streaks::default();
        // References and times in one loop each, which keeps either
        // loop's state in registers.
        self.run.extend(pairs.clone().map(|(reference, _)| {
            streaks.see(reference);
            reference
        }));
        self.times.extend(pairs.map(|(_, time)| time));
        self.code_run();
        self.close(hash, start, at, streaks.longest);
    }

    /// Code the references gathered for the run being written, if any.
    fn code_run(&mut self) {
        let (Some(&first), Some(&last)) = (self.run.first(), self.run.last()) else {
            return;
        };
        match self.layout {
            Layout::Packed(_) => self.refs.extend(self.run.iter().copied()),
            Layout::Gaps => refs::push_run(&mut self.refs, &self.run),
        }
        self.include(Some((first, last)));
    }

    /// Count `ids` in the range the segment reports: what a merge adds for
    /// a source whose runs it copied without a look at their references.
    pub(crate) fn include(&mut self, ids: Option<(u32, u32)>) {
        self.ids = union_ids(self.ids, ids);
    }

    /// Record the run of `hash` that began `start` postings and `at` bits
    /// of references into the segment and has multiplicity `mult`, if it
    /// got a posting.
    #[inline]
    fn close(&mut self, hash: u32, start: usize, at: u64, mult: u32) {
        if self.postings() > start {
            debug_assert!(
                self.keys.last().is_none_or(|&last| last < hash),
                "runs out of order"
            );
            self.keys.push(hash);
            self.starts.push(start as u64);
            self.ref_starts.push(at);
            self.mults.push(u8::try_from(mult).unwrap_or(u8::MAX));
        }
    }

    /// Copy the references of `run` of `source`, whose layout this
    /// writer's [copies](Layout::copies): packed values from width to
    /// width, gaps as bits.
    #[inline]
    fn copy_refs<F>(&mut self, source: &Source<'_, '_, F>, run: Span) {
        debug_assert!(
            self.layout.copies(source.refs.layout()),
            "a run to code again"
        );
        let view = source.refs.bits().view();
        match self.layout {
            Layout::Packed(_) => self.refs.extend_from(view, run.lo, run.hi),
            Layout::Gaps => self
                .refs
                .extend_from(view, run.at as usize, run.end as usize),
        }
    }

    /// Gather the postings of `run` that its source's `keep` accepts
    /// (every posting without a `keep`) into the run being written: the
    /// references into [`Writer::run`] and the times onto the times,
    /// copied as bits in the stretches between rejected references.
    /// Returns their multiplicity and whether every posting was kept.
    fn gather<F: Fn(u32) -> bool>(&mut self, source: &Source<'_, '_, F>, run: Span) -> (u32, bool) {
        let mut streaks = Streaks::default();
        let (mut from, mut i, mut whole) = (run.lo, run.lo, true);
        for reference in source.refs.run(run.at, run.lo, run.hi - run.lo).cursor() {
            if source.keep.is_none_or(|keep| keep(reference)) {
                streaks.see(reference);
                self.run.push(reference);
            } else {
                self.times.extend_from(source.times, from, i);
                from = i + 1;
                whole = false;
            }
            i += 1;
        }
        self.times.extend_from(source.times, from, run.hi);
        (streaks.longest, whole)
    }

    /// Append the run of `hash` as the union of `runs`, each with the
    /// index of its source (a posting in several of them once), filtered
    /// by its source's `keep` if it has one.
    ///
    /// A run that is the only one of its hash, from a source of a layout
    /// this writer copies, is copied without a look at its references when
    /// its source has no `keep`, and also when a `keep` rejects none of
    /// them. Otherwise the references are decoded and the run is coded
    /// again.
    fn merge_runs<F: Fn(u32) -> bool>(
        &mut self,
        hash: u32,
        sources: &[Source<'_, '_, F>],
        runs: &mut [(usize, Span)],
    ) {
        let (start, at) = (self.postings(), self.ref_len());
        self.run.clear();
        let mult = if let &mut [(source, run)] = runs {
            let source = &sources[source];
            let copies = self.layout.copies(source.refs.layout());
            if source.keep.is_none() && copies {
                self.copy_refs(source, run);
                self.times.extend_from(source.times, run.lo, run.hi);
                u32::from(run.mult)
            } else {
                let (mult, whole) = self.gather(source, run);
                if whole && copies {
                    self.copy_refs(source, run);
                    let ids = Some((self.run[0], self.run[self.run.len() - 1]));
                    self.include(ids);
                } else {
                    self.code_run();
                }
                mult
            }
        } else {
            let mult = self.append_union(sources, runs);
            self.code_run();
            mult
        };
        self.close(hash, start, at, mult);
    }

    /// Gather the union of `runs` into the run being written and return
    /// its multiplicity.
    fn append_union<F: Fn(u32) -> bool>(
        &mut self,
        sources: &[Source<'_, '_, F>],
        runs: &mut [(usize, Span)],
    ) -> u32 {
        let mut union = core::mem::take(&mut self.union);
        union.clear();
        // In a catalog a reference lives in one segment, so the runs of a
        // hash rarely interleave: ordered by their first reference, they
        // are mostly the union already.
        runs.sort_unstable_by_key(|&(source, run)| {
            sources[source]
                .refs
                .run(run.at, run.lo, run.hi - run.lo)
                .first()
        });
        for &(source, run) in &*runs {
            let source = &sources[source];
            let times = source.times;
            let pairs = source
                .refs
                .run(run.at, run.lo, run.hi - run.lo)
                .cursor()
                .zip(run.lo..run.hi)
                .map(|(reference, i)| (reference, times.get(i)));
            // A source without a `keep` holds nothing another's rejects.
            match source.keep {
                Some(keep) => union.extend(pairs.filter(|&(reference, _)| keep(reference))),
                None => union.extend(pairs),
            }
        }
        if !union.windows(2).all(|w| w[0] < w[1]) {
            union.sort_unstable();
            union.dedup();
        }
        let mut streaks = Streaks::default();
        self.run.extend(union.iter().map(|&(reference, _)| {
            streaks.see(reference);
            reference
        }));
        self.times.extend(union.iter().map(|&(_, time)| time));
        self.union = union;
        streaks.longest
    }

    /// The keys, the start of each key's run and of its references, the
    /// multiplicities and the finished arrays, for a caller that joins
    /// several writers' output.
    #[cfg(feature = "std")]
    fn into_parts(self) -> Parts {
        Parts {
            keys: self.keys,
            starts: self.starts,
            ref_starts: self.ref_starts,
            mults: self.mults,
            refs: self.refs.finish(),
            times: self.times.finish(),
            ids: self.ids,
        }
    }

    pub(crate) fn finish(mut self) -> Segment {
        let postings = self.postings() as u64;
        assert!(
            postings < MAX_POSTINGS,
            "a segment holds fewer than 2^40 postings, not {postings}"
        );
        self.starts.push(postings);
        self.ref_starts.push(self.ref_len());
        let (table, pages) = build_table(
            Index {
                keys: &self.keys,
                offsets: &self.starts,
                ref_starts: &self.ref_starts,
                mults: &self.mults,
            },
            self.layout,
        );
        let mut keys = self.keys;
        keys.shrink_to_fit();
        Segment::assemble(
            keys,
            Refs::new(self.refs.finish(), self.layout),
            self.times.finish(),
            (table, pages),
            self.ids,
        )
    }
}

/// One piece of a merge: its keys, the start of each key's run counted
/// from the piece's first posting and of its references counted from the
/// piece's first bit, the multiplicities, the references (the array the
/// merge's layout holds them in), the packed times, and the range of the
/// references it coded.
#[cfg(feature = "std")]
pub(crate) struct Parts {
    pub(crate) keys: Vec<u32>,
    pub(crate) starts: Vec<u64>,
    pub(crate) ref_starts: Vec<u64>,
    pub(crate) mults: Vec<u8>,
    pub(crate) refs: Packed,
    pub(crate) times: Packed,
    pub(crate) ids: Option<(u32, u32)>,
}

/// The longest streak of one reference in a run's references, which
/// arrive in ascending order: the run's multiplicity.
#[derive(Default)]
struct Streaks {
    last: u32,
    current: u32,
    longest: u32,
}

impl Streaks {
    #[inline]
    fn see(&mut self, reference: u32) {
        // Before the first reference the streak is zero, so whatever
        // `last` holds the first one starts a streak of one.
        self.current = if reference == self.last {
            self.current.saturating_add(1)
        } else {
            1
        };
        self.last = reference;
        self.longest = self.longest.max(self.current);
    }
}

/// One run of a source: entries `lo..hi` of its times, and bits
/// `at..end` of its coded references.
#[derive(Clone, Copy)]
struct Span {
    lo: usize,
    hi: usize,
    /// Most postings any one reference has in the run (saturating).
    mult: u8,
    at: u64,
    end: u64,
}

/// A segment being merged: its arrays, its filter, and a walk over its
/// runs in key order (see [`Segment::source`]).
struct Source<'a, 'k, F> {
    refs: &'a Refs,
    times: View<'a>,
    /// Which references to keep; `None` keeps every posting unread.
    keep: Option<&'k F>,
    keys: &'a [u32],
    /// Per key, `start << 8 | multiplicity` of its run and where its
    /// coded references start.
    order: Vec<(u64, u64)>,
    /// Where the last run ends: the posting count.
    end: usize,
    /// Where the last run's coded references end.
    end_bit: u64,
    /// Index of the next key.
    next: usize,
}

impl<F> Source<'_, '_, F> {
    /// Stands in for the hash of a source with no run left; above every
    /// hash.
    const DONE: u64 = u64::MAX;

    /// The hash of the next run, or [`Source::DONE`].
    #[inline]
    fn head(&self) -> u64 {
        self.keys
            .get(self.next)
            .map_or(Self::DONE, |&key| u64::from(key))
    }

    /// Take the next run; there must be one.
    #[inline]
    fn take(&mut self) -> Span {
        let k = self.next;
        self.next += 1;
        let (run, at) = self.order[k];
        // Runs are laid out in key order: one ends where the next begins.
        let (hi, end) = self
            .order
            .get(k + 1)
            .map_or((self.end, self.end_bit), |&(next, next_at)| {
                ((next >> 8) as usize, next_at)
            });
        Span {
            lo: (run >> 8) as usize,
            hi,
            mult: run as u8,
            at,
            end,
        }
    }
}

impl Segment {
    /// Load the directory slots of `hashes` and discard them, so their
    /// cache misses overlap instead of waiting one lookup at a time.
    #[inline]
    fn touch_directory(&self, hashes: &[u32]) {
        self.table.touch(hashes);
    }

    /// Locate each of `hashes` in turn, calling `f` with its index and
    /// its run for those the segment holds. The slots of each batch of
    /// `batch` hashes are loaded while the batch before it resolves, so the
    /// lookups' cache misses overlap; the loop is compiled for the kind of
    /// slot the directory holds, which a lookup then does not test.
    #[inline]
    pub(crate) fn locate_each(
        &self,
        hashes: &[u32],
        batch: usize,
        mut f: impl FnMut(usize, Located),
    ) {
        match &self.table {
            Table::Short(t) => self.locate_in(t, hashes, batch, &mut f),
            Table::Whole(t) => self.locate_in(t, hashes, batch, &mut f),
        }
    }

    #[inline(always)]
    fn locate_in<T: Stored>(
        &self,
        table: &[T],
        hashes: &[u32],
        batch: usize,
        f: &mut impl FnMut(usize, Located),
    ) {
        let width = u64::from(self.refs.layout().array_width());
        let n = hashes.len();
        touch_slots(table, &hashes[..batch.min(n)]);
        for k in (0..n).step_by(batch) {
            let end = (k + batch).min(n);
            touch_slots(table, &hashes[end..(end + batch).min(n)]);
            for (i, &hash) in hashes[k..end].iter().enumerate() {
                let Some(slot) = find_in(table, hash) else {
                    continue;
                };
                let at = if T::REFS {
                    self.pages[(slot.start() >> PAGE_BITS) as usize] + u64::from(slot.refs)
                } else {
                    slot.start() * width
                };
                f(k + i, self.located(&slot, hash, at));
            }
        }
    }

    /// Start a walk of the directory in key order: load the slots of the
    /// first batch of keys. [`Segment::touch_ahead`] keeps one batch ahead
    /// from there.
    #[inline]
    fn touch_first(&self) {
        self.touch_directory(&self.keys[..AHEAD.min(self.keys.len())]);
    }

    /// Call before key `k` is looked up in a walk in key order: at the
    /// first key of a batch, the slots of the batch after it are loaded.
    #[inline]
    fn touch_ahead(&self, k: usize) {
        if k.is_multiple_of(AHEAD) {
            let n = self.keys.len();
            let from = (k + AHEAD).min(n);
            self.touch_directory(&self.keys[from..(from + AHEAD).min(n)]);
        }
    }

    /// This segment as a source for a merge, filtered by `keep` if given:
    /// a walk over every hash and its run, hashes ascending.
    ///
    /// The walk never looks a key up. A lookup per key is a cache miss per
    /// key once the directory outgrows the cache, which is most of the
    /// cost of merging segments with few postings per hash. Instead the
    /// directory is read once in memory order and its runs are sorted by
    /// where they start, which is key order because runs are laid out in
    /// key order: sequential passes only, over 24 bytes per key that are
    /// released when the merge ends.
    fn source<'k, F>(&self, keep: Option<&'k F>) -> Source<'_, 'k, F> {
        let mut order: Vec<(u64, u64)> = Vec::with_capacity(self.keys.len());
        self.table.for_each_in(0, self.table.len(), |slot| {
            order.push(self.walk_entry(&slot));
        });
        debug_assert_eq!(order.len(), self.keys.len(), "every key has a run");
        self.source_of(keep, &self.keys, order, (self.len(), self.refs.bit_len()))
    }

    /// What a merge's walk keeps of a slot: [`Slot::order`], and where its
    /// run's coded references start.
    #[inline]
    fn walk_entry(&self, slot: &Slot) -> (u64, u64) {
        (slot.order(), self.ref_at(slot))
    }

    /// Where the coded references of `slot`'s run start.
    #[inline]
    fn ref_at(&self, slot: &Slot) -> u64 {
        match self.refs.layout() {
            Layout::Packed(width) => slot.start() * u64::from(width),
            Layout::Gaps => self.pages[(slot.start() >> PAGE_BITS) as usize] + u64::from(slot.refs),
        }
    }

    /// A walk over `keys`, a stretch of this segment's keys whose runs
    /// end at entry `end.0` and bit `end.1` of the references, from their
    /// runs as [`Segment::walk_entry`] gives them in any order.
    fn source_of<'a, 'k, F>(
        &'a self,
        keep: Option<&'k F>,
        keys: &'a [u32],
        mut order: Vec<(u64, u64)>,
        (end, end_bit): (usize, u64),
    ) -> Source<'a, 'k, F> {
        let start_bits = u64::BITS - (self.len() as u64).leading_zeros();
        crate::sort::radix_sort_by_key(&mut order, &mut Vec::new(), start_bits, |&(run, _)| {
            run >> 8
        });
        Source {
            refs: &self.refs,
            times: self.times.view(),
            keep,
            // Every key has a run; the shorter of the two bounds the walk
            // all the same.
            keys: &keys[..keys.len().min(order.len())],
            order,
            end,
            end_bit,
            next: 0,
        }
    }

    /// Load every cache line of the coded references of a run of `n`
    /// postings from bit `at`, as far as the segment's density says it
    /// reaches, and those up to `reach` bytes past it (the caller discards
    /// the value the same way; see [`Packed::touch`]).
    #[inline]
    pub(crate) fn touch_refs(&self, at: u64, n: usize, reach: usize) -> u32 {
        debug_assert!(n > 0, "a run is never empty");
        self.refs
            .touch(at, n as u64 * u64::from(self.density), reach)
    }

    /// The references of the run of postings `lo..lo + n`, whose stream
    /// starts at bit `at`.
    #[inline]
    pub(crate) fn run_refs(&self, at: u64, lo: usize, n: usize) -> refs::Run<'_> {
        self.refs.run(at, lo, n)
    }

    /// Load the time at entry `i` (the caller discards it the same way).
    #[inline]
    pub(crate) fn touch_time(&self, i: usize) -> u32 {
        self.times.get(i)
    }
}

impl Default for Segment {
    fn default() -> Self {
        Writer::with_capacity(0, 0, 0, 0, Layout::Packed(0)).finish()
    }
}

impl Segment {
    /// Build a segment from postings in any order, its references packed
    /// ([`RefCoding::Packed`]).
    ///
    /// Sorts the rows in place (a radix sort on the hash, so close to linear
    /// time), drops exact duplicates, and copies them into the segment's
    /// packed arrays (the bits of a reference and a time per posting, about
    /// `4n` bytes at the usual widths) and its keys, while the `12n`-byte
    /// `rows` are still alive. The rows are freed before the directory is
    /// built. With `std`, a million rows or more are sorted and packed on
    /// every core.
    #[must_use]
    pub fn from_postings(rows: Vec<Posting>) -> Self {
        Self::from_postings_in(rows, RefCoding::Packed)
    }

    /// [`Segment::from_postings`], with the references stored as `coding`
    /// says.
    #[must_use]
    pub fn from_postings_coded(rows: Vec<Posting>, coding: RefCoding) -> Self {
        Self::from_postings_in(rows, coding)
    }

    /// [`Segment::from_postings_coded`].
    pub(crate) fn from_postings_in(mut rows: Vec<Posting>, coding: RefCoding) -> Self {
        crate::sort::sort_postings(&mut rows);
        // A large input is packed on every core, in pieces cut at hash
        // boundaries; the result is the same segment.
        #[cfg(feature = "std")]
        let mut rows = match parallel::pack_sorted(rows, coding) {
            Ok(segment) => return segment,
            Err(rows) => rows,
        };
        rows.dedup();
        // The widths and the number of distinct hashes in one pass, so the
        // key arrays are allocated once at their final size.
        let (mut max_ref, mut max_time, mut keys) = (0, 0, 0usize);
        let mut prev = None;
        for p in &rows {
            max_ref = max_ref.max(p.reference);
            max_time = max_time.max(p.time);
            if prev != Some(p.hash) {
                keys += 1;
                prev = Some(p.hash);
            }
        }
        let mut out = Writer::with_capacity(
            rows.len(),
            keys,
            ref_room(rows.len(), keys, max_ref),
            Packed::bits_for(max_time),
            Layout::new(coding, max_ref),
        );
        for run in rows.chunk_by(|a, b| a.hash == b.hash) {
            out.push_run(run[0].hash, run.iter().map(|p| (p.reference, p.time)));
        }
        // The directory is built without the rows, which are the larger
        // part of the memory held here.
        drop(rows);
        out.finish()
    }

    /// Assemble from arrays the caller has validated (`index.offsets` and
    /// `index.ref_starts` one longer than the keys and spanning
    /// `times` and `refs`, each run's references at most
    /// [`refs::max_run_bits`] long, fewer than [`MAX_POSTINGS`] postings,
    /// `mults[k]` the most postings one reference has in key `k`'s run,
    /// saturating at 255, and `ids` the smallest and largest reference).
    /// The offsets and multiplicities go into the directory and are not
    /// kept.
    #[cfg(any(feature = "std", test))]
    pub(crate) fn from_parts(
        keys: Vec<u32>,
        index: (&[u64], &[u64], &[u8]),
        refs: Refs,
        times: Packed,
        ids: Option<(u32, u32)>,
    ) -> Self {
        let (offsets, ref_starts, mults) = index;
        debug_assert_eq!(offsets.len(), keys.len() + 1);
        debug_assert_eq!(ref_starts.len(), keys.len() + 1);
        debug_assert_eq!(offsets.last().copied(), Some(times.len() as u64));
        debug_assert_eq!(ref_starts.last().copied(), Some(refs.bit_len()));
        assert!(
            (times.len() as u64) < MAX_POSTINGS,
            "a segment holds fewer than 2^40 postings"
        );
        let directory = build_table(
            Index {
                keys: &keys,
                offsets,
                ref_starts,
                mults,
            },
            refs.layout(),
        );
        Self::assemble(keys, refs, times, directory, ids)
    }

    /// The segment of these arrays and this directory.
    fn assemble(
        keys: Vec<u32>,
        refs: Refs,
        times: Packed,
        (table, pages): (Table, Vec<u64>),
        ids: Option<(u32, u32)>,
    ) -> Self {
        let postings = times.len().max(1) as u64;
        let density = u32::try_from(refs.bit_len().div_ceil(postings)).unwrap_or(u32::MAX);
        Self {
            keys,
            refs,
            times,
            table,
            pages,
            ids,
            density,
        }
    }

    /// Number of postings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// `true` if the segment holds no postings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Number of distinct hashes.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// The distinct hashes, ascending.
    #[must_use]
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// Where each key's run starts, then the posting count: `key_count + 1`
    /// values, one directory lookup each, read a batch of keys ahead.
    pub(crate) fn offsets(&self) -> impl Iterator<Item = u64> + '_ {
        self.index_walk()
            .map(|(start, _, _)| start)
            .chain(core::iter::once(self.len() as u64))
    }

    /// Per key in key order, where its run starts, where its coded
    /// references start and the most postings any one reference has in
    /// it: one directory lookup each, read a batch of keys ahead.
    pub(crate) fn index_walk(&self) -> impl Iterator<Item = (u64, u64, u8)> + '_ {
        self.touch_first();
        self.keys.iter().enumerate().map(|(k, &key)| {
            self.touch_ahead(k);
            self.slot(key)
                .map_or((0, 0, 0), |s| (s.start(), self.ref_at(&s), s.mult()))
        })
    }

    /// The smallest and the largest reference, if there is a posting.
    #[cfg(any(feature = "std", test))]
    #[inline]
    pub(crate) fn ids(&self) -> Option<(u32, u32)> {
        self.ids
    }

    /// Plain arrays, the references packed at the width of the largest, in
    /// any order (tests).
    #[cfg(all(test, feature = "std"))]
    pub(crate) fn from_arrays(keys: &[u32], offsets: &[u64], refs: &[u32], times: &[u32]) -> Self {
        Self::from_arrays_coded(keys, offsets, refs, times, RefCoding::Packed)
    }

    /// Plain arrays, the references stored as `coding` says (ascending in
    /// each run when coded as gaps) and the times packed at the width of
    /// the largest (tests).
    #[cfg(all(test, feature = "std"))]
    pub(crate) fn from_arrays_coded(
        keys: &[u32],
        offsets: &[u64],
        refs: &[u32],
        times: &[u32],
        coding: RefCoding,
    ) -> Self {
        let mults: Vec<u8> = offsets
            .windows(2)
            .map(|w| {
                let run = &refs[w[0] as usize..w[1] as usize];
                let (mut best, mut streak) = (0u32, 0u32);
                for (i, &r) in run.iter().enumerate() {
                    streak = if i > 0 && run[i - 1] == r {
                        streak + 1
                    } else {
                        1
                    };
                    best = best.max(streak);
                }
                u8::try_from(best).unwrap_or(u8::MAX)
            })
            .collect();
        let layout = Layout::new(coding, refs.iter().copied().max().unwrap_or(0));
        let width = u64::from(layout.array_width());
        let mut coded = PackedBuilder::with_capacity(0, layout.array_width());
        let mut ref_starts = Vec::with_capacity(offsets.len());
        for w in offsets.windows(2) {
            ref_starts.push(coded.len() as u64 * width);
            let run = &refs[w[0] as usize..w[1] as usize];
            if !run.is_empty() {
                match layout {
                    Layout::Packed(_) => coded.extend(run.iter().copied()),
                    Layout::Gaps => refs::push_run(&mut coded, run),
                }
            }
        }
        ref_starts.push(coded.len() as u64 * width);
        let ids = refs
            .iter()
            .fold(None, |ids, &r| union_ids(ids, Some((r, r))));
        Self::from_parts(
            keys.to_vec(),
            (offsets, &ref_starts, &mults),
            Refs::new(coded.finish(), layout),
            Packed::from_values(times),
            ids,
        )
    }

    /// Mark `hash`'s slot as too long for a slot, so its lookups take the
    /// sorted-keys path (tests).
    #[cfg(test)]
    fn mark_long(&mut self, hash: u32) {
        fn mark<T: Stored>(table: &mut [T], hash: u32) {
            let slots = table.len();
            let mut i = slot_of(hash, slots);
            while table[i].key() != hash {
                i = (i + 1) % slots;
            }
            let slot = table[i].slot();
            table[i] = T::of(Slot::new(
                hash,
                slot.start(),
                u64::from(LONG),
                slot.mult(),
                slot.refs,
            ));
        }
        match &mut self.table {
            Table::Short(t) => mark(t, hash),
            Table::Whole(t) => mark(t, hash),
        }
    }

    /// Rebuild the directory in a table of `slots`, the size a test picks
    /// instead of the one [`slots_for`] gives the key count (tests).
    #[cfg(test)]
    fn retable(&mut self, slots: usize) {
        let (mut offsets, mut ref_starts, mut mults) = (Vec::new(), Vec::new(), Vec::new());
        for (start, at, mult) in self.index_walk() {
            offsets.push(start);
            ref_starts.push(at);
            mults.push(mult);
        }
        offsets.push(self.len() as u64);
        ref_starts.push(self.refs.bit_len());
        let index = Index {
            keys: &self.keys,
            offsets: &offsets,
            ref_starts: &ref_starts,
            mults: &mults,
        };
        self.table = Table::build(index, &self.pages, slots, self.refs.layout());
    }

    #[cfg(any(feature = "std", test))]
    #[inline]
    pub(crate) fn refs(&self) -> &Refs {
        &self.refs
    }

    #[inline]
    pub(crate) fn times(&self) -> &Packed {
        &self.times
    }

    /// The directory slot of `hash`, if the segment holds the hash.
    #[inline]
    fn slot(&self, hash: u32) -> Option<Slot> {
        self.table.find(hash)
    }

    /// Entry range of `hash`'s run, if the segment holds the hash.
    #[cfg(any(feature = "std", test))]
    #[inline]
    pub(crate) fn range(&self, hash: u32) -> Option<(usize, usize)> {
        self.locate(hash).map(|run| (run.lo, run.hi))
    }

    /// Entry range of `hash`'s run and the most postings any one reference
    /// has in it (saturating at 255), if the segment holds the hash.
    #[cfg(test)]
    pub(crate) fn range_mult(&self, hash: u32) -> Option<(usize, usize, u8)> {
        self.locate(hash).map(|run| (run.lo, run.hi, run.mult))
    }

    /// Where `hash`'s run is, if the segment holds the hash.
    #[inline]
    pub(crate) fn locate(&self, hash: u32) -> Option<Located> {
        let slot = self.slot(hash)?;
        Some(self.located(&slot, hash, self.ref_at(&slot)))
    }

    /// The run of `hash`, whose slot is `slot` and whose references start
    /// at bit `at`.
    #[inline]
    fn located(&self, slot: &Slot, hash: u32, at: u64) -> Located {
        let lo = slot.start() as usize;
        let hi = if slot.len() == LONG {
            self.end_by_search(hash)
        } else {
            lo + slot.len() as usize
        };
        Located {
            lo,
            hi,
            mult: slot.mult(),
            at,
        }
    }

    /// End of `hash`'s run through the sorted keys (the start of the next
    /// key's run): the path for runs too long for a slot, and the reference
    /// the fast path is checked against.
    fn end_by_search(&self, hash: u32) -> usize {
        let k = self
            .keys
            .binary_search(&hash)
            .expect("a key in the directory is in the sorted keys");
        match self.keys.get(k + 1) {
            Some(&next) => self.slot(next).expect("every key has a slot").start() as usize,
            None => self.len(),
        }
    }

    /// Probes a lookup of `hash` makes before it resolves.
    #[cfg(test)]
    fn probes(&self, hash: u32) -> usize {
        fn probes_in<T: Stored>(table: &[T], hash: u32) -> usize {
            let slots = table.len();
            let mut i = slot_of(hash, slots);
            let mut n = 1;
            while table[i].len() != 0 && table[i].key() != hash {
                i = (i + 1) % slots;
                n += 1;
            }
            n
        }
        match &self.table {
            Table::Short(t) => probes_in(t, hash),
            Table::Whole(t) => probes_in(t, hash),
        }
    }

    /// The postings of `hash`, if the segment holds it.
    #[must_use]
    pub fn run(&self, hash: u32) -> Option<Run<'_>> {
        let run = self.locate(hash)?;
        Some(Run {
            segment: self,
            lo: run.lo,
            hi: run.hi,
            at: run.at,
        })
    }

    /// Every posting in `(hash, reference, time)` order.
    #[must_use]
    pub fn postings(&self) -> Postings<'_> {
        self.touch_first();
        self.touch_ahead(0);
        let first = self.keys.first().and_then(|&k| self.locate(k));
        Postings {
            segment: self,
            times: self.times.view(),
            key: 0,
            index: first.map_or(0, |run| run.lo),
            end: first.map_or(0, |run| run.hi),
            refs: first.map(|run| self.refs.run(run.at, run.lo, run.hi - run.lo).cursor()),
        }
    }

    /// The largest reference id with a posting, if any.
    #[must_use]
    pub fn max_reference(&self) -> Option<u32> {
        self.ids.map(|(_, max)| max)
    }

    /// Bits the references and the packed times take, in total. The times
    /// are packed at the width of the largest; the references take the
    /// width of the largest each when packed, and about the bits of the gap
    /// between neighbours in a run when coded as gaps (see [`RefCoding`]).
    #[must_use]
    pub fn bits(&self) -> (u64, u64) {
        (
            self.refs.bit_len(),
            self.times.len() as u64 * u64::from(self.times.bits()),
        )
    }

    /// Bits per time, the width of the largest.
    #[must_use]
    pub fn time_bits(&self) -> u32 {
        self.times.bits()
    }

    /// Heap bytes held by the arrays and the directory.
    ///
    /// A posting costs its share of the coded references and its time
    /// (see [`Segment::bits`]); a distinct hash costs 4 bytes of key and
    /// about 27 of directory, or 4 bytes of key and 64 of directory in a
    /// segment of fewer than 2<sup>20</sup> keys.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.keys.capacity() * size_of::<u32>()
            + self.refs.heap_bytes()
            + self.times.heap_bytes()
            + self.table.heap_bytes()
            + self.pages.capacity() * size_of::<u64>()
    }

    /// Merge two segments, keeping only postings whose reference `keep`
    /// accepts, into a segment that stores its references as `left`
    /// does. Linear in the two sizes; a posting present in both is kept
    /// once. With `std`, a large merge runs on every core, so `keep` is
    /// called from several threads.
    ///
    /// # Panics
    ///
    /// Panics if the result would hold 2<sup>40</sup> postings or more.
    #[must_use]
    pub fn merge(left: &Segment, right: &Segment, keep: impl Fn(u32) -> bool + Sync) -> Segment {
        Self::merge_all(
            &[(left, Some(&keep)), (right, Some(&keep))],
            left.ref_coding(),
        )
    }

    /// A copy without the postings of references `keep` rejects, its
    /// references stored as this segment's are. With `std`, a large
    /// segment is filtered on every core, so `keep` is called from several
    /// threads.
    #[must_use]
    pub fn filter(&self, keep: impl Fn(u32) -> bool + Sync) -> Segment {
        Self::merge_all(&[(self, Some(&keep))], self.ref_coding())
    }

    /// How this segment stores its references.
    #[must_use]
    pub fn ref_coding(&self) -> RefCoding {
        self.refs.layout().coding()
    }

    /// The same postings with the references stored as `coding` says: a
    /// copy that decodes and codes every run again, or a plain copy when
    /// the segment already stores them that way.
    #[must_use]
    pub fn to_ref_coding(&self, coding: RefCoding) -> Segment {
        if coding == self.ref_coding() {
            return self.clone();
        }
        Self::merge_all::<fn(u32) -> bool>(&[(self, None)], coding)
    }

    /// Merge any number of segments in one pass, each filtered by its
    /// `keep` if it has one, into a segment whose references are stored as
    /// `coding` says; a posting present in several is kept once.
    ///
    /// A segment without a `keep` must hold nothing another's `keep`
    /// rejects; its runs are then copied without a look at their
    /// references. Linear in the total size, times the number of segments
    /// for the distinct hashes.
    ///
    /// With `std`, a large merge runs on every core: the hash space is cut
    /// into pieces of about equal postings, each merged on its own, and the
    /// pieces are joined in order. The result is the same posting for
    /// posting.
    pub(crate) fn merge_all<F: Fn(u32) -> bool + Sync>(
        sides: &[(&Segment, Option<&F>)],
        coding: RefCoding,
    ) -> Segment {
        let layout = layout_of(sides, coding);
        #[cfg(feature = "std")]
        if let Some(merged) = parallel::merge_all(sides, layout) {
            return merged;
        }
        // Every source's keys, an upper bound on the merge's distinct
        // hashes: one allocation each for the keys, their starts and their
        // multiplicities instead of a doubling per batch of runs.
        let mut out = Writer::with_capacity(
            postings(sides),
            keys(sides),
            ref_bits(sides),
            time_bits(sides),
            layout,
        );
        let mut sources: Vec<_> = sides
            .iter()
            .map(|&(segment, keep)| segment.source(keep))
            .collect();
        merge_sources(&mut out, &mut sources);
        // The walks are spent: the directory is built without them.
        drop(sources);
        out.include(copied_ids(sides));
        out.finish()
    }

    /// The segment of one reference, stored as `coding` says: `rows` are
    /// its `hash << 32 | time` values, ascending and distinct.
    pub(crate) fn from_reference(reference: u32, rows: &[u64], coding: RefCoding) -> Segment {
        debug_assert!(
            rows.windows(2).all(|w| w[0] < w[1]),
            "rows not sorted and unique"
        );
        let max_time = rows.iter().fold(0, |max, &row| max.max(row as u32));
        // Every row could be a hash of its own, so the rows bound the keys:
        // one allocation for them rather than one per doubling.
        let mut out = Writer::with_capacity(
            rows.len(),
            rows.len(),
            ref_room(rows.len(), rows.len(), reference),
            Packed::bits_for(max_time),
            Layout::new(coding, reference),
        );
        for run in rows.chunk_by(|a, b| a >> 32 == b >> 32) {
            out.push_run(
                (run[0] >> 32) as u32,
                run.iter().map(|&row| (reference, row as u32)),
            );
        }
        out.finish()
    }
}

/// Postings across `sides`, saturating.
fn postings<F>(sides: &[(&Segment, Option<&F>)]) -> usize {
    sides
        .iter()
        .fold(0usize, |n, (s, _)| n.saturating_add(s.len()))
}

/// Distinct hashes across `sides`, saturating: an upper bound on a merge's
/// key count, since a hash in several of them is one key.
fn keys<F>(sides: &[(&Segment, Option<&F>)]) -> usize {
    sides
        .iter()
        .fold(0usize, |n, (s, _)| n.saturating_add(s.key_count()))
}

/// The widest times across `sides`: the width of their merge's.
fn time_bits<F>(sides: &[(&Segment, Option<&F>)]) -> u32 {
    sides
        .iter()
        .map(|&(s, _)| s.times.bits())
        .max()
        .unwrap_or(0)
}

/// The bits of coded references across `sides`, saturating: about what
/// their merge's take.
fn ref_bits<F>(sides: &[(&Segment, Option<&F>)]) -> u64 {
    sides
        .iter()
        .fold(0u64, |n, (s, _)| n.saturating_add(s.refs.bit_len()))
}

/// The layout of a merge of `sides` stored as `coding` says: packed, at
/// the width of the largest reference any side holds.
pub(crate) fn layout_of<F>(sides: &[(&Segment, Option<&F>)], coding: RefCoding) -> Layout {
    let max_ref = sides
        .iter()
        .filter_map(|(s, _)| s.ids)
        .map(|(_, max)| max)
        .max()
        .unwrap_or(0);
    Layout::new(coding, max_ref)
}

/// The range of the references of the sides without a `keep`, whose
/// runs a merge may copy without a look at them: every one of their
/// postings is in the merge.
fn copied_ids<F>(sides: &[(&Segment, Option<&F>)]) -> Option<(u32, u32)> {
    sides
        .iter()
        .filter(|(_, keep)| keep.is_none())
        .fold(None, |ids, (s, _)| union_ids(ids, s.ids))
}

/// About the bits the coded references of `postings` postings in `keys`
/// runs take, references up to `max_ref`: room a writer reserves, which
/// is at most a little more than they take.
fn ref_room(postings: usize, keys: usize, max_ref: u32) -> u64 {
    // A run's first reference at its width, then gaps that are mostly
    // well below it.
    let per_posting = u64::from(Packed::bits_for(max_ref)).max(1);
    (postings as u64)
        .saturating_mul(per_posting)
        .saturating_add(keys as u64 * 10)
}

/// Where a run is: entries `lo..hi` of the times, its multiplicity, and
/// the bit its coded references start at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Located {
    pub(crate) lo: usize,
    pub(crate) hi: usize,
    pub(crate) mult: u8,
    pub(crate) at: u64,
}

/// Write the union of every run of `sources` to `out`, hashes ascending.
fn merge_sources<F: Fn(u32) -> bool>(out: &mut Writer, sources: &mut [Source<'_, '_, F>]) {
    // The runs of the smallest hash at the head of any source, each with
    // the index of its source.
    let mut same = Vec::with_capacity(sources.len());
    loop {
        let hash = sources
            .iter()
            .map(Source::head)
            .fold(Source::<F>::DONE, u64::min);
        if hash == Source::<F>::DONE {
            break;
        }
        same.clear();
        for (i, source) in sources.iter_mut().enumerate() {
            if source.head() == hash {
                same.push((i, source.take()));
            }
        }
        out.merge_runs(hash as u32, sources, &mut same);
    }
}

impl PartialEq for Segment {
    /// Same postings, whatever the widths and however the references are
    /// stored.
    fn eq(&self, other: &Self) -> bool {
        let same_refs = match (self.refs.layout(), other.refs.layout()) {
            // Canonical, so the same bits.
            (Layout::Gaps, Layout::Gaps) => self.refs == other.refs,
            _ => self.postings().eq(other.postings()),
        };
        self.keys == other.keys
            && self.times == other.times
            && self.offsets().eq(other.offsets())
            && self.ids == other.ids
            && same_refs
    }
}

impl Eq for Segment {}

impl FromIterator<Posting> for Segment {
    /// [`Segment::from_postings`] over the collected items.
    fn from_iter<I: IntoIterator<Item = Posting>>(iter: I) -> Self {
        Self::from_postings(iter.into_iter().collect())
    }
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("postings", &self.len())
            .field("keys", &self.keys.len())
            .field("slots", &self.table.len())
            .field("refs", &self.refs.layout())
            .field("ref_bits", &self.refs.bit_len())
            .field("time_bits", &self.times.bits())
            .finish_non_exhaustive()
    }
}

/// Iterator over a segment's postings in sorted order
/// (see [`Segment::postings`]).
#[derive(Clone)]
pub struct Postings<'a> {
    segment: &'a Segment,
    times: View<'a>,
    key: usize,
    /// Next entry, inside `key`'s run or at its end.
    index: usize,
    /// End of `key`'s run.
    end: usize,
    /// The references of `key`'s run not yet returned.
    refs: Option<refs::Cursor<'a>>,
}

impl Iterator for Postings<'_> {
    type Item = Posting;

    #[inline]
    fn next(&mut self) -> Option<Posting> {
        let s = self.segment;
        // Runs are laid out in key order, so a run's end is the next run's
        // start; only the end needs a lookup.
        while self.index >= self.end {
            self.key += 1;
            let &key = s.keys.get(self.key)?;
            s.touch_ahead(self.key);
            let run = s.locate(key).expect("every key has a run");
            self.end = run.hi;
            self.refs = Some(s.refs.run(run.at, run.lo, run.hi - run.lo).cursor());
        }
        let reference = self
            .refs
            .as_mut()
            .and_then(Iterator::next)
            .expect("a run's references are as many as its postings");
        let p = Posting {
            hash: s.keys[self.key],
            reference,
            time: self.times.get(self.index),
        };
        self.index += 1;
        Some(p)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.segment.len() - self.index;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Postings<'_> {}

impl fmt::Debug for Postings<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Postings")
            .field("remaining", &self.len())
            .finish_non_exhaustive()
    }
}

impl FusedIterator for Postings<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every way two segments of a merge can store their references.
    const CODING_PAIRS: [(RefCoding, RefCoding); 4] = [
        (RefCoding::Packed, RefCoding::Packed),
        (RefCoding::Gaps, RefCoding::Gaps),
        (RefCoding::Packed, RefCoding::Gaps),
        (RefCoding::Gaps, RefCoding::Packed),
    ];
    use alloc::vec;

    fn p(h: u32, r: u32, t: u32) -> Posting {
        Posting::new(h, r, t)
    }

    #[test]
    fn empty_segment_has_no_runs() {
        let s = Segment::default();
        assert!(s.is_empty());
        assert_eq!(s.key_count(), 0);
        assert!(s.run(0).is_none());
        assert!(s.run(5).is_none());
        assert!(s.run(u32::MAX).is_none());
        assert_eq!(s.postings().count(), 0);
        assert!(s.max_reference().is_none());
        let s = Segment::from_postings(Vec::new());
        assert!(s.run(7).is_none());
        assert_eq!(s, Segment::default());
    }

    #[test]
    fn sorts_dedups_and_groups_runs() {
        let s: Segment = vec![
            p(5, 1, 30),
            p(5, 0, 10),
            p(2, 7, 1),
            p(5, 1, 20),
            p(5, 1, 20),
            p(u32::MAX, 3, 0),
            p(0, 3, 9),
        ]
        .into_iter()
        .collect();
        assert_eq!(s.len(), 6);
        assert_eq!(s.keys(), &[0, 2, 5, u32::MAX]);
        let run = s.run(5).unwrap();
        assert_eq!(run.len(), 3);
        assert!(!run.is_empty());
        assert_eq!(run.iter().len(), 3);
        assert_eq!(
            run.iter().collect::<Vec<_>>(),
            vec![(0, 10), (1, 20), (1, 30)]
        );
        assert_eq!(
            run.iter().collect::<Vec<_>>(),
            vec![(0, 10), (1, 20), (1, 30)]
        );
        assert_eq!(s.run(0).unwrap().iter().collect::<Vec<_>>(), vec![(3, 9)]);
        assert_eq!(
            s.run(u32::MAX).unwrap().iter().collect::<Vec<_>>(),
            vec![(3, 0)]
        );
        assert!(s.run(1).is_none());
        assert!(s.run(6).is_none());
        assert_eq!(s.max_reference(), Some(7));
        let all: Vec<Posting> = s.postings().collect();
        assert_eq!(
            all,
            vec![
                p(0, 3, 9),
                p(2, 7, 1),
                p(5, 0, 10),
                p(5, 1, 20),
                p(5, 1, 30),
                p(u32::MAX, 3, 0)
            ]
        );
        assert_eq!(s.postings().len(), 6);
        let mut it = s.postings();
        for _ in 0..6 {
            it.next();
        }
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn lookup_agrees_with_linear_scan_across_the_hash_space() {
        // Hashes spread over the whole u32 range, including both extremes.
        let rows: Vec<Posting> = (0..5_000u32)
            .map(|i| p(i.wrapping_mul(0x9E37_79B1), i % 13, i))
            .chain([p(0, 1, 1), p(u32::MAX, 2, 2)])
            .collect();
        let s = Segment::from_postings(rows.clone());
        for r in &rows {
            let run = s.run(r.hash).expect("present hash");
            assert!(run.iter().any(|(rr, t)| rr == r.reference && t == r.time));
        }
        for probe in [1u32, 2, 3, 0x8000_0000, u32::MAX - 1] {
            assert_eq!(s.run(probe).is_some(), rows.iter().any(|r| r.hash == probe));
        }
    }

    #[test]
    fn directory_load_switches_at_the_sparse_threshold() {
        // Four slots per key below `SPARSE_KEYS`, and fewer than two from
        // there: a lookup that misses walks to an empty slot, so the load
        // is what decides how far it walks.
        for keys in [0usize, 1, 7, 1_000, SPARSE_KEYS - 1] {
            let slots = slots_for(keys);
            assert_eq!(slots, 4 * keys + 1, "{keys} keys");
            // Three empty slots per key, which ends a miss in under two
            // probes.
            assert_eq!(slots - keys, 3 * keys + 1, "{keys} keys");
        }
        for keys in [SPARSE_KEYS, SPARSE_KEYS + 1, 3 * SPARSE_KEYS + 2] {
            let slots = slots_for(keys);
            assert_eq!(slots, keys + keys / 3 * 2 + 1, "{keys} keys");
            assert!(
                slots < 2 * keys,
                "{keys} keys: fewer than two slots per key"
            );
            assert!(slots > keys, "{keys} keys: a key needs a free slot");
        }
    }

    #[test]
    fn directory_spreads_keys_that_vary_in_few_bits() {
        // Keys below 2^16, keys with a constant top byte, and multiples of
        // 2^16: each set must resolve, miss cleanly, and probe briefly at
        // the sparse load a segment this small gets, and at the dense one
        // every large segment uses.
        let sets: [Vec<u32>; 3] = [
            (0..4_000u32).map(|i| i * 7 % 65_536).collect(),
            (0..3_000u32).map(|i| 0xAB00_0000 | (i * 1_031)).collect(),
            (0..3_000u32).map(|i| i << 16).collect(),
        ];
        for keys in sets {
            let rows: Vec<Posting> = keys
                .iter()
                .enumerate()
                .map(|(i, &h)| p(h, 0, i as u32))
                .collect();
            let mut s = Segment::from_postings(rows.clone());
            let n = s.key_count();
            assert_eq!(s.table.len(), slots_for(n));
            // The sparse load this size gets, then the dense one of a
            // large segment.
            for slots in [sparse_slots(n), dense_slots(n)] {
                s.retable(slots);
                assert_eq!(s.table.len(), slots);
                let mut probes = 0;
                for r in &rows {
                    assert!(s.run(r.hash).is_some());
                    assert_eq!(s.range(r.hash).unwrap().1, s.end_by_search(r.hash));
                    probes += s.probes(r.hash);
                }
                assert!(
                    probes <= rows.len() * 2,
                    "{} probes for {} keys in {slots} slots",
                    probes,
                    rows.len()
                );
                assert!(s.table.len() > s.key_count());
                for probe in [1u32, 65_537, 1 << 20, 0xAB00_0000 | 1, u32::MAX] {
                    assert_eq!(s.run(probe).is_some(), rows.iter().any(|r| r.hash == probe));
                }
            }
        }
        assert!(Segment::default().run(0).is_none());
    }

    #[test]
    fn slot_packs_length_multiplicity_and_start() {
        let s = Slot::new(7, (1 << 40) - 1, u64::from(LONG) - 1, 255, u32::MAX);
        assert_eq!(
            (s.key, s.start(), s.len(), s.mult(), s.refs),
            (7, (1 << 40) - 1, LONG - 1, 255, u32::MAX)
        );
        assert_eq!(Slot::new(1, 0, 1 << 20, 0, 0).len(), LONG);
        assert_eq!(size_of::<Slot>(), 16);
        assert_eq!(size_of::<Short>(), 12);
        let short = Short::of(s);
        assert_eq!((short.key(), short.len()), (7, LONG - 1));
        assert_eq!(short.slot().order(), s.order());
        assert_eq!(Slot::EMPTY.len(), 0);
    }

    #[test]
    fn slots_record_the_run_multiplicity() {
        // Hash 1: one posting per reference. Hash 2: reference 5 three
        // times. Hash 3: 300 postings of one reference, past the byte.
        let mut rows = vec![
            p(1, 0, 0),
            p(1, 1, 0),
            p(2, 4, 0),
            p(2, 5, 0),
            p(2, 5, 1),
            p(2, 5, 2),
            p(2, 6, 0),
        ];
        rows.extend((0..300).map(|t| p(3, 9, t)));
        let s = Segment::from_postings(rows);
        assert_eq!(s.range_mult(1).map(|r| r.2), Some(1));
        assert_eq!(s.range_mult(2).map(|r| r.2), Some(3));
        assert_eq!(s.range_mult(3).map(|r| r.2), Some(255));
        assert_eq!(s.range_mult(4), None);
        let m = Segment::merge(&s, &Segment::from_postings(vec![p(2, 5, 3)]), |_| true);
        assert_eq!(m.range_mult(2).map(|r| r.2), Some(4));
    }

    #[test]
    fn overlong_slots_resolve_through_the_sorted_keys() {
        let rows: Vec<Posting> = (0..600u32).map(|i| p(i / 3, i % 3, i)).collect();
        let mut s = Segment::from_postings(rows.clone());
        let expected: Vec<Option<(usize, usize)>> = (0..=200u32).map(|h| s.range(h)).collect();
        // Every key, including the last, whose end is the posting count.
        for h in 0..200u32 {
            s.mark_long(h);
        }
        for h in 0..=200u32 {
            assert_eq!(s.range(h), expected[h as usize], "hash {h}");
        }
        assert_eq!(s.postings().collect::<Vec<_>>(), rows);
    }

    #[test]
    fn merge_interleaves_dedups_and_filters() {
        let a = Segment::from_postings(vec![p(1, 0, 5), p(3, 0, 1), p(3, 2, 2), p(9, 0, 0)]);
        let b = Segment::from_postings(vec![p(2, 1, 0), p(3, 1, 7), p(3, 2, 2), p(3, 2, 3)]);
        let m = Segment::merge(&a, &b, |_| true);
        let got: Vec<Posting> = m.postings().collect();
        assert_eq!(
            got,
            vec![
                p(1, 0, 5),
                p(2, 1, 0),
                p(3, 0, 1),
                p(3, 1, 7),
                p(3, 2, 2),
                p(3, 2, 3),
                p(9, 0, 0)
            ]
        );
        let m = Segment::merge(&a, &b, |r| r != 2);
        let got: Vec<Posting> = m.postings().collect();
        assert_eq!(
            got,
            vec![p(1, 0, 5), p(2, 1, 0), p(3, 0, 1), p(3, 1, 7), p(9, 0, 0)]
        );
        let m = Segment::merge(&a, &Segment::default(), |_| true);
        assert_eq!(m, a);
        let f = a.filter(|r| r == 2);
        assert_eq!(f.postings().collect::<Vec<_>>(), vec![p(3, 2, 2)]);
        assert!(a.filter(|_| false).is_empty());
        assert_ne!(a, b);
    }

    /// Merges and filters against the sorted, deduplicated union, over
    /// reference layouts that take each path of the run merge (left before
    /// right, right before left, interleaved with shared postings) and
    /// hash ranges that leave runs on one side only.
    #[test]
    fn merge_and_filter_agree_with_the_sorted_union() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut below = move |n: u32| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % u64::from(n)) as u32
        };
        let keeps: [fn(u32) -> bool; 4] = [|_| true, |r| r % 3 != 0, |r| r >= 25, |_| false];
        for (left, right) in [(0..20u32, 20..40u32), (20..40, 0..20), (0..30, 10..40)] {
            let mut rows = |hashes: core::ops::Range<u32>, refs: &core::ops::Range<u32>, n| {
                (0..n)
                    .map(|_| {
                        p(
                            hashes.start + below(hashes.end - hashes.start),
                            refs.start + below(refs.end - refs.start),
                            below(50),
                        )
                    })
                    .collect::<Vec<Posting>>()
            };
            let a_rows = rows(0..40, &left, 3_000);
            let b_rows = rows(20..60, &right, 1_500);
            // Every pair of codings: merged into the left one's, so runs of
            // the other are decoded and coded again.
            for (ca, cb) in CODING_PAIRS {
                let a = Segment::from_postings_coded(a_rows.clone(), ca);
                let b = Segment::from_postings_coded(b_rows.clone(), cb);
                for keep in keeps {
                    let sorted = |rows: &mut dyn Iterator<Item = Posting>| {
                        let mut v: Vec<Posting> = rows.collect();
                        v.sort_unstable();
                        v.dedup();
                        v
                    };
                    let kept = |p: &Posting| keep(p.reference);
                    let union = sorted(&mut a_rows.iter().chain(&b_rows).copied().filter(kept));
                    // A reference only the left side holds: rejecting it (and
                    // what `keep` rejects there) leaves the right side with
                    // nothing to filter, so it may go without a `keep`.
                    let only_left = |r: u32| left.contains(&r) && !right.contains(&r);
                    let keep_left = |r: u32| !only_left(r) || (keep(r) && r.is_multiple_of(2));
                    let one_sided = sorted(
                        &mut a_rows
                            .iter()
                            .copied()
                            .filter(|p| keep_left(p.reference))
                            .chain(b_rows.iter().copied()),
                    );
                    let unfiltered = sorted(&mut a_rows.iter().chain(&b_rows).copied());
                    for (got, want) in [
                        (Segment::merge(&a, &b, keep), union.clone()),
                        (Segment::merge(&b, &a, keep).to_ref_coding(ca), union),
                        (
                            a.filter(keep),
                            sorted(&mut a_rows.iter().copied().filter(kept)),
                        ),
                        (
                            Segment::merge_all(&[(&a, Some(&keep_left)), (&b, None)], ca),
                            one_sided.clone(),
                        ),
                        (
                            Segment::merge_all(&[(&b, None), (&a, Some(&keep_left))], ca),
                            one_sided,
                        ),
                        (
                            Segment::merge_all::<fn(u32) -> bool>(&[(&a, None), (&b, None)], ca),
                            unfiltered.clone(),
                        ),
                        // Four ways at once, two of them the same segment.
                        (
                            Segment::merge_all::<fn(u32) -> bool>(
                                &[
                                    (&b, None),
                                    (&a, None),
                                    (&Segment::default(), None),
                                    (&b, None),
                                ],
                                ca,
                            ),
                            unfiltered,
                        ),
                    ] {
                        assert_eq!(got.postings().collect::<Vec<_>>(), want);
                        assert_eq!(got.ref_coding(), ca);
                        // The directory agrees with a segment built from rows,
                        // multiplicities included, and so do the bits.
                        let fresh = Segment::from_postings_coded(want, ca);
                        assert_eq!(got, fresh);
                        if ca == RefCoding::Gaps {
                            assert!(got.refs == fresh.refs, "the canonical coding");
                        }
                        for h in 0..60 {
                            assert_eq!(got.range_mult(h), fresh.range_mult(h), "hash {h}");
                        }
                    }
                }
            }
        }
    }

    /// A walk in key order crosses many directory batches and agrees with
    /// lookups one key at a time.
    #[test]
    fn walks_in_key_order_agree_with_lookups() {
        let rows: Vec<Posting> = (0..10_000u32)
            .map(|i| p(i.wrapping_mul(0x9E37_79B1) >> 8, i % 7, i % 5))
            .collect();
        let s = Segment::from_postings(rows);
        assert!(s.key_count() > 4 * AHEAD);
        let mut source = s.source::<fn(u32) -> bool>(None);
        let mut at = 0;
        for &hash in s.keys() {
            assert_eq!(source.head(), u64::from(hash));
            let run = source.take();
            assert_eq!((run.lo, run.hi, run.mult), s.range_mult(hash).unwrap());
            assert_eq!(run.lo, at);
            at = run.hi;
        }
        assert_eq!(source.head(), u64::MAX);
        assert_eq!(at, s.len());
        let offsets: Vec<u64> = s.offsets().collect();
        assert_eq!(offsets.len(), s.key_count() + 1);
        assert!(offsets.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(offsets.last(), Some(&(s.len() as u64)));
        let mut walked = s.postings();
        assert_eq!(walked.len(), s.len());
        for &hash in s.keys() {
            for (reference, time) in s.run(hash).unwrap().iter() {
                assert_eq!(walked.next(), Some(p(hash, reference, time)));
            }
        }
        assert_eq!(walked.next(), None);
        assert!(alloc::format!("{walked:?}").contains("remaining: 0"));
    }

    /// A real segment of `SPARSE_KEYS` keys, the smallest that takes the
    /// dense load: every key resolves to its run, hashes it does not hold
    /// miss, and the directory has the dense size rather than the sparse
    /// one a smaller segment gets.
    #[test]
    fn a_segment_at_the_sparse_threshold_uses_the_dense_directory() {
        // A bijection of `0..SPARSE_KEYS` over the hash space, one posting
        // of reference 0 per key, its ordinal as the time.
        let mut keys: Vec<(u32, u32)> = (0..SPARSE_KEYS as u32)
            .map(|i| (i.wrapping_mul(0x9E37_79B1), i))
            .collect();
        let rows: Vec<Posting> = keys.iter().map(|&(hash, t)| p(hash, 0, t)).collect();
        let s = Segment::from_postings(rows);
        assert_eq!(s.key_count(), SPARSE_KEYS);
        assert_eq!(s.len(), SPARSE_KEYS);
        // Three keys per five slots, not the four per key a smaller segment
        // gets: the dense directory is the one every large segment has.
        assert_eq!(s.table.len(), dense_slots(SPARSE_KEYS));
        assert!(s.table.len() < sparse_slots(SPARSE_KEYS));
        // Every key resolves to its run, one posting with its time.
        keys.sort_unstable();
        for (rank, &(hash, t)) in keys.iter().enumerate() {
            let (lo, hi, mult) = s.range_mult(hash).expect("a key the segment holds");
            assert_eq!((lo, hi, mult), (rank, rank + 1, 1), "hash {hash}");
            assert_eq!(s.times().get(rank), t, "time of hash {hash}");
        }
        // Hashes the keys miss. They cover the image of a bijection, so a
        // sorted copy answers the lookups as the directory must.
        for j in 0..4_096u32 {
            let probe = j.wrapping_mul(0x85EB_CA6B) | 1;
            assert_eq!(
                s.run(probe).is_some(),
                keys.binary_search_by_key(&probe, |&(hash, _)| hash).is_ok(),
                "hash {probe}"
            );
        }
        for probe in [u32::MAX, 0x8000_0001] {
            assert_eq!(
                s.run(probe).is_some(),
                keys.binary_search_by_key(&probe, |&(hash, _)| hash).is_ok(),
                "hash {probe}"
            );
        }
    }

    #[test]
    fn heap_bytes_counts_every_array_and_finish_trims_capacity() {
        let rows = vec![p(1, 0, 0), p(2, 0, 0)];
        let s = Segment::from_postings(rows.clone());
        // Two keys get a nine-slot directory, the sparse load of a small
        // segment, of 12-byte slots over packed references; the references
        // and the times are all zero, so they pack to no data bytes plus
        // slack.
        assert_eq!(s.bits(), (0, 0));
        assert!(s.heap_bytes() >= 2 * 4 + 8 + 8 + 9 * 12);
        // Coded as gaps, each run codes its one reference in a width field
        // of 5 bits, and the directory's slots are 16 bytes and count from
        // a page start of 8.
        let gaps = Segment::from_postings_coded(rows, RefCoding::Gaps);
        assert_eq!(gaps.bits(), (10, 0));
        assert_eq!(gaps.heap_bytes() - s.heap_bytes(), 9 * 4 + 8 + 2);
        // Ten duplicates of one posting: the hint was 10, the segment holds 1
        // reference, of one bit packed, and a width field and one bit as
        // gaps.
        let s = Segment::from_postings(vec![p(1, 1, 0); 10]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bits(), (1, 0));
        assert_eq!(s.refs.heap_bytes(), 1 + Packed::SLACK);
        let gaps = Segment::from_postings_coded(vec![p(1, 1, 0); 10], RefCoding::Gaps);
        assert_eq!(gaps.bits(), (6, 0));
        assert_eq!(gaps.refs.heap_bytes(), 1 + Packed::SLACK);
        assert!(alloc::format!("{s:?}").starts_with("Segment { postings: 1"));
    }
}
