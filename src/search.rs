use alloc::vec::Vec;
use core::fmt;

use crate::packed::Packed;
use crate::simd::{self, Decoder, LANES, Pass};
use crate::{Catalog, Query, SearchError, Segment};

/// Bounds and statistics parameters for a search.
///
/// Start from [`SearchOptions::default`] and assign the fields to change:
///
/// ```
/// # use afpindex::SearchOptions;
/// let mut options = SearchOptions::default();
/// options.min_total_votes = 84;
/// ```
///
/// Fields may be added in `0.x` releases, which is why the struct cannot
/// be built with a literal, even one that fills the rest from the default:
///
/// ```compile_fail,E0639
/// let options = afpindex::SearchOptions { min_total_votes: 84, ..Default::default() };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SearchOptions {
    /// Most postings of one `(hash, reference)` pair that may vote: the
    /// first `fanout` by time. `None` lets every posting vote, and
    /// `Some(0)` none, so that search finds nothing. Default 64.
    ///
    /// Together with the query's `max_times_per_hash` this bounds the votes
    /// one pair can cast and so the cost of a hash that is common in both.
    pub fanout: Option<usize>,
    /// Vote floor: a reference needs at least this many total votes to
    /// become a survivor and get a histogram. `0` and `1` are equivalent.
    /// Default 1; see [`vote_floor`] for a query-proportional value.
    pub min_total_votes: u32,
    /// Bins within `±offset_tolerance` of a candidate's best Δ are excluded
    /// from its noise statistics. Default 2.
    pub offset_tolerance: u32,
    /// Fewer noise bins than this report zero mean, deviation and max
    /// (`NoiseStats::bins` still counts them). Default 10.
    pub min_noise_bins: usize,
    /// Most votes the survivors of one search may scatter in total; the
    /// histogram pass holds 16 bytes per vote and 32 per group of a
    /// survivor's postings in one run, so 48 bytes per vote at worst, when
    /// every posting is a group of its own. Exceeding it returns
    /// [`SearchError::TooManyVotes`]. Default `1 << 26` (about 1 GiB of
    /// votes, up to 3 GiB with the groups). The scratch keeps what a search
    /// grew; [`Scratch::shrink_to_fit`] gives it back.
    pub max_votes: usize,
    /// Threads one search may run on (`std`). A search that reads many
    /// postings cuts the reference ids into one range per thread and counts
    /// each range on a thread of its own, at most as many as the hardware
    /// has and only when each gets enough postings to be worth a thread;
    /// the calling thread is one of them. Results do not depend on it.
    /// Each thread beyond the first keeps buffers of its own in the
    /// [`Scratch`], as large as a scratch's on one thread. Default 1, which
    /// suits a server that runs as many searches at once as it has cores;
    /// more lowers the latency of a large search where cores are idle.
    /// Without `std` every search runs on the calling thread.
    pub threads: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            fanout: Some(64),
            min_total_votes: 1,
            offset_tolerance: 2,
            min_noise_bins: 10,
            max_votes: 1 << 26,
            threads: 1,
        }
    }
}

/// Votes of one survivor at one offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Bin {
    /// `reference_time − query_time` shared by every vote in the bin.
    pub delta: i64,
    /// Votes in the bin.
    pub votes: u32,
    /// Smallest query time among the votes.
    pub first_time: u32,
    /// Largest query time among the votes.
    pub last_time: u32,
}

impl Bin {
    /// A bin of `votes` at `delta` whose query times span `first_time` to
    /// `last_time`, for callers that reduce their own bins with
    /// [`Candidate::from_bins`].
    #[must_use]
    pub const fn new(delta: i64, votes: u32, first_time: u32, last_time: u32) -> Self {
        Self {
            delta,
            votes,
            first_time,
            last_time,
        }
    }
}

/// Statistics over a candidate's bins outside `±offset_tolerance` of its
/// best Δ.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct NoiseStats {
    /// Number of such bins.
    pub bins: usize,
    /// Mean votes per bin, or `0.0` below `min_noise_bins`.
    pub mean: f64,
    /// Population standard deviation of votes per bin, or `0.0` below
    /// `min_noise_bins`.
    pub std_dev: f64,
    /// Largest votes in one bin, or `0` below `min_noise_bins`.
    pub max: u32,
}

/// A survivor's best bin and noise statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Candidate {
    /// Reference id.
    pub reference: u32,
    /// Votes over all of the reference's bins.
    pub total_votes: u32,
    /// Δ of the best bin (most votes; ties go to the smaller Δ).
    pub delta: i64,
    /// Votes in the best bin.
    pub votes: u32,
    /// Smallest query time in the best bin.
    pub first_time: u32,
    /// Largest query time in the best bin.
    pub last_time: u32,
    /// Statistics over the other bins.
    pub noise: NoiseStats,
}

impl Candidate {
    /// Reduce a survivor's bins (sorted by Δ ascending) to a candidate.
    /// `None` if `bins` is empty.
    #[must_use]
    pub fn from_bins(
        reference: u32,
        total_votes: u32,
        bins: &[Bin],
        options: &SearchOptions,
    ) -> Option<Self> {
        let mut best = *bins.first()?;
        for b in &bins[1..] {
            if b.votes > best.votes || (b.votes == best.votes && b.delta < best.delta) {
                best = *b;
            }
        }
        let tol = u64::from(options.offset_tolerance);
        let noise = || bins.iter().filter(|x| x.delta.abs_diff(best.delta) > tol);
        let (mut n, mut sum, mut max) = (0usize, 0.0f64, 0u32);
        for x in noise() {
            n += 1;
            sum += f64::from(x.votes);
            max = max.max(x.votes);
        }
        let noise = if n > 0 && n >= options.min_noise_bins {
            let mean = sum / n as f64;
            let var = noise()
                .map(|x| {
                    let d = f64::from(x.votes) - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            NoiseStats {
                bins: n,
                mean,
                std_dev: libm::sqrt(var),
                max,
            }
        } else {
            NoiseStats {
                bins: n,
                ..NoiseStats::default()
            }
        };
        Some(Self {
            reference,
            total_votes,
            delta: best.delta,
            votes: best.votes,
            first_time: best.first_time,
            last_time: best.last_time,
            noise,
        })
    }
}

/// A reference that reached the vote floor, as [`Catalog::search_with`]
/// hands it to the visitor.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Survivor<'a> {
    /// Reference id.
    pub reference: u32,
    /// Votes over all of the reference's bins.
    pub total_votes: u32,
    /// The bins, sorted by Δ ascending; never empty.
    pub bins: &'a [Bin],
}

impl Survivor<'_> {
    /// The reduction [`Catalog::search`] applies: best bin and noise
    /// statistics.
    ///
    /// # Panics
    ///
    /// Never for a survivor this crate hands out: its bins are never empty.
    #[must_use]
    pub fn candidate(&self, options: &SearchOptions) -> Candidate {
        Candidate::from_bins(self.reference, self.total_votes, self.bins, options)
            .expect("a survivor has at least one bin")
    }
}

/// A vote floor proportional to the query: the smallest `v` with
/// `v / query_rows >= min_density`, but at least `min_votes`.
///
/// `query_rows` of `0`, or a `min_density` that is not positive, yields
/// `min_votes`; a product beyond `u32::MAX` yields `u32::MAX`. With
/// `min_votes = 16` and `min_density = 0.02`, a query of 4,200 rows needs
/// 84 votes and one of 100 rows needs 16.
#[must_use]
pub fn vote_floor(query_rows: usize, min_votes: u32, min_density: f64) -> u32 {
    if query_rows == 0 || min_density.is_nan() || min_density <= 0.0 {
        return min_votes;
    }
    let n = query_rows as f64;
    let want = libm::ceil(min_density * n);
    if want >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    let mut v = want.max(0.0) as u64;
    // Correct for rounding in the product: step down while the previous
    // integer still meets the density, then up until this one does.
    while v > 0 && (v - 1) as f64 / n >= min_density {
        v -= 1;
    }
    while (v as f64) / n < min_density {
        v += 1;
    }
    u32::try_from(v).unwrap_or(u32::MAX).max(min_votes)
}

/// Where one query hash's run lives: segment, query hash index, entries.
#[derive(Clone, Copy, Debug)]
struct RunLoc {
    segment: u32,
    hash: u32,
    lo: usize,
    hi: usize,
    /// Most postings any one reference has in the run (saturating).
    mult: u8,
    /// Query times of the hash: the vote weight of a posting.
    weight: u32,
}

/// Per-thread buffers a search reuses.
///
/// Holds one 32-bit counter, one bit and a sixty-fourth of a byte per
/// reference id of the largest catalog searched (a bit for the survivor
/// test of the histogram pass, and a byte per block of 64 counters, which
/// a search that reads few postings sets as it writes a block so that its
/// survivor scan can leave the others out), plus the run, group, vote and
/// bin buffers.
/// After the first search on a catalog, [`Catalog::search_with`] on a
/// query of similar size allocates nothing but the small vector of
/// segment handles it collects on every search; [`Catalog::search`] also
/// allocates its result. A search on several threads
/// ([`SearchOptions::threads`]) also starts its threads and keeps a
/// scratch per thread beyond the first in this one. One scratch serves any number of catalogs.
/// Searches leave it clean, including on an error, and so does a panic
/// inside a visitor. A search that panics part-way, which a damaged
/// catalog loaded without its checks can cause, leaves counters behind;
/// the scratch remembers that, and the next search clears every counter
/// before it counts.
#[derive(Default)]
pub struct Scratch {
    /// Vote total per reference during the totals pass; survivor index
    /// plus one per survivor during the histogram pass; zero otherwise.
    totals: Vec<u32>,
    /// One byte per block of [`SCAN_BLOCK`] counters, set by the totals
    /// pass of a search that tracks and cleared by the survivor scan: the
    /// blocks that hold a counter to look at.
    touched: Vec<u8>,
    /// One bit per reference, set for survivors during the histogram pass:
    /// an L1-resident membership test for the walk of a run.
    survivor_bits: Vec<u64>,
    runs: Vec<RunLoc>,
    groups: Vec<Group>,
    survivors: Vec<u32>,
    survivor_totals: Vec<u32>,
    slots: Vec<usize>,
    cursor: Vec<usize>,
    votes: Vec<(i64, u32)>,
    bins: Vec<Bin>,
    /// Set while a search may have counters, marks or survivor bits
    /// written, cleared once it has reset them: still set at the next
    /// search only if a search stopped by panicking in between.
    dirty: bool,
    /// The scratches of the threads of a search that runs on several.
    #[cfg(feature = "std")]
    workers: Vec<std::sync::Mutex<Scratch>>,
}

impl Scratch {
    /// An empty scratch; buffers grow on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Release every buffer; the next search grows them again.
    pub fn shrink_to_fit(&mut self) {
        *self = Self::default();
    }

    /// Heap bytes currently held.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.totals.capacity() * size_of::<u32>()
            + self.touched.capacity()
            + self.survivor_bits.capacity() * size_of::<u64>()
            + self.runs.capacity() * size_of::<RunLoc>()
            + self.groups.capacity() * size_of::<Group>()
            + self.survivors.capacity() * size_of::<u32>()
            + self.survivor_totals.capacity() * size_of::<u32>()
            + self.slots.capacity() * size_of::<usize>()
            + self.cursor.capacity() * size_of::<usize>()
            + self.votes.capacity() * size_of::<(i64, u32)>()
            + self.bins.capacity() * size_of::<Bin>()
            + self.workers_heap_bytes()
    }

    #[cfg(feature = "std")]
    fn workers_heap_bytes(&self) -> usize {
        self.workers.capacity() * size_of::<std::sync::Mutex<Scratch>>()
            + self
                .workers
                .iter()
                .map(|w| crate::par::lock(w).heap_bytes())
                .sum::<usize>()
    }

    #[cfg(not(feature = "std"))]
    #[expect(clippy::unused_self, reason = "the std build reads the workers")]
    fn workers_heap_bytes(&self) -> usize {
        0
    }

    fn fit(&mut self, n_refs: usize) {
        if self.dirty {
            self.totals.fill(0);
            self.touched.fill(0);
            self.survivor_bits.fill(0);
            self.dirty = false;
        }
        if self.totals.len() < n_refs {
            if self.totals.is_empty() {
                // Zeroed memory from the allocator: pages are taken as
                // counters in them are first written, so a thread of a
                // parallel search holds only the pages of its own range.
                self.totals = alloc::vec![0; n_refs];
            } else {
                self.totals.resize(n_refs, 0);
            }
            self.touched.resize(n_refs.div_ceil(SCAN_BLOCK), 0);
            self.survivor_bits.resize(n_refs.div_ceil(64), 0);
        }
    }
}

impl fmt::Debug for Scratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scratch")
            .field("reference_capacity", &self.totals.len())
            .field("vote_capacity", &self.votes.capacity())
            .field("heap_bytes", &self.heap_bytes())
            .finish_non_exhaustive()
    }
}

/// Postings in `runs`, saturating.
fn postings_of(runs: &[RunLoc]) -> usize {
    runs.iter()
        .fold(0usize, |n, run| n.saturating_add(run.hi - run.lo))
}

/// The votes the survivors will scatter, or the error when they pass
/// `options.max_votes`.
fn vote_count(survivor_totals: &[u32], options: &SearchOptions) -> Result<usize, SearchError> {
    let n_votes = survivor_totals
        .iter()
        .fold(0u64, |acc, &t| acc.saturating_add(u64::from(t)));
    if n_votes > options.max_votes as u64 {
        return Err(SearchError::TooManyVotes {
            votes: n_votes,
            max: options.max_votes,
        });
    }
    Ok(n_votes as usize)
}

/// The histogram pass for the survivors `scratch` holds: each survivor
/// gets a slice of the vote array sized by its exact total (a counting
/// sort keyed by survivor), its groups are found in the runs, and its
/// votes are scattered there. The counters and the survivor bits are
/// zero again at the end.
fn scatter_phase(
    segments: &[&Segment],
    query: &Query,
    fanout: Option<usize>,
    scratch: &mut Scratch,
    n_votes: usize,
) {
    let Scratch {
        totals,
        survivor_bits,
        runs,
        groups,
        survivors,
        survivor_totals,
        slots,
        cursor,
        votes,
        dirty,
        ..
    } = scratch;
    // The zeroed counters hold the survivor indices (plus one) until the
    // scatter is done.
    slots.clear();
    slots.push(0);
    *dirty = true;
    for (i, &r) in survivors.iter().enumerate() {
        totals[r as usize] = i as u32 + 1;
        survivor_bits[(r >> 6) as usize] |= 1u64 << (r & 63);
        slots.push(slots[i] + survivor_totals[i] as usize);
    }
    // The scatter writes every slot of `votes[..n_votes]` exactly once
    // (the cursors end at the slice ends), so stale contents need no zero
    // fill; only grow when a search needs more.
    if votes.len() < n_votes {
        votes.resize(n_votes, (0, 0));
    }
    let votes = &mut votes[..n_votes];
    cursor.clear();
    cursor.extend_from_slice(&slots[..survivors.len()]);
    simd::run(FindGroups {
        segments,
        fanout,
        runs,
        survivors,
        survivor_bits,
        index: totals,
        groups,
    });
    write_votes(segments, query, groups, cursor, votes);
    for &r in survivors.iter() {
        totals[r as usize] = 0;
        survivor_bits[(r >> 6) as usize] &= !(1u64 << (r & 63));
    }
    *dirty = false;
    debug_assert!(
        cursor.iter().zip(&slots[1..]).all(|(c, s)| c == s),
        "scatter count mismatch"
    );
}

/// Bin each survivor's votes and hand it to `visit`, in the order of
/// `survivors`.
fn emit<F: FnMut(Survivor<'_>)>(
    votes: &mut [(i64, u32)],
    slots: &[usize],
    survivors: &[u32],
    survivor_totals: &[u32],
    bins: &mut Vec<Bin>,
    visit: &mut F,
) {
    for (sv, &r) in survivors.iter().enumerate() {
        bins_of(&mut votes[slots[sv]..slots[sv + 1]], bins);
        visit(Survivor {
            reference: r,
            total_votes: survivor_totals[sv],
            bins,
        });
    }
}

/// Postings a search reads before it is cut across threads. Starting the
/// threads and adding their counters together costs a few tenths of a
/// millisecond; on the real clips that paid only for searches of 5 ms and
/// more on one thread, about two million postings.
/// Unit tests lower it so that their small catalogs take the parallel
/// path.
#[cfg(feature = "std")]
const PARALLEL_POSTINGS: usize = if cfg!(test) { 64 } else { 1 << 21 };

/// The threads a search over `runs` runs on, or `None` for one: what the
/// options ask for, at most the hardware's, and only for a search that
/// reads enough postings.
#[cfg(feature = "std")]
fn parallel_threads(options: &SearchOptions, runs: &[RunLoc]) -> Option<usize> {
    if options.threads <= 1 {
        return None;
    }
    let postings = postings_of(runs);
    if postings < PARALLEL_POSTINGS {
        return None;
    }
    // Every thread gets a quarter of the threshold's postings at least.
    let threads = options
        .threads
        .min(crate::par::hardware())
        .min(postings / (PARALLEL_POSTINGS / 4));
    (threads > 1).then_some(threads)
}

/// Query hashes per directory batch. The batch after the one being
/// resolved is loaded first, so its slots arrive while this one resolves:
/// plain loads whose results go nowhere, which overlap in the memory
/// system the way a prefetch would, on every target and without `unsafe`.
/// The batch sizes were swept: half of these cost 13 % of search time,
/// twice these gained nothing more.
const LOOKUP_BATCH: usize = 32;
/// Runs per totals batch, loaded one batch ahead the same way.
const RUN_BATCH: usize = 16;
/// Vote groups per batch of the histogram pass, loaded one batch ahead.
const GROUP_BATCH: usize = 32;
/// Counters per block of the survivor scan; a block is reduced with a
/// bitwise OR, which a 32-bit lane does without SSE4.1, and all-zero
/// blocks are skipped. In a search that records which blocks its totals
/// pass wrote, one byte of the scratch stands for each block.
const SCAN_BLOCK: usize = 64;
/// Fanouts up to this answer the fanout test from a fixed ring of the
/// last references instead of a second read of the run.
const RING: usize = 64;

/// Locate every query hash's run in every segment.
///
/// A lookup is one directory slot whose address depends only on the hash,
/// so the lookups are independent: each batch's slots are loaded while the
/// batch before it resolves. Hashes the query kept no times for cast no
/// votes and are not located.
fn locate(segments: &[&Segment], query: &Query, runs: &mut Vec<RunLoc>) {
    runs.clear();
    // A hash keeps no times only when the cap is zero, and then none does.
    if query.is_empty() {
        return;
    }
    let hashes = query.hashes();
    let n = hashes.len();
    for (si, seg) in segments.iter().enumerate() {
        seg.touch_directory(&hashes[..LOOKUP_BATCH.min(n)]);
        for k in (0..n).step_by(LOOKUP_BATCH) {
            let end = (k + LOOKUP_BATCH).min(n);
            seg.touch_directory(&hashes[end..(end + LOOKUP_BATCH).min(n)]);
            for (i, &hash) in hashes[k..end].iter().enumerate() {
                let qi = k + i;
                if let Some((lo, hi, mult)) = seg.range_mult(hash) {
                    runs.push(RunLoc {
                        segment: si as u32,
                        hash: qi as u32,
                        lo,
                        hi,
                        mult,
                        weight: u32::try_from(query.times(qi).len()).unwrap_or(u32::MAX),
                    });
                }
            }
        }
    }
}

/// How the totals pass records the blocks of counters it writes.
///
/// A search picks one of the two implementations once, before the pass,
/// and the pass is compiled for it: whether a counter store also marks
/// its block costs no branch per posting. What a store needs sits beside
/// the counters in [`Counters`], and is nothing at all for [`ScanAll`],
/// so that path's store is the bare counter add.
trait Marks: Copy {
    /// What marks the blocks written, if anything.
    type Store<'a>;

    /// Note that the counter of reference `r` is written.
    fn mark(store: &mut Self::Store<'_>, r: u32);
}

/// Record nothing: the survivor scan reads every counter, which on a
/// catalog of few references is cheaper than a byte store per posting.
#[derive(Clone, Copy)]
struct ScanAll;

impl Marks for ScanAll {
    type Store<'a> = ();

    #[inline(always)]
    fn mark((): &mut Self::Store<'_>, _: u32) {}
}

/// Record the block of [`SCAN_BLOCK`] counters that holds reference `r`.
#[derive(Clone, Copy)]
struct MarkBlocks;

impl Marks for MarkBlocks {
    type Store<'a> = &'a mut [u8];

    #[inline(always)]
    fn mark(touched: &mut Self::Store<'_>, r: u32) {
        touched[r as usize / SCAN_BLOCK] = 1;
    }
}

/// The counters of the totals pass, and the record of which blocks of
/// them were written. A pass borrows the two halves once with
/// [`Counters::parts`] and then writes both together, so the record stays
/// in step with the counters whichever way the pass counts.
struct Counters<'a, M: Marks> {
    /// One vote total per reference.
    totals: &'a mut [u32],
    /// What marks the blocks written: one byte per [`SCAN_BLOCK`]
    /// counters, or nothing.
    marks: M::Store<'a>,
}

/// The two halves of a [`Counters`], borrowed separately so a hot loop
/// loads each pointer once.
struct CounterParts<'b, 'a, M: Marks> {
    totals: &'b mut [u32],
    marks: &'b mut M::Store<'a>,
}

impl<'a, M: Marks> Counters<'a, M> {
    /// Borrow the counters and the record of written blocks apart.
    fn parts<'b>(&'b mut self) -> CounterParts<'b, 'a, M> {
        CounterParts {
            totals: self.totals,
            marks: &mut self.marks,
        }
    }
}

/// Totals pass: each run adds `query_times × min(count, fanout)` to each
/// reference in it.
///
/// When no counter can overflow (every run's full contribution fits in
/// 32 bits, which is every practical query) the runs are decoded eight
/// references at a time where the CPU can (the `simd` feature), as a
/// stream otherwise, and counted with plain adds; when one can, they are
/// counted with saturating adds, so an overflow stays detectable
/// afterwards.
fn totals_pass<'a, M: Marks>(
    counters: Counters<'a, M>,
    segments: &'a [&'a Segment],
    fanout: Option<usize>,
    runs: &'a [RunLoc],
) {
    // No reference has more than `mult` postings in a run (unless that is
    // saturated), and no more than `fanout` of them vote, so a counter
    // gains at most this much from each run.
    let cap = fanout.unwrap_or(usize::MAX);
    let bound = runs.iter().fold(0u64, |b, run| {
        let most = if run.mult == u8::MAX {
            usize::MAX
        } else {
            usize::from(run.mult)
        };
        let per_reference = (run.hi - run.lo).min(cap).min(most) as u64;
        b.saturating_add(per_reference.saturating_mul(u64::from(run.weight)))
    });
    if u32::try_from(bound).is_ok() {
        simd::run(CountPlain {
            segments,
            fanout,
            runs,
            counters,
        });
    } else {
        count_direct(segments, fanout, runs, counters);
    }
}

/// The body of [`for_each_run`], and of its copy inside a block-decoding
/// pass.
macro_rules! for_each_run {
    ($segments:expr, $runs:expr, $reach:expr, $visit:expr) => {{
        let (segments, runs): (&[&Segment], &[RunLoc]) = ($segments, $runs);
        let mut visit = $visit;
        let n = runs.len();
        let touch = |from: usize, to: usize| {
            let acc = runs[from.min(n)..to.min(n)].iter().fold(0u32, |acc, run| {
                acc ^ segments[run.segment as usize].touch_refs(run.lo, run.hi, $reach)
            });
            core::hint::black_box(acc);
        };
        touch(0, RUN_BATCH);
        for k in (0..n).step_by(RUN_BATCH) {
            touch(k + RUN_BATCH, k + 2 * RUN_BATCH);
            for run in &runs[k..(k + RUN_BATCH).min(n)] {
                visit(
                    segments[run.segment as usize].refs(),
                    run.lo,
                    run.hi,
                    run.weight,
                    run.mult,
                );
            }
        }
    }};
}

/// Visit every run's references with its vote weight. Each batch of runs
/// is loaded while the batch before it is visited.
fn for_each_run(
    segments: &[&Segment],
    runs: &[RunLoc],
    visit: impl FnMut(&Packed, usize, usize, u32, u8),
) {
    // One value at a time reads 8 bytes from the byte a value starts in.
    for_each_run!(segments, runs, Packed::SLACK, visit);
}

/// The plain totals pass over one run: `$whole` reads a run in which
/// every posting counts, which is how the stream and the block passes
/// differ.
///
/// Inside a run a reference's postings are contiguous, so entry `i` is its
/// reference's `fanout + 1`th or later posting exactly when entry
/// `i − fanout` has the same reference; a ring of the last [`RING`]
/// references answers that without a second read of the run.
macro_rules! count_run {
    ($fanout:expr, $whole:expr, $totals:expr, $marks:expr, $each:expr) => {{
        let (fanout, whole): (Option<usize>, usize) = ($fanout, $whole);
        let each = $each;
        move |refs: &Packed, lo: usize, hi: usize, qc: u32, mult: u8| match fanout {
            // Every posting counts, which on real catalogs is nearly every
            // run.
            Some(_) if usize::from(mult) <= whole => {
                each(refs, lo, hi, qc, &mut *$totals, &mut *$marks)
            }
            Some(f) if f <= RING => {
                let head = lo + (hi - lo).min(f);
                let mut ring = [0u32; RING];
                let mut i = 0usize;
                refs.for_each(lo, head, |r| {
                    ring[i % RING] = r;
                    i += 1;
                    $totals[r as usize] += qc;
                    M::mark(&mut *$marks, r);
                });
                refs.for_each(head, hi, |r| {
                    let counts = ring[(i - f) % RING] != r;
                    ring[i % RING] = r;
                    i += 1;
                    $totals[r as usize] += qc * u32::from(counts);
                    M::mark(&mut *$marks, r);
                });
            }
            Some(f) => {
                let head = lo + (hi - lo).min(f);
                refs.for_each(lo, head, |r| {
                    $totals[r as usize] += qc;
                    M::mark(&mut *$marks, r);
                });
                for i in head..hi {
                    let r = refs.get(i);
                    $totals[r as usize] += qc * u32::from(refs.get(i - f) != r);
                    M::mark(&mut *$marks, r);
                }
            }
            None => each(refs, lo, hi, qc, &mut *$totals, &mut *$marks),
        }
    }};
}

/// The largest multiplicity that proves no reference of a run reaches the
/// fanout. A saturated multiplicity is only a lower bound, so it proves
/// nothing whatever the fanout.
fn whole_below(fanout: Option<usize>) -> usize {
    fanout.map_or(0, |f| f.min(usize::from(u8::MAX) - 1))
}

/// The totals pass with plain adds.
struct CountPlain<'a, M: Marks> {
    segments: &'a [&'a Segment],
    fanout: Option<usize>,
    runs: &'a [RunLoc],
    counters: Counters<'a, M>,
}

impl<M: Marks> Pass for CountPlain<'_, M> {
    type Out = ();

    /// Without a kernel this is [`count_plain`], a function of its own
    /// whose code does not depend on this module; with one, the same pass
    /// with whole runs decoded in blocks, inlined into the caller that
    /// enabled the kernel's instructions.
    #[inline(always)]
    fn run<D: Decoder>(self, decoder: D) {
        let Self {
            segments,
            fanout,
            runs,
            mut counters,
        } = self;
        if !D::BLOCKS {
            return count_plain(segments, fanout, runs, counters);
        }
        let CounterParts { totals, marks } = counters.parts();
        let each = |refs: &Packed,
                    lo: usize,
                    hi: usize,
                    qc: u32,
                    totals: &mut [u32],
                    marks: &mut M::Store<'_>| {
            refs.for_each_in(
                lo,
                hi,
                decoder,
                // Forced in with the block loop around it: see
                // `Packed::for_each_in`.
                #[inline(always)]
                |r| {
                    totals[r as usize] += qc;
                    M::mark(marks, r);
                },
            );
        };
        // A block reads a window from the byte its first value starts in.
        for_each_run!(
            segments,
            runs,
            simd::WINDOW - 1,
            count_run!(fanout, whole_below(fanout), totals, marks, each)
        );
    }
}

/// Count every run with plain adds, decoding each run as a stream.
fn count_plain<M: Marks>(
    segments: &[&Segment],
    fanout: Option<usize>,
    runs: &[RunLoc],
    mut counters: Counters<'_, M>,
) {
    let CounterParts { totals, marks } = counters.parts();
    let each = |refs: &Packed,
                lo: usize,
                hi: usize,
                qc: u32,
                totals: &mut [u32],
                marks: &mut M::Store<'_>| {
        refs.for_each(lo, hi, |r| {
            totals[r as usize] += qc;
            M::mark(marks, r);
        });
    };
    for_each_run(
        segments,
        runs,
        count_run!(fanout, whole_below(fanout), totals, marks, each),
    );
}

/// Count every run straight into the counters, saturating at `u32::MAX`
/// so an overflow is detectable afterwards. Entry `i` counts unless entry
/// `i − fanout` has the same reference.
fn count_direct<M: Marks>(
    segments: &[&Segment],
    fanout: Option<usize>,
    runs: &[RunLoc],
    mut counters: Counters<'_, M>,
) {
    let CounterParts { totals, marks } = counters.parts();
    for_each_run(segments, runs, |refs, lo, hi, qc, _| match fanout {
        Some(f) => {
            let head = lo + (hi - lo).min(f);
            for i in lo..head {
                let r = refs.get(i);
                let t = &mut totals[r as usize];
                *t = t.saturating_add(qc);
                M::mark(marks, r);
            }
            for i in head..hi {
                let r = refs.get(i);
                let v = qc * u32::from(refs.get(i - f) != r);
                let t = &mut totals[r as usize];
                *t = t.saturating_add(v);
                M::mark(marks, r);
            }
        }
        None => {
            for i in lo..hi {
                let r = refs.get(i);
                let t = &mut totals[r as usize];
                *t = t.saturating_add(qc);
                M::mark(marks, r);
            }
        }
    });
}

/// One survivor's postings in one run, `fanout`-capped: entries
/// `lo..hi` of `segment`, each paired with every time of query hash
/// `hash`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Group {
    segment: u32,
    hash: u32,
    survivor: u32,
    lo: usize,
    hi: usize,
}

/// The first half of the histogram pass, as [`find_groups`] does it.
struct FindGroups<'a> {
    segments: &'a [&'a Segment],
    fanout: Option<usize>,
    runs: &'a [RunLoc],
    survivors: &'a [u32],
    survivor_bits: &'a [u64],
    index: &'a [u32],
    groups: &'a mut Vec<Group>,
}

impl Pass for FindGroups<'_> {
    type Out = ();

    #[inline(always)]
    fn run<D: Decoder>(self, decoder: D) {
        find_groups(
            self.segments,
            self.fanout,
            self.runs,
            self.survivors,
            self.survivor_bits,
            self.index,
            self.groups,
            decoder,
        );
    }
}

/// Longest range a search for a reference reads in blocks, where the
/// decoder has a kernel: a block answers for eight entries without a
/// branch that depends on them, which beats halving up to three blocks.
/// Measured on runs of 17, 48 and 136 postings: reading runs of 17 this
/// way took a tenth off the pass; reading up to 64 entries, or halving a
/// longer range down to 32 first, cost a sixth to a half more.
const SCAN: usize = 24;

/// First entry in `lo..hi` whose reference is not below `r`, or `hi`; the
/// references in the range are ascending.
#[inline(always)]
fn first_not_below<D: Decoder>(
    refs: &Packed,
    mut lo: usize,
    hi: usize,
    r: u32,
    decoder: D,
) -> usize {
    if hi - lo >= Packed::BLOCK_MIN
        && hi - lo <= SCAN
        && let Some(mut blocks) = refs.blocks(lo, decoder)
    {
        while lo < hi {
            let Some(block) = blocks.next() else {
                return refs.partition_point(lo, hi, |x| x < r);
            };
            let n = (hi - lo).min(LANES);
            // Ascending, so the entries below `r` are the block's
            // first ones and their count is where `r` belongs.
            let below = block
                .iter()
                .enumerate()
                .filter(|&(lane, &x)| lane < n && x < r)
                .count();
            lo += below;
            if below < n {
                break;
            }
        }
        return lo;
    }
    refs.partition_point(lo, hi, |x| x < r)
}

/// First half of the histogram pass: find every survivor's group in every
/// run, in run order then reference order.
///
/// A run is grouped by reference, ascending, so a run with few survivors
/// relative to its length is searched for each survivor, and a run with
/// many is walked once with a bitmap membership test per group (`index`
/// then holds each survivor's index plus one). A group is no longer than
/// the run's multiplicity, so its end is not searched for when that is
/// one. Only the references are read here, through `decoder`, and they
/// are still cached from the totals pass.
#[expect(clippy::too_many_arguments, reason = "the fields of `FindGroups`")]
#[inline(always)]
fn find_groups<D: Decoder>(
    segments: &[&Segment],
    fanout: Option<usize>,
    runs: &[RunLoc],
    survivors: &[u32],
    survivor_bits: &[u64],
    index: &[u32],
    groups: &mut Vec<Group>,
    decoder: D,
) {
    groups.clear();
    let n_surv = survivors.len();
    let is_survivor = |r: u32| survivor_bits[(r >> 6) as usize] & (1u64 << (r & 63)) != 0;
    for run in runs {
        let refs = segments[run.segment as usize].refs();
        let (run_lo, run_hi) = (run.lo, run.hi);
        let len = run_hi - run_lo;
        let mut push = |i: usize, j: usize, survivor: u32| {
            let end = fanout.map_or(j, |f| j.min(i.saturating_add(f)));
            groups.push(Group {
                segment: run.segment,
                hash: run.hash,
                survivor,
                lo: i,
                hi: end,
            });
        };
        // Probing costs about `survivors × log2(len)`; walking costs `len`.
        let probe_cost = n_surv.saturating_mul((usize::BITS - len.leading_zeros()) as usize + 1);
        if probe_cost < len {
            // The most postings a group can have; a saturated multiplicity
            // is only a lower bound.
            let longest = if run.mult == u8::MAX {
                len
            } else {
                usize::from(run.mult)
            };
            let mut lo = run_lo;
            for (sv, &r) in survivors.iter().enumerate() {
                let i = first_not_below(refs, lo, run_hi, r, decoder);
                if i == run_hi {
                    break;
                }
                lo = i;
                if refs.get(i) != r {
                    continue;
                }
                let j = if longest == 1 {
                    i + 1
                } else {
                    let end = run_hi.min(i.saturating_add(longest));
                    refs.partition_point(i + 1, end, |x| x == r)
                };
                push(i, j, sv as u32);
                lo = j;
            }
        } else if run.mult == 1 {
            // Every posting is its own group: a membership test per
            // posting, no search for the group's end.
            let mut i = run_lo;
            refs.for_each_in(run_lo, run_hi, decoder, |r| {
                if is_survivor(r) {
                    push(i, i + 1, index[r as usize] - 1);
                }
                i += 1;
            });
        } else {
            // One read of the run: a group ends where the reference
            // changes.
            let (mut i, mut start, mut current) = (run_lo, run_lo, refs.get(run_lo));
            refs.for_each_in(run_lo, run_hi, decoder, |r| {
                if r != current {
                    if is_survivor(current) {
                        push(start, i, index[current as usize] - 1);
                    }
                    (start, current) = (i, r);
                }
                i += 1;
            });
            if is_survivor(current) {
                push(start, run_hi, index[current as usize] - 1);
            }
        }
    }
}

/// Second half of the histogram pass: write each group's `(Δ, query_time)`
/// votes at its survivor's cursor. The times of a group are the first
/// cold lines of the search, so each batch of groups is loaded while the
/// batch before it is written.
fn write_votes(
    segments: &[&Segment],
    query: &Query,
    groups: &[Group],
    cursor: &mut [usize],
    votes: &mut [(i64, u32)],
) {
    let n = groups.len();
    let touch = |from: usize, to: usize| {
        let acc = groups[from.min(n)..to.min(n)].iter().fold(0u32, |acc, g| {
            acc ^ segments[g.segment as usize].touch_time(g.lo)
        });
        core::hint::black_box(acc);
    };
    touch(0, GROUP_BATCH);
    for k in (0..n).step_by(GROUP_BATCH) {
        touch(k + GROUP_BATCH, k + 2 * GROUP_BATCH);
        for g in &groups[k..(k + GROUP_BATCH).min(n)] {
            let times = segments[g.segment as usize].times();
            let qts = query.times(g.hash as usize);
            let c = &mut cursor[g.survivor as usize];
            for t in times.iter(g.lo..g.hi) {
                for &qt in qts {
                    votes[*c] = (i64::from(t) - i64::from(qt), qt);
                    *c += 1;
                }
            }
        }
    }
}

/// Sort one survivor's votes by Δ and merge equal Δs into bins.
fn bins_of(votes: &mut [(i64, u32)], bins: &mut Vec<Bin>) {
    votes.sort_unstable_by_key(|x| x.0);
    bins.clear();
    let mut i = 0;
    while i < votes.len() {
        let delta = votes[i].0;
        let (mut n, mut first, mut last) = (0u32, u32::MAX, 0u32);
        while i < votes.len() && votes[i].0 == delta {
            n += 1;
            first = first.min(votes[i].1);
            last = last.max(votes[i].1);
            i += 1;
        }
        bins.push(Bin {
            delta,
            votes: n,
            first_time: first,
            last_time: last,
        });
    }
}

impl Catalog {
    /// Search for references that reach the vote floor and return one
    /// [`Candidate`] per survivor, in ascending reference order.
    ///
    /// # Errors
    ///
    /// [`SearchError`] when a reference's vote total overflows or the
    /// survivors would scatter more than `options.max_votes` votes. Both
    /// are detected after the totals pass, before any histogram work.
    pub fn search(
        &self,
        query: &Query,
        options: &SearchOptions,
        scratch: &mut Scratch,
    ) -> Result<Vec<Candidate>, SearchError> {
        let mut out = Vec::new();
        self.search_with(query, options, scratch, |s| out.push(s.candidate(options)))?;
        Ok(out)
    }

    /// Like [`Catalog::search`], but hand each [`Survivor`] to `visit`
    /// instead of reducing it, in ascending reference order.
    /// [`Survivor::candidate`] is the reduction `search` applies.
    ///
    /// # Errors
    ///
    /// As [`Catalog::search`].
    pub fn search_with<F>(
        &self,
        query: &Query,
        options: &SearchOptions,
        scratch: &mut Scratch,
        visit: F,
    ) -> Result<(), SearchError>
    where
        F: FnMut(Survivor<'_>),
    {
        self.search_tracked(query, options, scratch, None, visit)
    }

    /// [`Catalog::search_with`], with `track` choosing how the survivor
    /// scan finds the counters the totals pass wrote: `Some(true)` for the
    /// record of written blocks, `Some(false)` for a scan of every
    /// counter, `None` for the choice the search makes for itself once
    /// its runs are located. Tests pass the first two to compare the paths
    /// on one catalog.
    fn search_tracked<F>(
        &self,
        query: &Query,
        options: &SearchOptions,
        scratch: &mut Scratch,
        track: Option<bool>,
        mut visit: F,
    ) -> Result<(), SearchError>
    where
        F: FnMut(Survivor<'_>),
    {
        let fanout = options.fanout;
        // A fanout of zero lets no posting vote; the passes below assume
        // at least one posting per `(hash, reference)` pair counts.
        if fanout == Some(0) {
            return Ok(());
        }
        let n_refs = self.reference_count();
        let floor = options.min_total_votes.max(1);
        // Segments are shared handles; the passes take a view of them. The
        // small vector is the only per-search allocation beyond the scratch.
        let segments: Vec<&Segment> = self.segments().collect();
        let segments: &[&Segment] = &segments;
        scratch.fit(n_refs);
        locate(segments, query, &mut scratch.runs);
        #[cfg(feature = "std")]
        if let Some(threads) = parallel_threads(options, &scratch.runs) {
            return self.search_parallel(segments, query, options, scratch, threads, floor, visit);
        }
        self.count_phase(segments, fanout, scratch, n_refs, track, floor)?;
        if scratch.survivors.is_empty() {
            return Ok(());
        }
        let n_votes = vote_count(&scratch.survivor_totals, options)?;
        scatter_phase(segments, query, fanout, scratch, n_votes);
        let Scratch {
            survivors,
            survivor_totals,
            slots,
            votes,
            bins,
            ..
        } = scratch;
        emit(
            &mut votes[..n_votes],
            slots,
            survivors,
            survivor_totals,
            bins,
            &mut visit,
        );
        Ok(())
    }

    /// The totals pass and the survivor scan over the runs `scratch` has
    /// located, by the path `track` chooses (`None` for the search's own
    /// choice): the survivors and their totals are left in the scratch,
    /// and the counters are zero again, error or not.
    fn count_phase(
        &self,
        segments: &[&Segment],
        fanout: Option<usize>,
        scratch: &mut Scratch,
        n_refs: usize,
        track: Option<bool>,
        floor: u32,
    ) -> Result<(), SearchError> {
        let Scratch {
            totals,
            touched,
            runs,
            survivors,
            survivor_totals,
            dirty,
            ..
        } = scratch;
        let totals = &mut totals[..n_refs];
        let n_blocks = n_refs.div_ceil(SCAN_BLOCK);
        // The record costs a byte store per posting and spares the scan
        // the blocks nothing wrote. With as many postings as blocks or
        // more, nearly every block is written and the scan has all of them
        // to read anyway, so the stores are pure cost; with fewer, the
        // blocks left out are worth more than the stores. A short query
        // over a catalog of many references records; a long one over the
        // same catalog does not.
        let track = track.unwrap_or_else(|| postings_of(runs) < n_blocks);
        *dirty = true;
        let taken = self.count_and_take(
            segments,
            fanout,
            runs,
            totals,
            touched,
            n_blocks,
            track,
            floor,
            survivors,
            survivor_totals,
        );
        // The counters are clean again, whether or not the pass found an
        // overflow.
        *dirty = false;
        taken
    }

    /// [`Catalog::search_tracked`] with its totals pass on `threads`
    /// threads, for runs already located in `scratch`.
    ///
    /// The runs are cut into one share per thread of about equal postings,
    /// and each thread counts its share with the totals pass of a search on
    /// one thread into counters of its own, marking the blocks it writes.
    /// The calling thread then adds the marked blocks into the scratch's
    /// counters, saturating so that an overflow is caught as it is on one
    /// thread, and the survivor scan, the histogram pass and the visits run
    /// as they do there. A run is never cut, so no thread looks anything up
    /// to split the work.
    #[cfg(feature = "std")]
    #[expect(clippy::too_many_arguments, reason = "the parts of a search")]
    fn search_parallel<F>(
        &self,
        segments: &[&Segment],
        query: &Query,
        options: &SearchOptions,
        scratch: &mut Scratch,
        threads: usize,
        floor: u32,
        mut visit: F,
    ) -> Result<(), SearchError>
    where
        F: FnMut(Survivor<'_>),
    {
        use crate::par::{lock, run_tasks};
        let n_refs = self.reference_count();
        let n_blocks = n_refs.div_ceil(SCAN_BLOCK);
        let fanout = options.fanout;
        while scratch.workers.len() < threads {
            scratch
                .workers
                .push(std::sync::Mutex::new(Scratch::default()));
        }
        // Share `t` is `runs[cuts[t]..cuts[t + 1]]`.
        let runs: &[RunLoc] = &scratch.runs;
        let total = postings_of(runs);
        let mut cuts = Vec::with_capacity(threads + 1);
        cuts.push(0);
        let mut seen = 0usize;
        for (i, run) in runs.iter().enumerate() {
            if cuts.len() < threads && seen >= total / threads * cuts.len() {
                cuts.push(i);
            }
            seen += run.hi - run.lo;
        }
        while cuts.len() <= threads {
            cuts.push(runs.len());
        }
        let workers = &scratch.workers[..threads];
        run_tasks(threads, threads, |t| {
            let mut worker = lock(&workers[t]);
            let worker = &mut *worker;
            worker.fit(n_refs);
            worker.dirty = true;
            totals_pass(
                Counters::<MarkBlocks> {
                    totals: &mut worker.totals[..n_refs],
                    marks: &mut worker.touched[..n_blocks],
                },
                segments,
                fanout,
                &runs[cuts[t]..cuts[t + 1]],
            );
        });
        let Scratch {
            totals,
            touched,
            survivors,
            survivor_totals,
            dirty,
            workers,
            ..
        } = scratch;
        let (totals, touched) = (&mut totals[..n_refs], &mut touched[..n_blocks]);
        *dirty = true;
        for worker in &workers[..threads] {
            let mut worker = lock(worker);
            let Scratch {
                totals: theirs,
                touched: marked,
                dirty: their_dirty,
                ..
            } = &mut *worker;
            for (b, mark) in marked[..n_blocks].iter_mut().enumerate() {
                if *mark == 0 {
                    continue;
                }
                *mark = 0;
                touched[b] = 1;
                let lo = b * SCAN_BLOCK;
                let hi = (lo + SCAN_BLOCK).min(n_refs);
                for (sum, count) in totals[lo..hi].iter_mut().zip(&mut theirs[lo..hi]) {
                    *sum = sum.saturating_add(*count);
                    *count = 0;
                }
            }
            *their_dirty = false;
        }
        let taken = self.take_touched(totals, touched, floor, survivors, survivor_totals);
        *dirty = false;
        taken?;
        if scratch.survivors.is_empty() {
            return Ok(());
        }
        let n_votes = vote_count(&scratch.survivor_totals, options)?;
        scatter_phase(segments, query, fanout, scratch, n_votes);
        let Scratch {
            survivors,
            survivor_totals,
            slots,
            votes,
            bins,
            ..
        } = scratch;
        emit(
            &mut votes[..n_votes],
            slots,
            survivors,
            survivor_totals,
            bins,
            &mut visit,
        );
        Ok(())
    }

    /// Fill the counters from the runs and take the survivors out of them,
    /// by the path `track` chooses. The counters are zero at entry and at
    /// exit, whether the search goes on to the histogram pass or stops
    /// here with an error.
    #[expect(clippy::too_many_arguments, reason = "the fields of the scratch")]
    fn count_and_take(
        &self,
        segments: &[&Segment],
        fanout: Option<usize>,
        runs: &[RunLoc],
        totals: &mut [u32],
        touched: &mut [u8],
        n_blocks: usize,
        track: bool,
        floor: u32,
        survivors: &mut Vec<u32>,
        survivor_totals: &mut Vec<u32>,
    ) -> Result<(), SearchError> {
        // The path is chosen once, and the pass is compiled for it, so
        // marking a block costs no branch per posting.
        if track {
            totals_pass(
                Counters::<MarkBlocks> {
                    totals,
                    marks: &mut touched[..n_blocks],
                },
                segments,
                fanout,
                runs,
            );
            self.take_touched(
                totals,
                &mut touched[..n_blocks],
                floor,
                survivors,
                survivor_totals,
            )
        } else {
            totals_pass(
                Counters::<ScanAll> { totals, marks: () },
                segments,
                fanout,
                runs,
            );
            self.take_survivors(totals, floor, survivors, survivor_totals)
        }
    }

    /// Move the survivors out of the counters, reading every block: the
    /// references that were touched, are live and reach `floor`, in
    /// ascending order, each with its total. The scan also resets the
    /// counters, so the scratch is clean from here on. A block's bitwise
    /// OR decides whether it holds anything to look at: every counter is
    /// at most the OR of its block, and the OR is zero exactly when the
    /// block is all zero.
    fn take_survivors(
        &self,
        totals: &mut [u32],
        floor: u32,
        survivors: &mut Vec<u32>,
        survivor_totals: &mut Vec<u32>,
    ) -> Result<(), SearchError> {
        survivors.clear();
        survivor_totals.clear();
        let mut scratch = SurvivorScan::new(survivors, survivor_totals);
        for (b, block) in totals.chunks_mut(SCAN_BLOCK).enumerate() {
            self.drain_block(b, block, floor, &mut scratch);
        }
        scratch.finish()
    }

    /// Move the survivors out of the counters, reading the blocks the
    /// totals pass marked and clearing each mark: the counters outside
    /// those blocks were zero before the pass and are zero now, so they
    /// hold no survivor.
    fn take_touched(
        &self,
        totals: &mut [u32],
        touched: &mut [u8],
        floor: u32,
        survivors: &mut Vec<u32>,
        survivor_totals: &mut Vec<u32>,
    ) -> Result<(), SearchError> {
        survivors.clear();
        survivor_totals.clear();
        let mut scratch = SurvivorScan::new(survivors, survivor_totals);
        // Eight mark bytes per word, read as one little-endian `u64`: a
        // zero word skips eight blocks, and `set` names the marked bytes
        // of a word that has any. A bit index names a byte, so a set bit
        // `k` marks block `8 * w + k / 8`.
        let whole = touched.len() / size_of::<u64>() * size_of::<u64>();
        let (words, rest) = touched.split_at_mut(whole);
        for (w, word) in words.as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let mut set = u64::from_le_bytes(*word);
            if set == 0 {
                continue;
            }
            word.fill(0);
            while set != 0 {
                let bit = set.trailing_zeros() as usize;
                set &= set - 1;
                self.scan_block(w * 8 + bit / 8, totals, floor, &mut scratch);
            }
        }
        // The bytes after the last whole word are the blocks after it, one
        // byte per block.
        for (w, byte) in rest.iter_mut().enumerate() {
            if *byte == 0 {
                continue;
            }
            *byte = 0;
            self.scan_block(whole + w, totals, floor, &mut scratch);
        }
        debug_assert!(touched.iter().all(|&t| t == 0), "marks left behind");
        scratch.finish()
    }

    /// Take and reset the survivors of block `b`, which the totals pass
    /// marked: its counters are read, the ones at the floor collected, and
    /// the block zeroed. Only the block a mark names is ever scanned, so a
    /// block past the counters is a wrong block number, not an empty
    /// block: the slice is taken from the counters' start, and only the
    /// end of the last block is cut to their length.
    fn scan_block(&self, b: usize, totals: &mut [u32], floor: u32, scan: &mut SurvivorScan<'_>) {
        let lo = b * SCAN_BLOCK;
        assert!(lo < totals.len(), "block {b} is past the counters");
        let hi = (lo + SCAN_BLOCK).min(totals.len());
        self.drain_block(b, &mut totals[lo..hi], floor, scan);
    }

    /// Collect the survivors of block `b` and zero it. A full block is
    /// an array of known length, so the zeroing is plain stores.
    #[inline]
    fn drain_block(&self, b: usize, block: &mut [u32], floor: u32, scan: &mut SurvivorScan<'_>) {
        if block.len() == SCAN_BLOCK {
            let block: &mut [u32; SCAN_BLOCK] = block.try_into().expect("a full block");
            let any = block.iter().fold(0u32, |m, &t| m | t);
            if any != 0 {
                self.collect_block(b, block, any, floor, scan);
                *block = [0; SCAN_BLOCK];
            }
        } else {
            let any = block.iter().fold(0u32, |m, &t| m | t);
            if any != 0 {
                self.collect_block(b, block, any, floor, scan);
                block.fill(0);
            }
        }
    }

    /// Append the survivors of block `b`, whose counters OR to `any`.
    fn collect_block(
        &self,
        b: usize,
        block: &mut [u32],
        any: u32,
        floor: u32,
        scan: &mut SurvivorScan<'_>,
    ) {
        if any < floor {
            return;
        }
        for (i, &t) in block.iter().enumerate() {
            if t >= floor {
                let r = (b * SCAN_BLOCK + i) as u32;
                // A removed reference is skipped before anything else: its
                // postings are nobody's problem.
                if !self.is_live(r) {
                    continue;
                }
                if t == u32::MAX {
                    scan.overflow.get_or_insert(r);
                } else {
                    scan.survivors.push(r);
                    scan.totals.push(t);
                }
            }
        }
    }
}

/// The survivors of one scan, and the first reference whose counter
/// overflowed.
struct SurvivorScan<'a> {
    survivors: &'a mut Vec<u32>,
    totals: &'a mut Vec<u32>,
    overflow: Option<u32>,
}

impl<'a> SurvivorScan<'a> {
    fn new(survivors: &'a mut Vec<u32>, totals: &'a mut Vec<u32>) -> Self {
        Self {
            survivors,
            totals,
            overflow: None,
        }
    }

    fn finish(self) -> Result<(), SearchError> {
        match self.overflow {
            Some(reference) => Err(SearchError::TotalsOverflow { reference }),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Posting;
    use alloc::vec;

    fn bin(delta: i64, votes: u32) -> Bin {
        Bin {
            delta,
            votes,
            first_time: 0,
            last_time: 0,
        }
    }

    #[test]
    fn vote_floor_matches_the_documented_values() {
        assert_eq!(vote_floor(0, 16, 0.02), 16);
        assert_eq!(vote_floor(4_200, 16, 0.02), 84);
        assert_eq!(vote_floor(100, 16, 0.02), 16);
        assert_eq!(vote_floor(1_000, 16, 0.02), 20);
        assert_eq!(vote_floor(50, 1, 0.02), 1);
        assert_eq!(vote_floor(51, 1, 0.02), 2);
        assert_eq!(vote_floor(10, 3, 0.0), 3);
        assert_eq!(vote_floor(10, 3, f64::NAN), 3);
        assert_eq!(vote_floor(10, 3, -1.0), 3);
        assert_eq!(vote_floor(10, 0, 1.0), 10);
        assert_eq!(vote_floor(10, 0, 5.0), 50);
        assert_eq!(vote_floor(10, 0, f64::INFINITY), u32::MAX);
        assert_eq!(vote_floor(10, 0, 1e30), u32::MAX);
        assert_eq!(vote_floor(usize::MAX, 0, 1.0), u32::MAX);
        // Every answer is the smallest integer meeting the density.
        for rows in 1..300usize {
            for density in [0.01, 0.02, 0.1, 0.333, 0.5] {
                let v = vote_floor(rows, 0, density);
                assert!(f64::from(v) / rows as f64 >= density);
                assert!(v == 0 || f64::from(v - 1) / (rows as f64) < density);
            }
        }
    }

    #[test]
    fn best_bin_ties_go_to_the_smaller_delta() {
        let o = SearchOptions::default();
        let bins = [bin(-5, 2), bin(0, 3), bin(4, 3), bin(100, 1)];
        let c = Candidate::from_bins(7, 9, &bins, &o).unwrap();
        assert_eq!((c.reference, c.total_votes, c.delta, c.votes), (7, 9, 0, 3));
        // Bins at -5, 4 and 100 lie outside ±2 of 0; below the 10-bin
        // minimum they are counted but not summarised.
        assert_eq!(
            c.noise,
            NoiseStats {
                bins: 3,
                ..NoiseStats::default()
            }
        );
        assert!(Candidate::from_bins(0, 0, &[], &o).is_none());
        // Extreme deltas do not overflow the distance.
        let bins = [bin(i64::MIN, 1), bin(i64::MAX, 1)];
        let c = Candidate::from_bins(0, 2, &bins, &o).unwrap();
        assert_eq!(c.delta, i64::MIN);
        assert_eq!(c.noise.bins, 1);
    }

    #[test]
    fn noise_statistics_above_the_minimum() {
        let o = SearchOptions {
            min_noise_bins: 3,
            offset_tolerance: 0,
            ..SearchOptions::default()
        };
        let bins = [bin(0, 10), bin(1, 1), bin(2, 2), bin(3, 3)];
        let c = Candidate::from_bins(1, 16, &bins, &o).unwrap();
        assert_eq!(c.delta, 0);
        assert_eq!(c.noise.bins, 3);
        assert!((c.noise.mean - 2.0).abs() < 1e-12);
        assert!((c.noise.std_dev - libm::sqrt(2.0 / 3.0)).abs() < 1e-12);
        assert_eq!(c.noise.max, 3);
        let o = SearchOptions {
            min_noise_bins: 0,
            ..o
        };
        let c = Candidate::from_bins(1, 10, &[bin(0, 10)], &o).unwrap();
        assert_eq!(c.noise, NoiseStats::default());
    }

    #[test]
    fn options_default_is_the_documented_one() {
        let o = SearchOptions::default();
        assert_eq!(o.fanout, Some(64));
        assert_eq!(o.min_total_votes, 1);
        assert_eq!(o.offset_tolerance, 2);
        assert_eq!(o.min_noise_bins, 10);
        assert_eq!(o.max_votes, 1 << 26);
    }

    /// Totals through every path of the plain pass: the fixed ring, a
    /// fanout past the ring, and no fanout; the saturating path is covered
    /// by the overflow tests in `tests/model.rs`.
    #[test]
    fn totals_agree_across_fanouts_and_paths() {
        let mut rows: Vec<Posting> = (0..150).map(|t| Posting::new(7, 0, t)).collect();
        rows.extend((0..10).map(|t| Posting::new(7, 1, t)));
        let cat = Catalog::from_segment(Segment::from_postings(rows));
        let q = Query::from_pairs((0..3).map(|t| (7, t)), None);
        let mut s = Scratch::new();
        for (fanout, want0, want1) in [
            (Some(3), 9, 9),
            (Some(64), 192, 30),
            (Some(100), 300, 30),
            (None, 450, 30),
        ] {
            let o = SearchOptions {
                fanout,
                ..SearchOptions::default()
            };
            let got = cat.search(&q, &o, &mut s).unwrap();
            let totals: Vec<(u32, u32)> =
                got.iter().map(|c| (c.reference, c.total_votes)).collect();
            assert_eq!(totals, [(0, want0), (1, want1)], "fanout {fanout:?}");
        }
        // Many query times per hash: the weight is applied once per posting.
        let many = Query::from_pairs((0..1 << 19).map(|t| (7, t)), None);
        let o = SearchOptions {
            fanout: Some(3),
            max_votes: usize::MAX,
            ..SearchOptions::default()
        };
        let got = cat.search(&many, &o, &mut s).unwrap();
        assert_eq!(got[0].total_votes, 3 << 19);
        assert_eq!(got[1].total_votes, 3 << 19);
    }

    /// A run where no reference reaches the fanout counts every posting on
    /// the plain path and agrees with the capped paths.
    #[test]
    fn plain_path_agrees_with_capped_paths_below_the_fanout() {
        // Hash 5: reference 0 twice, 1 once, 2 twice. Hash 6: one each.
        let rows = vec![
            Posting::new(5, 0, 0),
            Posting::new(5, 0, 1),
            Posting::new(5, 1, 0),
            Posting::new(5, 2, 3),
            Posting::new(5, 2, 9),
            Posting::new(6, 0, 0),
            Posting::new(6, 1, 0),
            Posting::new(6, 2, 0),
        ];
        let cat = Catalog::from_segment(Segment::from_postings(rows));
        let q = Query::from_pairs([(5, 0), (5, 1), (5, 2), (5, 3), (6, 0)], None);
        let mut s = Scratch::new();
        for (fanout, want) in [
            (Some(1), [5, 5, 5]),
            (Some(2), [9, 5, 9]),
            (Some(64), [9, 5, 9]),
            (None, [9, 5, 9]),
        ] {
            let o = SearchOptions {
                fanout,
                ..SearchOptions::default()
            };
            let got = cat.search(&q, &o, &mut s).unwrap();
            let totals: Vec<u32> = got.iter().map(|c| c.total_votes).collect();
            assert_eq!(totals, want, "fanout {fanout:?}");
        }
    }

    /// Many references, every one a survivor.
    #[test]
    fn totals_over_many_references() {
        let n = 3 * 8192 + 5;
        let rows: Vec<Posting> = (0..n as u32).map(|r| Posting::new(1, r, r % 7)).collect();
        let cat = Catalog::from_segment(Segment::from_postings(rows));
        let q = Query::from_pairs([(1, 0), (1, 10)], Some(64));
        let o = SearchOptions {
            min_total_votes: 2,
            ..SearchOptions::default()
        };
        let mut s = Scratch::new();
        let got = cat.search(&q, &o, &mut s).unwrap();
        assert_eq!(got.len(), n);
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, c)| c.reference as usize == i && c.total_votes == 2)
        );
    }

    /// The totals and the groups are the same through the scalar decoder
    /// and through whatever `simd::run` picks, and the groups are the ones
    /// a walk of every run finds: over runs of 1 to 700 postings, with and
    /// without repeated references, for one survivor up to all of them.
    #[test]
    fn passes_agree_across_decoders() {
        use crate::simd::Scalar;

        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut below = move |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n) as u32
        };
        let mut rows = Vec::new();
        for hash in 0..48u32 {
            let len = [1, 3, 4, 7, 8, 9, 31, 32, 33, 64, 100, 700][hash as usize % 12];
            let mut reference = below(4);
            for _ in 0..len {
                // Odd hashes repeat some references, a few past the fanouts
                // below.
                let burst = if hash % 2 == 1 && below(4) == 0 {
                    1 + below(6)
                } else {
                    1
                };
                rows.extend((0..burst).map(|t| Posting::new(hash, reference, t)));
                reference += 1 + below(5);
            }
        }
        let seg = Segment::from_postings(rows);
        let n_refs = seg.max_reference().unwrap() as usize + 1;
        let segments = [&seg];
        let query = Query::from_pairs((0..48).flat_map(|h| [(h, 0), (h, 5)]), None);
        let mut runs = Vec::new();
        locate(&segments, &query, &mut runs);
        assert_eq!(runs.len(), 48);

        for fanout in [Some(64), Some(3), Some(1), None] {
            let (mut scalar, mut picked) = (vec![0u32; n_refs], vec![0u32; n_refs]);
            let pass = |totals| CountPlain {
                segments: &segments,
                fanout,
                runs: &runs,
                counters: Counters::<ScanAll> { totals, marks: () },
            };
            pass(&mut scalar).run(Scalar);
            simd::run(pass(&mut picked));
            assert_eq!(scalar, picked, "totals, fanout {fanout:?}");

            for step in [1, 2, 7, 50, 400, n_refs] {
                let survivors: Vec<u32> = (0..n_refs as u32).rev().step_by(step).rev().collect();
                let mut bits = vec![0u64; n_refs.div_ceil(64)];
                let mut index = vec![0u32; n_refs];
                for (i, &r) in survivors.iter().enumerate() {
                    bits[(r >> 6) as usize] |= 1u64 << (r & 63);
                    index[r as usize] = i as u32 + 1;
                }
                let mut want = Vec::new();
                for run in &runs {
                    let refs = seg.refs();
                    let mut i = run.lo;
                    while i < run.hi {
                        let r = refs.get(i);
                        let j = (i..run.hi).find(|&j| refs.get(j) != r).unwrap_or(run.hi);
                        if index[r as usize] != 0 {
                            want.push(Group {
                                segment: 0,
                                hash: run.hash,
                                survivor: index[r as usize] - 1,
                                lo: i,
                                hi: fanout.map_or(j, |f| j.min(i + f)),
                            });
                        }
                        i = j;
                    }
                }
                let (mut scalar, mut picked) = (Vec::new(), Vec::new());
                let pass = |groups| FindGroups {
                    segments: &segments,
                    fanout,
                    runs: &runs,
                    survivors: &survivors,
                    survivor_bits: &bits,
                    index: &index,
                    groups,
                };
                pass(&mut scalar).run(Scalar);
                simd::run(pass(&mut picked));
                assert_eq!(
                    scalar, want,
                    "scalar groups, fanout {fanout:?}, step {step}"
                );
                assert_eq!(picked, want, "groups, fanout {fanout:?}, step {step}");
            }
        }
    }

    #[test]
    fn scratch_reports_and_releases_memory() {
        let mut s = Scratch::new();
        assert_eq!(s.heap_bytes(), 0);
        s.fit(100);
        // A counter per reference and a bit per reference.
        assert!(s.heap_bytes() >= 400 + 16);
        assert!(alloc::format!("{s:?}").contains("reference_capacity: 100"));
        s.shrink_to_fit();
        assert_eq!(s.heap_bytes(), 0);
    }

    /// The block filter is an OR, so a block can pass it with no counter at
    /// the floor: 16 and 4 OR to 20 while neither reaches it. Nothing comes
    /// out, and the counters are still reset.
    #[test]
    fn block_or_at_the_floor_without_a_survivor() {
        let rows = vec![Posting::new(1, 0, 0), Posting::new(1, 1, 0)];
        let cat = Catalog::from_segment(Segment::from_postings(rows));
        let mut totals = vec![16u32, 4];
        let (mut survivors, mut survivor_totals) = (Vec::new(), Vec::new());
        cat.take_survivors(&mut totals, 20, &mut survivors, &mut survivor_totals)
            .expect("no counter overflows");
        assert_eq!(survivors, Vec::<u32>::new());
        assert_eq!(survivor_totals, Vec::<u32>::new());
        assert_eq!(totals, [0, 0]);

        // The same block with one counter that reaches the floor keeps it.
        let mut totals = vec![20u32, 4];
        cat.take_survivors(&mut totals, 20, &mut survivors, &mut survivor_totals)
            .expect("no counter overflows");
        assert_eq!(survivors, [0]);
        assert_eq!(survivor_totals, [20]);
        assert_eq!(totals, [0, 0]);
    }

    /// Every search of a pair below runs with the block record on and off,
    /// so both survivor scans see the same catalog, and the pair agrees on
    /// the candidates. Each search runs on the other one's scratch too, so
    /// a search after a search must see clean counters.
    fn both_paths(
        cat: &Catalog,
        query: &Query,
        options: &SearchOptions,
    ) -> (Vec<Candidate>, Vec<Candidate>) {
        let mut first = Scratch::new();
        let mut second = Scratch::new();
        let with = |track, scratch: &mut Scratch| {
            let mut out = Vec::new();
            cat.search_tracked(query, options, scratch, Some(track), |s| {
                out.push(s.candidate(options));
            })
            .map(|()| out)
        };
        // A search on each scratch first, then the opposite path on the
        // same scratch: neither leaves a counter, a mark or a survivor bit
        // behind.
        let _ = with(true, &mut first).expect("the recorded path succeeds");
        assert_scratch_clean(cat, &first);
        let scanned = with(false, &mut first).expect("the scanned path succeeds");
        assert_scratch_clean(cat, &first);
        let _ = with(false, &mut second).expect("the scanned path succeeds");
        assert_scratch_clean(cat, &second);
        let tracked = with(true, &mut second).expect("the recorded path succeeds");
        assert_scratch_clean(cat, &second);
        (tracked, scanned)
    }

    /// The counters and the block marks a search's totals pass writes, as
    /// the pass leaves them and before the survivor scan reads them: what
    /// [`Catalog::take_touched`] is handed.
    fn marked(cat: &Catalog, query: &Query, options: &SearchOptions) -> (Vec<u32>, Vec<u8>) {
        let n = cat.reference_count();
        let blocks = n.div_ceil(SCAN_BLOCK);
        let mut totals = alloc::vec![0u32; n];
        let mut touched = alloc::vec![0u8; blocks];
        let segments: Vec<&Segment> = cat.segments().collect();
        let mut runs = Vec::new();
        locate(&segments, query, &mut runs);
        totals_pass(
            Counters::<MarkBlocks> {
                totals: &mut totals,
                marks: &mut touched,
            },
            &segments,
            options.fanout,
            &runs,
        );
        (totals, touched)
    }

    /// A catalog of 3,001 references in four segments whose searches mark
    /// every block of counters: every reference has a posting of hash 1, so
    /// a query of hash 1 touches all of them, and the even ones also have
    /// hash 2, so a floor of two keeps them and drops the odd ones. The
    /// last block holds 57 counters, so the marks fill five whole words and
    /// seven bytes past them, and every byte position of a word carries a
    /// mark.
    fn many_block_catalog() -> Catalog {
        let mut catalog = Catalog::new();
        let mut next = 0u32;
        // Sizes that do not merge, so four segments stay.
        for batch in [1_800usize, 800, 300, 101] {
            let rows: Vec<Vec<(u32, u32)>> = (0..batch)
                .map(|_| {
                    let id = next;
                    next += 1;
                    let mut rows = vec![(1, id)];
                    if id.is_multiple_of(2) {
                        rows.push((2, id));
                    }
                    // One reference with many postings of one hash, so the
                    // fanout caps its contribution and the ring answers
                    // the test. It sits in the first batch so the extra
                    // postings do not grow the last segment enough to
                    // merge the tail.
                    if id == 1_799 {
                        rows.extend((0..200u32).map(|t| (7, t)));
                    }
                    rows
                })
                .collect();
            catalog.add_many(rows);
        }
        assert_eq!(catalog.reference_count(), 3_001);
        assert!(catalog.segment_count() > 2, "several segments");
        // A removed reference at the floor: it has hash 1 and hash 2, two
        // votes, and no longer surfaces.
        assert!(catalog.remove(1_000));
        catalog
    }

    /// The block record and the full scan agree on a catalog whose
    /// references span many blocks, so that marks land in every byte
    /// position of a word, in several words and in the trailing bytes after
    /// the last whole word: over survivors, a removed reference at the
    /// floor and references touched below it, through every arm of the
    /// totals pass and both of the errors it can report.
    #[test]
    fn both_survivor_scans_agree_across_many_blocks() {
        let catalog = many_block_catalog();
        let n_blocks = catalog.reference_count().div_ceil(SCAN_BLOCK);
        assert_eq!(n_blocks, 47, "five words and seven trailing bytes");
        assert_ne!(n_blocks % size_of::<u64>(), 0, "trailing bytes");

        // Hash 1 is in every reference, so a floor of one makes every live
        // reference a survivor and hash 2 with a floor of two keeps the
        // even ones and leaves the odd ones below the floor: both are
        // spread over every block.
        let query = Query::from_pairs([(1, 0), (2, 0)], Some(64));
        for floor in [1u32, 2, 3] {
            let o = SearchOptions {
                min_total_votes: floor,
                ..SearchOptions::default()
            };
            // The marks reach every block the counters have, byte positions
            // and trailing bytes alike.
            let (totals, touched) = marked(&catalog, &query, &o);
            assert_eq!(touched, alloc::vec![1u8; n_blocks], "floor {floor}");
            assert_eq!(totals[..2], [2, 1], "floor {floor}");
            let (tracked, scanned) = both_paths(&catalog, &query, &o);
            assert_eq!(tracked, scanned, "floor {floor}");
            let want = match floor {
                1 => 3_000,
                2 => 1_500,
                _ => 0,
            };
            assert_eq!(tracked.len(), want, "floor {floor}");
            assert!(tracked.iter().all(|c| c.reference != 1_000));
            if floor == 2 {
                assert!(tracked.iter().all(|c| c.reference.is_multiple_of(2)));
            }
        }

        // Every arm of `count_run!`: the whole run (multiplicity below the
        // fanout), the fixed ring, a fanout past the ring, and no fanout.
        let query = Query::from_pairs([(1, 0), (2, 0), (7, 0), (7, 5)], None);
        for fanout in [Some(64), Some(32), Some(1), None] {
            let o = SearchOptions {
                fanout,
                ..SearchOptions::default()
            };
            let (tracked, scanned) = both_paths(&catalog, &query, &o);
            assert_eq!(tracked, scanned, "fanout {fanout:?}");
        }

        // The saturating path, and the totals error it reports, on a
        // catalog of many blocks: 70 000 postings of one hash against
        // 65 000 query times need 4.55 billion votes for reference 0.
        let mut rows: Vec<Posting> = (0..70_000u32).map(|t| Posting::new(5, 0, t)).collect();
        rows.extend((0..70_000u32).map(|r| Posting::new(6, r, 0)));
        let sat = Catalog::from_segment(Segment::from_postings(rows));
        assert!(sat.reference_count().div_ceil(SCAN_BLOCK) > 1_000);
        let wide = Query::from_pairs((0..65_000u32).map(|t| (5, t)), None);
        let o = SearchOptions {
            fanout: None,
            max_votes: usize::MAX,
            ..SearchOptions::default()
        };
        let mut s = Scratch::new();
        for track in [true, false] {
            assert_eq!(
                sat.search_tracked(&wide, &o, &mut s, Some(track), |_| {})
                    .unwrap_err(),
                SearchError::TotalsOverflow { reference: 0 },
                "overflow, tracking {track}"
            );
            assert_scratch_clean(&sat, &s);
        }
        assert_eq!(
            sat.search(&wide, &o, &mut s).unwrap_err(),
            SearchError::TotalsOverflow { reference: 0 },
            "the same scratch still sees clean counters"
        );

        // The other error, which the same catalog reports once many
        // references survive: hash 5 gives reference 0 64 × 65,000 votes
        // and hash 6 gives each of the 70,000 references 65.
        let mut pairs: Vec<(u32, u32)> = (0..65_000u32).map(|t| (5, t)).collect();
        pairs.extend((0..65u32).map(|t| (6, t)));
        let q = Query::from_pairs(pairs, None);
        let tight = SearchOptions {
            fanout: Some(64),
            max_votes: 1_000,
            ..SearchOptions::default()
        };
        for track in [true, false] {
            assert_eq!(
                sat.search_tracked(&q, &tight, &mut s, Some(track), |_| {})
                    .unwrap_err(),
                SearchError::TooManyVotes {
                    votes: 64 * 65_000 + 70_000 * 65,
                    max: 1_000
                },
                "too many votes, tracking {track}"
            );
            assert_scratch_clean(&sat, &s);
        }
        let small = Query::from_pairs([(6, 0)], Some(64));
        let got = sat
            .search(&small, &SearchOptions::default(), &mut s)
            .unwrap();
        let (tracked, scanned) = both_paths(&sat, &small, &SearchOptions::default());
        assert_eq!(got, tracked);
        assert_eq!(got, scanned);
    }

    /// A catalog searched with the block record forced on and forced off
    /// gives the same candidates, through several segments, a removed
    /// reference at the floor, every arm of the totals pass and both of the
    /// errors it can report.
    #[test]
    fn both_survivor_scans_agree() {
        let mut x = 0x2F6E_2B1D_9C4E_7A31u64;
        let mut below = move |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        // Four segments of growing size, so a search looks in all of them.
        let mut catalog = Catalog::new();
        for n_rows in [40u32, 90, 200, 400] {
            catalog.add((0..n_rows).map(|_| {
                let hash = below(48) as u32;
                (hash, below(3_000) as u32)
            }));
        }
        // A reference with many postings of one hash, so the fanout caps
        // its contribution and the ring answers the test.
        catalog.add((0..200u32).map(|t| (7, t)));
        // A removed reference whose counter still reaches the floor.
        let doomed = catalog.add((0..40u32).map(|t| (11, t)));
        assert!(catalog.remove(doomed));
        assert!(
            catalog.segment_count() > 2,
            "several segments, {}",
            catalog.segment_count()
        );

        let o = SearchOptions::default();
        for qrows in [1usize, 30, 800] {
            let pairs: Vec<(u32, u32)> = (0..qrows)
                .map(|_| (below(48) as u32, below(3_000) as u32))
                .collect();
            let query = Query::from_pairs(pairs, Some(64));
            let (tracked, scanned) = both_paths(&catalog, &query, &o);
            assert_eq!(tracked, scanned, "{qrows} rows");
        }

        // Every arm of `count_run!`: the whole run (multiplicity below the
        // fanout), the fixed ring, a fanout past the ring, and no fanout.
        let query = Query::from_pairs((0..48u32).flat_map(|h| [(h, 0), (h, 5)]), None);
        for fanout in [Some(64), Some(32), Some(1), None] {
            let o = SearchOptions {
                fanout,
                ..SearchOptions::default()
            };
            let (tracked, scanned) = both_paths(&catalog, &query, &o);
            assert_eq!(tracked, scanned, "fanout {fanout:?}");
        }

        // The saturating path, and the totals error it reports: 70 000
        // postings of one hash against 65 000 query times need 4.55
        // billion votes, past what a 32-bit counter holds.
        let many: Vec<Posting> = (0..70_000u32).map(|t| Posting::new(5, 0, t)).collect();
        let sat = Catalog::from_segment(Segment::from_postings(many));
        let wide = Query::from_pairs((0..65_000u32).map(|t| (5, t)), None);
        let o = SearchOptions {
            fanout: None,
            max_votes: usize::MAX,
            ..SearchOptions::default()
        };
        let mut s = Scratch::new();
        for track in [true, false] {
            assert_eq!(
                sat.search_tracked(&wide, &o, &mut s, Some(track), |_| {})
                    .unwrap_err(),
                SearchError::TotalsOverflow { reference: 0 },
                "overflow, tracking {track}"
            );
            assert_scratch_clean(&sat, &s);
        }
        assert_eq!(
            sat.search(&wide, &o, &mut s).unwrap_err(),
            SearchError::TotalsOverflow { reference: 0 },
            "the same scratch still sees clean counters"
        );

        // The other error, on its own scratch, which it also leaves clean
        // before a search of the same catalog answers.
        let mut rows: Vec<Posting> = (0..70_000u32).map(|i| Posting::new(5, 0, i)).collect();
        rows.extend((0..10u32).map(|i| Posting::new(5, 1, i)));
        let big = Catalog::from_segment(Segment::from_postings(rows));
        let q = Query::from_pairs((0..65_000u32).map(|t| (5, t)), None);
        let tight = SearchOptions {
            fanout: Some(64),
            max_votes: 1_000,
            ..SearchOptions::default()
        };
        for track in [true, false] {
            assert!(matches!(
                big.search_tracked(&q, &tight, &mut s, Some(track), |_| {}),
                Err(SearchError::TooManyVotes { .. })
            ));
            assert_scratch_clean(&big, &s);
        }
        let small = Query::from_pairs([(5, 0)], Some(64));
        let got = big
            .search(&small, &SearchOptions::default(), &mut s)
            .unwrap();
        let (tracked, scanned) = both_paths(&big, &small, &SearchOptions::default());
        assert_eq!(got, tracked);
        assert_eq!(got, scanned);
    }

    /// A small deterministic generator for the randomised comparison.
    struct Gen(u64);

    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A random catalog of `refs` references over `hashes` hashes in
    /// several `add_many` batches, with some references removed. Every
    /// reference has a posting of hash 3, so a query of hash 3 lights up
    /// every counter and marks every block.
    fn random_catalog(rng: &mut Gen, refs: usize, hashes: u32) -> Catalog {
        let mut catalog = Catalog::new();
        let mut next = 0usize;
        while next < refs {
            // Batches of unrelated sizes, so the segments stay apart.
            let batch = (1 + rng.below(700) as usize).min(refs - next);
            let rows: Vec<Vec<(u32, u32)>> = (0..batch)
                .map(|_| {
                    let mut rows: Vec<(u32, u32)> = (0..=rng.below(4))
                        .map(|_| (rng.below(u64::from(hashes)) as u32, rng.below(4_000) as u32))
                        .collect();
                    rows.push((3, rng.below(4_000) as u32));
                    rows
                })
                .collect();
            catalog.add_many(rows);
            next += batch;
        }
        assert_eq!(catalog.reference_count(), refs);
        assert!(catalog.segment_count() > 1, "several segments");
        for _ in 0..refs / 40 {
            let id = rng.below(refs as u64) as u32;
            let _ = catalog.remove(id);
        }
        catalog
    }

    /// For a few seeds, a random catalog of a few thousand references in
    /// several segments with some removed, searched with random queries and
    /// floors: the recorded path and the full scan give identical
    /// candidates, and afterwards every counter, mark and survivor bit of
    /// the scratch is zero.
    #[test]
    fn both_survivor_scans_agree_on_random_catalogs() {
        for seed in [0x5EED_0001u64, 0x5EED_0002, 0x5EED_0003] {
            let mut rng = Gen(seed);
            let refs = 2_500 + rng.below(2_000) as usize;
            let hashes = 40 + rng.below(60) as u32;
            let catalog = random_catalog(&mut rng, refs, hashes);
            assert!(refs.div_ceil(SCAN_BLOCK) > 8, "many blocks of counters");
            let mut scratch = Scratch::new();
            let with = |track, scratch: &mut Scratch, q: &Query, o: &SearchOptions| {
                let mut out = Vec::new();
                catalog
                    .search_tracked(q, o, scratch, Some(track), |s| {
                        out.push(s.candidate(o));
                    })
                    .map(|()| out)
            };
            for _ in 0..12 {
                let qrows = 1 + rng.below(60) as usize;
                let mut pairs: Vec<(u32, u32)> = (0..qrows)
                    .map(|_| (rng.below(u64::from(hashes)) as u32, rng.below(4_000) as u32))
                    .collect();
                pairs.push((3, rng.below(4_000) as u32));
                let query = Query::from_pairs(pairs, Some(64));
                let o = SearchOptions {
                    min_total_votes: 1 + rng.below(4) as u32,
                    fanout: [Some(64), Some(3), Some(1), None][rng.below(4) as usize],
                    ..SearchOptions::default()
                };
                let tracked = with(true, &mut scratch, &query, &o).expect("recorded path");
                assert_scratch_clean(&catalog, &scratch);
                let scanned = with(false, &mut scratch, &query, &o).expect("scanned path");
                assert_scratch_clean(&catalog, &scratch);
                assert_eq!(tracked, scanned, "seed {seed:#x}");
            }
        }
    }

    /// A catalog of `n` references with no postings, so that a scratch
    /// sized for it has exactly `n.div_ceil(SCAN_BLOCK)` blocks of counters
    /// and the marks can be set by hand.
    fn empty_catalog(n: usize) -> Catalog {
        let mut b = Catalog::builder();
        for _ in 0..n {
            b.add(Vec::<(u32, u32)>::new());
        }
        b.build()
    }

    /// Set the counters and the marks of a hand-built scan: every block of
    /// `marked` holds a survivor at its first counter, one vote below
    /// `floor` at its second and zero elsewhere, and its mark is set; every
    /// other block holds a counter that would be a survivor if the scan
    /// read it. Returns the survivors a scan of the marked blocks must
    /// find.
    fn mark_blocks(
        totals: &mut [u32],
        touched: &mut [u8],
        marked: &[usize],
        floor: u32,
    ) -> Vec<(u32, u32)> {
        touched.fill(0);
        for (b, byte) in touched.iter_mut().enumerate() {
            let lo = b * SCAN_BLOCK;
            if marked.contains(&b) {
                *byte = 1;
                totals[lo] = floor + b as u32 % 3;
                totals[lo + 1] = floor - 1;
            } else {
                totals[lo] = floor + 100;
            }
        }
        marked
            .iter()
            .map(|&b| ((b * SCAN_BLOCK) as u32, totals[b * SCAN_BLOCK]))
            .collect()
    }

    /// [`Catalog::take_touched`] alone: marks set at the first block, the
    /// last, each byte position of a word and the trailing bytes after the
    /// last whole word name exactly the counters the scan reads and resets;
    /// the counters of unmarked blocks are neither read nor reset.
    #[test]
    fn take_touched_reads_exactly_the_marked_blocks() {
        // 47 blocks of 64 counters: five whole words of marks (blocks 0 to
        // 39) and seven bytes past them (blocks 40 to 46), so both loops of
        // the scan run. The last block is cut to 57 counters.
        let n_blocks = 47usize;
        let n = (n_blocks - 1) * SCAN_BLOCK + 57;
        let mut cat = empty_catalog(n);
        assert_eq!(cat.reference_count(), n);

        let floor = 7u32;
        // Every byte position of word 0, the first bytes of the next words,
        // the last byte of a word, and three of the trailing bytes.
        let marked = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 32, 39, 40, 43, 46];
        let mut totals = alloc::vec![0u32; n];
        let mut touched = alloc::vec![0u8; n_blocks];
        let want = mark_blocks(&mut totals, &mut touched, &marked, floor);
        let (mut survivors, mut survivor_totals) = (Vec::new(), Vec::new());
        cat.take_touched(
            &mut totals,
            &mut touched,
            floor,
            &mut survivors,
            &mut survivor_totals,
        )
        .expect("no counter overflows");
        let got: Vec<(u32, u32)> = survivors
            .iter()
            .copied()
            .zip(&survivor_totals)
            .map(|(r, &t)| (r, t))
            .collect();
        assert_eq!(got, want, "the survivors of the marked blocks, ascending");
        assert_eq!(touched, alloc::vec![0u8; n_blocks], "every mark cleared");
        for (b, &mark) in touched.iter().enumerate() {
            let lo = b * SCAN_BLOCK;
            let hi = (lo + SCAN_BLOCK).min(n);
            if marked.contains(&b) {
                assert_eq!(
                    totals[lo..hi],
                    alloc::vec![0u32; hi - lo],
                    "block {b} reset"
                );
            } else {
                assert_eq!(mark, 0, "block {b}'s mark");
                assert_eq!(totals[lo], floor + 100, "block {b} left alone");
            }
        }

        // A removed reference at the floor in a marked block is not a
        // survivor, and its counter is still reset.
        let gone = 8 * SCAN_BLOCK + 1;
        assert!(cat.remove(gone as u32));
        mark_blocks(&mut totals, &mut touched, &marked, floor);
        totals[gone] = floor + 1;
        cat.take_touched(
            &mut totals,
            &mut touched,
            floor,
            &mut survivors,
            &mut survivor_totals,
        )
        .expect("no counter overflows");
        assert_eq!(survivors.len(), marked.len());
        assert!(survivors.iter().all(|&r| r as usize != gone));
        assert_eq!(totals[gone], 0, "a removed reference's counter is reset");
        assert_eq!(touched, alloc::vec![0u8; n_blocks]);

        // An overflowing counter in a marked block is the error the scan
        // reports, and the marks and counters are still clean afterwards.
        mark_blocks(&mut totals, &mut touched, &marked, floor);
        totals[16 * SCAN_BLOCK + 5] = u32::MAX;
        assert_eq!(
            cat.take_touched(
                &mut totals,
                &mut touched,
                floor,
                &mut survivors,
                &mut survivor_totals,
            )
            .unwrap_err(),
            SearchError::TotalsOverflow {
                reference: (16 * SCAN_BLOCK + 5) as u32
            }
        );
        assert_eq!(touched, alloc::vec![0u8; n_blocks]);
        for &b in &marked {
            let lo = b * SCAN_BLOCK;
            let hi = (lo + SCAN_BLOCK).min(n);
            assert_eq!(
                totals[lo..hi],
                alloc::vec![0u32; hi - lo],
                "block {b} reset"
            );
        }
    }

    /// The counters, the block marks and the survivor bits of `scratch`
    /// are as a fresh scratch leaves them: the search that ran before it
    /// left nothing behind. The counters and marks are checked over the ids
    /// and blocks the catalog uses.
    fn assert_scratch_clean(cat: &Catalog, scratch: &Scratch) {
        let n = cat.reference_count();
        assert_eq!(scratch.totals[..n], alloc::vec![0u32; n]);
        let blocks = n.div_ceil(SCAN_BLOCK);
        assert_eq!(scratch.touched[..blocks], alloc::vec![0u8; blocks]);
        assert!(scratch.survivor_bits.iter().all(|&b| b == 0));
    }

    /// A search that panics in the totals pass, as one over a damaged
    /// catalog loaded without its checks does, leaves counters behind; the
    /// next search on the same scratch still counts from zero.
    #[cfg(feature = "std")]
    #[test]
    fn a_panic_in_the_totals_pass_does_not_leak_into_the_next_search() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        // References 0 to 39 in a catalog that says it holds ten ids: what
        // a flipped bit in a trusted snapshot can give.
        let rows: Vec<Posting> = (0..40u32).map(|r| Posting::new(1, r, 0)).collect();
        let damaged = Catalog::from_parts(
            vec![(Segment::from_postings(rows), Some((0, 9)))],
            vec![true; 10],
        );
        let mut healthy = Catalog::new();
        for _ in 0..10 {
            healthy.add([(1, 0), (2, 0)]);
        }
        let query = Query::from_pairs([(1, 0), (2, 0)], None);
        let options = SearchOptions {
            min_total_votes: 1,
            ..SearchOptions::default()
        };
        let expected = healthy
            .search(&query, &options, &mut Scratch::new())
            .expect("healthy search");
        let mut scratch = Scratch::new();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            damaged.search(&query, &options, &mut scratch)
        }));
        assert!(outcome.is_err(), "the damaged catalog panics");
        let got = healthy
            .search(&query, &options, &mut scratch)
            .expect("search after the panic");
        assert_eq!(got, expected);
    }

    /// A search on several threads hands out the survivors, totals and
    /// bins a search on one thread does, in the same order, and fails with
    /// the same error: over several segments, with removed references, a
    /// fanout that cuts runs, queries of every size, and every thread
    /// count from one to seven.
    #[cfg(feature = "std")]
    #[test]
    fn a_search_on_several_threads_finds_what_one_finds() {
        let mut seed = 0x005E_A2C4_u64;
        let mut next = |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(n)) as u32
        };
        let mut catalog = Catalog::new();
        for r in 0..400u32 {
            // A clip of each reference, and hashes shared by many.
            let rows: Vec<(u32, u32)> = (0..60)
                .map(|i| {
                    (
                        if i % 5 == 0 { next(30) } else { next(3_000) },
                        i * 3 + r % 7,
                    )
                })
                .collect();
            catalog.add(rows);
            if r % 37 == 0 {
                catalog.remove(r / 2);
            }
        }
        assert!(catalog.segment_count() > 1);
        let collect = |options: &SearchOptions, query: &Query, scratch: &mut Scratch| {
            let mut out = Vec::new();
            let result = catalog.search_with(query, options, scratch, |s| {
                out.push((s.reference, s.total_votes, s.bins.to_vec()));
            });
            result.map(|()| out)
        };
        let mut one = Scratch::new();
        let mut many = Scratch::new();
        for q in 0..12u32 {
            let r = next(400);
            let mut pairs: Vec<(u32, u32)> = (0..(20 + q * 9)).map(|i| (next(3_000), i)).collect();
            pairs.extend((0..40).map(|i| (next(30), i + 1_000)));
            pairs.push((r, 0));
            let query = Query::from_pairs(pairs, Some(8));
            for (fanout, floor) in [(Some(64), 2), (Some(2), 3), (None, 1)] {
                let mut options = SearchOptions {
                    fanout,
                    min_total_votes: floor,
                    ..SearchOptions::default()
                };
                let want = collect(&options, &query, &mut one).expect("one thread");
                for threads in [2, 3, 7] {
                    options.threads = threads;
                    let got = collect(&options, &query, &mut many).expect("several threads");
                    assert_eq!(got, want, "query {q}, {threads} threads, fanout {fanout:?}");
                }
                options.threads = 1;
                // The bound on the votes holds over all the ranges at once.
                let votes: u64 = want.iter().map(|s| u64::from(s.1)).sum();
                if votes > 1 {
                    options.max_votes = (votes - 1) as usize;
                    let one_err = collect(&options, &query, &mut one).expect_err("too many");
                    options.threads = 4;
                    let many_err = collect(&options, &query, &mut many).expect_err("too many");
                    assert_eq!(one_err, many_err);
                }
            }
        }
        // The scratches are clean: a search on one thread with the scratch
        // that served several finds the same.
        let query = Query::from_pairs((0..300).map(|i| (next(3_000), i)), None);
        let options = SearchOptions::default();
        assert_eq!(
            collect(&options, &query, &mut many).unwrap(),
            collect(&options, &query, &mut Scratch::new()).unwrap()
        );
    }
}
