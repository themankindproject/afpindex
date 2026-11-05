use alloc::vec::Vec;
use core::fmt;

use crate::packed::Packed;
use crate::{Catalog, Query, SearchError, Segment};

/// Bounds and statistics parameters for a search.
///
/// Start from [`SearchOptions::default`] and assign the fields to change:
///
/// ```
/// # use afindex::SearchOptions;
/// let mut options = SearchOptions::default();
/// options.min_total_votes = 84;
/// ```
///
/// Fields may be added in `0.x` releases, which is why the struct cannot
/// be built with a literal, even one that fills the rest from the default:
///
/// ```compile_fail,E0639
/// let options = afindex::SearchOptions { min_total_votes: 84, ..Default::default() };
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
    /// histogram pass holds 16 bytes per vote. Exceeding it returns
    /// [`SearchError::TooManyVotes`]. Default `1 << 26` (about 1 GiB).
    pub max_votes: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            fanout: Some(64),
            min_total_votes: 1,
            offset_tolerance: 2,
            min_noise_bins: 10,
            max_votes: 1 << 26,
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
}

/// Per-thread buffers a search reuses.
///
/// Holds one 32-bit counter and one bit per reference id (4 bytes and a
/// bit per id of the largest catalog searched), plus the run, group, vote
/// and bin buffers.
/// After the first search on a catalog, [`Catalog::search_with`] on a
/// query of similar size allocates nothing; [`Catalog::search`] also
/// allocates its result. One scratch serves any number of catalogs.
/// Searches leave it clean, including on an error, and so does a panic
/// inside a visitor.
#[derive(Default)]
pub struct Scratch {
    /// Vote total per reference during the totals pass; survivor index
    /// plus one per survivor during the histogram pass; zero otherwise.
    totals: Vec<u32>,
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
            + self.survivor_bits.capacity() * size_of::<u64>()
            + self.runs.capacity() * size_of::<RunLoc>()
            + self.groups.capacity() * size_of::<Group>()
            + self.survivors.capacity() * size_of::<u32>()
            + self.survivor_totals.capacity() * size_of::<u32>()
            + self.slots.capacity() * size_of::<usize>()
            + self.cursor.capacity() * size_of::<usize>()
            + self.votes.capacity() * size_of::<(i64, u32)>()
            + self.bins.capacity() * size_of::<Bin>()
    }

    fn fit(&mut self, n_refs: usize) {
        if self.totals.len() < n_refs {
            self.totals.resize(n_refs, 0);
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
/// Counters per block of the survivor scan; a block is reduced to its
/// maximum on vector instructions, and all-zero blocks are skipped.
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
    let hashes = query.hashes();
    let n = hashes.len();
    for (si, seg) in segments.iter().enumerate() {
        seg.touch_directory(&hashes[..LOOKUP_BATCH.min(n)]);
        for k in (0..n).step_by(LOOKUP_BATCH) {
            let end = (k + LOOKUP_BATCH).min(n);
            seg.touch_directory(&hashes[end..(end + LOOKUP_BATCH).min(n)]);
            for (i, &hash) in hashes[k..end].iter().enumerate() {
                let qi = k + i;
                if query.times(qi).is_empty() {
                    continue;
                }
                if let Some((lo, hi, mult)) = seg.range_mult(hash) {
                    runs.push(RunLoc {
                        segment: si as u32,
                        hash: qi as u32,
                        lo,
                        hi,
                        mult,
                    });
                }
            }
        }
    }
}

/// Totals pass: each run adds `query_times × min(count, fanout)` to each
/// reference in it.
///
/// When no counter can overflow (every run's full contribution fits in
/// 32 bits, which is every practical query) the runs are decoded as a
/// stream and counted with plain adds; otherwise they are counted with
/// saturating adds, so an overflow stays detectable afterwards.
fn totals_pass(
    segments: &[&Segment],
    query: &Query,
    fanout: Option<usize>,
    runs: &[RunLoc],
    totals: &mut [u32],
) {
    let bound = runs.iter().fold(0u64, |b, run| {
        b.saturating_add((run.hi - run.lo) as u64 * u64::from(query_count(query, run)))
    });
    if u32::try_from(bound).is_ok() {
        count_plain(segments, query, fanout, runs, totals);
    } else {
        count_direct(segments, query, fanout, runs, totals);
    }
}

/// Query times of a run's hash, as a vote weight.
fn query_count(query: &Query, run: &RunLoc) -> u32 {
    u32::try_from(query.times(run.hash as usize).len()).unwrap_or(u32::MAX)
}

/// Visit every run's references with its vote weight. Each batch of runs
/// is loaded while the batch before it is visited.
fn for_each_run(
    segments: &[&Segment],
    query: &Query,
    runs: &[RunLoc],
    mut visit: impl FnMut(&Packed, usize, usize, u32, u8),
) {
    let n = runs.len();
    let touch = |from: usize, to: usize| {
        let acc = runs[from.min(n)..to.min(n)].iter().fold(0u32, |acc, run| {
            acc ^ segments[run.segment as usize].touch_refs(run.lo, run.hi)
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
                query_count(query, run),
                run.mult,
            );
        }
    }
}

/// Count every run with plain adds, decoding each run as a stream.
///
/// Inside a run a reference's postings are contiguous, so entry `i` is its
/// reference's `fanout + 1`th or later posting exactly when entry
/// `i − fanout` has the same reference; a ring of the last [`RING`]
/// references answers that without a second read of the run.
fn count_plain(
    segments: &[&Segment],
    query: &Query,
    fanout: Option<usize>,
    runs: &[RunLoc],
    totals: &mut [u32],
) {
    // The largest multiplicity that proves no reference of a run reaches
    // the fanout. A saturated multiplicity is only a lower bound, so it
    // proves nothing whatever the fanout.
    let whole = fanout.map_or(0, |f| f.min(usize::from(u8::MAX) - 1));
    for_each_run(
        segments,
        query,
        runs,
        |refs, lo, hi, qc, mult| match fanout {
            // Every posting counts, which on real catalogs is nearly every
            // run.
            Some(_) if usize::from(mult) <= whole => {
                refs.for_each(lo, hi, |r| totals[r as usize] += qc);
            }
            Some(f) if f <= RING => {
                let head = lo + (hi - lo).min(f);
                let mut ring = [0u32; RING];
                let mut i = 0usize;
                refs.for_each(lo, head, |r| {
                    ring[i % RING] = r;
                    i += 1;
                    totals[r as usize] += qc;
                });
                refs.for_each(head, hi, |r| {
                    let counts = ring[(i - f) % RING] != r;
                    ring[i % RING] = r;
                    i += 1;
                    totals[r as usize] += qc * u32::from(counts);
                });
            }
            Some(f) => {
                let head = lo + (hi - lo).min(f);
                refs.for_each(lo, head, |r| totals[r as usize] += qc);
                for i in head..hi {
                    let r = refs.get(i);
                    totals[r as usize] += qc * u32::from(refs.get(i - f) != r);
                }
            }
            None => refs.for_each(lo, hi, |r| totals[r as usize] += qc),
        },
    );
}

/// Count every run straight into the counters, saturating at `u32::MAX`
/// so an overflow is detectable afterwards. Entry `i` counts unless entry
/// `i − fanout` has the same reference.
fn count_direct(
    segments: &[&Segment],
    query: &Query,
    fanout: Option<usize>,
    runs: &[RunLoc],
    totals: &mut [u32],
) {
    for_each_run(segments, query, runs, |refs, lo, hi, qc, _| match fanout {
        Some(f) => {
            let head = lo + (hi - lo).min(f);
            for i in lo..head {
                let t = &mut totals[refs.get(i) as usize];
                *t = t.saturating_add(qc);
            }
            for i in head..hi {
                let r = refs.get(i);
                let t = &mut totals[r as usize];
                *t = t.saturating_add(qc * u32::from(refs.get(i - f) != r));
            }
        }
        None => {
            for i in lo..hi {
                let t = &mut totals[refs.get(i) as usize];
                *t = t.saturating_add(qc);
            }
        }
    });
}

/// One survivor's postings in one run, `fanout`-capped: entries
/// `lo..hi` of `segment`, each paired with every time of query hash
/// `hash`.
#[derive(Clone, Copy, Debug)]
struct Group {
    segment: u32,
    hash: u32,
    survivor: u32,
    lo: usize,
    hi: usize,
}

/// First half of the histogram pass: find every survivor's group in every
/// run, in run order then reference order.
///
/// A run is grouped by reference, ascending, so a run with few survivors
/// relative to its length is probed by binary search per survivor, and a
/// run with many is walked once with a bitmap membership test per group
/// (`index` then holds each survivor's index plus one). Only the
/// references are read here, and they are still cached from the totals
/// pass.
fn find_groups(
    segments: &[&Segment],
    fanout: Option<usize>,
    runs: &[RunLoc],
    survivors: &[u32],
    survivor_bits: &[u64],
    index: &[u32],
    groups: &mut Vec<Group>,
) {
    groups.clear();
    let n_surv = survivors.len();
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
            let mut lo = run_lo;
            for (sv, &r) in survivors.iter().enumerate() {
                let i = refs.partition_point(lo, run_hi, |x| x < r);
                if i == run_hi {
                    break;
                }
                lo = i;
                if refs.get(i) != r {
                    continue;
                }
                let j = refs.partition_point(i, run_hi, |x| x == r);
                push(i, j, sv as u32);
                lo = j;
            }
        } else if run.mult == 1 {
            // Every posting is its own group: a streamed membership test
            // per posting, no search for the group's end.
            let mut i = run_lo;
            refs.for_each(run_lo, run_hi, |r| {
                if survivor_bits[(r >> 6) as usize] & (1u64 << (r & 63)) != 0 {
                    push(i, i + 1, index[r as usize] - 1);
                }
                i += 1;
            });
        } else {
            let mut i = run_lo;
            while i < run_hi {
                let r = refs.get(i);
                let mut j = i + 1;
                while j < run_hi && refs.get(j) == r {
                    j += 1;
                }
                if survivor_bits[(r >> 6) as usize] & (1u64 << (r & 63)) != 0 {
                    push(i, j, index[r as usize] - 1);
                }
                i = j;
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
        scratch.fit(n_refs);
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
            bins,
        } = scratch;
        // Segments are shared handles; the passes take a view of them. The
        // small vector is the only per-search allocation beyond the scratch.
        let segments: Vec<&Segment> = self.segments().collect();
        let segments: &[&Segment] = &segments;

        locate(segments, query, runs);
        totals_pass(segments, query, fanout, runs, totals);
        let floor = options.min_total_votes.max(1);
        self.take_survivors(&mut totals[..n_refs], floor, survivors, survivor_totals)?;
        if survivors.is_empty() {
            return Ok(());
        }
        let n_votes = survivor_totals
            .iter()
            .fold(0u64, |acc, &t| acc.saturating_add(u64::from(t)));
        if n_votes > options.max_votes as u64 {
            return Err(SearchError::TooManyVotes {
                votes: n_votes,
                max: options.max_votes,
            });
        }
        let n_votes = n_votes as usize;

        // Each survivor owns a slice of the vote array sized by its exact
        // total: a counting sort keyed by survivor. The zeroed counters hold
        // the survivor indices (plus one) until the scatter is done.
        slots.clear();
        slots.push(0);
        for (i, &r) in survivors.iter().enumerate() {
            totals[r as usize] = i as u32 + 1;
            survivor_bits[(r >> 6) as usize] |= 1u64 << (r & 63);
            slots.push(slots[i] + survivor_totals[i] as usize);
        }
        // The scatter writes every slot of `votes[..n_votes]` exactly once
        // (the cursors end at the slice ends), so stale contents need no
        // zero fill; only grow when a search needs more.
        if votes.len() < n_votes {
            votes.resize(n_votes, (0, 0));
        }
        let votes = &mut votes[..n_votes];
        cursor.clear();
        cursor.extend_from_slice(&slots[..survivors.len()]);
        find_groups(
            segments,
            fanout,
            runs,
            survivors,
            survivor_bits,
            totals,
            groups,
        );
        write_votes(segments, query, groups, cursor, votes);
        for &r in survivors.iter() {
            totals[r as usize] = 0;
            survivor_bits[(r >> 6) as usize] &= !(1u64 << (r & 63));
        }

        for (sv, &r) in survivors.iter().enumerate() {
            debug_assert_eq!(cursor[sv], slots[sv + 1], "scatter count mismatch");
            bins_of(&mut votes[slots[sv]..slots[sv + 1]], bins);
            visit(Survivor {
                reference: r,
                total_votes: survivor_totals[sv],
                bins,
            });
        }
        Ok(())
    }

    /// Move the survivors out of the counters: the references that were
    /// touched, are live and reach `floor`, in ascending order, each with
    /// its total. The scan also resets the counters, so the scratch is
    /// clean from here on. A block's maximum decides whether it holds
    /// anything to look at.
    fn take_survivors(
        &self,
        totals: &mut [u32],
        floor: u32,
        survivors: &mut Vec<u32>,
        survivor_totals: &mut Vec<u32>,
    ) -> Result<(), SearchError> {
        survivors.clear();
        survivor_totals.clear();
        let mut overflow = None;
        for (b, block) in totals.chunks_mut(SCAN_BLOCK).enumerate() {
            let max = block.iter().fold(0u32, |m, &t| m.max(t));
            if max == 0 {
                continue;
            }
            if max >= floor {
                for (i, &t) in block.iter().enumerate() {
                    if t >= floor {
                        let r = (b * SCAN_BLOCK + i) as u32;
                        // A removed reference is skipped before anything
                        // else: its postings are nobody's problem.
                        if !self.is_live(r) {
                            continue;
                        }
                        if t == u32::MAX {
                            overflow.get_or_insert(r);
                        } else {
                            survivors.push(r);
                            survivor_totals.push(t);
                        }
                    }
                }
            }
            block.fill(0);
        }
        match overflow {
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
}
