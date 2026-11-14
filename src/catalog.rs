use alloc::vec::Vec;
use core::fmt;

use crate::{Posting, Segment};

/// Reference-counted segment handle: `Arc` where the target has atomics,
/// `Rc` otherwise (a catalog is then not `Send`).
#[cfg(target_has_atomic = "ptr")]
type Shared<T> = alloc::sync::Arc<T>;
#[cfg(not(target_has_atomic = "ptr"))]
type Shared<T> = alloc::rc::Rc<T>;

/// A merge the catalog's policy asks for, carried out by the caller.
///
/// [`Catalog::merge_tail_of`] and [`Catalog::merged_from`] decide when the
/// catalog merges and which parts it merges; a catalog on the heap and one
/// in a directory differ only in the implementation of this trait, so the
/// policy exists once.
pub(crate) trait MergeParts {
    /// Why a merge can fail; a merge in memory cannot.
    type Error;

    /// Merge `sides`, each filtered by its `keep` if it has one, into one
    /// part whose references lie in `ids`.
    fn merge<F: Fn(u32) -> bool + Sync>(
        &mut self,
        sides: &[(&Segment, Option<&F>)],
        ids: (u32, u32),
    ) -> Result<Option<Part>, Self::Error>;
}

/// The merge of catalog parts in memory: what a catalog without a
/// directory does.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InMemory;

impl MergeParts for InMemory {
    type Error = core::convert::Infallible;

    // Nothing a merge in memory returns is an error, so the caller's error
    // path compiles away once this is inlined.
    #[inline(always)]
    fn merge<F: Fn(u32) -> bool + Sync>(
        &mut self,
        sides: &[(&Segment, Option<&F>)],
        ids: (u32, u32),
    ) -> Result<Option<Part>, Self::Error> {
        let merged = Segment::merge_all(sides);
        Ok((!merged.is_empty()).then(|| Part::new(merged, ids)))
    }
}

/// The value of a result that cannot fail, for the in-memory merge.
pub(crate) fn never<T>(result: Result<T, core::convert::Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// A searchable set of references: segments, plus which reference ids are
/// live.
///
/// Reference ids are dense `u32`s. [`CatalogBuilder::add`] and
/// [`Catalog::add`] assign them in order, and they never change or get
/// reused; a catalog snapshot keeps the whole id space
/// ([`Catalog::write_to`](crate::Catalog::write_to), `std`). A removed
/// reference is tombstoned: searches skip it immediately, and its postings
/// leave memory the next time the segment holding them is merged or
/// compacted.
///
/// # Segments
///
/// A catalog keeps a list of immutable segments with geometrically
/// decreasing sizes: each is less than half of the one before it.
/// [`Catalog::add`] builds a one-reference segment and then merges from
/// the tail while a segment is at least half of its predecessor, so a
/// posting takes part in `O(log n)` merges over its life and the list
/// stays about `log2(largest / smallest)` long. A search looks every query
/// hash up in every segment, so call [`Catalog::compact_deltas`] (cheap)
/// or [`Catalog::compact`] (one pass over everything) when the list grows.
///
/// Segments are shared, so `clone` costs `O(segments + ids)` and the clone
/// keeps the postings alive: a server can clone, compact the clone, and
/// swap it in without blocking searches on the original.
///
/// # Memory
///
/// The liveness table costs one byte per id, and a [`Scratch`](crate::Scratch)
/// another four bytes and a bit per id, whether or not the id has postings.
///
/// A catalog is `Send + Sync` where the target has atomic pointers;
/// searches take `&self`, so threads share one catalog and own a
/// `Scratch` each.
#[derive(Clone, Default)]
pub struct Catalog {
    /// Sizes non-increasing; `parts[0]` is the snapshot when one exists.
    parts: Vec<Part>,
    live: Vec<bool>,
    live_count: usize,
}

/// A segment of a catalog and what the catalog knows of its references.
#[derive(Clone)]
pub(crate) struct Part {
    pub(crate) segment: Shared<Segment>,
    /// Every reference with a posting in the segment lies in
    /// `ids.0..=ids.1`.
    pub(crate) ids: (u32, u32),
    /// `true` while the segment may hold postings of removed references.
    /// A segment that cannot is merged without a look at its references.
    pub(crate) stale: bool,
    /// The segment file the part was read from or written to in a catalog
    /// directory, or [`NO_FILE`]: what `save` compares the directory with.
    /// A plain `u32` rather than an `Option` keeps the part at 24 bytes, of
    /// which a catalog of many segments holds one each.
    #[cfg(feature = "mmap")]
    pub(crate) file: u32,
}

/// The file id of a part that has no segment file.
#[cfg(feature = "mmap")]
pub(crate) const NO_FILE: u32 = u32::MAX;

impl Part {
    /// A part whose references all lie in `ids` and are all live.
    pub(crate) fn new(segment: Segment, ids: (u32, u32)) -> Self {
        Self {
            segment: Shared::new(segment),
            ids,
            stale: false,
            #[cfg(feature = "mmap")]
            file: NO_FILE,
        }
    }

    /// Hold the mapping of segment file `file` in place of the segment it
    /// was written from. `ids` is the range the file holds, when it holds
    /// a posting: it can only be narrower than the part's.
    #[cfg(feature = "mmap")]
    pub(crate) fn move_to_file(&mut self, mapped: Segment, ids: Option<(u32, u32)>, file: u32) {
        debug_assert_eq!(mapped.len(), self.segment.len(), "the same segment");
        self.segment = Shared::new(mapped);
        self.ids = ids.unwrap_or(self.ids);
        self.file = file;
    }

    /// The same part, recorded as segment file `file`.
    #[cfg(feature = "mmap")]
    pub(crate) fn with_file(mut self, file: u32) -> Self {
        self.file = file;
        self
    }

    /// The segment file of the part, if it has one.
    #[cfg(feature = "mmap")]
    pub(crate) fn file(&self) -> Option<u32> {
        (self.file != NO_FILE).then_some(self.file)
    }

    /// The same part, stale as `stale` says.
    #[cfg(feature = "mmap")]
    pub(crate) fn stale_if(mut self, stale: bool) -> Self {
        self.stale = stale;
        self
    }
}

impl Catalog {
    /// An empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bulk-build a catalog with one sort over every posting.
    #[must_use]
    pub fn builder() -> CatalogBuilder {
        CatalogBuilder::default()
    }

    /// Wrap a segment whose reference ids the caller assigned.
    ///
    /// Every id from `0` to the largest id in the segment is live, including
    /// ids with no postings, and [`Catalog::add`] continues after the
    /// largest. To persist a catalog that has seen removals, use the catalog
    /// snapshot instead, which keeps the id space and the liveness table.
    ///
    /// # Panics
    ///
    /// Panics if the largest id plus one does not fit in `usize` (only
    /// possible on 32-bit targets with the id `u32::MAX`).
    #[must_use]
    pub fn from_segment(segment: Segment) -> Self {
        let max = segment.max_reference();
        let n = max.map_or(0, |m| {
            usize::try_from(m)
                .ok()
                .and_then(|m| m.checked_add(1))
                .expect("reference id space exceeds the address space")
        });
        Self {
            parts: max
                .map(|max| Part::new(segment, (0, max)))
                .into_iter()
                .collect(),
            live: alloc::vec![true; n],
            live_count: n,
        }
    }

    /// Assemble from validated parts (snapshot reader): each segment with
    /// the smallest and largest reference it holds, if it holds any.
    #[cfg(feature = "std")]
    pub(crate) fn from_parts(
        segments: Vec<(Segment, Option<(u32, u32)>)>,
        live: Vec<bool>,
    ) -> Self {
        let live_count = live.iter().filter(|&&l| l).count();
        let parts = segments
            .into_iter()
            .filter_map(|(segment, ids)| {
                let ids = ids?;
                // Stale if any id in its range was removed (or lies outside
                // the id space, which the reader refuses anyway).
                let stale = live
                    .get(ids.0 as usize..=ids.1 as usize)
                    .is_none_or(|range| range.contains(&false));
                Some(Part {
                    stale,
                    ..Part::new(segment, ids)
                })
            })
            .collect();
        Self {
            parts,
            live,
            live_count,
        }
    }

    /// Assemble from parts a manifest validated: segments already placed
    /// with the file they came from, and the liveness table they share.
    #[cfg(feature = "mmap")]
    pub(crate) fn from_files(parts: Vec<Part>, live: Vec<bool>) -> Self {
        let live_count = live.iter().filter(|&&l| l).count();
        Self {
            parts,
            live,
            live_count,
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn live_table(&self) -> &[bool] {
        &self.live
    }

    /// The parts that make the catalog, in order.
    #[cfg(feature = "mmap")]
    pub(crate) fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// The parts that make the catalog, in order, for a store that
    /// rewrites them.
    #[cfg(feature = "mmap")]
    pub(crate) fn parts_mut(&mut self) -> &mut Vec<Part> {
        &mut self.parts
    }

    /// Add a reference from its `(hash, time)` rows and return its id.
    ///
    /// The rows become one sorted segment, which is then merged from the
    /// tail (see the type docs). Exact duplicate rows are dropped.
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    #[inline]
    pub fn add(&mut self, rows: impl IntoIterator<Item = (u32, u32)>) -> u32 {
        never(self.add_of(rows, &mut InMemory))
    }

    /// [`Catalog::add`], with `merges` deciding where a merge goes.
    ///
    /// A merge that fails leaves the catalog as it was.
    #[inline]
    pub(crate) fn add_of<M: MergeParts>(
        &mut self,
        rows: impl IntoIterator<Item = (u32, u32)>,
        merges: &mut M,
    ) -> Result<u32, M::Error> {
        let id = self.next_id();
        // One reference: a row is its hash and time, which sort as one
        // 64-bit key.
        let mut rows: Vec<u64> = rows
            .into_iter()
            .map(|(hash, time)| u64::from(hash) << 32 | u64::from(time))
            .collect();
        rows.sort_unstable();
        rows.dedup();
        self.live.push(true);
        self.live_count += 1;
        if rows.is_empty() {
            return Ok(id);
        }
        let part = Part::new(Segment::from_reference(id, &rows), (id, id));
        match self.merge_in(part, merges) {
            Ok(()) => Ok(id),
            Err(e) => {
                self.take_back(1);
                Err(e)
            }
        }
    }

    /// Push `part` and merge from the tail, with `merges` carrying the
    /// merges out. A merge that fails leaves the parts as they were, the
    /// new one included: [`merge_tail_in`] replaces the parts it joined
    /// only once their merge has returned.
    #[inline]
    fn merge_in<M: MergeParts>(&mut self, part: Part, merges: &mut M) -> Result<(), M::Error> {
        self.parts.push(part);
        let added = self.parts.len() - 1;
        match merge_tail_in(&mut self.parts, &self.live, merges) {
            Ok(()) => Ok(()),
            Err(e) => {
                // The merge left the parts it joined where they were, so
                // the one this call added comes off and nothing else moves.
                self.parts.truncate(added);
                Err(e)
            }
        }
    }

    /// Add several references, each from its `(hash, time)` rows, and
    /// return the range of ids they were given. One sort builds one
    /// segment for the batch, and one merge from the tail follows, where
    /// [`Catalog::add`] would sort and merge once per reference.
    ///
    /// Ids are assigned from the next one upward, in the order the
    /// references come, exactly as repeated [`Catalog::add`] calls would
    /// assign them. A reference without rows still takes an id. An empty
    /// batch changes nothing and returns the empty range at the id the
    /// next reference would get, so the returned range always starts at
    /// that id. The catalog is changed only once the input is consumed,
    /// so a panicking iterator leaves it as it was. Exact duplicate rows
    /// are dropped.
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    pub fn add_many<I, R>(&mut self, references: I) -> core::ops::Range<u32>
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = (u32, u32)>,
    {
        never(self.add_many_of(references, &mut InMemory))
    }

    /// [`Catalog::add_many`], with `merges` deciding where a merge goes.
    ///
    /// A merge that fails leaves the catalog as it was.
    pub(crate) fn add_many_of<I, R, M: MergeParts>(
        &mut self,
        references: I,
        merges: &mut M,
    ) -> Result<core::ops::Range<u32>, M::Error>
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = (u32, u32)>,
    {
        let start = self.live.len();
        let mut rows = Vec::new();
        let mut count = 0usize;
        for reference in references {
            // The check `next_id` makes, once per reference.
            let id = start
                .checked_add(count)
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| id < u32::MAX)
                .expect("reference ids exhausted");
            count += 1;
            rows.extend(reference.into_iter().map(|(hash, time)| Posting {
                hash,
                reference: id,
                time,
            }));
        }
        if count == 0 {
            // No id is taken, but the range still starts where the next
            // reference's would.
            let next = self.next_id();
            return Ok(next..next);
        }
        // Every id in the batch passed the check above, so both fit.
        let first = u32::try_from(start).expect("checked with the ids");
        let last = first + (count - 1) as u32;
        self.live.resize(start + count, true);
        self.live_count += count;
        if !rows.is_empty() {
            let part = Part::new(Segment::from_postings(rows), (first, last));
            if let Err(e) = self.merge_in(part, merges) {
                self.take_back(count);
                return Err(e);
            }
        }
        Ok(first..last + 1)
    }

    /// Give back the last `n` ids, all live, after a call that added them
    /// failed.
    fn take_back(&mut self, n: usize) {
        self.live.truncate(self.live.len() - n);
        self.live_count -= n;
    }

    fn next_id(&self) -> u32 {
        u32::try_from(self.live.len())
            .ok()
            .filter(|&id| id < u32::MAX)
            .expect("reference ids exhausted")
    }

    /// Tombstone a reference. Returns `false` if the id is unknown or
    /// already removed.
    pub fn remove(&mut self, id: u32) -> bool {
        match self.live.get_mut(id as usize) {
            Some(live) if *live => {
                *live = false;
                self.live_count -= 1;
                for part in &mut self.parts {
                    part.stale |= (part.ids.0..=part.ids.1).contains(&id);
                }
                true
            }
            _ => false,
        }
    }

    /// `true` if `id` was assigned and has not been removed.
    #[must_use]
    pub fn is_live(&self, id: u32) -> bool {
        self.live.get(id as usize).copied().unwrap_or(false)
    }

    /// Number of live references.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.live_count
    }

    /// Number of ids assigned so far, live or not. The next
    /// [`Catalog::add`] returns this value.
    #[must_use]
    pub fn reference_count(&self) -> usize {
        self.live.len()
    }

    /// Postings held across all segments, including those of removed
    /// references not yet compacted away.
    #[must_use]
    pub fn posting_count(&self) -> usize {
        self.segments().map(Segment::len).sum()
    }

    /// Number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.parts.len()
    }

    /// Segment `i`, largest first; `0` is the snapshot when one exists.
    #[must_use]
    pub fn segment(&self, i: usize) -> Option<&Segment> {
        self.parts.get(i).map(|part| &*part.segment)
    }

    /// The segments, largest first.
    pub fn segments(&self) -> impl ExactSizeIterator<Item = &Segment> + '_ {
        self.parts.iter().map(|part| &*part.segment)
    }

    /// Heap bytes held by the segments and the liveness table.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.segments().map(Segment::heap_bytes).sum::<usize>()
            + self.parts.capacity() * size_of::<Part>()
            + self.live.capacity()
    }

    /// Merge every segment but the largest into one, dropping the postings
    /// of removed references from them. The deltas are merged in one pass,
    /// which rewrites each of their postings once; the largest segment is
    /// not rewritten, so the result may break the half-size rule until the
    /// next [`Catalog::add`] merges them. A no-op when there is at most one
    /// delta and it has nothing to drop.
    pub fn compact_deltas(&mut self) {
        never(self.compact_deltas_of(&mut InMemory));
    }

    /// [`Catalog::compact_deltas`], with `merges` deciding where a merge
    /// goes. A merge that fails leaves the catalog as it was.
    pub(crate) fn compact_deltas_of<M: MergeParts>(
        &mut self,
        merges: &mut M,
    ) -> Result<(), M::Error> {
        let mut parts = core::mem::take(&mut self.parts);
        match merged_from(&parts, 1, &self.live, merges) {
            Ok(only) => {
                parts.truncate(1);
                parts.extend(only);
                self.parts = parts;
                Ok(())
            }
            Err(e) => {
                self.parts = parts;
                Err(e)
            }
        }
    }

    /// Merge everything into one segment, dropping the postings of removed
    /// references, in one pass that rewrites each posting once. While it
    /// runs, the result is held next to the segments it replaces. A no-op
    /// when there is one segment and it has nothing to drop.
    pub fn compact(&mut self) {
        never(self.compact_of(&mut InMemory));
    }

    /// [`Catalog::compact`], with `merges` deciding where a merge goes. A
    /// merge that fails leaves the catalog as it was.
    pub(crate) fn compact_of<M: MergeParts>(&mut self, merges: &mut M) -> Result<(), M::Error> {
        let mut parts = core::mem::take(&mut self.parts);
        match merged_from(&parts, 0, &self.live, merges) {
            Ok(only) => {
                parts.clear();
                parts.extend(only);
                self.parts = parts;
                Ok(())
            }
            Err(e) => {
                self.parts = parts;
                Err(e)
            }
        }
    }
}

impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog")
            .field("references", &self.live.len())
            .field("live", &self.live_count)
            .field("postings", &self.posting_count())
            .field("segments", &self.parts.len())
            .finish_non_exhaustive()
    }
}

/// Segments `first..` of `parts` as one part without the postings `live`
/// rejects, or `None` if no posting is left. A lone segment with nothing to
/// drop is shared rather than copied.
///
/// One pass over all of them, so a posting is rewritten once however many
/// segments are merged. Only segments that may hold postings of removed
/// references are filtered; the others are copied run by run. `merges`
/// carries out the merge, in memory or to a file.
fn merged_from<M: MergeParts>(
    parts: &[Part],
    first: usize,
    live: &[bool],
    merges: &mut M,
) -> Result<Option<Part>, M::Error> {
    let tail = parts.get(first..).unwrap_or_default();
    match tail {
        [] => return Ok(None),
        [only] if !only.stale => return Ok(Some(only.clone())),
        _ => {}
    }
    let keep = |reference: u32| live[reference as usize];
    let sides: Vec<(&Segment, Option<&_>)> = tail
        .iter()
        .map(|part| (&*part.segment, part.stale.then_some(&keep)))
        .collect();
    let ids = tail.iter().fold((u32::MAX, 0), |(lo, hi), part| {
        (lo.min(part.ids.0), hi.max(part.ids.1))
    });
    merges.merge(&sides, ids)
}

/// Restore the size invariant from the tail of `parts`: the last segment
/// joins the one before it while it is at least half of it, and the two
/// together may then be half of the next. The segments such a cascade
/// reaches are merged in one pass rather than two at a time.
///
/// One merge is all it takes: [`tail_start`] stops at a part more than
/// twice the size of everything after it, and the merge of those comes out
/// no larger than their sum (removed references and duplicates only
/// shrink it). So a merge that fails has changed nothing, and its result
/// replaces the parts it joined only once it has returned.
fn merge_tail_in<M: MergeParts>(
    parts: &mut Vec<Part>,
    live: &[bool],
    merges: &mut M,
) -> Result<(), M::Error> {
    let Some(first) = tail_start(parts) else {
        return Ok(());
    };
    let only = merged_from(parts, first, live, merges)?;
    parts.truncate(first);
    parts.extend(only);
    debug_assert!(
        tail_is_settled(parts, first),
        "one merge restores the size rule"
    );
    Ok(())
}

/// Whether the part a merge left at `at`, if it left one, is less than
/// half of the part before it.
fn tail_is_settled(parts: &[Part], at: usize) -> bool {
    match (at.checked_sub(1).and_then(|i| parts.get(i)), parts.get(at)) {
        (Some(before), Some(merged)) => merged.segment.len() * 2 < before.segment.len(),
        _ => true,
    }
}

/// The first part a merge from the tail has to reach, or `None` when the
/// sizes already obey the half-size rule.
fn tail_start(parts: &[Part]) -> Option<usize> {
    let last = parts.last()?;
    let mut first = parts.len() - 1;
    // Sizes are below 2^40, so the sums and doublings fit.
    let mut size = last.segment.len() as u64;
    while first > 0 && size * 2 >= parts[first - 1].segment.len() as u64 {
        first -= 1;
        size += parts[first].segment.len() as u64;
    }
    (first + 1 != parts.len()).then_some(first)
}

/// Collects references and builds a [`Catalog`] with one sort.
///
/// Ids are assigned in call order starting at `0`, exactly as
/// [`Catalog::add`] would, so a builder-built catalog and an add-built one
/// agree on every id.
#[derive(Clone, Debug, Default)]
pub struct CatalogBuilder {
    rows: Vec<Posting>,
    next: u32,
}

impl CatalogBuilder {
    /// Add a reference from its `(hash, time)` rows and return its id.
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    pub fn add(&mut self, rows: impl IntoIterator<Item = (u32, u32)>) -> u32 {
        let id = self.next;
        assert!(id < u32::MAX, "reference ids exhausted");
        self.next = id + 1;
        self.rows
            .extend(rows.into_iter().map(|(hash, time)| Posting {
                hash,
                reference: id,
                time,
            }));
        id
    }

    /// Postings collected so far.
    #[must_use]
    pub fn posting_count(&self) -> usize {
        self.rows.len()
    }

    /// Sort everything once and build the catalog.
    #[must_use]
    pub fn build(self) -> Catalog {
        let n = self.next as usize;
        let mut cat = Catalog {
            parts: Vec::new(),
            live: alloc::vec![true; n],
            live_count: n,
        };
        let segment = Segment::from_postings(self.rows);
        // A posting implies a reference, so `next` is at least one here.
        if !segment.is_empty() {
            cat.parts.push(Part::new(segment, (0, self.next - 1)));
        }
        cat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(seed: u32, n: u32) -> Vec<(u32, u32)> {
        (0..n)
            .map(|i| (seed.wrapping_mul(31).wrapping_add(i * 7), i))
            .collect()
    }

    #[test]
    fn ids_are_dense_and_never_reused() {
        let mut cat = Catalog::new();
        assert_eq!(cat.add(rows(1, 10)), 0);
        assert_eq!(cat.add(rows(2, 10)), 1);
        assert!(cat.remove(0));
        assert!(!cat.remove(0));
        assert!(!cat.remove(9));
        assert_eq!(cat.add(rows(3, 10)), 2);
        assert_eq!(cat.live_count(), 2);
        assert_eq!(cat.reference_count(), 3);
        assert!(!cat.is_live(0));
        assert!(cat.is_live(1));
        assert!(!cat.is_live(3));
    }

    #[test]
    fn segment_sizes_stay_geometric() {
        let mut cat = Catalog::new();
        for i in 0..200 {
            cat.add(rows(i, 100));
            let sizes: Vec<usize> = cat.segments().map(Segment::len).collect();
            assert!(sizes.windows(2).all(|w| w[1] * 2 < w[0]), "sizes {sizes:?}");
        }
        assert!(cat.segment_count() <= 9, "{} segments", cat.segment_count());
        assert_eq!(cat.posting_count(), 200 * 100);
    }

    #[test]
    fn compaction_drops_dead_postings_and_keeps_live_ones() {
        let mut cat = Catalog::new();
        for i in 0..20 {
            cat.add(rows(i, 50));
        }
        for id in [0, 5, 19] {
            assert!(cat.remove(id));
        }
        let before = cat.posting_count();
        cat.compact_deltas();
        assert!(cat.segment_count() >= 1);
        cat.compact();
        assert_eq!(cat.segment_count(), 1);
        assert_eq!(cat.posting_count(), 17 * 50);
        assert!(cat.posting_count() < before);
        assert!(
            cat.segment(0)
                .unwrap()
                .postings()
                .all(|p| cat.is_live(p.reference))
        );
        // Compacting again is a no-op: the segment is the same allocation.
        // A clone keeps the old segment alive, so a rewrite cannot land on
        // the same address through allocator reuse.
        let keep = cat.clone();
        let before: *const Segment = cat.segment(0).unwrap();
        cat.compact();
        assert!(core::ptr::eq(before, cat.segment(0).unwrap()));
        // A removal makes the next compaction rewrite it once, not forever.
        assert!(cat.remove(7));
        cat.compact();
        assert!(!core::ptr::eq(before, cat.segment(0).unwrap()));
        drop(keep);
        let keep = cat.clone();
        let after: *const Segment = cat.segment(0).unwrap();
        cat.compact();
        assert!(core::ptr::eq(after, cat.segment(0).unwrap()));
        drop(keep);
        // Removing everything leaves no segments.
        for id in 0..20 {
            cat.remove(id);
        }
        cat.compact();
        assert_eq!(cat.segment_count(), 0);
        assert_eq!(cat.live_count(), 0);
    }

    #[test]
    fn compact_deltas_leaves_the_snapshot_alone() {
        // A snapshot far larger than the deltas, so `add` never merges into it.
        let mut b = Catalog::builder();
        for i in 0..400 {
            b.add(rows(i, 50));
        }
        let mut cat = b.build();
        let keep = cat.clone();
        let snapshot: *const Segment = cat.segment(0).unwrap();
        for i in 400..420 {
            cat.add(rows(i, 50));
        }
        cat.remove(401);
        assert!(cat.segment_count() > 2);
        cat.compact_deltas();
        assert_eq!(cat.segment_count(), 2);
        assert!(core::ptr::eq(snapshot, cat.segment(0).unwrap()));
        assert_eq!(cat.segment(1).unwrap().len(), 19 * 50);
        assert!(
            cat.segment(1)
                .unwrap()
                .postings()
                .all(|p| p.reference != 401)
        );
        drop(keep);
    }

    #[test]
    fn clone_shares_segments() {
        let mut cat = Catalog::new();
        cat.add(rows(1, 10));
        let copy = cat.clone();
        assert!(core::ptr::eq(
            cat.segment(0).unwrap(),
            copy.segment(0).unwrap()
        ));
        cat.add(rows(2, 10));
        assert_eq!(copy.segment_count(), 1);
        assert_eq!(copy.reference_count(), 1);
    }

    #[test]
    fn builder_and_incremental_agree_on_ids_and_postings() {
        let mut b = Catalog::builder();
        let mut inc = Catalog::new();
        for i in 0..7 {
            assert_eq!(b.add(rows(i, 30)), inc.add(rows(i, 30)));
        }
        assert_eq!(b.posting_count(), 7 * 30);
        let built = b.build();
        inc.compact();
        assert_eq!(built.segment(0), inc.segment(0));
    }

    /// The same references through `add_many` and through repeated `add`
    /// calls give equal ids, counts and (after compaction, which puts them
    /// in one segment either way) equal segments.
    #[test]
    fn add_many_agrees_with_repeated_add() {
        let references: Vec<Vec<(u32, u32)>> = (0..30u32).map(|i| rows(i, 40 + i % 7)).collect();
        let mut many = Catalog::new();
        let mut one = Catalog::new();
        let range = many.add_many(references.iter().map(|r| r.iter().copied()));
        assert_eq!(range, 0..30);
        for (i, r) in references.iter().enumerate() {
            assert_eq!(one.add(r.iter().copied()), i as u32);
        }
        assert_eq!(many.reference_count(), one.reference_count());
        assert_eq!(many.live_count(), one.live_count());
        assert_eq!(many.posting_count(), one.posting_count());
        many.compact();
        one.compact();
        assert_eq!(many.segment_count(), 1);
        assert_eq!(many.segment(0), one.segment(0));

        // A second batch continues the ids where the first stopped.
        let more: Vec<Vec<(u32, u32)>> = (0..5u32).map(|i| rows(100 + i, 30)).collect();
        assert_eq!(
            many.add_many(more.iter().map(|r| r.iter().copied())),
            30..35
        );
        for (i, r) in more.iter().enumerate() {
            assert_eq!(one.add(r.iter().copied()), 30 + i as u32);
        }
        many.compact();
        one.compact();
        assert_eq!(many.segment(0), one.segment(0));
    }

    /// An empty batch changes nothing and takes no id, but its range still
    /// starts at the id the next reference would get; a batch of empty
    /// references takes their ids and adds no postings, as repeated `add`
    /// calls would.
    #[test]
    fn add_many_covers_empty_batches_and_empty_references() {
        let mut cat = Catalog::new();
        let nothing: Vec<Vec<(u32, u32)>> = Vec::new();
        assert_eq!(cat.add_many(nothing), 0..0);
        assert_eq!(cat.reference_count(), 0);
        assert_eq!(cat.segment_count(), 0);
        assert_eq!(cat.live_count(), 0);

        assert_eq!(cat.add_many([Vec::new(), rows(1, 5), Vec::new()]), 0..3);
        assert_eq!(cat.reference_count(), 3);
        assert_eq!(cat.live_count(), 3);
        assert!(cat.is_live(0) && cat.is_live(2));
        assert_eq!(cat.posting_count(), 5);
        assert_eq!(cat.segment_count(), 1);
        assert_eq!(cat.segment(0).unwrap().max_reference(), Some(1));

        // A batch of nothing but empty references leaves no segment.
        let before = cat.segment_count();
        assert_eq!(cat.add_many([Vec::new(), Vec::new()]), 3..5);
        assert_eq!(cat.segment_count(), before);
        assert_eq!(cat.reference_count(), 5);

        // An empty batch now starts its range at the next id, 5, and the
        // next `add` still gets that id.
        assert_eq!(cat.add_many(Vec::<Vec<(u32, u32)>>::new()), 5..5);
        assert_eq!(cat.reference_count(), 5);
        assert_eq!(cat.add(rows(2, 3)), 5);
        assert_eq!(cat.add_many(Vec::<Vec<(u32, u32)>>::new()), 6..6);
    }

    #[test]
    fn from_segment_marks_every_id_live() {
        let seg = Segment::from_postings(alloc::vec![Posting::new(1, 4, 0), Posting::new(2, 1, 0)]);
        let mut cat = Catalog::from_segment(seg);
        assert_eq!(cat.reference_count(), 5);
        assert_eq!(cat.live_count(), 5);
        assert!(cat.is_live(2));
        assert_eq!(cat.add(rows(1, 3)), 5);
        let empty = Catalog::from_segment(Segment::default());
        assert_eq!(empty.reference_count(), 0);
        assert_eq!(empty.segment_count(), 0);
    }

    #[test]
    fn empty_references_take_an_id_but_no_segment() {
        let mut cat = Catalog::new();
        assert_eq!(cat.add(Vec::<(u32, u32)>::new()), 0);
        assert_eq!(cat.segment_count(), 0);
        assert_eq!(cat.live_count(), 1);
        assert!(cat.heap_bytes() >= 1);
        let dbg = alloc::format!("{cat:?}");
        assert!(dbg.contains("references: 1"));
    }

    #[test]
    fn compact_deltas_dedups_overlapping_segments() {
        // The public API keeps a reference in one segment; a crafted
        // snapshot need not. Overlapping deltas must compact to one copy
        // of each posting rather than a run with duplicates.
        let big = Segment::from_postings((0..100u32).map(|i| Posting::new(i, 0, i)).collect());
        let d1 = Segment::from_postings(alloc::vec![
            Posting::new(7, 1, 1),
            Posting::new(7, 1, 2),
            Posting::new(9, 2, 5),
        ]);
        let d2 = Segment::from_postings(alloc::vec![
            Posting::new(7, 1, 2),
            Posting::new(9, 2, 5),
            Posting::new(9, 2, 6),
        ]);
        let mut cat = Catalog {
            parts: [big, d1, d2]
                .into_iter()
                .map(|segment| Part::new(segment, (0, 2)))
                .collect(),
            live: alloc::vec![true; 3],
            live_count: 3,
        };
        cat.compact_deltas();
        assert_eq!(cat.segment_count(), 2);
        let delta: Vec<Posting> = cat.segment(1).unwrap().postings().collect();
        assert_eq!(
            delta,
            [
                Posting::new(7, 1, 1),
                Posting::new(7, 1, 2),
                Posting::new(9, 2, 5),
                Posting::new(9, 2, 6),
            ]
        );
        assert_eq!(cat.posting_count(), 104);
    }
}
