use alloc::vec::Vec;

/// A query clip's rows grouped by hash.
///
/// Hashes are distinct and ascending; each hash carries the times it
/// appeared at, in input order, capped at `max_times_per_hash`. A search
/// looks each hash up once and casts `times × postings` votes for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    hashes: Vec<u32>,
    /// `hashes.len() + 1` entries: hash `i`'s times are
    /// `times[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
    times: Vec<u32>,
    input_rows: usize,
}

impl Default for Query {
    /// The empty query: no hashes, no rows.
    fn default() -> Self {
        Self {
            hashes: Vec::new(),
            offsets: alloc::vec![0],
            times: Vec::new(),
            input_rows: 0,
        }
    }
}

impl Query {
    /// Group `(hash, time)` rows by hash, keeping each hash's first
    /// `max_times_per_hash` times in input order (`None` keeps all).
    ///
    /// Duplicate `(hash, time)` rows are kept: each is a query row and
    /// votes on its own.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = (u32, u32)>,
        max_times_per_hash: Option<usize>,
    ) -> Self {
        let mut rows: Vec<(u32, u32)> = pairs.into_iter().collect();
        let input_rows = rows.len();
        // A stable sort by hash groups a hash's rows in input order, which
        // makes "first N" deterministic.
        let hash_bits = rows.iter().fold(0, |bits, &(hash, _)| {
            bits.max(u32::BITS - hash.leading_zeros())
        });
        crate::sort::radix_sort_by_key(&mut rows, &mut Vec::new(), hash_bits, |&(hash, _)| {
            u64::from(hash)
        });
        let cap = max_times_per_hash.unwrap_or(usize::MAX);
        // The groups are counted first, so the arrays are allocated once
        // at their final size.
        let same_hash = |a: &(u32, u32), b: &(u32, u32)| a.0 == b.0;
        let (groups, kept) = rows
            .chunk_by(same_hash)
            .fold((0usize, 0usize), |(groups, kept), group| {
                (groups + 1, kept + group.len().min(cap))
            });
        let mut hashes = Vec::with_capacity(groups);
        let mut offsets = Vec::with_capacity(groups + 1);
        let mut times = Vec::with_capacity(kept);
        offsets.push(0);
        for group in rows.chunk_by(same_hash) {
            hashes.push(group[0].0);
            times.extend(group.iter().take(cap).map(|&(_, time)| time));
            offsets.push(times.len());
        }
        Self {
            hashes,
            offsets,
            times,
            input_rows,
        }
    }

    /// The distinct hashes, ascending.
    #[must_use]
    pub fn hashes(&self) -> &[u32] {
        &self.hashes
    }

    /// The kept times of hash `i` (an index into [`Query::hashes`]).
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.hash_count()`; [`Query::iter`] walks every hash
    /// without indexing.
    #[must_use]
    pub fn times(&self, i: usize) -> &[u32] {
        &self.times[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Every hash with its kept times, hashes ascending.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u32])> + '_ {
        self.hashes
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, self.times(i)))
    }

    /// Number of distinct hashes.
    #[must_use]
    pub fn hash_count(&self) -> usize {
        self.hashes.len()
    }

    /// Rows kept after the per-hash cap.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// `true` if no rows were kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Rows given to [`Query::from_pairs`] before the cap, the usual input
    /// to [`vote_floor`](crate::vote_floor).
    #[must_use]
    pub fn input_rows(&self) -> usize {
        self.input_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_in_input_order_and_caps_per_hash() {
        let q = Query::from_pairs([(9, 5), (3, 1), (9, 2), (3, 0), (9, 9), (3, 1)], Some(2));
        assert_eq!(q.hashes(), &[3, 9]);
        assert_eq!(q.times(0), &[1, 0]);
        assert_eq!(q.times(1), &[5, 2]);
        assert_eq!(q.hash_count(), 2);
        assert_eq!(q.len(), 4);
        assert_eq!(q.input_rows(), 6);
        assert!(!q.is_empty());
        let walked: Vec<(u32, &[u32])> = q.iter().collect();
        assert_eq!(walked, alloc::vec![(3, &[1, 0][..]), (9, &[5, 2][..])]);
        assert_eq!(q.iter().len(), 2);
    }

    #[test]
    fn no_cap_keeps_duplicates() {
        let q = Query::from_pairs([(1, 4), (1, 4), (1, 4)], None);
        assert_eq!(q.times(0), &[4, 4, 4]);
        let q = Query::from_pairs([(1, 4), (1, 4)], Some(0));
        assert_eq!(q.hashes(), &[1]);
        assert!(q.is_empty());
        assert_eq!(q.input_rows(), 2);
    }

    #[test]
    fn empty_input() {
        let q = Query::from_pairs(Vec::<(u32, u32)>::new(), Some(64));
        assert!(q.is_empty());
        assert_eq!(q.hash_count(), 0);
        assert_eq!(q.iter().count(), 0);
        assert_eq!(q, Query::default());
    }
}
