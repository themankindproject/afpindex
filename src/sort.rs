//! Radix sorts.
//!
//! Postings are sorted in place: a most-significant-digit radix sort on the
//! hash, up to eleven bits per level over the bits the hashes of a bucket
//! differ in, finished by a comparison sort inside each small bucket (which
//! also orders equal hashes by reference and time). No buffer beyond the
//! bucket tables, so building a segment from `n` rows never holds more
//! than the rows and the segment's arrays.
//!
//! Small records that must keep their order among equal keys (a query's
//! rows, a segment's runs) go through [`radix_sort_by_key`], a stable
//! least-significant-digit sort through a buffer of the same length.

use alloc::vec::Vec;

use crate::Posting;

/// Bits per level: 2048 buckets, so a 32-bit hash takes at most three
/// levels, and the lines the permutation is writing stay in cache.
const BITS: u32 = 11;
/// Buckets at or below this size are finished by a comparison sort.
const SMALL: usize = 96;
/// Rows per bucket a level aims for when it has fewer rows than buckets
/// to spread them over.
const FILL: usize = 8;
// A level needs at least two buckets to make progress.
const _: () = assert!(SMALL >= 2 * FILL);
/// Rows below which [`radix_sort_by_key`] leaves the work to the standard
/// stable sort, which needs no counters and no buffer for so few.
const RADIX_MIN: usize = 64;
/// Inputs at or above this size sort their buckets on every core (`std`).
#[cfg(feature = "std")]
const PARALLEL_MIN: usize = 1 << 18;

/// Stable sort of `rows` by `key`, whose values are below `2^key_bits`.
///
/// One read of the rows counts every digit; each digit that orders
/// anything then costs one sequential pass from the rows to `scratch` or
/// back, so the sort is linear and never chases a pointer. `scratch` is
/// grown to the rows' length and left holding garbage.
pub(crate) fn radix_sort_by_key<T: Copy>(
    rows: &mut [T],
    scratch: &mut Vec<T>,
    key_bits: u32,
    key: impl Fn(&T) -> u64,
) {
    let n = rows.len();
    let Some(&first) = rows.first() else { return };
    if n < RADIX_MIN {
        rows.sort_by_key(key);
        return;
    }
    // A pass costs the rows plus the counters, so few rows take narrow
    // digits.
    let digit = if n < 1 << 12 { 8 } else { BITS };
    let buckets = 1usize << digit;
    let mask = (buckets - 1) as u64;
    let passes = key_bits.div_ceil(digit) as usize;
    let mut counts = alloc::vec![0usize; passes * buckets];
    for row in rows.iter() {
        let mut k = key(row);
        for pass in 0..passes {
            counts[pass * buckets + (k & mask) as usize] += 1;
            k >>= digit;
        }
    }
    if scratch.len() < n {
        scratch.resize(n, first);
    }
    let (mut from, mut to) = (&mut *rows, &mut scratch[..n]);
    let mut swapped = false;
    for (pass, counts) in counts.chunks_exact_mut(buckets).enumerate() {
        // Every row in one bucket: the digit orders nothing.
        if counts.contains(&n) {
            continue;
        }
        let mut at = 0;
        for count in counts.iter_mut() {
            at += core::mem::replace(count, at);
        }
        let shift = pass as u32 * digit;
        for row in from.iter() {
            let next = &mut counts[((key(row) >> shift) & mask) as usize];
            to[*next] = *row;
            *next += 1;
        }
        (from, to) = (to, from);
        swapped = !swapped;
    }
    if swapped {
        // An odd number of passes left the sorted rows in the scratch.
        to.copy_from_slice(from);
    }
}

/// Sort `rows` ascending by `(hash, reference, time)`.
pub(crate) fn sort_postings(rows: &mut [Posting]) {
    msd(rows, u32::BITS, true);
}

/// Sort `rows`, whose hashes agree at bit `hi` and above. Only the top
/// level may fan out to threads (`top`); a bucket sorted by a worker
/// recurses on that worker, so the thread count stays bounded.
fn msd(rows: &mut [Posting], hi: u32, top: bool) {
    // The bits the hashes really differ in: sparse hashes leave a bucket
    // agreeing on more of them than the levels above consumed, and a level
    // spent on those would order nothing.
    let hi = match rows.first() {
        Some(first) if hi > 0 => {
            let differ = rows.iter().fold(0, |bits, p| bits | (p.hash ^ first.hash));
            u32::BITS - differ.leading_zeros()
        }
        _ => 0,
    };
    if hi == 0 {
        // One hash: what is left of the order is a single comparison.
        rows.sort_unstable_by_key(|p| u64::from(p.reference) << 32 | u64::from(p.time));
        return;
    }
    if rows.len() <= SMALL {
        rows.sort_unstable();
        return;
    }
    // A level costs its rows plus its buckets, so few rows take a narrow
    // digit: about `FILL` rows to a bucket, which the comparison sort
    // finishes.
    let bits = BITS.min(hi).min((rows.len() / FILL).ilog2());
    let shift = hi - bits;
    let buckets = 1usize << bits;
    let mask = (buckets - 1) as u32;
    let digit = |p: &Posting| ((p.hash >> shift) & mask) as usize;

    let mut start = alloc::vec![0usize; buckets + 1];
    for p in rows.iter() {
        start[digit(p) + 1] += 1;
    }
    for b in 0..buckets {
        start[b + 1] += start[b];
    }

    // American flag permutation: every row not yet known to be in place
    // is swapped to the head of its own bucket. The row a swap brings back
    // is left for the next round rather than followed, so which rows a
    // round reads does not depend on what its swaps load, and the cache
    // misses of the bucket heads overlap instead of queueing behind one
    // another. Each swap puts one row in place for good, so the rounds
    // together make exactly one swap per row.
    let mut head: Vec<usize> = start[..buckets].to_vec();
    let mut open: Vec<usize> = (0..buckets).filter(|&b| head[b] < start[b + 1]).collect();
    while !open.is_empty() {
        open.retain(|&b| {
            let end = start[b + 1];
            for i in head[b]..end {
                let d = digit(&rows[i]);
                rows.swap(i, head[d]);
                head[d] += 1;
            }
            head[b] < end
        });
    }

    #[cfg(feature = "std")]
    if top && rows.len() >= PARALLEL_MIN {
        parallel::sort_buckets(rows, &start, shift);
        return;
    }
    #[cfg(not(feature = "std"))]
    let _ = top;
    sort_buckets(rows, &start, shift);
}

/// Sort every bucket of `rows` (bucket `b` is `start[b]..start[b + 1]`,
/// relative to `start[0]`) on the next radix level.
fn sort_buckets(rows: &mut [Posting], start: &[usize], shift: u32) {
    let base = start[0];
    for w in start.windows(2) {
        if w[1] - w[0] > 1 {
            msd(&mut rows[w[0] - base..w[1] - base], shift, false);
        }
    }
}

#[cfg(feature = "std")]
mod parallel {
    use alloc::vec::Vec;
    use std::sync::Mutex;

    use super::Posting;
    use crate::par::{lock, run_tasks, threads};

    /// A range of the buckets to sort, and the bounds of its buckets.
    type Piece<'a> = (&'a mut [Posting], &'a [usize]);

    /// Sort the buckets of `rows` on every core: the buckets are cut into
    /// contiguous ranges holding about equal shares of the rows, and each
    /// range is a task that [`run_tasks`] hands out. The calling thread works
    /// too, so a thread that cannot be spawned costs nothing but
    /// parallelism.
    pub(super) fn sort_buckets(rows: &mut [Posting], start: &[usize], shift: u32) {
        let threads = threads();
        if threads <= 1 {
            super::sort_buckets(rows, start, shift);
            return;
        }
        let total = rows.len();
        let pieces = threads * 4;
        let mut items: Vec<Piece<'_>> = Vec::with_capacity(pieces);
        let mut rest = rows;
        let mut b = 0;
        for piece in 1..=pieces {
            let target = total * piece / pieces;
            let mut b_end = b;
            while b_end + 1 < start.len() && start[b_end] < target {
                b_end += 1;
            }
            if piece == pieces {
                b_end = start.len() - 1;
            }
            if b_end == b {
                continue;
            }
            let (chunk, tail) = rest.split_at_mut(start[b_end] - start[b]);
            rest = tail;
            items.push((chunk, &start[b..=b_end]));
            b = b_end;
        }
        // One slot per piece, which the task that takes it empties, so the
        // pool hands every piece to exactly one task. Each piece carries
        // its own range of the rows, so the order the tasks take them in
        // does not matter.
        let slots: Vec<Mutex<Option<Piece<'_>>>> = items
            .into_iter()
            .map(|item| Mutex::new(Some(item)))
            .collect();
        run_tasks(threads, slots.len(), |piece| {
            let (chunk, bounds) = lock(&slots[piece])
                .take()
                .expect("each piece is sorted once");
            super::sort_buckets(chunk, bounds, shift);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::Rng;

    fn check(mut rows: Vec<Posting>) {
        let mut expected = rows.clone();
        expected.sort_unstable();
        sort_postings(&mut rows);
        assert_eq!(rows, expected);
    }

    #[test]
    fn radix_sort_by_key_is_stable_at_every_size_and_width() {
        let mut rng = Rng(0x0DDB_A110);
        let mut scratch = Vec::new();
        // Both digit widths, an odd and an even number of passes, digits
        // that order nothing, and keys up to 64 bits.
        for (n, key_bits) in [
            (0usize, 32u32),
            (1, 32),
            (2, 1),
            (RADIX_MIN - 1, 40),
            (RADIX_MIN, 40),
            (700, 5),
            (700, 22),
            (3_000, 64),
            (5_000, 11),
            (5_000, 13),
            (40_000, 33),
            (40_000, 64),
        ] {
            let mask = u64::MAX >> (64 - key_bits);
            // The sequence number tells equal keys apart; a few high bits
            // are held constant so that some digit orders nothing.
            let mut rows: Vec<(u64, usize)> = (0..n)
                .map(|i| ((rng.next() & mask & !(0b111 << 8)) >> (i % 3), i))
                .collect();
            let mut want = rows.clone();
            want.sort_by_key(|r| r.0);
            radix_sort_by_key(&mut rows, &mut scratch, key_bits, |r| r.0);
            assert_eq!(rows, want, "{n} rows, {key_bits} bits");
        }
    }

    #[test]
    fn agrees_with_the_comparison_sort() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let mut rows_with = |n: usize, hash: &mut dyn FnMut(u64) -> u32| -> Vec<Posting> {
            (0..n)
                .map(|_| {
                    let r = rng.next();
                    Posting::new(hash(r), (r >> 32) as u32 % 50, (r >> 40) as u32 % 1000)
                })
                .collect()
        };
        check(Vec::new());
        check(rows_with(1, &mut |r| r as u32));
        // Wide hashes, many more than one bucket's worth of rows.
        check(rows_with(300_000, &mut |r| r as u32));
        // Few distinct hashes: deep buckets, finished by reference and time.
        check(rows_with(50_000, &mut |r| (r % 7) as u32));
        // Hashes that vary only in their low bits, and only in their high.
        check(rows_with(50_000, &mut |r| (r % 300) as u32));
        check(rows_with(50_000, &mut |r| ((r % 300) as u32) << 23));
        // Both extremes of the u32 range, with duplicates.
        let mut rows = rows_with(20_000, &mut |r| r as u32);
        rows.extend([Posting::new(0, 0, 0), Posting::new(u32::MAX, 0, 0)]);
        rows.extend_from_slice(&rows.clone()[..100]);
        check(rows);
    }

    #[test]
    fn agrees_with_the_comparison_sort_on_awkward_shapes() {
        let mut rng = Rng(0x0BAD_5EED_0BAD_5EED);
        let mut rows_with = |n: usize, hash: &mut dyn FnMut(u64) -> u32| -> Vec<Posting> {
            (0..n)
                .map(|_| {
                    let r = rng.next();
                    Posting::new(hash(r), (r >> 32) as u32 % 50, (r >> 40) as u32 % 1000)
                })
                .collect()
        };
        // Every size around the hand-over to the comparison sort and the
        // narrow digits above it, with hashes over the whole range.
        for n in (SMALL - 2..SMALL + 40).chain([500, 1_500, 5_000, 17_000]) {
            check(rows_with(n, &mut |r| r as u32));
        }
        // A prefix every hash shares, above bits that vary.
        check(rows_with(40_000, &mut |r| 0xA5A0_0000 | (r % 5_000) as u32));
        // Few hashes scattered over the range, many rows each: a bucket
        // soon holds one hash, long before the hash bits run out.
        check(rows_with(60_000, &mut |r| {
            ((r % 400) as u32).wrapping_mul(0x9E37_79B1)
        }));
        // One hash with most of the rows, the rest spread thin.
        check(rows_with(60_000, &mut |r| {
            if r % 10 == 0 { (r >> 8) as u32 } else { 7 }
        }));
        // Rows that arrive sorted, reversed, and as two sorted halves
        // interleaved bucket against bucket.
        let mut rows = rows_with(50_000, &mut |r| (r % 20_000) as u32);
        rows.sort_unstable();
        check(rows.clone());
        rows.reverse();
        check(rows.clone());
        let (low, high) = rows.split_at(rows.len() / 2);
        check(high.iter().chain(low).copied().collect());
    }
}
