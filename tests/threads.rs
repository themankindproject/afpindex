//! The thread cap (`std` feature): a pass computes the same thing whatever
//! cap it runs under.
//!
//! The setting is process-wide, so this file holds a single test: another
//! one changing the cap while this runs would make its comparisons
//! meaningless. The work is sized so that every pass takes its parallel
//! path with no cap: sorts of more than 2^18 rows, a merge of more than
//! 2^20 postings, directories of more than 2^20 slots, and a segment of
//! more than 2^22 postings to walk on load. The same work then runs with
//! the cap at 1, which keeps every pass on the calling thread, and the two
//! must agree.
#![cfg(feature = "std")]

use hashtally::{
    Candidate, Catalog, Posting, Query, Scratch, SearchOptions, Segment, max_threads,
    set_max_threads,
};

/// A segment this large has its runs walked on every core when it is
/// loaded, and its directory filled on every core (it has more than 2^20
/// keys).
const BIG: usize = (1 << 22) + 60_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `n` postings from references `0..refs`, over `hashes` hashes, in an
/// order the sort has work to do on.
fn rows(seed: u64, n: usize, hashes: u64, refs: u32) -> Vec<Posting> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|i| {
            Posting::new(
                (rng.next() % hashes) as u32,
                (i as u32).wrapping_mul(2_654_435_761) % refs,
                rng.next() as u32 % 4_000,
            )
        })
        .collect()
}

fn query(hashes: u64) -> Query {
    let mut rng = Rng(0x0BAD_5EED);
    Query::from_pairs(
        (0..600).map(|_| ((rng.next() % hashes) as u32, rng.next() as u32 % 1_000)),
        Some(64),
    )
}

/// A catalog of several segments over more than 2^20 postings in its
/// deltas alone, so that both compactions take the parallel path: a base,
/// references added on top of it, and some of those removed so the
/// compactions have postings to drop.
fn catalog() -> Catalog {
    let mut builder = Catalog::builder();
    for i in 0..45u32 {
        builder.add(
            rows(u64::from(i) + 1, 12_000, 4_000, 500)
                .into_iter()
                .map(|p| (p.hash, p.time + i)),
        );
    }
    let mut cat = builder.build();
    for i in 45..200u32 {
        cat.add(
            rows(u64::from(i) + 1, 8_000, 4_000, 500)
                .into_iter()
                .map(|p| (p.hash, p.time + i)),
        );
    }
    for id in (0..200u32).step_by(9) {
        cat.remove(id);
    }
    // A merge takes threads from 2^20 postings on, and each compaction
    // merges everything it is given: the base for `compact`, the deltas
    // for `compact_deltas`.
    assert!(
        cat.posting_count() > 1 << 20,
        "both compactions take threads"
    );
    cat
}

fn segments(cat: &Catalog) -> Vec<Segment> {
    cat.segments().cloned().collect()
}

/// Everything the passes compute, for one cap.
struct Output {
    sorted: Segment,
    merged: Segment,
    loaded: Segment,
    deltas: Vec<Segment>,
    compacted: Vec<Segment>,
    candidates: Vec<Candidate>,
}

fn work(big: &[u8], base: &Catalog) -> Output {
    let a = Segment::from_postings(rows(1, 700_000, 300_000, 4_000));
    let b = Segment::from_postings(rows(2, 700_000, 300_000, 4_000));
    let merged = Segment::merge(&a, &b, |r| r % 5 != 0);
    let loaded = Segment::read_from(big).expect("a valid snapshot");
    assert_eq!(loaded.len(), BIG);
    let mut deltas = base.clone();
    deltas.compact_deltas();
    let mut compacted = base.clone();
    compacted.compact();
    let candidates = compacted
        .search(
            &query(4_000),
            &SearchOptions::default(),
            &mut Scratch::new(),
        )
        .expect("a search within the vote limit");
    assert!(!candidates.is_empty(), "the query finds something");
    Output {
        sorted: a,
        merged,
        loaded,
        deltas: segments(&deltas),
        compacted: segments(&compacted),
        candidates,
    }
}

#[test]
fn the_cap_changes_only_how_many_threads_a_pass_takes() {
    let hardware = max_threads();
    assert!(hardware >= 1, "at least one thread");

    // Written once and loaded under both caps.
    let big = Segment::from_postings(rows(3, BIG, 400_000, 50_000));
    let mut bytes = Vec::new();
    big.write_to(&mut bytes).expect("writing to memory");
    drop(big);
    let base = catalog();

    set_max_threads(0);
    let uncapped = work(&bytes, &base);
    set_max_threads(1);
    assert_eq!(max_threads(), 1, "the cap is what a pass takes");
    let capped = work(&bytes, &base);
    set_max_threads(0);

    // The sort, the directory and the merge give the same posting for
    // posting, and so does the walk over a loaded segment's runs.
    assert_eq!(capped.sorted, uncapped.sorted, "sort");
    assert_eq!(capped.merged, uncapped.merged, "merge");
    assert_eq!(capped.loaded, uncapped.loaded, "load");
    // The compactions, and the search over their result.
    for (capped, uncapped, what) in [
        (&capped.deltas, &uncapped.deltas, "compact_deltas"),
        (&capped.compacted, &uncapped.compacted, "compact"),
    ] {
        assert_eq!(capped.len(), uncapped.len(), "{what}: segments");
        for (i, (a, b)) in capped.iter().zip(uncapped).enumerate() {
            assert_eq!(a, b, "{what}: segment {i}");
        }
    }
    assert_eq!(capped.candidates, uncapped.candidates, "search");
    assert_eq!(max_threads(), hardware, "the cap is off again");
}
