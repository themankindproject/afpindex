//! The engine against a naive model of the same semantics, on random
//! catalogs and queries, through every public way of building a catalog
//! and through random add / remove / compact sequences.
//!
//! The model: for each query hash, the kept query times; for each
//! `(hash, reference)` the first `fanout` postings by time; one vote per
//! `(posting, query time)` at `Δ = time − query_time`; bins per
//! `(reference, Δ)` with first and last query time; references below the
//! floor or removed are dropped; the best bin and noise statistics follow
//! `Candidate::from_bins`.

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use afpindex::{Candidate, Catalog, Posting, Query, Scratch, SearchError, SearchOptions, Segment};

/// What the model predicts for one candidate. The engine's `Candidate` is
/// `#[non_exhaustive]`, so it is compared field by field.
#[derive(Debug, Clone, PartialEq)]
struct Expected {
    reference: u32,
    total_votes: u32,
    delta: i64,
    votes: u32,
    first_time: u32,
    last_time: u32,
    noise_bins: usize,
    noise_mean: f64,
    noise_std_dev: f64,
    noise_max: u32,
}

/// `SearchOptions` from its fields (the type cannot be built with a
/// literal).
fn options(
    fanout: Option<usize>,
    min_total_votes: u32,
    offset_tolerance: u32,
    min_noise_bins: usize,
    max_votes: usize,
) -> SearchOptions {
    let mut o = SearchOptions::default();
    o.fanout = fanout;
    o.min_total_votes = min_total_votes;
    o.offset_tolerance = offset_tolerance;
    o.min_noise_bins = min_noise_bins;
    o.max_votes = max_votes;
    o
}

struct Rng(u64);

impl Rng {
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

/// Grouped query as the engine sees it: `(hash, kept times in input order)`.
fn group(pairs: &[(u32, u32)], cap: Option<usize>) -> Vec<(u32, Vec<u32>)> {
    let mut by_hash: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(h, t) in pairs {
        by_hash.entry(h).or_default().push(t);
    }
    by_hash
        .into_iter()
        .map(|(h, ts)| (h, ts.into_iter().take(cap.unwrap_or(usize::MAX)).collect()))
        .collect()
}

fn model(
    rows: &[Posting],
    live: impl Fn(u32) -> bool,
    q: &[(u32, Vec<u32>)],
    o: &SearchOptions,
) -> Vec<Expected> {
    let mut by_hash: BTreeMap<u32, BTreeMap<u32, Vec<u32>>> = BTreeMap::new();
    for p in rows {
        by_hash
            .entry(p.hash)
            .or_default()
            .entry(p.reference)
            .or_default()
            .push(p.time);
    }
    let mut bins: BTreeMap<u32, BTreeMap<i64, (u32, u32, u32)>> = BTreeMap::new();
    for (h, qts) in q {
        let Some(refs) = by_hash.get(h) else { continue };
        for (&r, ts) in refs {
            let mut ts = ts.clone();
            ts.sort_unstable();
            ts.dedup();
            let take = o.fanout.unwrap_or(ts.len());
            for &t in ts.iter().take(take) {
                for &qt in qts {
                    let e = bins
                        .entry(r)
                        .or_default()
                        .entry(i64::from(t) - i64::from(qt))
                        .or_insert((0, u32::MAX, 0));
                    e.0 += 1;
                    e.1 = e.1.min(qt);
                    e.2 = e.2.max(qt);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (r, b) in bins {
        let total: u64 = b.values().map(|v| u64::from(v.0)).sum();
        if total < u64::from(o.min_total_votes) || !live(r) {
            continue;
        }
        let list: Vec<(i64, u32, u32, u32)> = b
            .iter()
            .map(|(&d, &(n, first, last))| (d, n, first, last))
            .collect();
        let mut best = 0;
        for k in 1..list.len() {
            if list[k].1 > list[best].1 || (list[k].1 == list[best].1 && list[k].0 < list[best].0) {
                best = k;
            }
        }
        let bd = list[best].0;
        let tol = u64::from(o.offset_tolerance);
        let noise: Vec<f64> = list
            .iter()
            .filter(|x| x.0.abs_diff(bd) > tol)
            .map(|x| f64::from(x.1))
            .collect();
        let (noise_mean, noise_std_dev, noise_max) =
            if !noise.is_empty() && noise.len() >= o.min_noise_bins {
                let m = noise.iter().sum::<f64>() / noise.len() as f64;
                let v = noise.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / noise.len() as f64;
                (
                    m,
                    v.sqrt(),
                    noise.iter().copied().fold(0.0, f64::max) as u32,
                )
            } else {
                (0.0, 0.0, 0)
            };
        out.push(Expected {
            reference: r,
            total_votes: total as u32,
            delta: bd,
            votes: list[best].1,
            first_time: list[best].2,
            last_time: list[best].3,
            noise_bins: noise.len(),
            noise_mean,
            noise_std_dev,
            noise_max,
        });
    }
    out
}

fn assert_same(got: &[Candidate], want: &[Expected], what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: candidate count\n got {got:?}\nwant {want:?}"
    );
    for (g, w) in got.iter().zip(want) {
        assert_eq!(
            (
                g.reference,
                g.total_votes,
                g.delta,
                g.votes,
                g.first_time,
                g.last_time,
                g.noise.bins,
                g.noise.max
            ),
            (
                w.reference,
                w.total_votes,
                w.delta,
                w.votes,
                w.first_time,
                w.last_time,
                w.noise_bins,
                w.noise_max
            ),
            "{what}: reference {}",
            w.reference
        );
        for (a, b) in [
            (g.noise.mean, w.noise_mean),
            (g.noise.std_dev, w.noise_std_dev),
        ] {
            assert!(
                (a - b).abs() <= 1e-9 * a.abs().max(1.0),
                "{what}: {a} vs {b}"
            );
        }
    }
}

/// Hash index to hash. `spread` maps indices over the whole `u32` range
/// (an odd multiplier is a bijection), so lookups cross every radix bucket.
fn hash_of(index: u32, spread: bool) -> u32 {
    if spread {
        index.wrapping_mul(0x9E37_79B1) ^ 0x5A5A_0000
    } else {
        index
    }
}

/// Rows of `refs` references over `hashes` hashes; some `(hash, reference)`
/// pairs get bursts past the fanout, and some rows repeat exactly.
fn random_rows(rng: &mut Rng, refs: u32, hashes: u32, n: usize, spread: bool) -> Vec<Posting> {
    let mut rows = Vec::new();
    for _ in 0..n {
        let h = hash_of(rng.below(u64::from(hashes)) as u32, spread);
        let r = rng.below(u64::from(refs)) as u32;
        let burst = if rng.below(50) == 0 { 90 } else { 1 };
        for _ in 0..burst {
            rows.push(Posting::new(h, r, rng.below(3_000) as u32));
        }
        if rng.below(20) == 0 {
            let last = *rows.last().unwrap();
            rows.push(last);
        }
    }
    rows
}

fn random_query(rng: &mut Rng, hashes: u32, n: usize, spread: bool) -> Vec<(u32, u32)> {
    let mut q = Vec::new();
    for _ in 0..n {
        let h = hash_of(rng.below(u64::from(hashes)) as u32, spread);
        let burst = if rng.below(40) == 0 { 80 } else { 1 };
        for _ in 0..burst {
            q.push((h, rng.below(2_000) as u32));
        }
    }
    q
}

fn dedup(rows: &[Posting]) -> Vec<Posting> {
    let mut v = rows.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

fn option_sets() -> [SearchOptions; 5] {
    [
        options(Some(64), 1, 2, 10, usize::MAX),
        options(None, 0, 0, 0, usize::MAX),
        options(Some(1), 16, 5, 3, usize::MAX),
        options(Some(3), 40, 2, 10, usize::MAX),
        options(Some(64), 200, 2, 10, usize::MAX),
    ]
}

#[test]
fn matches_the_model_on_random_catalogs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut scratch = Scratch::new();
    for case in 0..60 {
        let spread = case % 2 == 1;
        let refs = 1 + rng.below(40) as u32;
        let hashes = 1 + rng.below(200) as u32;
        let n_rows = 200 + rng.below(3_000) as usize;
        let rows = random_rows(&mut rng, refs, hashes, n_rows, spread);
        // Every construction path must agree with the model.
        let from_segment = Catalog::from_segment(Segment::from_postings(rows.clone()));
        let mut builder = Catalog::builder();
        let mut incremental = Catalog::new();
        for r in 0..refs {
            let mine: Vec<(u32, u32)> = rows
                .iter()
                .filter(|p| p.reference == r)
                .map(|p| (p.hash, p.time))
                .collect();
            assert_eq!(builder.add(mine.iter().copied()), r);
            assert_eq!(incremental.add(mine), r);
        }
        let built = builder.build();
        let n_query = 1 + rng.below(400) as usize;
        let pairs = random_query(&mut rng, hashes, n_query, spread);
        let cap = [Some(64), None, Some(2)][case % 3];
        let query = Query::from_pairs(pairs.iter().copied(), cap);
        let grouped = group(&pairs, cap);
        let clean = dedup(&rows);
        for (oi, o) in option_sets().iter().enumerate() {
            let want = model(&clean, |_| true, &grouped, o);
            for (name, cat) in [
                ("from_segment", &from_segment),
                ("builder", &built),
                ("incremental", &incremental),
            ] {
                let got = cat.search(&query, o, &mut scratch).unwrap();
                assert_same(&got, &want, &format!("case {case} options {oi} {name}"));
            }
        }
    }
}

#[test]
fn search_with_hands_out_the_bins_the_candidates_came_from() {
    let mut rng = Rng(7);
    let rows = random_rows(&mut rng, 25, 120, 2_500, true);
    let cat = Catalog::from_segment(Segment::from_postings(rows));
    let pairs = random_query(&mut rng, 120, 300, true);
    let query = Query::from_pairs(pairs, Some(64));
    let mut scratch = Scratch::new();
    for o in &option_sets() {
        let direct = cat.search(&query, o, &mut scratch).unwrap();
        let mut visited = Vec::new();
        let mut last_ref = None;
        cat.search_with(&query, o, &mut scratch, |s| {
            assert!(last_ref.is_none_or(|l| l < s.reference), "survivors ascend");
            last_ref = Some(s.reference);
            assert!(
                s.bins.windows(2).all(|w| w[0].delta < w[1].delta),
                "bins ascend"
            );
            assert_eq!(s.bins.iter().map(|b| b.votes).sum::<u32>(), s.total_votes);
            assert_eq!(
                Some(s.candidate(o)),
                Candidate::from_bins(s.reference, s.total_votes, s.bins, o)
            );
            visited.push(s.candidate(o));
        })
        .unwrap();
        assert_eq!(direct, visited);
    }
}

/// Live rows per reference, the model's view of a mutating catalog.
type Live = BTreeMap<u32, Vec<Posting>>;

fn reference_rows(rng: &mut Rng, hashes: u32) -> Vec<(u32, u32)> {
    let n = 50 + rng.below(400) as usize;
    (0..n)
        .map(|_| {
            (
                hash_of(rng.below(u64::from(hashes)) as u32, true),
                rng.below(3_000) as u32,
            )
        })
        .collect()
}

/// Record a reference's rows in the model under `id`.
fn record(live: &mut Live, id: u32, rows: &[(u32, u32)]) {
    let mut mine: Vec<Posting> = rows.iter().map(|&(h, t)| Posting::new(h, id, t)).collect();
    mine.sort_unstable();
    mine.dedup();
    live.insert(id, mine);
}

fn add_reference(cat: &mut Catalog, live: &mut Live, rng: &mut Rng, hashes: u32) -> u32 {
    let rows = reference_rows(rng, hashes);
    let id = cat.add(rows.iter().copied());
    record(live, id, &rows);
    id
}

/// One `add_many` of a random batch, recorded in the model.
fn add_batch(cat: &mut Catalog, live: &mut Live, rng: &mut Rng, hashes: u32) -> Vec<u32> {
    let n = 1 + rng.below(6) as usize;
    let batch: Vec<Vec<(u32, u32)>> = (0..n).map(|_| reference_rows(rng, hashes)).collect();
    let ids = cat.add_many(batch.iter().map(|r| r.iter().copied()));
    for (id, rows) in ids.clone().zip(&batch) {
        record(live, id, rows);
    }
    ids.collect()
}

fn check_against_model(
    cat: &Catalog,
    live: &Live,
    scratch: &mut Scratch,
    what: &str,
    rng: &mut Rng,
    hashes: u32,
) {
    let rows: Vec<Posting> = live.values().flatten().copied().collect();
    let pairs = random_query(rng, hashes, 300, true);
    let query = Query::from_pairs(pairs.iter().copied(), Some(64));
    let grouped = group(&pairs, Some(64));
    for (oi, o) in option_sets().iter().enumerate() {
        let want = model(&rows, |r| cat.is_live(r), &grouped, o);
        let got = cat.search(&query, o, scratch).unwrap();
        assert_same(&got, &want, &format!("{what} options {oi}"));
        assert!(got.iter().all(|c| cat.is_live(c.reference)));
    }
}

#[test]
fn additions_removals_and_compaction_match_the_model() {
    let mut rng = Rng(42);
    let (refs, hashes) = (30u32, 150u32);
    let base = random_rows(&mut rng, refs, hashes, 4_000, true);
    let mut cat = Catalog::from_segment(Segment::from_postings(base.clone()));
    let mut live: Live = BTreeMap::new();
    for p in dedup(&base) {
        live.entry(p.reference).or_default().push(p);
    }
    for _ in 0..40 {
        add_reference(&mut cat, &mut live, &mut rng, hashes);
    }
    for id in [3u32, 31, 45, 69, 5] {
        assert!(cat.remove(id));
        live.remove(&id);
    }
    assert!(!cat.remove(31));
    let mut scratch = Scratch::new();
    check_against_model(
        &cat,
        &live,
        &mut scratch,
        "before compaction",
        &mut rng,
        hashes,
    );
    assert!(cat.segment_count() > 1);
    cat.compact_deltas();
    assert!(cat.segment_count() <= 2);
    check_against_model(
        &cat,
        &live,
        &mut scratch,
        "after compact_deltas",
        &mut rng,
        hashes,
    );
    cat.compact();
    assert_eq!(cat.segment_count(), 1);
    assert_eq!(
        cat.posting_count(),
        live.values().map(Vec::len).sum::<usize>()
    );
    check_against_model(&cat, &live, &mut scratch, "after compact", &mut rng, hashes);
    // Keep adding after compaction; ids continue.
    let next = add_reference(&mut cat, &mut live, &mut rng, hashes);
    assert_eq!(next, refs + 40);
    check_against_model(
        &cat,
        &live,
        &mut scratch,
        "after compaction and add",
        &mut rng,
        hashes,
    );
}

#[test]
fn random_mutation_sequences_match_the_model() {
    let mut rng = Rng(0x00C0_FFEE);
    let hashes = 120u32;
    let mut scratch = Scratch::new();
    for seq in 0..6 {
        let mut cat = Catalog::new();
        let mut live: Live = BTreeMap::new();
        let mut ids: Vec<u32> = Vec::new();
        for step in 0..60 {
            match rng.below(12) {
                0..=3 => ids.push(add_reference(&mut cat, &mut live, &mut rng, hashes)),
                4..=5 => ids.extend(add_batch(&mut cat, &mut live, &mut rng, hashes)),
                6..=7 if !ids.is_empty() => {
                    let id = ids[rng.below(ids.len() as u64) as usize];
                    let was_live = live.remove(&id).is_some();
                    assert_eq!(cat.remove(id), was_live);
                }
                8 => cat.compact_deltas(),
                9 => cat.compact(),
                _ => {}
            }
            assert_eq!(cat.live_count(), live.len());
            assert_eq!(cat.reference_count(), ids.len());
            assert!(cat.posting_count() >= live.values().map(Vec::len).sum::<usize>());
            if step % 7 == 6 {
                check_against_model(
                    &cat,
                    &live,
                    &mut scratch,
                    &format!("seq {seq} step {step}"),
                    &mut rng,
                    hashes,
                );
            }
        }
        cat.compact();
        assert!(cat.segment_count() <= 1);
        assert_eq!(
            cat.posting_count(),
            live.values().map(Vec::len).sum::<usize>()
        );
        check_against_model(
            &cat,
            &live,
            &mut scratch,
            &format!("seq {seq} end"),
            &mut rng,
            hashes,
        );
    }
}

#[test]
fn results_are_deterministic_and_scratch_is_reusable_across_catalogs() {
    let mut rng = Rng(99);
    let a = Catalog::from_segment(Segment::from_postings(random_rows(
        &mut rng, 40, 100, 3_000, true,
    )));
    let b = Catalog::from_segment(Segment::from_postings(random_rows(
        &mut rng, 5, 100, 500, false,
    )));
    let query = Query::from_pairs(random_query(&mut rng, 100, 200, true), Some(64));
    let o = SearchOptions::default();
    let mut s1 = Scratch::new();
    let mut s2 = Scratch::new();
    let first_a = a.search(&query, &o, &mut s1).unwrap();
    let first_b = b.search(&query, &o, &mut s1).unwrap();
    for _ in 0..5 {
        assert_eq!(a.search(&query, &o, &mut s1).unwrap(), first_a);
        assert_eq!(b.search(&query, &o, &mut s2).unwrap(), first_b);
        assert_eq!(a.search(&query, &o, &mut s2).unwrap(), first_a);
    }
    assert_ne!(first_a.len(), 0);
}

#[test]
fn a_panicking_visitor_leaves_the_scratch_clean() {
    let mut rng = Rng(5);
    let cat = Catalog::from_segment(Segment::from_postings(random_rows(
        &mut rng, 20, 100, 2_000, true,
    )));
    let query = Query::from_pairs(random_query(&mut rng, 100, 200, true), Some(64));
    let o = SearchOptions::default();
    let mut scratch = Scratch::new();
    let expected = cat.search(&query, &o, &mut Scratch::new()).unwrap();
    assert_ne!(expected.len(), 0);
    let mut seen = 0;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        cat.search_with(&query, &o, &mut scratch, |_| {
            seen += 1;
            assert!(seen != 1, "visitor gave up");
        })
    }));
    assert!(outcome.is_err());
    assert_eq!(cat.search(&query, &o, &mut scratch).unwrap(), expected);
}

/// A batch with a panicking iterator in the middle leaves the catalog as
/// it was: no ids, no postings, no segment. The batch may be large enough
/// to build a segment, too.
#[test]
fn a_panicking_add_many_leaves_the_catalog_alone() {
    let mut rng = Rng(11);
    let mut cat = Catalog::new();
    let mut live: Live = BTreeMap::new();
    for _ in 0..6 {
        add_reference(&mut cat, &mut live, &mut rng, 100);
    }
    let (refs, segments, postings) = (
        cat.reference_count(),
        cat.segment_count(),
        cat.posting_count(),
    );
    let good = cat.clone();
    let before = cat.segment(0).unwrap().clone();
    for panic_at in [0usize, 1, 3] {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let batch: Vec<Vec<(u32, u32)>> =
                (0..6).map(|_| reference_rows(&mut rng, 100)).collect();
            cat.add_many(batch.into_iter().enumerate().map(|(i, rows)| {
                assert_ne!(i, panic_at, "the iterator gave up");
                rows
            }));
        }));
        assert!(outcome.is_err(), "panicking at {panic_at}");
        assert_eq!(cat.reference_count(), refs);
        assert_eq!(cat.segment_count(), segments);
        assert_eq!(cat.posting_count(), postings);
        assert_eq!(cat.segment(0), Some(&before));
        assert_eq!(cat.live_count(), good.live_count());
    }
    // And it still takes a batch afterwards.
    let ids = add_batch(&mut cat, &mut live, &mut rng, 100);
    assert_eq!(ids.first().copied(), Some(refs as u32));
    check_against_model(
        &cat,
        &live,
        &mut Scratch::new(),
        "after a panic",
        &mut rng,
        100,
    );
}

#[test]
fn catalog_and_friends_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Catalog>();
    assert_send_sync::<Segment>();
    assert_send_sync::<Query>();
    assert_send_sync::<Scratch>();
    assert_send_sync::<Candidate>();
    assert_send_sync::<SearchOptions>();
    assert_send_sync::<SearchError>();
}

#[test]
fn empty_inputs_yield_no_candidates() {
    let mut scratch = Scratch::new();
    let o = SearchOptions::default();
    let empty = Catalog::new();
    let q = Query::from_pairs([(1, 2), (3, 4)], Some(64));
    assert_eq!(empty.search(&q, &o, &mut scratch).unwrap().len(), 0);
    let cat = Catalog::from_segment(Segment::from_postings(vec![Posting::new(1, 0, 0)]));
    let none = Query::from_pairs(Vec::<(u32, u32)>::new(), Some(64));
    assert_eq!(cat.search(&none, &o, &mut scratch).unwrap().len(), 0);
    let miss = Query::from_pairs([(2, 0)], Some(64));
    assert_eq!(cat.search(&miss, &o, &mut scratch).unwrap().len(), 0);
    let hit = cat.search(&q, &o, &mut scratch).unwrap();
    assert_eq!(hit.len(), 1);
    assert_eq!(
        (
            hit[0].reference,
            hit[0].delta,
            hit[0].votes,
            hit[0].total_votes
        ),
        (0, -2, 1, 1)
    );
}

#[test]
fn removed_references_never_surface_even_before_compaction() {
    let mut cat = Catalog::new();
    let rows: Vec<(u32, u32)> = (0..20).map(|i| (i, i * 3)).collect();
    let a = cat.add(rows.iter().copied());
    let b = cat.add(rows.iter().map(|&(h, t)| (h, t + 7)));
    let q = Query::from_pairs(rows.iter().copied(), Some(64));
    let o = SearchOptions::default();
    let mut s = Scratch::new();
    assert_eq!(cat.search(&q, &o, &mut s).unwrap().len(), 2);
    assert!(cat.remove(a));
    let got = cat.search(&q, &o, &mut s).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].reference, got[0].delta, got[0].votes), (b, 7, 20));
    assert!(cat.posting_count() >= 40, "postings stay until compaction");
    cat.compact();
    assert_eq!(cat.posting_count(), 20);
    assert_eq!(cat.search(&q, &o, &mut s).unwrap(), got);
}

#[test]
fn from_segment_keeps_the_callers_ids_and_continues_after_the_largest() {
    let seg = Segment::from_postings(vec![Posting::new(1, 0, 0), Posting::new(2, 5, 0)]);
    let mut cat = Catalog::from_segment(seg);
    assert_eq!(cat.reference_count(), 6);
    assert!(cat.is_live(3), "ids without postings are live");
    assert_eq!(cat.add([(9, 9)]), 6);
    let q = Query::from_pairs([(2, 0)], Some(64));
    let got = cat
        .search(&q, &SearchOptions::default(), &mut Scratch::new())
        .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].reference, 5);
}

#[test]
fn vote_bounds_are_enforced_after_the_totals_pass() {
    // One reference with 70 000 postings of one hash, and a query holding
    // that hash 65 000 times: with no fanout its total would be 4.55 billion.
    let rows: Vec<Posting> = (0..70_000u32).map(|i| Posting::new(5, 0, i)).collect();
    let mut rows_b = rows.clone();
    rows_b.extend((0..10u32).map(|i| Posting::new(5, 1, i)));
    let cat = Catalog::from_segment(Segment::from_postings(rows_b));
    let q = Query::from_pairs((0..65_000u32).map(|t| (5, t)), None);
    let mut s = Scratch::new();
    let mut unlimited = SearchOptions::default();
    unlimited.fanout = None;
    unlimited.max_votes = usize::MAX;
    let err = cat.search(&q, &unlimited, &mut s).unwrap_err();
    assert_eq!(err, SearchError::TotalsOverflow { reference: 0 });
    assert!(err.to_string().contains("reference 0"));
    // With the fanout the totals fit, but the scatter would hold
    // 2 × 64 × 65 000 votes; a tight `max_votes` refuses it.
    let mut capped = SearchOptions::default();
    capped.fanout = Some(64);
    capped.max_votes = 1_000;
    let err = cat.search(&q, &capped, &mut s).unwrap_err();
    assert_eq!(
        err,
        SearchError::TooManyVotes {
            votes: 64 * 65_000 + 10 * 65_000,
            max: 1_000
        }
    );
    // A generous limit with a small fanout runs it.
    let mut generous = capped;
    generous.fanout = Some(2);
    generous.max_votes = usize::MAX;
    let got = cat.search(&q, &generous, &mut s).unwrap();
    assert_eq!(got.len(), 2);
    assert!(got.iter().all(|c| c.total_votes == 2 * 65_000));
    // The scratch is still clean after both errors: a small search behaves
    // normally, and so does a fresh scratch.
    let small = Query::from_pairs([(5, 0)], Some(64));
    let got = cat
        .search(&small, &SearchOptions::default(), &mut s)
        .unwrap();
    assert_eq!(
        got,
        cat.search(&small, &SearchOptions::default(), &mut Scratch::new())
            .unwrap()
    );
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].total_votes, 64);
    assert_eq!(got[1].total_votes, 10);
}

#[test]
fn a_removed_reference_cannot_overflow_the_search() {
    // Reference 0 would overflow its 32-bit total (70 000 postings of one
    // hash against 65 000 query times); reference 1 is small. Alive, the
    // search reports the overflow. Removed, its postings stay in the
    // segment until compaction but must no longer fail the search.
    let mut rows: Vec<Posting> = (0..70_000u32).map(|i| Posting::new(5, 0, i)).collect();
    rows.extend((0..10u32).map(|i| Posting::new(5, 1, i)));
    let mut cat = Catalog::from_segment(Segment::from_postings(rows));
    let q = Query::from_pairs((0..65_000u32).map(|t| (5, t)), None);
    let mut o = SearchOptions::default();
    o.fanout = None;
    o.max_votes = usize::MAX;
    let mut s = Scratch::new();
    assert_eq!(
        cat.search(&q, &o, &mut s).unwrap_err(),
        SearchError::TotalsOverflow { reference: 0 }
    );
    assert!(cat.remove(0));
    assert!(
        cat.posting_count() > 70_000,
        "postings stay until compaction"
    );
    let got = cat.search(&q, &o, &mut s).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].reference, got[0].total_votes), (1, 10 * 65_000));
    cat.compact();
    assert_eq!(cat.posting_count(), 10);
    assert_eq!(cat.search(&q, &o, &mut s).unwrap(), got);
}

#[test]
fn a_query_with_no_kept_times_casts_no_votes() {
    // `max_times_per_hash = Some(0)` keeps the hashes but none of their
    // times; the search skips their runs instead of scanning them.
    let rows: Vec<(u32, u32)> = (0..20).map(|i| (i, i * 3)).collect();
    let mut cat = Catalog::new();
    cat.add(rows.iter().copied());
    let mut s = Scratch::new();
    let none = Query::from_pairs(rows.iter().copied(), Some(0));
    let got = cat
        .search(&none, &SearchOptions::default(), &mut s)
        .unwrap();
    assert_eq!(got.len(), 0);
    let all = Query::from_pairs(rows.iter().copied(), Some(64));
    let got = cat.search(&all, &SearchOptions::default(), &mut s).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].total_votes, 20);
}

/// Fanouts at and around every boundary the totals pass branches on (zero,
/// the 64-entry ring, the saturating 8-bit run multiplicity), over runs
/// where one reference has far more postings than any of them.
#[test]
fn fanout_boundaries_match_the_model() {
    let mut rng = Rng(0xFA_0001);
    let mut rows = Vec::new();
    // Per hash: one reference with a long burst, a few short ones.
    for h in 0..12u32 {
        let burst = [1, 2, 63, 64, 65, 200, 254, 255, 256, 257, 300, 700][h as usize];
        rows.extend((0..burst).map(|t| Posting::new(h, 3, t * 2)));
        for r in [0u32, 1, 2, 4, 9] {
            let n = 1 + rng.below(4) as u32;
            rows.extend((0..n).map(|t| Posting::new(h, r, t * 5 + r)));
        }
    }
    let cat = Catalog::from_segment(Segment::from_postings(rows.clone()));
    let pairs: Vec<(u32, u32)> = (0..12u32)
        .flat_map(|h| (0..=(h % 3)).map(move |t| (h, t)))
        .collect();
    let query = Query::from_pairs(pairs.iter().copied(), None);
    let grouped = group(&pairs, None);
    let clean = dedup(&rows);
    let mut scratch = Scratch::new();
    let fanouts = [
        0usize, 1, 2, 63, 64, 65, 100, 254, 255, 256, 257, 300, 699, 700, 701,
    ];
    for fanout in fanouts.into_iter().map(Some).chain([None]) {
        let o = options(fanout, 1, 2, 10, usize::MAX);
        let want = model(&clean, |_| true, &grouped, &o);
        let got = cat.search(&query, &o, &mut scratch).unwrap();
        assert_same(&got, &want, &format!("fanout {fanout:?}"));
    }
}
