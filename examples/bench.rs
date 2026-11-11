//! Latency and throughput on a synthetic catalog.
//!
//! ```bash
//! cargo run --release --example bench -- [refs] [postings_per_ref] [hashes] [queries] [threads]
//! ```
//!
//! Each reference gets `postings_per_ref` rows with hashes uniform over
//! `hashes` and times uniform over 45 minutes of 16 ms frames. A query is a
//! window of one reference's rows, shifted to start at 0, plus half as many
//! random rows. Latency is one thread searching each query once after a
//! warm-up pass; throughput is `threads` threads searching for five
//! seconds. Every query must find its reference at the planted offset.

use std::time::{Duration, Instant};

use afpindex::{Catalog, Query, Scratch, SearchOptions, vote_floor};

struct Rng(u64);

impl Rng {
    fn seeded(seed: u64) -> Self {
        let mut r = Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }

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

const MAX_TIME: u64 = 168_750;
const WINDOW: u32 = 30_000;

/// A reference's rows are a pure function of its id, so the bench never
/// keeps a second copy of the catalog.
fn rows_of(reference: u32, per_ref: usize, hashes: u64) -> Vec<(u32, u32)> {
    let mut rng = Rng::seeded(u64::from(reference) + 1);
    (0..per_ref)
        .map(|_| (rng.below(hashes) as u32, rng.below(MAX_TIME) as u32))
        .collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer"))
        .collect();
    let arg = |i: usize, default: usize| args.get(i).copied().unwrap_or(default);
    let refs = arg(0, 20_000);
    let per_ref = arg(1, 3_000);
    let hashes = arg(2, 3_600_000) as u64;
    let n_queries = arg(3, 200);
    let threads = arg(
        4,
        std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get),
    );

    let t = Instant::now();
    let mut builder = Catalog::builder();
    for r in 0..refs as u32 {
        builder.add(rows_of(r, per_ref, hashes));
    }
    let catalog = builder.build();
    let build = t.elapsed();
    let seg = catalog.segment(0).expect("one segment");
    println!(
        "catalog: {} references, {} postings, {} keys, {:.2} GiB, built in {:.1} s",
        refs,
        catalog.posting_count(),
        seg.key_count(),
        catalog.heap_bytes() as f64 / (1u64 << 30) as f64,
        build.as_secs_f64()
    );

    let queries = make_queries(n_queries, refs, per_ref, hashes);
    let rows: Vec<usize> = queries.iter().map(|q| q.0.input_rows()).collect();
    println!(
        "queries: {} of {}..{} rows (median {})",
        n_queries,
        rows.iter().min().unwrap(),
        rows.iter().max().unwrap(),
        {
            let mut r = rows.clone();
            r.sort_unstable();
            r[r.len() / 2]
        }
    );

    let mut scratch = Scratch::new();
    let mut found = 0usize;
    let mut candidates = 0usize;
    for (q, o, r, t0) in &queries {
        let got = catalog.search(q, o, &mut scratch).unwrap();
        candidates += got.len();
        if got
            .iter()
            .any(|c| c.reference == *r && c.delta == i64::from(*t0))
        {
            found += 1;
        }
    }
    println!(
        "warm-up: {found}/{n_queries} queries found their reference at the planted offset, {candidates} candidates total"
    );

    let mut lat = Vec::with_capacity(n_queries);
    let total = Instant::now();
    for (q, o, _, _) in &queries {
        let t = Instant::now();
        let got = catalog.search(q, o, &mut scratch).unwrap();
        std::hint::black_box(got);
        lat.push(t.elapsed());
    }
    let total = total.elapsed();
    lat.sort_unstable();
    println!(
        "latency (1 thread): total {:.1} ms, p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms",
        total.as_secs_f64() * 1e3,
        percentile(&lat, 0.5).as_secs_f64() * 1e3,
        percentile(&lat, 0.95).as_secs_f64() * 1e3,
        lat.last().unwrap().as_secs_f64() * 1e3
    );

    let run_for = Duration::from_secs(5);
    let count = throughput(&catalog, &queries, threads, run_for);
    println!(
        "throughput ({threads} threads, {} s): {:.0} searches/s",
        run_for.as_secs(),
        count as f64 / run_for.as_secs_f64()
    );
}

/// A planted query per entry: `(query, options, reference, offset)`.
type Planted = (Query, SearchOptions, u32, u32);

fn make_queries(n_queries: usize, refs: usize, per_ref: usize, hashes: u64) -> Vec<Planted> {
    let mut rng = Rng::seeded(0xBEEF);
    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let r = rng.below(refs as u64) as u32;
        let t0 = rng.below(MAX_TIME - u64::from(WINDOW)) as u32;
        let mut pairs: Vec<(u32, u32)> = rows_of(r, per_ref, hashes)
            .into_iter()
            .filter(|&(_, t)| t >= t0 && t < t0 + WINDOW)
            .map(|(h, t)| (h, t - t0))
            .collect();
        let noise = pairs.len() / 2;
        pairs.extend((0..noise).map(|_| {
            (
                rng.below(hashes) as u32,
                rng.below(u64::from(WINDOW)) as u32,
            )
        }));
        let query = Query::from_pairs(pairs, Some(64));
        let mut options = SearchOptions::default();
        options.min_total_votes = vote_floor(query.input_rows(), 16, 0.02);
        queries.push((query, options, r, t0));
    }
    queries
}

/// Searches completed by `threads` threads in `run_for`.
fn throughput(catalog: &Catalog, queries: &[Planted], threads: usize, run_for: Duration) -> usize {
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|k| {
                s.spawn(move || {
                    let mut scratch = Scratch::new();
                    let mut n = 0usize;
                    let start = Instant::now();
                    let mut i = k;
                    while start.elapsed() < run_for {
                        let (q, o, _, _) = &queries[i % queries.len()];
                        std::hint::black_box(catalog.search(q, o, &mut scratch).unwrap());
                        n += 1;
                        i += 1;
                    }
                    n
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .sum::<usize>()
    })
}
