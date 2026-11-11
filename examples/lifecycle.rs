//! Cost of everything around a search: building a catalog, adding to it,
//! compacting it, writing and loading its snapshot, and grouping a query.
//!
//! ```bash
//! cargo run --release --features mmap --example lifecycle -- [refs] [postings_per_ref] [hashes] [adds]
//! ```
//!
//! The catalog is the synthetic one of `examples/bench.rs`: each reference
//! gets `postings_per_ref` rows with hashes uniform over `hashes`. After
//! the bulk build, `adds` more references are added one at a time and a
//! tenth of them removed again, so the compactions have deltas to merge
//! and postings to drop. The snapshot is written to and read from memory,
//! so the timings are the crate's own work rather than the disk's; with
//! the `mmap` feature it is also written to a temporary file and mapped.
//! A last section searches with vote floors from strict to none, where
//! the histogram pass takes over from the totals pass.

use std::time::{Duration, Instant};

use afpindex::{Catalog, Query, Scratch, SearchOptions};

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

fn rows_of(reference: u32, per_ref: usize, hashes: u64) -> Vec<(u32, u32)> {
    let mut rng = Rng::seeded(u64::from(reference) + 1);
    (0..per_ref)
        .map(|_| (rng.below(hashes) as u32, rng.below(MAX_TIME) as u32))
        .collect()
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// Run `f` and return its result with the time it took.
fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed())
}

fn describe(what: &str, catalog: &Catalog) {
    println!(
        "{what}: {} postings in {} segments, {:.3} GiB on the heap",
        catalog.posting_count(),
        catalog.segment_count(),
        catalog.heap_bytes() as f64 / (1u64 << 30) as f64
    );
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("integer"))
        .collect();
    let arg = |i: usize, default: usize| args.get(i).copied().unwrap_or(default);
    let shape = Shape {
        refs: arg(0, 20_000),
        per_ref: arg(1, 3_000),
        hashes: arg(2, 3_600_000) as u64,
    };

    let mut catalog = build(&shape);
    grow_and_compact(&mut catalog, &shape, arg(3, 2_000));
    snapshot(&catalog);
    search(&catalog, &shape);
}

/// The synthetic catalog: how many references, rows each, distinct hashes.
struct Shape {
    refs: usize,
    per_ref: usize,
    hashes: u64,
}

impl Shape {
    fn rows_of(&self, reference: u32) -> Vec<(u32, u32)> {
        rows_of(reference, self.per_ref, self.hashes)
    }
}

/// Build every reference at once.
fn build(shape: &Shape) -> Catalog {
    let (catalog, t) = timed(|| {
        let mut builder = Catalog::builder();
        for r in 0..shape.refs as u32 {
            builder.add(shape.rows_of(r));
        }
        builder.build()
    });
    describe("built", &catalog);
    println!(
        "  bulk build: {:.2} s ({:.0} ns per posting)",
        secs(t),
        secs(t) * 1e9 / catalog.posting_count().max(1) as f64
    );
    catalog
}

/// Add `adds` references one at a time, remove a tenth of them, and
/// compact: the deltas alone on a copy, then everything.
fn grow_and_compact(catalog: &mut Catalog, shape: &Shape, adds: usize) {
    // The rows are generated outside the clock.
    let (mut total, mut worst) = (Duration::ZERO, Duration::ZERO);
    let mut added = Vec::with_capacity(adds);
    for i in 0..adds {
        let rows = shape.rows_of((shape.refs + i) as u32);
        let (id, t) = timed(|| catalog.add(rows));
        added.push(id);
        total += t;
        worst = worst.max(t);
    }
    describe("after the additions", catalog);
    println!(
        "  {adds} additions: {:.2} s, {:.2} ms each on average, {:.1} ms at worst",
        secs(total),
        secs(total) * 1e3 / adds.max(1) as f64,
        secs(worst) * 1e3
    );

    for &id in added.iter().step_by(10) {
        catalog.remove(id);
    }
    let mut deltas = catalog.clone();
    let ((), t) = timed(|| deltas.compact_deltas());
    describe("after compact_deltas", &deltas);
    println!("  compact_deltas: {:.2} s", secs(t));
    drop(deltas);
    let ((), t) = timed(|| catalog.compact());
    describe("after compact", catalog);
    println!(
        "  compact: {:.2} s ({:.0} ns per posting)",
        secs(t),
        secs(t) * 1e9 / catalog.posting_count().max(1) as f64
    );
}

/// Write the snapshot to memory, read it back, and map it from a file.
fn snapshot(catalog: &Catalog) {
    let (bytes, t) = timed(|| {
        let mut bytes = Vec::new();
        catalog.write_to(&mut bytes).expect("writing to memory");
        bytes
    });
    let gib = bytes.len() as f64 / (1u64 << 30) as f64;
    println!(
        "snapshot: {gib:.3} GiB\n  write_to: {:.2} s ({:.2} GiB/s)",
        secs(t),
        gib / secs(t)
    );
    let (back, t) = timed(|| Catalog::read_from(bytes.as_slice()).expect("a valid snapshot"));
    println!("  read_from: {:.2} s ({:.2} GiB/s)", secs(t), gib / secs(t));
    assert_eq!(back.posting_count(), catalog.posting_count());
    drop(back);
    map_snapshot(&bytes, catalog);
}

/// Group 200 queries, then search them with vote floors from strict to
/// none: fewer references are filtered by the totals pass and more of the
/// work moves to the histogram pass.
fn search(catalog: &Catalog, shape: &Shape) {
    let queries = make_pairs(200, shape);
    let rows: usize = queries.iter().map(Vec::len).sum();
    let (grouped, t) = timed(|| {
        let mut grouped = Vec::new();
        for _ in 0..20 {
            grouped.clear();
            grouped.extend(
                queries
                    .iter()
                    .map(|q| Query::from_pairs(q.iter().copied(), Some(64))),
            );
        }
        grouped
    });
    println!(
        "query grouping: {:.1} µs per query of {} rows on average",
        secs(t) * 1e6 / (20 * queries.len()) as f64,
        rows / queries.len()
    );

    let mut scratch = Scratch::new();
    for floor in [64, 16, 8, 4, 1] {
        let mut options = SearchOptions::default();
        options.min_total_votes = floor;
        let mut candidates = 0usize;
        for q in &grouped {
            candidates += catalog
                .search(q, &options, &mut scratch)
                .expect("a search within the vote limit")
                .len();
        }
        let mut best = Duration::MAX;
        for _ in 0..5 {
            let ((), t) = timed(|| {
                for q in &grouped {
                    std::hint::black_box(
                        catalog
                            .search(q, &options, &mut scratch)
                            .expect("a search within the vote limit"),
                    );
                }
            });
            best = best.min(t);
        }
        println!(
            "search, floor {floor}: {:.3} ms per query, {} candidates per query",
            secs(best) * 1e3 / grouped.len() as f64,
            candidates / grouped.len()
        );
    }
}

/// Query rows as `examples/bench.rs` plants them: a window of one
/// reference plus half as many random rows.
fn make_pairs(n: usize, shape: &Shape) -> Vec<Vec<(u32, u32)>> {
    let mut rng = Rng::seeded(0xBEEF);
    (0..n)
        .map(|_| {
            let r = rng.below(shape.refs as u64) as u32;
            let t0 = rng.below(MAX_TIME - u64::from(WINDOW)) as u32;
            let mut pairs: Vec<(u32, u32)> = shape
                .rows_of(r)
                .into_iter()
                .filter(|&(_, t)| t >= t0 && t < t0 + WINDOW)
                .map(|(h, t)| (h, t - t0))
                .collect();
            let noise = pairs.len() / 2;
            pairs.extend((0..noise).map(|_| {
                (
                    rng.below(shape.hashes) as u32,
                    rng.below(u64::from(WINDOW)) as u32,
                )
            }));
            pairs
        })
        .collect()
}

#[cfg(feature = "mmap")]
fn map_snapshot(bytes: &[u8], catalog: &Catalog) {
    let path = std::env::temp_dir().join(format!("afpindex-lifecycle-{}.afc", std::process::id()));
    std::fs::write(&path, bytes).expect("writing the snapshot file");
    let (mapped, t) = timed(|| Catalog::map_from(&path).expect("a valid snapshot"));
    println!(
        "  map_from: {:.2} s, {:.3} GiB on the heap",
        secs(t),
        mapped.heap_bytes() as f64 / (1u64 << 30) as f64
    );
    assert_eq!(mapped.posting_count(), catalog.posting_count());
    drop(mapped);
    std::fs::remove_file(&path).expect("removing the snapshot file");
}

#[cfg(not(feature = "mmap"))]
fn map_snapshot(_: &[u8], _: &Catalog) {}
