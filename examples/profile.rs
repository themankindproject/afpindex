//! Instruction counts for a search or an addition, for `callgrind`.
//!
//! Builds a synthetic catalog into a snapshot, or maps one and searches or
//! adds to it inside a function that `--toggle-collect` selects, so the
//! profile holds the measured work and not the load or the result printing.
//!
//! ```bash
//! export CARGO_PROFILE_RELEASE_DEBUG=1
//! cargo build --release --features mmap,simd --example profile
//! B=target/release/examples/profile
//! $B build 120000 500 450000 /tmp/s1.afc    # real-like: 133 postings per hash
//! $B build 20000 3000 3600000 /tmp/s2.afc   # the bench shape: 17 per hash
//! valgrind --tool=callgrind --toggle-collect='profile::measured' \
//!   --callgrind-out-file=/tmp/cg.out \
//!   $B search 120000 500 450000 /tmp/s1.afc 750 0 0 2
//! callgrind_annotate --auto=yes /tmp/cg.out
//! ```
//!
//! Arguments after the path are query rows, floor (0 for a proportional
//! one), additions before the first search, and rounds of 40 searches.
//! `adds` mode profiles `n` single additions; `batch` mode profiles the
//! same `n` references added as one `add_many`. Setting `COMPACT_DELTAS`
//! or `COMPACT` in the environment compacts after the additions; without
//! them the additions are what leave many segments behind, and a search
//! then pays for a lookup in every one.
//!
//! `load` mode profiles one load of the snapshot, chosen by its last
//! argument: `map`, `map-trusted`, `read` or `read-trusted`.
//!
//! `spill` mode weighs the two ways of writing a merge out: `merge_all` on
//! the catalog's segments, its result written with `write_to`, against
//! `merge_to_file` over the same segments, which holds only its pieces.
//! Both are timed, the files they write are compared byte for byte, and the
//! mode fails if they differ. A last argument of `memory` or `file` runs
//! one half alone, so that a peak resident set belongs to that half.

use std::time::Instant;

use afpindex::{
    Catalog, Query, Scratch, SearchOptions, Segment, Written, merge_in_memory, merge_to_file,
    vote_floor,
};

/// A reference's filter, or none: these merges keep everything, the way a
/// compaction with nothing removed does.
type Keep = fn(u32) -> bool;

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

const MAX_TIME: u64 = 32_000;

fn rows_of(reference: u32, per_ref: usize, hashes: u64) -> Vec<(u32, u32)> {
    let mut rng = Rng::seeded(u64::from(reference) + 1);
    (0..per_ref)
        .map(|_| (rng.below(hashes) as u32, rng.below(MAX_TIME) as u32))
        .collect()
}

/// The searches `--toggle-collect` counts: the rounds, not the load.
#[inline(never)]
fn measured(
    catalog: &Catalog,
    queries: &[(Query, SearchOptions)],
    scratch: &mut Scratch,
    rounds: usize,
) -> usize {
    let mut n = 0;
    for _ in 0..rounds {
        for (q, o) in queries {
            n += catalog.search(q, o, scratch).unwrap().len();
        }
    }
    n
}

/// One addition per reference, the way a live catalog grows.
#[inline(never)]
fn measured_adds(catalog: &mut Catalog, rows: Vec<Vec<(u32, u32)>>) {
    for r in rows {
        catalog.add(r);
    }
}

/// The same references as one [`Catalog::add_many`] batch.
#[inline(never)]
fn measured_batch(catalog: &mut Catalog, rows: Vec<Vec<(u32, u32)>>) {
    catalog.add_many(rows);
}

/// How a `load` run reads the snapshot.
#[derive(Clone, Copy)]
enum Load {
    Map,
    MapTrusted,
    Read,
    ReadTrusted,
}

/// One load of the snapshot, with every check or trusting the file: the
/// work `load` counts.
#[inline(never)]
fn measured_load(path: &str, bytes: &[u8], load: Load) -> Catalog {
    match load {
        Load::Map => Catalog::map_from(path),
        Load::MapTrusted => Catalog::map_from_trusted(path),
        Load::Read => Catalog::read_from(bytes),
        Load::ReadTrusted => Catalog::read_from_trusted(bytes),
    }
    .unwrap()
}

/// One merge held in memory and written out, the way a catalog would: the
/// `--toggle-collect` target of the mode's first half.
#[inline(never)]
fn measured_merge_memory(sides: &[(&Segment, Option<&Keep>)], path: &str) -> u64 {
    let merged = merge_in_memory(sides);
    let file = std::fs::File::create(path).unwrap();
    let mut out = std::io::BufWriter::with_capacity(1 << 23, file);
    merged.write_to(&mut out).unwrap();
    out.into_inner().unwrap().sync_all().unwrap();
    std::fs::metadata(path).unwrap().len()
}

/// One merge written straight to the file: the mode's second target.
#[inline(never)]
fn measured_merge_file(sides: &[(&Segment, Option<&Keep>)], path: &str) -> (Written, u64) {
    let written = merge_to_file(sides, std::path::Path::new(path)).unwrap();
    let bytes = std::fs::metadata(path).unwrap().len();
    (written, bytes)
}

/// The `spill` mode: a merge of the catalog's segments both ways.
fn run_spill(a: &[String]) {
    let (refs, per_ref, hashes) = (
        a[1].parse::<u32>().unwrap(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<u64>().unwrap(),
    );
    let path = &a[4];
    let adds = a.get(5).map_or(0, |n| n.parse::<usize>().unwrap());
    // Which half to run: both write a file and compare them, one runs a
    // half alone, so that a peak resident set belongs to that half.
    let half = a.get(6).map_or("both", String::as_str);
    let mut catalog = Catalog::map_from_trusted(path).unwrap();
    for i in 0..adds {
        catalog.add(rows_of(refs + i as u32, per_ref, hashes));
    }
    // The segments as they stand, merged the way `compact` merges them.
    let segments: Vec<Segment> = catalog.segments().cloned().collect();
    let postings = catalog.posting_count();
    let sides: Vec<(&Segment, Option<&Keep>)> = segments.iter().map(|s| (s, None)).collect();
    let (memory_file, file_file) = (format!("{path}.memory"), format!("{path}.file"));
    if half != "file" {
        let t = Instant::now();
        let bytes = measured_merge_memory(&sides, &memory_file);
        println!(
            "{} segments, {} postings: merge_all + write_to {:.3} s, {} bytes",
            sides.len(),
            postings,
            t.elapsed().as_secs_f64(),
            bytes
        );
    }
    if half != "memory" {
        let t = Instant::now();
        let (written, bytes) = measured_merge_file(&sides, &file_file);
        assert_eq!(written.postings as usize, postings);
        println!(
            "{} segments, {} postings: merge_to_file {:.3} s, {} bytes",
            sides.len(),
            postings,
            t.elapsed().as_secs_f64(),
            bytes
        );
    }
    if half == "both" {
        let same = std::fs::read(&memory_file).unwrap() == std::fs::read(&file_file).unwrap();
        println!("the two files are identical: {same}");
        assert!(same, "the two files differ");
    }
    for file in [&memory_file, &file_file] {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file).unwrap();
        }
    }
}

/// The `load` mode: time one load, with the mode's name as the argument.
fn run_load(a: &[String]) {
    let (path, which) = (&a[4], a[5].as_str());
    let load = match which {
        "map" => Load::Map,
        "map-trusted" => Load::MapTrusted,
        "read" => Load::Read,
        "read-trusted" => Load::ReadTrusted,
        other => panic!("load mode {other}"),
    };
    // The read modes need the bytes; mapping only the path, but reading
    // them keeps every mode's work outside the load itself.
    let bytes = std::fs::read(path).unwrap();
    let t = Instant::now();
    let catalog = measured_load(path, &bytes, load);
    let secs = t.elapsed().as_secs_f64();
    println!(
        "{} segments, {} postings: {secs:.3} s",
        catalog.segment_count(),
        catalog.posting_count(),
    );
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| a[i].parse::<usize>().expect("integer");
    let (refs, per_ref, hashes) = (num(1), num(2), num(3) as u64);
    let path = &a[4];
    match a[0].as_str() {
        // build refs per_ref hashes path
        "build" => {
            let mut b = Catalog::builder();
            for r in 0..refs as u32 {
                b.add(rows_of(r, per_ref, hashes));
            }
            let c = b.build();
            let file = std::fs::File::create(path).unwrap();
            let mut out = std::io::BufWriter::with_capacity(1 << 23, file);
            c.write_to(&mut out).unwrap();
            println!("built {} postings", c.posting_count());
        }
        // search refs per_ref hashes path query_rows floor adds rounds
        "search" => {
            let (qrows, floor, adds, rounds) = (num(5), num(6), num(7), num(8));
            let mut catalog = Catalog::map_from_trusted(path).unwrap();
            for i in 0..adds {
                catalog.add(rows_of((refs + i) as u32, per_ref, hashes));
            }
            if std::env::var("COMPACT_DELTAS").is_ok() {
                catalog.compact_deltas();
            }
            if std::env::var("COMPACT").is_ok() {
                catalog.compact();
            }
            let sizes: Vec<_> = catalog
                .segments()
                .map(|s| (s.len(), s.key_count()))
                .collect();
            println!("segment sizes: {sizes:?}");
            let mut rng = Rng::seeded(0xBEEF);
            let queries: Vec<(Query, SearchOptions)> = (0..40)
                .map(|_| {
                    let r = rng.below(refs as u64) as u32;
                    let mut pairs: Vec<(u32, u32)> = rows_of(r, per_ref, hashes)
                        .into_iter()
                        .take(qrows * 2 / 3)
                        .collect();
                    let noise = qrows - pairs.len().min(qrows);
                    pairs.extend(
                        (0..noise).map(|_| (rng.below(hashes) as u32, rng.below(MAX_TIME) as u32)),
                    );
                    let q = Query::from_pairs(pairs, Some(64));
                    let mut o = SearchOptions::default();
                    o.min_total_votes = if floor == 0 {
                        vote_floor(q.input_rows(), 16, 0.02)
                    } else {
                        floor as u32
                    };
                    (q, o)
                })
                .collect();
            let mut scratch = Scratch::new();
            let warm = measured(&catalog, &queries, &mut scratch, 1);
            let t = Instant::now();
            let n = measured(&catalog, &queries, &mut scratch, rounds);
            let per = t.elapsed().as_secs_f64() * 1e6 / (rounds * queries.len()) as f64;
            println!(
                "segments {} postings {} | {warm} candidates/round | {per:.1} us/search",
                catalog.segment_count(),
                catalog.posting_count(),
            );
            std::hint::black_box(n);
        }
        // adds refs per_ref hashes path n
        "adds" => {
            let n = num(5);
            let mut catalog = Catalog::map_from_trusted(path).unwrap();
            let rows: Vec<_> = (0..n)
                .map(|i| rows_of((refs + i) as u32, per_ref, hashes))
                .collect();
            let t = Instant::now();
            measured_adds(&mut catalog, rows);
            println!(
                "{n} adds: {:.2} ms each, {} segments",
                t.elapsed().as_secs_f64() * 1e3 / n as f64,
                catalog.segment_count()
            );
        }
        // batch refs per_ref hashes path n
        "batch" => {
            let n = num(5);
            let mut catalog = Catalog::map_from_trusted(path).unwrap();
            let rows: Vec<_> = (0..n)
                .map(|i| rows_of((refs + i) as u32, per_ref, hashes))
                .collect();
            let t = Instant::now();
            measured_batch(&mut catalog, rows);
            println!(
                "{n} in one batch: {:.2} ms each, {} segments",
                t.elapsed().as_secs_f64() * 1e3 / n as f64,
                catalog.segment_count()
            );
        }
        // load refs per_ref hashes path map|map-trusted|read|read-trusted
        "load" => run_load(&a),
        // spill refs per_ref hashes path [adds]
        "spill" => run_spill(&a),
        _ => panic!("mode: build, search, adds, batch, load or spill"),
    }
}
