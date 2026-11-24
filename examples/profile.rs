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
//! `build` packs the references, or codes them as gaps when `GAPS` is set
//! in the environment.
//!
//! `load` mode profiles one load of the snapshot, chosen by its last
//! argument: `map`, `map-trusted`, `read` or `read-trusted`.
//!
//! `spill` mode weighs the two ways a catalog's segments become one: a
//! compaction held in memory, against one whose merge is written to a
//! segment file in a catalog directory. Both run on the same segments, are
//! timed, and the file they write is compared byte for byte with the
//! snapshot of the in-memory segment. A last argument of `memory` or `file`
//! runs one half alone, so that a peak resident set belongs to that half.
//!
//! `bulk` mode builds `refs × per_ref` postings into a catalog directory
//! through `CatalogStore::builder` and prints the time and the process's
//! peak resident set (from `/proc/self/status`, on Linux). An optional
//! last argument sets the chunk size in postings.

use std::time::Instant;

use afpindex::{Catalog, CatalogStore, Query, RefCoding, Scratch, SearchOptions, vote_floor};

mod support;
use support::Rng;

const MAX_TIME: u64 = 32_000;

/// The rows of synthetic reference `reference`, times below `MAX_TIME`.
fn rows_of(reference: u32, per_ref: usize, hashes: u64) -> Vec<(u32, u32)> {
    support::rows_in(reference, per_ref, hashes, MAX_TIME)
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

/// One compaction whose merge result is held in memory: the
/// `--toggle-collect` target of the mode's first half.
#[inline(never)]
fn measured_compact_memory(catalog: &mut Catalog) {
    catalog.compact();
}

/// One compaction whose merge is written to a segment file: the mode's
/// second target.
#[inline(never)]
fn measured_compact_file(store: &mut CatalogStore) {
    store.compact().unwrap();
}

/// The `spill` mode: a compaction of the catalog's segments both ways.
///
/// Both halves compact the same catalog by the two routes a store has. The
/// in-memory half's one segment is written out, and that snapshot is
/// compared byte for byte with the segment file the store's compaction
/// wrote.
fn run_spill(a: &[String]) {
    let (refs, per_ref, hashes) = (
        a[1].parse::<u32>().unwrap(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<u64>().unwrap(),
    );
    let path = &a[4];
    let adds = a.get(5).map_or(0, |n| n.parse::<usize>().unwrap());
    // Which half to run: both compare what they wrote, one runs a half
    // alone, so that a peak resident set belongs to that half.
    let half = a.get(6).map_or("both", String::as_str);
    let mut catalog = Catalog::map_from_trusted(path).unwrap();
    for i in 0..adds {
        catalog.add(rows_of(refs + i as u32, per_ref, hashes));
    }
    let segments = catalog.segment_count();
    let postings = catalog.posting_count();
    let dir = format!("{path}.store");
    let _ = std::fs::remove_dir_all(&dir);
    let (memory_file, file_file) = (format!("{path}.memory"), format!("{path}.file"));
    let mut in_memory = catalog.clone();
    if half != "file" {
        let t = Instant::now();
        measured_compact_memory(&mut in_memory);
        println!(
            "{segments} segments, {postings} postings: compact in memory {:.3} s",
            t.elapsed().as_secs_f64(),
        );
        let mut out = std::io::BufWriter::new(std::fs::File::create(&memory_file).unwrap());
        in_memory
            .segment(0)
            .expect("one segment")
            .write_to(&mut out)
            .unwrap();
    }
    if half != "memory" {
        // Every merge goes to a file: the merge in memory is not what this
        // half measures.
        let mut store = CatalogStore::create_from(&dir, catalog).unwrap();
        store.set_spill_postings(0);
        let t = Instant::now();
        measured_compact_file(&mut store);
        println!(
            "{segments} segments, {postings} postings: compact to a segment file {:.3} s",
            t.elapsed().as_secs_f64(),
        );
        let file = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "afs"))
            .expect("the spill wrote a segment file");
        std::fs::copy(&file, &file_file).unwrap();
    }
    if half == "both" {
        let want = std::fs::read(&memory_file).unwrap();
        let got = std::fs::read(&file_file).unwrap();
        let same = got == want;
        println!("the two files are identical: {same} ({} bytes)", got.len());
        assert!(same, "the two files differ");
    }
    for file in [&memory_file, &file_file] {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file).unwrap();
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
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

/// The `bulk` mode: a build through the store builder, timed, with the
/// peak resident set it reached.
fn run_bulk(a: &[String]) {
    let (refs, per_ref, hashes) = (
        a[1].parse::<u32>().unwrap(),
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<u64>().unwrap(),
    );
    let dir = &a[4];
    let _ = std::fs::remove_dir_all(dir);
    // The peak resident set counts the pages of mapped files, which the
    // kernel can drop and read again; the heap is the anonymous part,
    // sampled here since the kernel keeps no peak of it.
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let done = std::sync::Arc::clone(&done);
        std::thread::spawn(move || {
            let mut peak = 0u64;
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                peak = peak.max(status_kb("RssAnon").unwrap_or(0));
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            peak
        })
    };
    let t = Instant::now();
    let mut builder = CatalogStore::builder(dir).unwrap();
    if let Some(chunk) = a.get(5) {
        builder.chunk_postings(chunk.parse().unwrap());
    }
    for r in 0..refs {
        builder.add(rows_of(r, per_ref, hashes)).unwrap();
    }
    let store = builder.finish().unwrap();
    println!(
        "{} postings in {} segment: {:.3} s",
        store.catalog().posting_count(),
        store.catalog().segment_count(),
        t.elapsed().as_secs_f64()
    );
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    let anon = sampler.join().unwrap();
    if let Some(hwm) = status_kb("VmHWM") {
        println!("peak resident set {hwm} kB, of it anonymous at most {anon} kB");
    }
}

/// A `kB` field of `/proc/self/status`, where there is one.
fn status_kb(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with(field))?;
    line[field.len() + 1..]
        .trim()
        .trim_end_matches(" kB")
        .parse()
        .ok()
}

/// The `build` mode: a synthetic catalog, written to `path`.
fn build(refs: usize, per_ref: usize, hashes: u64, path: &str) {
    let mut b = Catalog::builder();
    if std::env::var("GAPS").is_ok() {
        b.ref_coding(RefCoding::Gaps);
    }
    for r in 0..refs as u32 {
        b.add(rows_of(r, per_ref, hashes));
    }
    let c = b.build();
    let file = std::fs::File::create(path).unwrap();
    let mut out = std::io::BufWriter::with_capacity(1 << 23, file);
    c.write_to(&mut out).unwrap();
    println!("built {} postings", c.posting_count());
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| a[i].parse::<usize>().expect("integer");
    let (refs, per_ref, hashes) = (num(1), num(2), num(3) as u64);
    let path = &a[4];
    match a[0].as_str() {
        // build refs per_ref hashes path
        "build" => build(refs, per_ref, hashes, path),
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
        // bulk refs per_ref hashes dir [chunk]
        "bulk" => run_bulk(&a),
        _ => panic!("mode: build, search, adds, batch, load, spill or bulk"),
    }
}
