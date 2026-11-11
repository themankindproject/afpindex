//! Replay real clips against a catalog dumped as raw arrays, and write
//! each clip's candidates and timings for comparison with another engine.
//!
//! `snapshot_dir` holds little-endian arrays: `keys.u32` (sorted distinct
//! hashes), `offsets.i64` (one more entry than keys: each hash's first
//! posting), `ref.u32` and `t.u32` (one entry per posting), `ref_hi.u64`
//! and `ref_lo.u64` (the UUID halves of each reference id). `fp_dir`
//! holds one `hash,time` CSV per clip. Options are the production ones:
//! fanout 64, the first 64 query times per hash, and the floor
//! `max(16, ceil(0.02 rows))`.
//!
//! If `snapshot_dir/catalog.afc` exists it is loaded with
//! `Catalog::read_from` instead of being built from the arrays, which is
//! the production load path; `--save <file>` writes that file after a
//! build from the arrays.
//!
//! usage: `replay <snapshot_dir> <fp_dir> <out_dir> [--save <file>] [--mmap]`
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use afpindex::{Candidate, Catalog, Posting, Query, Scratch, SearchOptions, Segment, vote_floor};

type Clip = (String, usize, Query, SearchOptions);

fn read_u32s(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u64s(p: &Path) -> Vec<u64> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn uuid(hi: u64, lo: u64) -> String {
    let h = format!("{hi:016x}{lo:016x}");
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// Every posting of the dump, streamed from the two per-posting arrays.
fn load_rows(snap: &Path) -> Vec<Posting> {
    let keys = read_u32s(&snap.join("keys.u32"));
    let offsets = read_u64s(&snap.join("offsets.i64"));
    let n = offsets[keys.len()] as usize;
    let t = Instant::now();
    let mut rows: Vec<Posting> = Vec::with_capacity(n);
    let mut rf = BufReader::with_capacity(1 << 23, File::open(snap.join("ref.u32")).unwrap());
    let mut tf = BufReader::with_capacity(1 << 23, File::open(snap.join("t.u32")).unwrap());
    let chunk = 1usize << 22;
    let (mut rb, mut tb) = (vec![0u8; chunk * 4], vec![0u8; chunk * 4]);
    let (mut k, mut i) = (0usize, 0usize);
    while i < n {
        let m = chunk.min(n - i);
        rf.read_exact(&mut rb[..m * 4]).unwrap();
        tf.read_exact(&mut tb[..m * 4]).unwrap();
        for (rc, tc) in rb[..m * 4].chunks_exact(4).zip(tb[..m * 4].chunks_exact(4)) {
            while offsets[k + 1] as usize <= i {
                k += 1;
            }
            let r = u32::from_le_bytes(rc.try_into().unwrap());
            let tm = u32::from_le_bytes(tc.try_into().unwrap());
            rows.push(Posting::new(keys[k], r, tm));
            i += 1;
        }
    }
    eprintln!("read {n} rows in {:.1} s", t.elapsed().as_secs_f64());
    rows
}

/// One query per clip CSV, with the production options.
fn load_clips(fp: &Path) -> Vec<Clip> {
    let mut paths: Vec<_> = std::fs::read_dir(fp)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "csv"))
        .collect();
    paths.sort();
    let mut clips = Vec::new();
    for p in &paths {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let pairs: Vec<(u32, u32)> = BufReader::new(File::open(p).unwrap())
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (h, t) = l.split_once(',').unwrap();
                (h.trim().parse().unwrap(), t.trim().parse().unwrap())
            })
            .collect();
        let q = Query::from_pairs(pairs.iter().copied(), Some(64));
        let mut o = SearchOptions::default();
        o.min_total_votes = vote_floor(pairs.len(), 16, 0.02);
        clips.push((name, pairs.len(), q, o));
    }
    clips
}

fn candidate_json(c: &Candidate, his: &[u64], los: &[u64]) -> String {
    format!(
        "[\"{}\",{},{},{},{},{},{:e},{:e},{:e}]",
        uuid(his[c.reference as usize], los[c.reference as usize]),
        c.delta,
        c.votes,
        c.first_time,
        c.last_time,
        c.noise.bins,
        c.noise.mean,
        c.noise.std_dev,
        f64::from(c.noise.max)
    )
}

/// Search every clip three times on one thread; write candidates and the
/// first and best timings.
fn search_clips(cat: &Catalog, clips: &[Clip], snap: &Path, out: &Path) {
    let his = read_u64s(&snap.join("ref_hi.u64"));
    let los = read_u64s(&snap.join("ref_lo.u64"));
    let mut jsonl = File::create(out.join("parity.jsonl")).unwrap();
    let mut csv = File::create(out.join("latency.csv")).unwrap();
    writeln!(csv, "name,rows,hashes,candidates,first_ms,best_ms").unwrap();
    let mut scratch = Scratch::new();
    for (name, rows, q, o) in clips {
        let (mut best, mut first, mut cands) = (f64::MAX, 0.0, Vec::new());
        for rep in 0..3 {
            let t = Instant::now();
            cands = cat.search(q, o, &mut scratch).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if rep == 0 {
                first = ms;
            }
            best = best.min(ms);
        }
        let items: Vec<String> = cands
            .iter()
            .map(|c| candidate_json(c, &his, &los))
            .collect();
        writeln!(
            jsonl,
            "{{\"name\":\"{name}\",\"rows\":{rows},\"candidates\":[{}]}}",
            items.join(",")
        )
        .unwrap();
        writeln!(
            csv,
            "{name},{rows},{},{},{first:.3},{best:.3}",
            q.hash_count(),
            cands.len()
        )
        .unwrap();
    }
    eprintln!("searched {} clips", clips.len());
}

/// Searches per second with one thread per hardware thread, each
/// cycling through the clips.
fn throughput(cat: &Catalog, clips: &[Clip], threads: usize) -> usize {
    let total = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for t in 0..threads {
            let total = &total;
            s.spawn(move || {
                let mut scratch = Scratch::new();
                let start = Instant::now();
                let (mut i, mut done) = (t, 0usize);
                while start.elapsed().as_secs_f64() < 5.0 {
                    let (_, _, q, o) = &clips[i % clips.len()];
                    std::hint::black_box(cat.search(q, o, &mut scratch).unwrap());
                    i += 1;
                    done += 1;
                }
                total.fetch_add(done, Ordering::Relaxed);
            });
        }
    });
    total.load(Ordering::Relaxed) / 5
}

#[cfg(feature = "mmap")]
fn map_catalog(path: &Path) -> Catalog {
    Catalog::map_from(path).unwrap()
}

#[cfg(not(feature = "mmap"))]
fn map_catalog(_: &Path) -> Catalog {
    panic!("--mmap needs a build with `--features mmap`")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (snap, fp, out) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    let save = (args.get(4).map(String::as_str) == Some("--save")).then(|| Path::new(&args[5]));
    let mmap = args.iter().any(|a| a == "--mmap");
    let afc = snap.join("catalog.afc");
    let cat = if afc.exists() && save.is_none() {
        let t = Instant::now();
        let cat = if mmap {
            map_catalog(&afc)
        } else {
            let file = BufReader::with_capacity(1 << 23, File::open(&afc).unwrap());
            Catalog::read_from(file).unwrap()
        };
        eprintln!(
            "catalog: {} {} in {:.1} s: {} postings, {} references, heap {:.2} GiB",
            if mmap { "mapped" } else { "loaded" },
            afc.display(),
            t.elapsed().as_secs_f64(),
            cat.posting_count(),
            cat.reference_count(),
            cat.heap_bytes() as f64 / f64::from(1u32 << 30)
        );
        cat
    } else {
        let rows = load_rows(snap);
        let t = Instant::now();
        let seg = Segment::from_postings(rows);
        eprintln!(
            "segment: {} postings, {} keys, widths {:?}, heap {:.2} GiB, built in {:.1} s",
            seg.len(),
            seg.key_count(),
            seg.widths(),
            seg.heap_bytes() as f64 / f64::from(1u32 << 30),
            t.elapsed().as_secs_f64()
        );
        let cat = Catalog::from_segment(seg);
        eprintln!("catalog: {} references", cat.reference_count());
        if let Some(path) = save {
            let t = Instant::now();
            cat.write_to(BufWriter::with_capacity(
                1 << 23,
                File::create(path).unwrap(),
            ))
            .unwrap();
            eprintln!(
                "wrote {} in {:.1} s",
                path.display(),
                t.elapsed().as_secs_f64()
            );
        }
        cat
    };
    let clips = load_clips(fp);
    search_clips(&cat, &clips, snap, out);
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    eprintln!(
        "throughput, {threads} threads, 5 s: {} searches/s",
        throughput(&cat, &clips, threads)
    );
}
