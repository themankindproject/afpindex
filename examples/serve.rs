//! Searches while the catalog changes: searches on several threads while
//! one writer adds and removes references and compacts as a
//! [`CompactionPolicy`] calls for, publishing each change for the next
//! search.
//!
//! ```bash
//! cargo run --release --example serve -- [refs] [postings_per_ref] [hashes] [readers] [seconds] [batch] [policy...]
//! ```
//!
//! The catalog is the synthetic one of `examples/bench.rs`. Each policy
//! runs for `seconds` from the same catalog, and once with no writes
//! first. A policy is `delta_share:deltas` (`0.05:4` is the default
//! policy) or `none`, which never compacts. The writer alternates an
//! addition of `batch` references (one `add_many`) and a removal of a
//! random live reference, works on a copy of the catalog (which shares its
//! segments), runs the compaction the policy calls for on that copy, and
//! swaps it in: a search never waits for a change, only for the swap.
//! Queries are windows of a reference's rows plus half as many random
//! rows, as in `bench`. `--gaps` codes the references as gaps.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use afpindex::{
    Catalog, Compaction, CompactionPolicy, Query, RefCoding, Scratch, SearchOptions, vote_floor,
};

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

/// The query of a window of `reference`'s rows, shifted to start at 0,
/// with half as many random rows.
fn query(rng: &mut Rng, reference: u32, per_ref: usize, hashes: u64) -> (Query, SearchOptions) {
    let start = rng.below(MAX_TIME - u64::from(WINDOW)) as u32;
    let mut rows: Vec<(u32, u32)> = rows_of(reference, per_ref, hashes)
        .into_iter()
        .filter(|&(_, t)| t >= start && t < start + WINDOW)
        .map(|(h, t)| (h, t - start))
        .collect();
    let noise = rows.len() / 2;
    rows.extend((0..noise).map(|_| {
        (
            rng.below(hashes) as u32,
            rng.below(u64::from(WINDOW)) as u32,
        )
    }));
    let mut options = SearchOptions::default();
    options.min_total_votes = vote_floor(rows.len(), 16, 0.02);
    (Query::from_pairs(rows, Some(64)), options)
}

/// The value at `p` of `values`, which it sorts, in milliseconds.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

/// What one phase of writes does.
struct Writes {
    policy: Option<CompactionPolicy>,
    batch: usize,
    per_ref: usize,
    hashes: u64,
}

/// One phase: `readers` threads search `served` for `time` while the
/// writer, if there is one, changes it. Prints what each side did.
fn phase(
    name: &str,
    served: &RwLock<Arc<Catalog>>,
    queries: &[(Query, SearchOptions)],
    readers: usize,
    time: Duration,
    plan: Option<&Writes>,
) {
    let stop = AtomicBool::new(false);
    let searches = AtomicUsize::new(0);
    let start = Instant::now();
    let (mut latencies, written) = std::thread::scope(|s| {
        let threads: Vec<_> = (0..readers)
            .map(|t| {
                let (stop, searches) = (&stop, &searches);
                s.spawn(move || {
                    let mut scratch = Scratch::new();
                    let mut latencies = Vec::new();
                    let mut i = t * 7;
                    while !stop.load(Ordering::Relaxed) {
                        let catalog = Arc::clone(&served.read().expect("lock"));
                        let (q, o) = &queries[i % queries.len()];
                        let t0 = Instant::now();
                        std::hint::black_box(catalog.search(q, o, &mut scratch).expect("search"));
                        latencies.push(t0.elapsed().as_secs_f64() * 1e3);
                        searches.fetch_add(1, Ordering::Relaxed);
                        i += 1;
                    }
                    latencies
                })
            })
            .collect();
        let written = plan.map(|w| write_for(served, w, time));
        if plan.is_none() {
            std::thread::sleep(time);
        }
        stop.store(true, Ordering::Relaxed);
        let latencies: Vec<f64> = threads
            .into_iter()
            .flat_map(|t| t.join().expect("a search thread"))
            .collect();
        (latencies, written)
    });
    let secs = start.elapsed().as_secs_f64();
    let catalog = Arc::clone(&served.read().expect("lock"));
    println!(
        "{name:>10} | {:>6.0} searches/s | p50 {:.3} ms, p99 {:.3} ms | {} segments, {} references at the end",
        searches.load(Ordering::Relaxed) as f64 / secs,
        percentile(&mut latencies, 0.5),
        percentile(&mut latencies, 0.99),
        catalog.segment_count(),
        catalog.live_count(),
    );
    if let Some(mut w) = written {
        println!(
            "{:>10} | {:>6.0} references added/s, change p50 {:.2} ms, p99 {:.2} ms | {:.0} removed/s | {} delta and {} full compactions, longest {:.0} ms",
            "",
            w.added as f64 / secs,
            percentile(&mut w.changes, 0.5),
            percentile(&mut w.changes, 0.99),
            w.removed as f64 / secs,
            w.deltas,
            w.fulls,
            w.compactions.iter().copied().fold(0.0, f64::max),
        );
    }
}

/// What the writer did in a phase.
struct Written {
    added: usize,
    removed: usize,
    /// Milliseconds per published change, compaction included.
    changes: Vec<f64>,
    compactions: Vec<f64>,
    deltas: usize,
    fulls: usize,
}

/// Change `served` for `time`: alternately add a batch and remove a
/// reference, on a copy, compacting it as the policy calls for.
fn write_for(served: &RwLock<Arc<Catalog>>, w: &Writes, time: Duration) -> Written {
    let mut rng = Rng::seeded(7);
    let mut out = Written {
        added: 0,
        removed: 0,
        changes: Vec::new(),
        compactions: Vec::new(),
        deltas: 0,
        fulls: 0,
    };
    let start = Instant::now();
    let mut add = true;
    while start.elapsed() < time {
        let t0 = Instant::now();
        let mut next = Catalog::clone(&served.read().expect("lock"));
        if add {
            let first = next.reference_count() as u32;
            let batch: Vec<_> = (first..first + w.batch as u32)
                .map(|r| rows_of(r, w.per_ref, w.hashes))
                .collect();
            out.added += next.add_many(batch).len();
        } else {
            let id = (0..64)
                .map(|_| rng.below(next.reference_count() as u64) as u32)
                .find(|&id| next.is_live(id));
            if id.is_some_and(|id| next.remove(id)) {
                out.removed += 1;
            }
        }
        add = !add;
        if let Some(due) = w.policy.and_then(|p| next.compaction_due(&p)) {
            let c0 = Instant::now();
            match due {
                Compaction::Deltas => {
                    next.compact_deltas();
                    out.deltas += 1;
                }
                Compaction::Full => {
                    next.compact();
                    out.fulls += 1;
                }
            }
            out.compactions.push(c0.elapsed().as_secs_f64() * 1e3);
        }
        *served.write().expect("lock") = Arc::new(next);
        out.changes.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    out
}

/// A policy argument: `delta_share:deltas`, or `none`.
fn policy_of(arg: &str) -> Option<CompactionPolicy> {
    if arg == "none" {
        return None;
    }
    let (share, deltas) = arg.split_once(':').expect("delta_share:deltas or none");
    let mut policy = CompactionPolicy::default();
    policy.delta_share = share.parse().expect("a share");
    policy.deltas = deltas.parse().expect("a count");
    Some(policy)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).filter(|a| a != "--gaps").collect();
    let gaps = std::env::args().any(|a| a == "--gaps");
    let num = |i: usize, default: u64| {
        args.get(i)
            .map_or(default, |a| a.parse().expect("a number"))
    };
    let refs = num(0, 20_000) as u32;
    let per_ref = num(1, 3_000) as usize;
    let hashes = num(2, 3_600_000);
    let readers = num(3, 3) as usize;
    let time = Duration::from_secs(num(4, 15));
    let batch = num(5, 1) as usize;
    let policies: Vec<String> = if args.len() > 6 {
        args[6..].to_vec()
    } else {
        vec!["0.05:4".into(), "none".into()]
    };

    let t = Instant::now();
    let mut builder = Catalog::builder();
    if gaps {
        builder.ref_coding(RefCoding::Gaps);
    }
    for r in 0..refs {
        builder.add(rows_of(r, per_ref, hashes));
    }
    let catalog = builder.build();
    println!(
        "{} postings, {} references, {:?}, built in {:.1} s; {readers} search threads, {} s a phase, {batch} references an addition",
        catalog.posting_count(),
        refs,
        catalog.ref_coding(),
        t.elapsed().as_secs_f64(),
        time.as_secs(),
    );
    let mut rng = Rng::seeded(99);
    let queries: Vec<_> = (0..200)
        .map(|_| {
            let r = rng.below(u64::from(refs)) as u32;
            query(&mut rng, r, per_ref, hashes)
        })
        .collect();
    let served = RwLock::new(Arc::new(catalog.clone()));
    phase("no writes", &served, &queries, readers, time, None);
    for arg in &policies {
        *served.write().expect("lock") = Arc::new(catalog.clone());
        let writes = Writes {
            policy: policy_of(arg),
            batch,
            per_ref,
            hashes,
        };
        phase(arg, &served, &queries, readers, time, Some(&writes));
    }
}
