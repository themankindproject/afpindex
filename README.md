# afpindex

Exact in-memory inverted index for landmark audio fingerprints.

A landmark fingerprinter (Wang / Shazam-style, Panako, and friends) turns
audio into rows of `(hash, time)`. A **posting** records one occurrence of a
hash in a reference: `(hash, reference, time)`. A copy of a reference keeps
its hashes and shifts their times by one constant, so every query row and
every posting with the same hash cast a **vote** for the bin
`(reference, Δ = time − query_time)`. A true copy piles votes into one bin;
chance collisions scatter single votes everywhere.

`afpindex` answers that search exactly, over hundreds of millions of postings,
in milliseconds:

- **Postings sorted by hash** in flat arrays, references and times packed
  to the width of the largest value (about 4 bytes per posting instead of
  8). Each hash is one contiguous run, found by a single directory read. A
  query's lookups are independent, and every pass loads its next batch of
  lines before working on the current one, so a search is bounded by memory
  parallelism rather than by one cache miss after another.
- **Count before histogram.** A branch-free first pass totals each
  reference's votes in a dense counter array; only references over a vote
  floor get a histogram. Chance matches never pay for a sort.
- **Immutable, shared segments.** Additions go into small delta segments
  merged geometrically; removals tombstone a reference, and compaction
  drops its postings. Reference ids never change, and a catalog clone
  shares its segments.
- **No unsafe code by default, one dependency (`libm`), `no_std + alloc`
  core.** The `std` feature adds checksummed snapshot files. Two more
  features bring one `unsafe` block each: `mmap` maps those files instead
  of reading them (the mapping call, and one more dependency, `memmap2`),
  and `simd` decodes packed postings eight at a time on x86-64 CPUs with
  AVX2 (the entry into the code compiled for it).

## Quick start

```rust
use afpindex::{Catalog, Query, Scratch, SearchOptions, vote_floor};

// Build a catalog with one sort over every posting.
let mut builder = Catalog::builder();
for reference in references {                    // your fingerprints
    let id: u32 = builder.add(reference.rows()); // impl IntoIterator<Item = (u32, u32)>
}
let catalog = builder.build();

// Group a query clip's rows by hash, keeping the first 64 times per hash.
let query = Query::from_pairs(clip.rows(), Some(64));
let mut options = SearchOptions::default();
options.min_total_votes = vote_floor(query.input_rows(), 16, 0.02);

// One scratch per thread; the catalog is shared by reference.
let mut scratch = Scratch::new();
for c in catalog.search(&query, &options, &mut scratch)? {
    println!(
        "reference {} at Δ {}: {} votes in the best bin, {} total, noise mean {:.2}",
        c.reference, c.delta, c.votes, c.total_votes, c.noise.mean
    );
}
```

`examples/quickstart.rs` is a runnable version with additions and removals.

## What a search computes

| Step | Rule |
|---|---|
| Query grouping | Distinct hashes, each with its first `max_times_per_hash` times in input order (`Query::from_pairs`). Duplicate rows vote on their own. |
| Fanout | Per `(hash, reference)`, only the first `fanout` postings by time vote (default 64; `None` for all). |
| Totals pass | Each reference's total votes, exactly, in one 32-bit counter per reference. |
| Vote floor | A reference needs `min_total_votes` to become a **survivor** (`vote_floor` gives a query-proportional value). |
| Histogram | Each survivor's votes, sorted by Δ and merged into `Bin`s with the first and last query time. |
| Candidate | The best bin (most votes; ties to the smaller Δ) and noise statistics over the bins more than `offset_tolerance` away from it: count, mean, standard deviation and max, or zeros below `min_noise_bins`. |
| Order | Candidates ascend by reference id. Everything is deterministic. |

`Catalog::search_with` hands each `Survivor` (reference, total votes, bins)
to a closure instead, for callers with their own scoring;
`Survivor::candidate` is the reduction `search` applies.

## Catalogs that change

```rust
let mut catalog = Catalog::builder().build();     // or Catalog::new()
let id = catalog.add(rows);                       // one sorted delta segment, merged in
catalog.remove(id);                               // tombstone: skipped by every later search
catalog.compact_deltas();                         // merge the deltas, drop dead postings
catalog.compact();                                // one segment for everything
```

Segments keep geometrically decreasing sizes: each is less than half of
the one before it, so a posting takes part in `O(log n)` merges over its
life and a catalog of one snapshot plus a day of additions holds about ten
segments. A search looks every query hash up in every segment, so compact
when the list grows; `compact_deltas` never rewrites the largest segment
and costs one pass over the deltas.

A merge is one pass over every segment it joins, however many: an
addition whose segment cascades through several sizes rewrites each
posting once, not once per size. A run whose hash is in one segment, or
whose segments' references do not interleave, is copied as packed bits
without being decoded, and only a segment that may hold a removed
reference has its references looked at.

Segments are shared between clones, so `catalog.clone()` is cheap and a
server can clone, compact the clone, and swap it in without blocking the
searches running on the original.

`Catalog::from_segment` wraps a `Segment` whose reference ids you assigned
yourself, for example from a database export; every id up to the largest
is live and the next `add` follows it.

A posting costs its two widths: a segment packs references at the bits of
its largest reference id and times at the bits of its largest time, so
20,000 references over 45 minutes of 16 ms frames cost 15 + 18 bits per
posting. `Segment::widths` reports them. The directory is an
open-addressing table rebuilt from the keys whenever a segment is built or
loaded (never stored), about 20 bytes per distinct hash on top of the
4-byte key; the slot of a hash depends only on the hash, so hash layouts
that vary in few bits still spread over it. A slot holds the run's start
and length and the most postings any one reference has in it, which is
what lets the search skip the fanout test on most runs. There is no offsets array:
iteration in key order looks each key up in the directory, and a merge
recovers the order of the runs from one scan of the directory and a sort
of what it finds (16 bytes per distinct hash while the merge runs). A
directory too large for the cache is filled a region at a time from a
copy of its entries sorted by region, another 12 bytes per hash until it
is built.

## Snapshots (`std`)

```rust
catalog.compact();                                 // optional: drop removed postings first
catalog.write_to(std::io::BufWriter::new(std::fs::File::create("catalog.afx")?))?;
let catalog = Catalog::read_from(std::io::BufReader::new(std::fs::File::open("catalog.afx")?))?;
```

A catalog snapshot stores the id space, the liveness table and every
segment, so ids and removals survive the round trip and the next `add`
returns the id it would have returned before. `Segment::write_to` and
`Segment::read_from` do the same for one segment. Both formats (version 2)
are little-endian arrays behind a 32-byte header, every section 8-byte
aligned, with a checksum trailer over the header and the arrays;
references and times are stored packed as they are in memory, so a file is
about the size of the catalog. Reading verifies the checksum and that the
arrays describe a valid segment before building the directory.

```rust
let catalog = Catalog::map_from("catalog.afx")?;   // `mmap` feature
```

With the `mmap` feature, `Catalog::map_from` and `Segment::map_from` map
the file instead of reading it. The packed postings stay in the file's
pages, shared by every process that maps the file and reclaimable by the
kernel without a write; only the keys and the directory are built on the
heap (a quarter of the catalog in the benchmark below), and the same
checksum and structural checks run. The file must not be modified or
truncated while anything mapped from it is alive: write a new snapshot to
a fresh name and rename it over the old one.

## Bounds

Two limits are checked after the totals pass, before any histogram work,
and return a `SearchError` with the scratch left clean:

- `TotalsOverflow` if a reference's vote total passed `u32::MAX`; the
  counters saturate so the overflow is detected, not wrapped. Cap the
  query's times per hash or the fanout.
- `TooManyVotes` if the survivors would scatter more than
  `SearchOptions::max_votes` votes (16 bytes each; default about 1 GiB).

Memory per search is 4 bytes and one bit per reference id (a counter that
doubles as the survivor index, and a survivor bit, reused by the `Scratch`)
plus 16 bytes per survivor vote and 32 bytes per survivor group.

## Performance

`examples/bench.rs` builds a synthetic catalog, plants a window of one
reference in each query with half as many random rows, and checks that
every query finds its reference at the planted offset. On an Intel
i5-1135G7 laptop (4 cores, 15 GB), 20,000 references × 3,000 postings over
3.6 M distinct hashes, on an otherwise idle machine:

| | |
|---|---|
| postings | 60 M, 15-bit references and 18-bit times |
| memory | 0.31 GiB with the directory (0.58 GiB with 32-bit arrays) |
| build | 2.4–2.5 s (radix sort, buckets sorted on all cores) |
| query | 799 rows at the median |
| search, one thread | p50 0.10–0.12 ms, p95 0.11–0.14 ms over four runs |
| searches per second, 4 threads | 33,000–34,000 from a cool start |

```bash
cargo run --release --example bench -- 20000 3000 3600000 200 4
```

Those numbers are for the generic x86-64 target. Building with
`RUSTFLAGS="-C target-cpu=native"` gained a further 5 % on this machine
(a vectorised unsigned maximum in the survivor scan, BMI2 shifts in the
packed loads); the crate sets no target features itself, and the `simd`
feature picks its code when a pass starts instead. The four-thread
figure is the first five seconds from a cool package: past 75 °C this
laptop throttles to about 20,000 searches per second, whatever the build.

A search here touches about 800 runs, each a few cache lines in a 0.5 GiB
array, so it is bound by memory latency and parallelism rather than by
arithmetic. The passes are arranged for that: a hash resolves in one
directory read whose address depends only on the hash; each pass loads its
next batch of lines (32 directory slots, every line of 16 runs, 32 vote
groups; half that costs 13 % of search time, double gains nothing more)
before working on the current batch, with plain loads that overlap the way a
prefetch would; the totals pass decodes each run as a bit stream, answers
the fanout test from a ring of the last 64 references instead of a second
read, and uses plain adds whenever no counter can overflow; a run where
no reference has as many postings as the fanout (every run but 2 % of
postings on the real catalog) skips the fanout test in both passes, which
took a third off the instructions per search on the real clips; the
survivor scan reduces each block of counters to its maximum on vector
instructions; and the
histogram pass finds a survivor's group in a run by binary search rather
than a walk. On the same machine the version without those arrangements
took 0.30 ms at the median and 6,700 searches per second on four threads:
user-mode cycles per search fell by three times, instructions by a third,
last-level cache misses by a third. Packing the postings afterwards cut
memory by 41 % for 3–8 % more cycles per search (each reference read is a
64-bit load, a shift and a mask, partly paid back by fewer cache lines per
run).

Four things measured and left out: transparent huge pages cut TLB misses
by a third but cycles by 2 %, since page walks overlap like the other
misses; and a cache-sized bucket table over sorted `(key, offset)` pairs
in place of the hash directory saved 10 % of memory but cost 11–15 % of
search time, because the posting stream evicts the table and every lookup
then misses twice; and a denser directory (three keys per four slots
instead of five) saved 3 % of memory but cost a third more per lookup of a
hash the segment does not hold, which real queries are full of; and a
bucketed histogram for the totals pass (scatter each counting posting by
the high bits of its reference, then count each bucket into counters that
fit L1) ran 62 % more instructions per search on the real clips, because
on real data the search is bound by instruction throughput at three
instructions per cycle, not by the counter array.

### Building, changing and loading

`examples/lifecycle.rs` times everything around a search on the same
synthetic catalog: the bulk build, 2,000 additions of 3,000 rows each (a
tenth of them removed again), both compactions, the snapshot written to
and read from memory and mapped from a file, and query grouping. On an
AWS `m7i-flex.large` (two hardware threads of one Sapphire Rapids core,
8 GiB), generic x86-64 build, next to the code before the changes the
changelog lists under *Unreleased*, both run on the same host:

| | before | now |
|---|---|---|
| bulk build, 60 M postings (rows generated on the clock) | 4.33 s | 2.65 s |
| `Catalog::add` of 3,000 rows into 60 M postings | 2.15 ms on average, 738 ms at worst | 1.14 ms on average, 249 ms at worst |
| `compact_deltas`, four deltas of 5.4 M postings | 0.66 s | 0.27 s |
| `compact`, 65 M postings | 2.81 s | 0.69 s |
| `write_to` memory, 0.29 GiB | 0.34 s | 0.31 s |
| `read_from` memory | 0.53 s | 0.46 s |
| `map_from` a file in the page cache | 0.44 s | 0.32 s |
| `Query::from_pairs`, 798 rows | 17.5 µs | 10.1 µs |
| search, one thread | 0.131 ms | 0.129 ms |

```bash
cargo run --release --features mmap --example lifecycle -- 20000 3000 3600000 2000
```

The build sorts in place with a radix sort whose swaps do not wait for one
another: the row a swap brings back is left for the next round instead of
being followed, so the cache misses of 2,048 bucket heads overlap. It
skips the hash bits a bucket's rows agree on and takes a narrower digit
for a small bucket, which is what sparse 32-bit hashes with a hundred
rows each need: `Segment::from_postings` on 60 M such rows fell from
4.4 s to 2.1 s, and from 1.8 s to 1.2 s when they arrive sorted. The worst
addition is the one whose merge cascades furthest, and it is now one pass
over the segments it reaches. A compaction copies runs as packed bits. A
load is bound by the structural check, which compares every posting with
the one before it, then by the checksum and the directory. The laptop
table above was measured before these changes; its build row is the
earlier sort.

Two things measured here and left out: checking a run's references and
its times as two streaming passes instead of one loop over both arrays
made a load a quarter slower, on runs of 17 postings and of 136, and huge pages
for the allocator (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`) took 4.5 % off
the additions, which is a deployment choice rather than the crate's.

### Real data

`examples/replay.rs` runs the crate over the catalog of the *afpindex*
design notes (7 November 2026): 499.4 M postings from 120,329 production
references, 3.66 M distinct hashes (136 postings per hash on average), and
the 104 real clips the notes compared with the production ClickHouse
query. Same machine as above, generic x86-64 build:

| | this crate | prototype in the notes |
|---|---|---|
| memory | 1.94 GiB (17-bit references, 15-bit times) | 3.0 GiB packed, 3.8 GiB raw |
| build from rows in memory | 8.3 s | 333 s including the export |
| search, one thread | p50 1.10 ms, p95 8.8 ms, max 14.2 ms | p50 2.9 ms |
| searches per second, 4 threads | 1,456 | 890 |
| candidates | identical to the prototype on all 104 clips | 103 of 104 match ClickHouse |

The one clip where both differ from ClickHouse is the one where ClickHouse's
`LIMIT 64 BY` keeps an arbitrary 64 of 74 postings of a hash and the exact
engines keep the first 64 by time.

On a production-sized host, an AWS `m7i-flex.large` (two hardware threads
of one Sapphire Rapids core, 8 GiB) loading the catalog from its snapshot
file, generic x86-64 build, five alternating rounds of each build on the
same host:

| | before the streaming totals pass | current |
|---|---|---|
| load from file | 3.8 s | 3.8 s |
| search, one thread | p50 0.98 ms, p95 8.7 ms, max 14.4 ms | p50 0.89 ms, p95 8.6 ms, max 14.2 ms |
| searches per second, two threads | 479–503 | 537–546 |

Candidates were identical between the two builds, the laptop run and the
prototype on all 104 clips. Hosts of this instance type differ by about
10 % (an earlier one gave p50 1.11 ms and 456–460 searches/s for the
older build, and 9 % less with `-C target-cpu=native`), so compare columns
within one table only. The hypervisor exposes no hardware counters on this
size, so cycle counts come from the laptop. The design notes estimated 50
to 450 searches per second for a two-vCPU task.

### Run lines, vector decoding and parallel loads

Three changes since the tables above, measured on an AWS
`m7i-flex.large` (two hardware threads of one Sapphire Rapids core),
generic x86-64 build, with the real catalog and its 104 clips, each
build in three alternating rounds on one host; candidates were
identical in every run:

| | before | default build | `simd` |
|---|---|---|---|
| search, one thread, best of three summed over the clips | 176.8 ms | 157.2 ms | 136.9 ms |
| search, one thread, p50 / p95 | 0.79 / 7.4 ms | 0.80 / 5.9 ms | 0.71 / 5.1 ms |
| searches per second, two threads | 620 | 629 | 737 |
| `read_from` / `map_from` | 2.8 / 1.8 s | 2.7 / 1.6 s | 2.1 / 1.0 s |
| synthetic, 136 postings per hash, searches per second | 2,067 | 2,596 | 3,008 |
| synthetic, 17 postings per hash, searches per second | 7,634 | 7,521 | 8,432 |

**Every line of a run.** A real run is 136 references of 17 bits, five
cache lines, and the totals pass used to load only each run's first and
last entry ahead of time; the middle lines were missed on demand. It now
loads every line of the next 16 runs, and for a run shorter than two
lines also the line past it, which the decoder reads into (8 bytes past
the last value one at a time, up to 31 in blocks). That took a fifth off
the p95 and a ninth off the summed latency; the catalogs of 17 postings
per hash, whose runs fit a line or two, are unchanged.

**Vector instructions (`simd`).** With the `simd` feature, on x86-64,
the passes that read many packed values ask the CPU for AVX2 when they
start and, if it has it, decode eight values at a time (one byte shuffle
and one variable shift per block, for widths up to 25 bits): the totals
pass, the search for a survivor's group in a short run, and the check of
every run when a snapshot is loaded. Everything else, and every other
CPU, runs the same code as without the feature.

**Parallel loads.** A large segment's runs are checked on every core,
and `map_from` checksums the packed arrays on a core of their own at the
same time; the format is unchanged. Two hardware threads of one core
gain little from it (the table's load column); more cores should gain
more, which was not measured.

**Huge pages.** Running the same builds with
`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` took a further 7–13 % off
search and 14–19 % off `read_from` on this host, depending on the build.
It also removed a gap that is not in the code: the build from before
these changes and the default build without the run-line change
measured 5–7 % apart on the real clips without it, and the same with
it. Compare builds with it set, or expect that much noise. For
deployments, see [Running it](#running-it).

Measured here and left out:

- Loading runs further ahead (32 or 64 runs instead of 16) was within
  noise of 16.
- Loading the line past a run whatever its length gained 8 % on runs of
  17 postings but cost 2.5 % on runs of 136, where that line is mostly
  the next run's; hence the two-line limit.
- Counters: folding every reference into 4,096 counters (wrong answers,
  timing only) took a third off the totals pass with `simd`, so the
  counter array is that much of it. Staging each counting posting in a
  buffer per 8,192 references and counting each full buffer into a
  32 KiB slice of the counters, the way to keep them in L1 with right
  answers, made searches 2.3 times slower: a run's references ascend, so
  consecutive postings append to the same buffer and wait on one
  another's length.
- Reading up to 64 entries of a run in blocks to find a survivor, or
  halving a longer run down to 32 entries first, cost a sixth to a half
  more of that pass than halving alone, so only ranges of 24 entries or
  fewer are read in blocks.
- Storing the directory in the snapshot instead of rebuilding it would
  save 0.05 s of a load; the run check and the checksum are the rest.
- mimalloc as the programs' global allocator (crate 0.1.52, its v3 and
  its v2) gains what huge pages gain and nothing more. v3 asks for them
  itself and then matched glibc with the tunable: 8 % off the summed
  latency of the real clips and 10–12 % more synthetic searches per
  second than plain glibc, within about 1 % of the tunable. With its huge
  pages turned off (`MIMALLOC_ALLOW_THP=0`) it matched plain glibc. It
  also loaded the real catalog in 1.9 s against 1.5 s with the tunable
  and peaked at 2.56 GB against 2.12 GB (1.88 against 1.19 GB in the
  lifecycle run). A load grows each array by doubling: glibc's `realloc`
  moves a large block by remapping its pages, mimalloc's copies it.

## Feeding it

Any fingerprinter whose output is `(hash: u32, time: u32)` rows works.
The hash space should be large enough that a hash's run stays short
(a few hundred postings at 500 M), and times should be in a unit that
makes a copy's Δ constant, such as STFT frames. The
[`audiofp`](https://crates.io/crates/audiofp) crate's Wang fingerprinter
produces exactly this shape (`WangHash { hash, t_anchor }` at 62.5 frames
per second).

Hashes that appear in nearly every reference (silence, hum) cost the most
and tell the least; drop them from queries before searching, or from
references before indexing.

## Running it

On Linux with glibc, start the process with huge pages for the heap:

```bash
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ./your-service
```

(`Environment=GLIBC_TUNABLES=glibc.malloc.hugetlb=1` in a systemd unit,
`ENV` in a Dockerfile.) On the real catalog this cut search latency by
7–8 %, raised synthetic throughput by 10–11 %, and brought `read_from`
down from 2.0 s to 1.5 s, at the same peak memory. These are figures
from an AWS `m7i-flex.large`; see
[Run lines, vector decoding and parallel loads](#run-lines-vector-decoding-and-parallel-loads).
The tunable needs transparent huge pages set to `madvise` (Ubuntu's
default) or `always` in `/sys/kernel/mm/transparent_hugepage/enabled`,
and it does nothing on `never` or outside glibc. It belongs to the
program, not to the crate: the crate allocates through the global
allocator and sets none.

Prefer the tunable to swapping the allocator. mimalloc gave the same
search gain, all of it from the huge pages it requests itself. It also
loaded more slowly and peaked 21–58 % higher in memory, because it
copies an array that grows where glibc remaps its pages.

## Features and `no_std`

| Feature | Default | Adds |
|---|---|---|
| `std` | yes | `Catalog` and `Segment` snapshot I/O, `std::error::Error` for `SnapshotError` |
| `mmap` | no | `Catalog::map_from` and `Segment::map_from` (implies `std`; adds `memmap2`) |
| `simd` | no | AVX2 decoding of packed postings on x86-64, chosen at run time (implies `std`; a no-op elsewhere) |

```toml
[dependencies]
afpindex = { version = "0.1", default-features = false }   # no_std + alloc
```

The core needs only `alloc`, and `Arc`-shared segments need a target with
atomic pointers (`Rc` is used otherwise, and the catalog is then not
`Send`).

## Roadmap

- Parallel merge, and a parallel first level of the build sort.

## License

MIT. See `LICENSE`.
