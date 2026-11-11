# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `SearchOptions::fanout` of `Some(0)` returned wrong candidates in
  release builds and failed an assertion in debug builds. No posting may
  vote, so the search now finds nothing.
- A `fanout` of 255 or more over a run in which one reference has more
  postings than the fanout counted every posting of the run: the run's
  recorded multiplicity saturates at 255, and a saturated value was read
  as exact. Totals came out too high and the bins were wrong. A saturated
  multiplicity now proves nothing.
- `cargo test --no-default-features` did not compile (`vec!` was not in
  scope in the search tests).

### Changed

Measured on an AWS `m7i-flex.large` with `examples/lifecycle.rs`, 60 M
postings, against the code before these changes on the same host.
Snapshot files are byte-identical to before (a test pins them), and so are
search results.

- Merges are one pass over every segment they join instead of two
  segments at a time, so an addition that cascades through several sizes
  rewrites each posting once. Runs that need no change are copied as
  packed bits rather than decoded and re-encoded, and only segments that
  may hold a removed reference have their references looked at (a catalog
  tracks the id range of each segment). 2,000 additions of 3,000 rows:
  4.30 s to 2.28 s, the slowest one 738 ms to 249 ms; `compact_deltas`
  0.66 s to 0.27 s; `compact` 2.81 s to 0.69 s.
- A merge finds the order of a segment's runs with one scan of its
  directory and a radix sort of the entries instead of a directory lookup
  per key, and the directory of a large segment is filled a cache-sized
  region at a time.
- Packed arrays are written through a 64-bit accumulator instead of a
  load, merge and store per value.
- The posting sort behind `Segment::from_postings` and
  `CatalogBuilder::build`: swaps no longer wait for one another, hash bits
  that a bucket's rows agree on are skipped, small buckets take a narrower
  digit, and a bucket of one hash is ordered by a single 64-bit
  comparison. Bulk build 4.33 s to 2.65 s; `from_postings` on 60 M rows of
  sparse 32-bit hashes 4.4 s to 2.1 s. Still in place, still no buffer
  beyond the bucket tables.
- `Catalog::add` sorts its one reference as 64-bit keys and writes its
  segment directly.
- Snapshot loads read each array straight into its final buffer and check
  runs through one 64-bit comparison per posting: `read_from` 0.53 s to
  0.46 s, `map_from` 0.44 s to 0.32 s.
- `Query::from_pairs` groups with a stable radix sort and allocates its
  arrays once at their final size: 17.5 µs to 10.1 µs for 798 rows.
- The minimum Rust version is 1.87 (was 1.85): the vector kernel uses
  intrinsics that are safe to call from 1.87 on, and the default build
  uses `is_multiple_of`, stable from 1.87.
- The totals pass loads every cache line of the next batch of runs
  before counting the current one, not just each run's first and last
  entry, and for a run shorter than two lines also the line past it that
  the decoder reads. On an AWS `m7i-flex.large` with the real catalog
  (499 M postings, 136 per hash) and its 104 clips, against the code
  before this release's search changes on the same host: summed
  best-of-three latency 176.8 ms to 157.2 ms, p95 7.4 ms to 5.9 ms,
  candidates identical; synthetic catalogs of 136 postings per hash
  2,067 to 2,596 searches per second, of 17 per hash unchanged.
- Snapshot loads check a large segment's runs on every core, and
  `map_from` checksums the packed arrays on a core of their own at the
  same time; the file format is unchanged. On the two-vCPU host above,
  `map_from` of the real catalog 1.8 s to 1.6 s (1.2 s to 1.0 s with
  `simd`); `read_from` is within 0.1 s, as its checksum runs while the
  file is read.
- The histogram pass no longer searches for the end of a survivor's
  group in a run where no reference has two postings, and bounds the
  search by the run's multiplicity elsewhere.
- CI: read-only token, superseded runs are cancelled, clippy also runs on
  the default features, and the lifecycle example runs as a smoke test.

### Added

- `examples/audiofp.rs`: audio fingerprinted by the `audiofp` crate's
  Wang fingerprinter, indexed and searched, with the offset of a clip
  checked. `audiofp` is a dev-dependency only, and it needs Rust 1.93;
  the minimum version of the library stays 1.87.
- `simd` feature: on x86-64 CPUs that report AVX2 at run time, the
  totals pass, the search for a survivor's group in a short run and the
  check of every run at snapshot load decode packed values eight at a
  time. The kernel is safe code; entering the code compiled for AVX2 is
  the feature's one `unsafe` block. On the host and catalog above, on
  top of the default build: summed latency 157.2 ms to 136.9 ms, p50
  0.80 ms to 0.71 ms, p95 5.9 ms to 5.1 ms, two threads 629 to 737
  searches per second, `read_from` 2.7 s to 2.1 s, `map_from` 1.6 s to
  1.0 s, candidates identical; synthetic catalogs 12 % (17 postings per
  hash) and 16 % (136) more searches per second. Off by default, and a
  no-op on other targets and CPUs.
- `examples/lifecycle.rs`: times the bulk build, additions, both
  compactions, the snapshot (written, read and mapped), query grouping and
  searches from a strict vote floor to none.
- README: the `mmap` feature (`Catalog::map_from`, `Segment::map_from`),
  which was undocumented there.

## [0.1.0] - 2026-11-07

First release: the engine of the *afpindex* design notes as a library.

### Added

- `Segment`: postings sorted by `(hash, reference, time)` in flat arrays,
  references and times bit-packed to the width of the largest value
  (`Segment::widths`), with an open-addressing directory from hash to run
  and no offsets array (iteration looks keys up); build from rows, look a
  hash's run up, iterate, merge two segments, filter by reference.
- `Catalog` and `CatalogBuilder`: dense reference ids, bulk build with one
  sort, incremental `add` into geometrically merged delta segments,
  `remove` by tombstone, `compact_deltas` and `compact`. Segments are
  shared (`Arc`), so a clone is cheap.
- `Query::from_pairs`: group a clip's `(hash, time)` rows by hash with a
  per-hash cap on the times kept.
- `SearchOptions`, `Candidate`, `Bin` and `NoiseStats` are
  `#[non_exhaustive]`: options are set by assigning fields on
  `SearchOptions::default()`, and `Bin::new` builds bins for
  `Candidate::from_bins`.
- `Catalog::search`: the two-pass exact search (branch-free saturating
  totals pass with a per-`(hash, reference)` fanout cap, vote floor,
  per-survivor counting-sort scatter, bins, best bin and noise statistics),
  and `Catalog::search_with` handing out a `Survivor` for callers with
  their own scoring.
- `vote_floor`: a query-proportional floor.
- `SearchError` bounds: 32-bit totals overflow and a `max_votes` limit on
  the histogram pass, both detected after the totals pass and before any
  histogram work.
- `std` feature: `Catalog::write_to` / `read_from` and `Segment::write_to`
  / `read_from` snapshot files (format version 2), 8-byte aligned with a
  checksum trailer that covers the header words as well as the arrays, and
  validation on load; references and times are stored packed, as in memory.
- In-place radix sort on the hash when a segment is built from rows, so a
  build holds no buffer beyond the rows and the segment's arrays; with
  `std`, the radix buckets are sorted on every core.
- Memory-level parallelism without `unsafe`: the lookup, totals and
  histogram passes load their next batch of lines (directory slots, run
  references, group times) before working on the current batch, and the
  survivor scan reduces counter blocks on vector instructions.
- Totals pass decodes each run as a bit stream, answers the fanout test
  from a ring of the last 64 references and adds without saturation when
  no counter can overflow: 10 % fewer instructions per search on the real
  clips, same candidates; on an AWS `m7i-flex.large` p50 fell from 0.98 to
  0.89 ms and two-thread throughput rose from 503 to 546 searches/s over
  five alternating rounds.
- Directory slots record each run's multiplicity (the most postings any
  one reference has in it), so a run that cannot reach the fanout skips
  the fanout test in the totals pass and the group search in the
  histogram pass: a third fewer instructions per search on the real clips
  (25.2 M to 16.9 M), same candidates.
- Touch-ahead batch sizes (32 directory slots, 16 runs, 32 vote groups)
  chosen by measurement: half of them cost 13 % of search time, double
  gained nothing more.
- `examples/replay.rs`: replay real clips against a catalog dumped as raw
  arrays and write candidates and timings. On the 499 M-posting catalog of
  the design notes the candidates are identical to the prototype's on all
  104 clips at 1.94 GiB, p50 1.10 ms and 1,456 searches/s on 4 threads;
  on an AWS `m7i-flex.large` (2 vCPUs) the snapshot loads in 3.8–4.2 s
  and the same clips give p50 0.89 ms and 537–546 searches/s on two
  threads.
- `no_std + alloc` core with no unsafe code; `libm` is the only dependency.
