# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First release: the engine of the *afpindex* design notes as a library.

### Added

- `Segment`: postings sorted by `(hash, reference, time)` in flat arrays,
  references and times bit-packed to the width of the largest value
  (`Segment::widths`), with an open-addressing directory from hash to run
  that also records each run's multiplicity (the most postings any one
  reference has in it) and no offsets array; build from rows, look a
  hash's run up, iterate, `merge` two segments, `filter` by reference.
- `Catalog` and `CatalogBuilder`: dense reference ids, bulk build with one
  sort, incremental `add` into geometrically merged delta segments,
  `add_many` to add a batch of references with one sort and one merge,
  `remove` by tombstone, `compact_deltas` and `compact`. A merge is one
  pass over every segment it joins, copies runs that need no change as
  packed bits, and looks at references only in segments that may hold a
  removed one. With `std`, a merge of a million postings or more runs on
  every core: the hash space is cut into pieces of about equal postings,
  merged on their own and copied into place in order, for the same
  segment; a compaction ran 3 times faster on 8 vCPUs and 4 times on 16.
  Segments are shared (`Arc`), so a clone is cheap. `add_many` takes a
  batch of references and assigns their ids in the order repeated `add`
  calls would, with one sort over the batch and one merge from the tail;
  a batch of 600 references is 6.6 times faster per reference than 600
  `add` calls.
- `Query::from_pairs`: group a clip's `(hash, time)` rows by hash with a
  per-hash cap on the times kept.
- `Catalog::search`: the two-pass exact search (branch-free totals pass
  with a per-`(hash, reference)` fanout cap, vote floor, per-survivor
  counting-sort scatter, bins, best bin and noise statistics), and
  `Catalog::search_with` handing out a `Survivor` for callers with their
  own scoring. With `std`, `SearchOptions::threads` lets a search that
  reads two million postings or more split its runs between threads,
  each counting into counters of its own that the calling thread then
  adds up; results do not change.
- `SearchOptions`, `Candidate`, `Bin` and `NoiseStats` are
  `#[non_exhaustive]`: options are set by assigning fields on
  `SearchOptions::default()`, and `Bin::new` builds bins for
  `Candidate::from_bins`.
- `vote_floor`: a query-proportional floor.
- `SearchError` bounds: 32-bit totals overflow and a `max_votes` limit on
  the histogram pass, both detected after the totals pass and before any
  histogram work.
- `std` feature: `Catalog::write_to` / `read_from` and `Segment::write_to`
  / `read_from` snapshot files (format version 5), 8-byte aligned, the
  packed postings first and every section with a checksum of its own (the
  packed arrays summed in four interleaved lanes),
  closed by one over the header words and the section checksums;
  references and times are stored packed, as in memory. A segment stores
  each run's multiplicity and its reference range next to the arrays,
  what a load would otherwise recompute from every posting. A load reads
  each array straight into its final buffer, and checks the checksum and
  that the arrays describe a valid segment, a large segment's runs on
  every core; it also refuses a file whose stored multiplicities or range
  are not the ones its runs hold. `read_from_trusted` skips the checksum
  and the order of the runs for files the caller wrote and has checked:
  it still checks the headers, sizes, keys, offsets and id space, and
  takes the stored multiplicities and range without reading a posting; a
  trusted map of the real catalog takes 0.14 s against 0.37 s with
  format 2.
  Version 2 files are not read (`SnapshotError::UnsupportedVersion`);
  write them out again with this version.
- `mmap` feature: `Catalog::map_from` and `Segment::map_from` leave the
  packed postings in the file's pages, shared between processes, with the
  same checks; the checksum runs on a core of its own alongside them.
  `map_from_trusted` maps as `read_from_trusted` reads. The mapping call
  is the feature's one `unsafe` block.
- `CatalogStore` and `Catalog::map_dir` (`mmap`): a catalog kept in a
  directory of immutable segment files and a manifest, for a catalog
  larger than memory. The store is the directory's one writer; a merge
  whose inputs hold at least `spill_postings()` postings is written
  straight to a segment file by the merge that never holds its result and
  mapped back, so a large segment's postings stay in its file's pages.
  `save` writes the segments that have no file, then the manifest, and only
  then deletes what the manifest no longer names, so a crash at any point
  leaves the catalog of the last `save` that returned. `map_dir` reads a
  directory without writing to it and without faulting its pages in, and
  `Catalog::warm_references` reads the packed references of every segment
  into memory once, so the first searches do not wait on the disk while
  the times, which only survivors read, stay there. A
  store (and a builder) holds an exclusive lock on the directory's `LOCK`
  file while it lives, so a second one fails to open instead of removing
  the first one's unsaved files. A store removes only its own segment and
  temporary files, never writes a segment file over an existing one, and
  refuses to save over a manifest another writer replaced, which guards a
  file system that does not honour the lock.
- `CatalogStore::builder` (`mmap`): a bulk build into a catalog directory
  in bounded memory. Rows are sorted a chunk at a time into temporary
  segment files, which `finish` merges eight at a time straight to files
  until one segment is left; ids and segment are those `CatalogBuilder`
  gives. A failed or dropped build leaves nothing behind.
- `simd` feature: on x86-64 CPUs that report AVX2 at run time, the totals
  pass, the search for a survivor's group in a short run and the check of
  every run at load decode packed values eight at a time. The kernel is
  safe code; entering the code compiled for AVX2 is the feature's one
  `unsafe` block. A no-op on other targets and CPUs.
- `set_max_threads` and `max_threads` (`std`): one process-wide cap on
  the threads a sort, merge, directory fill or load takes, where 0 removes
  the cap and 1 keeps every pass on the calling thread. A server can then
  keep a background compaction or a load off the cores its searches run
  on. What a pass computes does not depend on the cap.
- In-place radix sort on the hash when a segment is built from rows,
  whose swaps do not wait for one another and which skips the hash bits a
  bucket's rows agree on, so a build holds no buffer beyond the rows and
  the segment's arrays; with `std`, the buckets are sorted on every core,
  a build of a million postings or more is packed on every core in pieces
  cut at hash boundaries (joined as a merge's pieces are, for the same
  segment), and a large directory is filled a stretch of regions per
  thread.
- Searches arranged for memory-level parallelism without `unsafe`: the
  lookup, totals and histogram passes load their next batch of lines (32
  directory slots, every line of the next 16 runs, 32 vote groups) before
  working on the current batch. The totals pass decodes each run as a bit
  stream, answers the fanout test from a ring of the last 64 references,
  adds without saturation when no counter can overflow, and skips the
  fanout test on runs that cannot reach it; the survivor scan reduces
  each block of counters with a bitwise OR, which vectorises on any
  x86-64 CPU, rather than a maximum, which needs SSE4.1.
- A search that reads fewer postings than the catalog has blocks of 64
  vote counters records the blocks its totals pass writes, a byte per
  block, and its survivor scan reads those blocks only instead of every
  counter of the catalog: a 30-row query over 60 M postings and 2,000,000
  references takes 21.3 M instructions where it took 276.5 M, and an
  800-row one 277.6 M where it took 393.3 M. A search that reads more
  postings than that writes nearly every block and scans them all as
  before.
- A directory of fewer than 2<sup>20</sup> keys leaves four slots per key
  where a larger one leaves 1.7. A lookup that misses then reaches an
  empty slot in under two probes instead of five, and most lookups in a
  delta segment are misses: on a synthetic catalog of six segments a
  search takes 10 % fewer instructions and 31 % less time; single-segment
  catalogs are unchanged; additions are 2 % slower. The price is 48 bytes
  of directory per key instead of 20, at most about 50 MB per segment.
- A merge reserves its writer's key, start and multiplicity arrays at the
  sum of its sources' key counts rather than growing them by doubling,
  which took 2.8 % of the instructions off 600 single additions of 3,000
  rows each; the same goes for a segment built from rows.
- `no_std + alloc` core with no `unsafe` code by default; `libm` is the
  only dependency. Minimum Rust version 1.89.
- Examples: `quickstart` (build, search, add, remove), `bench`
  (synthetic catalog, latency and throughput), `lifecycle` (build,
  additions, compactions, snapshots, query grouping, searches), `replay`
  (real clips against a catalog dumped as raw arrays) and `audiofp`
  (audio fingerprinted by the `audiofp` crate, indexed and searched;
  `audiofp` is a dev-dependency only and needs Rust 1.93).
- Documentation: the README, `docs/USAGE.md`, `docs/PERFORMANCE.md`, and
  `docs/SCALING.md` on sharding by reference and serving replicas from
  segment files.

On an AWS `m7i-flex.large` (two hardware threads) with the 499 M-posting
catalog of the design notes (120,329 references) and its 104 real clips,
`simd` build: 1.94 GiB in memory, a search takes 0.65 ms at the median and
4.5 ms at p95 on one thread, two threads answer 779 searches per second,
and the snapshot loads in 2.0 s (`read_from`) or 0.9 s (`map_from`).
Candidates are identical to the design notes' prototype on all 104 clips.
