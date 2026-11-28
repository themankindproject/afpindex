# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-28

Additions for running replicas and exporting metrics. The file formats
are unchanged: 0.1.0 and 0.1.1 read and write the same directories.

### Added

- `Catalog::stats`, with `CatalogStats` and `SegmentStats`: the
  catalog's counters and each segment's postings, keys, heap bytes,
  reference coding and reference range, in one call.
- `Manifest` and `ManifestEntry` (`mmap`): `Manifest::read(dir)` and
  `Manifest::from_bytes` read a catalog directory's manifest, with every
  check of a load, and give its generation, id space, liveness, reference
  coding and segment files without mapping a segment.
- `replica::sync(src, dst)` (`mmap`): brings a directory up to another
  catalog directory, copying the segment files it lacks, checking each
  fully before it takes its name, and writing the manifest last. A file
  already present is kept only when its length and footer checksums match,
  so a source created again from nothing is copied again rather than mixed
  in. `replica::prune(dir)` removes the files a directory's manifest no
  longer names. Both hold the directory's lock.
- Fuzzing of the manifest reader and of `replica::sync` on damaged
  directories.

## [0.1.0] - 2026-11-25

First release: the engine of the *hashtally* design notes as a library.

### Added

- `Segment`: postings sorted by `(hash, reference, time)` in flat
  arrays, times bit-packed, an open-addressing directory from hash to run
  that also records each run's multiplicity; build from rows, look up,
  iterate, `merge`, `filter`.
- `RefCoding`: references packed at the width of the largest id (the
  default) or coded as the gaps inside each run (12 % smaller, 7–48 %
  slower to search), chosen per catalog or per segment; segments of both
  codings search and merge together.
- `Catalog` and `CatalogBuilder`: dense reference ids, bulk build with one
  sort, `add` and `add_many` into geometrically merged delta segments,
  `remove` by tombstone, `compact_deltas` and `compact`; merges of a
  million postings or more run on every core, and a clone shares its
  segments.
- `Catalog::compaction_due` and `CompactionPolicy`: when a compaction pays
  for itself, by the share of postings the deltas hold, the removals from
  the largest segment, and the number of deltas.
- `Query::from_pairs`, `Catalog::search` (the exact two-pass search:
  totals with a fanout cap, vote floor, histogram, best bin and noise
  statistics), `Catalog::search_with` for callers with their own scoring,
  `vote_floor`, and `SearchOptions::threads` to split one large search
  between threads. `SearchError` reports a totals overflow and the
  `max_votes` bound before any histogram work.
- `std` feature: checksummed snapshot files (format version 1) with
  `write_to`, `read_from` and `read_from_trusted`, references and times
  stored as in memory; `set_max_threads` caps the threads a sort, merge,
  directory fill or load takes.
- `mmap` feature: `map_from` and `map_from_trusted` leave the postings in
  the file's pages; `CatalogStore` keeps a catalog in a directory of
  immutable segment files and a manifest, with merges written straight to
  files, crash-safe `save`, a single-writer lock and a bounded-memory
  `CatalogStore::builder`; `Catalog::map_dir` and `warm_references` serve
  it read-only.
- `simd` feature: AVX2 decoding of packed values, picked at run time on
  x86-64 and a no-op elsewhere.
- `no_std + alloc` core with no `unsafe` code by default (`mmap` and
  `simd` bring one block each); `libm` is the only dependency. Minimum
  Rust version 1.89.
- Examples (`quickstart`, `bench`, `lifecycle`, `serve`, `replay`,
  `profile`, `audiofp`) and documentation: the README, `docs/USAGE.md`,
  `docs/PERFORMANCE.md` and `docs/SCALING.md`.

On an AWS `m7i-flex.large` (two hardware threads) with the 499 M-posting
catalog of the design notes (120,329 references) and its 104 real clips,
`simd` build, references packed: 1.94 GiB in memory (1.70 GiB coded as
gaps), a search takes 0.65 ms at the median and 4.5 ms at p95 on one
thread, two threads answer 779 searches per second, and the snapshot
loads in 2.0 s (`read_from`) or 0.9 s (`map_from`). Candidates are
identical to the design notes' prototype on all 104 clips.
