# Using afpindex

The [README](../README.md) covers installation and a first search. This is
the reference for what a search computes, how a catalog changes, what a
posting costs, the snapshot files, the limits a search checks, what to
feed the index and how to run it. The API itself is documented on
[docs.rs](https://docs.rs/afpindex).

- [What a search computes](#what-a-search-computes)
- [Catalogs that change](#catalogs-that-change)
- [Memory layout](#memory-layout)
- [Snapshots](#snapshots)
- [Bounds and memory per search](#bounds-and-memory-per-search)
- [Feeding it](#feeding-it)
- [Running in production](#running-in-production)

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

## Memory layout

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

## Snapshots

The `std` feature adds snapshot files; the `mmap` feature maps them.

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
heap (a quarter of the catalog in
[the lifecycle benchmark](PERFORMANCE.md#building-changing-and-loading)), and the same
checksum and structural checks run. The file must not be modified or
truncated while anything mapped from it is alive: write a new snapshot to
a fresh name and rename it over the old one.

## Bounds and memory per search

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

## Feeding it

Any fingerprinter whose output is `(hash: u32, time: u32)` rows works.
The hash space should be large enough that a hash's run stays short
(a few hundred postings at 500 M), and times should be in a unit that
makes a copy's Δ constant, such as STFT frames. The
[`audiofp`](https://crates.io/crates/audiofp) crate's Wang fingerprinter
produces exactly this shape (`WangHash { hash, t_anchor }` at 62.5 frames
per second).
`examples/audiofp.rs` joins the two: it fingerprints three tracks with
`audiofp`, indexes them, and finds ten seconds of one of them, with
another track playing underneath, at its offset (`cargo run --release
--example audiofp`). The glue is one line,
`fingerprint.hashes.iter().map(|h| (h.hash, h.t_anchor))`.

Hashes that appear in nearly every reference (silence, hum) cost the most
and tell the least; drop them from queries before searching, or from
references before indexing.

## Running in production

On Linux with glibc, start the process with huge pages for the heap:

```bash
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ./your-service
```

(`Environment=GLIBC_TUNABLES=glibc.malloc.hugetlb=1` in a systemd unit,
`ENV` in a Dockerfile.) On the real catalog this cut search latency by
7–8 %, raised synthetic throughput by 10–11 %, and brought `read_from`
down from 2.0 s to 1.5 s, at the same peak memory. These are figures
from an AWS `m7i-flex.large`; see
[Allocators and huge pages](PERFORMANCE.md#allocators-and-huge-pages).
The tunable needs transparent huge pages set to `madvise` (Ubuntu's
default) or `always` in `/sys/kernel/mm/transparent_hugepage/enabled`,
and it does nothing on `never` or outside glibc. It belongs to the
program, not to the crate: the crate allocates through the global
allocator and sets none.

Prefer the tunable to swapping the allocator. mimalloc gave the same
search gain, all of it from the huge pages it requests itself. It also
loaded more slowly and peaked 21–58 % higher in memory, because it
copies an array that grows where glibc remaps its pages.
