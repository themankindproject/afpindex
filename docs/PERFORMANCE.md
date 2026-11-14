# Performance

Every number behind the README, how each was measured, how the passes are
arranged, and what was tried and left out. Two machines appear:

- an Intel i5-1135G7 laptop (4 cores, 15 GB), on an otherwise idle
  machine, where cycle counts are available;
- an AWS `m7i-flex.large` (two hardware threads of one Sapphire Rapids
  core, 8 GiB), the size of a production task, whose hypervisor exposes
  no hardware counters.

Builds are for the generic x86-64 target unless a row says otherwise.
Where two builds are compared, they ran in alternating rounds on the same
host, and their candidates were identical. Hosts of the AWS type differ
by about 10 %, so compare columns within one table only.

- [Synthetic catalog](#synthetic-catalog)
- [Building, changing and loading](#building-changing-and-loading)
- [Real data](#real-data)
- [Run lines, vector decoding and parallel loads](#run-lines-vector-decoding-and-parallel-loads)
- [Sizing a merge's key arrays once](#sizing-a-merges-key-arrays-once)
- [Adding a batch](#adding-a-batch)
- [A sparser directory for small segments](#a-sparser-directory-for-small-segments)
- [Allocators and huge pages](#allocators-and-huge-pages)
- [Reproducing](#reproducing)

## Synthetic catalog

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
survivor scan reduces each block of counters with a bitwise OR, which
vectorises on any x86-64 CPU; and the
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

## Building, changing and loading

`examples/lifecycle.rs` times everything around a search on the
synthetic catalog of the section above: the bulk build, 2,000 additions
of 3,000 rows each (a tenth of them removed again), both compactions,
the snapshot written to and read from memory and mapped from a file, and
query grouping. On the AWS `m7i-flex.large`, next to an earlier version
of the code without the build, merge and load work described below, both
run on the same host:

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

## Real data

`examples/replay.rs` runs the crate over the catalog of the *afpindex*
design notes (7 November 2026): 499.4 M postings from 120,329 production
references, 3.66 M distinct hashes (136 postings per hash on average), and
the 104 real clips the notes compared with the production ClickHouse
query. On the laptop:

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

## Run lines, vector decoding and parallel loads

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
deployments, see [Running in production](USAGE.md#17-running-in-production).

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
- AVX-512 decoding made every pass slower. The kernel decoded sixteen
  values per block with one cross-lane byte permute (VBMI `vpermb`, its
  64-byte window as the operand) and one variable shift, and it was
  picked at run time ahead of AVX2. Measured against the AVX2 build of
  the same code on an AWS `m7i-flex.large` (Xeon 8488C), five
  alternating rounds with huge pages on:

  | | AVX2 | AVX-512 |
  |---|---|---|
  | real clips, summed latency, catalog loaded | 139.4 ms | 142.3 ms (+2 %) |
  | real clips, summed latency, catalog mapped | 145.9 ms | 150.0 ms (+3 %) |
  | real catalog, `read_from` | 1.74 s | 1.85 s (+6 %) |
  | real catalog, `map_from` | 1.03 s | 1.12 s (+9 %) |
  | synthetic, runs of 17, searches per second | 8,815 | 8,675 (−2 %) |
  | synthetic, runs of 136, searches per second | 3,121 | 2,788 (−11 %) |

  Decoding is a small share of each pass, and runs of 17 postings fill
  one block of sixteen. The loss was largest on runs of 136, where the
  kernel runs most. That points to a cost the 512-bit instructions put
  on the rest of the core rather than to the kernel. Most likely it is a
  lower clock and a vector port given up while they run; this host has
  no counters to confirm it. The kernel also needs Rust 1.89 for the
  AVX-512 intrinsics.

## Sizing a merge's key arrays once

A merge writes its runs into a writer whose key, start and
multiplicity arrays used to start empty and grow by doubling, one copy
of them after another; about 3 % of the instructions of an addition went
under `realloc` and the copies behind it. The writer now takes the number
of keys up front and reserves the three arrays: a merge passes the sum of
its sources' key counts, an upper bound, since a hash in several of them
is one key, and `finish` cuts the arrays the merge did not need.
`from_postings` and `from_reference` pass what bounds them the same way,
and `starts` reserves one past that for the count `finish` records.

On the synthetic 60 M-posting catalog, 600 additions of 3,000 rows each
(`examples/profile.rs`, `adds` mode, one thread so the merge is not
divided between tasks, callgrind):

| | 600 additions |
|---|---|
| before | 5,341,792,947 instructions |
| after | 5,191,399,029 instructions (−2.8 %) |

The `finish_grow` frame itself falls from 909,994 instructions to 39,418,
and what is left of it is on the snapshot reader's side and the search's
bins, not in a merge. A batch of 600 references in one `add_many` is
unchanged (751.6 M against 751.6 M): it merges once, and one merge was
not where the copies showed.

## Adding a batch

`Catalog::add` builds a one-reference segment and merges it from the
tail, so 600 additions of 3,000 rows each pay for 600 sorts, 600
directory builds and the merges between them. `Catalog::add_many`
collects the batch, sorts every row into one segment, pushes it as one
part and merges once; the ids it hands out are the ones repeated `add`
calls would have. On the synthetic catalog of 60 M postings
(`examples/profile.rs`, `simd` build, callgrind, `adds` and `batch` modes
with `n = 600` and 3,000 rows per reference):

| | instructions | segments left |
|---|---|---|
| 600 calls of `add` | 4,945,900,574 | 6 |
| the same 600 references as one `add_many` | 498,253,754 | 2 |

The batch costs a tenth of the instructions: the per-addition directory
builds and the tail merges between them are gone, and the batch segment
is large enough that the one merge it needs takes the whole of it in a
single piece. It also leaves fewer segments, so the searches that follow
look fewer places for their hashes.

## A sparser directory for small segments

A search looks every query hash up in every segment, and in a delta
segment most of those lookups miss. A miss walks the directory to an
empty slot: five slots on average at the load of three keys per five
slots a segment used to have, and most of what a lookup in a small
segment costs, on a branch the CPU cannot predict. Segments of fewer
than 2<sup>20</sup> keys now get four slots per key, which ends a miss
in under two probes, for 48 bytes of directory per key instead of 20 (at
most about 50 MB for a segment that small).

Instruction counts from `examples/profile.rs` under callgrind, `simd`
build, on the synthetic catalogs of the sections above; a count covers
120 searches (one warm-up round and two measured rounds of 40). The
cache columns are from `--cache-sim=yes`, on the laptop: D1 counts the
data reads that missed the first-level cache, LLd those that missed the
last.

| catalog, segments | instructions | D1 read misses | LLd read misses |
|---|---|---|---|
| s2 plus 300 adds, 6 segments, before | 127,402,436 | 1,600,402 | 556,449 |
| s2 plus 300 adds, 6 segments, after | 114,100,618 | 1,473,790 | 581,578 |
| s1, 1 segment of 450 k keys, before | 177,096,755 | 2,567,145 | 643,924 |
| s1, 1 segment of 450 k keys, after | 177,001,013 | 2,565,283 | 650,828 |

The 6-segment search is 10 % cheaper in instructions. Its last-level
read misses rise a little, by the extra directory bytes a lookup now
walks, and the first-level read misses fall by 8 %, which is the point:
fewer probes per miss. The 450 k-key segment is neutral, as intended.

The three single-segment sizes the audit named, measured the same way
before and after, are neutral too (a catalog of one small segment gains
little, since its hashes are there to be found, and pays for a larger
directory):

| catalog (keys) | instructions before | instructions after | D1 read misses before / after | LLd read misses before / after |
|---|---|---|---|---|
| 40 k keys, 600 k postings | 548,494,125 | 548,494,266 | 1,693,379 / 1,694,693 | 15,370 / 15,370 |
| 400 k keys, 6 M postings | 76,651,202 | 76,614,899 | 640,928 / 639,572 | 124,103 / 138,555 |
| 900 k keys, 15 M postings | 73,334,770 | 72,913,994 | 660,114 / 653,155 | 190,732 / 191,451 |

The 400 k-key catalog's directory grows past the laptop's 8 MiB last
level cache, which is the 12 % more last-level read misses; it costs
nothing measurable in instructions. On the laptop, where wall clock is
noisy (see the note at the top), the three single-segment sizes were not
slower by more than that noise in any round, so the threshold stayed at
2<sup>20</sup>.

## Parallel merge

With `std`, a merge of a million postings or more runs on every core.
The hash space is cut into pieces holding about equal numbers of
postings, eight per thread. One pass deals every source's directory to
the pieces, a stretch of slots per task. Each piece then merges its runs
into arrays of its own, and the pieces are copied into the output in
order as they finish, sharing the bytes where their packed bits meet; a
large directory is filled a stretch of regions per thread. The segment
is the same one a single thread builds.

`examples/lifecycle.rs` with huge pages on, in alternating rounds of the
code before and after on one host; build, write, read and map times did
not change beyond noise:

| host | catalog | `compact` | `compact_deltas` | worst `add` | peak memory |
|---|---|---|---|---|---|
| `m7i-flex.large`, 2 vCPUs | 60 M | 0.64 → 0.58 s | 0.27 → 0.23 s | 234 → 209 ms | +2 % |
| `m7i.2xlarge`, 8 vCPUs | 60 M | 0.67 → 0.22 s | 0.27 → 0.09 s | 246 → 88 ms | +12 % |
| `m7i.2xlarge`, 8 vCPUs | 180 M | 1.57 → 0.56 s | 0.35 → 0.12 s | 302 → 114 ms | +9 % |
| `c7i.4xlarge`, 16 vCPUs | 60 M | 0.62 → 0.15 s | 0.25 → 0.06 s | 223 → 62 ms | +15 % |
| `c7i.4xlarge`, 16 vCPUs | 180 M | 1.50 → 0.39 s | 0.33 → 0.09 s | 281 → 83 ms | +12 % |

The 60 M catalog is the one of the section above, five rounds; the
180 M one has three times the rows
(`-- 60000 3000 10800000 2000`), three rounds. The average addition fell
by about a fifth on 8 and 16 vCPUs, as the merges it waits on got shorter.
Two hardware threads of one core gain a tenth, since they share the
core's ports. On the real catalog, on that host, loads and searches
measured the same, and the candidates were identical.

**Memory.** Peak memory is that of the whole run: build, additions,
compactions and loads. A merge holds its output at full size from the
start. A piece's arrays are freed as soon as it is copied into place,
so besides the output only the pieces merged and not yet placed are
alive, about an eighth of the output with eight pieces per thread.
Copying every piece in at the end instead held a second copy of the
output, and the peak was 30 % higher; glibc keeps freed memory in its
per-thread arenas rather than returning it to the system, so the
pieces have to go as they are placed.

## Trusted loads
`read_from_trusted` and `map_from_trusted` skip the checksum and the
order check of every run, and read the references alone for each run's
multiplicity. On the AWS `m7i-flex.large`, huge pages on, `simd` build,
each load in a fresh process with the file in the page cache, five
alternating rounds on the real catalog and three on the synthetic one
of 65 M postings:

| | real, full | real, trusted | real, no pass | synthetic, full | synthetic, trusted |
|---|---|---|---|---|---|
| `read_from` | 1.65 s | 0.96 s | 0.71 s | 0.35 s | 0.24 s |
| `map_from` | 0.94 s | 0.39 s | 0.13 s | 0.26 s | 0.17 s |

Full loads measured the same as before the change, and searches after
each of the four loads of the real catalog returned identical
candidates. The "no pass" column is a variant that skips the pass over
the references as well (wrong multiplicities, timing only): it is what a
trusted load would cost if the snapshot stored each run's multiplicity,
a change of format. That pass is a quarter of a second for 499 M
references on two hardware threads. What remains of a trusted map is
faulting in the file's pages, copying the keys and offsets, and the
directory. The synthetic catalog's runs are 17 postings long, so its
keys and directory are a larger share of its load and it gains less.

Format 3 is that change: it stores the multiplicities and the reference
range, and a trusted load reads no posting. The counts are below, in
[Format 3: what a trusted load recomputes](#format-3-what-a-trusted-load-recomputes).

### Format 3: what a trusted load recomputes

Format 3 stores, per segment, each run's multiplicity and the segment's
reference range, the two things a trusted load used to recompute from the
references. Writing them costs one walk of the directory, which the writer
already made for the offsets; the file grows by a byte per distinct hash
and 8 bytes per segment. A trusted load now reads no posting at all.

Instruction counts from `examples/profile.rs` under callgrind, `simd`
build, one CPU (`taskset -c 0`, so the parallel load and the checksum on a
core of their own land on the counted thread), on the synthetic catalog
of 60 M postings (`build 20000 3000 3600000`):

| load | before | after |
|---|---|---|
| `map_from_trusted` | 1,414,460,075 | 613,834,915 |
| `read_from_trusted` | 1,583,889,579 | 822,315,819 |
| `map_from` | 2,577,832,442 | 2,590,960,794 |
| `read_from` | 2,718,786,263 | 2,735,514,418 |

A trusted map costs 57 % fewer instructions and a trusted read 48 %
fewer: what is left is copying the keys, the offsets and the
multiplicities, and building the directory, which is now most of a
trusted map. A full load costs 0.5 to 0.6 % more, for the multiplicities
it reads and the comparison of the stored values with what its walk
finds; that is the price of the format checking what it stores.

## Allocators and huge pages

The `simd` build on the real catalog under five allocator setups, on one
AWS `m7i-flex.large`, five alternating rounds (four for the synthetic
rows); candidates were identical in every run. Peak memory is the
largest resident set (GNU `time`), and the search rows are for a catalog
loaded with `read_from`:

| | glibc | glibc, `hugetlb=1` | mimalloc v3 | mimalloc v3, THP off | mimalloc v2 |
|---|---|---|---|---|---|
| search, best of three summed over the clips | 127.2 ms | 118.6 ms | 117.1 ms | 126.8 ms | 117.1 ms |
| search, one thread, p50 / p95 | 0.65 / 4.54 ms | 0.60 / 4.32 ms | 0.60 / 4.29 ms | 0.66 / 4.54 ms | 0.60 / 4.23 ms |
| searches per second, two threads | 779 | 795 | 813 | 774 | 797 |
| `read_from` | 2.0 s | 1.5 s | 1.9 s | 2.9 s | 2.0 s |
| peak memory, `read_from` | 2.12 GB | 2.12 GB | 2.56 GB | 3.05 GB | 2.57 GB |
| peak memory, `examples/lifecycle.rs` | 1.19 GB | 1.19 GB | 1.88 GB | 1.86 GB | 1.47 GB |
| synthetic, 17 postings per hash, searches per second, one thread | 9,618 | 10,666 | 10,731 | 9,742 | 11,264 |
| synthetic, 136 postings per hash, searches per second, one thread | 3,212 | 3,552 | 3,548 | 3,201 | 3,512 |

`glibc, hugetlb=1` is `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`; mimalloc is
crate 0.1.52 as the programs' global allocator, and "THP off" is
`MIMALLOC_ALLOW_THP=0`. The host had transparent huge pages set to
`madvise` (Ubuntu's default), so plain glibc got none: with the catalog
loaded, mimalloc v3 had 2.09 of its 2.11 GB on 2 MiB pages and glibc with
the tunable 1.5 GB.

mimalloc gains what huge pages gain and nothing more: v3 asks for them
itself, and with them turned off it matched plain glibc. It loads more
slowly and peaks higher because a load grows each array by doubling, and
glibc's `realloc` moves a large block by remapping its pages where
mimalloc's allocates a new block and copies. Mapped loads (`map_from`)
were 0.9 s under every setup, since their large arrays never pass
through the allocator; their searches gained 4–5 % from huge pages. The
tunable is the recommendation in
[Running in production](USAGE.md#17-running-in-production).

## Reproducing

```bash
# Synthetic catalog: refs, postings per ref, distinct hashes, queries, threads.
cargo run --release --example bench -- 20000 3000 3600000 200 4
# Everything around a search: build, additions, compactions, snapshots.
cargo run --release --features mmap --example lifecycle -- 20000 3000 3600000 2000
# A real catalog dumped as raw arrays and its clips (see the example's docs).
cargo run --release --features mmap,simd --example replay -- <snapshot_dir> <fp_dir> <out_dir> [--mmap]
```

Add `--features simd` to compare the vector build. Measure on a quiet
host, alternate the builds round by round, and set
`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` for every build: without it, page
placement alone moved the real clips by 5–7 %.
