# Performance

Every number behind the README, how each was measured, how the passes are
arranged, and what was tried and left out. Two machines appear:

- an Intel i5-1135G7 laptop (4 cores, 15 GB), on an otherwise idle
  machine, where cycle counts are available;
- an AWS `m7i-flex.large` (two hardware threads of one Sapphire Rapids
  core, 8 GiB), the size of a production task, whose hypervisor exposes
  no hardware counters.

Builds are for the generic x86-64 target unless a row says otherwise. Where
two builds are compared, they ran in alternating rounds on the same host,
and their candidates were identical. Hosts of the AWS type differ by about
10 %, so compare columns within one table only.

## Synthetic catalog

`examples/bench.rs` builds a synthetic catalog, plants a window of one
reference in each query with half as many random rows, and checks that every
query finds its reference at the planted offset. On an Intel i5-1135G7
laptop (4 cores, 15 GB), 20,000 references × 3,000 postings over 3.6 M
distinct hashes, on an otherwise idle machine:

| | |
|---|---|
| postings | 60 M, 15-bit references and 18-bit times |
| memory | 0.31 GiB with the directory (0.58 GiB with 32-bit arrays) |
| build | 2.4–2.5 s (radix sort, buckets sorted on all cores) |
| query | 799 rows at the median |
| search, one thread | p50 0.10–0.12 ms, p95 0.11–0.14 ms over four runs |
| searches per second, 4 threads | 33,000–34,000 from a cool start |

Those numbers are for the generic x86-64 target; `-C target-cpu=native`
gained a further 5 % on this machine. The four-thread figure is the first
five seconds from a cool package: past 75 °C this laptop throttles to about
20,000 searches per second, whatever the build.

A search here touches about 800 runs, each a few cache lines in a 0.5 GiB
array, so it is bound by memory latency and parallelism rather than by
arithmetic. The passes are arranged for that. A hash resolves in one
directory read whose address depends only on the hash. Each pass loads its
next batch of lines (32 directory slots, every line of 16 runs, 32 vote
groups; half that costs 13 % of search time, double gains nothing more)
before working on the current batch. The totals pass decodes each run as a
bit stream, answers the fanout test from a ring of the last 64 references
instead of a second read, and uses plain adds whenever no counter can
overflow; a run where no reference has as many postings as the fanout (every
run but 2 % of postings on the real catalog) skips the fanout test in both
passes, which took a third off the instructions per search on the real
clips. The survivor scan reduces each block of counters with a bitwise OR,
and the histogram pass finds a survivor's group in a run by binary search
rather than a walk.

On the same machine the version without those arrangements took 0.30 ms at
the median and 6,700 searches per second on four threads: user-mode cycles
per search fell by three times, instructions by a third, last-level cache
misses by a third. Packing the postings afterwards cut memory by 41 % for
3–8 % more cycles per search. What was tried and rejected is under [Measured
and left out](#measured-and-left-out).

## Building, changing and loading

`examples/lifecycle.rs` times everything around a search on the synthetic
catalog above: the bulk build, 2,000 additions of 3,000 rows each (a tenth
of them removed again), both compactions, snapshots written, read and
mapped, and query grouping. On the AWS `m7i-flex.large`:

| | |
|---|---|
| bulk build, 60 M postings (rows generated on the clock) | 2.65 s |
| `Catalog::add` of 3,000 rows into 60 M postings | 1.14 ms on average, 249 ms at worst |
| `compact_deltas`, four deltas of 5.4 M postings | 0.27 s |
| `compact`, 65 M postings | 0.69 s |
| `write_to` memory, 0.29 GiB | 0.31 s |
| `read_from` memory | 0.46 s |
| `map_from` a file in the page cache | 0.32 s |
| `Query::from_pairs`, 798 rows | 10.1 µs |
| search, one thread | 0.129 ms |

The build sorts in place with a radix sort whose swaps do not wait for one
another, so the cache misses of 2,048 bucket heads overlap, and it takes a
narrower digit for a small bucket: `Segment::from_postings` on 60 M sparse
32-bit hashes with a hundred rows each fell from 4.4 s to 2.1 s, and from
1.8 s to 1.2 s when they arrive sorted (the laptop build time above is for
the earlier sort). The worst addition is the one whose merge cascades
furthest, and it is one pass over the segments it reaches. A compaction
copies runs as packed bits. A load is bound by the structural check (every
posting against the one before it), then by the checksum and the directory.
Huge pages for the allocator took 4.5 % off the additions, a deployment
choice rather than the crate's (see [Allocators and huge
pages](#allocators-and-huge-pages)).

The same operations on one AWS `m7i-flex.large` (two hardware threads of one
Sapphire Rapids core, 8 GiB), `simd` and `mmap` features, huge pages on,
five rounds, the median of each:

| | |
|---|---|
| bulk build, 60 M postings | 2.45 s |
| 2,000 additions of 3,000 rows, total | 2.30 s |
| worst addition | 204.0 ms |
| heap after the additions, 5 segments | 0.444 GiB |
| `compact_deltas` | 0.22 s |
| `compact` | 0.57 s |
| snapshot size, 65.4 M postings | 0.295 GiB |
| `write_to` memory | 0.22 s |
| `read_from` | 0.33 s |
| `read_from_trusted` | 0.17 s |
| `map_from` | 0.25 s |
| `map_from_trusted` | 0.09 s |
| search, floor 16, one thread | 0.095 ms |

The trusted loads read no posting at all, because the snapshot stores each
run's multiplicity ([Trusted loads](#trusted-loads)), which `write_to`
writes too for a snapshot 1.4 % larger. The additions and the heap after
them reflect the [sparser
directory](#a-sparser-directory-for-small-segments).

## Real data

`examples/replay.rs` runs the crate over the catalog of the *hashtally*
design notes (7 November 2026): 499.4 M postings from 120,329 production
references, 3.66 M distinct hashes (136 postings per hash on average), and
the 104 real clips the notes compared with the production ClickHouse query.
On the laptop:

| | this crate | prototype in the notes |
|---|---|---|
| memory | 1.94 GiB (17-bit references, 15-bit times; 1.70 GiB with [references coded as gaps](#references-coded-as-gaps)) | 3.0 GiB packed, 3.8 GiB raw |
| build from rows in memory | 8.3 s | 333 s including the export |
| search, one thread | p50 1.10 ms, p95 8.8 ms, max 14.2 ms | p50 2.9 ms |
| searches per second, 4 threads | 1,456 | 890 |
| candidates | identical to the prototype on all 104 clips | 103 of 104 match ClickHouse |

The one clip where both differ from ClickHouse is the one where ClickHouse's
`LIMIT 64 BY` keeps an arbitrary 64 of 74 postings of a hash and the exact
engines keep the first 64 by time.

On a production-sized host, an AWS `m7i-flex.large` (two hardware threads of
one Sapphire Rapids core, 8 GiB) loading the catalog from its snapshot file,
generic x86-64 build:

| | |
|---|---|
| load from file | 3.8 s |
| search, one thread | p50 0.89 ms, p95 8.6 ms, max 14.2 ms |
| searches per second, two threads | 537–546 |

Candidates were identical between the laptop run, this host and the
prototype on all 104 clips, and the same 71 candidates came back in all four
load modes (`read_from`, `map_from` and their trusted forms), with each
clip's best time summing over the 104 clips to 121.2 ms after `read_from`
and 126.5 ms after `map_from`. The design notes estimated 50 to 450 searches
per second for a two-vCPU task.

## Search passes

Each change was measured against the tree without it, with identical
candidates.

| change | design and measured effect |
|---|---|
| Every line of a run | The totals pass loads every line of the next 16 runs ahead of time (and the line past a run shorter than two lines), not just each run's first and last entry: a fifth off the p95 and a ninth off the summed latency on the real clips |
| Vector instructions (`simd`) | On x86-64 the totals pass, the search for a survivor's group in a short run and the load check of every run decode eight values at a time with AVX2 (widths up to 25 bits) when the CPU has it; table below |
| Parallel loads | A large segment's runs are checked on every core and `map_from` checksums the packed arrays on a core of its own: little on two hardware threads of one core, more cores were not measured |
| Scanning only the counters a search wrote | A search records the blocks its totals pass writes (about four instructions per posting) and the survivor scan reads only those, when the postings are fewer than the blocks. A 30-row query on 2,000,000 references: 276,462,986 to 21,270,841 instructions per 120 searches (13 times fewer), no longer growing with the catalog from 65,536 references up; an 800-row query −5 % at 1,048,576 references and −29 % at 2,000,000, within 0.2 % of before below that |
| Whole blocks, run weights, overflow bound | A block of 64 counters is taken as an array, not a slice (a loop and a `memset`, about 230 instructions a block); a hash's vote weight is stored with its run instead of looked up for every run in every segment; the overflow bound counts at most `min(multiplicity, fanout)` postings per run. Instructions per 120 searches −4.5 % and −5.1 % on single segments, −11.1 % with six, −19.3 % (800 rows) and −30.3 % (30 rows) on 2,000,000 references |
| One search on several threads | `SearchOptions::threads` gives each thread a share of the runs with about equal postings, and counters of its own; the calling thread adds the marked blocks and runs the rest as on one thread. It applies from 2,097,152 postings on (from 262,144, searches of 0.5 to 2 ms got 20 to 46 % slower). On the real catalog on the laptop, clips of 5 ms and more (12 or 13 of 104) got 21, 35 and 37 % faster with 2, 4 and 8 threads, those of 2 to 5 ms 1, 4 and 3 %, shorter ones none; all 104 clips summed −10 %, −18 % and −18 % |

The first three together, on an AWS `m7i-flex.large`, generic x86-64 build,
the real catalog and its 104 clips, three alternating rounds:

| | before | default build | `simd` |
|---|---|---|---|
| search, one thread, best of three summed over the clips | 176.8 ms | 157.2 ms | 136.9 ms |
| searches per second, two threads | 620 | 629 | 737 |
| `read_from` / `map_from` | 2.8 / 1.8 s | 2.7 / 1.6 s | 2.1 / 1.0 s |
| synthetic, 136 / 17 postings per hash, searches per second | 2,067 / 7,634 | 2,596 / 7,521 | 3,008 / 8,432 |

## A sparser directory for small segments

In a delta segment most directory lookups miss, walking to an empty slot
(five on average at the old load of three keys per five slots). Segments of
fewer than 2<sup>20</sup> keys now get four slots per key, which ends a miss
in under two probes, for 48 bytes of directory per key instead of 20.
Searches of 800 rows on the synthetic catalog of 60 M postings with delta
segments from `add` calls of 3,000 rows (AWS `m7i-flex.large`, `simd` build,
five alternating rounds of each build):

| catalog | before | after |
|---|---|---|
| 1 segment, no additions | 71.4 µs | 68.5 µs |
| 2 segments, 1 addition | 85.9 µs | 75.4 µs |
| 6 segments, 300 additions | 169.7 µs | 117.4 µs |

That is 4 % faster on one segment, 12 % on two and 31 % on six: a search
pays about 20 µs per extra delta segment before and about 10 µs after. None
of the single segments of 40 k, 400 k and 900 k keys is slower (404.4 to
404.6 µs, 77.1 to 76.8 µs, 69.9 to 65.4 µs), so the threshold stayed at
2<sup>20</sup>. Additions pay: 0.92 ms per reference instead of 0.87 ms, and
the heap after them is 0.027 GiB higher (0.417 to 0.444 GiB).

## Adding a batch

`Catalog::add` builds a one-reference segment and merges it from the tail,
so 600 additions of 3,000 rows each pay for 600 sorts, 600 directory builds
and the merges between them. `Catalog::add_many` sorts the batch into one
segment and merges once, with the ids repeated `add` calls would have given.
On the synthetic catalog of 60 M postings (`simd` build; wall clock in five
alternating rounds of `add` and three of the batch):

| | instructions | per reference | segments left |
|---|---|---|---|
| 600 calls of `add` | 4,945,900,574 | 0.92 ms | 6 |
| the same 600 references as one `add_many` | 498,253,754 | 0.14 ms | 2 |

Separately, a merge's writer reserves the sum of its sources' key counts (an
upper bound) up front instead of growing its arrays by doubling (about 3 %
of an addition's instructions went under `realloc`): 600 additions fell from
5,341,792,947 to 5,191,399,029 instructions (−2.8 %), and an addition from
0.90 to 0.87 ms per reference.

## Parallel merge

With `std`, a merge of a million postings or more runs on every core. The
hash space is cut into pieces holding about equal numbers of postings, eight
per thread; each piece merges its runs into arrays of its own, and the
pieces are copied into the output in order as they finish. The segment is
the same one a single thread builds. `examples/lifecycle.rs` with huge pages
on, alternating rounds of the code before and after on one host (five
rounds, three on the 180 M catalog, which has three times the rows); build,
write, read and map times did not change beyond noise:

| host | catalog | `compact` | `compact_deltas` | worst `add` | peak memory |
|---|---|---|---|---|---|
| `m7i-flex.large`, 2 vCPUs | 60 M | 0.64 → 0.58 s | 0.27 → 0.23 s | 234 → 209 ms | +2 % |
| `m7i.2xlarge`, 8 vCPUs | 60 M | 0.67 → 0.22 s | 0.27 → 0.09 s | 246 → 88 ms | +12 % |
| `m7i.2xlarge`, 8 vCPUs | 180 M | 1.57 → 0.56 s | 0.35 → 0.12 s | 302 → 114 ms | +9 % |
| `c7i.4xlarge`, 16 vCPUs | 60 M | 0.62 → 0.15 s | 0.25 → 0.06 s | 223 → 62 ms | +15 % |
| `c7i.4xlarge`, 16 vCPUs | 180 M | 1.50 → 0.39 s | 0.33 → 0.09 s | 281 → 83 ms | +12 % |

The average addition fell by about a fifth on 8 and 16 vCPUs; two hardware
threads of one core gain a tenth, since they share the core's ports. A merge
holds its output at full size, and each piece's arrays are freed as soon as
it is copied into place, so only the pieces merged and not yet placed (about
an eighth of the output) are alive beside it; copying every piece in at the
end held a second copy and the peak was 30 % higher, since glibc keeps freed
memory in per-thread arenas.

A build of a million rows or more is packed the same way:
`Segment::from_postings` had sorted on every core but removed duplicates,
found the widths and packed every row on one thread (0.75 G of 1.59 G
instructions for 6 M postings were the sort). It now does all of it per
piece on every core, giving the same segment byte for byte for about 2 %
more peak memory; wall-clock numbers are not measured yet.

## Merging straight into a file

A merge can write its result to a segment snapshot without holding it in
memory: it is cut into pieces as the in-memory merge is, and each piece is
appended to the file in piece order as soon as the pieces before it are. The
snapshot puts the packed arrays first and gives each section a checksum of
its own for exactly this; a layout without that counted every piece in a
first pass and read the finished file back to sum it. Instruction counts
under callgrind (`simd` build, one CPU), synthetic catalog of 60 M postings
with 20 additions on top, compacting four segments of 60.06 M postings into
a file of 272,025,088 bytes:

| | instructions |
|---|---|
| `Catalog::compact`, the result on the heap | 3,947,659,752 |
| `CatalogStore::compact`, an earlier format: count, write, read back | 9,408,883,295 |
| `CatalogStore::compact`, one pass | 5,335,772,190 |

The file is byte for byte the segment the first builds. The store's route
costs the copy of every piece into the file (0.44 G instructions) and the
map of the result; in exchange it never holds a result larger than a piece
(by default 2<sup>22</sup> postings of input) plus the 13 bytes per distinct
hash its keys, run starts and multiplicities take. A plain catalog's merges
use the same code: 600 `Catalog::add` calls take 5,410,907,419 instructions
against 5,330,486,224 before (1.5 % more), and 120 searches take 81,028,707,
as before. Counts here move by about 6 % between runs unless the threads are
pinned to one CPU (`taskset -c 0`).

### A catalog in a directory

A `CatalogStore` owns a directory of segment files and a manifest, and sends
a merge there instead of to the heap when its inputs reach the spill
threshold. The heap then holds the unsaved segments below the threshold, the
keys and directory of every segment (24 bytes per distinct hash at the usual
load) and the pieces in flight; a segment with a file keeps its postings in
the file's pages, which the kernel can drop and read again. A piece holds
whole runs, so a hash with many postings is merged inside one piece, which
bounds a merge's memory from below.

## Trusted loads

`read_from_trusted` and `map_from_trusted` skip the checksum and the order
check of every run and, because the snapshot stores each run's multiplicity,
read no posting at all. On the AWS `m7i-flex.large`, huge pages on, `simd`
build, the file in the page cache, five alternating rounds, on the real
catalog (2.04 GiB) and the synthetic one (0.29 GiB):

| | full | trusted | trusted, before multiplicities were stored |
|---|---|---|---|
| real, `read_from` | 1.62 s | 0.71 s | 0.93 s |
| real, `map_from` | 0.91 s | 0.14 s | 0.37 s |
| synthetic, `read_from` | 0.33 s | 0.17 s | 0.23 s |
| synthetic, `map_from` | 0.25 s | 0.09 s | 0.16 s |

Full loads are unchanged, and searches after each load returned identical
candidates. What is left of a trusted map is faulting in the file's pages,
copying the keys, offsets and multiplicities, and building the directory:
the 0.14 s is close to the 0.13 s measured for a variant that also skipped
the references (wrong multiplicities, timing only), a pass that had cost a
quarter of a second for 499 M references.

### What a trusted load recomputes

The snapshot stores, per segment, each run's multiplicity and the segment's
reference range, which a trusted load would otherwise recompute from the
references; writing them costs one walk of the directory the writer already
made for the offsets, and the file grows by a byte per distinct hash and 8
bytes per segment. Under callgrind (`simd` build, one CPU, synthetic catalog
of 60 M postings) a trusted map fell from 1,414,460,075 to 613,834,915
instructions (−57 %) and a trusted read from 1,583,889,579 to 822,315,819
(−48 %); a full load costs 0.5 to 0.6 % more (`map_from` 2,577,832,442 to
2,590,960,794, `read_from` 2,718,786,263 to 2,735,514,418), the price of the
format checking what it stores.

## Warming the references of a mapped catalog

A catalog directory opened with `CatalogStore::open_trusted` reads no
posting, so the first searches fault the pages they touch in from the disk
one run at a time. A search reads the references of every run it matches and
the times of its survivors only, so the references (1.06 GB of the real
catalog's 2.0 GB segment file) are what has to be in memory:
`Catalog::warm_references` reads one byte of every page of them, in order,
and the kernel's read-ahead streams the file behind it. On the laptop's NVMe
drive, page cache emptied before each run, four alternating rounds, the 104
clips searched twice after the open (candidates identical):

| after `open_trusted` | first pass | worst clip | second pass | file in memory after |
|---|---|---|---|---|
| nothing | 4.42–4.63 s | 1.06–1.10 s | 0.14–0.23 s | 1.03 GB |
| `warm_references`, 1.59–1.83 s | 0.24–0.26 s | 16–20 ms | 0.15 s | 1.06 GB |

## Measured and left out

One line each, with the number that decided it. MaxScore-style pruning has
its own section below.

| tried | result |
|---|---|
| Transparent huge pages for the crate's arrays | TLB misses down by a third but cycles by 2 %: page walks overlap like the other misses |
| A cache-sized bucket table over sorted `(key, offset)` pairs, not the hash directory | 10 % less memory, 11–15 % more search time: the posting stream evicts the table, so every lookup misses twice |
| A denser directory (three keys per four slots instead of five) | 3 % less memory, a third more per lookup of a hash the segment does not hold, which real queries are full of |
| A bucketed histogram for the totals pass (scatter postings by the high bits of the reference, count each bucket in L1) | 62 % more instructions per search on the real clips, which are bound by instruction throughput (three per cycle), not by the counter array |
| Staging counting postings in a buffer per 8,192 references, to keep the counters in L1 with right answers | 2.3 times slower: a run's references ascend, so consecutive postings wait on one buffer's length. Folding every reference into 4,096 counters (wrong answers, timing only) took a third off the totals pass with `simd`, so the counter array is that much of it |
| Loading runs 32 or 64 ahead instead of 16 | Within noise |
| Loading the line past a run whatever its length | 8 % faster on runs of 17 postings, 2.5 % slower on runs of 136; hence the two-line limit |
| Reading up to 64 entries of a run in blocks to find a survivor, or halving a longer run to 32 first | A sixth to a half more of that pass than halving alone; only ranges of 24 entries or fewer are read in blocks |
| Masking the counter index (`totals[r & (n - 1)]`) to drop the totals pass's bounds check | The check is 4 instructions per posting and a quarter of a search, but LLVM keeps it in a standalone build of the loop and removing it takes `unsafe` |
| An interpolation guess for a survivor's group in a run | 3 % at most by estimate (the binary search is 9 % of a search on the real-like catalog); not built |
| Storing the directory in the snapshot instead of rebuilding it | Would save 0.05 s of a load |
| Checking a run's references and times as two streaming passes | A load a quarter slower, on runs of 17 postings and of 136 |
| Cutting the reference ids between threads instead of the runs | Twice as slow at the median: finding where each run crosses a range is a binary search per run and thread whose probes wait on one another |
| `madvise` with `MADV_WILLNEED` over the references, not `warm_references` | On this kernel it read the first 128 KiB of the range and no more; the first pass took as long as with nothing |
| AVX-512 decoding (sixteen values per block, one VBMI `vpermb` and one variable shift, chosen at run time ahead of AVX2) | Slower than AVX2 on an AWS `m7i-flex.large` (Xeon 8488C), five alternating rounds, huge pages on: summed clip latency +2 % loaded and +3 % mapped, `read_from` +6 %, `map_from` +9 %, synthetic searches per second −2 % on runs of 17 and −11 % on runs of 136; most likely a lower clock and a vector port given up under 512-bit instructions (no counters on this host to confirm); needs Rust 1.89 |

## Skipping runs that cannot reach the floor (left out)

MaxScore-style pruning was simulated on the real catalog before building it:
take a query's runs in order of their weight, and once the runs left could
not lift a reference with no votes yet to the floor, stop counting whole
runs and only look up, in each remaining run, the references that already
have votes. On the 102 clips with production options the counted postings
were 1.44 % fewer and the lookups that replaced them cost another 0.24 %,
for a net 1.2 % (candidates identical): the floor is a few dozen votes
against thousands of runs per query, so almost every run has to be counted
before the remaining ones can no longer matter. Not worth the code.

## Allocators and huge pages

The `simd` build on the real catalog under five allocator setups, on one AWS
`m7i-flex.large`, five alternating rounds (four for the synthetic rows),
candidates identical in every run. Peak memory is the largest resident set
(GNU `time`); the search rows are for a catalog loaded with `read_from`:

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
`MIMALLOC_ALLOW_THP=0`. The host had transparent huge pages set to `madvise`
(Ubuntu's default), so plain glibc got none: with the catalog loaded,
mimalloc v3 had 2.09 of its 2.11 GB on 2 MiB pages, glibc with the tunable
1.5 GB.

mimalloc gains what huge pages gain and nothing more: v3 asks for them
itself, and with them turned off it matched plain glibc. It loads more
slowly and peaks higher because a load grows each array by doubling, and
glibc's `realloc` moves a large block by remapping its pages where
mimalloc's allocates a new block and copies. Mapped loads (`map_from`) were
0.9 s under every setup, since their large arrays never pass through the
allocator; their searches gained 4–5 % from huge pages. On the builds of
[Search passes](#search-passes) the tunable took a further 7–13 % off search
and 14–19 % off `read_from`, so compare builds with it set. It is the
recommendation in [Running in
production](USAGE.md#17-running-in-production).

## References coded as gaps

Inside a run the references ascend, so `RefCoding::Gaps` stores them as the
gaps between neighbours instead of packing each at the width of the largest
id, which stays the default (`RefCoding::Packed`). A run is cut into blocks
of 65 references; each block holds its first reference whole (at a width
shared by the run), a five-bit width, and 64 gaps at the width of its widest
(`src/refs.rs`). A segment of gaps has 16-byte directory slots instead of 12
to hold where a run's coded references start, and its file stores that
offset per key; a packed segment needs neither.

On the real catalog with a laptop (i5-1135G7), against packed references at
17 bits; the two builds alternated, five rounds, best time per clip, and
candidates were byte-identical on all 104 clips:

| | packed | gaps |
|---|---|---|
| reference bits per posting | 17 | 12.50 |
| file | 2.05 GB | 1.79 GB (−12.5 %) |
| heap, `read_from` / mapped | 1.94 / 0.08 GiB | 1.70 / 0.10 GiB |
| first search of each clip, summed | 1 | 1.12 (median clip 1.09) |
| all 104 clips in turn, trusted map | 127.6 ms | 136.7 ms (+7 %) |
| a clip searched again at once, best of three | 1 | 1.48 |
| searches per second, eight threads | 1,426–1,466 | 1,287–1,312 |
| instructions, all clips twice (callgrind) | 1.61 G | 2.21 G |
| `read_from` / checked `map_from` | 0.99 / 0.75 s | 1.1 / 0.6–0.9 s |

A run takes a quarter fewer bits, so a search loads fewer lines, but
decoding costs a running sum per reference and the fields between blocks.
Where a run is not in the cache yet the two nearly cancel (+7–12 %); where
it is, the decoding shows (+48 %), and eight threads sharing the memory
system answer 10 % fewer searches.

The layout is the third tried. One stream of gaps per run, in 32-gap blocks
carrying only a width (12.36 bits), made the histogram pass decode a whole
run to find a few survivors (2.73 G instructions over all clips, 1.70 times
the packed ones). A whole reference at the head of every block lets it
decode only the blocks whose span holds a survivor, for 0.07 bits a
reference (2.37 G with blocks of 33); blocks of 65 halve the fields read
between blocks, for another 0.07 bits (2.21 G).

The load check decodes the gaps eight at a time: a sum that passes the
largest id wraps below the reference before it, so one compare per reference
finds it (a reference at a time had taken `read_from` from 0.99 s to 1.74
s). Elias–Fano coding would take 11.6 bits a reference on the same runs
(sized, not built) but has no eight-at-a-time decode with these kernels.

### Packed stays the default

Since the costs above fall on every search, the coding is a choice per
catalog rather than the format: each segment records its own coding, search
and merge take both, and a packed segment is what it was before gaps existed
(the same file size within 8 bytes, the same 1.94 GiB of heap on the real
catalog, byte-identical candidates on all 104 clips). A search costs within
about 1 % of the instructions it did then, the lookups compiled once for
each kind of directory slot (`profile search … 750 0 0 2`, callgrind):

| synthetic catalog | packed, before the choice | packed | gaps |
|---|---|---|---|
| 120,000 references × 500 over 450,000 hashes | 169.2 M | 170.6 M | 298.0 M |
| 20,000 references × 3,000 over 3.6 M hashes | 72.2 M | 73.0 M | 111.0 M |

On the same two catalogs the files take 245.9 MB and 271.8 MB packed, 217.0
MB and 283.7 MB coded as gaps: with 17 postings per hash, as in the second,
the larger directory slots and the stored offsets outweigh the bits the gaps
save.

## Segments and a catalog that changes

A search pays for every segment: a directory lookup per query hash, and a
run for each hash the segment holds. Splitting the same postings over more
segments costs more than the extra postings: on the synthetic catalog with
3,000 references added (69 M postings, queries of 750 rows), `profile search
… 750 0 3000 2` counted:

| the 69 M postings as | instructions | µs a search |
|---|---|---|
| one segment (`compact`) | 70.9 M | 95 |
| the largest and one delta (`compact_deltas`) | 108.5 M | 147 |
| five segments, as `add` left them | 128.1 M | 192 |

A delta's short runs (2.7 postings against 19 in the largest) each cost
nearly as much as a long one. A lookup now works out a hash's home slot once
per batch, not twice, and probes in a plain loop: about 28 instructions, 1.9
% off a search of one segment and 5.3 % off one of five (128.1 M to 121.2
M).

The rest is the work a delta brings, so it is what `Catalog::compaction_due`
weighs. `examples/serve.rs` runs search threads beside a writer that adds
and removes references and compacts on a copy as a policy says, every policy
from the same catalog; on the laptop, three search threads, one reference
added and one removed per change, 15 s a policy, two rounds in opposite
orders:

| policy (`delta_share`:`deltas`) | searches per second | references added per second | compactions, deltas / full |
|---|---|---|---|
| no writes | 12,635 / 19,703 | | |
| never compact | 6,347 / 5,812 | 414 / 397 | 0 / 0 |
| 0.10:8 | 7,755 / 7,583 | 352 / 359 | 0 / 4, 0 / 4 |
| 0.05:8 (the default) | – / 7,547 | – / 331 | 0 / 4 |
| 0.05:4 | 6,842 / 7,674 | 171 / 204 | 72 / 2, 87 / 2 |
| 0.02:2 | 9,959 / – | 52 / – | 190 / 1 |

Removals as frequent as additions call for most of the full compactions here
(5 % of 20,000 ids every 1,000 removals). Capping the deltas at four rewrote
the largest delta every few changes for no faster searches, hence eight. In
batches of 100 references the default policy added 593 a second beside 7,191
searches, and 912 without compacting.

## Reproducing

```bash
# Synthetic catalog: refs, postings per ref, distinct hashes, queries, threads.
cargo run --release --example bench -- 20000 3000 3600000 200 4
# Everything around a search: build, additions, compactions, snapshots.
cargo run --release --features mmap --example lifecycle -- 20000 3000 3600000 2000
# Searches beside a writer: refs, postings per ref, hashes, search threads,
# seconds per policy, references per addition, then the policies.
cargo run --release --example serve -- 20000 3000 3600000 3 15 1 0.05:8 none
# A real catalog dumped as raw arrays and its clips (see the example's docs).
cargo run --release --features mmap,simd --example replay -- <snapshot_dir> <fp_dir> <out_dir> [--mmap]
```

Add `--features simd` to compare the vector build. Measure on a quiet host,
alternate the builds round by round, and set
`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` for every build: without it, page
placement alone moved the real clips by 5–7 %.
