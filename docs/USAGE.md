# Using afpindex

The [README](../README.md) covers installation and a first search. This
guide is the reference for everything after that: what the index stores,
what a search computes and costs, how to read and act on its results, how
a catalog changes while it serves, what it holds in memory, how snapshots
work, and how to run it in production. The API itself is documented on
[docs.rs](https://docs.rs/afpindex); measurements are in
[PERFORMANCE.md](PERFORMANCE.md).

Every Rust example here is a complete function or type, compiled as a
doctest of the crate (with the `mmap` feature; the first one also uses
`audiofp`), so it matches the API as published.

1. [Concepts](#1-concepts)
2. [Feeding the index](#2-feeding-the-index)
3. [Reference ids](#3-reference-ids)
4. [Building a catalog](#4-building-a-catalog)
5. [Queries](#5-queries)
6. [Searching](#6-searching)
7. [Search options](#7-search-options)
8. [Reading candidates](#8-reading-candidates)
9. [Your own scoring: `search_with`](#9-your-own-scoring-search_with)
10. [Changing a catalog](#10-changing-a-catalog)
11. [Threads and serving](#11-threads-and-serving)
12. [Memory](#12-memory)
13. [Snapshots](#13-snapshots)
14. [Working with segments directly](#14-working-with-segments-directly)
15. [Errors](#15-errors)
16. [Features, platforms and `no_std`](#16-features-platforms-and-no_std)
17. [Running in production](#17-running-in-production)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Concepts

A landmark fingerprinter turns audio into **rows** `(hash, time)`: a 32-bit
hash of a local feature (for Wang, a pair of spectral peaks) and the frame
where that feature starts. Two recordings of the same audio produce many of
the same hashes, and the times of those shared hashes differ by one
constant: the offset between the two recordings. Everything in this crate
follows from that.

| Term | Meaning |
|---|---|
| **reference** | One indexed recording (a track, an ad, an episode), identified by a dense `u32` id |
| **posting** | One row of a reference: `(hash, reference, time)` ([`Posting`](https://docs.rs/afpindex/latest/afpindex/struct.Posting.html)) |
| **run** | All postings of one hash in one segment, grouped by reference, times ascending |
| **segment** | An immutable, sorted, packed set of postings with a directory from hash to run |
| **catalog** | A list of segments plus the table of which ids are live |
| **query** | A clip's rows grouped by hash |
| **vote** | One query row and one posting with the same hash: a vote for `(reference, Δ)` |
| **Δ (delta)** | `reference_time − query_time` of a vote, in frames; signed (`i64`) |
| **bin** | All votes of one reference at one Δ |
| **survivor** | A reference whose total votes reach the vote floor |
| **candidate** | A survivor reduced to its best bin plus statistics over its other bins |

A true copy piles its votes into one bin, at the Δ where the clip sits in
the reference. A chance collision casts single votes at scattered offsets.
The search finds every reference with enough votes in total, builds its
histogram over Δ, and reports the best bin.

**What "exact" means.** Every vote the rules below allow is counted;
nothing is sampled, pruned by a heuristic or approximated. The rules are
the caps you choose (the query's times per hash and the fanout per
`(hash, reference)`, [section 7](#7-search-options)) and the vote floor.
Results are deterministic: the same catalog and query give the same
candidates in the same order on every build, feature set and CPU, with or
without `simd`.

**What it does not do.** It does not fingerprint audio, decide what counts
as a match (it reports evidence; [section 8](#8-reading-candidates) shows
how to decide), store names or metadata for references, or find copies
played at a different speed (their Δ drifts, so their votes spread over
many bins; see [section 9](#9-your-own-scoring-search_with) for a
tolerance).

---

## 2. Feeding the index

### What a row must be

- **`hash: u32`** - any 32-bit value. Hashes should spread: the cost of a
  search is the number of postings in the runs it reads, so a hash shared
  by a large share of the catalog makes every query that contains it
  slow. On the real catalog of 499 M postings, 3.66 M distinct hashes
  averaged 136 postings each, which is a comfortable shape; aim for runs
  of a few hundred postings at that scale.
- **`time: u32`** - in a unit where a copy's offset is constant: STFT
  frames, not wall-clock milliseconds rounded per row. Query and reference
  times must use the same unit and frame rate.

Times are relative to the start of each recording (reference times from
the start of the reference, query times from the start of the clip, or
any constant origin; see Δ in [section 8](#8-reading-candidates)).

### From audiofp

[`audiofp`](https://crates.io/crates/audiofp) produces exactly these rows.
Its Wang fingerprinter takes 8 kHz mono `f32` audio and returns
`WangHash { hash, t_anchor }`, where `t_anchor` is an STFT frame at 62.5
frames per second:

```rust
use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

/// The rows of 8 kHz mono audio, one per Wang hash.
pub fn wang_rows(samples_8k_mono: &[f32]) -> Result<Vec<(u32, u32)>, audiofp::AfpError> {
    let fingerprint = Wang::default().extract(samples_8k_mono, SampleRate::HZ_8000)?;
    Ok(fingerprint
        .hashes
        .iter()
        .map(|h| (h.hash, h.t_anchor))
        .collect())
}
```

audiofp's Panako hashes (`PanakoHash { hash, t_anchor, .. }`) feed the
index the same way, as `(h.hash, h.t_anchor)`. Panako's hashes survive a
change of tempo, but the index's Δ voting still assumes the same speed:
an exact-speed copy is found as with Wang, a tempo-shifted one spreads
its votes (audiofp's `PanakoMatcher` handles tempo).

Rules that keep it working:

- **The same fingerprinter configuration on both sides.** A different
  `WangConfig` (fan-out, target zone) or sample rate produces different
  hashes, and nothing matches.
- **Decode, mix to mono and resample before fingerprinting.** audiofp's
  `io` module does this with its `std-*` codec features
  (`decode_to_mono_at(path, 8_000)`).
- **Frames, not `TimestampMs`.** audiofp's streaming fingerprinters also
  return a timestamp in milliseconds; index `t_anchor`.
- `examples/audiofp.rs` runs the whole path, including a clip with
  another track mixed underneath, and checks the offset it finds.

### Other fingerprinters

Anything that emits `(u32, u32)` rows with the properties above works:
your own landmark hashes, a 64-bit hash folded to 32 bits (`(h ^ (h >> 32))
as u32`; collisions only add chance votes), or per-frame sub-fingerprints
used as exact-match keys. Hashes wider than 32 bits must be folded; the
index keys on `u32`.

### Long references and the width of a time

A segment packs every time at the width of the largest time it holds
([section 12](#12-memory)). At 62.5 frames per second, 8.7 minutes fit 15
bits, 45 minutes 18 bits and 3 hours 20 bits. One very long reference
therefore widens every posting of the segment it lands in. If your
catalog mixes songs with multi-hour recordings, consider indexing long
recordings as consecutive chunks (each a reference of its own, with your
own map from chunk id to recording and start frame): every chunk then
keeps times small, and a match's position in the recording is the chunk's
start plus Δ.

### Hashes that appear everywhere

Silence, hum, clicks and test tones produce the same hashes in a large
share of the references. They cost the most (long runs) and tell the
least (votes everywhere). Drop them from queries before searching, or
from references before indexing. [Section 14](#14-working-with-segments-directly)
shows how to find them from the catalog itself.

### Duplicates

Exact duplicate postings (same hash, reference and time) are dropped when
a segment is built: a reference votes at most once per row. Duplicate
query rows are kept: each is a query row and votes on its own.

---

## 3. Reference ids

Ids are dense `u32`s assigned in call order, starting at `0`:

- `CatalogBuilder::add` and `Catalog::add` return the next id; a
  builder-built catalog and an add-built one agree on every id.
- An id is never reused. `Catalog::remove` tombstones it; it stays
  assigned, and `Catalog::reference_count()` (the number of ids assigned,
  which is also the next id `add` returns) counts it.
- A catalog snapshot keeps the whole id space and the liveness table, so
  ids survive a restart and the next `add` returns the id it would have
  returned before.

The catalog stores no names. Keep your own table from id to whatever
identifies a recording, and persist it next to the snapshot:

```rust
use afpindex::Catalog;

/// A catalog plus the name of every id it assigned, removed ones included.
#[derive(Default)]
pub struct Library {
    pub catalog: Catalog,
    names: Vec<String>,
}

impl Library {
    /// Index a recording and remember its name; returns its id.
    pub fn add(&mut self, name: String, rows: Vec<(u32, u32)>) -> u32 {
        let id = self.catalog.add(rows);
        // Ids are dense and in call order, so the table stays aligned.
        debug_assert_eq!(id as usize, self.names.len());
        self.names.push(name);
        id
    }

    /// Stop matching a recording. Its id and name stay assigned.
    pub fn remove(&mut self, id: u32) -> bool {
        self.catalog.remove(id)
    }

    /// The name of a live id.
    pub fn name(&self, id: u32) -> Option<&str> {
        self.catalog
            .is_live(id)
            .then(|| self.names[id as usize].as_str())
    }
}
```

If your ids come from elsewhere (a database export), build a `Segment`
with them and wrap it in `Catalog::from_segment`
([section 14](#14-working-with-segments-directly)): every id from `0` to the
largest in the segment is then live, including ids without postings, and
the next `add` continues after the largest. Keep such ids dense: a scratch
spends four bytes and a bit per id up to the largest, whether or not the
id has postings ([section 12](#12-memory)).

---

## 4. Building a catalog

### All at once: `CatalogBuilder`

```rust
use afpindex::Catalog;

/// Build a catalog from every reference's rows with one sort.
pub fn build(references: &[Vec<(u32, u32)>]) -> Catalog {
    let mut builder = Catalog::builder();
    for rows in references {
        builder.add(rows.iter().copied());
    }
    builder.build()
}
```

The builder collects postings (12 bytes each, up to twice that while its
vector grows) and sorts them once, in place, with a radix sort on the hash; with `std`, the buckets of that
sort are sorted on every core. The segment's arrays (about 8 bytes per
posting before packing) are allocated while the collected rows are still
alive, so peak memory is about 20 bytes per posting during `build`, then
drops to the packed size. 60 M postings build in 2.4–2.7 s; the real
catalog of 499 M postings built from rows in memory in 8.3 s.

### One at a time: `Catalog::add`

`Catalog::new()` (or a loaded snapshot) plus `add` per reference. Each
`add` sorts its reference into a small segment and merges segments from
the tail while one is at least half the size of the one before it, so a
posting takes part in `O(log n)` merges over its life. Adding 3,000 rows
to a catalog of 60 M postings took 1.15 ms on average and 209 ms at
worst on two hardware threads, the worst being the addition whose merge
reaches furthest; with `std`, a merge of a million postings or more runs
on every core, and the worst took 88 ms on 8 vCPUs and 62 ms on 16.

Use the builder for an initial or periodic full build, and `add` for
references that arrive while the catalog serves
([section 10](#10-changing-a-catalog)).

---

## 5. Queries

```rust
use afpindex::Query;

/// Group a clip's rows by hash, keeping the first 64 times of each hash.
pub fn query_of(clip_rows: Vec<(u32, u32)>) -> Query {
    Query::from_pairs(clip_rows, Some(64))
}
```

`Query::from_pairs(rows, max_times_per_hash)`:

- groups the rows by hash (distinct hashes, ascending);
- keeps, for each hash, its **first** `max_times_per_hash` times **in
  input order** (`None` keeps all). Feed rows in time order if the cap
  should keep the earliest times; audiofp's output is already sorted by
  `(t_anchor, hash)`;
- keeps duplicate rows (each votes on its own).

What it reports: `hash_count()` distinct hashes, `len()` rows kept after
the cap, `input_rows()` rows given before the cap (the usual input to
`vote_floor`), and `hashes()`, `times(i)` and `iter()` to look inside.

A `Query` is independent of any catalog: build it once and search it
against as many catalogs as you like. Building one is cheap (about 10 µs
for 800 rows).

**Clip length.** A true match's votes grow with the clip: about 300 Wang
hashes per second of music means a 10-second clip has about 3,000 rows. A
proportional floor (`vote_floor`, next section) keeps the bar the same
for short and long clips. Very short clips (a second or two) carry few
votes and are hard to tell from chance at any floor.

---

## 6. Searching

```rust
use afpindex::{Candidate, Catalog, Query, Scratch, SearchError, SearchOptions, vote_floor};

/// Search one clip with the options used on the real catalog.
pub fn search_clip(
    catalog: &Catalog,
    clip_rows: Vec<(u32, u32)>,
    scratch: &mut Scratch,
) -> Result<Vec<Candidate>, SearchError> {
    let query = Query::from_pairs(clip_rows, Some(64));
    let mut options = SearchOptions::default(); // fanout 64
    options.min_total_votes = vote_floor(query.input_rows(), 16, 0.02);
    catalog.search(&query, &options, scratch)
}
```

- The catalog is borrowed (`&self`): any number of threads search one
  catalog at the same time.
- A [`Scratch`](https://docs.rs/afpindex/latest/afpindex/struct.Scratch.html)
  holds the per-search buffers. Keep one per thread and reuse it: after
  the first search, `search_with` allocates nothing for a query of similar
  size, and `search` allocates only its result. One scratch serves any
  number of catalogs. It is left clean after every search, including one
  that fails and one whose visitor panics.
- Candidates come back one per survivor, in ascending reference order.

### What a search does, step by step

1. **Lookup.** Every query hash is looked up in every segment's directory
   (one read each; a hash's slot depends only on the hash, so the lookups
   of a query are independent and overlap in the memory system). Each hit
   is a run.
2. **Totals pass.** For every run, each posting adds its query hash's
   number of kept times to its reference's counter, counting at most
   `fanout` postings per `(hash, reference)` (the first by time). The
   counters are a dense array of one `u32` per reference id. Runs where no
   reference has as many postings as the fanout skip the fanout test.
3. **Survivors.** One scan of the counters collects the references whose
   total reaches `min_total_votes`, skips removed references, and resets
   the counters. Two bounds are checked here, before any histogram work
   ([section 15](#15-errors)).
4. **Histogram pass.** The runs are walked again; each vote of a survivor
   (query time, reference time) becomes `(Δ, query time)`, scattered by
   survivor with a counting sort, then sorted by Δ and merged into bins
   that record the votes and the first and last query time.
5. **Reduction.** Each survivor's bins become a `Candidate`: the best bin
   (most votes; ties to the smaller Δ) and noise statistics over the bins
   outside `±offset_tolerance` of it (`search`), or are handed to your
   closure as they are (`search_with`).

Chance matches stop at step 3: only survivors pay for a histogram, which
is why a floor that rejects them early matters more than anything else
for speed.

### What a search costs

- **Time** is dominated by the postings of the runs the query reads,
  decoded in the totals pass (three quarters of a search on the real
  catalog) and again for survivors in the histogram pass. Long runs
  (common hashes) and many segments (a lookup per hash per segment) make
  it slower; see [section 18](#18-troubleshooting).
- **Memory** is the scratch: 4 bytes and a bit per reference id, plus 16
  bytes per survivor vote and 32 bytes per survivor group while the
  histogram is built.

On the real catalog (120,329 references, 499 M postings) a search of one
of 104 real clips takes 0.65 ms at the median and 4.5 ms at p95 on one
hardware thread of an AWS `m7i-flex.large`, `simd` build.

---

## 7. Search options

`SearchOptions` is `#[non_exhaustive]`: start from the default and assign
fields.

| Field | Default | What it does | Cost and tuning |
|---|---|---|---|
| `fanout` | `Some(64)` | At most this many postings of one `(hash, reference)` vote: the first by time. `None`: all. `Some(0)`: none (the search finds nothing). | Bounds the votes one reference can collect from a hash it repeats (a sustained tone). Lower is faster on such runs; higher rarely changes results. |
| `min_total_votes` | `1` | The vote floor: a reference needs this many votes in total to survive. `0` and `1` are equivalent. | The most important knob. Too low and chance matches pay for histograms (slow, many weak candidates); too high and short or noisy true matches are lost. Use `vote_floor`. |
| `offset_tolerance` | `2` | Bins within `±offset_tolerance` frames of the best Δ are left out of the noise statistics. | At 62.5 frames per second, 2 frames is ±32 ms. Raise it if true matches smear over neighbouring bins (resampling, slight drift). Does not change `votes`. |
| `min_noise_bins` | `10` | With fewer noise bins than this, the noise mean, deviation and max are reported as `0`. | Avoids statistics from a handful of bins. `noise.bins` always counts them. |
| `max_votes` | `1 << 26` | Most votes the survivors of one search may scatter; the histogram holds 16 bytes per vote (default about 1 GiB). Exceeding it returns `TooManyVotes`. | A guard against a query with a floor so low that the histogram would exhaust memory. Lower it on small machines. |

Together, the query's `max_times_per_hash` and the `fanout` bound the
votes one `(hash, reference)` pair can cast: `times × min(fanout,
postings)`, at most 64 × 64 with the usual values.

### `vote_floor`

```rust
use afpindex::vote_floor;

/// The floor used on the real catalog: 2 % of the clip's rows, at least 16.
pub fn floor_for(rows: usize) -> u32 {
    vote_floor(rows, 16, 0.02)
}
```

`vote_floor(query_rows, min_votes, min_density)` is the smallest `v` with
`v / query_rows >= min_density`, but at least `min_votes`: 16 for a clip
of 100 rows, 84 for 4,200 rows. A `query_rows` of `0` or a non-positive
density yields `min_votes`.

Calibrate the two numbers on labelled clips: the floor should sit above
the totals of non-matching references and below those of true matches at
the shortest clip length you need. The values above (16 votes, 2 %) are
the ones used against the production catalog of 120,329 references, and
gave the same candidates as the production ClickHouse query on 103 of
104 clips (the last differs because ClickHouse keeps an arbitrary 64 of a
hash's 74 postings).

---

## 8. Reading candidates

| Field | Meaning |
|---|---|
| `reference` | The reference id |
| `total_votes` | Votes over all of the reference's bins (what passed the floor) |
| `delta` | Δ of the best bin, in frames |
| `votes` | Votes in the best bin |
| `first_time`, `last_time` | Smallest and largest query time among the best bin's votes |
| `noise.bins` | Bins outside `±offset_tolerance` of the best Δ |
| `noise.mean`, `noise.std_dev`, `noise.max` | Votes per noise bin: mean, population standard deviation, largest (zero below `min_noise_bins`) |

### Where the clip is

Δ is `reference_time − query_time`. The query's frame `t` lines up with
the reference's frame `t + Δ`:

- if the clip's frames start at `0`, Δ is where the clip starts in the
  reference; at 62.5 frames per second, `Δ / 62.5` seconds;
- the matched stretch of the clip is `first_time..=last_time` (query
  frames), which is `first_time + Δ ..= last_time + Δ` in the reference;
- a negative Δ means the clip starts before the reference does (the
  clip has audio before the reference's first frame).

### Is it a match?

The crate reports evidence, not a verdict. A true match concentrates its
votes: `votes` is far above the noise bins of the same reference, and a
large share of `total_votes`. A starting point to calibrate on labelled
data:

```rust
use afpindex::Candidate;

/// A first rule to calibrate: enough votes in the best bin, standing well
/// above that reference's other bins.
pub fn looks_like_a_match(c: &Candidate, min_votes: u32, min_z: f64) -> bool {
    if c.votes < min_votes {
        return false;
    }
    if c.noise.std_dev > 0.0 {
        (f64::from(c.votes) - c.noise.mean) / c.noise.std_dev >= min_z
    } else {
        // Too few noise bins to judge: rely on the vote count alone.
        true
    }
}
```

Other useful signals: `votes as f64 / total_votes as f64` (the share in
the best bin), `votes` against the clip's row count (a density), and
`last_time - first_time` against the clip length (how much of the clip
matched). When several candidates come back, rank them by `votes`; two
strong candidates usually mean the same audio is in two references
(re-releases, compilations).

---

## 9. Your own scoring: `search_with`

`Catalog::search_with` hands each survivor's bins to a closure instead of
reducing them, in ascending reference order, and allocates nothing after
warm-up. `Survivor { reference, total_votes, bins }` has its bins sorted by
Δ ascending and never empty; `survivor.candidate(&options)` is the
reduction `search` applies, and `Candidate::from_bins` with `Bin::new`
applies it to bins you build yourself.

A copy that is slightly faster or slower than the reference (resampling,
broadcast speed-up) drifts by a frame every so often, spreading its votes
over neighbouring bins. Summing a window of bins recovers them:

```rust
use afpindex::{Catalog, Query, Scratch, SearchError, SearchOptions};

/// For each survivor, the most votes in any window of Δ at most `2 * half_width`
/// frames wide: `(reference, first Δ of that window, votes in it)`.
pub fn windowed_votes(
    catalog: &Catalog,
    query: &Query,
    options: &SearchOptions,
    scratch: &mut Scratch,
    half_width: i64,
) -> Result<Vec<(u32, i64, u32)>, SearchError> {
    let mut out = Vec::new();
    catalog.search_with(query, options, scratch, |survivor| {
        let bins = survivor.bins; // sorted by Δ, never empty
        let (mut lo, mut sum) = (0, 0u32);
        let mut best = (bins[0].delta, 0u32);
        for hi in 0..bins.len() {
            sum += bins[hi].votes;
            while bins[hi].delta - bins[lo].delta > 2 * half_width {
                sum -= bins[lo].votes;
                lo += 1;
            }
            if sum > best.1 {
                best = (bins[lo].delta, sum);
            }
        }
        out.push((survivor.reference, best.0, best.1));
    })?;
    Ok(out)
}
```

Other uses: keeping only the top N survivors by votes without building
every candidate, writing bins to your own store for offline analysis,
or scoring with weights per Δ.

---

## 10. Changing a catalog

```rust
use afpindex::Catalog;

/// The whole life of a reference in a live catalog.
pub fn lifecycle(catalog: &mut Catalog, rows: Vec<(u32, u32)>) {
    let id = catalog.add(rows); // one sorted delta segment, merged in
    assert!(catalog.is_live(id));
    catalog.remove(id); // tombstone: skipped by every later search
    catalog.compact_deltas(); // merge the deltas, drop dead postings
    catalog.compact(); // one segment for everything
}
```

### Segments

A catalog is a list of immutable segments, largest first, each less than
half the size of the one before it: a catalog of one snapshot plus a day
of additions holds about ten. A search looks every query hash up in
every segment, so more segments mean more lookups (cheap individually,
but they add up), and a hash split across segments is read as several
shorter runs.

| Operation | What it does | Cost |
|---|---|---|
| `add(rows)` | Sorts the reference into a segment and merges from the tail while a segment is at least half its predecessor | Usually about a millisecond; occasionally a merge reaches a large segment (209 ms at worst at 60 M postings) |
| `remove(id)` | Tombstones the id; returns `false` if unknown or already removed | Constant. The postings stay until a merge or compaction drops them |
| `compact_deltas()` | Merges every segment but the largest into one, dropping removed references' postings | One pass over the deltas (0.23 s for four deltas of 5.4 M postings) |
| `compact()` | Merges everything into one segment, dropping removed postings | One pass over everything (0.58 s for 65 M postings); holds the result next to the segments it replaces while it runs |

The times are for two hardware threads of one core (an AWS
`m7i-flex.large`). With `std`, a merge of a million postings or more runs
on every core: `compact` took 0.22 s on 8 vCPUs and 0.15 s on 16, and the
result is the same segment as on one thread. Besides the result, it holds
the pieces being merged at the time, about an eighth of the output; the
peak memory of a whole build, change and reload cycle was 2 % higher on
two hardware threads and 9–15 % higher on 8 and 16 vCPUs; see
[Parallel merge](PERFORMANCE.md#parallel-merge).

A merge is one pass over every segment it joins, however many, so an
addition that cascades through several sizes rewrites each posting once.
Runs that need no change are copied as packed bits without being decoded,
and only segments that may hold a removed reference have their references
looked at (a catalog tracks the id range of each segment).

### Removed references still cost until compaction

A tombstoned reference never appears in results, but its postings stay in
their segment until that segment is merged: the totals pass still reads
them, and `posting_count()` and `heap_bytes()` still count them. After
removing many references, `compact_deltas` (if they were recent) or
`compact` gives the time and memory back.

### When to compact

- `segment_count()` grows by about one per doubling of what has been
  added since the last compaction. Compact the deltas when it passes 10
  or so, or on a timer.
- `posting_count()` against the postings of live references tells you how
  much removed data is still held.
- Compact on a clone, not on the catalog that serves (next section).
- Compact before writing a snapshot to leave removed postings out of it.

---

## 11. Threads and serving

`Catalog` is `Send + Sync` on targets with atomic pointers. Searches take
`&self`; each thread owns a `Scratch`:

```rust
use afpindex::{Candidate, Catalog, Query, Scratch, SearchError, SearchOptions};

/// Search many queries on `threads` threads, results in query order.
pub fn search_many(
    catalog: &Catalog,
    queries: &[Query],
    options: &SearchOptions,
    threads: usize,
) -> Vec<Result<Vec<Candidate>, SearchError>> {
    let chunk = queries.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = queries
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut scratch = Scratch::new(); // one per thread, reused
                    part.iter()
                        .map(|q| catalog.search(q, options, &mut scratch))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a search thread panicked"))
            .collect()
    })
}
```

A search uses one thread; run as many search threads as hardware
threads. Two hardware threads of one core share its caches and execution
units: on an `m7i-flex.large` (two hardware threads of one core), two
threads answered 779 searches per second on the real catalog. Separate
cores scale further: four laptop cores answered 33,000–34,000 searches per
second on the synthetic catalog.

### Changing a catalog while it serves

Mutations take `&mut self`, and a clone shares its segments (it costs
`O(segments + ids)`, not a copy of the postings). The usual pattern is a
writer that changes a private copy and publishes it:

```rust
use std::sync::{Arc, RwLock};

use afpindex::Catalog;

/// The catalog searches read, swapped whole by a single writer.
pub struct Served {
    current: RwLock<Arc<Catalog>>,
}

impl Served {
    pub fn new(catalog: Catalog) -> Self {
        Self {
            current: RwLock::new(Arc::new(catalog)),
        }
    }

    /// For a search: the catalog as it is now. Hold it for the search only.
    pub fn get(&self) -> Arc<Catalog> {
        Arc::clone(&self.current.read().expect("lock poisoned"))
    }

    /// For the writer: add references and compact on a copy, then swap it in.
    /// Searches never wait for the work, only for the swap.
    pub fn add_all(&self, references: Vec<Vec<(u32, u32)>>) -> Vec<u32> {
        let mut next = (*self.get()).clone(); // shares the segments
        let ids = references.into_iter().map(|rows| next.add(rows)).collect();
        if next.segment_count() > 10 {
            next.compact_deltas();
        }
        *self.current.write().expect("lock poisoned") = Arc::new(next);
        ids
    }
}
```

Keep one writer: two writers working on copies of the same catalog would
assign the same ids. Searches running on the old `Arc` finish on it; its
segments are freed when the last search lets go.

---

## 12. Memory

| What | Cost |
|---|---|
| A posting | Its reference width plus its time width, in bits (`Segment::widths`) |
| A distinct hash, per segment | 4 bytes of key plus about 20 of directory, or plus about 48 in a segment of fewer than 2<sup>20</sup> keys |
| An id | 1 byte of liveness table |
| A `Scratch`, per thread | 4 bytes and a bit per id of the largest catalog it searched, plus buffers that grow with the queries |
| A search's histogram | 16 bytes per survivor vote and 32 bytes per survivor group, while it runs |
| Building with `CatalogBuilder` | 12 bytes per collected posting, about 20 while `build` runs |

**Widths.** A segment packs references at the bits of its largest
reference id and times at the bits of its largest time. 120,329
references need 17 bits; times up to 32,767 frames (8.7 minutes at 62.5
frames per second) 15. The real catalog: 499.4 M postings × 32 bits =
1.86 GiB, plus 3.66 M hashes × 24 bytes = 0.08 GiB, is the 1.94 GiB it
measures. Packing saves 41 % against 32-bit arrays, for 3–8 % more
cycles per search.

**The directory** is an open-addressing table rebuilt from the keys
whenever a segment is built or loaded (never stored). A slot holds the
run's start and length and the most postings any one reference has in
it, which is what lets the search skip the fanout test on most runs.
A directory of fewer than 2<sup>20</sup> keys leaves four slots per key
instead of the 1.7 of a larger one: a lookup that misses then reaches an
empty slot in under two probes instead of five, which is what most
lookups in a delta segment are, at 48 bytes per key instead of 20 (at
most about 50 MB for a segment that small). There is no offsets array:
iteration in key order looks each key up in the directory, and a merge
recovers the order of the runs from one scan of the directory and a sort
of what it finds (16 bytes per distinct hash while the merge runs). A
directory too large for the cache is filled a region at a time from a
copy of its entries sorted by region, another 12 bytes per hash until it
is built.

**Measuring.** `Catalog::heap_bytes()` (segments and liveness table),
`Segment::heap_bytes()`, `Scratch::heap_bytes()`, and
`Scratch::shrink_to_fit()` to give a scratch's buffers back after an
unusually large query. A mapped catalog's `heap_bytes()` counts only what
is on the heap (keys and directories, about a quarter of the catalog).

---

## 13. Snapshots

The `std` feature writes and reads checksummed snapshot files; `mmap`
maps them.

### Writing and reading

```rust
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use afpindex::{Catalog, SnapshotError};

/// Write a snapshot so that a crash leaves either the old file or the new one.
pub fn save(catalog: &Catalog, path: &Path) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut out = BufWriter::with_capacity(1 << 23, File::create(&tmp)?);
    catalog.write_to(&mut out)?;
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    // Make the rename itself durable (directories can be synced on Unix).
    #[cfg(unix)]
    {
        let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
        File::open(dir.unwrap_or(Path::new(".")))?.sync_all()?;
    }
    Ok(())
}

/// Read a snapshot into memory.
pub fn load(path: &Path) -> Result<Catalog, SnapshotError> {
    Catalog::read_from(BufReader::with_capacity(1 << 23, File::open(path)?))
}

/// Map a snapshot instead (`mmap` feature).
pub fn map(path: &Path) -> Result<Catalog, SnapshotError> {
    Catalog::map_from(path)
}
```

A **catalog snapshot** stores the id space, the liveness table and every
segment as it stands: ids and removals survive the round trip, and the
next `add` returns the id it would have returned before. Call `compact`
first to leave removed postings out. `Segment::write_to`, `read_from` and
`map_from` do the same for one segment.

Use a large buffer (8 MiB above): the reader and writer move gigabytes.

### What a load checks

Reading verifies, before the catalog is usable:

- the magic (`AFCT` for a catalog, `AFIX` for a segment) and the format
  version (2);
- a checksum over the header and every array;
- that the arrays describe a valid segment: keys ascending, offsets
  consistent, each run sorted by `(reference, time)` without duplicates,
  and every reference inside the id space.

A large segment's runs are checked on every core. A file that fails any
check returns a `SnapshotError` ([section 15](#15-errors)) instead of a
catalog. The checksum is a fast 64-bit hash that catches accidents
(truncation, bit rot, a partial copy), not tampering: load snapshots only
from where you write them.

The load is bound by that check, then by the checksum and the directory.
On the real catalog (1.94 GiB) on two hardware threads, `read_from` takes
about 2 s and `map_from` about 1 s with the `simd` feature (2.7 s and
1.6 s without it), and huge pages for the heap take `read_from` down to
1.5 s.

### Trusted loads

`Catalog::read_from_trusted` and, with `mmap`, `Catalog::map_from_trusted`
(and the same pair on `Segment`) skip the two checks that read every
posting: the checksum of each segment, and the order of each run. The
header, the sizes, the keys and offsets, the liveness table and its
checksum, and the id space are still checked. The references are still
read once, because the search needs to know the most postings one
reference has in each run.

```rust
use std::path::Path;

use afpindex::{Catalog, SnapshotError};

/// Map a snapshot this service wrote and has loaded with every check before.
pub fn restart(path: &Path) -> Result<Catalog, SnapshotError> {
    Catalog::map_from_trusted(path)
}
```

Use them for files you wrote and have checked once, when the time to load
matters: a server restarting on the snapshot it saved, or a fleet that
checks a new snapshot on one host before every host maps it. A file
damaged since loads without an error, and searches and changes on it
return wrong candidates or panic; nothing undefined happens. On the real
catalog on two hardware threads with huge pages, `read_from_trusted` took
0.96 s against 1.65 s for `read_from`, and `map_from_trusted` 0.39 s
against 0.94 s.

### Mapping: `map_from`

With the `mmap` feature, `Catalog::map_from(path)` and
`Segment::map_from(path)` map the file instead of reading it:

- The packed postings stay in the file's pages: shared by every process
  that maps the same file, and reclaimable by the kernel without a write
  (they are clean file pages). Only the keys and the directories are
  built on the heap.
- The pages are faulted in before the call returns, and the same checksum
  and structural checks run in full.
- **The file must not be modified or truncated while the catalog, any
  clone of it, or any catalog still sharing one of its segments is
  alive.** A write through another handle is undefined behaviour for the
  mapping. Replace a snapshot by writing a new file and renaming it over
  the old one (as `save` above does): processes that mapped the old file
  keep its pages until they let go.

Prefer `map_from` when several processes serve the same catalog on one
host, when restarts should be fast, or when the kernel should be able to
reclaim the catalog's memory under pressure; prefer `read_from` for a
catalog you will change heavily (merges copy the postings to the heap
anyway).

### The format

Both formats are version 2: little-endian arrays behind a 32-byte header,
every section 8-byte aligned, with a checksum trailer. References and
times are stored packed as they are in memory, so a file is about the
size of the catalog.

```text
Segment snapshot
0   "AFIX"            4   version u32 = 2
8   ref_bits u32      12  time_bits u32
16  key_count u64     24  posting_count u64
32  keys    u32 × key_count                             (zero-padded to 8)
    offsets u64 × (key_count + 1)
    refs    ceil(posting_count × ref_bits / 8) bytes    (zero-padded to 8)
    times   ceil(posting_count × time_bits / 8) bytes   (zero-padded to 8)
    checksum u64

Catalog snapshot
0   "AFCT"            4   version u32 = 2
8   flags u32 = 0     12  reserved u32 = 0
16  reference_count u64   24  segment_count u64
32  live bitmap, one bit per id, LSB first              (zero-padded to 8)
    checksum u64
    segment_count segment snapshots, back to back
```

Flags and reserved fields are written as zero and ignored on read. The
directory is not stored; it is rebuilt from the keys on load (0.05 s on
the real catalog).

---

## 14. Working with segments directly

A `Segment` is the immutable unit a catalog is made of, usable on its
own: for an offline batch job that builds once and serves many times, for
ids you assign yourself, or to inspect what the index holds.

```rust
use afpindex::{Catalog, Posting, Segment};

/// Build, inspect, filter and merge segments, then serve one as a catalog.
pub fn segment_tour() -> Catalog {
    let segment = Segment::from_postings(vec![
        Posting::new(7, 0, 10), // hash 7 in reference 0 at frame 10
        Posting::new(7, 1, 3),
        Posting::new(9, 0, 4),
        Posting::new(7, 1, 3), // an exact duplicate: dropped
    ]);
    assert_eq!(segment.len(), 3);
    assert_eq!(segment.keys(), &[7, 9]);
    assert_eq!(segment.max_reference(), Some(1));

    // A hash's run: (reference, time), grouped by reference, times ascending.
    let run = segment.run(7).expect("hash 7 is indexed");
    assert_eq!(run.iter().collect::<Vec<_>>(), [(0, 10), (1, 3)]);
    assert!(segment.run(8).is_none());

    // Bits per posting: references up to 1 take 1 bit, times up to 10 take 4.
    assert_eq!(segment.widths(), (1, 4));

    // Every posting in (hash, reference, time) order.
    let all: Vec<Posting> = segment.postings().collect();
    assert_eq!(all[0], Posting::new(7, 0, 10));

    // Filter by reference, merge two segments (a posting in both is kept once).
    let without_1 = segment.filter(|reference| reference != 1);
    let merged = Segment::merge(&segment, &without_1, |_| true);
    assert_eq!(merged, segment);

    // Serve a segment whose ids you assigned: ids 0..=1 are live.
    let catalog = Catalog::from_segment(merged);
    assert_eq!(catalog.reference_count(), 2);
    catalog
}
```

With `std`, `merge` and `filter` run on every core once the inputs hold a
million postings, so the `keep` closure must be `Sync` and may be called
from several threads at once.

`Segment` also implements `FromIterator<Posting>`
(`postings.into_iter().collect::<Segment>()`), and a catalog exposes its
segments through `segment_count()`, `segment(i)` (largest first) and
`segments()`.

### Finding the hashes that appear everywhere

```rust
use std::collections::{BTreeMap, BTreeSet};

use afpindex::Catalog;

/// Hashes with more than `max_postings` postings across the catalog's segments.
pub fn common_hashes(catalog: &Catalog, max_postings: usize) -> BTreeSet<u32> {
    let mut postings: BTreeMap<u32, usize> = BTreeMap::new();
    for segment in catalog.segments() {
        for &hash in segment.keys() {
            let run = segment.run(hash).expect("a key has a run");
            *postings.entry(hash).or_default() += run.len();
        }
    }
    postings
        .into_iter()
        .filter(|&(_, n)| n > max_postings)
        .map(|(hash, _)| hash)
        .collect()
}
```

Then drop them from each clip before building its query
(`rows.retain(|(h, _)| !common.contains(h))`). The walk visits every
distinct hash once per segment: seconds on a catalog of millions of
hashes, so run it offline and keep the set.

---

## 15. Errors

### `SearchError`

Both variants are bounds checked after the totals pass and before any
histogram work. The catalog is never modified and the scratch is left
clean.

| Variant | Cause | What to do |
|---|---|---|
| `TotalsOverflow { reference }` | A live reference's total reached `u32::MAX` | Cap the query's times per hash, lower the fanout, or split the query. Only absurd queries get here (billions of votes). |
| `TooManyVotes { votes, max }` | The survivors would scatter more than `max_votes` votes (16 bytes each) | Raise the floor (usually the floor is far too low), lower the fanout, or raise `max_votes` if the memory is there |

```rust
use afpindex::{Candidate, Catalog, Query, Scratch, SearchError, SearchOptions};

/// Search, treating an over-broad query as "no match" rather than a failure.
pub fn search_or_nothing(
    catalog: &Catalog,
    query: &Query,
    options: &SearchOptions,
    scratch: &mut Scratch,
) -> Vec<Candidate> {
    match catalog.search(query, options, scratch) {
        Ok(candidates) => candidates,
        Err(SearchError::TooManyVotes { votes, max }) => {
            eprintln!("query too broad: {votes} votes over the limit of {max}");
            Vec::new()
        }
        Err(SearchError::TotalsOverflow { reference }) => {
            eprintln!("reference {reference} overflowed a vote counter");
            Vec::new()
        }
        // The enum is non-exhaustive: later versions may add bounds.
        Err(other) => {
            eprintln!("search failed: {other}");
            Vec::new()
        }
    }
}
```

### `SnapshotError` (`std`)

| Variant | Cause |
|---|---|
| `Io(e)` | The reader failed (the error's `source()` is the I/O error) |
| `Truncated` | The input ended before the snapshot did |
| `BadMagic` | Not a snapshot of this kind (a segment snapshot read as a catalog lands here too) |
| `UnsupportedVersion(v)` | Written by a format version this crate does not read |
| `Checksum` | The data does not hash to the stored checksum |
| `Corrupt(what)` | The header or arrays are inconsistent; `what` names the check |

All but `Io` mean the file is not a usable snapshot: rebuild it from the
source rows, or fall back to the previous file.

---

## 16. Features, platforms and `no_std`

| Feature | Default | Adds |
|---|---|---|
| `std` | yes | Snapshots (`write_to`, `read_from`, `read_from_trusted`), `std::error::Error` for `SnapshotError`, sorting the build's buckets, merging and checking loads on every core |
| `mmap` | no | `Catalog::map_from`, `map_from_trusted` and the same on `Segment` (implies `std`; adds `memmap2`). The mapping call is the feature's one `unsafe` block |
| `simd` | no | AVX2 decoding of packed postings (implies `std`) |

**`simd`.** On x86-64, each pass that reads many packed values asks the
CPU for AVX2 when it starts; if present, it decodes eight values per block
(one byte shuffle and one variable shift) for widths up to 25 bits:
the totals pass, the search for a survivor's group in a short run, and
the check of every run at load. Wider fields, ranges shorter than four
entries, other CPUs and other architectures run the same scalar code as
without the feature, so results never change. Entering the AVX2 code is
the feature's one `unsafe` block; the kernel itself is safe code. On the
real catalog it took 13 % off the summed latency of the clips and a
fifth off `read_from` and over a third off `map_from`.

**`-C target-cpu=native`** gained a further 5 % on a laptop (a vectorised
maximum in the survivor scan, BMI2 shifts in the packed loads) but 9 %
less on an AWS host: measure on yours. The crate sets no target features
itself.

**`no_std + alloc`.** With `default-features = false` the crate needs only
`alloc`: building, changing and searching catalogs, queries and segments
all work; snapshots need `std`.

```toml
[dependencies]
afpindex = { version = "0.1", default-features = false }
```

Segments are shared through `Arc` on targets with atomic pointers and
`Rc` elsewhere, where a catalog is then not `Send`.

**Limits.** A segment holds fewer than 2<sup>40</sup> postings. Ids are
`u32`; `add` panics once every id below `u32::MAX` is assigned. The crate
is tested on 32-bit x86, where `Catalog::from_segment` with the id
`u32::MAX` would not fit `usize` and panics.

**Minimum Rust version** 1.87 for the library. `examples/audiofp.rs`
needs 1.93 (its dev-dependency `audiofp` does).

---

## 17. Running in production

### Huge pages for the heap

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

### A checklist

- **Build with `simd`** on x86-64 hosts (`features = ["mmap", "simd"]`).
- **One `Scratch` per worker thread**, created once; as many search
  threads as hardware threads.
- **Load with `map_from`** for fast restarts and shared pages across
  processes; `read_from` for a catalog that changes heavily. Once a
  snapshot has passed a full load, `map_from_trusted` maps it again in
  under half the time ([section 13](#trusted-loads)).
- **Write snapshots atomically** (temporary file, sync, rename) and
  compact first.
- **Keep the id table** (id to recording) next to the snapshot, written in
  the same step.
- **Change on a copy and swap** ([section 11](#11-threads-and-serving));
  one writer.
- **Compact the deltas** when `segment_count()` passes about 10, and
  everything after large removals.
- **Bound the work**: a proportional floor (`vote_floor`), `Some(64)` times
  per hash and fanout, `max_votes` sized to the memory you can spend per
  search.
- **Watch** `segment_count()`, `posting_count()`, `live_count()`,
  `heap_bytes()` and search latency percentiles.
- **Measure on your hardware** with `examples/bench.rs`,
  `examples/lifecycle.rs` and `examples/replay.rs`
  ([PERFORMANCE.md](PERFORMANCE.md#reproducing)).

### Sizing

Memory is about 4 bytes per posting with typical widths, plus 24 bytes per
distinct hash, plus 4 bytes and a bit per id per search thread. Estimate
postings from your fingerprinter's density: audiofp's Wang gives about
300 hashes per second of music at its defaults, so a 3-minute track is
about 54,000 postings and 100,000 such tracks about 5.4 G postings, or
20 GiB. The real catalog averages 4,150 postings per reference (499 M
over 120,329) and holds them in 1.94 GiB. Searches are bound by memory
latency: more memory channels and cores scale throughput, and the target
is a host whose catalog fits in RAM with room for the page cache (for
`map_from`).

---

## 18. Troubleshooting

| Symptom | Likely cause | What to try |
|---|---|---|
| No candidates for a clip that should match | Different fingerprinter settings or sample rate on the two sides; times in different units; floor too high for the clip length | Fingerprint both with the same config at the same rate; check `query.len()` and a candidate's `total_votes` with a floor of 1 |
| Many weak candidates | Floor too low | `vote_floor(rows, 16, 0.02)` or higher; calibrate on labelled clips |
| The right reference, wrong Δ | Times not in frames, or a time origin that differs per row | Index the fingerprinter's frame indices as they are |
| Votes smeared over neighbouring Δ | A slight speed difference | `search_with` with a window ([section 9](#9-your-own-scoring-search_with)), a larger `offset_tolerance` |
| Slow searches | Hashes that appear everywhere; many segments; a floor so low that chance matches get histograms; no huge pages | Drop common hashes ([section 14](#14-working-with-segments-directly)); `compact_deltas`; raise the floor; `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`; the `simd` feature |
| `TooManyVotes` | Floor far too low, or a query full of common hashes | Raise the floor; drop common hashes; lower the fanout |
| Memory above the estimate | Removed references not yet compacted; one long reference widening a segment's times; a scratch kept large by one huge query | `compact`; split long recordings into chunks; `Scratch::shrink_to_fit` |
| `SnapshotError::Checksum` or `Corrupt` | A partial copy or a file changed in place | Write snapshots atomically; rebuild from rows |
| `SnapshotError::BadMagic` | A segment snapshot read as a catalog or the reverse, or not a snapshot | Use the matching `read_from` |
| A mapped catalog crashes or reads garbage | The file was modified or truncated while mapped | Never write to a mapped snapshot; write a new file and rename it |
