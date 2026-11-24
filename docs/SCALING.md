# Scaling past one host

What this crate gives a service that outgrows one machine, and what the
service has to add. Nothing here is a feature of the crate beyond what
[USAGE.md](USAGE.md) describes; it is how those features fit together.
The numbers come from [PERFORMANCE.md](PERFORMANCE.md).

- [When one host is not enough](#when-one-host-is-not-enough)
- [Shard by reference](#shard-by-reference)
- [A search across shards](#a-search-across-shards)
- [Replicas from segment files](#replicas-from-segment-files)
- [A writer that fails](#a-writer-that-fails)
- [Sizing a shard](#sizing-a-shard)
- [What stays outside the crate](#what-stays-outside-the-crate)

## When one host is not enough

A search reads every posting of every run its query matches: on the real
catalog of 499 M postings, the clips read 0.42 M postings at the median
and 5.1 M at most, and skipping the runs that cannot lift a reference to
the floor would save 1.2 % of that
([measured](PERFORMANCE.md#skipping-runs-that-cannot-reach-the-floor-left-out)). So the work of a search grows with the catalog, and so
does the memory it needs: the packed references of every segment, which
each search reads, must be in memory for searches to stay in the
millisecond range. The times can stay on disk; only survivors read them.

One host stops being enough when either is too much for it:

- **Memory**: the references (their bit width per posting, 17 bits on the
  real catalog) plus about 24 bytes per distinct hash per segment for the
  keys and the directory no longer fit beside the page cache the times
  need.
- **Latency**: the slowest searches take too long. `SearchOptions::threads`
  takes about a third off searches of 5 ms and more on four idle cores;
  past that, the catalog has to be split.

Both point the same way: split the catalog so that each part holds and
reads fewer postings.

## Shard by reference

Give every reference to exactly one shard, and keep each shard a complete
catalog of its references: all of their postings, every hash.

A reference's votes then all land in one shard, so each shard computes the
reference's exact total, applies the floor and builds its histogram on its
own. Nothing has to be combined across shards except the candidates, and
the results are the ones one catalog of every reference gives.

Sharding by hash instead would put a reference's votes for different
hashes on different shards. Each shard would have to send the vote total
of every reference it saw, the totals would have to be added before the
floor could be applied, and the histogram pass would need the survivors'
postings from every shard: a round of messages as large as the counters
on every search.

Assign references to shards by something stable about them (a hash of the
external id, or ranges of it), and balance by postings rather than by
reference count: a long recording costs a shard as much as many short
ones. Each shard has its own dense id space, assigned by its own
`Catalog::add`, so each shard keeps its own table from id to external id
([USAGE.md, section 3](USAGE.md#3-reference-ids)).

## A search across shards

Group the clip once (`Query::from_pairs`), compute the floor once
(`vote_floor` depends only on the query's rows), and send the same query
and options to every shard. Each shard returns its candidates; the
coordinator maps each `(shard, id)` to the external id and merges the
lists by score.

- **Latency** is the slowest shard's. A shard with a quarter of the
  postings reads about a quarter of them per search.
- **Total work** does not fall: the shards together read the same postings
  one catalog would, and every shard looks every query hash up in its own
  directory, so a search costs one directory lookup per hash per shard
  more. Sharding buys memory per host and latency, not throughput per
  core.
- **A shard that fails** to answer makes the result incomplete, not wrong:
  its references are missing. Whether to wait, retry on a replica or
  answer without it is the coordinator's decision.

## Replicas from segment files

A shard is a catalog directory ([USAGE.md, section 13](USAGE.md#a-catalog-in-a-directory)):
one writer (`CatalogStore`), segment files that never change once they
have a name, and a manifest that lists the ones that make the catalog. A
segment file's name is never reused. That is what read replicas need:
copying a file that cannot change needs no coordination.

The writer of a shard, after each `save`:

1. uploads every segment file the manifest names that the object store
   does not have yet;
2. uploads the manifest last, as one object, once every file it names is
   there.

A replica:

1. reads the manifest from the object store and compares its generation
   with the one it serves;
2. downloads the segment files it does not have into its own local
   directory, and checks each new one once with `Segment::map_from`, which
   runs every check;
3. writes the manifest into its directory through a temporary name and a
   rename;
4. opens the directory with `Catalog::map_dir_trusted` (its files have
   all been checked by now), calls `warm_references`, and swaps the new
   catalog in for searches
   ([USAGE.md, section 11](USAGE.md#changing-a-catalog-while-it-serves));
5. deletes the segment files the new manifest no longer names once the
   catalog that mapped them is gone.

A replica's directory is its own: the writer's lock and its removal of
files it does not recognise only make sense for one writer per directory.
`map_dir_trusted` takes no lock and writes nothing, so any number of
replicas can each keep their own copy. The files a writer adds are mostly
small (additions go to small segments; only compactions write large
ones), so a replica's refresh usually downloads little.

## A writer that fails

The lock on a store's directory belongs to the process and goes with it,
so a new writer on the same host or volume opens the directory as soon as
the old process is gone, and finds the catalog of its last `save`. What
was added after that `save` is lost; `Catalog::reference_count` of the
opened catalog is the id the next addition gets, so the service replays
its own log of additions from there.

If the host is lost with its disk, the writer's directory is rebuilt from
the object store the way a replica's is, and opened with
`CatalogStore::open`. Additions after the last upload are lost the same
way and replayed the same way.

## Sizing a shard

| per | in memory for searches | on disk only |
|---|---|---|
| posting | its reference width in bits (17 on the real catalog) | its time width (15 on the real catalog) |
| distinct hash, per segment | 4 bytes of key and about 20 of directory (about 48 below 2<sup>20</sup> keys) | |
| reference id | 1 byte of liveness table, and 4 bytes per search thread | |

On the real catalog that is 1.06 GB of references and 0.09 GB of keys and
directory (3.66 M hashes) for 499 M postings, against 2.0 GB of segment file. A shard sized
so that its references and directories fit in about half its memory
leaves the rest for the scratches of its search threads and the page cache
that serves the times.

## What stays outside the crate

- The network: sending queries to shards and candidates back.
- Membership: which shards and replicas exist, and which reference goes to
  which shard.
- The table from each shard's ids to external ids, and the log of
  additions and removals a writer replays after a failure.
- Uploading and downloading segment files and manifests.
