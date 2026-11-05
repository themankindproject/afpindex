//! Exact in-memory inverted index for landmark audio fingerprints.
//!
//! A landmark fingerprinter (Wang / Shazam-style, Panako, and friends) turns
//! audio into rows of `(hash, time)`. A **posting** records one occurrence of
//! a hash in a reference: `(hash, reference, time)`. A query clip yields its
//! own `(hash, time)` rows. A copy of a reference keeps its hashes and shifts
//! all their times by one constant, so for every query row and every posting
//! with the same hash the matcher casts a **vote** for the bin
//! `(reference, Δ = time − query_time)`. A true copy piles many votes into one
//! bin; chance collisions scatter single votes across many offsets.
//!
//! `afindex` answers that search exactly and fast over hundreds of millions
//! of postings:
//!
//! - **Postings sorted by hash** in flat arrays, references and times packed
//!   to the width of the largest value, so each hash is one contiguous run
//!   located by a single directory read. A query's lookups
//!   are independent, and every pass loads the next batch of lines before
//!   it works on the current one, so a search is bounded by memory
//!   parallelism rather than by one cache miss after another.
//! - **Count before histogram.** A branch-free first pass totals each
//!   reference's votes in a dense counter array; only references that reach
//!   a vote floor get a histogram. Chance matches never pay for a sort.
//! - **Immutable, shared segments.** Additions go into small delta segments
//!   that are merged geometrically; removals tombstone a reference and
//!   compaction drops its postings. Reference ids never change, and a
//!   catalog clone shares its segments.
//! - **No unsafe code, no dependencies beyond `libm`,** and a `no_std + alloc`
//!   core. The `std` feature adds checksummed snapshot files for segments
//!   and whole catalogs; the `mmap` feature maps those files instead of
//!   reading them, and its one mapping call is the crate's only `unsafe`
//!   block.
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), afindex::SearchError> {
//! use afindex::{Catalog, Query, Scratch, SearchOptions};
//!
//! // Two references with eight postings each. Reference 1 is reference 0
//! // shifted 500 frames later.
//! let rows: Vec<(u32, u32)> = (0..8).map(|i| (100 + i, 10 * i)).collect();
//! let mut builder = Catalog::builder();
//! let r0 = builder.add(rows.iter().copied());
//! let r1 = builder.add(rows.iter().map(|&(h, t)| (h, t + 500)));
//! let catalog = builder.build();
//!
//! // A query clip that is reference 1's audio starting 30 frames in.
//! let query = Query::from_pairs(
//!     rows.iter().map(|&(h, t)| (h, t + 500 - 30)),
//!     Some(64),
//! );
//! let mut options = SearchOptions::default();
//! options.min_total_votes = 4;
//! let mut scratch = Scratch::new();
//! let candidates = catalog.search(&query, &options, &mut scratch)?;
//!
//! // Both references reach the floor; the best bin of each tells them apart.
//! assert_eq!(candidates.len(), 2);
//! let best = candidates.iter().max_by_key(|c| c.votes).unwrap();
//! assert_eq!(best.reference, r1);
//! assert_eq!(best.delta, 30);
//! assert_eq!(best.votes, 8);
//! let other = candidates.iter().find(|c| c.reference == r0).unwrap();
//! assert_eq!(other.delta, -470);
//! # Ok(())
//! # }
//! ```
//!
//! # What a search computes
//!
//! For each query hash and each segment holding it, the search walks the
//! hash's run. Within a run each reference's postings are contiguous with
//! times ascending, and at most [`SearchOptions::fanout`] of them count. The
//! query side keeps at most `max_times_per_hash` times per hash
//! ([`Query::from_pairs`]). A reference whose total votes reach
//! [`SearchOptions::min_total_votes`] is a **survivor**; every survivor's
//! votes are sorted by Δ, merged into [`Bin`]s, and reduced to one
//! [`Candidate`]: its best bin (most votes, ties to the smaller Δ) plus noise
//! statistics over the bins outside `±offset_tolerance` of it. Candidates
//! come back in ascending reference order, and the whole computation is
//! deterministic.
//!
//! [`Catalog::search_with`] hands each [`Survivor`] with its bins to a
//! closure instead, for callers with their own scoring.
//!
//! # Scale
//!
//! `examples/bench.rs` (shipped with the crate) builds a synthetic catalog
//! and reports latency and throughput; see the README for measured numbers.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// The one `unsafe` block is the mapping call behind the `mmap` feature.
#![cfg_attr(not(feature = "mmap"), forbid(unsafe_code))]

extern crate alloc;

mod catalog;
mod error;
mod packed;
mod posting;
mod query;
mod search;
mod segment;
#[cfg(feature = "std")]
mod snapshot;
mod sort;

pub use catalog::{Catalog, CatalogBuilder};
pub use error::SearchError;
#[cfg(feature = "std")]
pub use error::SnapshotError;
pub use posting::Posting;
pub use query::Query;
pub use search::{Bin, Candidate, NoiseStats, Scratch, SearchOptions, Survivor, vote_floor};
pub use segment::{Postings, Run, Segment};
