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
//! `hashtally` answers that search exactly over hundreds of millions of
//! postings; the README lists the features in full:
//!
//! - **Postings sorted by hash,** packed to the width of the largest value:
//!   each hash is one contiguous run found by a single directory read.
//! - **Count before histogram.** A branch-free pass totals each reference's
//!   votes; only references that reach a vote floor get a histogram.
//! - **Immutable, shared segments.** Additions go into delta segments merged
//!   geometrically; removals tombstone a reference; ids never change.
//! - **No unsafe code by default, no dependencies beyond `libm`,** and a
//!   `no_std + alloc` core. `std` adds snapshot files; `mmap` and `simd`
//!   bring one `unsafe` block each.
//! - **A catalog in a directory.** With `mmap`, [`CatalogStore`] keeps
//!   segment files and a manifest, [`Catalog::map_dir`] searches them
//!   with the postings left in the files' pages, and [`replica::sync`]
//!   keeps a copy of the directory up to date.
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), hashtally::SearchError> {
//! use hashtally::{Catalog, Query, Scratch, SearchOptions};
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
//! A reference whose votes reach [`SearchOptions::min_total_votes`] (at
//! most [`SearchOptions::fanout`] postings of a hash vote per reference) is
//! a **survivor**. Its votes are merged into [`Bin`]s by Δ and reduced to a
//! [`Candidate`], in ascending reference order and deterministically.
//! [`Catalog::search_with`] hands each [`Survivor`] with its bins to a
//! closure instead, for callers with their own scoring.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Two `unsafe` blocks, each behind a feature: the mapping call of `mmap`
// and the entry into code compiled for AVX2 of `simd`.
#![cfg_attr(not(any(feature = "mmap", feature = "simd")), forbid(unsafe_code))]

extern crate alloc;

mod catalog;
mod error;
#[cfg(feature = "mmap")]
mod manifest;
mod packed;
#[cfg(feature = "std")]
mod par;
mod posting;
mod query;
mod refs;
#[cfg(feature = "mmap")]
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
pub mod replica;
mod search;
mod segment;
mod simd;
#[cfg(feature = "std")]
mod snapshot;
mod sort;
#[cfg(feature = "mmap")]
mod store;
#[cfg(test)]
mod test_util;

// The guide's examples run as doctests, so they cannot drift from the API.
// They map snapshots, hence `mmap`; the first one uses the `audiofp`
// dev-dependency.
#[cfg(all(doctest, feature = "mmap"))]
#[doc = include_str!("../docs/USAGE.md")]
mod usage_guide {}

pub use catalog::{
    Catalog, CatalogBuilder, CatalogStats, Compaction, CompactionPolicy, SegmentStats,
};
pub use error::SearchError;
#[cfg(feature = "std")]
pub use error::SnapshotError;
#[cfg(feature = "mmap")]
pub use manifest::{Manifest, ManifestEntry};
#[cfg(feature = "std")]
pub use par::{set_max_threads, threads as max_threads};
pub use posting::Posting;
pub use query::Query;
pub use refs::RefCoding;
pub use search::{Bin, Candidate, NoiseStats, Scratch, SearchOptions, Survivor, vote_floor};
pub use segment::{Postings, Run, Segment};
#[cfg(feature = "mmap")]
pub use store::{CatalogStore, CatalogStoreBuilder};
