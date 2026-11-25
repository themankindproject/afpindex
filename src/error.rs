use core::fmt;

/// Why a search produced no result.
///
/// Both variants are bounds the caller controls through
/// [`SearchOptions`](crate::SearchOptions) and
/// [`Query::from_pairs`](crate::Query::from_pairs). The catalog is never
/// modified, and the scratch is left clean for the next search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SearchError {
    /// A live reference's vote total reached `u32::MAX`, the most the
    /// 32-bit counters of the totals pass can hold. Detected at the end of that
    /// pass; no histogram work was done.
    ///
    /// Cap the query's times per hash, lower the fanout, or split the query.
    TotalsOverflow {
        /// The first reference (by id) whose total overflowed.
        reference: u32,
    },
    /// The survivors would scatter more votes than
    /// [`SearchOptions::max_votes`](crate::SearchOptions::max_votes) allows.
    /// Detected after the totals pass; no histogram work was done.
    ///
    /// Raise the limit, raise the vote floor, or lower the fanout.
    TooManyVotes {
        /// Votes the survivors would scatter (saturated at `u64::MAX`).
        votes: u64,
        /// The configured limit.
        max: usize,
    },
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TotalsOverflow { reference } => {
                write!(
                    f,
                    "reference {reference}'s vote total exceeds a 32-bit counter"
                )
            }
            Self::TooManyVotes { votes, max } => {
                write!(
                    f,
                    "survivors would scatter {votes} votes, above the limit of {max}"
                )
            }
        }
    }
}

impl core::error::Error for SearchError {}

/// Why a snapshot could not be read.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[derive(Debug)]
#[non_exhaustive]
pub enum SnapshotError {
    /// The reader failed; the cause is the error's source.
    Io(std::io::Error),
    /// The input ended before the snapshot did.
    Truncated,
    /// The stream does not start with the magic of the snapshot kind being
    /// read (a catalog snapshot read as a segment, or vice versa, lands
    /// here too).
    BadMagic,
    /// The snapshot was written by a format version this crate does not read.
    UnsupportedVersion(u32),
    /// The data does not hash to the checksum stored with it.
    Checksum,
    /// The header or the arrays are inconsistent; the message names the check.
    Corrupt(&'static str),
}

#[cfg(feature = "std")]
impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(_) => f.write_str("snapshot I/O error"),
            Self::Truncated => f.write_str("snapshot ends early"),
            Self::BadMagic => f.write_str("not an hashtally snapshot of this kind"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported snapshot version {v}"),
            Self::Checksum => f.write_str("snapshot checksum mismatch"),
            Self::Corrupt(what) => write!(f, "corrupt snapshot: {what}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Self::Truncated
        } else {
            Self::Io(e)
        }
    }
}
