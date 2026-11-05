/// One occurrence of a hash in a reference.
///
/// The derived ordering is `(hash, reference, time)`, which is the order a
/// [`Segment`](crate::Segment) stores postings in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Posting {
    /// The landmark hash.
    pub hash: u32,
    /// Reference id, assigned by a [`Catalog`](crate::Catalog) or by the
    /// caller when building a [`Segment`](crate::Segment) directly.
    pub reference: u32,
    /// Anchor time of the hash inside the reference, in the fingerprinter's
    /// frames.
    pub time: u32,
}

impl Posting {
    /// Build a posting.
    #[must_use]
    pub const fn new(hash: u32, reference: u32, time: u32) -> Self {
        Self {
            hash,
            reference,
            time,
        }
    }
}
