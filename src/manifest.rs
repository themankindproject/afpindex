//! The manifest of a catalog directory (`mmap`), and its reader.
//!
//! A directory holds one catalog: a `MANIFEST` and one segment snapshot
//! per segment (see [`crate::CatalogStore`] for the names and the order
//! they are written in). The manifest says which segments make the catalog
//! and what the catalog knows of its references; the segments are
//! immutable, so a change writes new segment files and a new manifest and
//! lets the old ones go.
//!
//! ```text
//! 0   "AFCM"            4   version u32 = 1
//! 8   generation u64    16  reference_count u64
//! 24  segment_count u64 32  next_file u32
//! 36  reserved u32 = 0
//! 40  live bitmap, one bit per id, LSB first, ceil(n / 8) bytes
//!     (zero-padded to 8)
//!     segment_count entries of 40 bytes, in catalog order (largest first)
//!     checksum u64
//! ```
//!
//! An entry is the segment's file id (`u32`), a reserved word, its length
//! in bytes (`u64`), its key count (`u64`), its posting count (`u64`) and
//! its reference range (`min u32`, `max u32`, both zero without postings):
//! what a load checks the file against before it trusts a mapping of it.
//!
//! The checksum folds the header words, the bitmap bytes as little-endian
//! 64-bit words (the last one zero-padded) and then each entry's five
//! words, as [`Checksum`] does elsewhere.
//!
//! The generation rises by one with every manifest written, and a store
//! refuses to replace a manifest whose generation is not the one it last
//! read or wrote: that is what a second writer looks like. File ids are
//! never reused, so a file a crash left behind can never be mistaken for a
//! newer segment that happens to take its name.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use std::io::Write;

use crate::SnapshotError;
use crate::snapshot::{Checksum, pad_len};

/// The first bytes of a manifest.
const MAGIC: &[u8; 4] = b"AFCM";

/// Bytes before the liveness bitmap.
const HEADER: usize = 40;
/// Bytes of one segment entry.
const ENTRY: usize = 40;
/// Version of the manifest format.
const VERSION: u32 = 1;
/// Name of the file that names the catalog.
pub(super) const MANIFEST: &str = "MANIFEST";
/// Name of a manifest being written.
pub(super) const MANIFEST_TEMP: &str = "MANIFEST.tmp";
/// Bytes of a manifest up to the end of its generation.
pub(super) const GENERATION_END: usize = 16;
/// A segment count no real catalog reaches; a larger one is a corrupt file
/// rather than a reason to allocate.
const MAX_SEGMENTS: u64 = 1 << 24;

/// One segment of the catalog: its file, its shape and its references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub(super) file: u32,
    pub(super) bytes: u64,
    pub(super) keys: u64,
    pub(super) postings: u64,
    /// Smallest and largest reference of the segment, or `None` without
    /// postings.
    pub(super) ids: Option<(u32, u32)>,
}

/// What a manifest holds: a catalog's id space, its segments and the
/// generation a writer compares before it replaces the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Manifest {
    pub(super) generation: u64,
    pub(super) reference_count: u64,
    /// The id the next segment file gets; ids are never reused.
    pub(super) next_file: u32,
    pub(super) live: Vec<bool>,
    pub(super) entries: Vec<Entry>,
}

impl Manifest {
    /// Write the manifest.
    pub(super) fn write(&self, w: &mut impl Write) -> std::io::Result<()> {
        let mut ck = Checksum::new();
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        ck.word(u64::from(VERSION));
        for word in [
            self.generation,
            self.reference_count,
            self.entries.len() as u64,
        ] {
            ck.word(word);
            w.write_all(&word.to_le_bytes())?;
        }
        ck.word(u64::from(self.next_file));
        w.write_all(&self.next_file.to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        let mut bitmap = alloc::vec![0u8; self.live.len().div_ceil(8)];
        for (i, &live) in self.live.iter().enumerate() {
            if live {
                bitmap[i / 8] |= 1 << (i % 8);
            }
        }
        ck.words(&bitmap);
        w.write_all(&bitmap)?;
        let pad = pad_len((HEADER + bitmap.len()) as u64);
        w.write_all(&[0u8; 8][..pad])?;
        for entry in &self.entries {
            let (min, max) = entry.ids.unwrap_or((0, 0));
            w.write_all(&entry.file.to_le_bytes())?;
            w.write_all(&0u32.to_le_bytes())?;
            for word in [entry.bytes, entry.keys, entry.postings] {
                ck.word(word);
                w.write_all(&word.to_le_bytes())?;
            }
            w.write_all(&min.to_le_bytes())?;
            w.write_all(&max.to_le_bytes())?;
            ck.word(u64::from(entry.file));
            ck.word(u64::from(min) | u64::from(max) << 32);
        }
        w.write_all(&ck.finish().to_le_bytes())?;
        w.flush()
    }
}

/// The file name of segment `file`.
pub(super) fn segment_name(file: u32) -> String {
    format!("seg-{file:08x}.afs")
}

/// The file name a segment is written to before it is renamed.
pub(super) fn temp_name(file: u32) -> String {
    format!("seg-{file:08x}.tmp")
}

/// The id of the segment file called `name`, if that is a segment's name.
pub(super) fn segment_id(name: &str) -> Option<u32> {
    file_id(name, ".afs")
}

/// Whether `name` is the name of a file being written.
pub(super) fn is_temp(name: &str) -> bool {
    name == MANIFEST_TEMP || file_id(name, ".tmp").is_some()
}

/// The id in `seg-<8 hex><suffix>`, as [`segment_name`] spells it.
fn file_id(name: &str, suffix: &str) -> Option<u32> {
    let digits = name.strip_prefix("seg-")?.strip_suffix(suffix)?;
    let spelled = digits.len() == 8
        && digits
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    spelled
        .then(|| u32::from_str_radix(digits, 16).ok())
        .flatten()
}

/// The generation of the manifest that starts with `head`, or `None` when
/// those bytes do not start a manifest.
pub(super) fn generation(head: &[u8; GENERATION_END]) -> Option<u64> {
    let (magic, rest) = head.split_at(4);
    (magic == MAGIC).then(|| u64::from_le_bytes(rest[4..].try_into().expect("8 bytes")))
}

/// Read a manifest from `bytes`, the whole file's contents.
pub(super) fn read(bytes: &[u8]) -> Result<Manifest, SnapshotError> {
    let head = bytes.get(..HEADER).ok_or(SnapshotError::Truncated)?;
    if head[..4] != *MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let version = u32::from_le_bytes(head[4..8].try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(SnapshotError::UnsupportedVersion(version));
    }
    let mut words = head[8..32].chunks_exact(8);
    let mut next_word = || {
        let chunk = words.next()?;
        Some(u64::from_le_bytes(chunk.try_into().expect("8 bytes")))
    };
    let generation = next_word().ok_or(SnapshotError::Truncated)?;
    let reference_count = next_word().ok_or(SnapshotError::Truncated)?;
    let n_segments = next_word().ok_or(SnapshotError::Truncated)?;
    let next_file = u32::from_le_bytes(head[32..36].try_into().expect("4 bytes"));
    if n_segments > MAX_SEGMENTS {
        return Err(SnapshotError::Corrupt("implausible segment count"));
    }
    if reference_count > u64::from(u32::MAX) {
        return Err(SnapshotError::Corrupt("id space exceeds u32"));
    }
    let mut ck = Checksum::new();
    for word in [
        u64::from(version),
        generation,
        reference_count,
        n_segments,
        u64::from(next_file),
    ] {
        ck.word(word);
    }
    let n_refs = reference_count as usize;
    let bitmap_len = n_refs.div_ceil(8);
    let bitmap = bytes
        .get(HEADER..HEADER + bitmap_len)
        .ok_or(SnapshotError::Truncated)?;
    ck.words(bitmap);
    let after = HEADER + bitmap_len + pad_len((HEADER + bitmap_len) as u64);
    let entries_len = (n_segments as usize)
        .checked_mul(ENTRY)
        .ok_or(SnapshotError::Corrupt(
            "the segment count exceeds the address space",
        ))?;
    let end = after
        .checked_add(entries_len)
        .ok_or(SnapshotError::Corrupt(
            "the segment count exceeds the address space",
        ))?;
    let body = bytes.get(after..end).ok_or(SnapshotError::Truncated)?;
    let mut entries = Vec::with_capacity(n_segments as usize);
    for entry in body.chunks_exact(ENTRY) {
        let word = |at: usize| u64::from_le_bytes(entry[at..at + 8].try_into().expect("8 bytes"));
        let file = u32::from_le_bytes(entry[0..4].try_into().expect("4 bytes"));
        let (bytes, keys, postings) = (word(8), word(16), word(24));
        let min = u32::from_le_bytes(entry[32..36].try_into().expect("4 bytes"));
        let max = u32::from_le_bytes(entry[36..40].try_into().expect("4 bytes"));
        ck.word(bytes);
        ck.word(keys);
        ck.word(postings);
        ck.word(u64::from(file));
        ck.word(u64::from(min) | u64::from(max) << 32);
        if postings == 0 && (min != 0 || max != 0) {
            return Err(SnapshotError::Corrupt("a reference range without postings"));
        }
        if postings > 0 && min > max {
            return Err(SnapshotError::Corrupt("an inverted reference range"));
        }
        if postings > 0 && u64::from(max) >= reference_count {
            return Err(SnapshotError::Corrupt(
                "a segment references an id outside the id space",
            ));
        }
        entries.push(Entry {
            file,
            bytes,
            keys,
            postings,
            ids: (postings > 0).then_some((min, max)),
        });
    }
    let stored = bytes.get(end..end + 8).ok_or(SnapshotError::Truncated)?;
    if u64::from_le_bytes(stored.try_into().expect("8 bytes")) != ck.finish() {
        return Err(SnapshotError::Checksum);
    }
    let live = (0..n_refs)
        .map(|i| bitmap[i / 8] & (1 << (i % 8)) != 0)
        .collect();
    Ok(Manifest {
        generation,
        reference_count,
        next_file,
        live,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        let mut live = alloc::vec![true; 5];
        live[2] = false;
        Manifest {
            generation: 7,
            reference_count: 5,
            next_file: 9,
            live,
            entries: alloc::vec![
                Entry {
                    file: 3,
                    bytes: 1_000,
                    keys: 10,
                    postings: 50,
                    ids: Some((0, 4)),
                },
                Entry {
                    file: 8,
                    bytes: 32,
                    keys: 0,
                    postings: 0,
                    ids: None,
                },
            ],
        }
    }

    #[test]
    fn round_trips() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        assert_eq!(read(&bytes).unwrap(), sample());
        assert_eq!(bytes.len(), HEADER + 1 + 7 + 2 * ENTRY + 8);
    }

    #[test]
    fn refuses_a_truncated_or_flipped_file() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        for cut in [0, 4, 40, bytes.len() - 1] {
            assert!(
                matches!(read(&bytes[..cut]), Err(SnapshotError::Truncated)),
                "{cut} bytes"
            );
        }
        let mut flipped = bytes.clone();
        flipped[HEADER] ^= 1;
        assert!(matches!(read(&flipped), Err(SnapshotError::Checksum)));
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(read(&bad_magic), Err(SnapshotError::BadMagic)));
        let mut later = bytes;
        later[4] = 2;
        assert!(matches!(
            read(&later),
            Err(SnapshotError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn refuses_entries_that_do_not_fit_the_id_space() {
        let mut m = sample();
        m.entries[0].ids = Some((0, 5));
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        assert!(matches!(read(&bytes), Err(SnapshotError::Corrupt(m)) if m.contains("id space")));
        let mut m = sample();
        m.entries[0].ids = Some((4, 1));
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        assert!(matches!(read(&bytes), Err(SnapshotError::Corrupt(m)) if m.contains("inverted")));
        let mut m = sample();
        m.entries[1].ids = Some((0, 1));
        m.entries[1].postings = 0;
        let mut bytes = Vec::new();
        m.write(&mut bytes).unwrap();
        assert!(
            matches!(read(&bytes), Err(SnapshotError::Corrupt(m)) if m.contains("without postings"))
        );
    }

    #[test]
    fn names_segments_by_id() {
        assert_eq!(segment_name(0x2a), "seg-0000002a.afs");
        assert_eq!(temp_name(0x2a), "seg-0000002a.tmp");
        for id in [0, 0x2a, u32::MAX] {
            assert_eq!(segment_id(&segment_name(id)), Some(id));
            assert_eq!(segment_id(&temp_name(id)), None);
            assert!(is_temp(&temp_name(id)));
            assert!(!is_temp(&segment_name(id)));
        }
        assert!(is_temp(MANIFEST_TEMP));
        // Names a store never gives are not its files.
        for other in [
            "MANIFEST",
            "MANIFEST.bak",
            "notes.tmp",
            "seg-notes.afs",
            "seg-0000002A.afs",
            "seg-+000002a.afs",
            "seg-2a.afs",
            "seg-000000000000002a.afs",
            "seg-0000002a.afs.bak",
            "xseg-0000002a.afs",
        ] {
            assert_eq!(segment_id(other), None, "{other}");
            assert!(!is_temp(other), "{other}");
        }
    }

    #[test]
    fn reads_the_generation_from_the_head() {
        let mut bytes = Vec::new();
        sample().write(&mut bytes).unwrap();
        let head: [u8; GENERATION_END] = bytes[..GENERATION_END].try_into().unwrap();
        assert_eq!(generation(&head), Some(7));
        let mut other = head;
        other[0] = b'X';
        assert_eq!(generation(&other), None);
    }
}
