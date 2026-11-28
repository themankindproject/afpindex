//! Copies of a catalog directory (`mmap`): [`sync`] brings a replica's
//! directory up to a source directory, and [`prune`] removes the segment files
//! its manifest no longer names.
//!
//! A replica serves searches from its own directory, mapped with
//! [`Catalog::map_dir_trusted`](crate::Catalog::map_dir_trusted), and refreshes
//! it from a directory the writer's [`CatalogStore`](crate::CatalogStore) saves
//! to (a shared volume, or a local copy of what an object store holds):
//!
//! ```no_run
//! # fn main() -> Result<(), hashtally::SnapshotError> {
//! use hashtally::{Catalog, replica};
//!
//! let synced = replica::sync("/mnt/primary/catalog", "/var/lib/replica/catalog")?;
//! if synced.manifest_replaced {
//!     // Every file was checked as it was copied.
//!     let catalog = Catalog::map_dir_trusted("/var/lib/replica/catalog")?;
//!     catalog.warm_references();
//!     // ... swap `catalog` in for searches, drop the old one, then:
//!     replica::prune("/var/lib/replica/catalog")?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! A replica's directory is an ordinary catalog directory: promoting a
//! replica to writer is [`CatalogStore::open`](crate::CatalogStore::open) on it.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::SnapshotError;
use crate::manifest::{self, MANIFEST, MANIFEST_TEMP, Manifest, ManifestEntry};
use crate::snapshot::{FOOTER, map_segment};
use crate::store::{lock_dir, sweep, sync_dir, sync_file};

/// Times a sync starts over because a segment file the source's manifest
/// names was gone: the source's writer saved in between.
const RETRIES: usize = 8;

/// What [`sync`] did.
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Synced {
    /// The generation of the manifest the replica now holds, the source's.
    pub generation: u64,
    /// Segment files copied; the others were in the replica already.
    pub files_copied: usize,
    /// Bytes of the segment files copied.
    pub bytes_copied: u64,
    /// `false` when the replica's manifest was already the source's, byte
    /// for byte: the catalog did not change, and nothing need be reloaded.
    pub manifest_replaced: bool,
}

/// Bring the catalog directory `dst` up to the catalog of `src`, making
/// `dst` if it does not exist.
///
/// The manifest of `src` is read, every segment file it names that `dst`
/// lacks is copied, and the manifest is written last, through a temporary
/// name and a rename. A process that maps `dst` meanwhile sees the
/// catalog before the sync or the one after it, never a mix:
///
/// 1. Each segment file is copied to a temporary name, synced, mapped and
///    checked with every check of
///    [`Segment::map_from`](crate::Segment::map_from) and against its
///    manifest entry, and only then renamed into place. A replica can
///    therefore map `dst` with
///    [`Catalog::map_dir_trusted`](crate::Catalog::map_dir_trusted).
/// 2. A file already in `dst` is kept when its length and its footer (the
///    checksums of its contents) are the source file's, and copied again
///    otherwise. A source directory created again from nothing gives its
///    files the same names as before with other contents; when such a
///    file must be replaced, `dst`'s manifest is removed first, so that no
///    reader pairs the old manifest with a new file. Until the sync
///    finishes, `dst` then holds no catalog.
/// 3. The directory is synced, then the manifest is copied and renamed
///    over `dst`'s, and the directory is synced again (on Unix).
///
/// Nothing is removed: the files the new manifest no longer names stay
/// until [`prune`], since a catalog mapped from the old manifest may still
/// be serving searches.
///
/// The source may be a directory its writer is saving to. A segment file
/// that its manifest named and the writer removed before the copy makes
/// the sync start over from the newer manifest, a few times. The manifest
/// copied is always the source's, so a source restored from an older
/// backup takes `dst` back with it.
///
/// `sync` holds `dst`'s lock while it runs, as a
/// [`CatalogStore`](crate::CatalogStore) does: it fails rather than write
/// a directory that a store or another sync is writing.
///
/// # Errors
///
/// [`SnapshotError::Io`] for any error reading `src` or writing `dst`, a
/// `WouldBlock` error when another store or sync holds `dst`'s lock, and
/// the errors of [`Manifest::from_bytes`] and
/// [`Segment::map_from`](crate::Segment::map_from) for a manifest or a
/// segment file of `src` that does not read. A sync that fails leaves the
/// catalog of `dst` as it was, or none if it had begun to replace files of
/// a recreated source; the files it finished stay for the next sync.
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
pub fn sync(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<Synced, SnapshotError> {
    let (src, dst) = (src.as_ref(), dst.as_ref());
    fs::create_dir_all(dst)?;
    let _lock = lock_dir(dst)?;
    for _ in 0..RETRIES {
        let bytes = fs::read(src.join(MANIFEST))?;
        let manifest = manifest::read(&bytes)?;
        let Some((files_copied, bytes_copied)) = copy_segments(src, dst, &manifest)? else {
            continue;
        };
        // The renames of the segment files are made durable before a
        // manifest names them.
        sync_dir(dst)?;
        let manifest_replaced = put_manifest(dst, &bytes)?;
        return Ok(Synced {
            generation: manifest.generation(),
            files_copied,
            bytes_copied,
            manifest_replaced,
        });
    }
    Err(SnapshotError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        "a segment file the source's manifest names is missing",
    )))
}

/// Remove from the catalog directory `dir` every segment file its manifest
/// does not name, and every temporary file a crash left. Returns the
/// number of files removed.
///
/// Call it once no catalog mapped from an older manifest of `dir` is
/// alive. On Unix a removed file stays readable through a mapping that
/// holds it, so a search on such a catalog still finishes; on Windows a
/// mapped file cannot be removed, and stays for the next call. Files of any
/// other name are left alone.
///
/// A [`CatalogStore`](crate::CatalogStore) prunes its own directory at
/// every save; `prune` is for the directories [`sync`] writes. It holds
/// `dir`'s lock while it runs.
///
/// # Errors
///
/// [`SnapshotError::Io`] when `dir` has no manifest (`NotFound`: nothing
/// is removed from a directory without a catalog) or cannot be read, a
/// `WouldBlock` error when a store or a sync holds the lock, and the errors
/// of [`Manifest::from_bytes`] for a manifest that does not read.
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
pub fn prune(dir: impl AsRef<Path>) -> Result<usize, SnapshotError> {
    let dir = dir.as_ref();
    let _lock = lock_dir(dir)?;
    let manifest = Manifest::read(dir)?;
    let keep: alloc::vec::Vec<u32> = manifest.entries().iter().map(|e| e.file).collect();
    Ok(sweep(dir, &keep))
}

/// Copy into `dst` every segment file of `manifest` that `dst` lacks, and
/// return how many files and bytes were copied. `None` when a source file
/// is gone: the source's writer saved after its manifest was read.
fn copy_segments(
    src: &Path,
    dst: &Path,
    manifest: &Manifest,
) -> Result<Option<(usize, u64)>, SnapshotError> {
    let (mut files, mut bytes) = (0, 0);
    let mut manifest_removed = false;
    for entry in manifest.entries() {
        let name = entry.file_name();
        let from = src.join(&name);
        let Some(source_tail) = gone_as_none(tail(&from, entry.bytes))? else {
            return Ok(None);
        };
        let to = dst.join(&name);
        let held = match fs::metadata(&to) {
            Ok(meta) => Some(meta.len() == entry.bytes && tail(&to, entry.bytes)? == source_tail),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        match held {
            Some(true) => continue,
            // A file of this name with other contents: the source was
            // created again. The manifest that names the old file goes
            // before the file does.
            Some(false) if !manifest_removed => {
                remove_manifest(dst)?;
                manifest_removed = true;
            }
            _ => {}
        }
        if copy_segment(&from, dst, &name, entry)?.is_none() {
            return Ok(None);
        }
        files += 1;
        bytes += entry.bytes;
    }
    Ok(Some((files, bytes)))
}

/// Copy the segment file `from` to `dst` under a temporary name, check it
/// fully and against `entry`, and rename it to `name`. `None` when the
/// source file is gone.
fn copy_segment(
    from: &Path,
    dst: &Path,
    name: &str,
    entry: &ManifestEntry,
) -> Result<Option<()>, SnapshotError> {
    let temp = dst.join(manifest::temp_name(entry.file));
    let checked = (|| {
        let Some(copied) = gone_as_none(fs::copy(from, &temp))? else {
            return Ok(None);
        };
        if copied != entry.bytes {
            return Err(SnapshotError::Corrupt(
                "a segment file is not the length its manifest entry declares",
            ));
        }
        sync_file(&temp)?;
        // The mapping goes before the rename, which Windows refuses while
        // a file is mapped.
        let (segment, ids) = map_segment(&temp, false)?;
        let fits = segment.key_count() as u64 == entry.keys
            && segment.len() as u64 == entry.postings
            && ids == entry.ids;
        if !fits {
            return Err(SnapshotError::Corrupt(
                "a segment does not match its manifest entry",
            ));
        }
        Ok(Some(()))
    })();
    let renamed = match checked {
        Ok(Some(())) => fs::rename(&temp, dst.join(name))
            .map(Some)
            .map_err(Into::into),
        other => other,
    };
    if !matches!(renamed, Ok(Some(()))) {
        let _ = fs::remove_file(&temp);
    }
    renamed
}

/// Write `bytes` as the manifest of `dst` unless it is that already, and
/// say whether it was written.
fn put_manifest(dst: &Path, bytes: &[u8]) -> io::Result<bool> {
    match fs::read(dst.join(MANIFEST)) {
        Ok(held) if held == bytes => return Ok(false),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let temp = dst.join(MANIFEST_TEMP);
    let written = fs::write(&temp, bytes)
        .and_then(|()| sync_file(&temp))
        .and_then(|()| fs::rename(&temp, dst.join(MANIFEST)));
    if let Err(e) = written {
        let _ = fs::remove_file(&temp);
        return Err(e);
    }
    sync_dir(dst)?;
    Ok(true)
}

/// Remove the manifest of `dst`, if it has one, durably.
fn remove_manifest(dst: &Path) -> io::Result<()> {
    match fs::remove_file(dst.join(MANIFEST)) {
        Ok(()) => sync_dir(dst),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// The last bytes of the file `path` of length `len`: its footer, which
/// holds the checksums of everything before it.
fn tail(path: &Path, len: u64) -> io::Result<alloc::vec::Vec<u8>> {
    let mut file = fs::File::open(path)?;
    let n = len.min(FOOTER as u64);
    file.seek(SeekFrom::Start(len - n))?;
    let mut out = alloc::vec![0u8; n as usize];
    file.read_exact(&mut out)?;
    Ok(out)
}

/// `None` for an error that says the file is not there.
fn gone_as_none<T>(result: io::Result<T>) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{Rng, TempDir};
    use crate::{Catalog, CatalogStore, Posting};
    use alloc::string::String;
    use alloc::vec::Vec;
    use std::path::PathBuf;

    fn rows(rng: &mut Rng) -> Vec<(u32, u32)> {
        (0..20 + rng.below(80))
            .map(|_| (rng.below(70) as u32, rng.below(400) as u32))
            .collect()
    }

    /// Everything a search can see of a catalog.
    fn state(catalog: &Catalog) -> (Vec<Vec<Posting>>, Vec<bool>) {
        (
            catalog.segments().map(|s| s.postings().collect()).collect(),
            (0..catalog.reference_count() as u32)
                .map(|id| catalog.is_live(id))
                .collect(),
        )
    }

    /// The segment files of `dir`, by name.
    fn segment_files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("the directory")
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| manifest::segment_id(n).is_some())
            .collect();
        names.sort();
        names
    }

    fn dirs(name: &str) -> (TempDir, PathBuf, PathBuf) {
        let scratch = TempDir::new(name);
        let (src, dst) = (scratch.0.join("src"), scratch.0.join("dst"));
        (scratch, src, dst)
    }

    /// A first sync copies every file; one with nothing new copies nothing
    /// and leaves the manifest; one after a save copies only the new files,
    /// and `prune` then leaves the source's files exactly.
    #[test]
    fn copies_a_directory_and_then_only_what_changed() {
        let mut rng = Rng(0xC0FF_EE15);
        let (_scratch, src, dst) = dirs("replica-sync");
        let mut store = CatalogStore::create(&src).expect("a store");
        store.set_spill_postings(0);
        for _ in 0..12 {
            store.add(rows(&mut rng)).expect("an addition");
        }
        store.save().expect("a save");
        let first = sync(&src, &dst).expect("a sync");
        let files = segment_files(&src);
        assert_eq!(first.files_copied, files.len());
        assert!(first.manifest_replaced);
        assert_eq!(first.generation, 1);
        assert_eq!(segment_files(&dst), files);
        let mapped = Catalog::map_dir_trusted(&dst).expect("the replica");
        assert_eq!(state(&mapped), state(store.catalog()));

        let again = sync(&src, &dst).expect("a second sync");
        assert_eq!((again.files_copied, again.bytes_copied), (0, 0));
        assert!(!again.manifest_replaced);

        for _ in 0..5 {
            store.add(rows(&mut rng)).expect("an addition");
        }
        store.remove(3);
        store.compact_deltas().expect("a compaction");
        store.save().expect("a save");
        let before = segment_files(&dst);
        let after = sync(&src, &dst).expect("a third sync");
        assert!(after.manifest_replaced);
        assert_eq!(after.generation, 2);
        let new: Vec<_> = segment_files(&src)
            .into_iter()
            .filter(|n| !before.contains(n))
            .collect();
        assert_eq!(after.files_copied, new.len());
        let mapped = Catalog::map_dir(&dst).expect("the replica");
        assert_eq!(state(&mapped), state(store.catalog()));
        // The old files stay until a prune, which removes only those.
        assert!(segment_files(&dst).len() >= segment_files(&src).len());
        let stale = segment_files(&dst).len() - segment_files(&src).len();
        assert_eq!(prune(&dst).expect("a prune"), stale);
        assert_eq!(segment_files(&dst), segment_files(&src));
        assert_eq!(
            state(&Catalog::map_dir(&dst).expect("the replica")),
            state(&mapped)
        );
    }

    /// A source created again from nothing names its files as the first one
    /// did; the replica copies them again and holds the new catalog.
    #[test]
    fn copies_a_recreated_source_again() {
        let mut rng = Rng(0x5EED_0001);
        let (_scratch, src, dst) = dirs("replica-recreated");
        let mut store = CatalogStore::create(&src).expect("a store");
        let batch: Vec<_> = (0..6).map(|_| rows(&mut rng)).collect();
        store.add_many(batch).expect("a batch");
        store.save().expect("a save");
        drop(store);
        sync(&src, &dst).expect("a sync");
        let old = segment_files(&dst);

        fs::remove_dir_all(&src).expect("the source gone");
        let mut store = CatalogStore::create(&src).expect("a new store");
        let batch: Vec<_> = (0..6).map(|_| rows(&mut rng)).collect();
        store.add_many(batch).expect("a batch");
        store.save().expect("a save");
        assert_eq!(segment_files(&src), old, "the same names");
        let synced = sync(&src, &dst).expect("a sync");
        assert_eq!(synced.files_copied, old.len());
        assert!(synced.manifest_replaced);
        let mapped = Catalog::map_dir(&dst).expect("the replica");
        assert_eq!(state(&mapped), state(store.catalog()));
    }

    /// A damaged segment file of the source is refused before it gets its
    /// name, and the replica keeps the catalog it had.
    #[test]
    fn refuses_a_damaged_source_file() {
        let mut rng = Rng(0xBAD5_EED5);
        let (_scratch, src, dst) = dirs("replica-damaged");
        let mut store = CatalogStore::create(&src).expect("a store");
        store.add(rows(&mut rng)).expect("an addition");
        store.save().expect("a save");
        sync(&src, &dst).expect("a sync");
        let had = state(&Catalog::map_dir(&dst).expect("the replica"));

        store.add(rows(&mut rng)).expect("an addition");
        store.save().expect("a save");
        let held = segment_files(&dst);
        let new = segment_files(&src)
            .into_iter()
            .find(|n| !held.contains(n))
            .expect("a new segment file");
        let path = src.join(&new);
        let mut bytes = fs::read(&path).expect("the file");
        let at = bytes.len() / 2;
        bytes[at] ^= 0x40;
        fs::write(&path, &bytes).expect("a rewrite");

        assert!(sync(&src, &dst).is_err());
        assert_eq!(segment_files(&dst), held, "nothing half-copied");
        assert!(!dst.join(MANIFEST_TEMP).exists());
        assert_eq!(state(&Catalog::map_dir(&dst).expect("the replica")), had);
    }

    /// A store holding a directory keeps a sync and a prune out of it, and a
    /// prune removes nothing from a directory without a manifest.
    #[test]
    fn keeps_out_of_a_locked_or_empty_directory() {
        let (_scratch, src, dst) = dirs("replica-locked");
        let mut store = CatalogStore::create(&src).expect("a store");
        store.add([(1, 2), (3, 4)]).expect("an addition");
        store.save().expect("a save");
        let held = CatalogStore::create(&dst).expect("a store on the replica");
        let busy = |e: SnapshotError| matches!(e, SnapshotError::Io(e) if e.kind() == io::ErrorKind::WouldBlock);
        assert!(busy(sync(&src, &dst).unwrap_err()));
        assert!(busy(prune(&dst).unwrap_err()));
        drop(held);

        let stray = dst.join(manifest::segment_name(0));
        fs::write(&stray, b"not yet named").expect("a stray file");
        assert!(matches!(
            prune(&dst),
            Err(SnapshotError::Io(e)) if e.kind() == io::ErrorKind::NotFound
        ));
        assert!(stray.exists());
    }

    /// A replica's directory is a catalog directory: a store opens it and
    /// takes over as its writer.
    #[test]
    fn a_replica_can_be_promoted() {
        let mut rng = Rng(0x0123_4567);
        let (_scratch, src, dst) = dirs("replica-promote");
        let mut store = CatalogStore::create(&src).expect("a store");
        for _ in 0..4 {
            store.add(rows(&mut rng)).expect("an addition");
        }
        store.save().expect("a save");
        sync(&src, &dst).expect("a sync");
        let mut promoted = CatalogStore::open(&dst).expect("the replica opened");
        assert_eq!(state(promoted.catalog()), state(store.catalog()));
        let id = promoted.add(rows(&mut rng)).expect("an addition");
        assert_eq!(id, 4);
        promoted.save().expect("a save");
        assert_eq!(Manifest::read(&dst).expect("a manifest").generation(), 2);
    }
}
