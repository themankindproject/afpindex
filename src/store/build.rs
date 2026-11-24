//! A bulk build into a catalog directory in bounded memory.

use alloc::vec::Vec;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::{CatalogStore, Faults, lock_dir, publish, sweep, sync_file, write_file};
use crate::catalog::Part;
use crate::manifest::{self, MANIFEST};
use crate::segment::to_file::{Settings, merge_to_file};
use crate::snapshot::map_segment;
use crate::{Catalog, Posting, RefCoding, Segment, SnapshotError};

/// Rows a builder holds before it sorts them into a chunk file: 16 M, a
/// few hundred megabytes while they are sorted.
const CHUNK: usize = 1 << 24;

/// Chunk files one merge joins. A merge holds the directory of every
/// source it reads, so this multiplies the memory of a merge.
const FAN_IN: usize = 8;

/// The keep of a merge that drops nothing.
type KeepAll = fn(u32) -> bool;

/// Builds a catalog directory from more postings than fit in memory.
///
/// Ids are assigned in call order from `0`, as [`crate::CatalogBuilder`]
/// assigns them. Rows are collected until a chunk's worth is held; the
/// chunk is then sorted into a segment, written to a temporary file in the
/// directory and freed, so memory peaks at one chunk however large the
/// catalog. [`CatalogStoreBuilder::finish`] merges the chunk files
/// eight at a time, each merge written straight to a file, until one
/// segment is left, and saves it as the catalog of the directory. The
/// result is the catalog [`crate::CatalogBuilder`] builds from the same
/// calls.
///
/// A builder that fails or is dropped before `finish` returns removes its
/// files and leaves no manifest, so the build can start again in the same
/// directory.
///
/// # Memory
///
/// While references are added, one chunk's rows (12 bytes a posting) and
/// then the segment sorted from them. While chunks are merged, each merge
/// holds the keys and the directory of every file it joins, about 32
/// bytes per distinct hash of each, beside the pieces in flight. Neither
/// grows with the number of postings.
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
#[derive(Debug)]
pub struct CatalogStoreBuilder {
    dir: PathBuf,
    rows: Vec<Posting>,
    chunk: usize,
    next: u32,
    /// The files of the chunks written so far, in order.
    chunks: Vec<u32>,
    next_file: u32,
    /// How the chunks and their merges store the references.
    coding: RefCoding,
    settings: Settings,
    faults: Faults,
    /// The directory's lock, handed to the store `finish` returns.
    lock: Option<std::fs::File>,
}

impl CatalogStore {
    /// Start a bulk build of a catalog in `dir`, making the directory if it
    /// does not exist.
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::create`]: an I/O error, `AlreadyExists` when a
    /// manifest is already in `dir`, and `WouldBlock` when a store or
    /// another builder has the directory.
    pub fn builder(dir: impl AsRef<Path>) -> Result<CatalogStoreBuilder, SnapshotError> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let lock = lock_dir(dir)?;
        if dir.join(MANIFEST).try_exists()? {
            return Err(SnapshotError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "a catalog manifest is already in this directory",
            )));
        }
        sweep(dir, &[]);
        Ok(CatalogStoreBuilder {
            dir: dir.to_path_buf(),
            rows: Vec::new(),
            chunk: CHUNK,
            next: 0,
            chunks: Vec::new(),
            next_file: 0,
            coding: RefCoding::default(),
            settings: Settings::default(),
            faults: Faults::default(),
            lock: Some(lock),
        })
    }
}

impl CatalogStoreBuilder {
    /// Set the rows held before they are sorted into a chunk file;
    /// 2<sup>24</sup> unless set. A reference's rows always go into one
    /// chunk, so a chunk can hold more.
    pub fn chunk_postings(&mut self, postings: usize) -> &mut Self {
        self.chunk = postings.max(1);
        self
    }

    /// Store the references as `coding` says (see [`RefCoding`]), in the
    /// segment the build makes and in the catalog's later merges; packed
    /// unless set.
    pub fn ref_coding(&mut self, coding: RefCoding) -> &mut Self {
        self.coding = coding;
        self
    }

    /// Add a reference from its `(hash, time)` rows and return its id.
    ///
    /// # Errors
    ///
    /// Any error of writing a chunk file. The builder is then of no further
    /// use: drop it, which removes its files.
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    pub fn add(&mut self, rows: impl IntoIterator<Item = (u32, u32)>) -> io::Result<u32> {
        let id = self.next;
        assert!(id < u32::MAX, "reference ids exhausted");
        self.next = id + 1;
        if self.rows.capacity() == 0 {
            // A chunk's rows and the slack of the reference that ends it,
            // so that the rows do not grow to twice a chunk by doubling.
            self.rows
                .reserve_exact(self.chunk.saturating_add(self.chunk / 8).min(1 << 30));
        }
        self.rows
            .extend(rows.into_iter().map(|(hash, time)| Posting {
                hash,
                reference: id,
                time,
            }));
        if self.rows.len() >= self.chunk {
            self.write_chunk()?;
        }
        Ok(id)
    }

    /// Sort the rows held into a segment and write it to a chunk file.
    fn write_chunk(&mut self) -> io::Result<()> {
        let segment = Segment::from_postings_in(core::mem::take(&mut self.rows), self.coding);
        if segment.is_empty() {
            return Ok(());
        }
        let file = self.take_file();
        let path = self.temp(file);
        // A chunk is merged and deleted, or synced in `finish` when it is
        // the only one, so it is written without a sync of its own.
        write_file(&mut self.faults, &path, false, |out| segment.write_to(out))?;
        self.chunks.push(file);
        Ok(())
    }

    /// Write the last chunk, merge the chunks into one segment and save it
    /// as the catalog of the directory.
    ///
    /// # Errors
    ///
    /// Any error of writing, merging or saving. Nothing is left in the
    /// directory then: no manifest and none of the builder's files.
    pub fn finish(mut self) -> io::Result<CatalogStore> {
        self.write_chunk()?;
        while self.chunks.len() > 1 {
            let level = core::mem::take(&mut self.chunks);
            let mut rest = level.as_slice();
            while !rest.is_empty() {
                let (group, after) = rest.split_at(FAN_IN.min(rest.len()));
                match self.merge(group) {
                    Ok(file) => self.chunks.push(file),
                    Err(e) => {
                        // What is not merged yet goes when the builder does.
                        self.chunks.extend_from_slice(rest);
                        return Err(e);
                    }
                }
                rest = after;
            }
        }
        let n = self.next as usize;
        let mut parts = Vec::new();
        if let Some(&file) = self.chunks.first() {
            let name = self.dir.join(manifest::segment_name(file));
            // The chunks and the merges between them were written without
            // a sync; the one file the manifest will name is synced here.
            sync_file(&self.temp(file))?;
            publish(&self.temp(file), &name)?;
            self.chunks.clear();
            let mapped = map_segment(&name, true, false).map_err(|e| {
                let _ = fs::remove_file(&name);
                io::Error::other(e)
            });
            let (segment, ids) = mapped?;
            if let Some(ids) = ids {
                parts.push(Part::new(segment, ids).with_file(file));
            }
        }
        let catalog = Catalog::from_files(parts, alloc::vec![true; n], self.coding);
        let mut store =
            CatalogStore::new(&self.dir, catalog, None, self.next_file, self.lock.take());
        store.settings = self.settings;
        store.faults = core::mem::take(&mut self.faults);
        if let Err(e) = store.save() {
            // Before the manifest is in place the segment is no catalog's;
            // after it, it is this directory's catalog and stays.
            if store.on_disk.is_some() {
                return Err(e);
            }
            for file in files_of(&store) {
                let _ = fs::remove_file(self.dir.join(manifest::segment_name(file)));
            }
            return Err(e);
        }
        Ok(store)
    }

    /// Merge the chunk files `group` into one, and remove them.
    fn merge(&mut self, group: &[u32]) -> io::Result<u32> {
        if let [only] = group {
            return Ok(*only);
        }
        let mut segments = Vec::with_capacity(group.len());
        for &file in group {
            let (segment, _) =
                map_segment(&self.temp(file), true, false).map_err(io::Error::other)?;
            segments.push(segment);
        }
        let sides: Vec<(&Segment, Option<&KeepAll>)> =
            segments.iter().map(|segment| (segment, None)).collect();
        self.faults.create()?;
        let settings = Settings {
            fail_after: self.faults.write().or(self.settings.fail_after),
            sync: false,
            ..self.settings
        };
        let file = self.take_file();
        merge_to_file(&sides, &self.temp(file), self.coding, settings)?;
        drop(sides);
        drop(segments);
        for &input in group {
            let _ = fs::remove_file(self.temp(input));
        }
        Ok(file)
    }

    fn take_file(&mut self) -> u32 {
        let file = self.next_file;
        self.next_file += 1;
        file
    }

    fn temp(&self, file: u32) -> PathBuf {
        self.dir.join(manifest::temp_name(file))
    }

    /// Merge with `threads` threads instead of what the machine offers.
    #[cfg(test)]
    pub(crate) fn with_threads(mut self, threads: usize) -> Self {
        self.settings.threads = threads;
        self
    }

    /// Make the next file written fail after `bytes`.
    #[cfg(test)]
    pub(crate) fn fail_next_write(&mut self, bytes: u64) {
        self.faults.write = Some(bytes);
    }

    /// Make the next creation of a file fail.
    #[cfg(test)]
    pub(crate) fn fail_next_create(&mut self) {
        self.faults.create = true;
    }
}

impl Drop for CatalogStoreBuilder {
    fn drop(&mut self) {
        for &file in &self.chunks {
            let _ = fs::remove_file(self.temp(file));
        }
    }
}

/// The segment files of a store's catalog.
fn files_of(store: &CatalogStore) -> Vec<u32> {
    super::files_of(&store.catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Stop;
    use crate::{CatalogBuilder, Query, Scratch, SearchOptions};
    use alloc::format;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("afpindex-build-{name}-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// References of random rows, some with none and some with rows
    /// repeated.
    fn references(rng: &mut Rng, n: usize) -> Vec<Vec<(u32, u32)>> {
        (0..n)
            .map(|_| {
                let len = match rng.below(10) {
                    0 => 0,
                    _ => 1 + rng.below(40) as usize,
                };
                let mut rows: Vec<(u32, u32)> = (0..len)
                    .map(|_| (rng.below(300) as u32, rng.below(200) as u32))
                    .collect();
                if len > 0 && rng.below(4) == 0 {
                    rows.push(rows[0]);
                }
                rows
            })
            .collect()
    }

    fn files(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    // The lock file is the store's, not the catalog's.
                    .filter(|name| name != crate::manifest::LOCK)
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    fn assert_same(got: &Catalog, want: &Catalog, what: &str) {
        assert_eq!(got.reference_count(), want.reference_count(), "{what}: ids");
        assert_eq!(got.live_count(), want.live_count(), "{what}: live");
        assert_eq!(
            got.segment_count(),
            want.segment_count(),
            "{what}: segments"
        );
        for (a, b) in got.segments().zip(want.segments()) {
            assert_eq!(a, b, "{what}: the segment");
        }
    }

    /// One reference per chunk gives more than 8 × 8 chunks, so the
    /// merges run three levels deep; the catalog is the one
    /// `CatalogBuilder` builds, on one thread and on several, and the
    /// directory opens as it.
    #[test]
    fn builds_what_the_catalog_builder_builds() {
        let mut rng = Rng(0xB01D_5EED);
        let refs = references(&mut rng, 300);
        let mut plain = CatalogBuilder::default();
        for rows in &refs {
            plain.add(rows.iter().copied());
        }
        let want = plain.build();
        let query = Query::from_pairs(
            refs[7].iter().copied().chain(refs[200].iter().copied()),
            Some(64),
        );
        let options = SearchOptions::default();
        let found = want.search(&query, &options, &mut Scratch::new()).unwrap();
        let scratch = TempDir::new("same");
        for threads in [1usize, 4] {
            let dir = scratch.0.join(format!("threads-{threads}"));
            let mut builder = CatalogStore::builder(&dir).unwrap().with_threads(threads);
            builder.chunk_postings(1);
            for (i, rows) in refs.iter().enumerate() {
                assert_eq!(builder.add(rows.iter().copied()).unwrap(), i as u32);
            }
            let store = builder.finish().unwrap();
            assert_same(store.catalog(), &want, "built");
            let got = store
                .catalog()
                .search(&query, &options, &mut Scratch::new())
                .unwrap();
            assert_eq!(got, found, "{threads} threads: candidates");
            assert_eq!(files(&dir).len(), 2, "the manifest and one segment");
            drop(store);
            assert_same(&Catalog::map_dir(&dir).unwrap(), &want, "mapped");
            assert_same(CatalogStore::open(&dir).unwrap().catalog(), &want, "opened");
        }
    }

    /// No reference, references without rows, and a chunk larger than the
    /// whole build.
    #[test]
    fn empty_and_small_builds() {
        let scratch = TempDir::new("empty");
        let dir = scratch.0.join("none");
        let store = CatalogStore::builder(&dir).unwrap().finish().unwrap();
        assert_eq!(store.catalog().reference_count(), 0);
        assert_eq!(files(&dir), ["MANIFEST"]);

        let dir = scratch.0.join("no-rows");
        let mut builder = CatalogStore::builder(&dir).unwrap();
        for _ in 0..5 {
            builder.add([]).unwrap();
        }
        let store = builder.finish().unwrap();
        assert_eq!(store.catalog().reference_count(), 5);
        assert_eq!(store.catalog().live_count(), 5);
        assert_eq!(store.catalog().segment_count(), 0);
        drop(store);
        assert_eq!(
            CatalogStore::open(&dir)
                .unwrap()
                .catalog()
                .reference_count(),
            5
        );

        let dir = scratch.0.join("one-chunk");
        let mut rng = Rng(0x0C40);
        let refs = references(&mut rng, 30);
        let mut builder = CatalogStore::builder(&dir).unwrap();
        let mut plain = CatalogBuilder::default();
        for rows in &refs {
            builder.add(rows.iter().copied()).unwrap();
            plain.add(rows.iter().copied());
        }
        assert_same(
            builder.finish().unwrap().catalog(),
            &plain.build(),
            "one chunk",
        );
        // A directory with a catalog is refused.
        assert!(CatalogStore::builder(&dir).is_err());
    }

    /// A failure while a chunk is written, while chunks are merged, or
    /// while the manifest is written leaves an empty directory, and a
    /// build in it starts again; so does a builder dropped half-way.
    #[test]
    fn a_failed_build_leaves_nothing() {
        let mut rng = Rng(0xFA11_B01D);
        let refs = references(&mut rng, 40);
        let scratch = TempDir::new("failed");
        let dir = scratch.0.join("store");
        for fault in 0..5 {
            let mut builder = CatalogStore::builder(&dir).unwrap();
            builder.chunk_postings(1);
            let mut failed = None;
            for (i, rows) in refs.iter().enumerate() {
                if fault == 0 && i == 20 {
                    builder.fail_next_write(16);
                }
                if let Err(e) = builder.add(rows.iter().copied()) {
                    failed = Some(e);
                    break;
                }
            }
            match fault {
                0 => assert!(failed.is_some(), "a chunk write fails"),
                1 => builder.fail_next_write(64),
                2 => builder.fail_next_create(),
                3 => builder.faults.stop = Some(Stop::AfterTempManifest),
                _ => {}
            }
            if fault == 0 || fault == 4 {
                drop(builder);
            } else {
                assert!(builder.finish().is_err(), "fault {fault}");
            }
            assert_eq!(files(&dir), Vec::<String>::new(), "fault {fault}");
        }
        let mut builder = CatalogStore::builder(&dir).unwrap();
        builder.chunk_postings(1);
        let mut plain = CatalogBuilder::default();
        for rows in &refs {
            builder.add(rows.iter().copied()).unwrap();
            plain.add(rows.iter().copied());
        }
        assert_same(builder.finish().unwrap().catalog(), &plain.build(), "again");
    }
}
