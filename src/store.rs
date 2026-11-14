//! A catalog kept in a directory (`mmap`): [`CatalogStore`], its one
//! writer, and [`Catalog::map_dir`] for everyone who only reads it. The
//! files and their order of writing are described on the store; the
//! manifest's layout is in [`crate::manifest`].

use alloc::vec::Vec;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::catalog::{MergeParts, NO_FILE, Part};
use crate::manifest::{self, Entry, MANIFEST, Manifest};
use crate::segment::to_file::{Settings, merge_to_file};
use crate::snapshot::map_segment;
use crate::{Catalog, Segment, SnapshotError};

mod build;
pub use build::CatalogStoreBuilder;

/// Postings of input from which a merge is written to a file rather than
/// built on the heap: 64 M, about 0.3 GiB of postings at the widths of a
/// real catalog.
const SPILL: usize = 1 << 26;

/// Bytes gathered in the writer of a segment file.
const BUFFER: usize = 1 << 23;

/// Times a read of a directory starts over because a segment file that its
/// manifest names was gone: a writer saved in between.
const RETRIES: usize = 8;

/// The one writer of a catalog directory.
///
/// A catalog of billions of postings does not fit in one process's heap.
/// It is a directory of segment files, each a segment snapshot that never
/// changes once it has a name, and a manifest that says which of them make
/// the catalog:
///
/// ```text
/// <dir>/MANIFEST           the segments, the id space, the removals
/// <dir>/seg-<8 hex>.afs    one segment snapshot per segment, immutable
/// <dir>/seg-<8 hex>.tmp    a segment being written
/// <dir>/MANIFEST.tmp       a manifest being written
/// ```
///
/// A store holds the catalog in memory as a plain [`Catalog`] does and
/// decides, merge by merge, whether the result goes to the heap or
/// straight into a new segment file. A merge whose inputs hold at least
/// [`CatalogStore::spill_postings`] postings is written by a merge that
/// never holds its result and is mapped back, so a large segment's
/// postings live in its file's pages and never on the heap. A smaller one
/// is merged in memory and stays there until a save writes it.
///
/// [`CatalogStore::save`] is the durable point: it writes every segment
/// that has no file yet, then the manifest, and only then removes the
/// segment files the new manifest does not name. A crash before the
/// manifest is replaced leaves the previous catalog, so after a crash the
/// directory opens as the catalog of the last `save` that returned. What
/// was added or removed since then is lost, and the id the next reference
/// gets is [`Catalog::reference_count`] of that catalog, so a caller
/// replays its own log of additions from there.
///
/// [`Catalog::map_dir`] reads such a directory without writing to it.
///
/// # Memory
///
/// The heap holds the segments below the spill threshold that no save has
/// written yet, the keys and the directory of every segment, and a
/// merge's pieces in flight. A merge to a file holds its pieces rather
/// than its result; in proportion to the result it holds 13 bytes per
/// distinct hash (the keys, the start of each run and its multiplicity).
/// A piece of a merge carries whole runs, so such a merge is also bounded
/// from below by the longest run among its inputs.
///
/// # One writer
///
/// A store is the only writer of its directory, and the directory is the
/// store's: [`CatalogStore::open`] and [`CatalogStore::save`] remove the
/// segment and temporary files (the names above) that the manifest does
/// not name, and leave files of any other name alone. Two things stand in
/// the way of a second writer, neither of them a lock: `save` refuses to
/// replace a manifest whose generation is not the one this store last read
/// or wrote, and no segment file is ever written over another.
#[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
#[derive(Debug)]
pub struct CatalogStore {
    dir: PathBuf,
    catalog: Catalog,
    /// The generation of the manifest on disk, as this store last read or
    /// wrote it; `None` before the first `save` of a new directory.
    on_disk: Option<u64>,
    /// The id the next segment file gets; ids are never reused.
    next_file: u32,
    /// The spill threshold, in postings of a merge's inputs.
    spill: usize,
    /// How a merge to a file runs.
    settings: Settings,
    /// Failures a test asks for.
    faults: Faults,
}

impl CatalogStore {
    /// Create the store of an empty catalog in `dir`, making the directory
    /// if it does not exist.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::Io`] when the directory cannot be made or read, and
    /// an `AlreadyExists` error when a manifest is already in `dir`:
    /// taking over a catalog is [`CatalogStore::open`]'s job.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        Self::create_from(dir, Catalog::new())
    }

    /// Create the store of `catalog` in `dir`, making the directory if it
    /// does not exist.
    ///
    /// Nothing is written until [`CatalogStore::save`]; the catalog is
    /// taken as it stands, and `save` writes its segments to files of this
    /// directory.
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::create`].
    pub fn create_from(dir: impl AsRef<Path>, mut catalog: Catalog) -> Result<Self, SnapshotError> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        if dir.join(MANIFEST).try_exists()? {
            return Err(SnapshotError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "a catalog manifest is already in this directory",
            )));
        }
        // Without a manifest there is no catalog here, so a segment file is
        // what a store left that never saved.
        sweep(dir, &[]);
        // The segments come from wherever the catalog was, and this
        // directory holds none of them yet.
        for part in catalog.parts_mut() {
            part.file = NO_FILE;
        }
        Ok(Self::new(dir, catalog, None, 0))
    }

    /// Open the catalog of `dir` and take over as its writer.
    ///
    /// Every check of [`Catalog::map_from`] runs, on every segment, and
    /// each file is checked against the shape its manifest entry declares.
    /// No page is faulted in, so a catalog larger than memory opens with
    /// only its keys and directories on the heap. Segment and temporary
    /// files the manifest does not name are removed: a crash can leave
    /// them.
    ///
    /// # Errors
    ///
    /// [`SnapshotError`] when the manifest is missing or does not read, a
    /// segment file is missing or is not the segment its entry describes,
    /// or a checksum or structural check fails.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        Self::open_with(dir.as_ref(), false)
    }

    /// Open `dir` as [`CatalogStore::open`] does, without the checks that
    /// read every posting, under the terms of
    /// [`Catalog::read_from_trusted`].
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::open`], except checksum mismatches in the
    /// segments and unsorted runs, which go unnoticed.
    pub fn open_trusted(dir: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        Self::open_with(dir.as_ref(), true)
    }

    fn open_with(dir: &Path, trusted: bool) -> Result<Self, SnapshotError> {
        let (catalog, manifest) = load(dir, trusted)?;
        sweep(dir, &files_of(&catalog));
        Ok(Self::new(
            dir,
            catalog,
            Some(manifest.generation),
            manifest.next_file,
        ))
    }

    fn new(dir: &Path, catalog: Catalog, on_disk: Option<u64>, next_file: u32) -> Self {
        Self {
            dir: dir.to_path_buf(),
            catalog,
            on_disk,
            next_file,
            spill: SPILL,
            settings: Settings::default(),
            faults: Faults::default(),
        }
    }

    /// The catalog as it stands: search it, or clone it for other threads.
    #[must_use]
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// The postings of a merge's inputs from which the merge is written to
    /// a segment file instead of built on the heap; 2<sup>26</sup> unless
    /// set.
    #[must_use]
    pub fn spill_postings(&self) -> usize {
        self.spill
    }

    /// Set the spill threshold: from this many postings of input on, a
    /// merge is written to a file and mapped back, and below it the result
    /// stays on the heap until it is saved.
    ///
    /// Zero sends every merge to a file; [`usize::MAX`] keeps every merge
    /// in memory. Both give the same catalog.
    pub fn set_spill_postings(&mut self, postings: usize) {
        self.spill = postings;
    }

    /// Add a reference from its `(hash, time)` rows and return its id, as
    /// [`Catalog::add`] does.
    ///
    /// # Errors
    ///
    /// Any error of a merge that writes a file. The store and its catalog
    /// are then what they were before the call.
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    pub fn add(&mut self, rows: impl IntoIterator<Item = (u32, u32)>) -> io::Result<u32> {
        self.change(|catalog, merges| catalog.add_of(rows, merges))
    }

    /// Add several references, each from its `(hash, time)` rows, and
    /// return the range of ids they were given, as [`Catalog::add_many`]
    /// does.
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::add`].
    ///
    /// # Panics
    ///
    /// Panics once every id below `u32::MAX` has been assigned.
    pub fn add_many<I, R>(&mut self, references: I) -> io::Result<core::ops::Range<u32>>
    where
        I: IntoIterator<Item = R>,
        R: IntoIterator<Item = (u32, u32)>,
    {
        self.change(|catalog, merges| catalog.add_many_of(references, merges))
    }

    /// Tombstone a reference. Returns `false` if the id is unknown or
    /// already removed.
    pub fn remove(&mut self, id: u32) -> bool {
        self.catalog.remove(id)
    }

    /// Merge every segment but the largest into one, as
    /// [`Catalog::compact_deltas`] does.
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::add`].
    pub fn compact_deltas(&mut self) -> io::Result<()> {
        self.change(|catalog, merges| catalog.compact_deltas_of(merges))
    }

    /// Merge everything into one segment, as [`Catalog::compact`] does.
    ///
    /// # Errors
    ///
    /// As [`CatalogStore::add`].
    pub fn compact(&mut self) -> io::Result<()> {
        self.change(|catalog, merges| catalog.compact_of(merges))
    }

    /// Run `change`, whose merges go where this store's threshold sends
    /// them, and remove the segment files a failed call wrote.
    fn change<T>(
        &mut self,
        change: impl FnOnce(&mut Catalog, &mut DirMerger<'_>) -> io::Result<T>,
    ) -> io::Result<T> {
        let mut merger = DirMerger {
            dir: &self.dir,
            next_file: &mut self.next_file,
            spill: self.spill,
            settings: self.settings,
            created: Vec::new(),
            faults: &mut self.faults,
        };
        let outcome = change(&mut self.catalog, &mut merger);
        if outcome.is_err() {
            // The catalog took none of them: a failed change leaves its
            // parts as they were.
            for file in merger.created {
                let _ = fs::remove_file(self.dir.join(manifest::segment_name(file)));
            }
        }
        outcome
    }

    /// Write the catalog to the directory and make it durable.
    ///
    /// Every segment that has no file is written to its temporary name,
    /// synced and renamed into place; then the manifest is written to
    /// `MANIFEST.tmp`, synced and renamed over `MANIFEST`, and the
    /// directory is synced (on Unix). Only then are the segment files the
    /// new manifest does not name removed; a file that cannot be removed
    /// (Windows refuses while a mapping holds it) is tried again by the
    /// next `save`.
    ///
    /// After a crash at any of those points the directory opens as the
    /// catalog of the last `save` that returned.
    ///
    /// # Errors
    ///
    /// Any error of writing, syncing or renaming a file, and an error when
    /// the manifest on disk is not the one this store read or wrote last:
    /// another writer has the directory. A call that fails before the
    /// manifest is replaced leaves the catalog on disk as it was and this
    /// store's catalog unchanged; the segment files it finished stay for
    /// the next `save`, which does not write them again.
    pub fn save(&mut self) -> io::Result<()> {
        if disk_generation(&self.dir)? != self.on_disk {
            return Err(io::Error::other(
                "another writer saved this catalog directory",
            ));
        }
        self.write_segments()?;
        let entries = self.entries()?;
        self.put_manifest(entries)?;
        self.faults.stop_at(Stop::BeforeDeletions)?;
        sweep(&self.dir, &files_of(&self.catalog));
        Ok(())
    }

    /// Give every segment that has none a file of its own, and hold the
    /// mapping of that file from then on instead of the segment on the
    /// heap. A clone of the catalog keeps the heap's copy alive until it
    /// goes.
    fn write_segments(&mut self) -> io::Result<()> {
        for i in 0..self.catalog.parts().len() {
            let part = &self.catalog.parts()[i];
            if part.file().is_some() {
                continue;
            }
            // A handle on the segment, so that writing it does not hold
            // the catalog borrowed.
            let segment = part.segment.clone();
            let file = self.next_file;
            let (mapped, ids) = self.write_segment(&segment, file)?;
            self.next_file = file + 1;
            self.catalog.parts_mut()[i].move_to_file(mapped, ids, file);
        }
        self.faults.stop_at(Stop::AfterSegments)
    }

    /// Write `segment` as segment file `file` and map it. The range is
    /// the one the file holds, which can be narrower than the part's.
    fn write_segment(
        &mut self,
        segment: &Segment,
        file: u32,
    ) -> io::Result<(Segment, Option<(u32, u32)>)> {
        let name = self.dir.join(manifest::segment_name(file));
        let temp = self.dir.join(manifest::temp_name(file));
        write_file(&mut self.faults, &temp, |out| segment.write_to(out))?;
        publish(&temp, &name)?;
        // This process wrote the file a moment ago, so it is mapped
        // without a second read of its postings.
        map_segment(&name, true, false).map_err(|e| {
            let _ = fs::remove_file(&name);
            io::Error::other(e)
        })
    }

    /// The manifest entries of the parts as they stand.
    fn entries(&self) -> io::Result<Vec<Entry>> {
        let mut entries = Vec::with_capacity(self.catalog.parts().len());
        for part in self.catalog.parts() {
            let Some(file) = part.file() else {
                return Err(io::Error::other("a segment of the catalog has no file"));
            };
            // The length is read here rather than remembered, so a file
            // that someone removed stops the save before the manifest
            // names it.
            let meta = fs::metadata(self.dir.join(manifest::segment_name(file)))?;
            entries.push(Entry {
                file,
                bytes: meta.len(),
                keys: part.segment.key_count() as u64,
                postings: part.segment.len() as u64,
                ids: (!part.segment.is_empty()).then_some(part.ids),
            });
        }
        Ok(entries)
    }

    /// Write the manifest of `entries`, rename it into place and sync the
    /// directory.
    fn put_manifest(&mut self, entries: Vec<Entry>) -> io::Result<()> {
        let generation = self.on_disk.map_or(1, |g| g + 1);
        let manifest = Manifest {
            generation,
            reference_count: self.catalog.reference_count() as u64,
            next_file: self.next_file,
            live: self.catalog.live_table().to_vec(),
            entries,
        };
        let temp = self.dir.join(manifest::MANIFEST_TEMP);
        write_file(&mut self.faults, &temp, |out| manifest.write(out))?;
        let renamed = self
            .faults
            .stop_at(Stop::AfterTempManifest)
            .and_then(|()| fs::rename(&temp, self.dir.join(MANIFEST)));
        if let Err(e) = renamed {
            let _ = fs::remove_file(&temp);
            return Err(e);
        }
        // The rename is where a save takes effect: the catalog on disk is
        // this store's from here on, whatever fails after it.
        self.on_disk = Some(generation);
        self.faults.stop_at(Stop::AfterRename)?;
        sync_dir(&self.dir)
    }

    /// Merge with `threads` threads instead of what the machine offers;
    /// what a test runs a merge to a file under.
    #[cfg(test)]
    pub(crate) fn with_threads(mut self, threads: usize) -> Self {
        self.settings.threads = threads;
        self
    }

    /// Make the next creation of a file fail.
    #[cfg(test)]
    pub(crate) fn fail_next_create(&mut self) {
        self.faults.create = true;
    }

    /// Make the next file written fail after `bytes`.
    #[cfg(test)]
    pub(crate) fn fail_next_write(&mut self, bytes: u64) {
        self.faults.write = Some(bytes);
    }

    /// Stop the next `save` after `at`.
    #[cfg(test)]
    pub(crate) fn stop_after(&mut self, at: Stop) {
        self.faults.stop = Some(at);
    }
}

impl Catalog {
    /// Map the catalog a directory holds, as [`Catalog::map_from`] maps a
    /// catalog snapshot: every segment file stays in the file's pages,
    /// while the keys and the directories are built in memory.
    ///
    /// Nothing is written. The pages are not faulted in, so a catalog
    /// larger than memory opens: only the pages a search touches arrive.
    /// A writer may replace the manifest and delete a segment file between
    /// the read of the manifest and the mapping of the file; the directory
    /// is then read again, a few times.
    ///
    /// As with [`Catalog::map_from`], nothing in the directory may be
    /// modified or truncated while this catalog, a clone of it or a
    /// catalog sharing one of its segments is alive.
    ///
    /// # Errors
    ///
    /// As [`Catalog::read_from`] for a manifest and every segment it names,
    /// plus any error opening or mapping a file, and a segment file that
    /// does not match the shape its manifest entry declares.
    pub fn map_dir(dir: impl AsRef<Path>) -> Result<Catalog, SnapshotError> {
        load(dir.as_ref(), false).map(|(catalog, _)| catalog)
    }

    /// Map a directory as [`Catalog::map_dir`] does, without the checks
    /// that read every posting, under the terms of
    /// [`Catalog::read_from_trusted`].
    ///
    /// # Errors
    ///
    /// As [`Catalog::map_dir`], except checksum mismatches in the segments
    /// and unsorted runs, which go unnoticed.
    pub fn map_dir_trusted(dir: impl AsRef<Path>) -> Result<Catalog, SnapshotError> {
        load(dir.as_ref(), true).map(|(catalog, _)| catalog)
    }
}

/// Read the manifest of `dir` and map its segments, starting over when a
/// writer saved in between.
fn load(dir: &Path, trusted: bool) -> Result<(Catalog, Manifest), SnapshotError> {
    for _ in 0..RETRIES {
        let bytes = fs::read(dir.join(MANIFEST))?;
        let manifest = manifest::read(&bytes)?;
        if let Some(parts) = map_all(dir, &manifest, trusted)? {
            let catalog = Catalog::from_files(parts, manifest.live.clone());
            return Ok((catalog, manifest));
        }
    }
    Err(SnapshotError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        "a segment file the manifest names is missing",
    )))
}

/// Map every segment a manifest names, checking each file against its
/// entry. `None` when one of the files is gone: either a writer saved
/// after the manifest was read, or the directory lost a file.
fn map_all(
    dir: &Path,
    manifest: &Manifest,
    trusted: bool,
) -> Result<Option<Vec<Part>>, SnapshotError> {
    let gone = |e: &io::Error| e.kind() == io::ErrorKind::NotFound;
    let mut parts = Vec::with_capacity(manifest.entries.len());
    for entry in &manifest.entries {
        let path = dir.join(manifest::segment_name(entry.file));
        let (segment, ids) = match map_segment(&path, trusted, false) {
            Ok(mapped) => mapped,
            Err(SnapshotError::Io(e)) if gone(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let bytes = match fs::metadata(&path) {
            Ok(meta) => meta.len(),
            Err(e) if gone(&e) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let fits = bytes == entry.bytes
            && segment.key_count() as u64 == entry.keys
            && segment.len() as u64 == entry.postings
            && ids == entry.ids;
        if !fits {
            return Err(SnapshotError::Corrupt(
                "a segment does not match its manifest entry",
            ));
        }
        let Some(ids) = ids else { continue };
        // The manifest's reader has checked that the range lies in the id
        // space.
        let stale = manifest.live[ids.0 as usize..=ids.1 as usize].contains(&false);
        parts.push(
            Part::new(segment, ids)
                .with_file(entry.file)
                .stale_if(stale),
        );
    }
    Ok(Some(parts))
}

/// The generation of the manifest in `dir`, or `None` when there is none.
fn disk_generation(dir: &Path) -> io::Result<Option<u64>> {
    let mut file = match File::open(dir.join(MANIFEST)) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    // A file too short to hold a generation is not a manifest a store
    // wrote, and the error is what keeps this store from replacing it.
    let mut head = [0u8; manifest::GENERATION_END];
    file.read_exact(&mut head)?;
    manifest::generation(&head)
        .map(Some)
        .ok_or_else(|| io::Error::other("not a catalog manifest"))
}

/// The segment files the parts of `catalog` have.
fn files_of(catalog: &Catalog) -> Vec<u32> {
    catalog.parts().iter().filter_map(Part::file).collect()
}

/// Remove what a crash, a failed write or a merge that replaced a segment
/// left in `dir`: every temporary file, and every segment file but those
/// of `keep`. Files of any other name are not the store's to remove.
///
/// A file that cannot be removed stays for the next sweep.
fn sweep(dir: &Path, keep: &[u32]) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let left = match manifest::segment_id(name) {
            Some(file) => !keep.contains(&file),
            None => manifest::is_temp(name),
        };
        if left {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Create `path`, write it with `fill` and sync it; the file is removed
/// again when any of that fails.
fn write_file(
    faults: &mut Faults,
    path: &Path,
    fill: impl FnOnce(&mut BufWriter<Sink>) -> io::Result<()>,
) -> io::Result<()> {
    faults.create()?;
    let after = faults.write();
    let written = File::create(path).and_then(|file| {
        let mut out = BufWriter::with_capacity(BUFFER, Sink::new(file, after));
        fill(&mut out)?;
        let sink = out.into_inner().map_err(io::IntoInnerError::into_error)?;
        sink.file.sync_all()
    });
    if written.is_err() {
        let _ = fs::remove_file(path);
    }
    written
}

/// Give the finished file `temp` its name. A file of that name is another
/// writer's segment, which a rename would silently replace.
fn publish(temp: &Path, name: &Path) -> io::Result<()> {
    let renamed = match name.try_exists() {
        Ok(false) => fs::rename(temp, name),
        Ok(true) => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "another writer wrote a segment file of this name",
        )),
        Err(e) => Err(e),
    };
    if renamed.is_err() {
        let _ = fs::remove_file(temp);
    }
    renamed
}

/// Sync a directory entry, where the platform has directory handles.
#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_: &Path) -> io::Result<()> {
    Ok(())
}

/// A merge that writes its result to a segment file when its inputs are
/// large enough, and keeps it on the heap otherwise.
struct DirMerger<'a> {
    dir: &'a Path,
    next_file: &'a mut u32,
    spill: usize,
    settings: Settings,
    /// The files this call wrote, for a caller that removes them again
    /// when it fails.
    created: Vec<u32>,
    faults: &'a mut Faults,
}

impl MergeParts for DirMerger<'_> {
    type Error = io::Error;

    fn merge<F: Fn(u32) -> bool + Sync>(
        &mut self,
        sides: &[(&Segment, Option<&F>)],
        ids: (u32, u32),
    ) -> io::Result<Option<Part>> {
        let total = sides
            .iter()
            .fold(0usize, |n, (segment, _)| n.saturating_add(segment.len()));
        if total < self.spill {
            let merged = Segment::merge_all(sides);
            return Ok((!merged.is_empty()).then(|| Part::new(merged, ids)));
        }
        self.faults.create()?;
        let settings = Settings {
            fail_after: self.faults.write().or(self.settings.fail_after),
            ..self.settings
        };
        let file = *self.next_file;
        let name = self.dir.join(manifest::segment_name(file));
        let temp = self.dir.join(manifest::temp_name(file));
        // A merge that fails removes what it wrote.
        let written = merge_to_file(sides, &temp, settings)?;
        if written.ids.is_none() {
            // Every posting was a removed reference's: no part, no file.
            let _ = fs::remove_file(&temp);
            return Ok(None);
        }
        publish(&temp, &name)?;
        *self.next_file = file + 1;
        self.created.push(file);
        // The merge summed what it wrote and read the sum back, so the
        // file is mapped without another read of its postings. Its range
        // is the one the file holds, which the removals can have made
        // narrower than `ids`.
        let (segment, held) = map_segment(&name, true, false).map_err(io::Error::other)?;
        Ok(held.map(|held| Part::new(segment, held).with_file(file)))
    }
}

/// A step of `save` after which a test stops it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    /// Every segment file is written and renamed.
    AfterSegments,
    /// The temporary manifest is written and synced.
    AfterTempManifest,
    /// The manifest is renamed into place.
    AfterRename,
    /// Everything is durable; only the deletions are left.
    BeforeDeletions,
}

/// Failures a test asks a store for. Nothing sets one outside the tests,
/// where each of these is one branch on a field that never changes.
#[derive(Clone, Copy, Debug, Default)]
struct Faults {
    /// Fail the next creation of a file.
    create: bool,
    /// Fail the next file written after this many bytes.
    write: Option<u64>,
    /// Stop `save` after this step.
    stop: Option<Stop>,
}

impl Faults {
    /// The error of creating a file, if the next creation is to fail.
    fn create(&mut self) -> io::Result<()> {
        if core::mem::take(&mut self.create) {
            return Err(io::Error::other("the store could not create a file"));
        }
        Ok(())
    }

    /// The byte count after which the next file written fails.
    fn write(&mut self) -> Option<u64> {
        self.write.take()
    }

    /// The error of stopping at `at`, if `save` is to stop there.
    fn stop_at(&mut self, at: Stop) -> io::Result<()> {
        if self.stop == Some(at) {
            self.stop = None;
            return Err(io::Error::other("the store stopped here"));
        }
        Ok(())
    }
}

/// The file a segment or a manifest is written to, which fails after a
/// byte count when a test asks for that.
struct Sink {
    file: File,
    /// Bytes after which a write fails, or `None` to write.
    after: Option<u64>,
    written: u64,
}

impl Sink {
    fn new(file: File, after: Option<u64>) -> Self {
        Self {
            file,
            after,
            written: 0,
        }
    }
}

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let take = match self.after {
            None => bytes.len(),
            Some(after) => {
                let left = after.saturating_sub(self.written);
                if left == 0 {
                    return Err(io::Error::other("the store could not write a file"));
                }
                bytes.len().min(usize::try_from(left).unwrap_or(usize::MAX))
            }
        };
        let n = self.file.write(&bytes[..take])?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Posting, Query, Scratch, SearchOptions};
    use alloc::format;

    /// A directory named after the test and the process, removed at the
    /// end.
    struct TempDir {
        dir: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("afpindex-store-{name}-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            Self { dir }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
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

    fn rows(rng: &mut Rng, hashes: u32) -> Vec<(u32, u32)> {
        let n = 20 + rng.below(120) as usize;
        (0..n)
            .map(|_| (rng.below(u64::from(hashes)) as u32, rng.below(500) as u32))
            .collect()
    }

    /// The segments of a catalog as they stand: their postings in order
    /// and the references of each.
    fn shape(cat: &Catalog) -> Vec<Vec<Posting>> {
        cat.segments().map(|s| s.postings().collect()).collect()
    }

    /// Everything a caller can see of a catalog.
    type State = (Vec<Vec<Posting>>, usize, usize, Vec<bool>);

    fn state(cat: &Catalog) -> State {
        (
            shape(cat),
            cat.reference_count(),
            cat.live_count(),
            (0..cat.reference_count() as u32)
                .map(|id| cat.is_live(id))
                .collect(),
        )
    }

    /// The files of a directory by name, with their lengths.
    fn files(dir: &Path) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = std::fs::read_dir(dir)
            .expect("the directory")
            .flatten()
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    e.metadata().map_or(0, |m| m.len()),
                )
            })
            .collect();
        out.sort();
        out
    }

    /// A catalog directory of random additions, removals, both compactions
    /// and saves, against a plain catalog given the same calls: after every
    /// step the segments, the ids and the liveness agree, and so does a
    /// map of the directory after every save and a store opened again.
    #[test]
    fn a_random_sequence_matches_a_plain_catalog() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let scratch = TempDir::new("model");
        let hashes = 60u32;
        for spill in [0usize, 5, usize::MAX] {
            let dir = scratch.dir.join(format!("spill-{spill}"));
            let mut store = CatalogStore::create(&dir).expect("a fresh directory");
            store.set_spill_postings(spill);
            let mut plain = Catalog::new();
            let mut ids: Vec<u32> = Vec::new();
            let mut saved: Option<State> = None;
            for step in 0..40 {
                match rng.below(10) {
                    0..=3 => {
                        let r = rows(&mut rng, hashes);
                        let id = store.add(r.clone()).expect("an addition");
                        let other = plain.add(r);
                        assert_eq!(id, other, "spill {spill}: step {step}");
                        ids.push(id);
                    }
                    4..=5 => {
                        let n = 1 + rng.below(3) as usize;
                        let batch: Vec<Vec<(u32, u32)>> =
                            (0..n).map(|_| rows(&mut rng, hashes)).collect();
                        let range = store
                            .add_many(batch.iter().map(|r| r.iter().copied()))
                            .expect("a batch");
                        let other = plain.add_many(batch.iter().map(|r| r.iter().copied()));
                        assert_eq!(range, other, "spill {spill}: step {step}");
                        ids.extend(range);
                    }
                    6..=7 if !ids.is_empty() => {
                        let id = ids[rng.below(ids.len() as u64) as usize];
                        assert_eq!(store.remove(id), plain.remove(id), "spill {spill}");
                    }
                    8 => {
                        if step % 2 == 0 {
                            store.compact_deltas().expect("a compaction");
                            plain.compact_deltas();
                        } else {
                            store.compact().expect("a compaction");
                            plain.compact();
                        }
                    }
                    _ => {
                        store.save().expect("a save");
                        saved = Some(state(&plain));
                        assert_same(
                            store.catalog(),
                            &plain,
                            &format!("spill {spill}: step {step}"),
                        );
                        let mapped = Catalog::map_dir(&dir).expect("a mapped directory");
                        assert_same(&mapped, &plain, &format!("spill {spill}: mapped"));
                        let trusted =
                            Catalog::map_dir_trusted(&dir).expect("a trusted mapped directory");
                        assert_same(&trusted, &plain, &format!("spill {spill}: trusted"));
                    }
                }
                assert_same(
                    store.catalog(),
                    &plain,
                    &format!("spill {spill}: step {step}"),
                );
            }
            // A store dropped without a save opens as the catalog of the
            // last save that returned.
            if let Some(want) = saved {
                drop(store);
                let reopened = CatalogStore::open(&dir).expect("an opened store");
                let got = state(reopened.catalog());
                assert_eq!(got, want, "spill {spill}: after a crash");
            }
        }
    }

    /// Two catalogs made by the same calls have the same segments, ids and
    /// liveness.
    fn assert_same(got: &Catalog, want: &Catalog, what: &str) {
        assert_eq!(
            got.segment_count(),
            want.segment_count(),
            "{what}: segments"
        );
        for (i, (a, b)) in got.segments().zip(want.segments()).enumerate() {
            assert_eq!(a.len(), b.len(), "{what}: segment {i} postings");
            assert_eq!(a.key_count(), b.key_count(), "{what}: segment {i} keys");
            assert_eq!(
                a.postings().collect::<Vec<_>>(),
                b.postings().collect::<Vec<_>>(),
                "{what}: segment {i}"
            );
        }
        assert_eq!(got.reference_count(), want.reference_count(), "{what}: ids");
        assert_eq!(got.live_count(), want.live_count(), "{what}: live");
        for id in 0..want.reference_count() as u32 {
            assert_eq!(got.is_live(id), want.is_live(id), "{what}: id {id}");
        }
    }

    /// A store's catalog gives the same search results as the plain catalog
    /// it mirrors, whatever the spill threshold: zero sends every merge to
    /// a file, `usize::MAX` keeps them all on the heap.
    #[test]
    fn every_spill_threshold_gives_the_same_catalog() {
        let mut rng = Rng(0x5A1B_7E57);
        let scratch = TempDir::new("spill");
        let hashes = 80u32;
        let mut plain = Catalog::new();
        let mut calls: Vec<Vec<Vec<(u32, u32)>>> = Vec::new();
        for _ in 0..14 {
            let batch: Vec<Vec<(u32, u32)>> =
                (0..=rng.below(3)).map(|_| rows(&mut rng, hashes)).collect();
            plain.add_many(batch.iter().map(|r| r.iter().copied()));
            calls.push(batch);
        }
        for id in [0u32, 4, 7] {
            plain.remove(id);
        }
        plain.compact_deltas();
        let query = Query::from_pairs(
            (0..300).map(|_| (rng.below(u64::from(hashes)) as u32, rng.below(500) as u32)),
            Some(64),
        );
        let options = SearchOptions::default();
        let want = plain
            .search(&query, &options, &mut Scratch::new())
            .expect("a search");
        let mut results = Vec::new();
        for spill in [0usize, 1, 200, usize::MAX] {
            let dir = scratch.dir.join(format!("spill-{spill}"));
            let mut store = CatalogStore::create(&dir).expect("a fresh directory");
            store.set_spill_postings(spill);
            for batch in &calls {
                store
                    .add_many(batch.iter().map(|r| r.iter().copied()))
                    .expect("a batch");
            }
            for id in [0u32, 4, 7] {
                assert!(store.remove(id));
            }
            store.compact_deltas().expect("a compaction");
            assert_same(store.catalog(), &plain, &format!("spill {spill}"));
            let got = store
                .catalog()
                .search(&query, &options, &mut Scratch::new())
                .expect("a search");
            assert_eq!(got, want, "spill {spill}: candidates");
            store.save().expect("a save");
            let mapped = Catalog::map_dir(&dir).expect("a mapped directory");
            let got = mapped
                .search(&query, &options, &mut Scratch::new())
                .expect("a search");
            assert_eq!(got, want, "spill {spill}: mapped candidates");
            results.push(got);
        }
        for got in &results[1..] {
            assert_eq!(got, &results[0], "every threshold");
        }
    }

    /// A store of `n` references over `hashes` hashes, every merge written
    /// to a file.
    fn small_store(dir: &Path, rng: &mut Rng, n: u32, hashes: u32) -> CatalogStore {
        let mut store = CatalogStore::create(dir).expect("a fresh directory");
        store.set_spill_postings(0);
        for _ in 0..n {
            store.add(rows(rng, hashes)).expect("an addition");
        }
        store
    }

    /// `save` stopped after each of its steps: the directory opens as the
    /// old catalog before the manifest is renamed and as the new one after
    /// it, and no file is left that the manifest does not name.
    #[test]
    fn a_save_stopped_at_any_step_leaves_a_catalog_that_opens() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let scratch = TempDir::new("crash-points");
        for (stop, old) in [
            (Stop::AfterSegments, true),
            (Stop::AfterTempManifest, true),
            (Stop::AfterRename, false),
            (Stop::BeforeDeletions, false),
        ] {
            let dir = scratch.dir.join(format!("{stop:?}"));
            // Segments stay on the heap, so every save has files to write
            // and each of its steps runs.
            let mut store = CatalogStore::create(&dir).expect("a fresh directory");
            for _ in 0..6 {
                store.add(rows(&mut rng, 40)).expect("an addition");
            }
            store.save().expect("the first save");
            let want = state(store.catalog());
            // A change the stopped save may or may not make durable. One
            // part is new and has no file, so the save writes a segment.
            for _ in 0..8 {
                store.add(rows(&mut rng, 40)).expect("an addition");
            }
            store.remove(1);
            let changed = state(store.catalog());
            store.stop_after(stop);
            let err = match store.save() {
                Ok(()) => panic!("{stop:?}: the save does not stop"),
                Err(e) => e,
            };
            assert_eq!(err.kind(), io::ErrorKind::Other, "{stop:?}");
            // What the directory opens as is the old catalog before the
            // rename and the new one after it. It is read, not opened: a
            // store that opens a directory takes it over.
            let on_disk = Catalog::map_dir(&dir).expect("the directory opens");
            let got = state(&on_disk);
            if old {
                assert_eq!(got, want, "{stop:?}: the catalog on disk");
                // The store still holds its change: nothing was durable,
                // and a save after the fault writes it.
                assert_eq!(state(store.catalog()), changed, "{stop:?}: the store");
                store.save().expect("a save after the fault");
                assert_eq!(state(store.catalog()), changed, "{stop:?}: saved");
                assert_eq!(
                    state(&Catalog::map_dir(&dir).expect("the directory")),
                    changed,
                    "{stop:?}: saved to disk"
                );
            } else {
                assert_eq!(got, changed, "{stop:?}: the new catalog on disk");
            }
            // Every file is the manifest or a segment it names, and a store
            // opens the directory and saves again.
            drop(store);
            let mut store = CatalogStore::open(&dir).expect("the directory opens");
            store
                .add(rows(&mut rng, 40))
                .expect("an addition after the crash");
            store.save().expect("a save after the crash");
            let segments = files(&dir)
                .into_iter()
                .filter(|(name, _)| name.starts_with("seg-"))
                .count();
            assert_eq!(
                segments,
                store.catalog().segment_count(),
                "{stop:?}: the segment files"
            );
            for (name, _) in files(&dir) {
                assert!(
                    name == MANIFEST || (name.starts_with("seg-") && name.contains(".afs")),
                    "{stop:?}: stray file {name}"
                );
            }
        }
    }

    /// A save that fails to write a segment or the manifest returns the
    /// error, leaves the catalog and the files the manifest names as they
    /// were, and the next save writes the catalog.
    #[test]
    fn a_failed_save_changes_nothing() {
        let mut rng = Rng(0x0DDB_A110_5EED_1234);
        let scratch = TempDir::new("failed-save");
        let dir = scratch.dir.join("store");
        let mut store = small_store(&dir, &mut rng, 5, 30);
        store.save().expect("the first save");
        for fault in 0..3 {
            for _ in 0..2 {
                store.add(rows(&mut rng, 30)).expect("an addition");
            }
            let want = state(store.catalog());
            let on_disk = state(&Catalog::map_dir(&dir).expect("the directory"));
            // The additions wrote segment files of their own; the failed
            // save adds none, since the file it fails on is its first.
            let before_files = files(&dir);
            match fault {
                0 => store.fail_next_create(),
                1 => store.fail_next_write(16),
                _ => store.fail_next_write(0),
            }
            let err = store.save().expect_err("the save fails");
            assert_eq!(err.kind(), io::ErrorKind::Other, "fault {fault}");
            assert_eq!(state(store.catalog()), want, "fault {fault}: the store");
            assert_eq!(files(&dir), before_files, "fault {fault}: the files");
            assert_eq!(
                state(&Catalog::map_dir(&dir).expect("the directory")),
                on_disk,
                "fault {fault}: the catalog on disk"
            );
            store.save().expect("a save without the fault");
            let opened = CatalogStore::open(&dir).expect("the directory opens");
            assert_eq!(state(opened.catalog()), state(store.catalog()));
        }
    }

    /// A merge that fails to create or write its file returns the error and
    /// leaves the catalog and the directory as they were; the next call
    /// succeeds.
    #[test]
    fn a_failed_merge_changes_nothing() {
        let mut rng = Rng(0x4146_4958_0000_0001);
        let scratch = TempDir::new("failed-merge");
        let dir = scratch.dir.join("store");
        let mut store = small_store(&dir, &mut rng, 6, 40);
        store.save().expect("the first save");
        for fault in 0..2 {
            for _ in 0..2 {
                store.add(rows(&mut rng, 40)).expect("an addition");
            }
            // More than one part, so the compaction merges, and the spill
            // threshold sends the merge to a file.
            assert!(store.catalog().segment_count() > 1);
            let want = state(store.catalog());
            let before = files(&dir);
            if fault == 0 {
                store.fail_next_create();
            } else {
                store.fail_next_write(8);
            }
            let err = store.compact().expect_err("the merge fails");
            assert_eq!(err.kind(), io::ErrorKind::Other, "fault {fault}");
            assert_eq!(state(store.catalog()), want, "fault {fault}: the catalog");
            assert_eq!(files(&dir), before, "fault {fault}: the files");
            store.compact().expect("a compaction without the fault");
            store.save().expect("a save without the fault");
            Catalog::map_dir(&dir).expect("the directory opens");
        }
    }

    /// A store whose every merge goes to a file builds the same catalog on
    /// one thread as on several.
    #[test]
    fn a_store_builds_the_same_catalog_on_one_thread_and_on_several() {
        let mut rng = Rng(0x7EAD_0F1E);
        let scratch = TempDir::new("threads");
        let hashes = 50u32;
        let mut calls: Vec<Vec<Vec<(u32, u32)>>> = Vec::new();
        let mut plain = Catalog::new();
        for _ in 0..12 {
            let batch: Vec<Vec<(u32, u32)>> =
                (0..=rng.below(2)).map(|_| rows(&mut rng, hashes)).collect();
            plain.add_many(batch.iter().map(|r| r.iter().copied()));
            calls.push(batch);
        }
        plain.remove(3);
        plain.compact();
        let mut built = Vec::new();
        for threads in [1usize, 4] {
            let dir = scratch.dir.join(format!("threads-{threads}"));
            let mut store = CatalogStore::create(&dir)
                .expect("a fresh directory")
                .with_threads(threads);
            store.set_spill_postings(0);
            for batch in &calls {
                store
                    .add_many(batch.iter().map(|r| r.iter().copied()))
                    .expect("a batch");
            }
            store.remove(3);
            store.compact().expect("a compaction");
            store.save().expect("a save");
            assert_same(store.catalog(), &plain, &format!("{threads} threads"));
            built.push(Catalog::map_dir(&dir).expect("a mapped directory"));
        }
        assert_same(&built[0], &built[1], "one thread against several");
    }

    /// A manifest that is truncated, flipped, of a later version or of
    /// another kind is a `SnapshotError`; so is a segment file that is gone
    /// or does not fit its entry.
    #[test]
    fn corruption_is_an_error_and_never_a_panic() {
        let mut rng = Rng(0xFA_0001);
        let scratch = TempDir::new("corruption");
        let dir = scratch.dir.join("store");
        let mut store = small_store(&dir, &mut rng, 6, 40);
        store.save().expect("the save");
        let manifest = dir.join(MANIFEST);
        let good = std::fs::read(&manifest).expect("the manifest");
        let segment = files(&dir)
            .into_iter()
            .map(|(name, _)| name)
            .find(|name| name.starts_with("seg-"))
            .expect("a segment file");
        let segment = dir.join(segment);
        let segment_bytes = std::fs::read(&segment).expect("the segment");
        let repaired = || {
            std::fs::write(&manifest, &good).expect("the manifest again");
            std::fs::write(&segment, &segment_bytes).expect("the segment again");
        };
        for what in [
            "truncated",
            "flipped",
            "later version",
            "another kind",
            "missing segment",
            "a segment of another shape",
        ] {
            let mut bad = good.clone();
            match what {
                "truncated" => std::fs::write(&manifest, &good[..good.len() / 2]).unwrap(),
                "flipped" => {
                    // The first byte of the liveness bitmap.
                    bad[40] ^= 0x40;
                    std::fs::write(&manifest, &bad).unwrap();
                }
                "later version" => {
                    bad[4] = 9;
                    std::fs::write(&manifest, &bad).unwrap();
                }
                "another kind" => {
                    bad[0] = b'X';
                    std::fs::write(&manifest, &bad).unwrap();
                }
                "missing segment" => std::fs::remove_file(&segment).unwrap(),
                _ => {
                    // A valid segment of one posting, which is not the
                    // segment the entry describes.
                    let mut bytes = Vec::new();
                    Segment::from_postings(alloc::vec![Posting::new(1, 0, 0)])
                        .write_to(&mut bytes)
                        .unwrap();
                    std::fs::write(&segment, bytes).unwrap();
                }
            }
            let mapped = Catalog::map_dir(&dir);
            assert!(mapped.is_err(), "{what}: mapped");
            if what == "missing segment" {
                assert!(
                    matches!(mapped, Err(SnapshotError::Io(e)) if e.kind() == io::ErrorKind::NotFound),
                    "a missing file is reported as one"
                );
            }
            let opened = CatalogStore::open(&dir);
            assert!(opened.is_err(), "{what}: opened");
            repaired();
        }
        Catalog::map_dir(&dir).expect("the repaired directory");
    }

    /// A second store on the same directory makes the first one's next
    /// save fail, and changes nothing.
    #[test]
    fn a_second_writer_makes_the_first_save_fail() {
        let mut rng = Rng(0xB0B0_1234);
        let scratch = TempDir::new("second-writer");
        let dir = scratch.dir.join("store");
        let mut first = small_store(&dir, &mut rng, 5, 30);
        first.save().expect("the first save");
        let mut second = CatalogStore::open(&dir).expect("a second store");
        second.add(rows(&mut rng, 30)).expect("an addition");
        second.save().expect("the second store saves");
        let before = files(&dir);
        first.add(rows(&mut rng, 30)).expect("an addition");
        let err = first.save().expect_err("the first store is stale");
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(files(&dir), before, "the directory");
        // And the directory is still the second store's catalog.
        let opened = CatalogStore::open(&dir).expect("the directory opens");
        assert_same(opened.catalog(), second.catalog(), "the second store's");
    }

    /// Two stores of one directory give their next segment file the same
    /// name. The one that comes second fails instead of writing over the
    /// file of the first, and its catalog stays as it was.
    #[test]
    fn a_segment_file_is_never_written_over() {
        let mut rng = Rng(0x0C10_BBE5);
        let scratch = TempDir::new("no-clobber");
        let dir = scratch.dir.join("store");
        let mut first = small_store(&dir, &mut rng, 5, 30);
        first.save().expect("the first save");
        let mut second = CatalogStore::open(&dir).expect("a second store");
        second.set_spill_postings(0);
        // An addition and a compaction merge at least once, and every
        // merge of this store writes a file.
        second.add(rows(&mut rng, 30)).expect("an addition");
        second.compact().expect("a compaction");
        let taken = files(&dir);
        let name = dir.join(manifest::segment_name(first.next_file));
        let bytes = std::fs::read(&name).expect("the second store's segment");
        let before = state(first.catalog());
        let id = first.add(rows(&mut rng, 30));
        let after_add = state(first.catalog());
        let err = id
            .and_then(|_| first.compact())
            .expect_err("the name is taken");
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        // Whichever of the two calls merged is the one that failed, and
        // that one changed nothing.
        let now = state(first.catalog());
        assert!(now == before || now == after_add, "the first store");
        assert_eq!(files(&dir), taken, "the directory");
        assert_eq!(std::fs::read(&name).expect("the segment"), bytes);
        second.save().expect("the second store saves");
        let opened = Catalog::map_dir(&dir).expect("the directory opens");
        assert_same(&opened, second.catalog(), "the second store's");
    }

    /// A directory with segment files and no manifest holds no catalog: a
    /// store that never saved left them, and a new store starts over.
    #[test]
    fn a_new_store_clears_what_an_unsaved_one_left() {
        let mut rng = Rng(0x1EF7_0E55);
        let scratch = TempDir::new("unsaved");
        let dir = scratch.dir.join("store");
        let mut unsaved = small_store(&dir, &mut rng, 8, 30);
        unsaved.compact().expect("a compaction");
        assert!(files(&dir).iter().any(|(name, _)| name.starts_with("seg-")));
        drop(unsaved);
        let mut store = small_store(&dir, &mut rng, 8, 30);
        store.compact().expect("a compaction");
        store.save().expect("a save");
        let opened = CatalogStore::open(&dir).expect("the directory opens");
        assert_same(opened.catalog(), store.catalog(), "the new store's");
        assert_eq!(files(&dir).len(), 2, "the manifest and one segment");
    }

    /// A part's id range can be wider than the references its segment
    /// holds: a batch whose first and last references have no rows, and a
    /// merge that drops the references at either end. The directory still
    /// opens as the same catalog, whichever way the merge went.
    #[test]
    fn ranges_wider_than_the_references_open_again() {
        let mut rng = Rng(0x0E0E_5EED);
        let scratch = TempDir::new("wide-ranges");
        for spill in [0usize, usize::MAX] {
            let dir = scratch.dir.join(format!("spill-{spill}"));
            let mut store = CatalogStore::create(&dir).expect("a fresh directory");
            store.set_spill_postings(spill);
            let batch = [
                Vec::new(),
                rows(&mut rng, 30),
                rows(&mut rng, 30),
                Vec::new(),
            ];
            let ids = store
                .add_many(batch.iter().map(|r| r.iter().copied()))
                .expect("a batch");
            assert_eq!(ids, 0..4);
            store.save().expect("a save of the batch");
            assert_same(
                &Catalog::map_dir(&dir).expect("the batch opens"),
                store.catalog(),
                "the batch",
            );
            for _ in 0..5 {
                store.add(rows(&mut rng, 30)).expect("an addition");
            }
            // The first and the last reference with postings go, so the
            // compacted segment holds a narrower range than its parts did.
            assert!(store.remove(1));
            assert!(store.remove(8));
            store.compact().expect("a compaction");
            store.save().expect("a save of the compaction");
            let want = state(store.catalog());
            assert_eq!(state(&Catalog::map_dir(&dir).expect("it maps")), want);
            assert_eq!(
                state(&Catalog::map_dir_trusted(&dir).expect("it maps trusted")),
                want
            );
            drop(store);
            let opened = CatalogStore::open(&dir).expect("it opens");
            assert_eq!(state(opened.catalog()), want, "spill {spill}");
        }
    }

    /// A store removes the segment files and temporaries a manifest does
    /// not name, and nothing else: a file of another name in the directory
    /// is not the store's to delete.
    #[test]
    fn files_of_other_names_are_left_alone() {
        let mut rng = Rng(0xF11E_5AFE);
        let scratch = TempDir::new("other-files");
        let dir = scratch.dir.join("store");
        std::fs::create_dir_all(&dir).expect("the directory");
        let others = ["notes.txt", "seg-notes.afs", "MANIFEST.bak", "data.tmp"];
        for name in others {
            std::fs::write(dir.join(name), b"not a segment").expect("a file of another name");
        }
        let mut store = small_store(&dir, &mut rng, 6, 30);
        store.save().expect("a save");
        store.compact().expect("a compaction");
        store.save().expect("a save that deletes the old segments");
        // What a crash leaves: a segment no manifest names and a temporary.
        let stray = manifest::segment_name(900);
        let temp = manifest::temp_name(901);
        std::fs::write(dir.join(&stray), b"left by a crash").expect("a stray segment");
        std::fs::write(dir.join(&temp), b"left by a crash").expect("a stray temporary");
        drop(store);
        let mut store = CatalogStore::open(&dir).expect("the directory opens");
        assert!(!dir.join(&stray).exists(), "the stray segment goes");
        assert!(!dir.join(&temp).exists(), "the stray temporary goes");
        store.add(rows(&mut rng, 30)).expect("an addition");
        store.save().expect("a save");
        for name in others {
            assert_eq!(
                std::fs::read(dir.join(name)).expect("the file is still there"),
                b"not a segment",
                "{name}"
            );
        }
    }

    /// A directory that does not exist has no catalog; a store created in
    /// one that already holds a catalog is refused.
    #[test]
    fn a_directory_of_its_own() {
        let mut rng = Rng(0x5151);
        let scratch = TempDir::new("own-directory");
        let missing = scratch.dir.join("nothing").join("here");
        assert!(Catalog::map_dir(&missing).is_err());
        let dir = scratch.dir.join("store");
        let mut store = CatalogStore::create(&dir).expect("a fresh directory");
        store.add(rows(&mut rng, 20)).expect("an addition");
        store.save().expect("the save");
        let err = CatalogStore::create(&dir).expect_err("the catalog is there");
        assert!(matches!(err, SnapshotError::Io(e) if e.kind() == io::ErrorKind::AlreadyExists));
        CatalogStore::create_from(&dir, Catalog::new()).expect_err("and stays there");
    }

    /// A store opened again finds the same catalog, and a clone of a
    /// catalog the store holds saves the same segments.
    #[test]
    fn a_clone_of_the_catalog_saves_the_same_segments() {
        let mut rng = Rng(0xC10E);
        let scratch = TempDir::new("clone");
        let dir = scratch.dir.join("store");
        let mut store = small_store(&dir, &mut rng, 6, 40);
        store.save().expect("the save");
        let copy = store.catalog().clone();
        let other = scratch.dir.join("copy");
        let mut store = CatalogStore::create_from(&other, copy).expect("a fresh directory");
        store.set_spill_postings(usize::MAX);
        store.save().expect("the copy saves");
        let a = Catalog::map_dir(&dir).expect("the original");
        let b = Catalog::map_dir(&other).expect("the copy");
        assert_same(&a, &b, "the copy");
    }

    /// A compaction that a keep empties leaves no part and no file.
    #[test]
    fn a_merge_that_removes_every_posting_leaves_no_file() {
        let mut rng = Rng(0xE3E3);
        let scratch = TempDir::new("emptied");
        let dir = scratch.dir.join("store");
        let mut store = small_store(&dir, &mut rng, 6, 40);
        store.save().expect("the save");
        for id in 0..store.catalog().reference_count() as u32 {
            assert!(store.remove(id));
        }
        store.compact().expect("the compaction");
        assert_eq!(store.catalog().segment_count(), 0);
        store.save().expect("the save");
        let opened = CatalogStore::open(&dir).expect("the directory opens");
        assert_eq!(opened.catalog().segment_count(), 0);
        assert_eq!(opened.catalog().reference_count(), 6);
        assert_eq!(opened.catalog().live_count(), 0);
        // Only the manifest is left: the segments went with the save.
        assert_eq!(files(&dir).len(), 1);
    }
}
