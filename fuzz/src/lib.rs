//! What the fuzz targets share: a small catalog drawn from the input, and
//! the work a catalog loaded under the full checks must survive.
//!
//! `cargo fuzz` builds the crate with `cfg(fuzzing)`, under which stored
//! checksums always match, so a mutated file reaches the checks past them.
//!
//! ```bash
//! cd fuzz
//! cargo +nightly fuzz run raw        # any bytes into every reader
//! cargo +nightly fuzz run snapshot   # changed catalog and segment snapshots, read and mapped
//! cargo +nightly fuzz run store      # a changed file in a saved directory, mapped and opened
//! ```

use afpindex::{Catalog, Query, RefCoding, Scratch, SearchOptions};
use arbitrary::Arbitrary;

/// A catalog of a few references, added in batches so it has several
/// segments, some removed, perhaps compacted.
#[derive(Arbitrary, Debug)]
pub struct Shape {
    refs: Vec<Vec<(u8, u16)>>,
    gaps: bool,
    batches: u8,
    removes: Vec<u8>,
    compact: u8,
}

impl Shape {
    pub fn catalog(&self) -> Catalog {
        let mut catalog = Catalog::new();
        if self.gaps {
            catalog.set_ref_coding(RefCoding::Gaps);
        }
        let refs: Vec<Vec<(u32, u32)>> = self
            .refs
            .iter()
            .take(48)
            .map(|rows| {
                rows.iter()
                    .take(48)
                    .map(|&(h, t)| (u32::from(h).wrapping_mul(0x9E37_79B1), u32::from(t)))
                    .collect()
            })
            .collect();
        let per = refs
            .len()
            .div_ceil(usize::from(self.batches % 4) + 1)
            .max(1);
        for batch in refs.chunks(per) {
            catalog.add_many(batch.to_vec());
        }
        for &r in &self.removes {
            if catalog.reference_count() > 0 {
                catalog.remove(u32::from(r) % catalog.reference_count() as u32);
            }
        }
        match self.compact % 3 {
            1 => catalog.compact_deltas(),
            2 => catalog.compact(),
            _ => {}
        }
        catalog
    }
}

/// Byte changes to a file: flips at offsets taken modulo its length, and
/// perhaps a cut.
#[derive(Arbitrary, Debug)]
pub struct Edits {
    flips: Vec<(u32, u8)>,
    cut: Option<u32>,
}

impl Edits {
    pub fn apply(&self, bytes: &mut Vec<u8>) {
        if bytes.is_empty() {
            return;
        }
        for &(at, x) in self.flips.iter().take(8) {
            let at = at as usize % bytes.len();
            bytes[at] ^= x;
        }
        if let Some(cut) = self.cut {
            bytes.truncate(cut as usize % (bytes.len() + 1));
        }
    }
}

/// Search, change, compact and snapshot a catalog that loaded under the
/// full checks: none of it may panic.
pub fn exercise(catalog: &Catalog) {
    let _ = (
        catalog.live_count(),
        catalog.posting_count(),
        catalog.heap_bytes(),
    );
    let mut rows = Vec::new();
    for segment in catalog.segments() {
        let _ = (
            segment.max_reference(),
            segment.bits(),
            segment.heap_bytes(),
        );
        let _ = segment.postings().count();
        for &key in segment.keys().iter().take(8) {
            if let Some(run) = segment.run(key) {
                rows.extend(run.iter().take(4).map(|(_, t)| (key, t)));
            }
        }
        let _ = segment.filter(|r| r % 2 == 0);
    }
    search(catalog, &rows);
    let mut bytes = Vec::new();
    catalog.write_to(&mut bytes).expect("a write to memory");
    let back = Catalog::read_from(bytes.as_slice()).expect("a catalog's own snapshot");
    assert_eq!(back.reference_count(), catalog.reference_count());
    assert_eq!(back.posting_count(), catalog.posting_count());

    let mut changed = catalog.clone();
    changed.add(rows.iter().copied());
    if changed.reference_count() > 0 {
        changed.remove(0);
    }
    changed.compact_deltas();
    search(&changed, &rows);
    changed.compact();
    search(&changed, &rows);
}

fn search(catalog: &Catalog, rows: &[(u32, u32)]) {
    let query = Query::from_pairs(rows.iter().copied(), Some(8));
    let mut scratch = Scratch::new();
    for fanout in [None, Some(2)] {
        let mut options = SearchOptions::default();
        options.fanout = fanout;
        options.min_total_votes = 1;
        let _ = catalog.search(&query, &options, &mut scratch);
    }
}

/// A file of its own for this process under the system's temporary
/// directory.
pub fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("afpindex-fuzz-{name}-{}", std::process::id()))
}
