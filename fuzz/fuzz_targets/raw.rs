//! Any bytes, read as a catalog and a segment snapshot and as a manifest.

#![no_main]

use hashtally::{Catalog, Manifest, Segment};
use hashtally_fuzz::exercise;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(catalog) = Catalog::read_from(data) {
        exercise(&catalog);
    }
    if let Ok(segment) = Segment::read_from(data) {
        exercise(&Catalog::from_segment(segment));
    }
    // A trusted load may take a damaged file, and what is done with it
    // after may then panic; the load itself may not.
    let _ = Catalog::read_from_trusted(data);
    let _ = Segment::read_from_trusted(data);
    if let Ok(manifest) = Manifest::from_bytes(data) {
        let _ = (manifest.live_count(), manifest.posting_count(), format!("{manifest:?}"));
        for entry in manifest.entries() {
            let _ = (entry.file_name(), entry.references());
        }
    }
});
