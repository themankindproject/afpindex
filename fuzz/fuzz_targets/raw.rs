//! Any bytes, read as a catalog and as a segment snapshot.

#![no_main]

use afpindex::{Catalog, Segment};
use afpindex_fuzz::exercise;
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
});
