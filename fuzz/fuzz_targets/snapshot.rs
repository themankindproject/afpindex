//! A catalog's or a segment's snapshot with bytes changed, read and mapped.

#![no_main]

use arbitrary::Arbitrary;
use hashtally::{Catalog, Segment};
use hashtally_fuzz::{Edits, Shape, exercise, temp};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    shape: Shape,
    segment: bool,
    edits: Edits,
}

fuzz_target!(|input: Input| {
    let catalog = input.shape.catalog();
    let mut bytes = Vec::new();
    if input.segment {
        let Some(segment) = catalog.segment(0) else {
            return;
        };
        segment.write_to(&mut bytes).expect("a write to memory");
    } else {
        catalog.write_to(&mut bytes).expect("a write to memory");
    }
    input.edits.apply(&mut bytes);

    let path = temp("snapshot");
    std::fs::write(&path, &bytes).expect("a temporary file");
    if input.segment {
        if let Ok(segment) = Segment::read_from(bytes.as_slice()) {
            exercise(&Catalog::from_segment(segment));
        }
        if let Ok(segment) = Segment::map_from(&path) {
            exercise(&Catalog::from_segment(segment));
        }
        let _ = Segment::read_from_trusted(bytes.as_slice());
        let _ = Segment::map_from_trusted(&path);
    } else {
        if let Ok(catalog) = Catalog::read_from(bytes.as_slice()) {
            exercise(&catalog);
        }
        if let Ok(catalog) = Catalog::map_from(&path) {
            exercise(&catalog);
        }
        let _ = Catalog::read_from_trusted(bytes.as_slice());
        let _ = Catalog::map_from_trusted(&path);
    }
    let _ = std::fs::remove_file(&path);
});
