//! A saved catalog directory with one of its files changed, mapped and
//! opened.

#![no_main]

use afpindex::{Catalog, CatalogStore};
use afpindex_fuzz::{Edits, Shape, exercise, temp};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    shape: Shape,
    file: u8,
    edits: Edits,
}

fuzz_target!(|input: Input| {
    let dir = temp("store");
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = CatalogStore::create_from(&dir, input.shape.catalog()).expect("a new store");
    store.save().expect("a save");
    drop(store);

    // The manifest and the segment files, in a fixed order.
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("the directory")
        .map(|e| e.expect("an entry").path())
        .filter(|p| p.file_name().is_some_and(|n| n != "LOCK"))
        .collect();
    files.sort();
    let target = &files[usize::from(input.file) % files.len()];
    let mut bytes = std::fs::read(target).expect("a file");
    input.edits.apply(&mut bytes);
    std::fs::write(target, &bytes).expect("a rewrite");

    if let Ok(catalog) = Catalog::map_dir(&dir) {
        exercise(&catalog);
    }
    if let Ok(store) = CatalogStore::open(&dir) {
        exercise(store.catalog());
    }
    let _ = Catalog::map_dir_trusted(&dir);
    let _ = std::fs::remove_dir_all(&dir);
});
