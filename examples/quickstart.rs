//! Build a small catalog, search it, add and remove a reference.

use afindex::{Catalog, Query, Scratch, SearchOptions, vote_floor};

fn main() {
    // Three references of 200 rows each; reference 2 shares its first 120
    // rows with reference 1, shifted 1 000 frames later.
    let rows = |seed: u32| -> Vec<(u32, u32)> {
        (0..200u32)
            .map(|i| (seed.wrapping_mul(2_654_435_761).wrapping_add(i * 7), i * 5))
            .collect()
    };
    let mut builder = Catalog::builder();
    let r0 = builder.add(rows(1));
    let r1 = builder.add(rows(2));
    let r2 = builder.add(
        rows(2)
            .into_iter()
            .take(120)
            .map(|(h, t)| (h, t + 1_000))
            .chain(rows(3)),
    );
    let mut catalog = builder.build();
    println!("{catalog:?}");

    // A query clip: reference 1's rows 40..160, shifted so the clip starts
    // at 0, plus a few rows that match nothing.
    let clip: Vec<(u32, u32)> = rows(2)[40..160]
        .iter()
        .map(|&(h, t)| (h, t - 200))
        .chain((0..30).map(|i| (0xDEAD_0000 + i, i)))
        .collect();
    let query = Query::from_pairs(clip, Some(64));
    let mut options = SearchOptions::default();
    options.min_total_votes = vote_floor(query.input_rows(), 16, 0.02);
    let mut scratch = Scratch::new();

    for c in catalog.search(&query, &options, &mut scratch).unwrap() {
        println!(
            "reference {} at Δ {} frames: {} votes in the best bin of {} total, {} noise bins",
            c.reference, c.delta, c.votes, c.total_votes, c.noise.bins
        );
    }
    assert!(!catalog.is_live(99) && catalog.is_live(r0));

    // Remove reference 1; reference 2 still matches the shared rows.
    catalog.remove(r1);
    let after = catalog.search(&query, &options, &mut scratch).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].reference, r2);
    assert_eq!(after[0].delta, 1_200);
    println!(
        "after removing {r1}: reference {} at Δ {}",
        after[0].reference, after[0].delta
    );

    // Add a copy of the clip itself and find it at Δ 0.
    let r3 = catalog.add((0..query.hash_count()).flat_map(|i| {
        let h = query.hashes()[i];
        query.times(i).iter().map(move |&t| (h, t))
    }));
    let again = catalog.search(&query, &options, &mut scratch).unwrap();
    let me = again.iter().find(|c| c.reference == r3).unwrap();
    assert_eq!(me.delta, 0);
    println!(
        "added {r3}: {} candidates, {} segments before compaction",
        again.len(),
        catalog.segment_count()
    );
    catalog.compact();
    assert_eq!(catalog.segment_count(), 1);
}
