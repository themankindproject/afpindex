//! Snapshot round trips and corruption checks (`std` feature).
#![cfg(feature = "std")]

use hashtally::{
    Catalog, Compaction, CompactionPolicy, Posting, Query, RefCoding, Scratch, SearchOptions,
    Segment, SnapshotError,
};

const HEADER: usize = 48;
/// Bytes of the four checksums that close a segment snapshot.
const FOOTER: usize = 32;
/// Bytes before a catalog snapshot's liveness bitmap.
const CATALOG_HEADER: usize = 32;

fn sample() -> Segment {
    let rows: Vec<Posting> = (0..5_000u32)
        .map(|i| Posting::new(i.wrapping_mul(0x9E37_79B1) % 700, i % 37, (i * 13) % 3_000))
        .collect();
    Segment::from_postings(rows)
}

/// Byte offsets of the keys, the offsets, the multiplicities, the coded
/// references, the times, the end of the references' data, and the four
/// checksums in a segment snapshot (version 1: the times first, then the
/// coded references, the keys, the two offsets and the multiplicities).
fn layout(seg: &Segment) -> (usize, usize, usize, usize, usize, usize, usize) {
    let k = seg.key_count();
    let (ref_bits, time_bits) = seg.bits();
    let times = HEADER;
    let refs = (times + time_bits.div_ceil(8) as usize).next_multiple_of(8);
    let refs_end = refs + ref_bits.div_ceil(8) as usize;
    let keys = refs_end.next_multiple_of(8);
    let offsets = (keys + 4 * k).next_multiple_of(8);
    // Packed references store no reference offsets.
    let offset_arrays = if seg.ref_coding() == RefCoding::Gaps {
        2
    } else {
        1
    };
    let mults = offsets + offset_arrays * 8 * (k + 1);
    let footer = (mults + k).next_multiple_of(8);
    (keys, offsets, mults, refs, times, refs_end, footer)
}

fn query() -> Query {
    Query::from_pairs(
        (0..200u32).map(|i| (i.wrapping_mul(0x9E37_79B1) % 700, i)),
        Some(64),
    )
}

#[test]
fn write_then_read_gives_the_same_segment_and_results() {
    for coding in [RefCoding::Packed, RefCoding::Gaps] {
        write_then_read(sample().to_ref_coding(coding));
    }
}

fn write_then_read(seg: Segment) {
    let mut bytes = Vec::new();
    seg.write_to(&mut bytes).unwrap();
    let (.., footer) = layout(&seg);
    assert_eq!(bytes.len(), footer + FOOTER);
    let back = Segment::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back, seg);
    let q = query();
    let o = SearchOptions::default();
    let mut s = Scratch::new();
    let a = Catalog::from_segment(seg).search(&q, &o, &mut s).unwrap();
    let b = Catalog::from_segment(back).search(&q, &o, &mut s).unwrap();
    assert_eq!(a, b);
    assert_ne!(a.len(), 0);
    let empty = Segment::default();
    let mut bytes = Vec::new();
    empty.write_to(&mut bytes).unwrap();
    assert!(Segment::read_from(bytes.as_slice()).unwrap().is_empty());
}

#[test]
fn corruption_is_reported_not_trusted() {
    for coding in [RefCoding::Packed, RefCoding::Gaps] {
        corruption_of(&sample().to_ref_coding(coding));
    }
}

fn corruption_of(seg: &Segment) {
    let mut good = Vec::new();
    seg.write_to(&mut good).unwrap();
    let (keys_at, offsets_at, mults_at, refs_at, times_at, refs_end, footer_at) = layout(seg);

    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(matches!(
        Segment::read_from(bad.as_slice()),
        Err(SnapshotError::BadMagic)
    ));

    let mut bad = good.clone();
    bad[4] = 9;
    assert!(matches!(
        Segment::read_from(bad.as_slice()),
        Err(SnapshotError::UnsupportedVersion(9))
    ));
    // No other version is read, a later one or none at all.
    for version in [0u8, 2] {
        let mut bad = good.clone();
        bad[4] = version;
        assert!(matches!(
            Segment::read_from(bad.as_slice()),
            Err(SnapshotError::UnsupportedVersion(v)) if v == u32::from(version)
        ));
    }

    for cut in [HEADER - 1, offsets_at + 3, footer_at, good.len() - 1] {
        let err = Segment::read_from(&good[..cut]).unwrap_err();
        assert!(
            matches!(err, SnapshotError::Truncated),
            "cut at {cut}: {err}"
        );
    }

    // Claim more postings than a segment may hold, or more keys than there
    // are hashes: refused from the header, before any allocation.
    let mut bad = good.clone();
    bad[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = Segment::read_from(bad.as_slice()).unwrap_err();
    assert!(matches!(err, SnapshotError::Corrupt(_)), "{err}");
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&((1u64 << 32) + 1).to_le_bytes());
    let err = Segment::read_from(bad.as_slice()).unwrap_err();
    assert!(matches!(err, SnapshotError::Corrupt(_)), "{err}");

    // Any flipped data byte fails the checksum (padding is not data).
    for at in [
        keys_at,
        offsets_at + 8,
        mults_at + 2,
        refs_at + 5,
        times_at + 9,
        refs_end - 1,
    ] {
        let mut bad = good.clone();
        bad[at] ^= 0x40;
        let err = Segment::read_from(bad.as_slice()).unwrap_err();
        assert!(matches!(err, SnapshotError::Checksum), "byte {at}: {err}");
    }
    let mut bad = good.clone();
    bad[footer_at] ^= 1;
    assert!(matches!(
        Segment::read_from(bad.as_slice()),
        Err(SnapshotError::Checksum)
    ));

    // The header is covered too: a width change that keeps the packed
    // length (two 1-bit times read as 2-bit) fails the checksum instead
    // of decoding wrong values.
    let tiny = Segment::from_postings(vec![Posting::new(7, 0, 0), Posting::new(9, 1, 1)]);
    let mut bytes = Vec::new();
    tiny.write_to(&mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 1);
    bytes[8] = 2;
    assert!(matches!(
        Segment::read_from(bytes.as_slice()),
        Err(SnapshotError::Checksum)
    ));

    // Error values display and chain.
    let err = Segment::read_from(&good[..10]).unwrap_err();
    assert_eq!(err.to_string(), "snapshot ends early");
    assert!(std::error::Error::source(&err).is_none());
}

#[test]
fn catalog_snapshot_keeps_ids_liveness_and_segments() {
    let mut cat = Catalog::builder();
    for i in 0..30u32 {
        cat.add(
            (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
        );
    }
    let mut cat = cat.build();
    for i in 30..45u32 {
        cat.add(
            (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
        );
    }
    // Remove some ids, including the last one, so the id space has holes
    // and a trailing gap that a bare segment could not represent.
    for id in [0u32, 7, 31, 44] {
        assert!(cat.remove(id));
    }
    let q = Query::from_pairs(
        (0..100u32).map(|j| {
            (
                33u32.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000,
                j,
            )
        }),
        Some(64),
    );
    let o = SearchOptions::default();
    let mut s = Scratch::new();
    let before = cat.search(&q, &o, &mut s).unwrap();
    assert!(before.iter().any(|c| c.reference == 33 && c.delta == 0));

    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let back = Catalog::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back.reference_count(), 45);
    assert_eq!(back.live_count(), 41);
    for id in 0..46 {
        assert_eq!(back.is_live(id), cat.is_live(id), "id {id}");
    }
    assert_eq!(back.segment_count(), cat.segment_count());
    assert_eq!(back.posting_count(), cat.posting_count());
    assert_eq!(back.search(&q, &o, &mut s).unwrap(), before);
    // A loaded catalog knows which segments hold removed references:
    // compacting it drops their postings like the original's.
    let mut purged = back.clone();
    purged.compact();
    assert_eq!(purged.posting_count(), 41 * 100);
    assert_eq!(purged.search(&q, &o, &mut s).unwrap(), before);
    // The next id continues after the removed tail; nothing is reused.
    let mut back = back;
    assert_eq!(back.add([(1, 1)]), 45);
    // A compacted catalog round-trips to one segment holding live postings only.
    cat.compact();
    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let back = Catalog::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back.segment_count(), 1);
    assert_eq!(back.posting_count(), 41 * 100);
    assert_eq!(back.search(&q, &o, &mut s).unwrap(), before);
    // An empty catalog round-trips too.
    let mut bytes = Vec::new();
    Catalog::new().write_to(&mut bytes).unwrap();
    let back = Catalog::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back.reference_count(), 0);
    assert_eq!(back.segment_count(), 0);
}

/// A loaded catalog knows what a full compaction would drop from its
/// largest segment, so it calls for the compactions the original does.
#[test]
fn catalog_snapshot_keeps_what_compaction_is_due() {
    let mut b = Catalog::builder();
    for i in 0..100u32 {
        b.add((0..50u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j) % 3_000, j)));
    }
    let mut cat = b.build();
    for id in 0..6 {
        assert!(cat.remove(id));
    }
    let policy = CompactionPolicy::default();
    assert_eq!(cat.compaction_due(&policy), Some(Compaction::Full));
    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let mut back = Catalog::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back.compaction_due(&policy), Some(Compaction::Full));
    back.compact();
    assert_eq!(back.compaction_due(&policy), None);
}

/// A catalog snapshot keeps the reference coding: a loaded gap-coded
/// catalog goes on building gap-coded segments and finds what the
/// original finds, and a packed catalog of gap-coded segments compacts
/// to packed ones.
#[test]
fn catalog_snapshot_keeps_the_reference_coding() {
    let mut cat = Catalog::new();
    cat.set_ref_coding(RefCoding::Gaps);
    for i in 0..20u32 {
        cat.add((0..80u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 5) % 3_000, j)));
    }
    cat.remove(3);
    let q = Query::from_pairs(
        (0..80u32).map(|j| {
            (
                11u32.wrapping_mul(2_654_435_761).wrapping_add(j * 5) % 3_000,
                j,
            )
        }),
        Some(64),
    );
    let o = SearchOptions::default();
    let mut s = Scratch::new();
    let before = cat.search(&q, &o, &mut s).unwrap();
    assert!(before.iter().any(|c| c.reference == 11 && c.delta == 0));

    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let mut back = Catalog::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back.ref_coding(), RefCoding::Gaps);
    assert!(
        back.segments()
            .all(|seg| seg.ref_coding() == RefCoding::Gaps)
    );
    assert_eq!(back.search(&q, &o, &mut s).unwrap(), before);
    back.add([(1, 1)]);
    assert!(
        back.segments()
            .all(|seg| seg.ref_coding() == RefCoding::Gaps)
    );

    let mut packed = Catalog::read_from(bytes.as_slice()).unwrap();
    packed.set_ref_coding(RefCoding::Packed);
    packed.compact();
    assert_eq!(packed.segment_count(), 1);
    assert!(
        packed
            .segments()
            .all(|seg| seg.ref_coding() == RefCoding::Packed)
    );
    assert_eq!(packed.search(&q, &o, &mut s).unwrap(), before);
}

/// A trusted load gives the catalog a full one gives: ids, liveness,
/// segments, search results, and which segments hold removed references.
#[test]
fn trusted_catalog_loads_match_full_ones() {
    let mut cat = Catalog::builder();
    for i in 0..30u32 {
        cat.add(
            (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
        );
    }
    let mut cat = cat.build();
    for i in 30..45u32 {
        cat.add(
            (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
        );
    }
    for id in [0u32, 7, 31, 44] {
        assert!(cat.remove(id));
    }
    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let full = Catalog::read_from(bytes.as_slice()).unwrap();
    let trusted = Catalog::read_from_trusted(bytes.as_slice()).unwrap();
    assert_eq!(trusted.reference_count(), full.reference_count());
    assert_eq!(trusted.live_count(), full.live_count());
    assert_eq!(trusted.segment_count(), full.segment_count());
    for i in 0..full.segment_count() {
        assert_eq!(trusted.segment(i), full.segment(i), "segment {i}");
    }
    let (o, mut s) = (SearchOptions::default(), Scratch::new());
    for i in 0..45u32 {
        let q = Query::from_pairs(
            (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
            Some(64),
        );
        assert_eq!(
            trusted.search(&q, &o, &mut s).unwrap(),
            full.search(&q, &o, &mut s).unwrap(),
            "query {i}"
        );
    }
    let mut purged = trusted;
    purged.compact();
    assert_eq!(purged.posting_count(), 41 * 100);

    // The liveness bitmap is still checked.
    let mut bad = bytes;
    bad[CATALOG_HEADER] ^= 1;
    assert!(matches!(
        Catalog::read_from_trusted(bad.as_slice()),
        Err(SnapshotError::Checksum)
    ));
}

#[test]
fn catalog_snapshot_corruption_is_reported() {
    let mut cat = Catalog::new();
    // Ten rows, then one: the small delta stays a segment of its own.
    cat.add((0..10u32).map(|i| (i, i)));
    cat.add([(1, 5)]);
    cat.remove(0);
    let mut good = Vec::new();
    cat.write_to(&mut good).unwrap();

    // The two kinds do not read as each other.
    assert!(matches!(
        Segment::read_from(good.as_slice()),
        Err(SnapshotError::BadMagic)
    ));
    let mut seg_bytes = Vec::new();
    cat.segment(0).unwrap().write_to(&mut seg_bytes).unwrap();
    assert!(matches!(
        Catalog::read_from(seg_bytes.as_slice()),
        Err(SnapshotError::BadMagic)
    ));

    // Flip a liveness bit: the bitmap checksum catches it.
    let mut bad = good.clone();
    bad[CATALOG_HEADER] ^= 1;
    assert!(matches!(
        Catalog::read_from(bad.as_slice()),
        Err(SnapshotError::Checksum)
    ));

    // Truncate inside a segment.
    let err = Catalog::read_from(&good[..good.len() - 4]).unwrap_err();
    assert!(matches!(err, SnapshotError::Truncated), "{err}");

    // An implausible segment count is refused before reading any segment.
    let mut bad = good.clone();
    bad[24..32].copy_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(matches!(
        Catalog::read_from(bad.as_slice()),
        Err(SnapshotError::Corrupt(_))
    ));

    // An id space wider than `u32` is refused before the bitmap is read.
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&(1u64 << 32).to_le_bytes());
    assert!(matches!(
        Catalog::read_from(bad.as_slice()),
        Err(SnapshotError::Corrupt(_))
    ));

    // The header counts are checksummed: a smaller segment count would
    // silently drop a segment, and an id space grown within the same
    // bitmap byte would hand out an id that was never assigned.
    assert_eq!(cat.segment_count(), 2, "a large and a small segment");
    let mut bad = good.clone();
    bad[24..32].copy_from_slice(&1u64.to_le_bytes());
    assert!(matches!(
        Catalog::read_from(bad.as_slice()),
        Err(SnapshotError::Checksum)
    ));
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&3u64.to_le_bytes());
    assert!(matches!(
        Catalog::read_from(bad.as_slice()),
        Err(SnapshotError::Checksum)
    ));
}

#[test]
fn full_width_references_and_times_round_trip() {
    // Both widths at 32 bits: the reader's run-length bound
    // (`2^(ref_bits + time_bits)`) is clamped rather than overflowed, and
    // the largest values survive the packing.
    let seg = Segment::from_postings(vec![
        Posting::new(1, u32::MAX, u32::MAX),
        Posting::new(1, u32::MAX - 1, 0),
        Posting::new(2, 0, u32::MAX),
    ]);
    let mut bytes = Vec::new();
    seg.write_to(&mut bytes).unwrap();
    let back = Segment::read_from(bytes.as_slice()).unwrap();
    assert_eq!(back, seg);
    assert_eq!(back.max_reference(), Some(u32::MAX));
    assert_eq!(back.postings().map(|p| p.time).max(), Some(u32::MAX));
}

/// The length and FNV-1a hash of a snapshot.
type Fingerprint = (usize, u64);

/// Length and FNV-1a hash of the fixed catalog's and the fixed segment's
/// snapshots (format version 1), references packed and coded as gaps.
const GOLDEN: [(RefCoding, Fingerprint, Fingerprint); 2] = [
    (
        RefCoding::Packed,
        (96_960, 1_463_847_483_902_720_414),
        (32_096, 8_595_667_794_029_778_045),
    ),
    (
        RefCoding::Gaps,
        (141_368, 15_643_221_215_204_798_549),
        (48_912, 11_881_422_705_215_068_046),
    ),
];

/// FNV-1a over `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The format is a contract with the files already written: a fixed
/// catalog and a fixed segment must keep producing the same bytes. A
/// deliberate format change gets a new version number and new constants
/// here.
#[test]
fn snapshot_bytes_are_stable() {
    for (coding, catalog, segment) in GOLDEN {
        let (got_catalog, got_segment) = fixed_snapshots(coding);
        assert_eq!(got_catalog, catalog, "{coding:?}: the catalog");
        assert_eq!(got_segment, segment, "{coding:?}: the segment");
    }
}

/// Length and FNV-1a hash of the fixed catalog's and the fixed segment's
/// snapshots, references stored as `coding` says.
fn fixed_snapshots(coding: RefCoding) -> (Fingerprint, Fingerprint) {
    let rows = |seed: u32| {
        (0..300u32).map(move |j| {
            (
                seed.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 4_000,
                j * 3 % 1_000,
            )
        })
    };
    let mut builder = Catalog::builder();
    builder.ref_coding(coding);
    for seed in 0..40 {
        builder.add(rows(seed));
    }
    let mut cat = builder.build();
    for seed in 40..47 {
        cat.add(rows(seed));
    }
    assert!(cat.remove(3) && cat.remove(41));
    assert!(cat.segment_count() > 1, "a base and at least one delta");
    let mut bytes = Vec::new();
    cat.write_to(&mut bytes).unwrap();
    let catalog = (bytes.len(), fnv1a(&bytes));

    let seg = Segment::from_postings_coded(
        (0..2_000u32)
            .map(|i| Posting::new(i.wrapping_mul(0x9E37_79B1) >> 20, i % 300, i * 11 % 70_000))
            .collect(),
        coding,
    );
    let mut bytes = Vec::new();
    seg.write_to(&mut bytes).unwrap();
    (catalog, (bytes.len(), fnv1a(&bytes)))
}

/// Memory-mapped snapshots (`mmap` feature).
#[cfg(feature = "mmap")]
mod mapped {
    use super::*;
    use std::path::PathBuf;

    /// A fresh file name under the temp dir for this test.
    fn temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("hashtally-{}-{name}", std::process::id()));
        p
    }

    fn sample_catalog() -> (Catalog, Query) {
        let mut cat = Catalog::builder();
        for i in 0..30u32 {
            cat.add(
                (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
            );
        }
        let mut cat = cat.build();
        for i in 30..45u32 {
            cat.add(
                (0..100u32).map(|j| (i.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000, j)),
            );
        }
        for id in [0u32, 7, 31, 44] {
            assert!(cat.remove(id));
        }
        let q = Query::from_pairs(
            (0..100u32).map(|j| {
                (
                    33u32.wrapping_mul(2_654_435_761).wrapping_add(j * 7) % 5_000,
                    j,
                )
            }),
            Some(64),
        );
        (cat, q)
    }

    #[test]
    fn mapped_catalog_matches_the_read_one_and_keeps_working() {
        let (cat, q) = sample_catalog();
        let path = temp("catalog.afc");
        cat.write_to(std::fs::File::create(&path).unwrap()).unwrap();
        let o = SearchOptions::default();
        let mut s = Scratch::new();
        let want = cat.search(&q, &o, &mut s).unwrap();
        assert!(want.iter().any(|c| c.reference == 33 && c.delta == 0));

        let read = Catalog::read_from(std::fs::File::open(&path).unwrap()).unwrap();
        let trusted = Catalog::map_from_trusted(&path).unwrap();
        assert_eq!(trusted.search(&q, &o, &mut s).unwrap(), want);
        for i in 0..read.segment_count() {
            assert_eq!(trusted.segment(i), read.segment(i), "segment {i}");
        }
        let mapped = Catalog::map_from(&path).unwrap();
        assert_eq!(mapped.reference_count(), read.reference_count());
        assert_eq!(mapped.live_count(), read.live_count());
        assert_eq!(mapped.posting_count(), read.posting_count());
        assert_eq!(mapped.segment_count(), read.segment_count());
        for id in 0..46 {
            assert_eq!(mapped.is_live(id), cat.is_live(id), "id {id}");
        }
        assert_eq!(mapped.search(&q, &o, &mut s).unwrap(), want);
        // The postings are not on the heap; the keys and directory are.
        assert!(mapped.heap_bytes() < read.heap_bytes());
        assert!(mapped.heap_bytes() > 0);

        // A mapped catalog changes like any other: the new postings are
        // owned, the mapped ones are read, and compaction copies them out.
        let mut m = mapped.clone();
        assert_eq!(m.add([(1, 1)]), 45);
        assert!(m.remove(33));
        assert!(
            m.search(&q, &o, &mut s)
                .unwrap()
                .iter()
                .all(|c| c.reference != 33)
        );
        m.compact();
        assert_eq!(m.segment_count(), 1);
        assert_eq!(m.posting_count(), 40 * 100 + 1);
        let mut bytes = Vec::new();
        m.write_to(&mut bytes).unwrap();
        let back = Catalog::read_from(bytes.as_slice()).unwrap();
        assert_eq!(back.posting_count(), m.posting_count());

        // The original mapping is untouched by all of that.
        assert_eq!(mapped.search(&q, &o, &mut s).unwrap(), want);
        drop((m, mapped, back));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn mapped_segment_round_trips_and_checks_every_data_byte() {
        for coding in [RefCoding::Packed, RefCoding::Gaps] {
            mapped_round_trip(&sample().to_ref_coding(coding));
        }
    }

    fn mapped_round_trip(seg: &Segment) {
        let mut good = Vec::new();
        seg.write_to(&mut good).unwrap();
        let path = temp("segment.afx");
        std::fs::write(&path, &good).unwrap();
        let back = Segment::map_from(&path).unwrap();
        assert_eq!(&back, seg);
        assert_eq!(back.max_reference(), seg.max_reference());
        drop(back);
        // Every data byte is checksummed in place, as on the read path.
        let (keys_at, offsets_at, mults_at, refs_at, times_at, refs_end, footer_at) = layout(seg);
        for at in [
            keys_at,
            offsets_at + 8,
            mults_at + 2,
            refs_at + 5,
            times_at + 9,
            refs_end - 1,
            footer_at,
        ] {
            let mut bad = good.clone();
            bad[at] ^= 0x40;
            std::fs::write(&path, &bad).unwrap();
            let err = Segment::map_from(&path).unwrap_err();
            assert!(matches!(err, SnapshotError::Checksum), "byte {at}: {err}");
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn mapped_corruption_is_reported() {
        let (cat, _) = sample_catalog();
        let mut good = Vec::new();
        cat.write_to(&mut good).unwrap();
        let path = temp("bad.afc");

        // The liveness bitmap and the packed arrays are checksummed in place.
        for at in [CATALOG_HEADER, good.len() / 2] {
            let mut bad = good.clone();
            bad[at] ^= 0x10;
            std::fs::write(&path, &bad).unwrap();
            let err = Catalog::map_from(&path).unwrap_err();
            assert!(matches!(err, SnapshotError::Checksum), "byte {at}: {err}");
        }
        // Truncated anywhere: refused, never read past the file.
        for cut in [
            CATALOG_HEADER - 1,
            good.len() / 3,
            good.len() - 4,
            good.len() - 1,
        ] {
            std::fs::write(&path, &good[..cut]).unwrap();
            let err = Catalog::map_from(&path).unwrap_err();
            assert!(
                matches!(err, SnapshotError::Truncated | SnapshotError::Corrupt(_)),
                "cut {cut}: {err}"
            );
        }
        std::fs::write(&path, b"AFCT").unwrap();
        assert!(matches!(
            Catalog::map_from(&path).unwrap_err(),
            SnapshotError::Truncated
        ));
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            Catalog::map_from(&path).unwrap_err(),
            SnapshotError::Io(_)
        ));
        // A segment file is not a catalog and the other way round.
        let seg_path = temp("seg.afx");
        cat.segment(0)
            .unwrap()
            .write_to(std::fs::File::create(&seg_path).unwrap())
            .unwrap();
        assert!(matches!(
            Catalog::map_from(&seg_path).unwrap_err(),
            SnapshotError::BadMagic
        ));
        std::fs::remove_file(&seg_path).unwrap();
    }
}
