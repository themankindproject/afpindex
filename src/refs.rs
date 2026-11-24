//! Reference ids coded run by run.
//!
//! Inside a run the references ascend, so the run is coded in blocks of
//! [`BLOCK`] references: each block holds its first reference whole (its
//! base, at a width shared by the run) and then the gaps between
//! neighbours, at the width of the block's widest gap:
//!
//! ```text
//! run    := width(5) block+
//! block  := base(width) [width(5) gap(width) × (count − 1)]
//! ```
//!
//! where `count` is `min(65, references left)` and the bracket is there
//! only when `count > 1`. Fields are packed back to back, little-endian
//! bit order, and the runs of a segment follow one another in key order in
//! one stream of bits. A width is 0 to 30 or 32, held in five bits (31
//! stands for 32; a value of 31 bits is stored at 32). The coding is
//! canonical: the run's width is that of its largest base and a block's
//! that of its widest gap, so a run has one coding, and a merge can copy a
//! run's bits or code its references again and get the same bytes.
//!
//! A gap is mostly much smaller than a reference: on the real catalog the
//! references take 12.5 bits each instead of 17. A block decodes with the
//! same kernels as a packed array (eight gaps a lane each, `crate::simd`),
//! and a running sum from its base turns the gaps back into references.
//! Because every block starts from a whole reference, a reader looking for
//! a few references steps from base to base and decodes only the blocks
//! that can hold them.

use crate::packed::{Packed, PackedBuilder};
use crate::simd::{Decoder, LANES, Scalar, WINDOW};

/// References per block: a base and up to 64 gaps, eight kernel blocks.
/// Measured on the real catalog against blocks of 33, the longer block
/// takes 0.07 bits more per reference and a tenth fewer instructions to
/// count, half the fields between blocks being read for half as many.
pub(crate) const BLOCK: usize = 65;

/// Bits of a width field.
const WIDTH_BITS: u32 = 5;

/// The width a field of value `x` is stored at: the bits of `x`, with 31
/// taken up to 32 so that five bits name every width.
#[inline]
pub(crate) const fn width_of(x: u32) -> u32 {
    let bits = u32::BITS - x.leading_zeros();
    if bits == 31 { 32 } else { bits }
}

/// The five-bit field that names `width`.
#[inline]
const fn code_of(width: u32) -> u64 {
    if width == 32 { 31 } else { width as u64 }
}

/// The width the five-bit field at the bottom of `code` names.
#[inline(always)]
const fn width_of_code(code: u64) -> u32 {
    let code = (code & 31) as u32;
    code + (code == 31) as u32
}

/// Mask of the low `bits` bits, for a width (at most 32).
#[inline(always)]
const fn mask_of(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

/// The most bits a run of `n` references (at least one) can take: every
/// field at 32 bits, and a gap width for each block that has a gap.
#[cfg(any(feature = "std", test))]
#[inline]
pub(crate) const fn max_run_bits(n: u64) -> u64 {
    let gap_blocks = n.saturating_sub(1).div_ceil(BLOCK as u64);
    WIDTH_BITS as u64 + n * 32 + gap_blocks * WIDTH_BITS as u64
}

/// The least bits a run takes: its width field.
#[cfg(any(feature = "std", test))]
pub(crate) const MIN_RUN_BITS: u64 = WIDTH_BITS as u64;

/// An upper bound on the bits of `runs` runs holding `postings` references
/// between them: what a buffer that must hold any coding of them reserves.
#[cfg(any(feature = "std", test))]
#[inline]
pub(crate) const fn bound_bits(postings: u64, runs: u64) -> u64 {
    // Per run its width and the gap width of a partial block; per
    // reference 32 bits and a share (5/65) of a full block's gap width.
    runs.saturating_mul(WIDTH_BITS as u64 * 2)
        .saturating_add(postings.saturating_mul(33))
}

/// The bits from position `bit` on, in the low end of a word: at least 57
/// of them are the stream's (or the slack after it). Indexing past the
/// bytes panics rather than read anything it should not.
#[inline]
fn window(bytes: &[u8], bit: u64) -> u64 {
    let byte = (bit >> 3) as usize;
    let w = u64::from_le_bytes(bytes[byte..byte + 8].try_into().expect("8 bytes"));
    w >> (bit & 7)
}

/// The coded references of a segment: every run, back to back in key
/// order, as one array of single bits.
#[derive(Clone)]
pub(crate) struct Refs {
    bits: Packed,
}

impl Default for Refs {
    fn default() -> Self {
        Self {
            bits: PackedBuilder::with_capacity(0, 1).finish(),
        }
    }
}

impl Refs {
    /// Wrap a stream of bits (an array of width 1, or an empty one).
    pub(crate) fn from_bits(bits: Packed) -> Self {
        assert!(
            bits.bits() == 1 || bits.is_empty(),
            "coded references are an array of single bits"
        );
        if bits.is_empty() {
            return Self::default();
        }
        Self { bits }
    }

    /// The stream as an array of single bits, for a caller that copies
    /// stretches of it.
    #[inline]
    pub(crate) fn bits(&self) -> &Packed {
        &self.bits
    }

    /// Bits in the stream.
    #[inline]
    pub(crate) fn bit_len(&self) -> u64 {
        self.bits.len() as u64
    }

    /// The bytes of the stream, without the slack.
    #[cfg(any(feature = "std", test))]
    pub(crate) fn data(&self) -> &[u8] {
        self.bits.data()
    }

    /// Bytes owned on the heap; a mapped stream owns none.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.bits.heap_bytes()
    }

    /// The run of `n` references (at least one) whose coding starts at bit
    /// `at`.
    #[inline]
    pub(crate) fn run(&self, at: u64, n: usize) -> Run<'_> {
        debug_assert!(n > 0, "a run is never empty");
        Run {
            bytes: self.bits.raw(),
            at,
            n,
        }
    }

    /// Load a byte of every cache line from bit `at` to `bits` bits past
    /// it, and those up to `reach` bytes further for a short stretch, and
    /// fold them together for a caller to discard (see
    /// [`Packed::touch`]).
    #[inline]
    pub(crate) fn touch(&self, at: u64, bits: u64, reach: usize) -> u32 {
        /// Bytes per cache line on the targets this is tuned for.
        const LINE: usize = 64;
        let bytes = self.bits.raw();
        let last_byte = bytes.len() - 1;
        let first = ((at >> 3) as usize).min(last_byte);
        let last = ((at + bits.max(1) - 1) >> 3).min(last_byte as u64) as usize;
        let reach = if last - first < 2 * LINE { reach } else { 0 };
        let last = last.saturating_add(reach).min(last_byte);
        let mut acc = bytes[last];
        let mut i = first;
        while i < last {
            acc ^= bytes[i];
            i += LINE;
        }
        u32::from(acc)
    }
}

impl PartialEq for Refs {
    /// The same bits: the coding is canonical, so the same runs.
    fn eq(&self, other: &Self) -> bool {
        self.bits.len() == other.bits.len()
            && self.bits.raw()[..data_len(&self.bits)] == other.bits.raw()[..data_len(&other.bits)]
    }
}

impl Eq for Refs {}

/// Data bytes of a stream.
fn data_len(bits: &Packed) -> usize {
    Packed::data_len(bits.len(), bits.bits())
}

impl core::fmt::Debug for Refs {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Refs")
            .field("bits", &self.bits.len())
            .finish_non_exhaustive()
    }
}

/// One run's coded references: `n` of them from bit `at` of a stream.
#[derive(Clone, Copy)]
pub(crate) struct Run<'a> {
    bytes: &'a [u8],
    at: u64,
    n: usize,
}

/// What a block's first fields say.
#[derive(Clone, Copy)]
struct Head {
    /// Its first reference.
    base: u32,
    /// The width of its gaps (0 for a block of one reference).
    width: u32,
    /// The bit its gaps start at.
    gaps: u64,
    /// The bit after it.
    end: u64,
}

/// The fields of a block of `count` references (at least one) from bit
/// `bit`, in a run whose bases are `base_width` wide. One window holds
/// both: a base is at most 32 bits and a width field 5.
#[inline(always)]
fn head(bytes: &[u8], bit: u64, base_width: u32, count: usize) -> Head {
    let w = window(bytes, bit);
    let base = (w & mask_of(base_width)) as u32;
    // A block of one has no gap width field: the bits after its base are
    // the next block's.
    let more = u32::from(count > 1);
    let width = width_of_code(w >> base_width) * more;
    let gaps = bit + u64::from(base_width + WIDTH_BITS * more);
    Head {
        base,
        width,
        gaps,
        end: gaps + (count as u64 - 1) * u64::from(width),
    }
}

/// Call `f` with the `count` references after `value` whose gaps are
/// `width` wide from bit `bit`, decoding them eight at a time where
/// `decoder` has a kernel for the width.
#[inline(always)]
fn gaps<D: Decoder>(
    bytes: &[u8],
    decoder: D,
    mut value: u32,
    width: u32,
    mut bit: u64,
    count: usize,
    f: &mut impl FnMut(u32),
) {
    if width == 0 {
        // One reference repeated: no bits to read.
        for _ in 0..count {
            f(value);
        }
        return;
    }
    let mut done = 0;
    if D::BLOCKS
        && count >= LANES
        && let Some(kernel) = decoder.kernel(width, (bit & 7) as usize)
    {
        // Eight gaps are `width` whole bytes, so every group of the block
        // starts at the same bit of its byte.
        let mut rest = &bytes[(bit >> 3) as usize..];
        while done + LANES <= count {
            let Some(w) = rest.first_chunk::<WINDOW>() else {
                break;
            };
            for gap in kernel(w) {
                value = value.wrapping_add(gap);
                f(value);
            }
            rest = &rest[width as usize..];
            done += LANES;
        }
        bit += done as u64 * u64::from(width);
    }
    let mask = mask_of(width);
    for _ in done..count {
        value = value.wrapping_add((window(bytes, bit) & mask) as u32);
        bit += u64::from(width);
        f(value);
    }
}

impl<'a> Run<'a> {
    /// The width of the run's bases and the bit its first block starts at.
    #[inline(always)]
    fn base_width(&self) -> (u32, u64) {
        (
            width_of_code(window(self.bytes, self.at)),
            self.at + u64::from(WIDTH_BITS),
        )
    }

    /// The first (smallest) reference.
    #[inline]
    pub(crate) fn first(&self) -> u32 {
        let (base_width, bit) = self.base_width();
        (window(self.bytes, bit) & mask_of(base_width)) as u32
    }

    /// The bit after the run: its end in the stream, found from the width
    /// fields without decoding a gap.
    #[cfg(test)]
    pub(crate) fn end(&self) -> u64 {
        let (base_width, mut bit) = self.base_width();
        let mut left = self.n;
        while left > 0 {
            let count = left.min(BLOCK);
            bit = head(self.bytes, bit, base_width, count).end;
            left -= count;
        }
        bit
    }

    /// The references in order, one at a time.
    #[inline]
    pub(crate) fn cursor(&self) -> Cursor<'a> {
        let (base_width, bit) = self.base_width();
        Cursor {
            bytes: self.bytes,
            bit,
            base_width,
            left: self.n,
            block: 0,
            width: 0,
            value: 0,
        }
    }

    /// Call `f` with every reference in order, decoded one at a time.
    #[inline]
    pub(crate) fn for_each(&self, mut f: impl FnMut(u32)) {
        self.for_each_in(Scalar, |_, r| f(r));
    }

    /// Call `f` with every reference in order and its position in the
    /// run. Where `decoder` has a kernel for a block's width, its gaps
    /// are decoded eight at a time; a block of one reference repeated
    /// reads no gaps at all.
    #[inline(always)]
    pub(crate) fn for_each_in<D: Decoder>(&self, decoder: D, f: impl FnMut(usize, u32)) {
        self.for_each_where(decoder, |_, _| true, f);
    }

    /// As [`Run::for_each_in`], for the blocks `want` accepts only. It is
    /// offered each block in order as `(base, bound)`: every reference of
    /// the block lies in `base..=bound` (the bound is the next block's
    /// base, or `u32::MAX` for the last block). A block passed over costs
    /// one read of its first fields.
    #[inline(always)]
    pub(crate) fn for_each_where<D: Decoder>(
        &self,
        decoder: D,
        mut want: impl FnMut(u32, u32) -> bool,
        mut f: impl FnMut(usize, u32),
    ) {
        let bytes = self.bytes;
        let (base_width, bit) = self.base_width();
        let mut left = self.n;
        let mut pos = 0;
        let mut block = head(bytes, bit, base_width, left.min(BLOCK));
        loop {
            let count = left.min(BLOCK);
            left -= count;
            let next = (left > 0).then(|| head(bytes, block.end, base_width, left.min(BLOCK)));
            if want(block.base, next.map_or(u32::MAX, |h| h.base)) {
                let mut i = pos;
                let mut each = |r| {
                    f(i, r);
                    i += 1;
                };
                each(block.base);
                gaps(
                    bytes,
                    decoder,
                    block.base,
                    block.width,
                    block.gaps,
                    count - 1,
                    &mut each,
                );
            }
            let Some(next) = next else {
                break;
            };
            pos += count;
            block = next;
        }
    }

    /// Decode the run as [`Run::for_each_in`] does, refusing a coding that
    /// is not the canonical one, references that descend or pass the
    /// largest id, or a coding that does not end exactly at bit `end`.
    /// Nothing is read past `end` but the window of a field that starts
    /// before it, so a damaged run costs at most its own references' work
    /// and reads inside the stream when `end` is.
    #[cfg(any(feature = "std", test))]
    #[inline(always)]
    pub(crate) fn check<D: Decoder>(
        &self,
        decoder: D,
        end: u64,
        mut f: impl FnMut(u32),
    ) -> Result<(), &'static str> {
        let bytes = self.bytes;
        let (base_width, mut bit) = self.base_width();
        let mut left = self.n;
        let (mut last, mut largest_base) = (0u32, 0u32);
        while left > 0 {
            if bit > end {
                return Err("a run runs past its end");
            }
            let count = left.min(BLOCK);
            let block = head(bytes, bit, base_width, count);
            if block.end > end {
                return Err("a run runs past its end");
            }
            if left < self.n && block.base < last {
                return Err("a run's references descend");
            }
            left -= count;
            largest_base = block.base;
            f(block.base);
            // A sum past the largest id wraps, to below the reference
            // before it.
            let (mut prev, mut widest, mut ascending) = (block.base, 0u32, true);
            gaps(
                bytes,
                decoder,
                block.base,
                block.width,
                block.gaps,
                count - 1,
                &mut |r| {
                    widest |= r.wrapping_sub(prev);
                    ascending &= r >= prev;
                    prev = r;
                    f(r);
                },
            );
            if !ascending {
                return Err("a run's references pass the largest id");
            }
            if count > 1 && width_of(widest) != block.width {
                return Err("a block of gaps is not at the width of its widest");
            }
            (last, bit) = (prev, block.end);
        }
        if width_of(largest_base) != base_width {
            return Err("a run's bases are not at the width of the largest");
        }
        if bit != end {
            return Err("a run does not end where the next one starts");
        }
        Ok(())
    }
}

/// The references of a run, one at a time (see [`Run::cursor`]).
#[derive(Clone)]
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    /// The next field.
    bit: u64,
    /// The width of the run's bases.
    base_width: u32,
    /// References not yet returned.
    left: usize,
    /// Gaps left in the current block.
    block: usize,
    /// Their width.
    width: u32,
    /// The last reference returned.
    value: u32,
}

impl Iterator for Cursor<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.left == 0 {
            return None;
        }
        if self.block == 0 {
            let count = self.left.min(BLOCK);
            let block = head(self.bytes, self.bit, self.base_width, count);
            (self.value, self.width, self.bit) = (block.base, block.width, block.gaps);
            self.block = count - 1;
        } else {
            self.block -= 1;
            let gap = (window(self.bytes, self.bit) & mask_of(self.width)) as u32;
            self.bit += u64::from(self.width);
            self.value = self.value.wrapping_add(gap);
        }
        self.left -= 1;
        Some(self.value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Cursor<'_> {}

/// Code the run `refs` (ascending, at least one) onto the end of `out`,
/// an array of single bits.
pub(crate) fn push_run(out: &mut PackedBuilder, refs: &[u32]) {
    debug_assert!(!refs.is_empty(), "a run is never empty");
    debug_assert!(refs.windows(2).all(|w| w[0] <= w[1]), "a run ascends");
    // The largest base is the last block's.
    let base_width = width_of(refs[(refs.len() - 1) / BLOCK * BLOCK]);
    out.push_bits(code_of(base_width), WIDTH_BITS);
    for block in refs.chunks(BLOCK) {
        out.push_bits(u64::from(block[0]), base_width);
        if block.len() == 1 {
            continue;
        }
        let widest = block.windows(2).fold(0u32, |w, p| w | (p[1] - p[0]));
        let width = width_of(widest);
        out.push_bits(code_of(width), WIDTH_BITS);
        if width > 0 {
            for p in block.windows(2) {
                out.push_bits(u64::from(p[1] - p[0]), width);
            }
        }
    }
}

/// The bits [`push_run`] codes `refs` in.
#[cfg(test)]
pub(crate) fn run_bits(refs: &[u32]) -> u64 {
    let mut out = PackedBuilder::with_capacity(0, 1);
    push_run(&mut out, refs);
    out.len() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Runs of every shape: one reference, repeats, every gap width, the
    /// largest id, and lengths around the block size.
    fn runs() -> Vec<Vec<u32>> {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![
            vec![0],
            vec![u32::MAX],
            vec![(1 << 30) + 5],
            vec![7; 100],
            vec![0, u32::MAX],
            vec![0, 1 << 30, (1 << 31) + 3, u32::MAX],
        ];
        for n in [
            2usize, 8, 31, 32, 33, 34, 40, 64, 65, 66, 67, 130, 131, 136, 1000,
        ] {
            for spread in [0u64, 1, 3, 1 << 10, 1 << 17, 1 << 26, 1 << 31] {
                let mut v = 0u64;
                let mut run = Vec::with_capacity(n);
                for _ in 0..n {
                    v += if spread == 0 {
                        0
                    } else {
                        rng.next() % (spread + 1)
                    };
                    run.push(v.min(u64::from(u32::MAX)) as u32);
                }
                out.push(run);
            }
        }
        out
    }

    fn coded(runs: &[Vec<u32>]) -> (Refs, Vec<u64>) {
        let mut out = PackedBuilder::with_capacity(0, 1);
        let mut starts = Vec::new();
        for run in runs {
            starts.push(out.len() as u64);
            push_run(&mut out, run);
        }
        starts.push(out.len() as u64);
        (Refs::from_bits(out.finish()), starts)
    }

    #[test]
    fn runs_decode_as_they_were_coded_every_way() {
        let runs = runs();
        let (refs, starts) = coded(&runs);
        for (k, want) in runs.iter().enumerate() {
            let run = refs.run(starts[k], want.len());
            assert_eq!(run.first(), want[0]);
            assert_eq!(run.end(), starts[k + 1], "run {k}");
            assert_eq!(run_bits(want), starts[k + 1] - starts[k]);
            assert!(starts[k + 1] - starts[k] <= max_run_bits(want.len() as u64));
            assert_eq!(run.cursor().collect::<Vec<_>>(), *want, "run {k}");
            let mut got = Vec::new();
            run.for_each(|r| got.push(r));
            assert_eq!(got, *want, "run {k}");
            got.clear();
            crate::simd::run(Collect { run, out: &mut got });
            assert_eq!(got, *want, "run {k}, blocks");
            got.clear();
            run.check(Scalar, starts[k + 1], |r| got.push(r)).unwrap();
            assert_eq!(got, *want, "run {k}, checked");
            got.clear();
            crate::simd::run(Checked {
                run,
                end: starts[k + 1],
                out: &mut got,
            })
            .unwrap();
            assert_eq!(got, *want, "run {k}, checked in blocks");
        }
        let total: u64 = runs.iter().map(|r| r.len() as u64).sum();
        assert!(refs.bit_len() <= bound_bits(total, runs.len() as u64));
    }

    /// Reads a run through whatever decoder it is given.
    struct Collect<'a, 'b> {
        run: Run<'a>,
        out: &'b mut Vec<u32>,
    }

    impl crate::simd::Pass for Collect<'_, '_> {
        type Out = ();

        #[inline(always)]
        fn run<D: Decoder>(self, decoder: D) {
            self.run.for_each_in(decoder, |i, r| {
                assert_eq!(i, self.out.len(), "positions count up");
                self.out.push(r);
            });
        }
    }

    /// Checks a run through whatever decoder it is given.
    struct Checked<'a, 'b> {
        run: Run<'a>,
        end: u64,
        out: &'b mut Vec<u32>,
    }

    impl crate::simd::Pass for Checked<'_, '_> {
        type Out = Result<(), &'static str>;

        #[inline(always)]
        fn run<D: Decoder>(self, decoder: D) -> Self::Out {
            self.run.check(decoder, self.end, |r| self.out.push(r))
        }
    }

    #[test]
    fn widths_name_every_field() {
        assert_eq!(width_of(0), 0);
        assert_eq!(width_of(1), 1);
        assert_eq!(width_of((1 << 30) - 1), 30);
        assert_eq!(width_of(1 << 30), 32);
        assert_eq!(width_of(u32::MAX), 32);
        for w in (0..=30).chain([32]) {
            assert_eq!(width_of_code(code_of(w)), w);
        }
        assert_eq!(max_run_bits(1), 37);
        let block = BLOCK as u64;
        assert_eq!(max_run_bits(block), 37 + 5 + (block - 1) * 32);
        assert_eq!(max_run_bits(block + 1), 37 + 5 + block * 32);
        assert_eq!(max_run_bits(block + 2), 37 + 10 + (block + 1) * 32);
    }

    #[test]
    fn a_check_refuses_what_is_not_the_canonical_coding() {
        // A first reference wider than it needs.
        let mut out = PackedBuilder::with_capacity(0, 1);
        out.push_bits(code_of(8), WIDTH_BITS);
        out.push_bits(3, 8);
        let refs = Refs::from_bits(out.finish());
        assert!(refs.run(0, 1).check(Scalar, 13, |_| {}).is_err());
        // A block wider than its widest gap.
        let mut out = PackedBuilder::with_capacity(0, 1);
        out.push_bits(code_of(2), WIDTH_BITS);
        out.push_bits(3, 2);
        out.push_bits(code_of(4), WIDTH_BITS);
        out.push_bits(1, 4);
        let refs = Refs::from_bits(out.finish());
        assert!(refs.run(0, 2).check(Scalar, 16, |_| {}).is_err());
        // The canonical coding of the same run passes, at its own end only.
        let mut out = PackedBuilder::with_capacity(0, 1);
        push_run(&mut out, &[3, 4]);
        let refs = Refs::from_bits(out.finish());
        assert!(refs.run(0, 2).check(Scalar, 13, |_| {}).is_ok());
        assert!(refs.run(0, 2).check(Scalar, 14, |_| {}).is_err());
        // A gap width that would read past the run's end and the stream's.
        let mut out = PackedBuilder::with_capacity(0, 1);
        out.push_bits(code_of(4), WIDTH_BITS);
        out.push_bits(3, 4);
        out.push_bits(code_of(32), WIDTH_BITS);
        let end = out.len() as u64;
        let refs = Refs::from_bits(out.finish());
        assert_eq!(
            refs.run(0, BLOCK).check(Scalar, end, |_| {}),
            Err("a run runs past its end")
        );
        // A gap past the largest id.
        let mut out = PackedBuilder::with_capacity(0, 1);
        push_run(&mut out, &[u32::MAX - 1]);
        out.push_bits(code_of(2), WIDTH_BITS);
        out.push_bits(2, 2);
        let refs = Refs::from_bits(out.finish());
        let end = refs.bit_len();
        assert!(refs.run(0, 2).check(Scalar, end, |_| {}).is_err());
    }

    #[test]
    fn a_check_refuses_references_that_descend_across_blocks() {
        // Width 4; a first block of fives, then a block whose base is 3.
        let mut out = PackedBuilder::with_capacity(0, 1);
        out.push_bits(code_of(4), WIDTH_BITS);
        out.push_bits(5, 4);
        out.push_bits(code_of(0), WIDTH_BITS);
        out.push_bits(3, 4);
        let end = out.len() as u64;
        let refs = Refs::from_bits(out.finish());
        assert_eq!(
            refs.run(0, BLOCK + 1).check(Scalar, end, |_| {}),
            Err("a run's references descend")
        );
    }

    /// Blocks passed over are skipped whole and the rest decode with their
    /// positions, every way.
    #[test]
    fn blocks_are_offered_with_bounds_and_skipped() {
        struct Where<'a, 'b> {
            run: Run<'a>,
            keep: u32,
            offered: &'b mut Vec<(u32, u32)>,
            out: &'b mut Vec<(usize, u32)>,
        }
        impl crate::simd::Pass for Where<'_, '_> {
            type Out = ();
            #[inline(always)]
            fn run<D: Decoder>(self, decoder: D) {
                let keep = self.keep;
                self.run.for_each_where(
                    decoder,
                    |lo, hi| {
                        self.offered.push((lo, hi));
                        lo <= keep && keep <= hi
                    },
                    |i, r| self.out.push((i, r)),
                );
            }
        }
        let runs = runs();
        let (refs, starts) = coded(&runs);
        for (k, want) in runs.iter().enumerate() {
            let run = refs.run(starts[k], want.len());
            for keep in [want[0], want[want.len() / 2], want[want.len() - 1]] {
                let (mut offered, mut out) = (Vec::new(), Vec::new());
                crate::simd::run(Where {
                    run,
                    keep,
                    offered: &mut offered,
                    out: &mut out,
                });
                // One offer per block, each bounding its own references.
                let blocks: Vec<&[u32]> = want.chunks(BLOCK).collect();
                assert_eq!(offered.len(), blocks.len(), "run {k}");
                for (b, (&(lo, hi), block)) in offered.iter().zip(&blocks).enumerate() {
                    assert_eq!(lo, block[0]);
                    assert!(
                        block.iter().all(|&r| lo <= r && r <= hi),
                        "run {k} block {b}"
                    );
                }
                // Exactly the references of the accepted blocks, with
                // their positions.
                let expected: Vec<(usize, u32)> = want
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(i, _)| {
                        let (lo, hi) = offered[i / BLOCK];
                        lo <= keep && keep <= hi
                    })
                    .collect();
                assert_eq!(out, expected, "run {k}");
                assert!(out.iter().any(|&(_, r)| r == keep), "run {k} finds {keep}");
            }
        }
    }

    #[test]
    fn touch_stays_in_bounds() {
        let (refs, starts) = coded(&runs());
        for w in starts.windows(2) {
            for reach in [0, 7, WINDOW - 1, usize::MAX] {
                refs.touch(w[0], w[1] - w[0], reach);
                refs.touch(w[0], 1 << 40, reach);
            }
        }
        let empty = Refs::default();
        assert_eq!(empty.bit_len(), 0);
        empty.touch(0, 0, 8);
    }
}
