//! Fixed-width integers packed back to back.

use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

/// Unsigned integers of one width, `bits` each, packed back to back in
/// little-endian bit order. Reading one is an unaligned 64-bit load, a
/// shift and a mask; eight slack bytes at the end keep every such window
/// inside the buffer. Width 0 holds only zeros and no data bytes.
#[derive(Clone)]
pub(crate) struct Packed {
    bytes: Bytes,
    bits: u32,
    mask: u64,
    len: usize,
}

/// Where a packed array's bytes live: a vector of its own, or a window of
/// a memory-mapped snapshot (`mmap`), slack included in both.
#[derive(Clone)]
enum Bytes {
    Owned(Vec<u8>),
    #[cfg(feature = "mmap")]
    Mapped {
        map: alloc::sync::Arc<memmap2::Mmap>,
        start: usize,
        len: usize,
    },
}

impl Bytes {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            #[cfg(feature = "mmap")]
            Self::Mapped { map, start, len } => &map[*start..*start + *len],
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Owned(v) => v.capacity(),
            #[cfg(feature = "mmap")]
            Self::Mapped { .. } => 0,
        }
    }
}

/// Mask of the low `bits` bits.
const fn mask_of(bits: u32) -> u64 {
    if bits == 0 {
        0
    } else {
        u64::MAX >> (64 - bits)
    }
}

impl Packed {
    /// Bytes past the data so the 64-bit window at the last value is in
    /// bounds.
    pub(crate) const SLACK: usize = 8;

    /// Width that holds every value up to `max`.
    pub(crate) const fn bits_for(max: u32) -> u32 {
        u32::BITS - max.leading_zeros()
    }

    /// Data bytes for `len` values of `bits` (saturating, so an absurd
    /// count becomes an absurd length rather than a small one).
    pub(crate) fn data_len(len: usize, bits: u32) -> usize {
        let bits_total = len as u128 * u128::from(bits);
        usize::try_from(bits_total.div_ceil(8)).unwrap_or(usize::MAX)
    }

    /// Wrap the data bytes of `len` values of `bits`, as written by
    /// [`Packed::data`]; `None` if the length or width is wrong.
    #[cfg(any(feature = "std", test))]
    pub(crate) fn from_bytes(mut bytes: Vec<u8>, len: usize, bits: u32) -> Option<Self> {
        if bits > u32::BITS || bytes.len() != Self::data_len(len, bits) {
            return None;
        }
        bytes.resize(bytes.len() + Self::SLACK, 0);
        Some(Self {
            bytes: Bytes::Owned(bytes),
            bits,
            mask: mask_of(bits),
            len,
        })
    }

    /// Wrap `len` values of `bits` whose data bytes (as written by
    /// [`Packed::data`]) start at `map[start]`; `None` if the width is
    /// wrong or the data plus the slack runs past the mapping.
    #[cfg(feature = "mmap")]
    pub(crate) fn mapped(
        map: alloc::sync::Arc<memmap2::Mmap>,
        start: usize,
        len: usize,
        bits: u32,
    ) -> Option<Self> {
        if bits > u32::BITS {
            return None;
        }
        let window = Self::data_len(len, bits).checked_add(Self::SLACK)?;
        if start.checked_add(window)? > map.len() {
            return None;
        }
        Some(Self {
            bytes: Bytes::Mapped {
                map,
                start,
                len: window,
            },
            bits,
            mask: mask_of(bits),
            len,
        })
    }

    /// Pack `values` at the width of the largest.
    #[cfg(test)]
    pub(crate) fn from_values(values: &[u32]) -> Self {
        let bits = Self::bits_for(values.iter().copied().max().unwrap_or(0));
        Self::pack(values, bits, values.len())
    }

    /// Pack `values` at `bits` through a builder with room for `capacity`.
    #[cfg(test)]
    fn pack(values: &[u32], bits: u32, capacity: usize) -> Self {
        let mut out = PackedBuilder::with_capacity(capacity, bits);
        out.extend(values.iter().copied());
        assert_eq!(out.len(), values.len());
        out.finish()
    }

    /// Value `i`.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> u32 {
        debug_assert!(i < self.len, "index {i} past {}", self.len);
        let bit = i as u64 * u64::from(self.bits);
        let byte = (bit >> 3) as usize;
        let bytes = self.bytes.as_slice();
        let w = u64::from_le_bytes(bytes[byte..byte + 8].try_into().expect("8 bytes"));
        ((w >> (bit & 7)) & self.mask) as u32
    }

    /// The values behind one borrow of the bytes, for loops that read many:
    /// where the bytes live is decided once rather than per value.
    #[inline]
    pub(crate) fn view(&self) -> View<'_> {
        View {
            bytes: self.bytes.as_slice(),
            bits: u64::from(self.bits),
            mask: self.mask,
            len: self.len,
        }
    }

    /// Call `f` with entries `lo..hi` in order. A streaming decode: one
    /// word load every few entries instead of one per entry, and no
    /// per-entry bounds check.
    #[inline]
    pub(crate) fn for_each(&self, lo: usize, hi: usize, mut f: impl FnMut(u32)) {
        debug_assert!(lo <= hi && hi <= self.len, "{lo}..{hi} past {}", self.len);
        if lo >= hi {
            return;
        }
        let bits = u64::from(self.bits);
        let mask = self.mask;
        let start = lo as u64 * bits;
        let mut byte = (start >> 3) as usize;
        // Bits of the word already consumed; at most 32 at the top of the
        // loop, so a 32-bit entry always lies inside the word.
        let mut shift = start & 7;
        let bytes = self.bytes.as_slice();
        let mut w = u64::from_le_bytes(bytes[byte..byte + 8].try_into().expect("8 bytes"));
        for _ in lo..hi {
            f(((w >> shift) & mask) as u32);
            shift += bits;
            if shift > 32 {
                byte += (shift >> 3) as usize;
                shift &= 7;
                w = u64::from_le_bytes(bytes[byte..byte + 8].try_into().expect("8 bytes"));
            }
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) const fn bits(&self) -> u32 {
        self.bits
    }

    /// The data bytes, without the slack.
    #[cfg(any(feature = "std", test))]
    pub(crate) fn data(&self) -> &[u8] {
        &self.bytes.as_slice()[..Self::data_len(self.len, self.bits)]
    }

    /// Bytes owned on the heap; a mapped array owns none.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.bytes.heap_bytes()
    }

    /// Values `range`, in order.
    #[inline]
    pub(crate) fn iter(&self, range: Range<usize>) -> impl ExactSizeIterator<Item = u32> + '_ {
        range.map(move |i| self.get(i))
    }

    pub(crate) fn max(&self) -> Option<u32> {
        let mut max = None;
        self.for_each(0, self.len, |v| max = max.max(Some(v)));
        max
    }

    /// First index in `lo..hi` whose value fails `pred`, or `hi`; the
    /// values in the range must be partitioned by `pred`.
    #[inline]
    pub(crate) fn partition_point(
        &self,
        mut lo: usize,
        mut hi: usize,
        pred: impl Fn(u32) -> bool,
    ) -> usize {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.get(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// A borrowed [`Packed`]: the same values, with the bytes already located.
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    bytes: &'a [u8],
    bits: u64,
    mask: u64,
    len: usize,
}

impl View<'_> {
    /// Value `i`.
    #[inline]
    pub(crate) fn get(&self, i: usize) -> u32 {
        debug_assert!(i < self.len, "index {i} past {}", self.len);
        (self.window(i as u64 * self.bits) & self.mask) as u32
    }

    /// The bits from position `bit` on, in the low end of a word: at least
    /// 57 of them are data (or slack past the last value).
    #[inline]
    fn window(&self, bit: u64) -> u64 {
        let byte = (bit >> 3) as usize;
        let w = u64::from_le_bytes(self.bytes[byte..byte + 8].try_into().expect("8 bytes"));
        w >> (bit & 7)
    }
}

/// Appends values of one width and finishes into a [`Packed`].
///
/// Values gather in a 64-bit accumulator that is written out 32 bits at a
/// time, so a push is a shift and an or rather than a load, a merge and a
/// store on bytes the previous push has just written.
pub(crate) struct PackedBuilder {
    bytes: Vec<u8>,
    /// Bits appended but not yet written: the low `fill` bits, `fill < 32`.
    acc: u64,
    fill: u32,
    bits: u32,
    mask: u64,
    /// Values appended so far.
    len: usize,
}

impl PackedBuilder {
    /// Room for `len` values of `bits`; appends past it grow the buffer.
    pub(crate) fn with_capacity(len: usize, bits: u32) -> Self {
        assert!(bits <= u32::BITS, "width of {bits} bits is above 32");
        let room = Packed::data_len(len, bits).saturating_add(Packed::SLACK);
        Self {
            bytes: Vec::with_capacity(room),
            acc: 0,
            fill: 0,
            bits,
            mask: mask_of(bits),
            len: 0,
        }
    }

    /// Values appended so far.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Append `v`, which must fit the width. For one value at a time;
    /// [`PackedBuilder::extend`] keeps a loop's state in registers.
    #[inline]
    pub(crate) fn push(&mut self, v: u32) {
        self.extend(core::iter::once(v));
    }

    /// Append `values`, each of which must fit the width.
    ///
    /// The accumulator is a local for the length of the loop, so it stays
    /// in registers; keep a caller's loop to one `extend` at a time.
    #[inline]
    pub(crate) fn extend(&mut self, values: impl Iterator<Item = u32>) {
        let (mut acc, mut fill, mut pushed) = (self.acc, self.fill, 0usize);
        for v in values {
            debug_assert!(
                u64::from(v) <= self.mask,
                "{v} does not fit {} bits",
                self.bits
            );
            // Masked as well as asserted: a value too wide for the width
            // must not spill into its neighbours in a release build. At
            // most 31 pending bits and 32 new ones, so nothing is lost.
            acc |= (u64::from(v) & self.mask) << fill;
            fill += self.bits;
            if fill >= u32::BITS {
                self.bytes.extend_from_slice(&(acc as u32).to_le_bytes());
                acc >>= u32::BITS;
                fill -= u32::BITS;
            }
            pushed += 1;
        }
        (self.acc, self.fill) = (acc, fill);
        self.len += pushed;
    }

    /// Append values `lo..hi` of `src`.
    ///
    /// At the same width a stretch of values is never decoded: its bits
    /// are copied from wherever they start in `src` to wherever this array
    /// ends. Decoding is the only way across widths, and the cheaper one
    /// for a few values.
    #[inline]
    pub(crate) fn extend_from(&mut self, src: View<'_>, lo: usize, hi: usize) {
        debug_assert!(lo <= hi && hi <= src.len, "{lo}..{hi} past {}", src.len);
        if src.bits == u64::from(self.bits) && hi - lo >= Self::COPY_MIN {
            self.extend_bits(src, lo, hi);
        } else {
            self.extend((lo..hi).map(|i| src.get(i)));
        }
    }

    /// Values from which a stretch is copied as bits rather than decoded.
    const COPY_MIN: usize = 4;

    /// Append the bits of values `lo..hi` of `src`, which has this
    /// builder's width, 32 at a time.
    fn extend_bits(&mut self, src: View<'_>, lo: usize, hi: usize) {
        /// Bytes gathered on the stack between appends, so the destination
        /// is written once and its length updated once per block.
        const BLOCK: usize = 512;
        let total = (hi - lo) as u64 * src.bits;
        let mut bit = lo as u64 * src.bits;
        // Whole 32-bit steps leave the count of pending bits as it is.
        let (mut acc, fill) = (self.acc, self.fill);
        let mut steps = (total / u64::from(u32::BITS)) as usize;
        let mut block = [0u8; BLOCK];
        while steps > 0 {
            let n = steps.min(BLOCK / 4);
            for out in block[..4 * n].chunks_exact_mut(4) {
                acc |= (src.window(bit) & u64::from(u32::MAX)) << fill;
                out.copy_from_slice(&(acc as u32).to_le_bytes());
                acc >>= u32::BITS;
                bit += u64::from(u32::BITS);
            }
            self.bytes.extend_from_slice(&block[..4 * n]);
            steps -= n;
        }
        (self.acc, self.len) = (acc, self.len + (hi - lo));
        let rest = (total % u64::from(u32::BITS)) as u32;
        if rest > 0 {
            self.acc |= (src.window(bit) & mask_of(rest)) << fill;
            self.fill += rest;
            if self.fill >= u32::BITS {
                self.bytes
                    .extend_from_slice(&(self.acc as u32).to_le_bytes());
                self.acc >>= u32::BITS;
                self.fill -= u32::BITS;
            }
        }
    }

    /// Write out the pending bits and the slack, and drop the capacity
    /// beyond them: a capacity hint can overshoot.
    pub(crate) fn finish(mut self) -> Packed {
        let pending = self.fill.div_ceil(8) as usize;
        self.bytes
            .extend_from_slice(&self.acc.to_le_bytes()[..pending]);
        debug_assert_eq!(self.bytes.len(), Packed::data_len(self.len, self.bits));
        self.bytes.resize(self.bytes.len() + Packed::SLACK, 0);
        self.bytes.shrink_to_fit();
        Packed {
            bytes: Bytes::Owned(self.bytes),
            bits: self.bits,
            mask: self.mask,
            len: self.len,
        }
    }
}

impl Default for Packed {
    fn default() -> Self {
        PackedBuilder::with_capacity(0, 0).finish()
    }
}

impl PartialEq for Packed {
    /// Same values, whatever the widths.
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter(0..self.len).eq(other.iter(0..other.len))
    }
}

impl Eq for Packed {}

impl fmt::Debug for Packed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Packed")
            .field("len", &self.len)
            .field("bits", &self.bits)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_width() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for bits in 0..=32u32 {
            let mask = if bits == 0 {
                0
            } else {
                u64::MAX >> (64 - bits)
            };
            let values: Vec<u32> = (0..1_000)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x & mask) as u32
                })
                .collect();
            // Through a small capacity, so pushes grow the buffer.
            let p = Packed::pack(&values, bits, 3);
            assert_eq!(p.len(), values.len());
            assert_eq!(p.bits(), bits);
            assert_eq!(p.iter(0..p.len()).collect::<Vec<_>>(), values);
            assert_eq!(p.data().len(), Packed::data_len(values.len(), bits));
            assert_eq!(p.heap_bytes(), p.data().len() + Packed::SLACK);
            let back = Packed::from_bytes(p.data().to_vec(), values.len(), bits).unwrap();
            assert_eq!(back, p);
            assert_eq!(back.max(), values.iter().copied().max());
            // A count the bytes cannot hold is refused (width 0 holds any).
            if bits > 0 {
                assert!(Packed::from_bytes(p.data().to_vec(), values.len() + 64, bits).is_none());
            }
        }
        assert!(Packed::from_bytes(Vec::new(), 0, 33).is_none());
        assert_eq!(Packed::bits_for(0), 0);
        assert_eq!(Packed::bits_for(1), 1);
        assert_eq!(Packed::bits_for(u32::MAX), 32);
    }

    /// Values appended in several calls, decoded or as raw bits, pack
    /// exactly as they do in one: every width, every alignment of the
    /// source range and of the destination's end.
    #[test]
    fn appends_resume_where_the_last_one_stopped() {
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        for bits in 0..=32u32 {
            let values: Vec<u32> = (0..300)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x & mask_of(bits)) as u32
                })
                .collect();
            let whole = Packed::pack(&values, bits, values.len());
            // The same width copies bits; a wider destination decodes.
            for out_bits in [bits, 32] {
                let mut b = PackedBuilder::with_capacity(0, out_bits);
                let (mut at, mut step) = (0, 0);
                while at < values.len() {
                    // Stretches of 0, 1, 2, ... values, alternately decoded
                    // and copied from the packed source.
                    let to = (at + step).min(values.len());
                    if step % 2 == 0 {
                        b.extend(values[at..to].iter().copied());
                    } else {
                        b.extend_from(whole.view(), at, to);
                    }
                    (at, step) = (to, step + 1);
                }
                assert_eq!(b.len(), values.len());
                let got = b.finish();
                assert_eq!(got.bits(), out_bits);
                assert_eq!(got, whole, "{bits} bits into {out_bits}");
                if out_bits == bits {
                    assert_eq!(got.data(), whole.data(), "{bits} bits");
                }
            }
        }
    }

    #[test]
    fn for_each_matches_get_at_every_width() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for bits in [0u32, 1, 7, 13, 17, 24, 31, 32] {
            let mask = if bits == 0 {
                0
            } else {
                u64::MAX >> (64 - bits)
            };
            let values: Vec<u32> = (0..500)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x & mask) as u32
                })
                .collect();
            let p = Packed::pack(&values, bits, values.len());
            for (lo, hi) in [
                (0, 0),
                (0, 1),
                (0, 500),
                (3, 4),
                (5, 499),
                (123, 321),
                (499, 500),
            ] {
                let mut got = Vec::new();
                p.for_each(lo, hi, |v| got.push(v));
                assert_eq!(got, values[lo..hi], "bits {bits}, {lo}..{hi}");
            }
        }
    }

    #[test]
    fn partition_point_and_value_equality() {
        let p = Packed::from_values(&[1, 3, 3, 3, 8, 9]);
        assert_eq!(p.partition_point(0, 6, |x| x < 3), 1);
        assert_eq!(p.partition_point(1, 6, |x| x == 3), 4);
        assert_eq!(p.partition_point(0, 6, |x| x < 100), 6);
        assert_eq!(p.partition_point(2, 2, |_| true), 2);
        let wide = Packed::pack(&[1, 3, 3, 3, 8, 9], 32, 6);
        assert_eq!(p, wide);
        assert_ne!(p, Packed::from_values(&[1, 3, 3, 3, 8]));
        assert!(Packed::default().is_empty());
        assert!(alloc::format!("{p:?}").starts_with("Packed { len: 6, bits: 4"));
    }

    /// A value wider than the width is a caller bug (asserted in debug);
    /// in release it is masked rather than spilled into its neighbours.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "does not fit 4 bits"))]
    fn push_masks_a_value_wider_than_the_width() {
        let p = Packed::pack(&[0xF, 0x1F5, 0x3], 4, 3);
        assert_eq!(p.iter(0..3).collect::<Vec<_>>(), [0xF, 0x5, 0x3]);
    }
}
