//! Eight packed values at a time, on vector instructions where the CPU
//! has them.
//!
//! A pass that reads many values of a [`Packed`](crate::packed::Packed)
//! array is written once, generic over a [`Decoder`], and started through
//! [`run`]. Without the `simd` feature, off x86-64, or on a CPU without
//! AVX2, the decoder is [`Scalar`] and the pass reads one value at a time
//! exactly as it would without this module. With all three, the whole pass
//! is compiled a second time with AVX2 enabled and handed a decoder whose
//! kernel unpacks a block of eight values with one byte shuffle and one
//! variable shift.
//!
//! The kernel itself is safe code: it builds its vectors from byte slices
//! and takes no pointer. Entering code compiled for AVX2 is the one thing
//! the compiler cannot check, so the call in [`run`], right after the CPU
//! has been asked, is the `unsafe` block of the `simd` feature.

/// Values per block.
pub(crate) const LANES: usize = 8;
/// Bytes a kernel may read from the byte holding a block's first value.
pub(crate) const WINDOW: usize = 32;

/// How a pass decodes blocks of a packed array: a kernel for a width and a
/// starting bit, or none, in which case the pass reads values one by one.
pub(crate) trait Decoder: Copy {
    /// Whether this decoder has kernels at all. A pass whose scalar form
    /// is better left as it was written branches on this.
    const BLOCKS: bool;

    /// Decodes the [`LANES`] values whose first one starts `phase` bits
    /// into the window. Lanes past the end of the array hold whatever the
    /// bytes there decode to.
    type Kernel: Fn(&[u8; WINDOW]) -> [u32; LANES];

    /// The kernel for values of `bits` when the first value of every
    /// block starts `phase` bits into its byte, if this decoder has one.
    /// A block is `bits` whole bytes, so the phase holds for a whole run.
    fn kernel(self, bits: u32, phase: usize) -> Option<Self::Kernel>;
}

/// The decoder without a kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scalar;

impl Decoder for Scalar {
    const BLOCKS: bool = false;

    type Kernel = fn(&[u8; WINDOW]) -> [u32; LANES];

    #[inline(always)]
    fn kernel(self, _: u32, _: usize) -> Option<Self::Kernel> {
        None
    }
}

/// Work over packed arrays that is the same whichever decoder reads them.
pub(crate) trait Pass {
    /// What the pass returns.
    type Out;

    /// Do the work. Implementations are `#[inline(always)]`, and so is
    /// everything between them and the decoder's kernel: the kernel can
    /// only be inlined into code compiled with its instructions enabled,
    /// which is the body of the caller in this module and nothing else.
    fn run<D: Decoder>(self, decoder: D) -> Self::Out;
}

/// Run `pass` with the best decoder this CPU has.
#[inline]
pub(crate) fn run<P: Pass>(pass: P) -> P::Out {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        #[expect(
            unsafe_code,
            reason = "entering code compiled for AVX2 cannot be checked by the compiler"
        )]
        // SAFETY: the CPU has just reported AVX2, the only feature
        // `avx2::run` is compiled with beyond the crate's own.
        return unsafe { avx2::run(pass) };
    }
    pass.run(Scalar)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm_set_epi64x, _mm256_and_si256, _mm256_extract_epi64, _mm256_set_epi64x,
        _mm256_set_m128i, _mm256_set1_epi32, _mm256_shuffle_epi8, _mm256_srlv_epi32,
    };

    use super::{Decoder, LANES, Pass, WINDOW};

    /// Widest value the kernel decodes: a value and the bits before it in
    /// its first byte must fit a 32-bit lane.
    const MAX_BITS: usize = 25;

    /// Where the eight values of a block lie, for one width and one
    /// starting bit.
    ///
    /// Values 0 to 3 are taken from the 16 bytes at the window's start and
    /// values 4 to 7 from the 16 bytes at `high`: a byte shuffle moves the
    /// four bytes each value starts in to its lane, within its half, and a
    /// shift per lane drops the bits before it.
    #[derive(Clone, Copy)]
    struct Phase {
        /// Byte shuffle control, as four little-endian words.
        shuffle: [i64; 4],
        /// Right shift per lane, two lanes to a word.
        shift: [i64; 4],
        /// Byte of the window the high half is loaded from, at most 13.
        high: usize,
    }

    const fn phase(bits: usize, start: usize) -> Phase {
        let high = (start + 4 * bits) >> 3;
        let mut shuffle = [0u8; 32];
        let mut shift = [0u64; LANES];
        let mut lane = 0;
        while lane < LANES {
            let bit = start + lane * bits;
            let from = (bit >> 3) - if lane < 4 { 0 } else { high };
            // Three values of at most 25 bits past a start of at most 7
            // begin in byte 10 or before, so the four bytes are in the half.
            assert!(from + 3 < 16);
            let mut byte = 0;
            while byte < 4 {
                shuffle[lane * 4 + byte] = (from + byte) as u8;
                byte += 1;
            }
            shift[lane] = (bit & 7) as u64;
            lane += 1;
        }
        assert!(high + 16 <= WINDOW);
        let mut words = Phase {
            shuffle: [0; 4],
            shift: [0; 4],
            high,
        };
        let mut word = 0;
        while word < 4 {
            let mut bytes = [0u8; 8];
            let mut byte = 0;
            while byte < 8 {
                bytes[byte] = shuffle[word * 8 + byte];
                byte += 1;
            }
            words.shuffle[word] = i64::from_le_bytes(bytes);
            words.shift[word] = (shift[word * 2] | shift[word * 2 + 1] << 32) as i64;
            word += 1;
        }
        words
    }

    /// Every width from 1 to [`MAX_BITS`] at each of the eight starting
    /// bits; row 0 is unused (width 0 has no bytes to decode).
    static PHASES: [[Phase; 8]; MAX_BITS + 1] = {
        let mut phases = [[Phase {
            shuffle: [0; 4],
            shift: [0; 4],
            high: 0,
        }; 8]; MAX_BITS + 1];
        let mut bits = 1;
        while bits <= MAX_BITS {
            let mut start = 0;
            while start < 8 {
                phases[bits][start] = phase(bits, start);
                start += 1;
            }
            bits += 1;
        }
        phases
    };

    /// A decoder whose kernels `prepare` builds: a closure made where AVX2
    /// is enabled, which is what lets it call the kernel below.
    #[derive(Clone, Copy)]
    struct Avx2<F>(F);

    impl<F, K> Decoder for Avx2<F>
    where
        F: Fn(&'static Phase, u32) -> K + Copy,
        K: Fn(&[u8; WINDOW]) -> [u32; LANES],
    {
        const BLOCKS: bool = true;
        type Kernel = K;

        #[inline(always)]
        fn kernel(self, bits: u32, phase: usize) -> Option<K> {
            let phases = PHASES.get(bits as usize).filter(|_| bits > 0)?;
            Some((self.0)(&phases[phase & 7], u32::MAX >> (u32::BITS - bits)))
        }
    }

    /// Run `pass` compiled for AVX2. The CPU must have it.
    #[target_feature(enable = "avx2")]
    pub(super) fn run<P: Pass>(pass: P) -> P::Out {
        pass.run(Avx2(|phase: &'static Phase, mask: u32| {
            let [s0, s1, s2, s3] = phase.shuffle;
            let shuffle = _mm256_set_epi64x(s3, s2, s1, s0);
            let [h0, h1, h2, h3] = phase.shift;
            let shift = _mm256_set_epi64x(h3, h2, h1, h0);
            let mask = _mm256_set1_epi32(mask as i32);
            let high = phase.high;
            move |window: &[u8; WINDOW]| unpack(window, high, shuffle, shift, mask)
        }))
    }

    /// The little-endian word at `at`.
    #[inline(always)]
    fn word(window: &[u8; WINDOW], at: usize) -> i64 {
        i64::from_le_bytes(window[at..at + 8].try_into().expect("8 bytes"))
    }

    /// Decode the eight values of a block: see [`Phase`].
    #[target_feature(enable = "avx2")]
    #[inline]
    fn unpack(
        window: &[u8; WINDOW],
        high: usize,
        shuffle: __m256i,
        shift: __m256i,
        mask: __m256i,
    ) -> [u32; LANES] {
        // At most 13; the mask lets the compiler see the loads are in the
        // window.
        let high = high & 15;
        let low = _mm_set_epi64x(word(window, 8), word(window, 0));
        let high = _mm_set_epi64x(word(window, high + 8), word(window, high));
        let bytes = _mm256_shuffle_epi8(_mm256_set_m128i(high, low), shuffle);
        let values = _mm256_and_si256(_mm256_srlv_epi32(bytes, shift), mask);
        let pairs = [
            _mm256_extract_epi64::<0>(values) as u64,
            _mm256_extract_epi64::<1>(values) as u64,
            _mm256_extract_epi64::<2>(values) as u64,
            _mm256_extract_epi64::<3>(values) as u64,
        ];
        core::array::from_fn(|lane| (pairs[lane / 2] >> (32 * (lane % 2))) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::Rng;
    use alloc::vec::Vec;

    use crate::packed::Packed;

    /// Reads every value of a range through whatever decoder it is given.
    struct Collect<'a> {
        packed: &'a Packed,
        lo: usize,
        hi: usize,
    }

    impl Pass for Collect<'_> {
        type Out = (Vec<u32>, usize);

        #[inline(always)]
        fn run<D: Decoder>(self, decoder: D) -> Self::Out {
            let mut values = Vec::new();
            let mut blocks = 0;
            let done = self
                .packed
                .for_each_block(self.lo, self.hi, decoder, |at, block, n| {
                    assert_eq!(at, self.lo + values.len());
                    values.extend_from_slice(&block[..n]);
                    blocks += 1;
                });
            assert_eq!(done, self.lo + values.len());
            self.packed
                .for_each_in(done, self.hi, decoder, |v| values.push(v));
            (values, blocks)
        }
    }

    /// Blocks agree with single reads at every width, at every starting
    /// bit, and up to the last value of the array, where the window runs
    /// out and the values come one at a time.
    #[test]
    fn blocks_agree_with_single_reads_at_every_width() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut vector_blocks = 0;
        for bits in 0..=32u32 {
            let values: Vec<u32> = (0..700).map(|_| rng.bits(bits)).collect();
            let packed = Packed::pack(&values, bits, values.len());
            let ranges = (0..40).map(|lo| (lo, lo + 3 * lo)).chain([
                (0, 0),
                (0, 700),
                (5, 699),
                (650, 700),
                (699, 700),
            ]);
            for (lo, hi) in ranges {
                let pass = Collect {
                    packed: &packed,
                    lo,
                    hi,
                };
                let (got, blocks) = run(pass);
                assert_eq!(got, values[lo..hi], "{bits} bits, {lo}..{hi}");
                vector_blocks += blocks;
                let pass = Collect {
                    packed: &packed,
                    lo,
                    hi,
                };
                let (got, blocks) = pass.run(Scalar);
                assert_eq!(got, values[lo..hi], "{bits} bits, {lo}..{hi}, scalar");
                assert_eq!(blocks, 0);
            }
        }
        // Where there is a kernel, it did the work.
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        assert_eq!(vector_blocks > 0, std::is_x86_feature_detected!("avx2"));
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        assert_eq!(vector_blocks, 0);
    }
}
