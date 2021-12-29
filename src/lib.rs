//! # Spectral density estimation with Welch method
//!
//! The Welch's method for the estimation of the spectral density of a *stationnary and
//! zero-mean* signal  involves splitting the signal in overlapping segments, multiplying
//! each segment with a predeterming window and computing the discrete Fourier transform
//! of each windowed segment.
//! The spectral density is then given by the average of the squared magnitude
//! of each Fourier transform.
//!
//! Assuming the signal is divided into `k` segments, each of length `l`, and each segment
//! overlap a fraction `a` of  the previous segment, the relation of `k`, `l` and `a` to
//! the signal length `n` is `n=kl - (k-1)la`.
//!
//! The minimum number of segment is chosen to be `k=4`, and the segment length is derived
//! from `l = trunc(n/(k(1-a)+a))`.
//!
//! Each segment of length `l` is  multiplied by the predetermined window and zero--padded
//! to the size `m = 2^p` where `p=ceil(log2(l))`.
//! The maximum allowed value for `p` is 12 (i.e. `m=4096`).
//! If with only 4 segments (`k=4`), `l` is greater than 4096, then `l` is set to 4096 and
//! the increased number of segments is derived from `k=(n-la)/(l(1-a))`.

mod welch;
mod window;
use num_traits::Float;
use rustfft::FftNum;
use std::ops::Deref;
pub use welch::{Builder, Welch};
pub use window::{Hann, Window};

/// The trait the signal type `T` must implement
pub trait Signal:
    Float + FftNum + std::iter::Sum + std::ops::SubAssign + std::ops::AddAssign
{
}
impl Signal for f64 {}
impl Signal for f32 {}

/// Signal spectral density
///
/// The spectral density is given in units of the signal units squared per Hertz
#[derive(Debug)]
pub struct SpectralDensity<T: Signal>((T, Vec<T>));
impl<T: Signal> Deref for SpectralDensity<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0 .1.as_slice()
    }
}
impl<T: Signal> SpectralDensity<T> {
    /// Returns the frequency vector in Hz for the given signal sampling frequency
    pub fn frequency(&self) -> Vec<T> {
        let n = self.0 .1.len();
        let fs = self.0 .0;
        (0..n)
            .map(|i| {
                T::from_usize(i).unwrap() * fs * T::from_f32(0.5).unwrap()
                    / T::from_usize(n - 1).unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn welch() {
        let n = 2000;
        let k = 8;
        let signal = vec![1f64; n];
        let welch: Welch<f64, Hann<f64>> = Welch::builder(&signal).n_segment(k).build();
        let ps = welch.power_spectrum();
        assert!(
            (1. - welch.window.sum_sqr() / welch.window.sqr_sum() / ps[0]).abs()
                < 10. * f64::EPSILON
        );
    }
}
