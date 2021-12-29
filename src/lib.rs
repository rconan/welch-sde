//! # Spectral density and power spectrum estimation with Welch method
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
//!
//! ## Examples
//! ### Spectral density
//! ```
//!use rand::prelude::*;
//!use rand_distr::StandardNormal;
//!use std::time::Instant;
//!use welch_sde::SpectralDensity;
//!
//!fn main() {
//!    let n = 1e5 as usize;
//!    let fs = 10e3_f64;
//!    let amp = 2. * 2f64.sqrt();
//!    let freq = 1550f64;
//!    let noise_power = 0.001 * fs / 2.;
//!    let sigma = noise_power.sqrt();
//!    let signal: Vec<f64> = (0..n)
//!        .map(|i| i as f64 / fs)
//!        .map(|t| {
//!            amp * (2. * std::f64::consts::PI * freq * t).sin()
//!                + thread_rng().sample::<f64, StandardNormal>(StandardNormal) * sigma
//!        })
//!        .collect();
//!
//!    let welch = SpectralDensity::builder(&signal, fs).build();
//!    println!("{}", welch);
//!    let now = Instant::now();
//!    let sd = welch.spectral_density();
//!    println!(
//!        "Spectral density estimated in {}ms",
//!        now.elapsed().as_millis()
//!    );
//!    let h = sd.len() / 2;
//!    let noise_floor = 2. * sd.iter().skip(h).cloned().sum::<f64>() / (h as f64);
//!    println!("Noise floor: {:.3e}", noise_floor);
//!
//!    let _: complot::LinLog = (
//!        sd.frequency()
//!            .into_iter()
//!            .zip(&(*sd))
//!            .map(|(x, &y)| (x, vec![y])),
//!        complot::complot!(
//!            "spectral_density.png",
//!            xlabel = "Frequency [Hz]",
//!            ylabel = "Spectral density [s^2/Hz]"
//!        ),
//!    )
//!        .into();
//!}
//!```
//! ### Power spectrum
//!```
//!use rand::prelude::*;
//!use rand_distr::StandardNormal;
//!use std::time::Instant;
//!use welch_sde::PowerSpectrum;
//!
//!fn main() {
//!    let n = 1e5 as usize;
//!    let signal: Vec<f64> = (0..n)
//!        .map(|_| thread_rng().sample::<f64, StandardNormal>(StandardNormal))
//!        .collect();
//!
//!    let welch = PowerSpectrum::builder(&signal).build();
//!    println!("{}", welch);
//!
//!    let now = Instant::now();
//!    let ps = welch.power_spectrum();
//!    println!(
//!        "Power spectrum estimated in {}ms",
//!        now.elapsed().as_millis()
//!    );
//!
//!    let mean = ps[0];
//!    let variance = 2. * ps.iter().sum::<f64>();
//!    println!("mean    : {:.3e}", mean);
//!    println!("variance: {:.3e}", variance);
//!
//!    let _: complot::LinLog = (
//!        ps.frequency()
//!            .into_iter()
//!            .zip(&(*ps))
//!            .map(|(x, &y)| (x, vec![y])),
//!        complot::complot!(
//!            "power_spectrum.png",
//!            xlabel = "Frequency [Hz]",
//!            ylabel = "Power spectrum [s^2]"
//!        ),
//!    )
//!        .into();
//!}
//!```

mod welch;
mod window;
use num_traits::Float;
use rustfft::FftNum;
use std::{marker::PhantomData, ops::Deref};
pub use welch::{Builder, Welch};
pub use window::{Hann, One, Window};

/// Power spectrum default type
pub type PowerSpectrum<'a, T> = Welch<'a, T, One<T>>;
/// Spectral density default type
pub struct SpectralDensity<'a, T: Signal> {
    phantom: PhantomData<&'a T>,
}
impl<'a, T: Signal> SpectralDensity<'a, T> {
    /// Returns [Welch] [Builder] providing the `signal` sampled at `fs`Hz
    pub fn builder(signal: &'a [T], fs: T) -> Builder<'a, T> {
        Builder::new(signal).sampling_frequency(fs)
    }
}

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
pub struct Periodogram<T: Signal>(T, Vec<T>);
impl<T: Signal> Deref for Periodogram<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.1.as_slice()
    }
}
impl<T: Signal> Periodogram<T> {
    /// Returns the frequency vector in Hz for the given signal sampling frequency
    pub fn frequency(&self) -> Vec<T> {
        let n = self.1.len();
        let fs = self.0;
        (0..n)
            .map(|i| {
                T::from_usize(i).unwrap() * fs * T::from_f32(0.5).unwrap()
                    / T::from_usize(n - 1).unwrap()
            })
            .collect()
    }
}
