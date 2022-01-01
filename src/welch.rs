use crate::{Build, Builder, Periodogram, Signal, Window};
use num_complex::Complex;
use num_traits::Zero;
use rustfft::{algorithm::Radix4, Fft, FftDirection};
use std::fmt::Display;

/// Welch spectral density estimator
pub struct Welch<'a, T: Signal, W: Window<T>> {
    /// number of segments (`k`)
    pub n_segment: usize,
    /// size of each segment (`l`)
    pub segment_size: usize,
    /// size of the discrete Fourier transform (`p`)
    pub dft_size: usize,
    /// overlaps starting points
    overlap_idx: usize,
    /// the signal to estimate the spectral density for
    signal: &'a [T],
    /// the signal sampling frequency `[Hz]`
    pub fs: T,
    /// segments windowing function
    pub window: W,
}
impl<'a, T: Signal, W: Window<T>> Display for Welch<'a, T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Welch spectral density estimator:")?;
        writeln!(f, " - number of segment: {:>6}", self.n_segment)?;
        writeln!(f, " - segment size     : {:>6}", self.segment_size)?;
        writeln!(
            f,
            " - overlap size     : {:>6}",
            self.segment_size - self.overlap_idx
        )?;
        write!(f, " - dft size         : {:>6}", self.dft_size)
    }
}
impl<'a, T: Signal, W: Window<T>> Build<T, W, Welch<'a, T, W>> for Builder<'a, T> {
    fn build(&self) -> Welch<'a, T, W> {
        let mut k = self.n_segment;
        let mut l = self.segment_size;
        let mut m = l.next_power_of_two();
        if m > self.dft_max_size {
            l = self.dft_max_size;
            let a = self.overlap;
            k = ((self.signal.len() as f64 - l as f64 * a) / (l as f64 * (1. - a))).trunc()
                as usize;
            m = l;
        }
        Welch {
            n_segment: k,
            segment_size: l,
            dft_size: m,
            overlap_idx: l - (l as f64 * self.overlap).round() as usize,
            signal: self.signal,
            fs: self.fs.unwrap_or_else(T::one),
            window: W::new(l),
        }
    }
}
impl<'a, T: Signal, W: Window<T>> Welch<'a, T, W> {
    /// Returns [Welch] [Builder] given the `signal`
    pub fn builder(signal: &'a [T]) -> Builder<'a, T> {
        Builder::new(signal)
    }
    // Splits the signal into overlapping segments and applies the window
    fn windowed_segments(&self) -> Vec<Complex<T>> {
        let n = self.segment_size;
        let d = self.overlap_idx;
        let m = self.dft_size;
        self.signal
            .windows(n)
            .step_by(d)
            .flat_map(|s| {
                let mut buffer: Vec<Complex<T>> = vec![Complex::zero(); m];
                s.iter()
                    .zip(self.window.weights())
                    .map(|(&x, &w)| x * w)
                    .zip(&mut buffer)
                    .for_each(|(v, c)| {
                        c.re = v;
                    });
                buffer
            })
            .collect()
    }
    // Fourier transform each segment
    fn dfts(&self) -> Vec<Complex<T>> {
        let mut buffer = self.windowed_segments();
        let fft = Radix4::new(self.dft_size, FftDirection::Forward);
        fft.process(&mut buffer);
        buffer
    }
}

/// Interface to the spatial density periodogram
pub trait SpectralDensityPeriodogram<T: Signal> {
    /// Returns the signal spectral density (signal unit squared per Hertz)
    fn periodogram(&self) -> Periodogram<T>;
}
/// Interface to the power spectrum periodogram
pub trait PowerSpectrumPeriodogram<T: Signal> {
    /// Returns the signal power spectrum (signal unit squared)
    fn periodogram(&self) -> Periodogram<T>;
}

impl<'a, T: Signal, W: Window<T>> SpectralDensityPeriodogram<T> for Welch<'a, T, W> {
    fn periodogram(&self) -> Periodogram<T> {
        let u = (self.window.sqr_sum() * T::from_usize(self.n_segment).unwrap() * self.fs).recip();
        Periodogram::new(self, u)
    }
}
impl<'a, T: Signal, W: Window<T>> PowerSpectrumPeriodogram<T> for Welch<'a, T, W> {
    fn periodogram(&self) -> Periodogram<T> {
        let u = (self.window.sum_sqr() * T::from_usize(self.n_segment).unwrap()).recip();
        Periodogram::new(self, u)
    }
}
impl<T: Signal> Periodogram<T> {
    /// Creates a new [Periodogram] from [Welch::periodogram] scaled with `u`
    fn new<W: Window<T>>(welch: &Welch<T, W>, u: T) -> Self {
        let n = welch.dft_size / 2;
        Self(
            welch.fs,
            welch
                .dfts()
                .chunks(welch.dft_size)
                .map(|dft| dft.iter().take(n).map(|x| x.norm_sqr()).collect::<Vec<T>>())
                .fold(vec![T::zero(); n], |mut a, p| {
                    a.iter_mut().zip(p).for_each(|(a, p)| *a += p);
                    a
                })
                .into_iter()
                .map(|x| x * u)
                .collect(),
        )
    }
}
