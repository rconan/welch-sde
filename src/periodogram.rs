use crate::{Signal, Welch, Window};
use std::ops::Deref;

/// Signal periodogram
#[derive(Debug, Clone)]
pub struct Periodogram<T: Signal>(T, Vec<T>);
impl<T: Signal> Deref for Periodogram<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.1.as_slice()
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
    /// Returns the frequency vector in Hz
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
