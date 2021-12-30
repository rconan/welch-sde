use crate::{Build, Builder, Hann, Periodogram, Signal, SpectralDensityPeriodogram, Welch, Window};
use std::{fmt::Display, marker::PhantomData, ops::Deref};

/// Spectral density
///
/// Computes a `signal` spectral density from [Welch] [Periodogram]
pub struct SpectralDensity<'a, T, W = Hann<T>, E = Welch<'a, T, W>>
where
    T: Signal,
    W: Window<T>,
    E: Display + SpectralDensityPeriodogram<T>,
{
    welch: E,
    num: PhantomData<&'a T>,
    win: PhantomData<W>,
}
impl<'a, T, W, E> SpectralDensity<'a, T, W, E>
where
    T: Signal,
    W: Window<T>,
    E: Display + SpectralDensityPeriodogram<T>,
{
    /// Returns [Welch] [Builder] providing the `signal` sampled at `fs`Hz
    pub fn builder(signal: &[T], fs: T) -> Builder<T> {
        Builder::new(signal).sampling_frequency(fs)
    }
    /// Returns the spectral density periodogram
    pub fn periodogram(&self) -> Periodogram<T> {
        self.welch.periodogram()
    }
}
impl<'a, T, W> Build<T, W, SpectralDensity<'a, T, W>> for Builder<'a, T>
where
    T: Signal,
    W: Window<T>,
{
    fn build(&self) -> SpectralDensity<'a, T, W> {
        SpectralDensity {
            welch: self.build(),
            num: PhantomData,
            win: PhantomData,
        }
    }
}
impl<'a, T, W> Deref for SpectralDensity<'a, T, W>
where
    T: Signal,
    W: Window<T>,
{
    type Target = Welch<'a, T, W>;

    fn deref(&self) -> &Self::Target {
        &self.welch
    }
}
impl<'a, T, W> Display for SpectralDensity<'a, T, W>
where
    T: Signal,
    W: Window<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.welch.fmt(f)
    }
}
