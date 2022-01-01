use crate::{Build, Builder, Hann, Periodogram, Signal, SpectralDensityPeriodogram, Welch};
use std::{fmt::Display, ops::Deref};

type WelchHann<'a, T> = Welch<'a, T, Hann<T>>;

/// Spectral density
///
/// Computes a `signal` spectral density from [Welch] [Periodogram] using [Hann] [Window](crate::Window)
#[derive(Debug, Clone)]
pub struct SpectralDensity<'a, T: Signal>(WelchHann<'a, T>);
impl<'a, T: Signal> SpectralDensity<'a, T> {
    /// Returns [Welch] [Builder] given the `signal` sampled at `fs`Hz
    pub fn builder(signal: &[T], fs: T) -> Builder<T> {
        Builder::new(signal).sampling_frequency(fs)
    }
    /// Returns the spectral density periodogram
    pub fn periodogram(&self) -> Periodogram<T> {
        <WelchHann<'a, T> as SpectralDensityPeriodogram<T>>::periodogram(&self.0)
    }
}
impl<'a, T: Signal> Build<T, Hann<T>, SpectralDensity<'a, T>> for Builder<'a, T> {
    fn build(&self) -> SpectralDensity<'a, T> {
        SpectralDensity(self.build())
    }
}
impl<'a, T: Signal> Deref for SpectralDensity<'a, T> {
    type Target = WelchHann<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a, T: Signal> Display for SpectralDensity<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
