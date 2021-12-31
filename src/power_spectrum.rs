use crate::{Build, Builder, One, Periodogram, PowerSpectrumPeriodogram, Signal, Welch, Window};
use std::{fmt::Display, ops::Deref};

/// Power spectrum
///
/// Computes a `signal` power spectrum from [Welch] [Periodogram]
pub struct PowerSpectrum<'a, T, W = One<T>>
where
    T: Signal,
    W: Window<T>,
{
    welch: Welch<'a, T, W>,
}
impl<'a, T, W> PowerSpectrum<'a, T, W>
where
    T: Signal,
    W: Window<T>,
{
    /// Returns [Welch] [Builder] providing the `signal`
    pub fn builder(signal: &[T]) -> Builder<T> {
        Builder::new(signal)
    }
    /// Returns the power spectrum periodogram
    pub fn periodogram(&self) -> Periodogram<T> {
        <Welch<'a, T, W> as PowerSpectrumPeriodogram<T>>::periodogram(&self.welch)
    }
}
impl<'a, T, W> Build<T, W, PowerSpectrum<'a, T, W>> for Builder<'a, T>
where
    T: Signal,
    W: Window<T>,
{
    fn build(&self) -> PowerSpectrum<'a, T, W> {
        PowerSpectrum {
            welch: self.build(),
        }
    }
}
impl<'a, T, W> Deref for PowerSpectrum<'a, T, W>
where
    T: Signal,
    W: Window<T>,
{
    type Target = Welch<'a, T, W>;

    fn deref(&self) -> &Self::Target {
        &self.welch
    }
}
impl<'a, T, W> Display for PowerSpectrum<'a, T, W>
where
    T: Signal,
    W: Window<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.welch.fmt(f)
    }
}
