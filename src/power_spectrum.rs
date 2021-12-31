use crate::{Build, Builder, One, Periodogram, PowerSpectrumPeriodogram, Signal, Welch};
use std::{fmt::Display, ops::Deref};

type WelchOne<'a, T> = Welch<'a, T, One<T>>;

/// Power spectrum
///
/// Computes a `signal` power spectrum from [Welch] [Periodogram] using [One] [Window](crate::Window)
pub struct PowerSpectrum<'a, T: Signal> {
    welch: WelchOne<'a, T>,
}
impl<'a, T: Signal> PowerSpectrum<'a, T> {
    /// Returns [Welch] [Builder] providing the `signal`
    pub fn builder(signal: &[T]) -> Builder<T> {
        Builder::new(signal)
    }
    /// Returns the power spectrum periodogram
    pub fn periodogram(&self) -> Periodogram<T> {
        <WelchOne<'a, T> as PowerSpectrumPeriodogram<T>>::periodogram(&self.welch)
    }
}
impl<'a, T: Signal> Build<T, One<T>, PowerSpectrum<'a, T>> for Builder<'a, T> {
    fn build(&self) -> PowerSpectrum<'a, T> {
        PowerSpectrum {
            welch: self.build(),
        }
    }
}
impl<'a, T: Signal> Deref for PowerSpectrum<'a, T> {
    type Target = WelchOne<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.welch
    }
}
impl<'a, T: Signal> Display for PowerSpectrum<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.welch.fmt(f)
    }
}
