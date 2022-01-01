use crate::Signal;

/// Generic builder
pub struct Builder<'a, T: Signal> {
    /// number of segments (`k`)
    pub n_segment: usize,
    /// size of each segment (`l`)
    pub segment_size: usize,
    /// segment overlapping fraction (`0<a<1`)
    pub overlap: f64,
    /// maximum size of the discrete Fourier transform (`p`)
    pub dft_max_size: usize,
    /// the signal to estimate the spectral density for
    pub signal: &'a [T],
    /// the signal sampling frequency `[Hz]`
    pub fs: Option<T>,
}
impl<'a, T: Signal> Builder<'a, T> {
    /// Creates a Welch [Builder] from a given signal with `k=4` and `a=0.5`
    pub fn new(signal: &'a [T]) -> Self {
        let k: usize = 4;
        let a: f64 = 0.5;
        let l = (signal.len() as f64 / (k as f64 * (1. - a) + a)).trunc() as usize;
        Self {
            n_segment: k,
            segment_size: l,
            overlap: a,
            dft_max_size: 4096,
            signal,
            fs: None,
        }
    }
    /// Sets the signal sampling frequency
    pub fn sampling_frequency(self, fs: T) -> Self {
        Self {
            fs: Some(fs),
            ..self
        }
    }
    /// Sets the segment overlapping fraction (`0<a<1`)
    pub fn overlap(self, overlap: f64) -> Self {
        let k = self.n_segment;
        let a = overlap;
        let l = (self.signal.len() as f64 / (k as f64 * (1. - a) + a)).trunc() as usize;
        Self {
            segment_size: l,
            overlap: a,
            ..self
        }
    }
    /// Sets the number of segments (`k`)
    pub fn n_segment(self, n_segment: usize) -> Self {
        let k = n_segment;
        let a = self.overlap;
        let l = (self.signal.len() as f64 / (k as f64 * (1. - a) + a)).trunc() as usize;
        Self {
            n_segment: k,
            segment_size: l,
            ..self
        }
    }
    /// Sets the log2 of the maximum size of the discrete Fourier transform (`p`)
    pub fn dft_log2_max_size(self, dft_log2_max_size: usize) -> Self {
        Self {
            dft_max_size: 2 << (dft_log2_max_size - 1),
            ..self
        }
    }
}
