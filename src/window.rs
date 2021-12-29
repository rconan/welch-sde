use crate::Signal;

/// Signal windowing interface
pub trait Window<T: Signal> {
    /// Creates a new window of size `n`
    fn new(n: usize) -> Self;
    /// Return the window sampling weights
    fn weights(&self) -> &[T];
    /// Return the sum of the squared weights
    fn sqr_sum(&self) -> T {
        self.weights().iter().map(|&w| w * w).sum()
    }
    /// Return the square of the weights sum
    fn sum_sqr(&self) -> T {
        self.weights().iter().cloned().sum::<T>().powi(2)
    }
}
/// Hann window
pub struct Hann<T> {
    weight: Vec<T>,
}
impl<T: Signal> Window<T> for Hann<T> {
    fn new(n: usize) -> Self {
        let pi = T::from_f64(std::f64::consts::PI).unwrap();
        let nm1 = T::from_usize(n - 1).unwrap();
        let weight: Vec<T> = (0..n)
            .map(|i| {
                let j = T::from_usize(i).unwrap();
                (pi * j / nm1).sin().powi(2)
            })
            .collect();
        Self { weight }
    }
    fn weights(&self) -> &[T] {
        self.weight.as_slice()
    }
}
