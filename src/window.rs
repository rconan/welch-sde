use crate::Signal;
use std::fmt::Debug;

/// Signal windowing interface
pub trait Window<T: Signal>: Debug + Clone {
    /// Creates a new window of size `n`
    fn new(n: usize) -> Self;
    /// Return the window sampling weights
    fn weights(&self) -> &[T];
    /// Return the sum of the squared weights
    fn sqr_sum(&self) -> T;
    /// Return the square of the weights sum
    fn sum_sqr(&self) -> T;
}
/// Hann window
#[derive(Debug, Clone)]
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
    fn sqr_sum(&self) -> T {
        self.weights().iter().map(|&w| w * w).sum()
    }
    fn sum_sqr(&self) -> T {
        self.weights().iter().cloned().sum::<T>().powi(2)
    }
}
/// One window
///
/// A window where all weights are 1
#[derive(Debug, Clone)]
pub struct One<T> {
    weight: Vec<T>,
}
impl<T: Signal> Window<T> for One<T> {
    fn new(n: usize) -> Self {
        Self {
            weight: vec![T::one(); n],
        }
    }
    fn weights(&self) -> &[T] {
        self.weight.as_slice()
    }
    fn sqr_sum(&self) -> T {
        T::from_usize(self.weight.len()).unwrap()
    }
    fn sum_sqr(&self) -> T {
        self.sqr_sum().powi(2)
    }
}
