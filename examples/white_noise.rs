use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;
use welch_sde::{Build, PowerSpectrum};

fn main() {
    let n = 1e6 as usize;
    let signal: Vec<f64> = (0..n)
        .map(|_| 1.234_f64.sqrt() * thread_rng().sample::<f64, StandardNormal>(StandardNormal))
        .collect();
    {
        let mean = signal.iter().cloned().sum::<f64>() / n as f64;
        let variance = signal.iter().map(|s| *s - mean).map(|x| x * x).sum::<f64>() / n as f64;
        println!("Signal variance: {:.3}", variance);
    }

    let welch: PowerSpectrum<f64> = PowerSpectrum::builder(&signal).build();
    println!("{}", welch);

    let now = Instant::now();
    let ps = welch.periodogram();
    println!(
        "Power spectrum estimated in {}ms",
        now.elapsed().as_millis()
    );
    {
        let variance = 2. * ps.iter().sum::<f64>();
        println!("Signal variance from power spectrum: {:.3}", variance);
    }
}
