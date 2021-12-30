use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;
use welch_sde::{Build, Hann, SpectralDensity};

fn main() {
    let n = 1e5 as usize;
    let fs = 10e3;
    let amp = 2. * 2f64.sqrt();
    let freq = 1550f64;
    let noise_power = 0.001 * fs / 2.;
    let signal: Vec<f64> = (0..n)
        .map(|i| i as f64 / fs)
        .map(|t| {
            amp * (2. * std::f64::consts::PI * freq * t).sin()
                + thread_rng().sample::<f64, StandardNormal>(StandardNormal) * noise_power.sqrt()
        })
        .collect();

    let welch: SpectralDensity<f64, Hann<f64>> =
        SpectralDensity::<f64, Hann<f64>>::builder(&signal, fs).build();
    println!("{}", welch);
    let now = Instant::now();
    let sd = welch.periodogram();
    println!(
        "Spectral density estimated in {}ms",
        now.elapsed().as_millis()
    );
    let noise_floor =
        2. * sd.iter().skip(sd.len() / 2).cloned().sum::<f64>() / ((sd.len() / 2) as f64);
    println!("Noise floor: {:.3}", noise_floor);

    let _: complot::LinLog = (
        sd.frequency()
            .into_iter()
            .zip(&(*sd))
            .map(|(x, &y)| (x, vec![2. * y])),
        complot::complot!(
            "spectral_density.png",
            xlabel = "Frequency [Hz]",
            ylabel = "Spectral density [s^2/Hz]"
        ),
    )
        .into();
}
