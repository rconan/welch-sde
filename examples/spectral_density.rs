use rand::prelude::*;
use rand_distr::StandardNormal;
use std::time::Instant;
use welch_sde::Welch;

fn main() {
    let n = 1e5 as usize;
    let fs = 10e3;
    let amp = 2. * 2f64.sqrt();
    let freq = 1550f64;
    let noise_power = 0.001 * fs;
    let signal: Vec<f64> = (0..n)
        .map(|i| i as f64 / fs)
        .map(|t| {
            amp * (2. * std::f64::consts::PI * freq * t).sin()
                + thread_rng().sample::<f64, StandardNormal>(StandardNormal) * noise_power.sqrt()
        })
        .collect();

    let welch = Welch::builder(&signal, fs).n_segment(8).build();
    println!("{}", welch);
    let now = Instant::now();
    let ps = welch.spectral_density();
    println!(
        "Spectral density esimated in {}ms",
        now.elapsed().as_millis()
    );
    let noise_floor = ps.iter().skip(ps.len() / 2).cloned().sum::<f64>() / ((ps.len() / 2) as f64);
    println!("Noise floor: {:.3e}", noise_floor);

    let _: complot::LinLog = (
        ps.frequency()
            .into_iter()
            .zip(&(*ps))
            .map(|(x, &y)| (x, vec![y])),
        None,
    )
        .into();
}
