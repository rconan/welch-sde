fn main() {
    println!("{}", 2 << 11);

    let a = 0.5;
    let k = 4;
    let s: Vec<_> = (0..15).collect();
    let l = (s.len() as f64 / (k as f64 * (1. - a) + a)).trunc() as usize;
    println!("l: {}", l);
    let d = (l as f64 * a).round() as usize;
    println!("d: {}", d);
    s.windows(l)
        .step_by(l - d)
        .for_each(|w| println!("{:?}", w));
}
