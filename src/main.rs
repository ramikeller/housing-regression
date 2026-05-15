mod dataset;
mod model;

use burn::backend::Wgpu;
use model::LinearRegressionConfig;

fn main() {
    let (train, test, _normalizer) = dataset::load("data/housing.csv");
    println!("Train rows: {}", train.len());
    println!("Test rows:  {}", test.len());

    // Set up the GPU device (Metal on M4 via wgpu).
    let device = Default::default();

    // Build the model: 8 inputs (our features), 1 output (house price).
    let model = LinearRegressionConfig::new(8, 1).init::<Wgpu>(&device);
    println!("{}", model);
}
