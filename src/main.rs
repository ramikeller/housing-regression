mod dataset;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use model::LinearRegressionConfig;

fn main() {
    let (train, test, _normalizer) = dataset::load("data/housing.csv");
    println!("Train rows: {}  Test rows: {}", train.len(), test.len());

    // Autodiff<Wgpu> = GPU backend (Metal on M4) with automatic differentiation.
    type Backend = Autodiff<Wgpu>;
    let device = Default::default();

    let model = LinearRegressionConfig::new(8, 1).init::<Backend>(&device);

    let _model = training::train(
        model,
        &train,
        &test,
        &device,
        100,   // num_epochs
        32,    // batch_size
        0.001, // learning_rate
    );
}
