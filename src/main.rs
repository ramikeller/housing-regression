mod dataset;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use model::MLPConfig;

fn main() {
    let (train, test, normalizer) = dataset::load("data/housing.csv");
    println!("Train rows: {}  Test rows: {}", train.len(), test.len());

    // Autodiff<Wgpu> = GPU backend (Metal on M4) with automatic differentiation.
    type Backend = Autodiff<Wgpu>;
    let device = Default::default();

    let model = MLPConfig::new(8, 64, 1).init::<Backend>(&device);

    let model = training::train(
        model,
        &train,
        &test,
        &device,
        100,   // num_epochs
        32,    // batch_size
        0.001, // learning_rate
    );

    // Convert normalized errors back to dollars using the target range.
    let target_range = (normalizer.maxs[8] - normalizer.mins[8]) as f64;
    let (mse, mae) = training::evaluate(&model, &test, &device, 32);
    let rmse_dollars = mse.sqrt() * target_range;
    let mae_dollars = mae * target_range;
    println!("\n--- Final test evaluation ---");
    println!("RMSE: ${:.0}", rmse_dollars);
    println!("MAE:  ${:.0}", mae_dollars);

    // Sample inference: a house in San Francisco (near the bay).
    // [longitude, latitude, age, total_rooms, total_bedrooms,
    //  population, households, median_income]
    let sample = [-122.4, 37.75, 20.0, 2000.0, 400.0, 1000.0, 380.0, 5.5];
    let price = training::predict(&model, &normalizer, sample, &device);
    println!("\n--- Sample prediction ---");
    println!("Predicted house value: ${:.0}", price);
}
