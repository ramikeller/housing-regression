mod dataset;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use model::MLPConfig;

#[derive(Parser)]
#[command(about = "Train a housing price regression model")]
struct Args {
    /// Device to run on: gpu or cpu
    #[arg(long, default_value = "gpu")]
    device: String,
}

fn run<B: AutodiffBackend>(device: B::Device) {
    // Fix the random seed so weight initialisation and batch shuffling
    // are identical across runs, making results reproducible.
    B::seed(&device, 42);

    let (train, test, normalizer) = dataset::load("data/housing.csv");
    println!("Train rows: {}  Test rows: {}", train.len(), test.len());

    let model = MLPConfig::new(8, 64, 32, 1).init::<B>(&device);

    let model = training::train(
        model,
        &train,
        &test,
        &device,
        100,   // num_epochs
        32,    // batch_size
        0.001, // learning_rate
    );

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

fn main() {
    let args = Args::parse();

    match args.device.as_str() {
        "cpu" => {
            println!("Running on CPU");
            run::<Autodiff<NdArray>>(Default::default());
        }
        "gpu" => {
            println!("Running on GPU");
            run::<Autodiff<Wgpu>>(Default::default());
        }
        other => {
            eprintln!("Unknown device '{other}'. Use --device cpu or --device gpu.");
            std::process::exit(1);
        }
    }
}
