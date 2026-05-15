# housing-regression

A linear regression model built in Rust using the [burn](https://github.com/tracel-ai/burn) deep learning framework. It predicts California median house prices from the 1990 US Census dataset.

## Goal

A learning project for understanding ML model training fundamentals: data loading, normalization, model definition, loss functions, GPU-accelerated training, and inference.

## Dataset

California Housing dataset (20,640 rows). Each row is a census block group with 8 numeric features:

- Longitude & latitude
- Median housing age
- Total rooms & bedrooms
- Population & households
- Median income

Target: median house value (USD).

See [data/README.md](data/README.md) for the full column descriptions and data source.

## Hardware

Targets the Apple M4 GPU via burn's `wgpu` backend (Metal). The device resolves to `BestAvailable` at runtime, which selects the Metal GPU automatically.

## Results

After 100 epochs (batch size 32, Adam optimizer, learning rate 0.001):

| Metric | Value |
|--------|-------|
| Test RMSE | ~$67,000 |
| Test MAE  | ~$49,000 |

The MAE means predictions are typically off by ~$49k. This is the practical limit of linear regression on this dataset — the relationship between location and price is non-linear, which a more complex model would capture better.

## Project structure

```
src/
  main.rs       — entry point: training, evaluation, and sample inference
  dataset.rs    — CSV loading, cleaning, normalization, train/test split
  model.rs      — linear regression model definition (8 inputs, 1 output)
  training.rs   — batcher, MSE loss, training loop, evaluation, inference
data/
  housing.csv   — raw dataset
  README.md     — data source and column descriptions
```

## Running

```bash
cargo run --release
```

Output includes per-epoch train/test MSE, final RMSE and MAE in dollars, and a sample house price prediction.

## Dependencies

- [burn](https://crates.io/crates/burn) 0.21 — ML framework (wgpu + autodiff + train)
- [csv](https://crates.io/crates/csv) 1.4 — CSV parsing
- [serde](https://crates.io/crates/serde) 1 — deserialization
- [rand](https://crates.io/crates/rand) 0.8 — batch shuffling
