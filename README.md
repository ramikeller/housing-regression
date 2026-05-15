# housing-regression

A linear regression model built in Rust using the [burn](https://github.com/tracel-ai/burn) deep learning framework. It predicts California median house prices from the 1990 US Census dataset.

## Goal

A learning project for understanding ML model training fundamentals: data loading, normalization, model definition, loss functions, and GPU-accelerated training.

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

Targets the Apple M4 GPU via burn's `wgpu` backend (Metal).

## Project structure

```
src/
  main.rs       — entry point
  dataset.rs    — CSV loading, cleaning, normalization, train/test split
  model.rs      — linear regression model definition
data/
  housing.csv   — raw dataset
  README.md     — data source and column descriptions
```

## Running

```bash
cargo run --release
```

## Dependencies

- [burn](https://crates.io/crates/burn) 0.21 — ML framework (wgpu + autodiff + train)
- [csv](https://crates.io/crates/csv) 1.4 — CSV parsing
- [serde](https://crates.io/crates/serde) 1 — deserialization
