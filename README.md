# housing-regression

A 2-layer MLP built in Rust using the [burn](https://github.com/tracel-ai/burn) deep learning framework. It predicts California median house prices from the 1990 US Census dataset.

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

## Model

2-layer MLP: `Linear(8→64) → ReLU → Linear(64→1)`

577 trainable parameters. Trained with Adam optimizer, MSE loss, 100 epochs, batch size 32, learning rate 0.001.

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | ~$66,000 |
| Test MAE  | ~$47,000 |

Predictions are typically off by ~$47k. The main limitation is the non-linear relationship between location and price, which a deeper model would capture better.

## Hardware

Supports both GPU and CPU backends:

| Backend | Device | Runtime (100 epochs) |
|---------|--------|----------------------|
| `wgpu` (default) | M4 GPU via Metal | ~114s |
| `ndarray` | CPU (all cores) | ~6s |

For this model size, CPU is faster — GPU overhead dominates at small scales.

## Running

```bash
cargo run --release                    # GPU (default)
cargo run --release -- --device cpu   # CPU
cargo run --release -- --help         # show options
```

Output includes per-epoch train/test MSE, final RMSE and MAE in dollars, and a sample house price prediction.

## Project structure

```
src/
  main.rs       — entry point: CLI args, training, evaluation, inference
  dataset.rs    — CSV loading, cleaning, normalization, train/test split
  model.rs      — 2-layer MLP definition (8 → 64 → 1)
  training.rs   — batcher, MSE loss, training loop, evaluation, inference
data/
  housing.csv   — raw dataset
  README.md     — data source and column descriptions
```

## Dependencies

- [burn](https://crates.io/crates/burn) 0.21 — ML framework (wgpu + ndarray + autodiff + train)
- [csv](https://crates.io/crates/csv) 1.4 — CSV parsing
- [serde](https://crates.io/crates/serde) 1 — deserialization
- [rand](https://crates.io/crates/rand) 0.8 — batch shuffling
- [clap](https://crates.io/crates/clap) 4 — CLI argument parsing
