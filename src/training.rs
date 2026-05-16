use burn::{
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::dataset::{HousingRow, Normalizer};
use crate::model::MLP;

// A batch of data converted to tensors, ready for the GPU.
// features shape: [batch_size, 8]
// targets shape:  [batch_size, 1]
#[derive(Debug, Clone)]
pub struct HousingBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

// Converts a Vec<HousingRow> into a HousingBatch by stacking the rows
// into tensors.
pub struct HousingBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> HousingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn batch(&self, rows: Vec<HousingRow>) -> HousingBatch<B> {
        let batch_size = rows.len();

        let features_data: Vec<f32> = rows
            .iter()
            .flat_map(|r| r.features.iter().map(|&v| v as f32))
            .collect();

        let targets_data: Vec<f32> = rows.iter().map(|r| r.target as f32).collect();

        let features = Tensor::<B, 1>::from_floats(features_data.as_slice(), &self.device)
            .reshape([batch_size, 8]);

        let targets = Tensor::<B, 1>::from_floats(targets_data.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        HousingBatch { features, targets }
    }
}

// Runs one forward pass and returns the MSE loss as a 1D
// single-element tensor.
pub fn mse_loss<B: Backend>(model: &MLP<B>, batch: &HousingBatch<B>) -> Tensor<B, 1> {
    let predictions = model.forward(batch.features.clone());
    let diff = predictions - batch.targets.clone();
    diff.powf_scalar(2.0).mean()
}

// Extracts the scalar f32 value from a single-element 1D loss tensor.
fn loss_scalar<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    // burn's .mean() returns a 1D single-element tensor, not a 0D
    // scalar — extract the number by indexing into the data vec.
    loss.into_data().to_vec::<f32>().unwrap()[0]
}

// Runs the full training loop and returns the trained model.
// Uses Adam optimiser and reports train + test MSE after each epoch.
pub fn train<B: AutodiffBackend>(
    mut model: MLP<B>,
    train_data: &[HousingRow],
    test_data: &[HousingRow],
    device: &B::Device,
    num_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> MLP<B> {
    // L2 regularisation via weight decay — penalises large weights to
    // reduce overfitting. λ=0.001 is a common starting value.
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(0.001)))
        .init();
    let batcher = HousingBatcher::<B>::new(device.clone());
    let mut rng = StdRng::seed_from_u64(42);

    for epoch in 1..=num_epochs {
        // Shuffle training indices so each epoch sees a different batch order.
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        indices.shuffle(&mut rng);

        // --- Training pass ---
        let mut train_loss_sum = 0.0f64;
        let mut n_train_batches = 0usize;

        for chunk in indices.chunks(batch_size) {
            let rows: Vec<HousingRow> = chunk.iter().map(|&i| train_data[i].clone()).collect();
            let batch = batcher.batch(rows);

            let loss = mse_loss(&model, &batch);
            train_loss_sum += loss_scalar(loss.clone()) as f64;
            n_train_batches += 1;

            // Backward pass: compute gradients and update weights.
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }

        // --- Test pass (forward only, no weight updates) ---
        let mut test_loss_sum = 0.0f64;
        let mut n_test_batches = 0usize;

        for chunk in test_data.chunks(batch_size) {
            let batch = batcher.batch(chunk.to_vec());
            let loss = mse_loss(&model, &batch);
            test_loss_sum += loss_scalar(loss) as f64;
            n_test_batches += 1;
        }

        println!(
            "Epoch {:>3}/{} — train MSE: {:.6}  test MSE: {:.6}",
            epoch,
            num_epochs,
            train_loss_sum / n_train_batches as f64,
            test_loss_sum / n_test_batches as f64,
        );
    }

    model
}

// Evaluates the model on a dataset and returns (mse, mae) in normalized space.
// Multiply by the target range to convert to dollars.
pub fn evaluate<B: Backend>(
    model: &MLP<B>,
    data: &[HousingRow],
    device: &B::Device,
    batch_size: usize,
) -> (f64, f64) {
    let batcher = HousingBatcher::<B>::new(device.clone());
    let mut mse_sum = 0.0f64;
    let mut mae_sum = 0.0f64;
    let mut n_batches = 0usize;

    for chunk in data.chunks(batch_size) {
        let batch = batcher.batch(chunk.to_vec());
        let predictions = model.forward(batch.features);
        let diff = predictions - batch.targets;
        mse_sum += loss_scalar(diff.clone().powf_scalar(2.0).mean()) as f64;
        mae_sum += loss_scalar(diff.abs().mean()) as f64;
        n_batches += 1;
    }

    (mse_sum / n_batches as f64, mae_sum / n_batches as f64)
}

// Predicts the house price in dollars for a single set of raw feature values.
// The features must be in the same order as the training data:
//   [longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
//    population, households, median_income]
pub fn predict<B: Backend>(
    model: &MLP<B>,
    normalizer: &Normalizer,
    raw_features: [f64; 8],
    device: &B::Device,
) -> f64 {
    // Normalize the input features using the training set's min/max.
    let row = HousingRow { features: raw_features, target: 0.0 };
    let normalized = normalizer.normalize(&row);

    let input = Tensor::<B, 1>::from_floats(
        normalized.features.map(|v| v as f32).as_slice(),
        device,
    )
    .reshape([1, 8]); // shape [1, 8]: a batch of one house

    let output = model.forward(input); // shape [1, 1]
    let normalized_prediction = loss_scalar(output.mean()) as f64;

    // Reverse the min-max scaling to get the dollar value.
    let target_min = normalizer.mins[8];
    let target_max = normalizer.maxs[8];
    normalized_prediction * (target_max - target_min) + target_min
}
