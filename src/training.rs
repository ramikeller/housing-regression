use burn::prelude::*;

use crate::dataset::HousingRow;
use crate::model::LinearRegression;

// A batch of data converted to tensors, ready for the GPU.
// features shape: [batch_size, 8]
// targets shape:  [batch_size, 1]
#[derive(Debug, Clone)]
pub struct HousingBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

// Converts a Vec<HousingRow> into a HousingBatch by stacking the rows into tensors.
pub struct HousingBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> HousingBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn batch(&self, rows: Vec<HousingRow>) -> HousingBatch<B> {
        // Build a flat Vec of feature values, then reshape into [batch_size, 8].
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

// Runs one forward pass and returns the MSE loss (a 0D scalar tensor).
pub fn mse_loss<B: Backend>(
    model: &LinearRegression<B>,
    batch: &HousingBatch<B>,
) -> Tensor<B, 1> {
    let predictions = model.forward(batch.features.clone());

    // (prediction - actual)², averaged over the batch.
    let diff = predictions - batch.targets.clone();
    let loss = diff.powf_scalar(2.0).mean();

    loss
}
