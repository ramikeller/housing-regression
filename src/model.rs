use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

// Config holds the parameters needed to construct the model.
// burn requires this pattern — you build a config first, then call .init() to get the model.
#[derive(Config, Debug)]
pub struct LinearRegressionConfig {
    pub input_size: usize,
    pub output_size: usize,
}

impl LinearRegressionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearRegression<B> {
        LinearRegression {
            layer: LinearConfig::new(self.input_size, self.output_size)
                .with_bias(true)
                .init(device),
        }
    }
}

// The model itself. It holds one Linear layer which contains the weights and bias.
// The <B: Backend> generic means it works on any burn backend — CPU, GPU, etc.
#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    layer: Linear<B>,
}

impl<B: Backend> LinearRegression<B> {
    // The forward pass: takes a batch of feature vectors, returns a batch of predictions.
    // Input shape:  [batch_size, 8]
    // Output shape: [batch_size, 1]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(x)
    }
}
