use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

// Config holds the parameters needed to construct the model.
// burn requires this pattern — you build a config first, then call .init() to get the model.
#[derive(Config, Debug)]
pub struct MLPConfig {
    pub input_size: usize,
    pub hidden1_size: usize,
    pub hidden2_size: usize,
    pub output_size: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            layer1: LinearConfig::new(self.input_size, self.hidden1_size)
                .with_bias(true)
                .init(device),
            layer2: LinearConfig::new(self.hidden1_size, self.hidden2_size)
                .with_bias(true)
                .init(device),
            layer3: LinearConfig::new(self.hidden2_size, self.output_size)
                .with_bias(true)
                .init(device),
            activation: Relu::new(),
        }
    }
}

// A 3-layer MLP: two hidden layers with ReLU activations, then an output layer.
// The <B: Backend> generic means it works on any burn backend — CPU, GPU, etc.
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    layer1: Linear<B>,     // input → hidden1
    layer2: Linear<B>,     // hidden1 → hidden2
    layer3: Linear<B>,     // hidden2 → output
    activation: Relu,
}

impl<B: Backend> MLP<B> {
    // The forward pass: takes a batch of feature vectors, returns a batch of predictions.
    // Input shape:  [batch_size, 8]
    // Output shape: [batch_size, 1]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.layer1.forward(x));
        let x = self.activation.forward(self.layer2.forward(x));
        self.layer3.forward(x)
    }
}
