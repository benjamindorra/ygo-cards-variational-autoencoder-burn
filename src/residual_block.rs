use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    GroupNorm, GroupNormConfig,
    PaddingConfig2d,
    Sigmoid,
};

use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Swish<B: Backend> {
    beta: Tensor<B,4>,
}

#[derive(Config, Debug)]
pub struct SwishConfig {
}

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    act1: Swish<B>,
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    act2: Swish<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl SwishConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Swish<B> {
        Swish {
            beta: Tensor::ones(Shape::new([1, 1, 1, 1]), device), 
        }
    }
}

impl<B: Backend> Swish<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.clone() * Sigmoid.forward( self.beta.clone() * input )
    }
}

impl ResBlockConfig { 
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            act1: SwishConfig::new().init(device),
            norm1: GroupNormConfig::new(1, self.in_channels).init(device),
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3,3]).with_padding(PaddingConfig2d::Same).init(device),
            act2: SwishConfig::new().init(device),
            norm2: GroupNormConfig::new(1, self.out_channels).init(device),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3,3]).with_padding(PaddingConfig2d::Same).init(device),
            conv_shortcut: match self.in_channels==self.out_channels {
                true => None,
                false => Some(Conv2dConfig::new([self.in_channels, self.out_channels], [1,1]).init(device)),
            },
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = match self.conv_shortcut.clone() {
            None => input.clone(),
            Some(conv) => conv.forward(input.clone()),
        };
        let x = self.act1.forward(input);
        let x = self.norm1.forward(x);
        let x = self.conv1.forward(x);
        let x = self.act2.forward(x);
        let x = self.norm2.forward(x);
        let x = self.conv2.forward(x);
        // Residual junction with rescaling by 1/sqrt(2) to simplify normalization  
        (x + shortcut) * 0.707
    }
}
