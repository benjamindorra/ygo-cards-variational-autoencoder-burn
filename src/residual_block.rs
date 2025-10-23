use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    GroupNorm, GroupNormConfig,
    Gelu,
};

use burn::prelude::*;


#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    act: Gelu,
    conv_shortcut: Option<Conv2d<B>>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    in_channels: usize,
    out_channels: usize,
}


impl ResBlockConfig { 
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            norm1: GroupNormConfig::new(1, self.in_channels).init(device),
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3,3]).init(device),
            norm2: GroupNormConfig::new(1, self.out_channels).init(device),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3,3]).init(device),
            act: Gelu::new(),
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
        let x = self.act.forward(input);
        let x = self.norm1.forward(x);
        let x = self.conv1.forward(x);
        let x = self.act.forward(x);
        let x = self.norm2.forward(x);
        let x = self.conv2.forward(x);
        // Residual junction with rescaling by 1/sqrt(2) to simplify normalization  
        (x + shortcut) * 0.707
    }
}
