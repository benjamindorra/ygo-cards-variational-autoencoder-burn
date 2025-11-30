use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    GroupNorm, GroupNormConfig,
    PaddingConfig2d,
    Gelu,
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct GRN<B: Backend> {
    gamma: Tensor<B,4>,
    beta: Tensor<B,4>,
}

#[derive(Config, Debug)]
pub struct GRNConfig {
}

#[derive(Module, Debug)]
pub struct ConvNext2Block<B: Backend> {
    norm: GroupNorm<B>,
    act: Gelu,
    grn: GRN<B>,
    depthwise_conv: Conv2d<B>,
    pointwise_conv1: Conv2d<B>,
    pointwise_conv2: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct ConvNext2BlockConfig {
    in_channels: usize,
    out_channels: usize,
} 

impl GRNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GRN<B> {
        GRN {
            gamma: Tensor::zeros(Shape::new([1,1,1,1]), device), 
            beta: Tensor::zeros(Shape::new([1,1,1,1]), device), 
        }
    }
}

impl<B: Backend> GRN<B> {
    pub fn forward(&self, x: Tensor<B,4>) -> Tensor<B,4> {
        let gx = l2_norm(x.clone(), &vec![2,3]);
        let nx = gx.clone() / (gx.mean_dim(1)+1e-6);
        self.gamma.clone() * ( x.clone() * nx ) + self.beta.clone() + x
    }
}

fn l2_norm<B: Backend>(input: Tensor<B, 4>, dims:&[usize]) -> Tensor<B,4> {
    let mut sq = input.powi_scalar(2);
    for d in dims {
        sq = sq.sum_dim(*d);
    }
    sq.sqrt()
}

impl ConvNext2BlockConfig { 
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvNext2Block<B> {
        ConvNext2Block {
            norm: GroupNormConfig::new(1, self.in_channels).init(device),
            act: Gelu::new(),
            grn: GRNConfig::new().init(device),
            depthwise_conv: Conv2dConfig::new([self.in_channels, self.in_channels], [7,7])
                .with_padding(PaddingConfig2d::Same)
                .with_groups(self.in_channels)
                .init(device),
            pointwise_conv1: Conv2dConfig::new([self.in_channels, 4*self.in_channels], [1,1]).init(device),
            pointwise_conv2: Conv2dConfig::new([4*self.in_channels, self.out_channels], [1,1]).init(device),
        }
    }
}

impl<B: Backend> ConvNext2Block<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = input.clone();
        let x = self.depthwise_conv.forward(input);
        let x = self.norm.forward(x);
        let x = self.pointwise_conv1.forward(x);
        let x = self.act.forward(x);
        let x = self.grn.forward(x);
        let x = self.pointwise_conv2.forward(x);
        x + shortcut
    }
}
