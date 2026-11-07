use burn::nn::{
    Gelu, GroupNorm, GroupNormConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct GRN {
    gamma: Tensor<4>,
    beta: Tensor<4>,
}

#[derive(Config, Debug)]
pub struct GRNConfig {}

#[derive(Module, Debug)]
pub struct ConvNext2Block {
    norm: GroupNorm,
    act: Gelu,
    grn: GRN,
    depthwise_conv: Conv2d,
    pointwise_conv1: Conv2d,
    pointwise_conv2: Conv2d,
}

#[derive(Config, Debug)]
pub struct ConvNext2BlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl GRNConfig {
    pub fn init(&self, device: &Device) -> GRN {
        GRN {
            gamma: Tensor::zeros(Shape::new([1, 1, 1, 1]), device),
            beta: Tensor::zeros(Shape::new([1, 1, 1, 1]), device),
        }
    }
}

impl GRN {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let gx = l2_norm(x.clone(), &vec![2, 3]);
        let nx = gx.clone() / (gx.mean_dim(1) + 1e-6);
        self.gamma.clone() * (x.clone() * nx) + self.beta.clone() + x
    }
}

fn l2_norm(input: Tensor<4>, dims: &[usize]) -> Tensor<4> {
    let mut sq = input.powi_scalar(2);
    for d in dims {
        sq = sq.sum_dim(*d);
    }
    sq.sqrt()
}

impl ConvNext2BlockConfig {
    pub fn init(&self, device: &Device) -> ConvNext2Block {
        ConvNext2Block {
            norm: GroupNormConfig::new(1, self.in_channels).init(device),
            act: Gelu::new(),
            grn: GRNConfig::new().init(device),
            depthwise_conv: Conv2dConfig::new([self.in_channels, self.in_channels], [7, 7])
                .with_padding(PaddingConfig2d::Same)
                .with_groups(self.in_channels)
                .init(device),
            pointwise_conv1: Conv2dConfig::new([self.in_channels, 4 * self.in_channels], [1, 1])
                .init(device),
            pointwise_conv2: Conv2dConfig::new([4 * self.in_channels, self.out_channels], [1, 1])
                .init(device),
        }
    }
}

impl ConvNext2Block {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
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
