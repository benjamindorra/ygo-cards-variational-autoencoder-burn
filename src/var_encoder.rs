use burn::{
    tensor::Distribution::Normal,
    nn::{
        conv::{Conv2d,Conv2dConfig},
        pool::{AvgPool2d, AvgPool2dConfig},
        GroupNorm, GroupNormConfig, Gelu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct VarEncoder<B: Backend> {
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    norm4:GroupNorm<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    act: Gelu,
    pool: AvgPool2d,
}

#[derive(Config, Debug)]
pub struct VarEncConfig {
    widths: [usize; 6],
}

impl VarEncConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarEncoder<B> {
        VarEncoder {
            conv1: Conv2dConfig::new([self.widths[0],self.widths[1]], [3,3]).init(device),
            conv2: Conv2dConfig::new([self.widths[1],self.widths[2]], [3,3]).init(device),
            conv3: Conv2dConfig::new([self.widths[2],self.widths[3]], [3,3]).init(device),
            conv4: Conv2dConfig::new([self.widths[3],self.widths[4]], [3,3]).init(device),
            conv5: Conv2dConfig::new([self.widths[4],self.widths[5]], [1,1]).init(device),
            norm2: GroupNormConfig::new(1, self.widths[1]).init(device),
            norm4: GroupNormConfig::new(1, self.widths[3]).init(device),
            act: Gelu::new(),
            pool: AvgPool2dConfig::new([2, 2]).init(),
        }
    }
}

// Hold both the encoding, its mean and std to help compute the loss 
pub struct SampleMeanStd<B: Backend> {
    pub sample: Tensor<B, 4>,
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> VarEncoder<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> SampleMeanStd<B> {
        // Forward pass through the layers
        let x = self.conv1.forward(images);
        let x = self.pool.forward(x);
        let x = self.act.forward(x);
        let x = self.norm2.forward(x);
        let x = self.conv2.forward(x);
        let x = self.pool.forward(x);
        let x = self.act.forward(x);
        let x = self.conv3.forward(x);
        let x = self.pool.forward(x);
        let x = self.act.forward(x);
        let x = self.norm4.forward(x);
        let x = self.conv4.forward(x);
        let x = self.pool.forward(x);
        let x = self.act.forward(x);
        let x =  self.conv5.forward(x);
        // Trick to get a distributed output
        let shape = x.shape();
        let width = shape.dims[1];
        let mean_std = x.split(width/2, 1);
        let m = mean_std[0].clone();
        let s = mean_std[1].clone().abs();
        let noise = Tensor::<B, 4>::random_like(&s, Normal(0.,1.));
        let encoding = m.clone() + s.clone() * noise;
        SampleMeanStd { 
            sample: encoding,
            mean: m,
            std: s,
        }
    }
}
