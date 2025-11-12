use crate::residual_block::{ResBlock, ResBlockConfig};

use burn::{
    nn::{
        conv::{Conv2d,Conv2dConfig},
    }, prelude::*, tensor::Distribution::Normal
};

#[derive(Module, Debug)]
pub struct VarEncoder<B: Backend> {
    conv1: Conv2d<B>,
    //block1: ResBlock<B>,
    conv2: Conv2d<B>,
    block2: ResBlock<B>,
    conv3: Conv2d<B>,
    block3: ResBlock<B>,
    conv4: Conv2d<B>,
    block4: ResBlock<B>,
    //conv5: Conv2d<B>,
    block4_2: ResBlock<B>,
    block4_3: ResBlock<B>,
    block4_4: ResBlock<B>,
    block4_5: ResBlock<B>,
    conv_latent: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct VarEncConfig {
    channels: [usize; 6],
}

impl VarEncConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarEncoder<B> {
        VarEncoder {
            conv1: Conv2dConfig::new([self.channels[0],self.channels[1]], [2,2]).with_stride([2,2]).init(device),
            //block1: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            conv2: Conv2dConfig::new([self.channels[1],self.channels[2]], [2,2]).with_stride([2,2]).init(device),
            block2: ResBlockConfig::new(self.channels[2], self.channels[2]).init(device),
            conv3: Conv2dConfig::new([self.channels[2],self.channels[3]], [2,2]).with_stride([2,2]).init(device),
            block3: ResBlockConfig::new(self.channels[3], self.channels[3]).init(device),
            conv4: Conv2dConfig::new([self.channels[3],self.channels[4]], [2,2]).with_stride([2,2]).init(device),
            block4: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            //conv5: Conv2dConfig::new([self.channels[4],self.channels[5]], [2,2]).with_stride([2,2]).init(device),
            block4_2: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_3: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_4: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_5: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            conv_latent: Conv2dConfig::new([self.channels[4],self.channels[5]], [1,1]).init(device),
        }
    }
}

// Hold both the encoding, its mean and std to help compute the loss 
pub struct SampleDist<B: Backend> {
    pub sample: Tensor<B, 4>,
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
    pub log_var: Tensor<B,4>,
}

impl<B: Backend> VarEncoder<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> SampleDist<B> {
        // Forward pass through the layers
        let x = self.conv1.forward(images);
        //let x = self.block1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.block2.forward(x);
        let x = self.conv3.forward(x);
        let x = self.block3.forward(x);
        let x = self.conv4.forward(x);
        let x = self.block4.forward(x);
        //let x = self.conv5.forward(x);
        let x = self.block4_2.forward(x);
        let x = self.block4_3.forward(x);
        let x = self.block4_4.forward(x);
        let x = self.block4_5.forward(x);
        let x =  self.conv_latent.forward(x);
        // Trick to get a distributed output
        let shape = x.shape();
        let width = shape.dims[1];
        let mean_logvar = x.split(width/2, 1);
        let m = mean_logvar[0].clone();
        let log_var = mean_logvar[1].clone();
        let s = (log_var.clone() * 0.5).exp();
        let noise = Tensor::<B, 4>::random_like(&m, Normal(0.,1.));
        let encoding = m.clone() + s.clone() * noise;
        SampleDist { 
            sample: encoding,
            mean: m,
            std: s,
            log_var: log_var,
        }
    }
}
