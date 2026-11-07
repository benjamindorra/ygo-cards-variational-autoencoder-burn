use crate::convnext2_block::{ConvNext2Block, ConvNext2BlockConfig};
use crate::residual_block::{ResBlock, ResBlockConfig};

use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
    tensor::Distribution::Normal,
};

#[derive(Module, Debug)]
pub struct VarEncoder {
    conv1: Conv2d,
    //block1: ConvNext2Block,
    conv2: Conv2d,
    block2: ResBlock,
    conv3: Conv2d,
    block3: ConvNext2Block,
    //conv4: Conv2d,
    block4: ConvNext2Block,
    //conv5: Conv2d,
    block4_2: ConvNext2Block,
    block4_3: ConvNext2Block,
    block4_4: ConvNext2Block,
    block4_5: ConvNext2Block,
    block4_6: ConvNext2Block,
    conv_latent: Conv2d,
}

#[derive(Config, Debug)]
pub struct VarEncConfig {
    channels: [usize; 6],
}

impl VarEncConfig {
    pub fn init(&self, device: &Device) -> VarEncoder {
        VarEncoder {
            conv1: Conv2dConfig::new([self.channels[0], self.channels[1]], [2, 2])
                .with_stride([2, 2])
                .init(device),
            //block1: ConvNext2BlockConfig::new(self.channels[1], self.channels[1]).init(device),
            conv2: Conv2dConfig::new([self.channels[1], self.channels[2]], [2, 2])
                .with_stride([2, 2])
                .init(device),
            block2: ResBlockConfig::new(self.channels[2], self.channels[2]).init(device),
            conv3: Conv2dConfig::new([self.channels[2], self.channels[3]], [2, 2])
                .with_stride([2, 2])
                .init(device),
            block3: ConvNext2BlockConfig::new(self.channels[3], self.channels[3]).init(device),
            //conv4: Conv2dConfig::new([self.channels[3],self.channels[4]], [2,2]).with_stride([2,2]).init(device),
            block4: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_2: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_3: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_4: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_5: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            block4_6: ConvNext2BlockConfig::new(self.channels[4], self.channels[4]).init(device),
            conv_latent: Conv2dConfig::new([self.channels[4], self.channels[5]], [1, 1])
                .init(device),
        }
    }
}

// Hold both the encoding, its mean and std to help compute the loss
pub struct SampleDist {
    pub sample: Tensor<4>,
    pub mean: Tensor<4>,
    pub std: Tensor<4>,
    pub log_var: Tensor<4>,
}

impl VarEncoder {
    pub fn forward(&self, images: Tensor<4>) -> SampleDist {
        // Forward pass through the layers
        let x = self.conv1.forward(images);
        //let x = self.block1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.block2.forward(x);
        let x = self.conv3.forward(x);
        let x = self.block3.forward(x);
        //let x = self.conv4.forward(x);
        let x = self.block4.forward(x);
        //let x = self.conv5.forward(x);
        let x = self.block4_2.forward(x);
        let x = self.block4_3.forward(x);
        let x = self.block4_4.forward(x);
        let x = self.block4_5.forward(x);
        let x = self.block4_6.forward(x);
        let x = self.conv_latent.forward(x);
        // Trick to get a distributed output
        let width = x.dims()[1];
        let mean_logvar = x.split(width / 2, 1);
        let m = mean_logvar[0].clone();
        let log_var = mean_logvar[1].clone();
        let s = (log_var.clone() * 0.5).exp();
        let noise = Tensor::<4>::random_like(&m, Normal(0., 1.));
        let encoding = m.clone() + s.clone() * noise;
        SampleDist {
            sample: encoding,
            mean: m,
            std: s,
            log_var: log_var,
        }
    }
}
