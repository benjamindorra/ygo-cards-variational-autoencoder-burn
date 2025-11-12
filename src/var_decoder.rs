use crate::residual_block::{ResBlock, ResBlockConfig};

use burn::{
    nn::{
        conv::{ConvTranspose2d,ConvTranspose2dConfig, Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
        //Tanh,
        Sigmoid,
    }, prelude::*
};

#[derive(Module, Debug)]
pub struct VarDecoder<B: Backend> {
    conv_latent: Conv2d<B>,
    block1: ResBlock<B>,
    block1_2: ResBlock<B>,
    block1_3: ResBlock<B>,
    block1_4: ResBlock<B>,
    block1_5: ResBlock<B>,
    conv1: ConvTranspose2d<B>,
    block2: ResBlock<B>,
    conv2: ConvTranspose2d<B>,
    block3: ResBlock<B>,
    conv3: ConvTranspose2d<B>,
    //block4: ResBlock<B>,
    conv4: ConvTranspose2d<B>,
    //conv5: ConvTranspose2d<B>,
    sigmoid: Sigmoid,
    // Scaling factors for the sigmoid 
    a: Tensor<B,4>, 
    interp: Interpolate2d,
}

#[derive(Config, Debug)]
pub struct VarDecConfig {
    channels: [usize; 6],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],
}

impl VarDecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarDecoder<B> {
        VarDecoder {
            conv_latent: Conv2dConfig::new([self.channels[0], self.channels[1]], [1,1]).init(device),
            block1: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            block1_2: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            block1_3: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            block1_4: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            block1_5: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            conv1: ConvTranspose2dConfig::new([self.channels[1],self.channels[2]], [2,2]).with_stride([2,2]).init(device),
            block2: ResBlockConfig::new(self.channels[2], self.channels[2]).init(device),
            conv2: ConvTranspose2dConfig::new([self.channels[2],self.channels[3]], [2,2]).with_stride([2,2]).init(device),
            block3: ResBlockConfig::new(self.channels[3], self.channels[3]).init(device),
            conv3: ConvTranspose2dConfig::new([self.channels[3],self.channels[4]], [2,2]).with_stride([2,2]).init(device),
            //block4: ResBlockConfig::new(self.channels[4], self.channels[4]).init(device),
            conv4: ConvTranspose2dConfig::new([self.channels[4],self.channels[5]], [2,2]).with_stride([2,2]).init(device),
            //conv5: ConvTranspose2dConfig::new([self.channels[5],self.channels[6]], [2,2]).with_stride([2,2]).init(device),
            sigmoid: Sigmoid::new(),
            a: Tensor::ones(Shape::new([1, 1, 1, 1]), device), 
            interp: Interpolate2dConfig::new().with_output_size(Some(self.output_size)).init(),
        }
    }
}

impl<B: Backend> VarDecoder<B> {
    pub fn forward(&self, encoding: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_latent.forward(encoding);
        let x = self.block1.forward(x);
        let x = self.block1_2.forward(x);
        let x = self.block1_3.forward(x);
        let x = self.block1_4.forward(x);
        let x = self.block1_5.forward(x);
        let x = self.conv1.forward(x);
        let x = self.block2.forward(x);
        let x = self.conv2.forward(x);
        let x = self.block3.forward(x);
        let x = self.conv3.forward(x);
        //let x = self.block4.forward(x);
        let x = self.conv4.forward(x);
        //let x = self.conv5.forward(x);
        let x = self.sigmoid.forward( self.a.clone() * x );
        self.interp.forward(x)
    }
}
