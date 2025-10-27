use crate::residual_block::{ResBlock, ResBlockConfig};

use burn::{
    nn::{
        conv::{ConvTranspose2d,ConvTranspose2dConfig, Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig},
        Tanh,
    }, prelude::*
};

#[derive(Module, Debug)]
pub struct VarDecoder<B: Backend> {
    conv0: Conv2d<B>,
    block1: ResBlock<B>,
    block1_2: ResBlock<B>,
    //block1_3: ResBlock<B>,
    conv1: ConvTranspose2d<B>,
    block2: ResBlock<B>,
    conv2: ConvTranspose2d<B>,
    block3: ResBlock<B>,
    conv3: ConvTranspose2d<B>,
    tanh: Tanh,
    // Scaling factors for the tanh 
    tanh_a: Tensor<B,4>, 
    tanh_b: Tensor<B,4>, 
    interp: Interpolate2d,
}

#[derive(Config, Debug)]
pub struct VarDecConfig {
    channels: [usize; 5],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],
}

impl VarDecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarDecoder<B> {
        VarDecoder {
            conv0: Conv2dConfig::new([self.channels[0], self.channels[1]], [1,1]).init(device),
            block1: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            block1_2: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            //block1_3: ResBlockConfig::new(self.channels[1], self.channels[1]).init(device),
            conv1: ConvTranspose2dConfig::new([self.channels[1],self.channels[2]], [2,2]).with_stride([2,2]).init(device),
            block2: ResBlockConfig::new(self.channels[2], self.channels[2]).init(device),
            conv2: ConvTranspose2dConfig::new([self.channels[2],self.channels[3]], [2,2]).with_stride([2,2]).init(device),
            block3: ResBlockConfig::new(self.channels[3], self.channels[3]).init(device),
            conv3: ConvTranspose2dConfig::new([self.channels[3],self.channels[4]], [2,2]).with_stride([2,2]).init(device),
            tanh: Tanh::new(),
            tanh_a: Tensor::ones(Shape::new([1, 1, 1, 1]), device), 
            tanh_b: Tensor::ones(Shape::new([1, 1, 1 , 1]), device), 
            interp: Interpolate2dConfig::new().with_output_size(Some(self.output_size)).init(),
        }
    }
}

impl<B: Backend> VarDecoder<B> {
    pub fn forward(&self, encoding: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv0.forward(encoding);
        let x = self.block1.forward(x);
        let x = self.block1_2.forward(x);
        //let x = self.block1_3.forward(x);
        let x = self.conv1.forward(x);
        let x = self.block2.forward(x);
        let x = self.conv2.forward(x);
        let x = self.block3.forward(x);
        let x = self.conv3.forward(x);
        let x = self.tanh_a.clone() * self.tanh.forward( self.tanh_b.clone() * x );
        self.interp.forward(x)
    }
}
