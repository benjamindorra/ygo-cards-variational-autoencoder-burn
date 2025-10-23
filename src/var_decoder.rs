use burn::{
    nn::{
        conv::{ConvTranspose2d,ConvTranspose2dConfig}, interpolate::{Interpolate2d, Interpolate2dConfig}, Gelu, GroupNorm, GroupNormConfig,
    }, prelude::*
};

#[derive(Module, Debug)]
pub struct VarDecoder<B: Backend> {
    conv1: ConvTranspose2d<B>,
    norm2: GroupNorm<B>,
    conv2: ConvTranspose2d<B>,
    conv3: ConvTranspose2d<B>,
    norm4: GroupNorm<B>,
    conv4: ConvTranspose2d<B>,
    act: Gelu, 
    interp: Interpolate2d,
}

#[derive(Config, Debug)]
pub struct VarDecConfig {
    widths: [usize; 5],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],
}

impl VarDecConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarDecoder<B> {
        VarDecoder {
            conv1: ConvTranspose2dConfig::new([self.widths[0],self.widths[1]], [3,3]).with_stride([2,2]).init(device),
            conv2: ConvTranspose2dConfig::new([self.widths[1],self.widths[2]], [3,3]).with_stride([2,2]).init(device),
            conv3: ConvTranspose2dConfig::new([self.widths[2],self.widths[3]], [3,3]).with_stride([2,2]).init(device),
            conv4: ConvTranspose2dConfig::new([self.widths[3],self.widths[4]], [3,3]).with_stride([2,2]).init(device),
            norm2: GroupNormConfig::new(1, self.widths[1]).init(device),
            norm4: GroupNormConfig::new(1, self.widths[3]).init(device),
            act: Gelu::new(),
            interp: Interpolate2dConfig::new().with_output_size(Some(self.output_size)).init(),
        }
    }
}

impl<B: Backend> VarDecoder<B> {
    pub fn forward(&self, encoding: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(encoding);
        let x = self.act.forward(x);
        let x = self.norm2.forward(x);
        let x = self.conv2.forward(x);
        let x = self.act.forward(x);
        let x = self.conv3.forward(x);
        let x = self.act.forward(x);
        let x = self.norm4.forward(x);
        let x = self.conv4.forward(x);
        self.interp.forward(x)
    }
}
