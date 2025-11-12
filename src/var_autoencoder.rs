use crate::{
    var_encoder::{VarEncoder, VarEncConfig, SampleDist},
    var_decoder::{VarDecoder, VarDecConfig},
};

use burn::prelude::*;

#[derive(Module, Debug)]
pub struct VarAutoencoder<B: Backend> {
    pub encoder: VarEncoder<B>,
    pub decoder: VarDecoder<B>,
}

#[derive(Config, Debug)]
pub struct VarAutoencoderConfig {
    #[config(default = "[3,8,16,32,64,128]")]
    encoder_channels: [usize; 6],
    #[config(default = "[64,64,32,16,8,3]")]
    decoder_channels: [usize; 6],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],

}

impl VarAutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarAutoencoder<B> {
        VarAutoencoder {
            encoder: VarEncConfig::new(self.encoder_channels).init(device),
            decoder: VarDecConfig::new(self.decoder_channels).with_output_size(self.output_size).init(device),
        }
    }
}

impl<B: Backend> VarAutoencoder<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> SampleDist<B> {
        let encoding_dist = self.encoder.forward(images);
        let x = encoding_dist.sample;
        let x = self.decoder.forward(x);
        SampleDist {
            sample: x,
            mean: encoding_dist.mean,
            log_var: encoding_dist.log_var,
            std: encoding_dist.std,
        }
    }
}
