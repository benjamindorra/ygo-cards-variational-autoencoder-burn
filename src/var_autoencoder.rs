use crate::{
    var_encoder::{VarEncoder, VarEncConfig, SampleMeanStd},
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
    #[config(default = "[3,8,16,32,32,16]")]
    encoder_widths: [usize; 6],
    #[config(default = "[8,32,16,8,3]")]
    decoder_widths: [usize; 5],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],

}

impl VarAutoencoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VarAutoencoder<B> {
        VarAutoencoder {
            encoder: VarEncConfig::new(self.encoder_widths).init(device),
            decoder: VarDecConfig::new(self.decoder_widths).with_output_size(self.output_size).init(device),
        }
    }
}

impl<B: Backend> VarAutoencoder<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> SampleMeanStd<B> {
        let encoding_mean_std = self.encoder.forward(images);
        let x = encoding_mean_std.sample;
        let x = self.decoder.forward(x);
        SampleMeanStd {
            sample: x,
            mean: encoding_mean_std.mean,
            std: encoding_mean_std.std,
        }
    }
}
