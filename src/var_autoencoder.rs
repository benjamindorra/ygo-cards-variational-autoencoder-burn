use crate::{
    var_decoder::{VarDecConfig, VarDecoder},
    var_encoder::{SampleDist, VarEncConfig, VarEncoder},
};

use burn::prelude::*;

#[derive(Module, Debug)]
pub struct VarAutoencoder {
    pub encoder: VarEncoder,
    pub decoder: VarDecoder,
}

#[derive(Config, Debug)]
pub struct VarAutoencoderConfig {
    #[config(default = "[3,8,16,32,32,64]")]
    encoder_channels: [usize; 6],
    #[config(default = "[32,32,32,16,8,3]")]
    decoder_channels: [usize; 6],
    #[config(default = "[391, 268]")]
    output_size: [usize; 2],
}

impl VarAutoencoderConfig {
    pub fn init(&self, device: &Device) -> VarAutoencoder {
        VarAutoencoder {
            encoder: VarEncConfig::new(self.encoder_channels).init(device),
            decoder: VarDecConfig::new(self.decoder_channels)
                .with_output_size(self.output_size)
                .init(device),
        }
    }
}

impl VarAutoencoder {
    pub fn forward(&self, images: Tensor<4>) -> SampleDist {
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
