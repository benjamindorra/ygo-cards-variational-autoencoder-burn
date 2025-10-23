#![recursion_limit = "256"]

mod var_encoder;
mod var_decoder;
mod var_autoencoder;
mod training;
mod data;
mod dataset;
mod inference;

use crate::{
    var_autoencoder::VarAutoencoderConfig,
    training::TrainingConfig,
    dataset::CardsLoader,
};

use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamWConfig,
    data::dataset::Dataset,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "training";
     
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(VarAutoencoderConfig::new(), AdamWConfig::new()),
        device.clone(),
    );
    
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::ImageFolderDataset::cards_test()
            .get(0)
            .unwrap(),
    );
}
