#![recursion_limit = "256"]

mod convnext2_block;
mod data;
mod dataset;
//mod export;
mod inference;
mod residual_block;
mod training;
mod var_autoencoder;
mod var_decoder;
mod var_encoder;

use crate::{
    dataset::CardsLoader, training::TrainingConfig, var_autoencoder::VarAutoencoderConfig,
};

use burn::{data::dataset::Dataset, optim::AdamWConfig, tensor::Device};

use std::env;
use std::fs;

fn main() {
    let device = Device::wgpu(Default::default());

    let args: Vec<String> = env::args().collect();

    let mut artifact_dir = "training";
    if args.len() > 1 {
        artifact_dir = &args[2];
    }

    match args[1].as_str() {
        "train" => {
            if !fs::exists(artifact_dir).unwrap() {
                crate::training::train(
                    artifact_dir,
                    TrainingConfig::new(VarAutoencoderConfig::new(), AdamWConfig::new()),
                    device,
                );
            } else {
                println!("Error, training dir already exists");
            }
        }
        "test" => {
            crate::inference::infer(
                artifact_dir,
                device,
                burn::data::dataset::vision::ImageFolderDataset::cards_test()
                    .get(0)
                    .unwrap(),
                burn::data::dataset::vision::ImageFolderDataset::cards_test()
                    .get(1)
                    .unwrap(),
            );
        }
        //"export" => {
        //    crate::export::export_burnpack::<MyBackend>(artifact_dir, device);
        //}
        &_ => {}
    }
}
