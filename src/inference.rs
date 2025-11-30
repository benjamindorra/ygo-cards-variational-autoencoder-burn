use crate::training::TrainingConfig;
use crate::data::{CardsBatcher, CardsBatch};
use burn::{
    data::{dataset::vision::ImageDatasetItem, dataloader::batcher::Batcher},
    record::{CompactRecorder, Recorder},
    tensor::{Distribution::Normal, Tensor}
};
use image::{RgbImage, Rgb};
use burn::prelude::*;
use std::path::Path;
use std::{fs, path};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item1: ImageDatasetItem, item2: ImageDatasetItem) {

    let generated_images_dir = path::Path::new(artifact_dir).join("generated_images");
    let _ = fs::remove_dir_all(&generated_images_dir);
    let _ = fs::create_dir(&generated_images_dir);
    let num_generated = 10;

    // Load model and select the decoder
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/var_autoencoder").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.var_autoencoder
        .init::<B>(&device)
        .load_record(record);

     
    // Decoder inference from noise
    let encoder = model.encoder;
    let decoder = model.decoder;

    
    // Reconstruction of a test image 
    let batcher = CardsBatcher::new();
    let batch1: CardsBatch<B> = batcher.batch(vec![item1], &device);
    let encoding1 = encoder.forward(batch1.images);
    //println!("encoded size: {:?}", encoding1.sample.shape());
    let output = decoder.forward(encoding1.sample);
    let out_path = generated_images_dir.join("reconstructed_image_1.jpg");
    write_image(output, &out_path);
    let batch2: CardsBatch<B> = batcher.batch(vec![item2], &device);
    let encoding2 = encoder.forward(batch2.images);
    let output = decoder.forward(encoding2.sample);
    let out_path = generated_images_dir.join("reconstructed_image_2.jpg");
    write_image(output, &out_path);

    for i in 0..num_generated {
        let noise = Tensor::<B, 4>::random(Shape::new([1,32,48,33]),Normal(0., 1.), &device);
        let output = decoder.forward(noise);
        let out_path = generated_images_dir.join(format!("generated_image_{}.jpg", i));
        write_image(output, &out_path);
    }

    

}

fn write_image<B: Backend>(output: Tensor<B,4>, out_path: &Path) {
        // Reshape output image and denormalize
        // Denormalize
        //let output = batcher.normalizer.to_device(&device).denormalize(output);
        let output = output.clamp(0., 1.) * 255;
        // [C,H,W] -> [H,W,C]
        let output = output.swap_dims(2,1).swap_dims(3,2);

        // Convert to uint8 image array and save to the artifact directory
        let [_, height, width, _] = output.dims();
        let height = height.try_into().unwrap();
        let width = width.try_into().unwrap();
        // To flat u8 array
        let output = output
            .into_data()
            .convert_dtype(burn::tensor::DType::U8)
            .into_vec::<u8>()
            .expect("The data should be converted to a vector");
        // Write to an imager buffer and save
        let mut imgbuf = RgbImage::new(width, height);
        for x in 0..height {
            for y in 0..width {
                // index into the H,W,3 array
                let index = (x * width * 3 + y * 3).try_into().unwrap();
                let color = &output[index .. index + 3];
                imgbuf.put_pixel(y, x, Rgb(color.try_into().unwrap()));
            }
        }
        imgbuf.save(out_path)
            .expect("The output image should be saved normally");
}
