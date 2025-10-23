use crate::training::TrainingConfig;
use crate::data::{CardsBatcher, CardsBatch};
use burn::{
    data::{dataset::vision::ImageDatasetItem, dataloader::batcher::Batcher},
    record::{CompactRecorder, Recorder},
    tensor::{Distribution::Normal, Tensor}
};
use image::{RgbImage, Rgb};
use burn::prelude::*;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ImageDatasetItem) {

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
    let decoder = model.decoder;
    let noise = Tensor::<B, 4>::random(Shape::new([1,8,16,24]),Normal(0., 1.), &device);
    let output = decoder.forward(noise);
    

    let batcher = CardsBatcher::<B>::new(device.clone());
   
    /*
    // Reconstruction of a test image 
    let batch: CardsBatch<B> = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images).sample;
    */
    
    // Reshape output image and denormalize
    // Denormalize
    let output = batcher.normalizer.to_device(&device).denormalize(output);
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
    imgbuf.save("generated_image.jpg")
        .expect("The output image should be saved normally");

    

}
