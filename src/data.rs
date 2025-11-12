// Largely inspired by https://github.com/tracel-ai/burn/blob/main/examples/custom-image-dataset/src/data.rs 

use burn::{
    data::{
        dataset::vision::{ImageDatasetItem, PixelDepth},
        dataloader::batcher::Batcher,
    },
    prelude::*,
};

/*
// Mean and std values computed on the training dataset
const MEAN: [f32; 3] = [0.51779658, 0.46859592, 0.46552556];
const STD: [f32; 3] = [0.27136453, 0.25286143, 0.25103455];

#[derive(Clone)]
pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1,3,1,1]);
        let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1,3,1,1]);
        Self { mean, std }
    }
    
    pub fn normalize(&self, input: Tensor::<B, 4>) -> Tensor::<B, 4> {
        (input - self.mean.clone()) / self.std.clone() 
    }

    pub fn denormalize(&self, input: Tensor::<B, 4>) -> Tensor::<B, 4> {
        input * self.std.clone() + self.mean.clone()
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.clone().to_device(device),
            std: self.std.clone().to_device(device),
        }
    }
}
*/

#[derive(Clone)]
pub struct CardsBatcher {}

#[derive(Clone, Debug)]
pub struct CardsBatch<B: Backend> {
    pub images: Tensor<B, 4>,
}

impl CardsBatcher {
    pub fn new() -> Self {
        Self {}
    }
}

impl<B: Backend> Batcher<B, ImageDatasetItem, CardsBatch<B>> for CardsBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> CardsBatch<B> {
        fn images_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
            item.image
                .into_iter()
                .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
                .collect::<Vec<u8>>()
        }
        let images = items
            .iter()
            .map(|item| TensorData::new(images_as_vec_u8(item.clone()), Shape::new([391, 268, 3])))
            .map(|data| {
                Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), device)
                    .swap_dims(2,1)
                    .swap_dims(1,0)
            })
            .map(|tensor| tensor / 255)
            .collect();

        let images = Tensor::stack(images, 0);
        //let images = self.normalizer.to_device(device).normalize(images);

        CardsBatch { images }

    }
}
