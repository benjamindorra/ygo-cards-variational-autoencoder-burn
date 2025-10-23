use burn::data::{dataset::vision::ImageFolderDataset};

pub trait CardsLoader {
    fn cards_train() -> Self;
    fn cards_test() -> Self;
}

impl CardsLoader for ImageFolderDataset {
    fn cards_train() -> Self {
        let img_dir = "data/card_images_small/train";
        Self::new_classification(img_dir).unwrap()
    }

    fn cards_test() -> Self {
        let img_dir = "data/card_images_small/test";
        Self::new_classification(img_dir).unwrap()
    }
}


