use crate::{
    data::{CardsBatch, CardsBatcher},
    dataset::CardsLoader,
    var_autoencoder::{VarAutoencoder, VarAutoencoderConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    grad_clipping::GradientClippingConfig,
    nn::loss::{MseLoss, Reduction::Mean},
    optim::AdamWConfig,
    prelude::*,
    train::{
        InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput, TrainStep,
        metric::LossMetric,
    },
};

impl VarAutoencoder {
    pub fn forward_step(&self, images: Tensor<4>) -> RegressionOutput {
        let [batch_size, channels, height, width] = images.dims();
        let output_dist = self.forward(images.clone());
        let output = output_dist.sample;
        let output_flatten = output.reshape([batch_size, channels * height * width]);
        let targets_flatten = images.reshape([batch_size, channels * height * width]);
        // Reconstruction loss
        let reconstruction_loss =
            MseLoss::new().forward(output_flatten.clone(), targets_flatten.clone(), Mean);
        // Additional term to adjust the predicted mean and std to the observed distribution
        // Based on "Auto-Encoding Variational Bayes"
        let log_var = output_dist.log_var;
        let mean_square = output_dist.mean.powi_scalar(2);
        let std_square = output_dist.std.powi_scalar(2);
        let distribution_loss = log_var - mean_square - std_square;
        let distribution_loss: Tensor<1> = -0.5 - distribution_loss.mean() / 2.;
        let loss = 0.001 * distribution_loss + reconstruction_loss;

        RegressionOutput::new(loss, output_flatten, targets_flatten)
    }
}

impl TrainStep for VarAutoencoder {
    type Input = CardsBatch;
    type Output = RegressionOutput;
    fn step(&self, batch: CardsBatch) -> TrainOutput<RegressionOutput> {
        let item = self.forward_step(batch.images);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for VarAutoencoder {
    type Input = CardsBatch;
    type Output = RegressionOutput;
    fn step(&self, batch: CardsBatch) -> RegressionOutput {
        self.forward_step(batch.images)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub var_autoencoder: VarAutoencoderConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.)]
    pub clip_value: f32,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 2e-4)]
    pub lr: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train(artifact_dir: &str, config: TrainingConfig, device: impl Into<Device>) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    let device = device.into();
    device.seed(config.seed);
    let autodiff_device = device.autodiff();

    //let batcher = CardsBatcher::default();
    let batcher = CardsBatcher::new();
    let batcher_test = CardsBatcher::new();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cards_train());

    let dataloader_test = DataLoaderBuilder::new(batcher_test.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ImageFolderDataset::cards_test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((LossMetric::new(),))
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .grads_accumulation(4)
        .summary();

    let model = config.var_autoencoder.init(&autodiff_device);

    let result = training.launch(Learner::new(
        model,
        config
            .optimizer
            .init()
            .with_grad_clipping(GradientClippingConfig::Value(config.clip_value).init()),
        config.lr,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/var_autoencoder"))
        .expect("Unable to save trained model");
}
