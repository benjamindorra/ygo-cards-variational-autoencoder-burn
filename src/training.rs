use crate::{
    data::{CardsBatch, CardsBatcher},
    dataset::CardsLoader,
    var_autoencoder::{VarAutoencoder, VarAutoencoderConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset},
    nn::loss::{MseLoss, Reduction::Mean},
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        RegressionOutput,
        LearnerBuilder,
        TrainOutput,
        TrainStep,
        ValidStep,
        metric::LossMetric,
    },
    grad_clipping::GradientClippingConfig,
    lr_scheduler::exponential::{ExponentialLrSchedulerConfig},
};
        
impl<B: Backend> VarAutoencoder<B> {
    pub fn forward_step(
        &self,
        images: Tensor<B, 4>,
    ) -> RegressionOutput<B> {
        let [batch_size, channels, height, width] = images.dims();
        let output_mean_std = self.forward(images.clone());
        let output = output_mean_std.sample;
        let output_flatten = output.reshape([batch_size, channels * height * width]);
        let targets_flatten = images.reshape([batch_size, channels * height * width]);
        // Reconstruction loss
        let reconstruction_loss = MseLoss::new()
            .forward(output_flatten.clone(), targets_flatten.clone(), Mean);
        // Additional term to adjust the predicted mean and std to the observed distribution 
        // Based on "Auto-Encoding Variational Bayes"
        let mean_square = output_mean_std.mean.powi_scalar(2);
        let std_square = output_mean_std.std.powi_scalar(2);
        let distribution_loss = 
            std_square.clone().log() 
            - mean_square 
            - std_square;
        let distribution_loss: Tensor<B, 1> = - 1./2. - distribution_loss.mean() / 2.;
        let loss =  distribution_loss + reconstruction_loss;

        RegressionOutput::new(loss, output_flatten, targets_flatten)
    }
}


impl<B: AutodiffBackend> TrainStep<CardsBatch<B>,  RegressionOutput<B>> for VarAutoencoder<B> {
    fn step(&self, batch: CardsBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(batch.images);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CardsBatch<B>,  RegressionOutput<B>> for VarAutoencoder<B> {
    fn step(&self, batch: CardsBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch.images)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub var_autoencoder: VarAutoencoderConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1.)]
    pub clip_value: f32,
    #[config(default = 3)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub initial_lr: f64,
    #[config(default = 0.9995)]
    pub gamma: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    
    B::seed(config.seed);

    //let batcher = CardsBatcher::default();
    let batcher = CardsBatcher::<B>::new(device.clone());
    let batcher_test = CardsBatcher::<B::InnerBackend>::new(device.clone());

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

    let scheduler = ExponentialLrSchedulerConfig::new(config.initial_lr, config.gamma)
        .init()
        .expect("The lr scheduler should instantiate normally");

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.var_autoencoder.init::<B>(&device),
            config.optimizer.init().with_grad_clipping(GradientClippingConfig::Value(config.clip_value).init()),
            scheduler,
        );
    
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/var_autoencoder"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
