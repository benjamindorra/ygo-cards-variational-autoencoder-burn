# ygo-cards-variational-autoencoder-burn
Implementation of a variational autoencoder to test the deep learning framework Burn

## Requirements

uv

rustc

cargo

## Usage

### Setup the data

$ cd data

$ uv run download_images.py 

$ uv run exclude_invalid.py 

$ chmod +x create_test_set.sh

$ ./create_test_set.sh

### Start training

$ cd ..

$ cargo run --release

The model will train and generate an intermediate image between two of the test images.
