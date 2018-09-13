# Deep Learning Architectures for Image Demosaicing

Implementation of several Deep Learning architectures for image demosaicing with CNNs.

## Getting Started

Training files for running training of selected architectures are denoted by their architecture name.

"UNET_train", "RESNET_train", "Chang_Resnet" implement models, select directories, generate file structures for training and testing and run training/evaluation.

### Prerequisites

Keras, Tensorflow, python3, opencv2, sklearn

## Datasets

Collected Datasets:

MIT 5K (raw) - https://data.csail.mit.edu/graphics/fivek/

RAISE -  http://loki.disi.unitn.it/RAISE/

Gharbi - https://groups.csail.mit.edu/graphics/demosaicnet/

MIR FLIKR - https://press.liacs.nl/mirflickr/

Describable Textures Dataset (DTD) - https://www.robots.ox.ac.uk/~vgg/data/dtd/


## Training

Run "UNET_train"/"RESNET_train"/"Chang_Resnet" - Requires input directories, model parameters, training parameters, test parameters

Called:

model_train - runs training loop with inputted training parameters training parameters

train_backend - general functions: application of bayer pattern (mosaic), training sequencer (image pre-processing for training), loss functions, data shaping, misc.

Training loop includes end of epoch predictions, training loss.  Keyboard interrupt during training, interrupts trainin gloop and saves out model and results.

Training generates model folder with: model, epoch models, epoch training, epoch predictions


## Testing

Testing Class in "model_train.py".

Kodak and McMaster test sets used for testing.

PSNR and SSIM used as metrics.

Results stored in JSON format.


## Raw Demosaicing

Incomplete implementation with DCRAW.  Illustrative of required input shape, further work required, useful for algorithm comparison on raw images.


## Authors

* **Rhys Buggy- *MAI Thesis work 2018*
