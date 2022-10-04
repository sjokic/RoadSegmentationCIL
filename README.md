# Support Victory Machines CIL2021 Project 3
Stefan Jokic, Yan Walesch, Thomas Zhou

## 1. Setup
### Dependencies
Please use Python version >=3.7.4, <=3.8.5.
Install all required packages via `pip install -r requirements.txt`. You may also need to run `pip install ray[default]` and `pip install ray[tune]` if you are encountering issues with dependencies with Ray Tune, the package we use for hyper-parameter search.

**IMPORTANT:** If you are encountering issues with importing PyTorch Lightning on the ETH cluster, run `python -m venv env_name` followed by `source env_name/bin/activate` first before installing the packages specified in `requirements.txt`. The issue is that the python module of the cluster may already include several packages that are conflicting with the installation of PyTorch Lightning.

### Kaggle Data
Download the .zip from the [Kaggle](https://www.kaggle.com/c/cil-road-segmentation-2021) and unzip in the root directory.
Feel free to remove the `mask_to_submission.py` and `submission_to_mask.py` as there's a copy of them in the `utils` directory.

## 2. Training
Run the following command to train a model from scratch with default parameters:

`python train.py MODEL_NAME`

Read a longer description of the args using `python train.py -h`. Notable flags include `--n_filters N_FILTERS`, `--loss LOSS_FN`, `--ckpt CKPT_FILE`, `--augmentation`, `--hyperparam_search`, as well as regular hyperparameter flags like `--lr`, `--bs`, and `--n_epochs`. 

For our **best model**, we ran the following:

`python train.py lanet --n_workers 4 --gpus 1 --n_epochs 1000 --lr 5e-4 --bs 4 --loss bce-dice  --n_filters 128 --ckpt PATH_TO_PRETRAINED_WEIGHTS --augmentation`

This will require pre-trained weights. See **section 3**. 

**NOTE:** For convenience, we've uploaded the weight files for the best model (`model-best.ckpt`) and pretrained weights (`model-pretrained-weights.ckpt`) for `lanet` at https://polybox.ethz.ch/index.php/s/I7ZH05H0lZRyIkA. 

Outputs log to local directory `logs_training/`. Access training loss and validation score using tensorboard: `tensorboard --logdir logs_training`. Resulting model weights from training will be saved under `logs_training/MODEL_NAME/version_X/checkpoints/*.ckpt`.

For our three baselines, we ran the following:
- CNN patch classifier: `python train.py patchcnn --n_workers 4 --gpus 1 --n_epochs 300 --bs 128`
- UNet: `python train.py unet --n_workers 4 --gpus 1 --n_epochs 300`
- Pix2Pix: `python train.py pix2pix --n_workers 4 --gpus 1 --n_epochs 300 --n_filters 64`

For hyperparameters not explicitly set using the respective flags, default values were used, i.e. lr=0.001, bs=4, n_filters=32.

See **section 4** on how to produce predictions on test data after training and **section 5** on how to generate the output .csv file for submission to kaggle. **Section 5** also describes how post-processing can be applied to the predicted segmentation maps. 

#### Models
We include a model index at `models/models.py` which specifies the names that can be used for the `MODEL_NAME` arg.
Models mentioned in the report include:

- `patchcnn`: Patch-based shallow CNN (baseline 1)
- `unet`: U-Net (baseline 2)
- `pix2pix`: Pix2Pix cGAN using U-Net generator (baseline 3)
- `lanet`: Our modified LANet architecture
- `lanet-res`: Original LANet ResNet architecture

Models which were tested but not mentioned in the report include:

- `dinknet`: D-LinkNet from _D-linknet: Linknet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction_ by Zhou et al.
- `dunet`: D-UNet architecture using a dilation block from the aforementioned `dinknet`

#### Loss Functions
Likewise, a loss function index is specified in `losses.py` which specifies the functions for the `--loss LOSS_FN` flag:
- `bce`: Binary Cross Entropy
- `dice`: Dice coefficient loss
- `bce-dice`: Linear combination of BCE and Dice loss
- `focal`: Focal loss from _Focal loss for dense object detection_ by Lin et a.
- `jaccard`: Jaccard/IoU loss

And several other linear cominations of loss functions.

## 3. Pre-training
If you want to pre-train your network, download the DeepGlobe training set [here](https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset?select=train) into `deepglobe/training_raw` and then generate random crops:

`python utils/generate_random_crops.py deepglobe/training_raw deepglobe/training --n 2`

To pre-train the network:

`python train.py MODEL_NAME --pretrain`

After pre-training, you can load the pre-trained checkpoint using the `--ckpt` flag for `train.py` to start training with pre-trained weights. For our **best model**, we use the following command.

`python train.py lanet --pretrain --n_workers 4 --gpus 1 --n_epochs 50 --patience 12 --lr 5e-4 --bs 4 --loss bce-dice --n_filters 128`

## 4. Predictions
To predict on test data, run the following:

`python predict.py MODEL_NAME --ckpt CHECKPOINT_FILE --out_dir OUTPUT_DIRECTORY`

where `CHECKPOINT_FILE` is the path to the `.ckpt` file containing the model weights. After training using `train.py`, you should find this file under `logs_training/MODEL_NAME/version_X/checkpoints/*.ckpt`. Segmentation maps (probabilities) produced by predicting on the test data will be stored in the directory specified by `OUTPUT_DIRECTORY`.

Likewise, for a full list of flags refer to `python predict.py -h`. Important flags include `--augmentation` for test-time augmentation, `--gpu`, and `--n_filters` to ensure the weights are compatible. For our **best model**, we used the following command:

`python predict.py lanet --ckpt CHECKPOINT_FILE --out_dir predictions/lanet/preds --n_workers 4 --gpu --n_filters 128 --augmentation`

## 5. Post-process masks and generate submission
As `predict.py` only outputs prediction probabilities, we use a final script to apply any ensembling or post-processing and generate the Kaggle submission CSV.
To generate a submission.csv using a simple default binary 0.5 threshold on the input prediction probabilities:

`python submit.py OUTPUT_DIR TEST_DIR --pred_dirs PRED_DIR`

where `OUTPUT_DIR` specifies the directory of where the resulting post-processed masks and output .csv file should be stored and `TEST_DIR` specifies the directory containing the test images. The directory or directories specified by the `--pred_dirs` flag must contain predicted segmentation maps as produced by `predict.py` (see previous **section 4**).

To generate a submission.csv using an ensemble of multiple prediction probabilities:

`python submit.py OUTPUT_DIR TEST_DIR --pred_dirs PRED_DIR_1 PRED_DIR_2 ... PRED_DIR_N [--val_weighted]`

Set the `--weighted` flag in order to weight models by their validation score (or any manual weight), but *there needs to be a file called weight.txt in the prediction directory that contains the score*. Newer versions of predict.py will generate this automatically. Note that we did not apply any ensembling for our final submission and hence ensembling was not mentioned in our report.

Finally, you can also specify alternate post-processing methods. For example, to apply the post-processing layer discussed in the report:

`python submit.py OUTPUT_DIR TEST_DIR --pred_dirs PRED_DIR --pp problayer`
