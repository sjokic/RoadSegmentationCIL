""" Script to train a pl.LightningModule using the CIL dataset. """

import os
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from models.models import get_model, get_model_from_ckpt, model_index
from losses import losses_index
from metrics import metrics_index
from utils.data.cilab import SatImageDataModule
from utils.data.deepglobe import DeepGlobeDataModule
from models.patchcnn.dataloader import SatImagePatchesDataModule

# Hyperparameters
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 40

# Stuff to save model checkpoints and logs
LOG_DIR = 'logs_training/'

# Local paths to directories with training data
cwd = os.getcwd()

TRAIN_IMG_DIR = cwd + '/training/training/images/'
TRAIN_MASK_DIR = cwd + '/training/training/groundtruth/'
TEST_IMG_DIR = cwd + '/test_images/test_images/'

PRE_TRAIN_IMG_DIR = cwd + '/deepglobe/training/images/'
PRE_TRAIN_MASK_DIR = cwd + '/deepglobe/training/groundtruth/'


def train_net(args):
    def train(config, num_epochs, num_gpus):
        # Initialize an instance of the DataModule for Kaggle road seg data
        if args.pretrain:
            # Pretraining mode, initialise deepglobe datamodule
            dm = DeepGlobeDataModule(PRE_TRAIN_IMG_DIR, PRE_TRAIN_MASK_DIR,
                                     TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                     config['bs'], n_workers=args.n_workers)
        else:
            # Normal, initialise Kaggle road seg datamodule
            if(args.model == 'patchcnn'):
                dm = SatImagePatchesDataModule(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                               TEST_IMG_DIR, config['bs'], n_workers=args.n_workers,
                                               gpus=(num_gpus if (num_gpus is not None) else 0))
            else:
                dm = SatImageDataModule(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                                        TEST_IMG_DIR, config['bs'], n_workers=args.n_workers,
                                        data_aug=args.augmentation, model=args.model)

        dm.setup('fit')

        # Get an instance of the metric
        metric = metrics_index[args.val_metric]
        metric_name = 'validation_{}'.format(metric.name)
        metric_short = 'val_{}'.format(metric.name)

        # Initilize an instance of your model
        if args.ckpt is not None:
            model = get_model_from_ckpt(args.model, os.path.join(cwd, args.ckpt), n_channels=3,
                                        n_filters=config['n_filters'], n_classes=1,
                                        loss_fn=losses_index[config['loss']],
                                        metric=metric, lr=config['lr'])

            if(args.model == 'pix2pix'):
                model.reset_discriminator()
        else:
            model = get_model(args.model, n_channels=3, n_filters=config['n_filters'],
                              n_classes=1, loss_fn=losses_index[config['loss']],
                              metric=metric, lr=config['lr'])

        # Early stopping
        early_stop = EarlyStopping(
            monitor=metric_short,
            mode='max',
            patience=args.patience,
            strict=False,
            verbose=False,
        )

        if not args.hyperparam_search:
            # Initialize an instance of tensorboard logger
            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=LOG_DIR,
                name=args.name
            )

            tb_logger.log_hyperparams({
                'name': args.name,
                'epochs': args.n_epochs,
                'earlystop': args.patience,
                'lr': config['lr'],
                'batchsize': config['bs'],
                'n_filters': config['n_filters'],
                'loss': losses_index[config['loss']].name,
                'metric': metric_name
            })

            # Model checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor=metric_short,
                filename='{epoch}-{val_' + metric.name + ':.3f}',
                mode='max',
                save_top_k=args.save_top_k,
                verbose=True
            )

            # Initialize an instance of the trainer
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                gpus=num_gpus,
                logger=tb_logger,
                flush_logs_every_n_steps=50,
                callbacks=[checkpoint_callback, early_stop],
                fast_dev_run=args.test
            )

            # Fit data
            trainer.fit(model, dm)

            # Save logger
            tb_logger.save()

        else:
            # Use Ray Tune for hyperparameter optimization
            tune_report = TuneReportCallback({metric_name: metric_short}, on="validation_end")

            tb_logger = pl_loggers.TensorBoardLogger(
                save_dir=tune.get_trial_dir(),
                name="",
                version="."
            )

            trainer = pl.Trainer(
                max_epochs=num_epochs,
                gpus=num_gpus,
                logger=tb_logger,
                flush_logs_every_n_steps=50,
                callbacks=[tune_report, early_stop],
                fast_dev_run=args.test
            )

            # Fit data
            trainer.fit(model, dm)

    return train


def hyperparam_opt(args, num_epochs=100, gpus_per_trial=0):
    config = {
        "loss": tune.grid_search(['focal-jacc', 'bce-jacc', 'jaccard']),
        "bs": tune.grid_search([1, 2, 4]),
        "lr": tune.grid_search([1e-5, 1e-4]),
        "n_filters": tune.grid_search([32]),
    }

    metric = metrics_index[args.val_metric]
    metric_name = 'validation_{}'.format(metric.name)

    reporter = CLIReporter(
        parameter_columns=["loss", "bs", "lr", "n_filters"],
        metric_columns=[metric_name, "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_net(args),
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": mp.cpu_count(),
            "gpu": gpus_per_trial
        },
        metric=metric_name,
        mode="max",
        config=config,
        progress_reporter=reporter,
        local_dir="./logs_hyperparamsearch",
        name=args.name + "_HyperparameterSearch")

    print("Best hyperparameters found were: ", analysis.best_config)

    with open('best_hyperparams.txt', 'w') as f:
        print("Best hyperparameters found were: {}".format(analysis.best_config), file=f)


def main(args):
    if(args.hyperparam_search):
        if(args.gpus != None):
            gpus = int(args.gpus)
        else:
            gpus = 0
        hyperparam_opt(args, num_epochs=args.n_epochs, gpus_per_trial=gpus)
    else:
        config = {'lr': args.lr,
                  'bs': args.bs,
                  'n_filters': args.n_filters,
                  'loss': args.loss}
        train = train_net(args)

        if(args.gpus != None):
            num_gpus = int(args.gpus)
        else:
            num_gpus = args.gpus

        train(config, args.n_epochs, num_gpus)


if __name__ == "__main__":
    parser = ArgumentParser(description='Train model on the training data.')

    parser.add_argument('model', metavar='MODEL_NAME', choices=list(model_index.keys()),
                        help='Name of the ML model to use.')

    parser.add_argument(
        '--name', help='Change the name of the training experiment (defaults to model name)')

    parser.add_argument('--test', action='store_true',
                        help='Perform a fast test run to check for bugs')

    # Override training dir
    parser.add_argument('--pretrain', action='store_true',
                        help='Pretraining mode for deepglobe')

    # Dataset stuff
    parser.add_argument('--augmentation', action='store_true',
                        help='Perform data augmentation')

    # Training hardware params
    parser.add_argument('--gpus', default=None,
                        help='How many GPUs to use, do not pass if using CPU')
    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of workers for DataLoaders')

    # Loss function
    parser.add_argument('--loss', choices=list(losses_index.keys()),
                        default='bce', help='Name of loss function to use.')

    # Metrics function
    parser.add_argument('--val_metric', choices=list(metrics_index.keys()),
                        default='iou', help='Name of metric to use for validation.')

    # CHECKPOINT if pre-training
    parser.add_argument('--ckpt', default=None,
                        help='Path to checkpoint file for pre-trained weights.')

    # Hyperparameters
    parser.add_argument('--hyperparam_search', action='store_true',
                        help='Perform a hyperparam search (does not early stop)')

    parser.add_argument('--n_epochs', default=100, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--patience', default=40, type=int,
                        help='Patience for early stopping.')
    parser.add_argument('--save_top_k', default=1, type=int,
                        help='Save the top k models with the best validation scores.')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--bs', default=4, type=int,
                        help='Batch size')
    parser.add_argument('--n_filters', default=32, type=int,
                        help='Number of filters of the first convolutional layer. The number of filters for the remaining layers are a multiple of n_filters.')

    args = parser.parse_args()

    # Set name to model if not specified
    args.name = args.name if args.name is not None else args.model

    main(args)
