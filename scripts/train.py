from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
from os import mkdir, remove
from os.path import isdir, isfile

from Utils.AttrDict import load_config
from Utils.Dataloader import Transforms, Dataset, dict_transform
from Utils.BaseModel import Classifier
from Utils.ResNet import resnet
from Utils.Utils import set_seed, set_device, TransformTestItems, path_to, load_args
import logging

logger = logging.getLogger()
device = set_device()

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

# TODO's
# -Prepare config files for all models to train
# -Train model on whole data (the best one)
# -Post Processing: plot mean val acc + std
# -Post Processing: plot expert prediciton vs.baseline raw vs.cropped vs.mar
# python train.py -i .\\config\\MAR_RESNET34_CROPPED_256_borehole_train.yaml


def train(config=''):

    if not isdir(path_to('outputs')):
        # Create new folder for the model
        mkdir(path_to('outputs'))

    # Load config_file
    inputs = load_config(config)

    # Handle config_file inputs
    n_epochs = int(inputs.NEpochs)
    batch_size = int(inputs.BatchSize)
    seed = int(inputs.Seed)
    layers = inputs.Model.Layers
    classes = inputs.Model.OutClasses
    channels = inputs.Model.Channels

    # Handle file paths
    # Workspace path to Cuttings_Characterisation
    path_model = path_to(inputs.PathSave, inputs.ModelName)
    logger.info(f"Path to model is '{path_model}'")

    if not isdir(path_model):  # Create new folder for the model
        mkdir(path_model)

    path_checkpoint = path_to(inputs.PathSave,
                              inputs.ModelName,
                              inputs.CheckpointName)
    logger.info(f"Saving checkpoints at: '{path_checkpoint}'")

    # Seed
    set_seed(seed)

    # Transforms Train and Test
    transforms_train = Transforms(TransformTestItems(
        inputs.TransformTrain.items(), dict_transform
    ))
    transforms_test = Transforms(TransformTestItems(
        inputs.TransformTest.items(), dict_transform
    ))

    for i_, (train_path, test_path) in enumerate(zip(
        inputs.Train, inputs.Test
    )):

        model_name = f"model_{i_}.pt"
        log_name = f"model_logs_{i_}.json"
        save_model_path = path_to(path_model, model_name)

        # Check if the the model already exists
        if isfile(save_model_path):
            logger.info(f"Model was already trained: '{save_model_path}'")
        else:
            logger.info(f"Will save model in: "
                        f"'{path_to(path_model, model_name)}'")
            save_log_path = path_to(path_model, log_name)
            logger.info(
                f"Will save logged results in: "
                f"'{path_to(path_model, model_name)}'"
            )

            train_dataframe = pd.read_csv(path_to(
                'config', 'sets', train_path), index_col=0
            )
            train_dataframe.Paths = train_dataframe.Paths.apply(path_to)
            # train_dataframe = train_dataframe.loc[::500].reset_index()
            logger.info(f"Training on: "
                        f"'{path_to('config', 'sets', train_path)}'")

            test_dataframe = pd.read_csv(path_to(
                'config', 'sets',  test_path), index_col=0
            )
            test_dataframe.Paths = test_dataframe.Paths.apply(path_to)
            logger.info(f"Testing  on: "
                        f"'{path_to('config', 'sets', test_path)}'")

            trainDataset = Dataset(
                train_dataframe,
                transforms=transforms_train.get_transforms()
            )
            testDataset = Dataset(
                test_dataframe,
                transforms=transforms_test.get_transforms()
            )

            train_dataloader = DataLoader(
                trainDataset,
                batch_size=batch_size,
                shuffle=True
            )
            test_dataloader = DataLoader(
                testDataset,
                batch_size=batch_size,
                shuffle=True
            )

            net = resnet(layers=layers, channels=channels, num_classes=classes)

            optimizer = optim.Adam(
                net.parameters(),
                lr=inputs.Optimizer.lr,
                weight_decay=inputs.Optimizer.weight_decay
            )

            sched = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=inputs.Scheduler.gamma
            )

            classifier = Classifier(
                net=net,
                opt=optimizer,
                sched=sched,
                device=device
            )

            classifier.train(
                n_epochs=n_epochs,
                train_loader=train_dataloader,
                valid_loader=test_dataloader,
                checkpoint_path=path_checkpoint,
                checkpoint_freq=inputs.CheckpointFreq
            )

            # Save model weights etc.
            classifier.save(save_model_path)
            classifier.save_outputs(save_log_path)

            # End of training remove checkpoint file
            # Remove checkpoint file at the end of the training
            if isfile(path_checkpoint):
                remove(path_checkpoint)


if __name__ == "__main__":
    train(load_args())
