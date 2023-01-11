import torch
import pandas as pd
import os

from sklearn.metrics import accuracy_score  # import scikit-learn
from numpy import array

from Utils.AttrDict import load_config
from Utils.Dataloader import Transforms, Dataset, dict_transform
from Utils.BaseModel import Classifier
from Utils.ResNet import resnet
from Utils.Utils import set_device, TransformTestItems, path_to, load_args
import logging

logger = logging.getLogger()
device = set_device()

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

# TODO to train a new set
# - Prepare config files for all models to train (*.yaml, *.csv and folder)
# - Train model (the best one) on whole data
# - ?: Post Processing : plot mean val acc + std
# - ?: Post Processing : plot expert prediciton vs. baseline raw vs. cropped vs. mar
### python train.py -i .\\config\\MAR_RESNET34_CROPPED_256_borehole_train.yaml


def test(config=""):
    # Load config_file
    inputs = load_config(load_args(config))

    layers = inputs.Model.Layers
    classes = inputs.Model.OutClasses
    channels = inputs.Model.Channels

    net = resnet(layers=layers, channels=channels, num_classes=classes)
    classifier = Classifier(net=net, device=device)

    path_model = path_to(inputs.PathSave, inputs.ModelName)
    logger.info(f"Path to model is '{path_model}'")

    # Handle config_file inputs
    batch_size = int(inputs.BatchSize)

    # Transforms Train and Test
    transforms_test = Transforms(TransformTestItems(
        inputs.TransformTest.items(), dict_transform
    ))

    valid_outputs = [None for _ in inputs.Test]
    losses = valid_outputs.copy()

    predictions, labels = [], []

    for i, data_path in enumerate(inputs.Test):
        print(f"##### Testing {data_path} #####")
        model_path = path_to(path_model, f"model_{i}.pt")
        logger.info(f"Test model: '{model_path}'")

        test_dataframe = pd.read_csv(path_to('config', 'sets',  data_path), index_col=0)
        test_dataframe.Paths = test_dataframe.Paths.apply(path_to)
        logger.info(f"Testing on: '{path_to('config', 'sets', data_path)}'")

        testDataset = Dataset(
            test_dataframe,
            transforms=transforms_test.get_transforms()
        )
        valid_loader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=batch_size,
            shuffle=True
        )
        # Check if the the model already exists
        if os.path.isfile(model_path):
            print(f"Loading classifier... {model_path}")
            classifier.load(model_path, map_location=torch.device(device))
            print(f"Loading done. Predicting...")
            pred, label, loss = [
                array(x) for x in classifier.predict(valid_loader)
            ]
            predictions.append(pred)
            labels.append(label)
            print(f"Calculating accuracy score...")
            valid_outputs[i] = accuracy_score(label, pred)
            print("Predictions done. Calculating loss...")
            losses[i] = loss.mean()
            print("Loss calculated.")
        else:
            logger.error(f"{model_path} is not a file")
            raise FileNotFoundError(f"{model_path} is not a file")
    return valid_outputs, losses, predictions, labels


if __name__=='__main__':
    acc, loss = test(config="debugconfig.yaml")
    print(acc, loss)
