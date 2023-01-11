from test import test
from Utils.Utils import path_to
import json


def main(*configs):

    for config in configs:
        accuracies, losses, predictions, labels = test(config)

        for i, (accuracy, loss, prediction, label) in enumerate(zip(
            accuracies, losses, predictions, labels
        )):
            with open(path_to('outputs', f'{config}-{i}'), "w") as outfile:
                outfile.write(json.dumps(dict(
                    accuracy=accuracy,
                    loss=loss,
                    prediction=prediction,
                    label=label
                )))


if __name__ == '__main__':
    main(
        'MAR_RESNET18_PADDED_256_lab_borehole-lab_borehole.yaml',
        'MAR_RESNET18_PADDED_256_lab_borehole-lab.yaml',
        'MAR_RESNET18_PADDED_256_lab_borehole-borehole.yaml',
        'MAR_RESNET18_PADDED_256_borehole-lab.yaml',
        'MAR_RESNET18_PADDED_256_lab-borehole.yaml',
        'MAR_RESNET34_CROPPED_256_lab_train.yaml',
        'RN34_CROPPED_256_borehole-lab.yaml',
        'MAR_RESNET34_CROPPED_256_lab-borehole.yaml',
        'MAR_RESNET34_CROPPED_256_borehole.yaml',
    )
