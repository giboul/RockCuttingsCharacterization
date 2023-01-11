import random
import numpy
import torch
import logging
from sys import argv
from os.path import dirname, join, isabs, realpath, isfile


project_path = realpath(__file__)
for _ in range(3):
    # As many times as needed so that project_path becomes the absolute path
    # '.../Rock-Cuttings-Characterization' or however you called this
    # project's folder
    project_path = dirname(project_path)


def path_to(*path: str):
    """
    Takes a relative path in the project as a single string
    or a sequence of paths to join as for this project
    -> returns an absolute path
    """
    joined = join(*path)
    if isabs(join(joined)):
        return joined
    else:
        return join(project_path, joined)


logging.basicConfig(
    filename=path_to('Scripts', 'scripts.log'),
    level=logging.INFO,
    format="{levelname:<7} {asctime} | ({filename}:{lineno:<3}) {message}",
    style="{",
    filemode='a'  # 'w'
)
logger = logging.getLogger()


def set_seed(seed):
    """ Set the random seed """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(
            f"CUDA device name is: {torch.cuda.get_device_name(device)}"
        )
    else:
        logger.warning(f"Running on {device}, not on CUDA")
    return device


def TransformTestItems(items, dic):
    return [dic[key]([k for k in item.values()] if len(item.values()) > 1
            else [k for k in item.values()][0])
            for key, item in items]


def load_args(input_file=''):

    if not input_file:
        nargs = len(argv)
        if nargs == 2:
            _, input_file = argv
        elif nargs > 2:
            print(argv)
            raise FileNotFoundError(
                f'Give at most one argument to this script '
                f'default: debug files. '
                f'If run from a notebook, pass input_file="..." '
                f'argument to load_args()'
            )

    if not input_file:
        logger.warning(
            "input config file option not given, running on debug set"
        )
        input_file = path_to('config', 'debugconfig.yaml')
    else:
        input_file = path_to('config', input_file)
        logger.info(f"Training on the set: '{input_file}'")

    if not isfile(input_file):
        raise FileNotFoundError(f"'{input_file}' is not an existing file")

    return input_file


if __name__ == '__main__':

    print(project_path)
    print(path_to('config', 'sets', 'borehole', 'test_mar_0.csv'))
    print(path_to(path_to(
        '/home/axel/Documents/GitHub/Rock-Cuttings-Characterization/config',
        'sets', 'borehole', 'test_mar_0.csv'))
    )
    logger.warning('Test')
    logger.info('Test')
    logger.error('Test')
