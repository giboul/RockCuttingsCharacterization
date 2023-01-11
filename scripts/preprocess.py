from Utils.Preprocessing import preprocess
from Utils.Utils import path_to, set_device
from os import listdir
from os.path import isfile, isdir
from sys import argv
import logging


"""This script is for pre-processing raw images"""


logger = logging.getLogger()
device = set_device()

def load_arguments(
    ifolder=path_to('data', 'Raw'),
    ofolder=path_to('data', 'New')
):
    if argv[1:]:
        for opt, arg in zip(argv[:-1], argv[1:]):
            if opt == '-i':
                ifolder = arg
            elif opt == '-o':
                ofolder = arg
    else:
        ifolder,
        ofolder

    if isdir(ifolder):
        return ifolder, ofolder
    else:
        logger.error(f"Folder '{ifolder}' is not a directory. "
                     f"If you want to preprocess a signle image, "
                     f"use the 'preprocessing.py' script")
        raise FileNotFoundError(f"{ifolder} is not a directory. "
                                f"If you want to preprocess a signle file,"
                                f"use the 'preprocessing.py' script")


def preprocess_folder(ifolder, ofolder, **kwargs):

    ifiles = [path_to(ifolder, f) for f in listdir(ifolder)]
    ifiles = [f for f in ifiles if isfile(f)]
    for file in ifiles:
        preprocess(file, ofolder, **kwargs)


def preprocess_folders(ifolder, ofolder, **kwargs):

    folders = [f for f in listdir(ifolder) if isdir(path_to(ifolder, f))]
    ofolders = [path_to(ofolder, folder) for folder in folders]
    ifolders = [path_to(ifolder, folder) for folder in folders]
    ifolders = [f for f in ifolders if isdir(f)]

    for ofolder, ifolder in zip(ofolders, ifolders):
        logger.info(f"Processing {ifolder = }")
        logger.info(f"To folder  {ofolder = }")
        preprocess_folder(ifolder, ofolder)


if __name__ == "__main__":
    args = load_arguments()
    print(args)
    # preprocess_folders(*args)