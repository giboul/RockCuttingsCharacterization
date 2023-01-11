import numpy as np
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, dilation, closing  # pip install scikit-image
from skimage.measure import regionprops, label
from PIL.ImageDraw import Draw
from PIL.Image import new,  composite
from PIL import Image
from os.path import basename, isdir, isfile, join, realpath, dirname
from os import mkdir
import logging
logger = logging.getLogger()


if __name__ == '__main__':
    PATH = realpath(__file__)
    for _ in range(3):
        PATH = dirname(PATH)
else:
    PATH = ""


def segregate(image, mask, upper_bound=1, lower_bound=0):
    """Find contours, separate them and crop the sub-images
    lower_bound in [0,1] defines the minimum area of a contour
    upper_bound in [0,1] defines the maximum area of a contour"""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.prod(image.shape)
    contours = np.array([
        contour for contour in contours
        if area*upper_bound > cv2.contourArea(contour) > lower_bound*area
    ], dtype=object)

    outs = [None for _ in contours]
    for i, _ in enumerate(contours):
        mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, i, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(image) # Extract out the object and place into output image
        out[mask == 255] = image[mask == 255]

        # Now crop
        y, x = np.where(mask == 255)
        topy, topx = np.min(y), np.min(x)
        bottomy, bottomx = np.max(y), np.max(x)
        outs[i] = out[topy:bottomy+1, topx:bottomx+1]
    
    return outs


def preprocess_image(image,
                     filename=join(PATH, "data", "Raw", "debug.jpeg"),
                     save_folder=join(PATH, "data", "New"),
                     erosion_radius=3, nerosions=2,
                     upper_bound=0.9, lower_bound=0.001,
                     show=False):
    """Takes in a single image path to preprocess
    The new image will be saved at 'save_foler' """
    erosion_disk = disk(radius=erosion_radius)
    # 1) Original scan
    #     'image' variable
    # 2) Threshold
    _, mask = cv2.threshold(image.astype('uint8')*255, 0, 255, cv2.THRESH_OTSU)
    # 3) Erosion & labelize (segregate samples)
    mask = cv2.erode(mask, erosion_disk, iterations=nerosions)
    # separate samples
    samples = segregate(image, mask, upper_bound, lower_bound)
    # 4) Dilation
    for i, sample in enumerate(samples):
        sample = cv2.dilate(sample, erosion_disk, iterations=nerosions)

        filename, ext = filename.split('.')
        filename = f"{filename}-{i}.{ext}"
        cv2.imwrite(join(save_folder, filename), sample)

        if show:
            plt.imshow(sample)
            plt.show()


def preprocess(
    path=join(PATH, "data", "Raw", "debug.jpeg"),
    save_folder=join(PATH, "data", "New"),
    **kwargs
):
    """This function checks that the paths are valid then calls 'preprocess_image'"""
    # Check that file and directory are fine
    if not isfile(path):
        logger.error(f"File '{path}' was not found")
        raise FileNotFoundError(path)
    if not isdir(save_folder):
        logger.warning(f"'{save_folder}' directory does not exist")
        if not isdir(dirname(save_folder)):
            mkdir(dirname(save_folder))
        mkdir(save_folder)
        logger.info(f"'{save_folder}' created")
    
    # Preprocess
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    preprocess_image(image, basename(path), save_folder, **kwargs)


if __name__ == '__main__':
    preprocess(show=True)
