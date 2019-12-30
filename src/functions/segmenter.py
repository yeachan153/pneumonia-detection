import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from keras.backend import flatten
from keras.backend import sum
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


class IncorrectInputDirException(Exception):
    """Exception raised when the input directory
    yields no items
    """
    pass


class LungSegmenter():
    """This class is used to segment the lung from x-rays.
    Original source code can be found here:
    https://www.kaggle.com/eduardomineo/lung-segmentation-
    of-rsna-challenge-data
    """

    def __init__(
        self, model_path='models/lung_segmentation/unet_lung_seg.hdf5'
    ):
        self.model = load_model(
            model_path,
            custom_objects={
                'dice_coef_loss': self._dice_coef_loss,
                'dice_coef': self._dice_coef
            })
        self.img_ends = ['jpg', 'png', 'jpeg']

    def _dice_coef(self, y_true, y_pred):
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        intersection = sum(y_true_f * y_pred_f)
        num = 2. * intersection + 1
        den = sum(y_true_f) + sum(y_pred_f) + 1
        return num/den

    def _dice_coef_loss(self, y_true, y_pred):
        return -self._dice_coef(y_true, y_pred)

    def _image_to_train(self, img):
        npy = img / 255
        npy = np.reshape(npy, npy.shape + (1,))
        npy = np.reshape(npy, (1,) + npy.shape)
        return npy

    def _train_to_image(self, npy):
        img = (npy[0, :, :, 0] * 255.).astype(np.uint8)
        return img

    def segment_image(self, img_path):
        pid, fileext = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img.img_to_array(img)
        img = cv2.resize(img, (512, 512))
        segm_ret = self.model.predict(
            self._image_to_train(img),
            verbose=0
        )
        img = cv2.bitwise_and(img, img, mask=self._train_to_image(segm_ret))
        return img, pid, fileext

    def write_to_disk(self, img, save_dir, pid, fileext):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        write_path = os.path.join(save_dir, f"{pid}{fileext}")
        cv2.imwrite(write_path, img)

    def segment_to_disk(self, save_dir, input_dir):
        paths = os.listdir(input_dir)
        img_paths = [
            os.path.join(input_dir, x) for x in paths if
            x.split('.')[-1] in self.img_ends
        ]
        if len(img_paths) == 0:
            raise IncorrectInputDirException(
                "Check that your input directory is correct"
            )
        img_paths = list(set(img_paths))
        for img_path in tqdm(img_paths, total=len(img_paths)):
            img, pid, fileext = self.segment_image(img_path)
            self.write_to_disk(img, save_dir, pid, fileext)
        return "Done"
