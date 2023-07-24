import os
os.environ['KMP_WARNINGS'] = 'off'
from albumentations import ImageOnlyTransform
import random
from stain_mixup.augment import stain_mixup
from stain_mixup.utils import get_stain_matrix
import numpy as np
import warnings
import dill as pickle
import cv2
from stain_mixup.utils import get_stain_matrix

class StainMix(ImageOnlyTransform):
    """
    wrapper around stain mix https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py#L2925
    """

    def __init__(
        self,
        stain_matrices,
        source_matrix,
        always_apply=False,
        p=0.5
    ):
        super(StainMix, self).__init__(always_apply, p)
        self.stain_matrices = stain_matrices if type(stain_matrices) == list else [stain_matrices]
        self.source_matrix = source_matrix

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self):

        return {
            "matrix": self.stain_matrices[random.randint(0, len(self.stain_matrices))-1]
        }

    def apply(self, img, matrix, alpha=0.2, **params):
        alpha = random.uniform(0.1, 0.9)
        return (stain_mixup(img, self.source_matrix, matrix, alpha=alpha)).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ('stain_matrices', 'source_matrix')


def load_stain_mix_matrix():
    with open('./stain_matrix_list.pkl', 'rb') as f:
        stain_matrices = pickle.load(f)
    return stain_matrices

def load_norm_source():
    source_m = get_stain_matrix(cv2.cvtColor(cv2.imread('./stain_mixup/reference.png'), 4))
    return source_m

def load_norm_source_img():
    source_m = cv2.cvtColor(cv2.imread('./stain_mixup/reference.png'), 4)
    return source_m

## add a pickle of a fit staintools norm?

if __name__ == '__main__':
    import cv2
    s = get_stain_matrix(cv2.cvtColor(cv2.imread('test_patches/test_img_ube_1.png'), 4))
    source = cv2.cvtColor(cv2.imread('test_patches/reference.png'), 4)
    sm =  get_stain_matrix(source)
    import matplotlib.pyplot as plt
    plt.imshow(stain_mixup(source,sm,s, alpha=0.9))
    smi = StainMix(s,sm)
    x=smi.apply(source[0:256,0:256,:], s)
    plt.imshow(x)
    smi.get_params()
