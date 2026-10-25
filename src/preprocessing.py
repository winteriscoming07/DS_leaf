"""Deterministic evaluation preprocessing (shared by train/eval/inference) and train-only augmentation."""
import numpy as np
from PIL import Image
from skimage.feature import hog

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def hog_features(img, size=128):
    """Grayscale -> resize -> HOG. Returns a 1-D float array."""
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    return hog(np.asarray(g, dtype=np.float32) / 255.0, orientations=9,
               pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")


def eval_transform(size=224):
    """Resize (whole leaf kept, no crop) + ImageNet normalisation. No randomness."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(size=224):
    """Mild augmentation. Applied to training images ONLY."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                      T.ColorJitter(brightness=0.2, contrast=0.2), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
