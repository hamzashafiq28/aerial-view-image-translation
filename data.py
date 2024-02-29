from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


class SAREOTranslationDataset_test(Dataset):
    def __init__(self, root_dir, transform=None, input_domain="SAR"):
        self.sar_root_dir = root_dir

        self.transform = transform


        self.sar_image_filenames = sorted(os.listdir(root_dir))





    def __len__(self):
        return len(self.sar_image_filenames)

    def __getitem__(self, idx):
        sar_filename = os.path.join(self.sar_root_dir, self.sar_image_filenames[idx])

        sar_image = Image.open(sar_filename).convert('RGB')

        if self.transform:
            sar_image = self.transform(sar_image)

        return sar_image, sar_filename



import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import glob

class SAREOTranslationDataset_test1(Dataset):
    def __init__(self, root_dir="/Volumes/Storage/train/", transform=None, input_domain="RGB"):
        """
        :param root_dir: Directory pointing to root training data folder
        :param transform: List of transforms to be applied to each image
        :param input_domain: Choose one of the input domains: "RGB", "IR", or "SAR"
        """
        self.root_dir = root_dir
        self.transform = transform
        self.input_domain = input_domain


        # Collect images based on the selected input domain
        self.images = glob.glob(os.path.join(root_dir, '*.tiff'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Read the image
        image_path = self.images[idx]
        image = cv2.imread(image_path)



        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, image_path


