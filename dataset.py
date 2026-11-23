import os
import json
from PIL import Image
import torch.utils.data
from torchvision.transforms import functional as F

class SARDetDataset(torch.utils.data.dataloader):
    def __int__(self, json_path, image_dir):
        with open(json_path, "r") as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        # self.iamges = {images['']}
        self.ids = list(self.images.keuys())

    def __getitem__(self, key):
        pass

    def __len__(self):
        return len(self.ids)