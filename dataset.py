import os
import json
from PIL import Image
import torch.utils.data
from torchvision.transforms import functional as F

class SARDet100KDataset(torch.utils.data.dataloader):
    def __int__(self, json_path, image_dir):
        self.image_dir = image_dir

        with open(json_path, "r") as f:
            self.annotations = json.load(f)

        self.images = self.annotations['images']

        catagories = self.annotations.get['categories', []]
        if catagories:
            self.catagory_id_to_label = {catagorie['id']: i + 1 for i, catagorie in enumerate(catagories)}
        else:
            self.catagory_id_to_label = None

        self.image_to_annotation = {}
        for annotation in self.annotations.get('annotation', []):
            image_id = self.annotation['image_id']

            if image_id not in self.image_to_annotation:
                self.image_to_annotation[image_id].append(annotation)
   
    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path)
        width, height = image.size
        image_id = image_info['id']
        annotations = self.image_to_annotation(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annotation in annotations:
            x, y, w, h = annotation['bbox']

            # Convert to [xmin, ymin, xmax, ymax]
            xmin = max(0.0, float(x))
            ymin = max(0.0, float(y))
            xmax = min(float(width), float(x + w))
            ymax = min(float(height), float(y + h))

            # Filter zero-area or inverted bounding boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])

            category_id = annotation['category_id']
            if self.catagory_id_to_label:
                label = self.catagory_id_to_label[category_id]
            else:
                label = category_id if category_id > 0 else category_id + 1

            areas.append(annotation.get('area', (xmax - xmin) * (ymax - ymin)))
            iscrowd.append(annotation.get('iscrowd', 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))