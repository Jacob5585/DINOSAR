import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

from dataset import SARDet100KDataset, collate_fn
import load_models

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if (i +1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")

    print(f"Epoch [{epoch+1}] Complete | Avg Loss: {total_loss / len(data_loader):.4f}")

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    IMAGE_DIR = "datasets/SARDet_100K/JPEGImages/train_val"
    ANNOTATION_FILE = "datasets/SARDet_100K/Annotations/train_val.json"
    NUM_CLASSES = 7 # background + 6 catagories

    # model = load_model()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight="DEFAULT")
    model = load_models.load_single_channel_model(model, 'backbone.body.conv1')

    for param in model.backbone.parameters():
        param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model.to(device)

    dataset = SARDet100KDataset(image_dir=IMAGE_DIR, annotation_file=ANNOTATION_FILE, transforms=ToTensor())
    data_loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=8, 
        collate_fn=collate_fn
    )
    
    # 4. Filter parameters to pass ONLY trainable parameters to SGD
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    epochs = 10 # 5
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch)

    torch.save(model.state_dict(), "output_dir_detection_head/fasterrcnn_head_sardet100k.pth")
    print("Training finished. Checkpoint saved to fasterrcnn_head_sardet100k.pth")

if __name__ == "__main__":
    main()
