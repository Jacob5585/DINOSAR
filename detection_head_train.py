import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import argparse

from model_adapter import load_adapted_model
from dataset import SARDet100KDataset, collate_fn

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def train_one_epoch(model, frozen_backbone, optimizer, data_loader, device, epoch):
    model.train()
    frozen_backbone.eval()
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
    parser = argparse.ArgumentParser(
        description="Train a Faster R-CNN detection head directly on a frozen DINO ViT/Swin backbone."
    )
    parser.add_argument("--model_type", required=True, choices=["vit", "swin"])
    parser.add_argument("--backbone_checkpoint", required=True, help="Path to the DINO-pretrained (optionally LoRA) backbone checkpoint.")
    parser.add_argument("--backbone_checkpoint_key", default="student", help="Which entry of the DINO checkpoint dict to read ('student' or 'teacher').")
    parser.add_argument("--arch", default=None, help="Defaults to vit_small for vit, swin_tiny for swin.")
    parser.add_argument("--patch_size", type=int, default=None, help="Defaults to 16 for vit, 4 for swin.")
    parser.add_argument("--window_size", type=int, default=7, help="Swin only.")
    parser.add_argument("--in_chans", type=int, default=1)
    # parser.add_argument("--image_dir", default="datasets/SARDet_100K/JPEGImages/train_val")
    # parser.add_argument("--annotation_file", default="datasets/SARDet_100K/Annotations/train_val.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    # parser.add_argument("--output_dir", default="output_dir_detection_head")
    args = parser.parse_args()
 
    arch = args.arch or ("vit_small" if args.model_type == "vit" else "swin_tiny")
    patch_size = args.patch_size or (16 if args.model_type == "vit" else 4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    IMAGE_DIR = "datasets/SARDet_100K/JPEGImages/train_val"
    ANNOTATION_FILE = "datasets/SARDet_100K/Annotations/train_val.json"
    OUTPUT_DIR = "output_dir_detection_head"
    NUM_CLASSES = 7 # background + 6 catagories

    dataset = SARDet100KDataset(image_dir=IMAGE_DIR, annotation_file=ANNOTATION_FILE, transforms=ToTensor())
    data_loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=8, 
        collate_fn=collate_fn
    )

    adapted_backbone = load_adapted_model(
        args.model_type,
        checkpoint_path=args.backbone_checkpoint,
        in_chans=args.in_chans,
        arch=arch,
        patch_size=patch_size,
        window_size=args.window_size,
    )

    frozen_adapted_backbone = adapted_backbone.model
    for p in frozen_adapted_backbone.parameters():
        p.requires_grad = False
    
    model = FasterRCNN(adapted_backbone, num_classes=NUM_CLASSES)
    model.transform.image_mean = [0.449]
    model.transform.image_std = [0.226]
    model.to(device)
    
    # 4. Filter parameters to pass ONLY trainable parameters to SGD
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    epochs = 10 # 5
    for epoch in range(epochs):
        train_one_epoch(model, frozen_adapted_backbone, optimizer, data_loader, device, epoch)

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/{arch}_fasterrcnn_head_sardet100k.pth")
    print("Training finished. Checkpoint saved to fasterrcnn_head_sardet100k.pth")

if __name__ == "__main__":
    main()
