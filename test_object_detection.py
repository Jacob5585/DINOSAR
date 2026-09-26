import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from dataset import SARDet100KDataset, collate_fn
from model_adapter import combine_backbone_detection_head
from metrics import evaluate_predictions, save_result

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

@torch.no_grad()
def test(model, data_loader, device, label_to_category_id, score_threshold=0.0):
    model.eval()
    predictions = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue

                x1, y1, x2, y2 = box.tolist()

                predictions.append({
                    "image_id": image_id,
                    "category_id": label_to_category_id[int(label)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

        if (batch_idx + 1) % 20 == 0:
            print(f"  processed {batch_idx + 1}/{len(data_loader)} batches "
                  f"({len(predictions)} detections so far)")

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate a DINO ViT/Swin + Faster R-CNN model on SARDet-100K.")
    parser.add_argument("--model_type", required=True, choices=["vit", "swin"])
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--detection_head_checkpoint", required=True)
    # parser.add_argument("--image_dir", required=True)
    # parser.add_argument("--annotation_file", required=True)
 
    parser.add_argument("--backbone_checkpoint_key", default="student", help="Which entry of the DINO checkpoint dict to read ('student' or 'teacher').")
    parser.add_argument("--arch", default=None, help="Defaults to vit_small for --model_type vit, swin_tiny for swin.")
    parser.add_argument("--patch_size", type=int, default=None, help="Defaults to 16 for ViT, 4 for Swin.")
    parser.add_argument("--window_size", type=int, default=7, help="Swin only.")
    parser.add_argument("--in_chans", type=int, default=1)
 
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Drop detections below this confidence before scoring mAP.")
    parser.add_argument("--predictions_out", default=None, help="Optional path to save raw COCO-format predictions as json.")
    parser.add_argument("--metrics_output", default="metrics_report.txt", help="Path to write the metrics report to (a .json with the same base name is written alongside it with raw numbers).")
    args = parser.parse_args()
 
    default_arch = "vit_small" if args.model_type == "vit" else "swin_tiny"
    default_patch = 16 if args.model_type == "vit" else 4
    arch = args.arch or default_arch
    patch_size = args.patch_size or default_patch

    image_dir = "datasets/SARDet_100K/JPEGImages/test/"
    annotation_file = "datasets/SARDet_100K/test.json"
 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")


    dataset = SARDet100KDataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transforms=ToTensor(),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print(f"Loaded {len(dataset)} images from {image_dir}")
    print(f"Building {args.model_type} backbone from {args.backbone_checkpoint}")
    print(f"(checkpoint_key={args.backbone_checkpoint_key}")

    label_to_category_id = {v: k for k, v in dataset.catagory_id_to_label.items()}

    model = combine_backbone_detection_head(
        args.model_type,
        args.backbone_checkpoint,
        args.detection_head_checkpoint,
        backbone_checkpoint_key=args.backbone_checkpoint_key,
        in_chans=args.in_chans,
        arch=arch,
        patch_size=patch_size,
        window_size=args.window_size,
        num_classes=7
    )
    model.to(device)

    print("================")
    print("===== TEST =====")
    print("================")

    predictions = test(model, data_loader, device, label_to_category_id, score_threshold=args.score_threshold)

    result = evaluate_predictions(annotation_file, predictions)
    save_result(result, args.metrics_output)

if __name__ == "__main__":
    main()