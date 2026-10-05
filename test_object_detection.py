import argparse
import json
import os
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from dataset import SARDet100KDataset, collate_fn
from model_adapter import combine_backbone_detection_head
from metrics import evaluate_predictions, save_result

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def save_visualizations(images, targets, outputs, label_to_category_id, output_dir, sample_rate=0.10, threshold=0.3):
    """Draws Ground Truth (Green) and Predicted (Red) boxes on ~10% of testing images."""
    viz_dir = os.path.join(output_dir, "eval_images")
    os.makedirs(viz_dir, exist_ok=True)

    category_id_to_name = {v: str(k) for k, v in label_to_category_id.items()}

    for img_tensor, target, output in zip(images, targets, outputs):
        if random.random() > sample_rate:
            continue

        image_id = int(target["image_id"].item())
        
        # Convert tensor back to PIL Image (handles grayscale and RGB)
        img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(axis=2)
            pil_img = Image.fromarray(img_np, mode="L").convert("RGB")
        else:
            pil_img = Image.fromarray(img_np, mode="RGB")

        draw = ImageDraw.Draw(pil_img)

        # Draw Ground Truth Boxes (Green)
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            cat_name = category_id_to_name.get(int(label), str(label))
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, max(0, y1 - 10)), f"GT: {cat_name}", fill="green")

        # Draw Predictions (Red)
        pred_boxes = output["boxes"].cpu().numpy()
        pred_scores = output["scores"].cpu().numpy()
        pred_labels = output["labels"].cpu().numpy()

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            cat_name = category_id_to_name.get(label_to_category_id.get(int(label)), str(label))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{cat_name} {score:.2f}", fill="red")

        pil_img.save(os.path.join(viz_dir, f"img_{image_id}.png"))

@torch.no_grad()
def test(model, data_loader, device, label_to_category_id, output_dir, score_threshold=0.0, sample_rate=0.10):
    model.eval()
    predictions = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        images_device = [img.to(device) for img in images]
        outputs = model(images_device)

        # Save visual comparison for ~10% of images
        save_visualizations(
            images, targets, outputs, 
            label_to_category_id, output_dir, 
            sample_rate=sample_rate, 
            threshold=max(0.30, score_threshold)
        )

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
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches "
                  f"({len(predictions)} detections so far)")

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate a DINO ViT/Swin + Faster R-CNN model on SARDet-100K.")
    parser.add_argument("--model_type", required=True, choices=["vit", "swin"])
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--detection_head_checkpoint", required=True)
    parser.add_argument("--backbone_checkpoint_key", default="student")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--in_chans", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--viz_sample_rate", type=float, default=0.10, help="Fraction of images (0.10 = 10%) to output with GT/Pred overlay.")
    args = parser.parse_args()

    output_dir = os.path.join("results", args.backbone_checkpoint.strip('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    arch = args.arch or ("vit_small" if args.model_type == "vit" else "swin_tiny")
    patch_size = args.patch_size or (16 if args.model_type == "vit" else 4)

    image_dir = "datasets/SARDet_100K/JPEGImages/test/"
    annotation_file = "datasets/SARDet_100K/test.json"
    NUM_CLASSES = 7

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
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

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
        num_classes=NUM_CLASSES
    )
    model.to(device)

    print("==========================================")
    print(f" ===== EVALUATING: {args.model_name} =====")
    print("==========================================")

    predictions = test(
        model, data_loader, device, 
        label_to_category_id, output_dir, 
        score_threshold=args.score_threshold, 
        sample_rate=args.viz_sample_rate
    )

    # Save raw predictions JSON inside output directory
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    result = evaluate_predictions(annotation_file, predictions)
    save_result(result, output_dir)

if __name__ == "__main__":
    main()