import kagglehub
import shutil
import os
import json

dest_path = "datasets"
train_path = f"{dest_path}/SARDet_100K/JPEGImages/train"
val_path = f"{dest_path}/SARDet_100K/JPEGImages/val"
train_val_path = f"{dest_path}/SARDet_100K/JPEGImages/train_val"

train_json_path = f"{dest_path}/SARDet_100K/Annotations/train.json"
val_json_path = f"{dest_path}/SARDet_100K/Annotations/val.json"
train_val_json_path = f"{dest_path}/SARDet_100K/Annotations/train_val.json"

def download():
    path = kagglehub.dataset_download("greatbird/sardet-100k", output_dir=dest_path)
    print("Path to dataset files:", path)

def combine_train_val():
    print("Merging validation into train:")

    train_images = os.listdir(train_path)
    val_images = os.listdir(val_path)
    os.makedirs(train_val_path, exist_ok=True)

    for image in train_images:
        shutil.copy(os.path.join(train_path, image), train_val_path)

    for image in val_images:
        shutil.copy(os.path.join(val_path, image), train_val_path)

    with open(f'{train_json_path}', 'r') as f:
        train_json = json.load(f)

    with open(f'{val_json_path}', 'r') as f:
        val_json = json.load(f)

    merged = {
        "images": train_json["images"].copy(),
        "annotations": train_json["annotations"].copy(),
        "categories": train_json["categories"].copy()
    }

    # Add images, annotations from file val
    merged["images"].extend(val_json["images"])
    merged["annotations"].extend(val_json["annotations"])

    with open(f"{train_val_json_path}", "w") as f:
        json.dump(merged, f, indent=2)

def main():
    download()
    combine_train_val()