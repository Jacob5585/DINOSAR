import kagglehub
import shutil
import os
import json

# Download latest version
path = kagglehub.dataset_download("greatbird/sardet-100k")
dest_path = "datasets"
val_path = f"{dest_path}/SARDet_100K/JPEGImages/val"
train_path = f"{dest_path}/SARDet_100K/JPEGImages/train"

print("Path to dataset files:", path)

print("Moving dataset to:", dest_path)
shutil.move(path, dest_path)

print("Merging validation into train:")

val_images = os.listdir(val_path)
for image in val_images:
    shutil.move(os.path.join(val_path, image), train_path)

os.rmdir(val_path)

# This isnt needed since the annotations won't be used during DINO traing, but added incase there every comes a need for it
with open(f'{dest_path}/SARDet_100K/Annotations/train.json', 'r') as f:
    train_json = json.load(f)

with open(f'{dest_path}/SARDet_100K/Annotations/val.json', 'r') as f:
    val_json = json.load(f)

merged = {
    "images": train_json["images"].copy(),
    "annotations": train_json["annotations"].copy(),
    "categories": train_json["categories"].copy()
}

# Add images from file 2
merged["images"].extend(val_json["images"])

# Add annotations from file 2
merged["annotations"].extend(val_json["annotations"])

with open(f"{dest_path}/SARDet_100K/Annotations/train_val.json", "w") as f:
    json.dump(merged, f, indent=2)