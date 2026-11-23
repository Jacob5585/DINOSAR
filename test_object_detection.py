import torch

checkpoint = torch.load(
    "output_dir_reg_vit/checkpoint.pth",
    map_location="cpu",
    weights_only=False
)

print(checkpoint.keys())


checkpoint = torch.load(
    "output_dir_swin/checkpoint.pth",
    map_location="cpu",
    weights_only=False
)

print(checkpoint.keys())