import torch
import torch.nn as nn
import torchvision

# def avg_weights_across_input_channel(model, new_conv, old_conv):
#     # Average existing RGB weights across input channels to preserve pretrained weights
#         with torch.no_grad():
#             new_conv.weight = nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
#         model.backbone.body.conv1 = new_conv
    
#         return model

# def load_single_channel_fasterrcnn_model():
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight="DEFAULT")
#     old_conv = model.backbone.body.conv1
#     single_channel_conv = nn.Conv2d(
#         in_channels=1,
#         out_channels=old_conv.out_channels,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=old_conv.bias
#     )

#     return avg_weights_across_input_channel(model, single_channel_conv, old_conv)

def load_single_channel_fasterrcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight="DEFAULT")
    load_single_channel_model(model, 'backbone.body.conv1')

# def load_single_channel_vit_model():
#     pass

# def load_single_channel_swin_model():
#     pass

def load_single_channel_model(model, layer_path):
    parts = layer_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    old_conv = getattr(parent, parts[-1])

    single_channel_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias
    )

    # Average existing RGB weights across input channels to preserve pretrained weights
    with torch.no_grad():
        single_channel_conv.weight = nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
    model.backbone.body.conv1 = single_channel_conv


    # Only for model.transform.image_mean only exist in Torchvision object detection models (like Faster R-CNN),
    if hasattr(model, 'transform'):
        model.transform.image_mean = [0.5]
        model.transform.image_std = [0.5]

    setattr(parent, parts[-1], single_channel_conv)
    return model