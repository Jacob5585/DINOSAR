import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from peft import get_peft_model, LoraConfig
from collections import OrderedDict

import vision_transformer as vits
import swin_transformer as swins

lora_config = LoraConfig(
    r = 16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["qkv", "proj"],
    bias="none"
)

FPN_OUT_CHANNELS = 256

def extract_backbone(checkpoint, key):
    state_dict = checkpoint.get(key, checkpoint)

    for prefix in ("module.backbone.", "backbone."):
        matched = { k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix) }

        if matched:
            lora_state = any(k.startswith("base_model.model.") for k in matched)
            return matched, lora_state

def load_model(model, backbone, lora_state):
    if lora_state:
            model = get_peft_model(model, lora_config()).merge_and_unload()

    model = model.load_state_dict(backbone, strict=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model

def load_vit_backbone(checkpoint_path, arch, patch_size, in_chans, key='student'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    backbone, lora_state = extract_backbone(checkpoint, key)

    model = vits.__dict__[arch](
        patch_size=patch_size,
        in_chans=in_chans
    )

    load_model(model, backbone, lora_state)

    return model

def load_swin_backbone(checkpoint_path, arch, patch_size, window_size, in_chans, key='student'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    backbone, lora_state = extract_backbone(checkpoint, key)

    model = swins.__dict__[arch](
        patch_size=patch_size,
        in_chans=in_chans,
        window_size=window_size,
        use_dense_prediction=True
    )

    model = load_model(model, backbone, lora_state)

    return model

def vit_spatial_map(model, x):
    """
    Explain what is going on and why
    """

    B, _, H_image, W_image = x.shape
    patch_size = model.patch_embed.patch_size
    Hp, Wp = H_image // patch_size, W_image // patch_size

    tokens = model.prepare_tokens(x)

    for blk in model.blocks:
        tokens = blk(tokens)

    tokens = model.norm(tokens)

    patch_tokens = tokens[:, 1:, :]
    C = patch_tokens.shape[-1]
    feature_map = patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H_image, W_image)

    return feature_map

def swin_spatial_map(model, x):
    """
    """
    tokens, H, W = model.patch_embed(x)
    tokens = model.pos_drop(tokens)

    stage_maps = []
    for i, layer in enumerate(model.layers):
        x_out, H_out, W_out, tokens, H, W = layer(tokens, H, W)

        if hasattr(model, "stage_norms"):
            x_out = model.stage_norm[i](x_out)

        C = x.out.shape[-1]
        feature_map = x_out.transpose(1, 2).reshape(x_out.shape[0], C, H_out, W_out)
        stage_maps.append(feature_map)

    return stage_maps

class ViTFeaturePyramidbackboneAdapter(nn.Module):
    """
    Explain what is going on and why
    """
    def __init__(self, model, out_channels=FPN_OUT_CHANNELS):
        super().__init__()
        self.model = model
        dim = model.embed_dim
        self.out_channels = out_channels

        self.stride4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )
        self.stride8 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.stride16 = nn.Identitiy()
        self.stride32 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[dim, dim, dim, dim],
            out_channels=out_channels,
            extract_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        base = vit_spatial_map(self.model, x)
        levels = OrderedDict()
        levels["0"] = self.stride4(base)
        levels["1"] = self.stride8(base)
        levels["2"] = self.stride16(base)
        levels["3"] = self.stride32(base)

        return self.fpn(levels)

class SwinFeaturePyramidbackboneAdapter(nn.Module):
    """
    """
    def __init__(self, model, out_channels=FPN_OUT_CHANNELS):
        super().__init__()
        self.model = model
        self.out_channels = out_channels

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(model.embed_dims),
            out_channels=out_channels,
            extra_block=LastLevelMaxPool(),
        )

    def forward(self, x):
        stage_maps = swin_spatial_map(self.model, x)
        levels = OrderedDict((str(i), fm) for i, fm in enumerate(stage_maps))

        return self.fpn(levels)