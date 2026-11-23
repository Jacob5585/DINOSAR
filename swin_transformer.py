####################################################################################################
# https://github.com/microsoft/Swin-Transformer
# @misc{hwang2022tutel,
#       title={Tutel: Adaptive Mixture-of-Experts at Scale}, 
#       author={Changho Hwang and Wei Cui and Yifan Xiong and Ziyue Yang and Ze Liu and Han Hu and Zilong Wang and Rafael Salas and Jithin Jose and Prabhat Ram and Joe Chau and Peng Cheng and Fan Yang and Mao Yang and Yongqiang Xiong},
#       year={2022},
#       eprint={2206.03382},
#       archivePrefix={arXiv}
# }
# Claude to help with implentation/connecting swin transformer implentation into DINO
####################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial

from utils import trunc_normal_

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()

    return x.div(keep_prob) * random_tensor

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def window_partition(x, window_size):
    """
    (B, H, W, C) -> (num_windows*B, window_size, window_size, C)
    """
    
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size, H, W):
    """
    (num_windows*B, window_size, window_size, C) -> (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x

def get_window_size(x_size, window_size, shift_size):
    """
    Shrink window_size/shift_size for axes where the feature map is
    smaller than the requested window. This is what lets 96px local crops
    (which end up as tiny 6x6 / 3x3 feature maps in the deeper stages) run
    through a backbone whose window_size was chosen for 224px crops
    """

    use_window_size = list(window_size)
    use_shift_size = list(shift_size)

    for i in range(len(x_size)):
        if x_size[i] <=window_size[i]:
            use_window_size[i] = x_size[i]
            use_shift_size[i] = 0

    return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
    """
    Windowed multi-head self-attention (W-MSA / SW-MSA) with relative position bias.
    This replaces DINO's global `Attention` (vision_transformer.py) -- instead of every token attending to every other token in the image,
    each token only attends to the other tokens inside its own local `window_size` x `window_size` window,
    which is what makes Swin linear (not quadratic) in image size and what produces the hierarchical feature maps in the first place
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        """
        """

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self._relative_pos_index_cache = {}

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_relative_pos_index(self, window_size, device):
        """
        Relative-position index for the window size actually in use this call.
        Cached per (Wh, Ww) since it's pure index arithmetic --
        only the gather into `relative_position_bias_table` needs gradients
        """

        key = (window_size[0], window_size[1], device)
        if key not in self._relative_pos_index_cache:
            Wh, Ww = window_size
            Wh_max, Ww_max = self.window_size
            coords_h = torch.arange(Wh, device=device)
            coords_w = torch.arange(Ww, device=device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            # 
            relative_coords[:, :, 0] += Wh_max - 1
            relative_coords[:, :, 1] += Ww_max - 1
            relative_coords[:, :, 0] *= 2 * Ww_max - 1
            self._relative_pos_index_cache[key] = relative_coords.sum(-1)
        
        return self._relative_pos_index_cache[key]
    
    def forward(self, x, mask=None, window_size=None):
        """
        x: (num_windows*B, N, C), N = window_size[0]*window_size[1]
        mask: (num_windows, N, N) or None -- shifted-window attention mask
        window_size: the (Wh, Ww) actually used this call. 
        Defaults to the max configured window_size when not given (the common case)
        """

        B_, N, C = x.shape

        if window_size is None:
            window_size = self.window_size

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_pos_index = self._get_relative_pos_index(window_size, x.device)
        relative_position_bias = self.relative_position_bias_table[relative_pos_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PatchMerging(nn.Module):
    """
    Downsamples resolution by 2x, doubles channel dim. This is what produces the hierarchical pyramid
    (each stage operates at half the resolution / 2x the channels of the previous one)
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    """
    Image to non-overlapping patch embedding via strided conv, same idea as ViT's PatchEmbed but
    typically with a smaller patch_size (4 instead of 16) since Swin relies on later PatchMerging stages
    -- rather than a single patchify step -- to reach ViT-comparable receptive fields
    """
    
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.shape
        
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size))
        
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size - H % self.patch_size))
             
        x = self.proj(x)
        Wh, Ww = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        return x, Wh, Ww

class SwinTransformerBlock(nn.Module):
    """
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    
    def get_attn_mask(self, H, W, window_size, shift_size, device):
        if shift_size == 0:
            return None
        
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.00)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # adapt window/shift size to this crop's resolution, and pad up to
        # a multiple of the (possibly-adapted) window size
        window_size, shift_size = get_window_size((H, W), (self.window_size, self.window_size), (self.shift_size, self.shift_size))
        
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if shift_size[0] > 0 or shift_size[1] > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = self.get_attn_mask(Hp, Wp, window_size[0], shift_size[0], x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size[0])
        x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)

        attn_windows = self.attn(x_windows, mask=attn_mask, window_size=window_size)
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], C)

        shifted_x = window_reverse(attn_windows, window_size[0], Hp, Wp)

        if shift_size[0] > 0 or shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """
    One Swin stage: a stack of W-MSA/SW-MSA block pairs at a fixed resolution/channel-width,
    optionally followed by a PatchMerging downsample into the next stage
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
    
    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        
        x_out = x
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_out, H, W, x_down, Wh, Ww
        else:
            return x_out, H, W, x, H, W


class SwinTransformer(nn.Module):
    """
    Swin Transformer backbone, DINO-compatible

    Drop-in replacement for `vision_transformer.VisionTransformer`:
    exposes the same `.forward(x) -> (B, C)` interface so it works unmodified inside the existing `MultiCropWrapper` for standard (single-vector) DINO. 
    Additionally exposes `.forward_features(x) -> [ (B, C_1), (B, C_2), ... ]` returning one globally-pooled feature vector *per stage*
    for the hierarchical / multi-scale DINO variant
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=0, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoints=False, use_dense_prediction=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # [96, 192, 384, 768] for swin_tiny
        self.embed_dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # 
        self.num_features = self.embed_dims[-1]

        # 
        self.use_dense_prediction = use_dense_prediction

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers -1 else None,
                use_checkpoint=use_checkpoints)
            self.layers.append(layer)
        
        if self.use_dense_prediction:
            # 
            self.stage_norms = nn.ModuleList([norm_layer(d) for d in self.embed_dims])
            self.norm = self.stage_norms[-1]
        else:
            # 
            self.norm = norm_layer(self.num_features)
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_all_stages(self, x):
        """
        Runs the full hierarchy and returns a list of pooled (B, C_i)vectors, one per stage.
        Requires use_dense_prediction=True at construction (that's what allocates the per-stage norms this needs)
        """

        assert self.use_dense_prediction, "forward_all_stages() requires the model to be built with use_dense_prediction=True"

        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        pooled_per_stage = []
        for i, layer in enumerate(self.layers):
            x_out, H_out, W_out, x, H, W = layer(x, H, W)
            x_out = self.stage_norms[i](x_out)
            pooled_per_stage.append(x_out.mean(dim=1))

        return pooled_per_stage
    
    def forward(self, x):
        """
        Standard single-vector interface (matches VisionTransformer.forward):
        runs the backbone once and returns only the final stage's pooled feature, (B, C_last).
        Deliberately does NOT route through forward_all_stages -- it must not touch any per-stage norm other than the final one,
        or it would create DDP-unused parameters whenever use_dense_prediction=False (see __init__)
        """

        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x_out, H_out, W_out, x, H, W = layer(x, H, W)
        
        x_out = self.norm(x_out)

        return x_out.mean(dim=1)

    def forward_features(self, x, return_all_stages=False):
        """
        Convenience entry point used by MultiCropWrapper
        """

        if return_all_stages:
            return self.forward_all_stages(x)
        
        return self.forward(x)

def load_pretrained_swin(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu'
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)

    msg = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print("Loaded pretrained Swin-Tiny weights.")

    for key in msg.missing_keys:
        print("Key is missing: ", key)

    for key in msg.unexpected_keys:
        print("Key is unexpected: ", key)

    return model

def swin_tiny(patch_size=4, in_chans=3, window_size=7, pretrained=False, pretrained_path=None, **kwargs):
    model = SwinTransformer(
        patch_size=patch_size, in_chans=in_chans, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=window_size, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained:
        if pretrained_path is None:
            raise ValueError("pretrained_path must be provided when pretrained=True")

        load_pretrained_swin(
            model,
            pretrained_path,
        )


    return model