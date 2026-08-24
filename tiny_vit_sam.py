# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------
# The TinyViT model is adapted from MobileSAM's variant.
# --------------------------------------------------------

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        #self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.patches_resolution = img_size
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        #self.seq = nn.Sequential(
        #    Conv2d_BN(in_chans, n // 2, 3, 2, 1),
        #    activation(),
        #    Conv2d_BN(n // 2, n, 3, 2, 1),
        #)
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 1, 1, 0),
            activation(),
            Conv2d_BN(n // 2, n, 1, 1, 0),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c=2
        if(out_dim==320 or out_dim==448 or out_dim==576):
            stride_c=1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.register_buffer('ab',
                                 self.attention_biases[:, self.attention_bias_idxs],
                                 persistent=False)

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
            (q @ k.transpose(-2, -1)) * self.scale
            +
            (self.attention_biases[:, self.attention_bias_idxs]
             if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class TinyViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 #num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()
        self.img_size=img_size
        #self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                        input_resolution=(
                            patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
                            patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))
                        ),
                        #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                        #                     patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)

        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


# ========================================
# 跨模态注意力模块 - 用于MRI引导TRUS分割
# ========================================

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    MRI特征作为Key和Value, TRUS特征作为Query
    让TRUS学习MRI的高质量特征表达
    """
    def __init__(self, in_channels=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels必须能被num_heads整除"
        
        # 线性变换层
        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.out_linear = nn.Linear(in_channels, in_channels)
        
        # 层归一化和dropout
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, trus_feat, mri_feat):
        """
        Args:
            trus_feat: TRUS特征 (B, C, H, W) -> (B, H*W, C)
            mri_feat: MRI特征 (B, C, H, W) -> (B, H*W, C)
        Returns:
            enhanced_trus_feat: 增强后的TRUS特征 (B, C, H, W)
        """
        B, C, H, W = trus_feat.shape
        
        # 转换为序列格式 (B, H*W, C)
        trus_seq = trus_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        mri_seq = mri_feat.flatten(2).transpose(1, 2)    # (B, H*W, C)
        
        # 残差连接
        residual = trus_seq
        
        # 层归一化
        trus_seq = self.norm1(trus_seq)
        mri_seq = self.norm1(mri_seq)
        
        # 线性变换
        Q = self.q_linear(trus_seq)  # Query from TRUS
        K = self.k_linear(mri_seq)   # Key from MRI
        V = self.v_linear(mri_seq)   # Value from MRI
        
        # 重塑为多头格式 (B, num_heads, H*W, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, num_heads, H*W, H*W)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, H*W, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, C)
        
        # 输出投影
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout(attn_output)
        
        # 残差连接
        enhanced_seq = residual + attn_output
        
        # 最终层归一化
        enhanced_seq = self.norm2(enhanced_seq)
        
        # 转换回特征图格式 (B, C, H, W)
        enhanced_trus_feat = enhanced_seq.transpose(1, 2).view(B, C, H, W)
        
        return enhanced_trus_feat


class MMDLoss(nn.Module):
    """
    最大均值差异损失
    用于对齐MRI和TRUS特征分布,减少模态间差异
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):
        """计算高斯核"""
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        """
        Args:
            source: TRUS特征 (B, C, H, W)
            target: MRI特征 (B, C, H, W)
        Returns:
            mmd_loss: MMD损失值
        """
        # 展平特征 (B, C*H*W)
        source = source.reshape(source.size(0), -1)
        target = target.reshape(target.size(0), -1)
        
        # 计算MMD损失
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, self.kernel_mul, self.kernel_num)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        mmd_loss = torch.mean(XX + YY - XY - YX)
        return mmd_loss


class DepthPriorMLP(nn.Module):
    """Relative-depth prior that predicts apex/base conservativeness rho."""

    def __init__(self, embed_dim=32, hidden_dim=64, use_u_shape_regularization=True):
        super().__init__()
        self.use_u_shape_regularization = bool(use_u_shape_regularization)
        self.embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.rho_head = nn.Linear(embed_dim, 1)

    def forward(self, relative_depth):
        d = relative_depth.float().view(-1).clamp(0.0, 1.0)
        basis = torch.stack(
            [
                d,
                d.pow(2),
                torch.sin(torch.pi * d),
                torch.cos(torch.pi * d),
            ],
            dim=1,
        )
        depth_embed = self.embed(basis)
        rho = torch.sigmoid(self.rho_head(depth_embed)).view(-1)
        u_depth = (4.0 * (d - 0.5).pow(2)).clamp(0.0, 1.0)
        if self.use_u_shape_regularization:
            prior_loss = F.smooth_l1_loss(rho, u_depth)
        else:
            prior_loss = rho.new_zeros(())
        return depth_embed, rho, prior_loss, u_depth


class MultiKernelRBFMMD(nn.Module):
    """Multi-kernel RBF MMD with median-distance bandwidth."""

    def __init__(self, sigma_multipliers=None, eps=1e-6):
        super().__init__()
        self.sigma_multipliers = tuple(sigma_multipliers or (0.25, 0.5, 1.0, 2.0, 4.0))
        self.eps = float(eps)

    def _pairwise_sqdist(self, x, y):
        x_norm = (x * x).sum(dim=1, keepdim=True)
        y_norm = (y * y).sum(dim=1, keepdim=True).t()
        dist = x_norm + y_norm - 2.0 * x @ y.t()
        return dist.clamp_min(0.0)

    def _kernel(self, x, y, base_sigma):
        dist = self._pairwise_sqdist(x, y)
        kernels = []
        for multiplier in self.sigma_multipliers:
            sigma = base_sigma * float(multiplier)
            gamma = 1.0 / (2.0 * sigma.clamp_min(self.eps))
            kernels.append(torch.exp(-dist * gamma))
        return torch.stack(kernels, dim=0).mean(dim=0)

    def forward(self, source, target):
        if source.size(0) <= 1 or target.size(0) <= 1:
            return (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()

        total = torch.cat([source, target], dim=0)
        all_dist = self._pairwise_sqdist(total, total).detach()
        positive = all_dist[all_dist > self.eps]
        if positive.numel() == 0:
            base_sigma = source.new_tensor(1.0)
        else:
            base_sigma = positive.median().clamp_min(self.eps)

        k_xx = self._kernel(source, source, base_sigma)
        k_yy = self._kernel(target, target, base_sigma)
        k_xy = self._kernel(source, target, base_sigma)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


class AnatomicalDescriptorProjector(nn.Module):
    """Global anatomical descriptor projection; box/boundary hooks can be added later."""

    def __init__(self, in_channels=256, project_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(in_channels, project_dim),
            nn.LayerNorm(project_dim),
        )

    def forward(self, feat, boxes=None, image_hw=None):
        del boxes, image_hw
        return self.project(self.pool(feat).flatten(1))


class DepthGatedAnatomicalMMD(nn.Module):
    """Depth-gated anatomical MMD over TRUS, center-MRI, and privileged MRI features."""

    def __init__(
        self,
        in_channels=256,
        project_dim=128,
        lambda_center=0.005,
        lambda_priv=0.005,
        min_priv_weight=0.2,
        sigma_multipliers=None,
    ):
        super().__init__()
        self.lambda_center = float(lambda_center)
        self.lambda_priv = float(lambda_priv)
        self.min_priv_weight = float(min_priv_weight)
        self.descriptor = AnatomicalDescriptorProjector(in_channels, project_dim)
        self.mmd = MultiKernelRBFMMD(sigma_multipliers=sigma_multipliers)

    def forward(self, trus_feat, center_feat, privileged_feat, rho=None, alpha=None, boxes=None, image_hw=None):
        z_trus = self.descriptor(trus_feat, boxes=boxes, image_hw=image_hw)
        z_center = self.descriptor(center_feat, boxes=boxes, image_hw=image_hw)
        z_priv = self.descriptor(privileged_feat, boxes=boxes, image_hw=image_hw)

        mmd_center = self.mmd(z_trus, z_center)
        mmd_priv = self.mmd(z_trus, z_priv)
        B = trus_feat.size(0)
        if rho is None:
            rho = trus_feat.new_zeros(B)
        else:
            rho = rho.to(device=trus_feat.device, dtype=trus_feat.dtype).view(B).clamp(0.0, 1.0)
        if alpha is None:
            gate_trust = trus_feat.new_ones(B)
        else:
            gate_trust = alpha.to(device=trus_feat.device, dtype=trus_feat.dtype).view(B, -1).mean(dim=1).clamp(0.0, 1.0)

        depth_trust = (1.0 - rho).detach()
        gate_trust = gate_trust.detach()
        min_weight = min(max(self.min_priv_weight, 0.0), 1.0)
        mmd_priv_weight = min_weight + (1.0 - min_weight) * depth_trust * gate_trust
        loss = self.lambda_center * mmd_center + self.lambda_priv * mmd_priv_weight.mean() * mmd_priv
        info = {
            "mmd_center": mmd_center,
            "mmd_priv": mmd_priv,
            "mmd_priv_weight": mmd_priv_weight,
            "weighted_mmd_total": loss,
        }
        return loss, info


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    动态调节MRI引导的权重,平衡原始TRUS特征和增强特征
    """
    def __init__(self, in_channels=256, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, original_trus, enhanced_trus, confidence=None, return_gate=False):
        """
        Args:
            original_trus: 原始TRUS特征 (B, C, H, W)
            enhanced_trus: 增强后的TRUS特征 (B, C, H, W)
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        B, C, H, W = original_trus.shape
        
        # 计算注意力权重
        original_global = self.gap(original_trus).view(B, C)  # (B, C)
        enhanced_global = self.gap(enhanced_trus).view(B, C)  # (B, C)
        
        # 拼接全局特征
        combined_global = torch.cat([original_global, enhanced_global], dim=1)  # (B, 2*C)
        
        # 生成融合权重
        fusion_weight = self.attention(combined_global)  # (B, 1)
        if confidence is not None:
            fusion_weight = fusion_weight * confidence.view(B, 1).clamp(0.0, 1.0)
        fusion_weight = fusion_weight.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        
        # 加权融合
        weighted_enhanced = enhanced_trus * fusion_weight
        weighted_original = original_trus * (1 - fusion_weight)
        
        # 特征拼接和融合
        combined_feat = torch.cat([weighted_original, weighted_enhanced], dim=1)  # (B, 2*C, H, W)
        fused_feat = self.fusion_conv(combined_feat)  # (B, C, H, W)
        
        if return_gate:
            return fused_feat, fusion_weight
        return fused_feat


class LinearFusion(nn.Module):
    """
    线性融合模块（用于消融实验）
    简单的加权相加融合，替代自适应融合
    """
    def __init__(self, in_channels=256):
        super().__init__()
        # 可学习的融合权重（初始化为0.5，表示平衡融合）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, original_trus, enhanced_trus, confidence=None, return_gate=False):
        """
        Args:
            original_trus: 原始TRUS特征 (B, C, H, W)
            enhanced_trus: 增强后的TRUS特征 (B, C, H, W)
        Returns:
            fused_feat: 融合后的特征 (B, C, H, W)
        """
        # 线性加权融合: fused = w * original + (1-w) * enhanced
        # 使用sigmoid确保权重在[0, 1]范围内
        w = torch.sigmoid(self.fusion_weight)
        if confidence is not None:
            w = w * confidence.view(-1, 1, 1, 1).clamp(0.0, 1.0)
        fused_feat = w * original_trus + (1 - w) * enhanced_trus
        if return_gate:
            if not torch.is_tensor(w) or w.dim() == 0:
                w = torch.ones(
                    original_trus.size(0), 1, 1, 1,
                    device=original_trus.device,
                    dtype=original_trus.dtype,
                ) * w
            return fused_feat, w
        return fused_feat


class SoftSliceCorrespondenceAttention(nn.Module):
    """
    Soft Slice Correspondence Attention for one TRUS slice and an MRI slice set.

    FT: [B, C, H, W]
    FM_set: [B, S, C, H, W]
    """
    def __init__(
        self,
        in_channels=256,
        num_heads=8,
        dropout=0.1,
        cross_modal_attn=None,
        fusion=None,
        use_fusion=True,
        entropy_gate=True,
        descriptor_dim=None,
        beta_temperature=0.1,
        min_confidence=0.2,
        position_prior_weight=0.0,
        position_prior_sigma=0.75,
        max_window_size=7,
        use_box_aware_pooling=False,
        boundary_ring_width=3,
        slice_utility_temperature=0.5,
        use_transition_aware_beta=False,
        jump_logit_penalty=1.0,
        use_reliability_gate=False,
        reliability_use_candidate_agreement=True,
        transition_cls_loss_weight=0.05,
        transition_reg_loss_weight=0.01,
        use_dynamic_bandwidth_beta=False,
        sigma_min=0.30,
        sigma_max=1.25,
        dynamic_bandwidth_use_gt_transition_prob=0.0,
        dynamic_bandwidth_warmup_epochs=0,
        dynamic_beta_mode="content_plus_prior",
        dynamic_prior_weight=1.0,
        use_neighbor_residual_fusion=False,
        use_gt_transition_for_beta=False,
        use_gt_transition_for_neighbor_trust=False,
        depth_embed_dim=32,
        depth_gate_enabled=False,
        depth_gate_alpha_min=0.05,
        depth_gate_alpha_max=0.60,
        beta_modulation_enabled=False,
        beta_modulation_mix_max=0.0,
        content_logit_scale_init=1.0,
        disable_transition_gate_without_supervision=True,
    ):
        super().__init__()
        self.entropy_gate = entropy_gate
        self.use_fusion = use_fusion
        self.beta_temperature = float(beta_temperature)
        self.min_confidence = float(min_confidence)
        self.position_prior_weight = float(position_prior_weight)
        self.position_prior_sigma = float(position_prior_sigma)
        self.max_window_size = int(max_window_size)
        self.use_box_aware_pooling = bool(use_box_aware_pooling)
        self.boundary_ring_width = int(boundary_ring_width)
        self.slice_utility_temperature = float(slice_utility_temperature)
        self.use_transition_aware_beta = bool(use_transition_aware_beta)
        self.jump_logit_penalty = float(jump_logit_penalty)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.reliability_use_candidate_agreement = bool(reliability_use_candidate_agreement)
        self.transition_cls_loss_weight = float(transition_cls_loss_weight)
        self.transition_reg_loss_weight = float(transition_reg_loss_weight)
        self.use_dynamic_bandwidth_beta = bool(use_dynamic_bandwidth_beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.dynamic_bandwidth_use_gt_transition_prob = float(dynamic_bandwidth_use_gt_transition_prob)
        self.dynamic_bandwidth_warmup_epochs = int(dynamic_bandwidth_warmup_epochs)
        self.dynamic_beta_mode = str(dynamic_beta_mode)
        self.dynamic_prior_weight = float(dynamic_prior_weight)
        self.use_neighbor_residual_fusion = bool(use_neighbor_residual_fusion)
        self.use_gt_transition_for_beta = bool(use_gt_transition_for_beta)
        self.use_gt_transition_for_neighbor_trust = bool(use_gt_transition_for_neighbor_trust)
        self.depth_embed_dim = int(depth_embed_dim)
        self.depth_gate_enabled = bool(depth_gate_enabled)
        self.depth_gate_alpha_min = float(depth_gate_alpha_min)
        self.depth_gate_alpha_max = float(depth_gate_alpha_max)
        self.beta_modulation_enabled = bool(beta_modulation_enabled)
        self.beta_modulation_mix_max = float(beta_modulation_mix_max)
        self.disable_transition_gate_without_supervision = bool(
            disable_transition_gate_without_supervision
        )
        self.current_epoch = 0
        if self.dynamic_beta_mode not in ("prior_only", "content_plus_prior"):
            raise ValueError(
                f"Unsupported dynamic_beta_mode={self.dynamic_beta_mode}. "
                "Use 'prior_only' or 'content_plus_prior'."
            )
        self.cross_modal_attn = cross_modal_attn or CrossModalAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fusion = fusion if fusion is not None else AdaptiveFusion(in_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        descriptor_dim = int(descriptor_dim or in_channels)
        descriptor_in_dim = in_channels * (3 if self.use_box_aware_pooling else 1)
        self.trus_descriptor = nn.Sequential(
            nn.Linear(descriptor_in_dim, descriptor_dim),
            nn.GELU(),
            nn.LayerNorm(descriptor_dim),
        )
        self.mri_descriptor = nn.Sequential(
            nn.Linear(descriptor_in_dim, descriptor_dim),
            nn.GELU(),
            nn.LayerNorm(descriptor_dim),
        )
        self.transition_head = nn.Sequential(
            nn.Linear(descriptor_in_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, 1),
        )
        self.depth_gate = nn.Sequential(
            nn.Linear(in_channels * 2 + self.depth_embed_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, 1),
            nn.Sigmoid(),
        )
        self.candidate_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )
        self.relative_position_bias = nn.Parameter(torch.zeros(self.max_window_size))
        # 问题2修复:可学习的内容logit缩放。content_logits经过/temperature后量级可达±10,
        # 远压过固定位置先验(~1)。用一个初始化较小的可学习标量把内容项缩放到与先验同量级,
        # 让固定先验真正能影响beta。以log参数化保证为正。
        self.content_logit_log_scale = nn.Parameter(
            torch.log(torch.tensor(float(max(content_logit_scale_init, 1e-4))))
        )

    @property
    def content_logit_scale(self):
        return torch.exp(self.content_logit_log_scale)

    def _position_logits(self, window_size, device, dtype, valid_mask=None):
        if window_size > self.max_window_size:
            raise ValueError(
                f"SSCA window size {window_size} exceeds max_window_size={self.max_window_size}."
            )

        center = window_size // 2
        max_center = self.max_window_size // 2
        start = max_center - center
        bias = self.relative_position_bias[start:start + window_size].to(device=device, dtype=dtype)
        bias = bias.view(1, window_size)

        if self.position_prior_weight > 0 and window_size > 1:
            offsets = torch.arange(window_size, device=device, dtype=dtype) - center
            sigma = max(self.position_prior_sigma, 1e-4)
            prior = torch.exp(-0.5 * (offsets / sigma) ** 2).view(1, window_size)
            # 问题3修复:边界切片(apex/base)窗口含padding无效切片时,先验必须只在有效位置上
            # 归一化,否则概率质量被分给随后会被mask掉的无效位置,扭曲有效切片上的先验分布。
            if valid_mask is not None:
                mask = valid_mask.to(device=device, dtype=dtype)  # (B, S)
                prior = prior * mask
                prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
            bias = bias + self.position_prior_weight * torch.log(prior.clamp_min(1e-8))

        return bias

    def _learned_position_bias(self, window_size, device, dtype):
        if window_size > self.max_window_size:
            raise ValueError(
                f"SSCA window size {window_size} exceeds max_window_size={self.max_window_size}."
            )
        center = window_size // 2
        max_center = self.max_window_size // 2
        start = max_center - center
        return self.relative_position_bias[start:start + window_size].to(
            device=device,
            dtype=dtype,
        ).view(1, window_size)

    def _dynamic_transition_value(self, transition_score, transition_target, use_target=None, allow_probabilistic_gt=True):
        t = transition_score
        if transition_target is not None:
            target = transition_target.to(device=transition_score.device, dtype=transition_score.dtype).view_as(transition_score)
            if use_target is None:
                use_target = self.use_gt_transition_for_beta
            if (
                not use_target
                and allow_probabilistic_gt
                and self.training
                and self.dynamic_bandwidth_use_gt_transition_prob > 0
                and self.current_epoch >= self.dynamic_bandwidth_warmup_epochs
            ):
                prob = min(max(self.dynamic_bandwidth_use_gt_transition_prob, 0.0), 1.0)
                use_target = bool(torch.rand((), device=transition_score.device) < prob)
            if use_target:
                t = target
        return t.clamp(0.0, 1.0)

    def _dynamic_bandwidth_logits(self, window_size, transition_value, valid_mask):
        center = window_size // 2
        offsets = torch.arange(
            window_size,
            device=transition_value.device,
            dtype=transition_value.dtype,
        ) - center
        sigma_min = max(self.sigma_min, 1e-4)
        sigma_max = max(self.sigma_max, sigma_min)
        sigma = sigma_max * (1.0 - transition_value) + sigma_min * transition_value
        prior_logits = -(offsets.view(1, window_size) ** 2) / (2.0 * sigma.view(-1, 1).pow(2).clamp_min(1e-8))
        prior_logits = prior_logits.masked_fill(~valid_mask, -1e9)
        return prior_logits, sigma, offsets

    def _flatten_boxes(self, boxes):
        if boxes is None:
            return None
        if boxes.dim() == 4:
            boxes = boxes[:, 0, 0, :]
        elif boxes.dim() == 3:
            boxes = boxes[:, 0, :]
        return boxes.float()

    def _feature_box_masks(self, boxes, feature_hw, image_hw, device, dtype):
        boxes = self._flatten_boxes(boxes)
        if boxes is None:
            return None, None

        feat_h, feat_w = feature_hw
        img_h, img_w = image_hw
        scale_x = feat_w / max(float(img_w), 1.0)
        scale_y = feat_h / max(float(img_h), 1.0)

        box_mask = torch.zeros((boxes.size(0), 1, feat_h, feat_w), device=device, dtype=dtype)
        for b in range(boxes.size(0)):
            x0, y0, x1, y1 = boxes[b].tolist()
            fx0 = int(max(min(round(x0 * scale_x), feat_w - 1), 0))
            fx1 = int(max(min(round(x1 * scale_x), feat_w - 1), 0))
            fy0 = int(max(min(round(y0 * scale_y), feat_h - 1), 0))
            fy1 = int(max(min(round(y1 * scale_y), feat_h - 1), 0))
            if fx1 < fx0:
                fx0, fx1 = fx1, fx0
            if fy1 < fy0:
                fy0, fy1 = fy1, fy0
            box_mask[b, 0, fy0:fy1 + 1, fx0:fx1 + 1] = 1.0

        if self.boundary_ring_width <= 0:
            return box_mask, None

        kernel = 2 * self.boundary_ring_width + 1
        dilated = F.max_pool2d(box_mask, kernel_size=kernel, stride=1, padding=self.boundary_ring_width)
        ring_mask = (dilated - box_mask).clamp_min(0.0)
        return box_mask, ring_mask

    def _masked_average(self, feat, mask):
        denom = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = (feat * mask).sum(dim=(2, 3)) / denom
        fallback = self.gap(feat).flatten(1)
        valid = (mask.sum(dim=(2, 3)) > 0).expand_as(pooled)
        return torch.where(valid, pooled, fallback)

    def _descriptor_input(self, feat, boxes=None, image_hw=None):
        global_desc = self.gap(feat).flatten(1)
        if not self.use_box_aware_pooling or boxes is None or image_hw is None:
            return global_desc

        box_mask, ring_mask = self._feature_box_masks(
            boxes,
            feat.shape[-2:],
            image_hw,
            feat.device,
            feat.dtype,
        )
        if box_mask is None:
            return global_desc
        box_desc = self._masked_average(feat, box_mask)
        ring_desc = self._masked_average(feat, ring_mask) if ring_mask is not None else global_desc
        return torch.cat([global_desc, box_desc, ring_desc], dim=1)

    def _compute_candidate_losses(self, candidate_logits, mask_gt, valid_mask):
        if candidate_logits is None or mask_gt is None:
            return None, None, None

        B, S = candidate_logits.shape[:2]
        if mask_gt.dim() == 3:
            mask_gt = mask_gt.unsqueeze(1)
        target = mask_gt.float()
        if target.shape[-2:] != candidate_logits.shape[-2:]:
            target = F.interpolate(target, size=candidate_logits.shape[-2:], mode="nearest")

        target = target[:, None].expand(B, S, 1, target.shape[-2], target.shape[-1]).reshape_as(candidate_logits)
        logits_flat = candidate_logits.reshape(B * S, 1, candidate_logits.shape[-2], candidate_logits.shape[-1])
        target_flat = target.reshape_as(logits_flat)
        probs_flat = torch.sigmoid(logits_flat)
        intersection = (probs_flat * target_flat).sum(dim=(1, 2, 3))
        denom = probs_flat.sum(dim=(1, 2, 3)) + target_flat.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_flat,
            target_flat,
            reduction="none",
        ).mean(dim=(1, 2, 3))
        candidate_losses = (dice_loss + bce_loss).view(B, S)
        candidate_probs = probs_flat.view(B, S, 1, logits_flat.shape[-2], logits_flat.shape[-1])
        oracle_best_candidate_dice = (1.0 - dice_loss.view(B, S)).masked_fill(~valid_mask, -1.0).max(dim=1).values
        return candidate_losses, candidate_probs, oracle_best_candidate_dice

    def _candidate_agreement(self, candidate_probs, valid_mask):
        if candidate_probs is None:
            return None

        B, S = candidate_probs.shape[:2]
        flat_probs = candidate_probs.flatten(3)
        agreements = []
        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() <= 1:
                agreements.append(candidate_probs.new_tensor(1.0))
                continue
            distances = []
            for i in range(valid_idx.numel()):
                for j in range(i + 1, valid_idx.numel()):
                    p1 = flat_probs[b, valid_idx[i], 0]
                    p2 = flat_probs[b, valid_idx[j], 0]
                    inter = (p1 * p2).sum()
                    denom = p1.sum() + p2.sum()
                    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
                    distances.append(1.0 - dice)
            if distances:
                agreements.append(1.0 - torch.stack(distances).mean())
            else:
                agreements.append(candidate_probs.new_tensor(1.0))
        return torch.stack(agreements, dim=0).clamp(0.0, 1.0)

    def _margin_confidence(self, beta, valid_mask):
        masked_beta = beta.masked_fill(~valid_mask, float("-inf"))
        topk = torch.topk(masked_beta, k=min(2, beta.size(1)), dim=1).values
        if topk.size(1) == 1:
            return topk[:, 0].clamp(0.0, 1.0)
        top1 = topk[:, 0]
        top2 = torch.where(torch.isfinite(topk[:, 1]), topk[:, 1], torch.zeros_like(topk[:, 0]))
        return (top1 - top2).clamp(0.0, 1.0)

    def _depth_modulate_beta(self, beta, rho):
        if (
            not self.beta_modulation_enabled
            or self.beta_modulation_mix_max <= 0
            or rho is None
            or beta.size(1) <= 1
        ):
            return beta

        B, S = beta.shape
        center = S // 2
        if S == 5:
            template = beta.new_tensor([0.02, 0.08, 0.80, 0.08, 0.02])
        else:
            offsets = torch.arange(S, device=beta.device, dtype=beta.dtype) - center
            template = torch.exp(-0.5 * (offsets / 0.50).pow(2))
            template = template / template.sum().clamp_min(1e-8)
        template = template.view(1, S).expand(B, S)
        mix = (self.beta_modulation_mix_max * rho.to(device=beta.device, dtype=beta.dtype).view(B, 1)).clamp(0.0, 1.0)
        beta = (1.0 - mix) * beta + mix * template
        return beta / beta.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _depth_gate_fuse(self, trus_feat, privileged_feat, depth_embed):
        B, C, _, _ = trus_feat.shape
        z_trus = self.gap(trus_feat).view(B, C)
        z_priv = self.gap(privileged_feat).view(B, C)
        gate_input = torch.cat([z_trus, z_priv, depth_embed.to(device=trus_feat.device, dtype=trus_feat.dtype)], dim=1)
        alpha_raw = self.depth_gate(gate_input)
        alpha_min = min(max(self.depth_gate_alpha_min, 0.0), 1.0)
        alpha_max = min(max(self.depth_gate_alpha_max, alpha_min), 1.0)
        alpha = alpha_min + (alpha_max - alpha_min) * alpha_raw
        alpha = alpha.view(B, 1, 1, 1)
        fused = trus_feat + alpha * (privileged_feat - trus_feat)
        return fused, alpha

    def _fuse(self, trus_feat, privileged_feat, confidence, depth_embed=None):
        if self.depth_gate_enabled and depth_embed is not None:
            return self._depth_gate_fuse(trus_feat, privileged_feat, depth_embed)

        gate_confidence = confidence if self.entropy_gate else None
        if self.use_fusion and self.fusion is not None:
            try:
                return self.fusion(
                    trus_feat,
                    privileged_feat,
                    confidence=gate_confidence,
                    return_gate=True,
                )
            except TypeError:
                fused = self.fusion(trus_feat, privileged_feat)
                gate = torch.ones(
                    trus_feat.size(0), 1, 1, 1,
                    device=trus_feat.device,
                    dtype=trus_feat.dtype,
                )
                return fused, gate

        gate = torch.ones(
            trus_feat.size(0), 1, 1, 1,
            device=trus_feat.device,
            dtype=trus_feat.dtype,
        )
        return privileged_feat, gate

    def forward(
        self,
        FT,
        FM_set,
        entropy_gate=None,
        mri_valid_mask=None,
        boxes=None,
        image_hw=None,
        mask_gt=None,
        transition_target=None,
        transition_label=None,
        transition_valid=None,
        relative_depth=None,
        depth_embed=None,
        rho=None,
    ):
        if FM_set.dim() != 5:
            raise ValueError("FM_set must have shape [B, S, C, H, W]")

        B, S, C, H, W = FM_set.shape
        if FT.shape != (B, C, H, W):
            raise ValueError(
                f"FT shape {tuple(FT.shape)} is incompatible with FM_set shape {tuple(FM_set.shape)}"
            )

        old_entropy_gate = self.entropy_gate
        if entropy_gate is not None:
            self.entropy_gate = entropy_gate

        if mri_valid_mask is None:
            mri_valid_mask = torch.ones(B, S, device=FT.device, dtype=torch.bool)
        else:
            mri_valid_mask = mri_valid_mask.to(device=FT.device, dtype=torch.bool)

        if image_hw is None:
            image_hw = (FT.shape[-2] * 4, FT.shape[-1] * 4)

        trus_desc_input = self._descriptor_input(FT, boxes=boxes, image_hw=image_hw)
        trus_desc = F.normalize(self.trus_descriptor(trus_desc_input), dim=1)
        mri_desc_input = self._descriptor_input(
            FM_set.reshape(B * S, C, H, W),
            boxes=boxes.repeat_interleave(S, dim=0) if boxes is not None else None,
            image_hw=image_hw,
        ).view(B, S, -1)
        mri_desc = self.mri_descriptor(mri_desc_input.reshape(B * S, -1)).view(B, S, -1)
        mri_desc = F.normalize(mri_desc, dim=2)
        temperature = max(self.beta_temperature, 1e-4)
        content_logits = torch.sum(trus_desc[:, None, :] * mri_desc, dim=2) / temperature
        raw_content_beta = F.softmax(
            content_logits.masked_fill(~mri_valid_mask, -1e9),
            dim=1,
        )
        # 问题2修复:把内容logits缩放到与固定位置先验同量级,使先验真正能影响beta。
        scaled_content_logits = content_logits * self.content_logit_scale

        transition_logit = self.transition_head(trus_desc_input).view(B)
        transition_score = torch.sigmoid(transition_logit)
        sigma = None
        offsets = torch.arange(S, device=FT.device, dtype=FT.dtype) - (S // 2)
        position_prior_logits = None
        if self.use_dynamic_bandwidth_beta and S > 1:
            transition_value = self._dynamic_transition_value(
                transition_score,
                transition_target,
                use_target=self.use_gt_transition_for_beta,
                allow_probabilistic_gt=True,
            )
            prior_logits, sigma, offsets = self._dynamic_bandwidth_logits(S, transition_value, mri_valid_mask)
            if self.dynamic_beta_mode == "prior_only":
                logits = prior_logits
            else:
                logits = (
                    scaled_content_logits
                    + self._learned_position_bias(S, content_logits.device, content_logits.dtype)
                    + self.dynamic_prior_weight * prior_logits
                )
        else:
            position_prior_logits = self._position_logits(
                S, content_logits.device, content_logits.dtype, valid_mask=mri_valid_mask
            )
            logits = scaled_content_logits + position_prior_logits

        if self.use_transition_aware_beta and not self.use_dynamic_bandwidth_beta and S > 1:
            offsets = torch.arange(S, device=FT.device, dtype=FT.dtype) - (S // 2)
            logits = logits - transition_score[:, None] * offsets.abs().view(1, S) * self.jump_logit_penalty

        if position_prior_logits is not None:
            position_prior_beta = F.softmax(
                position_prior_logits.expand(B, S).masked_fill(~mri_valid_mask, -1e9),
                dim=1,
            )
        else:
            position_prior_beta = None
        logits = logits.masked_fill(~mri_valid_mask, -1e9)
        beta = F.softmax(logits, dim=1)
        beta = self._depth_modulate_beta(beta, rho)

        enhanced_slices = []
        for slice_idx in range(S):
            enhanced_slices.append(self.cross_modal_attn(FT, FM_set[:, slice_idx]))
        enhanced_stack = torch.stack(enhanced_slices, dim=1)
        center_idx = S // 2
        center_feat = enhanced_stack[:, center_idx]
        if self.use_neighbor_residual_fusion and S > 1:
            residual = enhanced_stack - center_feat[:, None]
            transition_value = self._dynamic_transition_value(
                transition_score,
                transition_target,
                use_target=self.use_gt_transition_for_neighbor_trust,
                allow_probabilistic_gt=False,
            )
            neighbor_trust = (1.0 - transition_value).clamp(0.0, 1.0)
            privileged_feat = center_feat + neighbor_trust.view(B, 1, 1, 1) * torch.sum(
                residual * beta.view(B, S, 1, 1, 1),
                dim=1,
            )
        else:
            privileged_feat = torch.sum(enhanced_stack * beta.view(B, S, 1, 1, 1), dim=1)
            transition_value = self._dynamic_transition_value(
                transition_score,
                transition_target,
                use_target=self.use_gt_transition_for_neighbor_trust,
                allow_probabilistic_gt=False,
            )
            neighbor_trust = (1.0 - transition_value).clamp(0.0, 1.0)

        entropy = -(beta * torch.log(beta.clamp_min(1e-8))).sum(dim=1)
        valid_counts = mri_valid_mask.sum(dim=1).clamp_min(1).to(dtype=beta.dtype)
        entropy_norm = torch.log(valid_counts.clamp_min(2.0))
        entropy = torch.where(
            valid_counts > 1,
            entropy / entropy_norm.clamp_min(1e-8),
            torch.zeros_like(entropy),
        )
        entropy = entropy.clamp(0.0, 1.0)

        candidate_logits = self.candidate_head(enhanced_stack.reshape(B * S, C, H, W)).view(B, S, 1, H, W)
        if mask_gt is not None and candidate_logits.shape[-2:] != mask_gt.shape[-2:]:
            candidate_logits = F.interpolate(
                candidate_logits.reshape(B * S, 1, H, W),
                size=mask_gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).view(B, S, 1, mask_gt.shape[-2], mask_gt.shape[-1])

        candidate_losses, candidate_probs, oracle_best_candidate_dice = self._compute_candidate_losses(
            candidate_logits if mask_gt is not None else None,
            mask_gt,
            mri_valid_mask,
        )

        utility_target = None
        slice_utility_loss = None
        if candidate_losses is not None:
            utility_logits = (-candidate_losses / max(self.slice_utility_temperature, 1e-4)).masked_fill(
                ~mri_valid_mask,
                -1e9,
            )
            utility_target = F.softmax(utility_logits, dim=1)
            slice_utility_loss = F.kl_div(
                torch.log(beta.clamp_min(1e-8)),
                utility_target.detach(),
                reduction="batchmean",
            )

        entropy_conf = 1.0 - entropy
        margin_conf = self._margin_confidence(beta, mri_valid_mask)
        candidate_agreement = None
        if self.reliability_use_candidate_agreement:
            candidate_agreement = self._candidate_agreement(candidate_probs, mri_valid_mask)

        if self.use_reliability_gate:
            if candidate_agreement is None:
                reliability = 0.5 * entropy_conf + 0.5 * margin_conf
            else:
                reliability = 0.3 * entropy_conf + 0.3 * margin_conf + 0.4 * candidate_agreement
        else:
            reliability = entropy_conf

        # 问题5修复:当本batch没有transition监督(transition_target为None)时,transition_head
        # 完全无监督,却仍通过(1-transition_score)影响融合门控,等于往门控塞了个无约束自由标量。
        # 此时让neighbor_trust只由reliability决定,避免引入无意义噪声。
        use_transition_in_gate = not (
            self.disable_transition_gate_without_supervision and transition_target is None
        )
        if not self.use_dynamic_bandwidth_beta:
            if use_transition_in_gate:
                neighbor_trust = reliability * (1.0 - transition_score)
            else:
                neighbor_trust = reliability

        if self.min_confidence > 0:
            min_confidence = min(max(self.min_confidence, 0.0), 1.0)
            confidence = min_confidence + (1.0 - min_confidence) * neighbor_trust
        else:
            confidence = neighbor_trust
        confidence = confidence.clamp(0.0, 1.0)

        fused_feat, fusion_gate = self._fuse(FT, privileged_feat, confidence, depth_embed=depth_embed)

        transition_reg_loss = None
        transition_cls_loss = None
        transition_loss = None
        if transition_target is not None:
            transition_target = transition_target.to(device=FT.device, dtype=FT.dtype).view(B)
            if transition_valid is None:
                transition_valid = torch.ones_like(transition_target, dtype=torch.bool)
            else:
                transition_valid = transition_valid.to(device=FT.device, dtype=torch.bool).view(B)
            if transition_valid.any():
                transition_reg_loss = F.smooth_l1_loss(
                    transition_score[transition_valid],
                    transition_target[transition_valid],
                )
                if transition_label is not None:
                    label = transition_label.to(device=FT.device, dtype=FT.dtype).view(B)
                else:
                    label = (transition_target >= 0.5).to(dtype=FT.dtype)
                valid_label = label[transition_valid]
                pos = valid_label.sum()
                neg = valid_label.numel() - pos
                pos_weight = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=10.0)
                transition_cls_loss = F.binary_cross_entropy_with_logits(
                    transition_logit[transition_valid],
                    valid_label,
                    pos_weight=pos_weight.detach(),
                )
            else:
                transition_reg_loss = transition_score.new_zeros(())
                transition_cls_loss = transition_score.new_zeros(())
            transition_loss = (
                self.transition_cls_loss_weight * transition_cls_loss
                + self.transition_reg_loss_weight * transition_reg_loss
            )

        info = {
            "beta": beta,
            "final_beta_used_for_fusion": beta,
            "raw_content_beta": raw_content_beta,
            "position_prior_beta": position_prior_beta,
            "entropy": entropy,
            "confidence": confidence,
            "fusion_gate": fusion_gate,
            "transition_logit": transition_logit,
            "transition_score": transition_score,
            "candidate_logits": candidate_logits,
            "candidate_losses": candidate_losses,
            "candidate_agreement": candidate_agreement,
            "slice_utility_target": utility_target,
            "slice_utility_loss": slice_utility_loss,
            "transition_loss": transition_loss,
            "transition_cls_loss": transition_cls_loss,
            "transition_reg_loss": transition_reg_loss,
            "oracle_best_candidate_dice": oracle_best_candidate_dice,
            "neighbor_trust": neighbor_trust,
            "margin_conf": margin_conf,
            "entropy_conf": entropy_conf,
            "reliability": reliability,
            "sigma": sigma,
            "center_feat": center_feat,
            "privileged_feat": privileged_feat,
            "rho": rho,
            "relative_depth": relative_depth,
        }
        self.entropy_gate = old_entropy_gate
        return fused_feat, privileged_feat, info


class CrossModalFeatureExtractor(nn.Module):
    """
    跨模态特征提取器
    整合所有跨模态组件
    """
    def __init__(
        self,
        in_channels=256,
        num_heads=8,
        mmd_weight=0.1,
        use_adaptive_fusion=True,
        use_fusion=True,
        ssca_descriptor_dim=128,
        ssca_beta_temperature=0.1,
        ssca_min_confidence=0.2,
        ssca_position_prior_weight=0.0,
        ssca_position_prior_sigma=0.75,
        ssca_max_window_size=7,
        slice_corr_loss_weight=0.0,
        slice_corr_prior_sigma=0.75,
        ssca_use_box_aware_pooling=False,
        ssca_boundary_ring_width=3,
        slice_utility_loss_weight=0.0,
        slice_utility_temperature=0.5,
        use_transition_aware_beta=False,
        transition_loss_weight=0.0,
        jump_logit_penalty=1.0,
        use_reliability_gate=False,
        reliability_use_candidate_agreement=True,
        transition_cls_loss_weight=0.05,
        transition_reg_loss_weight=0.01,
        use_dynamic_bandwidth_beta=False,
        sigma_min=0.30,
        sigma_max=1.25,
        dynamic_bandwidth_use_gt_transition_prob=0.0,
        dynamic_bandwidth_warmup_epochs=0,
        dynamic_beta_mode="content_plus_prior",
        dynamic_prior_weight=1.0,
        use_neighbor_residual_fusion=False,
        use_gt_transition_for_beta=False,
        use_gt_transition_for_neighbor_trust=False,
        depth_prior_enabled=False,
        depth_embed_dim=32,
        depth_prior_hidden_dim=64,
        lambda_depth_prior=0.01,
        depth_prior_use_u_shape_regularization=True,
        depth_gate_enabled=False,
        depth_gate_alpha_min=0.05,
        depth_gate_alpha_max=0.60,
        use_depth_gated_mmd=False,
        dg_mmd_project_dim=128,
        dg_mmd_lambda_center=0.005,
        dg_mmd_lambda_priv=0.005,
        dg_mmd_min_priv_weight=0.2,
        beta_modulation_enabled=False,
        beta_modulation_mix_max=0.0,
        content_logit_scale_init=1.0,
        disable_transition_gate_without_supervision=True,
        mmd_use_raw_mri_target=True,
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.slice_corr_loss_weight = float(slice_corr_loss_weight)
        self.slice_corr_prior_sigma = float(slice_corr_prior_sigma)
        self.slice_utility_loss_weight = float(slice_utility_loss_weight)
        self.transition_loss_weight = float(transition_loss_weight)
        self.depth_prior_enabled = bool(depth_prior_enabled)
        self.lambda_depth_prior = float(lambda_depth_prior)
        self.depth_gate_enabled = bool(depth_gate_enabled)
        self.use_depth_gated_mmd = bool(use_depth_gated_mmd)
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_fusion = use_fusion  # 是否使用融合（False时直接返回增强特征）
        self.mmd_use_raw_mri_target = bool(mmd_use_raw_mri_target)
        
        # 跨模态注意力
        self.cross_modal_attn = CrossModalAttention(in_channels, num_heads)
        
        # 融合模块：自适应融合或线性融合（仅在use_fusion=True时使用）
        if use_fusion:
            if use_adaptive_fusion:
                self.fusion = AdaptiveFusion(in_channels)
            else:
                self.fusion = LinearFusion(in_channels)
        else:
            self.fusion = None
        
        # MMD损失
        self.ssca = SoftSliceCorrespondenceAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            cross_modal_attn=self.cross_modal_attn,
            fusion=self.fusion,
            use_fusion=use_fusion,
            entropy_gate=True,
            descriptor_dim=ssca_descriptor_dim,
            beta_temperature=ssca_beta_temperature,
            min_confidence=ssca_min_confidence,
            position_prior_weight=ssca_position_prior_weight,
            position_prior_sigma=ssca_position_prior_sigma,
            max_window_size=ssca_max_window_size,
            use_box_aware_pooling=ssca_use_box_aware_pooling,
            boundary_ring_width=ssca_boundary_ring_width,
            slice_utility_temperature=slice_utility_temperature,
            use_transition_aware_beta=use_transition_aware_beta,
            jump_logit_penalty=jump_logit_penalty,
            use_reliability_gate=use_reliability_gate,
            reliability_use_candidate_agreement=reliability_use_candidate_agreement,
            transition_cls_loss_weight=transition_cls_loss_weight,
            transition_reg_loss_weight=transition_reg_loss_weight,
            use_dynamic_bandwidth_beta=use_dynamic_bandwidth_beta,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            dynamic_bandwidth_use_gt_transition_prob=dynamic_bandwidth_use_gt_transition_prob,
            dynamic_bandwidth_warmup_epochs=dynamic_bandwidth_warmup_epochs,
            dynamic_beta_mode=dynamic_beta_mode,
            dynamic_prior_weight=dynamic_prior_weight,
            use_neighbor_residual_fusion=use_neighbor_residual_fusion,
            use_gt_transition_for_beta=use_gt_transition_for_beta,
            use_gt_transition_for_neighbor_trust=use_gt_transition_for_neighbor_trust,
            depth_embed_dim=depth_embed_dim,
            depth_gate_enabled=depth_gate_enabled,
            depth_gate_alpha_min=depth_gate_alpha_min,
            depth_gate_alpha_max=depth_gate_alpha_max,
            beta_modulation_enabled=beta_modulation_enabled,
            beta_modulation_mix_max=beta_modulation_mix_max,
            content_logit_scale_init=content_logit_scale_init,
            disable_transition_gate_without_supervision=disable_transition_gate_without_supervision,
        )
        self.last_slice_attention = None
        self.mmd_loss = MMDLoss()
        self.depth_prior = DepthPriorMLP(
            embed_dim=depth_embed_dim,
            hidden_dim=depth_prior_hidden_dim,
            use_u_shape_regularization=depth_prior_use_u_shape_regularization,
        )
        self.depth_gated_mmd = DepthGatedAnatomicalMMD(
            in_channels=in_channels,
            project_dim=dg_mmd_project_dim,
            lambda_center=dg_mmd_lambda_center,
            lambda_priv=dg_mmd_lambda_priv,
            min_priv_weight=dg_mmd_min_priv_weight,
        )

    def _slice_corr_prior_loss(self, beta):
        if self.slice_corr_loss_weight <= 0 or beta is None or beta.size(1) <= 1:
            return beta.new_zeros(())

        S = beta.size(1)
        center = S // 2
        offsets = torch.arange(S, device=beta.device, dtype=beta.dtype) - center
        sigma = max(self.slice_corr_prior_sigma, 1e-4)
        target = torch.exp(-0.5 * (offsets / sigma) ** 2)
        target = target / target.sum().clamp_min(1e-8)
        target = target.view(1, S).expand_as(beta)
        return -(target * torch.log(beta.clamp_min(1e-8))).sum(dim=1).mean()
    
    def _fuse_enhanced(self, trus_feat, enhanced_trus):
        if self.use_fusion and self.fusion is not None:
            fused_trus = self.fusion(trus_feat, enhanced_trus)
            if fused_trus.shape != trus_feat.shape:
                fused_trus = F.adaptive_avg_pool2d(fused_trus, trus_feat.shape[-2:])
        else:
            fused_trus = enhanced_trus
        return fused_trus

    def _run_average_neighbor(self, trus_feat, mri_feat_set):
        B, S, C, H, W = mri_feat_set.shape
        enhanced = []
        for slice_idx in range(S):
            enhanced.append(self.cross_modal_attn(trus_feat, mri_feat_set[:, slice_idx]))
        enhanced_stack = torch.stack(enhanced, dim=1)
        enhanced_trus = enhanced_stack.mean(dim=1)
        beta = torch.full((B, S), 1.0 / S, device=trus_feat.device, dtype=trus_feat.dtype)
        return self._fuse_enhanced(trus_feat, enhanced_trus), enhanced_trus, beta

    def _run_random_neighbor(self, trus_feat, mri_feat_set):
        B, S, C, H, W = mri_feat_set.shape
        random_idx = torch.randint(0, S, (B,), device=mri_feat_set.device)
        selected = mri_feat_set[torch.arange(B, device=mri_feat_set.device), random_idx]
        beta = F.one_hot(random_idx, num_classes=S).to(dtype=trus_feat.dtype)
        enhanced_trus = self.cross_modal_attn(trus_feat, selected)
        return self._fuse_enhanced(trus_feat, enhanced_trus), selected, beta

    def _run_original_index(self, trus_feat, mri_feat_set):
        B, S, C, H, W = mri_feat_set.shape
        center_idx = S // 2
        center = mri_feat_set[:, center_idx]
        beta = torch.zeros(B, S, device=trus_feat.device, dtype=trus_feat.dtype)
        beta[:, center_idx] = 1.0
        enhanced_trus = self.cross_modal_attn(trus_feat, center)
        return self._fuse_enhanced(trus_feat, enhanced_trus), center, beta

    def forward(
        self,
        trus_feat,
        mri_feat,
        return_loss=True,
        slice_attention_mode="index_pairing",
        boxes=None,
        image_hw=None,
        mask_gt=None,
        mri_valid_mask=None,
        transition_target=None,
        transition_label=None,
        transition_valid=None,
        relative_depth=None,
    ):
        """
        Args:
            trus_feat: TRUS特征 (B, C, H, W)
            mri_feat: MRI特征 (B, C, H, W)
            return_loss: 是否返回MMD损失
        Returns:
            enhanced_trus_feat: 增强后的TRUS特征
            mmd_loss: MMD损失 (如果return_loss=True)
        """
        # 跨模态注意力增强
        self.last_slice_attention = None
        mmd_target = mri_feat
        depth_embed = None
        rho = None
        depth_prior_loss = None
        u_depth = None
        needs_depth = (
            self.depth_prior_enabled
            or self.depth_gate_enabled
            or self.use_depth_gated_mmd
            or getattr(self.ssca, "beta_modulation_enabled", False)
        )
        if needs_depth:
            if relative_depth is None:
                relative_depth = trus_feat.new_full((trus_feat.size(0),), 0.5)
            elif torch.is_tensor(relative_depth):
                relative_depth = relative_depth.to(device=trus_feat.device, dtype=trus_feat.dtype)
            else:
                relative_depth = torch.tensor(relative_depth, device=trus_feat.device, dtype=trus_feat.dtype)
            depth_embed, rho, depth_prior_loss, u_depth = self.depth_prior(relative_depth)

        if mri_feat.dim() == 5:
            B, S, C, H, W = mri_feat.shape
            mode_aliases = {
                "original_index": "index_pairing",
                "random_neighbor_sampling": "random_neighbor",
                "average_neighbor_fusion": "average_neighbor",
            }
            mode = mode_aliases.get(slice_attention_mode, slice_attention_mode)
            if mode == "index_pairing":
                fused_trus, mmd_target, beta = self._run_original_index(trus_feat, mri_feat)
                self.last_slice_attention = {"beta": beta}
            elif mode == "random_neighbor":
                fused_trus, mmd_target, beta = self._run_random_neighbor(trus_feat, mri_feat)
                self.last_slice_attention = {"beta": beta}
            elif mode == "average_neighbor":
                fused_trus, mmd_target, beta = self._run_average_neighbor(trus_feat, mri_feat)
                self.last_slice_attention = {"beta": beta}
            elif mode == "center_only_with_same_params":
                center_feat = mri_feat[:, S // 2:S // 2 + 1]
                center_valid_mask = None
                if mri_valid_mask is not None:
                    center_valid_mask = mri_valid_mask[:, S // 2:S // 2 + 1]
                fused_trus, mmd_target, ssca_info = self.ssca(
                    trus_feat,
                    center_feat,
                    entropy_gate=False,
                    mri_valid_mask=center_valid_mask,
                    boxes=boxes,
                    image_hw=image_hw,
                    mask_gt=mask_gt,
                    transition_target=transition_target,
                    transition_label=transition_label,
                    transition_valid=transition_valid,
                    relative_depth=relative_depth,
                    depth_embed=depth_embed,
                    rho=rho,
                )
                self.last_slice_attention = dict(ssca_info)
            elif mode in ("ssca_no_entropy", "ssca_entropy"):
                fused_trus, mmd_target, ssca_info = self.ssca(
                    trus_feat,
                    mri_feat,
                    entropy_gate=(mode == "ssca_entropy"),
                    mri_valid_mask=mri_valid_mask,
                    boxes=boxes,
                    image_hw=image_hw,
                    mask_gt=mask_gt,
                    transition_target=transition_target,
                    transition_label=transition_label,
                    transition_valid=transition_valid,
                    relative_depth=relative_depth,
                    depth_embed=depth_embed,
                    rho=rho,
                )
                self.last_slice_attention = dict(ssca_info)
            else:
                raise ValueError(f"Unknown slice_attention_mode: {mode}")

            if return_loss:
                if self.use_depth_gated_mmd and self.last_slice_attention is not None:
                    center_feat = self.last_slice_attention.get("center_feat")
                    privileged_feat = self.last_slice_attention.get("privileged_feat", mmd_target)
                    fusion_gate = self.last_slice_attention.get("fusion_gate")
                    if center_feat is None:
                        center_feat = mmd_target
                    mmd, dg_mmd_info = self.depth_gated_mmd(
                        trus_feat,
                        center_feat,
                        privileged_feat,
                        rho=rho,
                        alpha=fusion_gate,
                        boxes=boxes,
                        image_hw=image_hw,
                    )
                    self.last_slice_attention.update(dg_mmd_info)
                else:
                    # 问题6修复:原实现mmd_target=privileged_feat,而privileged_feat是trus_feat经
                    # cross_modal_attn生成的,等于让trus去对齐"自己生成的增强特征",是自指,削弱了
                    # 跨模态域对齐含义。这里改为对齐【原始center MRI特征】(未经trus-attention),
                    # 才是真正的TRUS->MRI分布对齐。
                    if self.mmd_use_raw_mri_target and mri_feat.dim() == 5:
                        raw_center_mri = mri_feat[:, mri_feat.size(1) // 2]
                        mmd = self.mmd_loss(trus_feat, raw_center_mri)
                    else:
                        mmd = self.mmd_loss(trus_feat, mmd_target)
                beta = self.last_slice_attention.get("beta") if self.last_slice_attention is not None else None
                slice_corr_loss = self._slice_corr_prior_loss(beta)
                utility_loss = (
                    self.last_slice_attention.get("slice_utility_loss")
                    if self.last_slice_attention is not None else None
                )
                if utility_loss is None:
                    utility_loss = trus_feat.new_zeros(())
                transition_loss = (
                    self.last_slice_attention.get("transition_loss")
                    if self.last_slice_attention is not None else None
                )
                if transition_loss is None:
                    transition_loss = trus_feat.new_zeros(())
                if self.last_slice_attention is not None:
                    self.last_slice_attention["slice_corr_loss"] = slice_corr_loss
                    self.last_slice_attention["utility_loss_weighted"] = self.slice_utility_loss_weight * utility_loss
                    self.last_slice_attention["transition_loss_weighted"] = self.transition_loss_weight * transition_loss
                    self.last_slice_attention["rho"] = rho
                    self.last_slice_attention["u_depth"] = u_depth
                    self.last_slice_attention["relative_depth"] = relative_depth
                    self.last_slice_attention["depth_prior_loss"] = depth_prior_loss
                    self.last_slice_attention["depth_prior_loss_weighted"] = (
                        self.lambda_depth_prior * depth_prior_loss
                        if depth_prior_loss is not None and self.depth_prior_enabled
                        else trus_feat.new_zeros(())
                    )
                # 问题4修复:原来total_aux = mmd + 已加权的utility/transition/depth三项,
                # 训练循环又对total_aux整体乘mmd_loss_weight(~0.1),导致后三项被二次缩水10倍,
                # 权重语义错乱(对D系列/utility/transition消融是真bug)。
                # 现在返回值只含【纯mmd】(由训练循环乘mmd_loss_weight),
                # 其余已加权项汇总到info["aux_weighted_extra"],训练循环直接相加(不再乘mmd_weight)。
                depth_weighted = (
                    self.lambda_depth_prior * depth_prior_loss
                    if depth_prior_loss is not None and self.depth_prior_enabled
                    else trus_feat.new_zeros(())
                )
                aux_weighted_extra = (
                    self.slice_utility_loss_weight * utility_loss
                    + self.transition_loss_weight * transition_loss
                    + depth_weighted
                )
                if self.last_slice_attention is not None:
                    self.last_slice_attention["mmd"] = mmd
                    self.last_slice_attention["aux_weighted_extra"] = aux_weighted_extra
                return fused_trus, mmd, aux_weighted_extra
            return fused_trus

        enhanced_trus = self.cross_modal_attn(trus_feat, mri_feat)
        
        # 融合（自适应、线性或直接使用增强特征）
        if self.use_fusion and self.fusion is not None:
            fused_trus = self.fusion(trus_feat, enhanced_trus)
            # 确保输出维度与输入一致
            if fused_trus.shape != trus_feat.shape:
                fused_trus = F.adaptive_avg_pool2d(fused_trus, trus_feat.shape[-2:])
        else:
            # 不使用融合，直接返回增强后的特征
            fused_trus = enhanced_trus
        
        if return_loss:
            # 计算MMD损失
            mmd_loss = self.mmd_loss(trus_feat, mri_feat)
            return fused_trus, mmd_loss
        else:
            return fused_trus
