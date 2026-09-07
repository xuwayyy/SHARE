import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


class SHARE(nn.Module):
    def __init__(self, in_channel, physics, window_size=8, layers=3, channel_dim=128, rank=4, memory_blocks=256):
        super(SHARE, self).__init__()
        self.res_block = UNet3D(in_channels=1, out_channels=1, # this is for 3D convolution
                                window_size=window_size, layers=layers, channels_out=in_channel,
                                channel_dim=channel_dim,
                                rank=rank, memory_blocks=memory_blocks)
        self.physics = physics

    def forward(self, x, physics=None):
        dagger = self.physics.A_adjoint(x)
        return self.res_block(dagger)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=1, layers=4, channel_dim=32,
                 window_size=6, channels_out=31, rank=4, memory_blocks=256):
        super(UNet3D, self).__init__()

        self.layers = layers

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attn = nn.ModuleList()
        if channel_dim == 16:
            attn_dim = 64
        else:
            attn_dim = 128 * (channel_dim // 32)

        channels = base_channels
        for i in range(layers):
            self.enc_blocks.append(ConvBlock(in_channels if i == 0 else channels, channels * 2))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.attn.append(
                SE_block(dim=attn_dim, window_size=window_size, input_resolution=window_size, num_heads=8,
                         down_rank=rank, memory_blocks=memory_blocks, qkv_bias=True))
            channels *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(channels, channels * 2)

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(layers):
            self.up_blocks.append(nn.ConvTranspose3d(channels * 2, channels, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(channels * 2, channels))
            channels //= 2

        # 最终输出层
        self.final_conv = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)
        # self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)

        self.first = nn.Conv2d(channels_out, channel_dim, kernel_size=3, padding=1)
        self.out = nn.Conv2d(channel_dim, channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.first(x)
        x = x.unsqueeze(1)
        enc_outputs = []
        for i in range(self.layers):
            x = self.enc_blocks[i](x)
            enc_outputs.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        for i in range(self.layers):
            x = self.up_blocks[i](x)
            x = torch.cat((enc_outputs[self.layers - i - 1], x), dim=1)  # 跳跃连接
            B, D, C, H, W = x.shape
            x = x.reshape(B, D * C, H, W)
            x = self.attn[i](x)
            x = x.reshape(B, D, C, H, W)
            x = self.dec_blocks[i](x)

        out = self.final_conv(x)
        out = out.squeeze(1)
        out = self.out(out)
        return out



def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16, memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(nn.Linear(num_feat, num_feat // squeeze_factor))
        self.upnet = nn.Sequential(nn.Linear(num_feat // squeeze_factor, num_feat), nn.Sigmoid())
        self.mb = torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
        b, n, c = x.shape
        t = x.transpose(1, 2)
        y = self.pool(t).squeeze(-1)
        low_rank_f = self.subnet(y).unsqueeze(2)
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        out = x * y2
        return out


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30, memory_blocks=128):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Linear(num_feat, num_feat // compress_ratio),
            nn.GELU(),
            nn.Linear(num_feat // compress_ratio, num_feat),
            ChannelAttention(num_feat, squeeze_factor, memory_blocks)
        )

    def forward(self, x):
        return self.cab(x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=0, qk_scale=None, memory_blocks=128, down_rank=16,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.c_attns = CAB(dim, compress_ratio=4, squeeze_factor=down_rank, memory_blocks=memory_blocks)

    def forward(self, x, mask=None):
        x3 = self.c_attns(x)
        x = self.proj(x3)
        x = self.proj_drop(x)
        return x


class SE_block(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, drop_path=0.0, memory_blocks=128, down_rank=16,
                 qkv_bias=True, qk_scale=None, drop=0., shift_size=0, attn_drop=0., act_layer=nn.GELU):
        super(SE_block, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.shift_size = shift_size
        self.attns = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                     memory_blocks=memory_blocks, down_rank=down_rank, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            H_pad = H
            W_pad = W


        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attns(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)


        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        x = x.permute(0, 3, 1, 2)
        return x