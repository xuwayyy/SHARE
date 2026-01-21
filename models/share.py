import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block import SE_block



class Share(nn.Module):
    def __init__(self, in_channel, physics, window_size=8, layers=3, channel_dim=128):
        super(Share, self).__init__()
        self.res_block = UNet3D(in_channels=1, out_channels=1,
                                window_size=window_size, layers=layers, channels_out=in_channel, channel_dim=channel_dim)
        self.physics = physics

    def forward(self, x):
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
    def __init__(self, in_channels=1, out_channels=1, base_channels=1, layers=4, channel_dim=32, window_size=6, channels_out=31):
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
                         down_rank=4, memory_blocks=256,
                         qkv_bias=True))
            channels *= 2

        self.bottleneck = ConvBlock(channels, channels * 2)

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(layers):
            self.up_blocks.append(nn.ConvTranspose3d(channels * 2, channels, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(channels * 2, channels))
            channels //= 2

        self.final_conv = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)
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




class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=True, window_size=8):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            DeepPoolLayer(in_channel, out_channel, window_size) if filter else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.physics = physics

    def forward(self, x):
        return self.main(x) + x


class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, window_size=8):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8, 4, 2]
        dilation = [3, 7, 9]
        pools, convs, dynas = [], [], []
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(DualBranch(in_channels=k, dilation=dilation[j], windown_size=window_size))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x) + y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes) - 1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Tanh()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation * (kernel_size - 1) // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels, 1, 1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        # low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group,
                                                                                                c // self.group,
                                                                                                self.kernel_size ** 2,
                                                                                                h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None, :, None, None]

        out_high = (identity_input) * (self.lamb_h[None, :, None, None] + 1.)

        return out_low + out_high


class DualBranch(nn.Module):
    def __init__(self, dilation, in_channels, dim=128, heads=8, windown_size=16, downrank=4, memory_blocks=256):
        super(DualBranch, self).__init__()
        self.spectral_attn = SE_block(dim=dim, window_size=windown_size, input_resolution=windown_size, num_heads=heads,
                                      down_rank=downrank, qkv_bias=True, memory_blocks=memory_blocks)
        self.spatial_attn = dynamic_filter(inchannels=dim, kernel_size=3, dilation=dilation, group=8)

        self.first = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj = nn.Conv2d(dim, in_channels, kernel_size=1)

    def forward(self, x):
        residual_1 = x.clone()
        x = self.first(x)
        spatial = self.spatial_attn(x)
        spectral = self.spectral_attn(x)
        attn = spatial + spectral
        attn = attn + x
        attn = self.proj(attn)
        out = attn + residual_1
        return out


if __name__ == '__main__':
    resblock = ResBlock(in_channel=31, out_channel=31, filter=True, window_size=8)
    x = torch.randn(2, 31, 128, 128)
    y = resblock(x)
    print(y.shape)