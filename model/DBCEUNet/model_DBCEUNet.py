import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet2020 import ResNetCt, Bottleneck


class Contrast(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Contrast, self).__init__()
        # Create a contrast module with fixed convolution kernels
        self.c_x = self._create_fixed_conv(in_channels, out_channels, [[-1, 1], [-2, 2]])
        self.c_y = self._create_fixed_conv(in_channels, out_channels, [[2, 1], [-2, -1]])

    @staticmethod
    def _create_fixed_conv(in_channels, out_channels, kernel_values):
        # Create a convolution layer with fixed weights
        kernel = torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(out_channels, in_channels, 1, 1)

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=1, bias=False)
        conv.weight.data = kernel
        conv.weight.requires_grad = False  # Fix weights
        return conv

    def forward(self, x):
        # Compute contrast features
        with torch.no_grad():
            x = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0)  # Zero padding
            x_x = self.c_x(x)
            x_y = self.c_y(x)
            return torch.sqrt(torch.abs(x_x) + torch.abs(x_y))


class ConvContrast(nn.Module):
    """Convolutional contrast module"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.contrast = Contrast(hidden_dim // 2, hidden_dim // 2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=1)
        )
        self.final_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        # Split input channels and process separately
        left, right = x.chunk(2, dim=1)
        right = right + self.contrast(right)
        left = self.conv_block(left)
        output = torch.cat((left, right), dim=1)
        return self.final_conv(output) + x


class FeatureBlock(nn.Module):
    """Multi-scale feature extraction block"""
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        # Two convolutions + concatenate input + pooling
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = torch.cat([out, x], dim=1)
        pooled = self.pool(out)
        return out, pooled


class MultiFeatureExtractor(nn.Module):
    """Multi-layer feature extractor"""
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.contrast = ConvContrast(16)

        self.block1 = FeatureBlock(16, 16)   # Output channels: 32
        self.block2 = FeatureBlock(32, 32)   # Output channels: 64
        self.block3 = FeatureBlock(64, 64)   # Output channels: 128
        self.block4 = FeatureBlock(128, 128) # Output channels: 256

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.contrast(x)

        f2, p1 = self.block1(x)
        f3, p2 = self.block2(p1)
        f4, p3 = self.block3(p2)
        f5, _  = self.block4(p3)

        return f2, f3, f4, f5  # Features from each layer


class DownSampler(nn.Module):
    """Down-sampling module"""
    def __init__(self,
                 inp_num=1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 norm_layer=nn.BatchNorm2d,
                 activation=nn.ReLU,
                 **kwargs):
        super().__init__()

        self.stem = nn.Sequential(
            norm_layer(inp_num, affine=False),
            nn.Conv2d(inp_num, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(stem_width * 2),
            activation()
        )

        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                       radix=2, groups=4, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.down(x)
        return x


class UpSampler(nn.Module):
    """Up-sampling module"""
    def __init__(self, channels, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()
        self.up_blocks = nn.ModuleList([
            self._create_up_block(channels[i], channels[i + 1], norm_layer, activation)
            for i in range(len(channels) - 1)
        ])

    @staticmethod
    def _create_up_block(in_channels, out_channels, norm_layer, activation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            activation()
        )

    def forward(self, features):
        # Perform up-sampling and feature fusion layer by layer
        x = features[-1]
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            x = features[-2 - i] + F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


class EDN(nn.Module):
    """Feature fusion module"""
    def __init__(self, channels):
        super(EDN, self).__init__()

        self.blocks = nn.ModuleList([
            ConvContrast(ch) for ch in channels
        ])

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch * 2, kernel_size=3, padding=1),  
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ) for ch in channels
        ])

    def forward(self, x, features):
        outputs = []

        for i, (block, conv) in enumerate(zip(self.blocks, self.convs)):
            x_block = block(x[i])
            f = features[i]
            cat = torch.cat((x_block, f), dim=1)
            out = conv(cat)
            outputs.append(out)

        return outputs


class Head(nn.Module):
    """Output head module"""
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channels // 4),
            activation(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.head(x)


class DBCEUNet(nn.Module):
    """Main network model"""
    def __init__(self):
        super(DBCEUNet, self).__init__()
        self.multi_feature = MultiFeatureExtractor()
        self.down = DownSampler(channels=[8, 16, 32, 64])
        self.up = UpSampler(channels=[512, 256, 128, 64])
        self.edn = EDN(channels=[32, 64, 128, 256])
        self.head_seg = Head(in_channels=64, out_channels=1)

    def forward(self, x):
        features = self.multi_feature(x)
        down_features = self.down(x)
        combined_features = self.edn(down_features, features)
        up_features = self.up(combined_features)
        return torch.sigmoid(self.head_seg(up_features))


if __name__ == '__main__':
    x = torch.rand((3, 1, 56, 776)).to('cuda')
    model = DBCEUNet().to('cuda')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    out = model(x)
    print(out.shape)