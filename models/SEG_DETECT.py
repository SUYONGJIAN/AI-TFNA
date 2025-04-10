import torch.nn as nn
import torch


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True,
                 beforeSkip=False):
        super(Block, self).__init__()
        self.do_beforeSkip = beforeSkip
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.99)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.99))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters, eps=1e-3, momentum=0.99))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters, eps=1e-3, momentum=0.99))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1 and not self.do_beforeSkip:
            rep.append(nn.ZeroPad2d((0, 1, 0, 1)))
            rep.append(nn.MaxPool2d(3, strides, 0))

        self.maxpool = nn.MaxPool2d(3, strides, 0)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        before_skip = x
        if self.do_beforeSkip:
            x = self.pad(x)
            x = self.maxpool(x)

        x += skip
        if self.do_beforeSkip:
            return x, before_skip
        else:
            return x


class convResize(nn.Module):
    def __init__(self, size):
        super(convResize, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.resize_size = size
        self.resize = nn.Upsample(scale_factor=size, align_corners=True, mode="bilinear")

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.resize_size != 1:
            x = self.resize(x)
        return x


class Xception_FPN_Unet(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self):
        super(Xception_FPN_Unet, self).__init__()
        self.block1_conv1_padding = nn.ZeroPad2d((0, 1, 0, 1))
        self.block1_conv1 = nn.Conv2d(3, 32, 3, 2, padding=0, bias=False)
        self.block1_conv1_bn = nn.BatchNorm2d(32, eps=1e-3, momentum=0.99)
        self.relu = nn.ReLU(inplace=True)
        self.block1_conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.block1_conv2_bn = nn.BatchNorm2d(64, eps=1e-3, momentum=0.99)
        # do relu here

        self.block2 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block3 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True, beforeSkip=True)
        self.block4 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True, beforeSkip=True)

        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block13 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False, beforeSkip=True)

        self.block14_sepconv1 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.block14_sepconv1_bn = nn.BatchNorm2d(1536, eps=1e-3, momentum=0.99)

        # do relu here
        self.block14_sepconv2 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.block14_sepconv2_bn = nn.BatchNorm2d(2048, eps=1e-3, momentum=0.99)

        # UNET layers
        self.upsample = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
        self.C2_reduced = nn.Conv2d(256, 256, 1, 1, padding=0)
        self.C3_reduced = nn.Conv2d(728, 256, 1, 1, padding=0)
        self.C4_reduced = nn.Conv2d(1024, 256, 1, 1, padding=0)

        self.P2 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.P3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.P4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.P5 = nn.Conv2d(2048, 256, 1, 1, padding=0)
        # self.upconv5_2 = nn.Conv2d(256, 256, 3, 1, padding=1)

        # FPN layers
        self.P5_conv_resize = convResize(8)
        self.P4_conv_resize = convResize(4)
        self.P3_conv_resize = convResize(2)
        self.P2_conv_resize = convResize(1)

        self.aggregation_conv = nn.Conv2d(512, 256, 3, 1, 1)
        self.aggregation_bn = nn.BatchNorm2d(num_features=256, eps=1e-5, momentum=0.99)
        # activation Relu
        # upsample

        self.up4_conv1_conv = nn.Conv2d(256, 128, 3, 1, 1)
        # Relu
        # Cat
        self.up4_conv2_conv = nn.Conv2d(192, 128, 3, 1, 1)

        # Relu
        # upsample
        self.up5_conv1_conv = nn.Conv2d(128, 64, 3, 1, 1)
        # Relu
        self.up5_conv2_conv = nn.Conv2d(64, 64, 3, 1, 1)
        # ReLu
        self.mask_softmax = nn.Conv2d(64, 1, 1, 1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1_conv1_padding(x)
        x = self.block1_conv1(x)
        x = self.block1_conv1_bn(x)
        x = self.relu(x)
        x = self.block1_conv2(x)
        x = self.block1_conv2_bn(x)
        x = self.relu(x)
        x1 = x

        x = self.block2(x)
        x, x2 = self.block3(x)
        x, x3 = self.block4(x)
        x3 = self.relu(x3)

        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x, x4 = self.block13(x)
        x4 = self.relu(x4)

        x = self.block14_sepconv1(x)
        x = self.block14_sepconv1_bn(x)
        x = self.relu(x)
        x = self.block14_sepconv2(x)
        x = self.block14_sepconv2_bn(x)

        x = self.relu(x)
        x5 = x

        p5 = self.P5(x5)

        p5_upsampled = self.upsample(p5)
        p4 = p5_upsampled + self.C4_reduced(x4)
        p4 = self.P4(p4)

        p4_upsampled = self.upsample(p4)
        p3 = p4_upsampled + self.C3_reduced(x3)
        p3 = self.P3(p3)

        p3_upsampled = self.upsample(p3)
        p2 = p3_upsampled + self.C2_reduced(x2)
        p2 = self.P2(p2)

        p5 = self.P5_conv_resize(p5)
        p4 = self.P4_conv_resize(p4)
        p3 = self.P3_conv_resize(p3)
        p2 = self.P2_conv_resize(p2)

        x = torch.cat([p5, p4, p3, p2], dim=1)

        x = self.aggregation_conv(x)
        x = self.aggregation_bn(x)
        x = self.relu(x)

        x = self.upsample(x)

        x = self.up4_conv1_conv(x)

        x = self.relu(x)
        x = torch.cat((x1, x), dim=1)
        x = self.up4_conv2_conv(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.up5_conv1_conv(x)
        x = self.relu(x)
        x = self.up5_conv2_conv(x)
        x = self.relu(x)
        x = self.mask_softmax(x)
        x = self.sigmoid(x)
        # x = x[:,0,:,:]
        return x


def make_model():
    torch_model = Xception_FPN_Unet()
    return torch_model
