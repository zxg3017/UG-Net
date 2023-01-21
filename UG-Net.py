import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from functools import partial
import segmentation_models_pytorch as smp

nonlinearity = partial(F.relu, inplace=True)

BN_EPS = 1e-4  # 1e-4  #1e-5


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False,
                 BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels // num_groups == 0:
                num_groups = 1
            self.gn = nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm = BatchNorm
        if is_relu is False: self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class DACblock137(nn.Module):
    def __init__(self, channel):
        super(DACblock137, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class CACblock_with_inception(nn.Module): # 1X1,3X3,5X5
    def __init__(self, channel):
        super(CACblock_with_inception, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        # self.conv7x7 = nn.Conv2d(channel, channel, kernel_size=7, dilation=1, padding=3)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        # dilate4_out = nonlinearity(self.conv7x7(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class M_Decoder_my_10(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv=False, bn=False,
                 BatchNorm=False, num_groups=32):
        super(M_Decoder_my_10, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation,
                             stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                             num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x):
        x = self.decode(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Coarse_SN_Efficient_b3_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(136, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


class Coarse_SN_Efficient_b3(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(136, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # decoder layer1
        up_conv1 = self.decoder4(e4)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)


class Coarse_SN_Efficient_b3_DAC_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        # feature_cat0 = torch.cat([e4, csv_p], 1)

        e4_cent = self.dblock(e4)
        feature_cat0 = self.cov_feature(e4_cent)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)


class CLS_Efficient_b3_DAC_CAC_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(CLS_Efficient_b3_DAC_CAC_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.CAC = CACblock_with_inception(136)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)
        # classfication layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512, out_ch)
    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        e4_cent = self.dblock(e4)
        e4_cent = self.CAC(e4_cent)
        feature_cat0 = self.cov_feature(e4_cent)

        # classfication layer
        cls_avg = self.avgpool(feature_cat0)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out

class CLS_Efficient_b3_DAC_CAC_input_cat_CAM(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(CLS_Efficient_b3_DAC_CAC_input_cat_CAM, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.CAC = CACblock_with_inception(512)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)
        # classfication layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512, out_ch)
    def forward(self, x,cam):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        feature_cat0 = torch.cat([e4, csv_p], 1)
        feature_cat0 = self.cov_feature(feature_cat0)

        e4_cent = self.dblock(feature_cat0)
        e4_cent = self.CAC(e4_cent)

        # classfication layer
        cls_avg = self.avgpool(e4_cent)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out

class Coarse_SN_Efficient_b3_DAC_CAC_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_CAC_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.CAC = CACblock_with_inception(136)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        # feature_cat0 = torch.cat([e4, csv_p], 1)

        e4_cent = self.dblock(e4)
        e4_cent = self.CAC(e4_cent)
        feature_cat0 = self.cov_feature(e4_cent)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)

class Coarse_SN_Efficient_b3_DAC_CAC_input_cat_CAM(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_CAC_input_cat_CAM, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(4, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.CAC = CACblock_with_inception(136)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(137, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x,cam):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]



        e4_cent = self.dblock(e4)
        e4_cent = self.CAC(e4_cent)
        feature_cat0 = self.cov_feature(e4_cent)

        csv_p = self.p_csv(cam)
        csv_cov = self.conv_csv_x(csv_p)
        feature_cat0 = torch.cat([e4, csv_p], 1)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)

class Coarse_SN_Efficient_b3_DAC_CAC_CAM(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_CAC_CAM, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.CAC = CACblock_with_inception(136)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(137, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x,cam):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]



        e4_cent = self.dblock(e4)
        e4_cent = self.CAC(e4_cent)
        feature_cat0 = self.cov_feature(e4_cent)

        csv_p = self.p_csv(cam)
        csv_cov = self.conv_csv_x(csv_p)
        feature_cat0 = torch.cat([e4, csv_p], 1)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)

class Coarse_SN_Efficient_b3_DAC_CAC_No_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_CAC_No_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.CAC = CACblock_with_inception(136)

        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,4,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        # feature_cat0 = torch.cat([e4, csv_p], 1)

        e4_cent = self.dblock(e4)
        e4_cent = self.CAC(e4_cent)
        feature_cat0 = self.cov_feature(e4_cent)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)

class Coarse_SN_Efficient_b3_DAC_No_input_cat(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_DAC_No_input_cat, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(136, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        # feature_cat0 = torch.cat([e4, csv_p], 1)

        e4_cent = self.dblock(e4)
        feature_cat0 = self.cov_feature(e4_cent)

        # decoder layer1
        up_conv1 = self.decoder4(feature_cat0)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)


class Coarse_SN_Efficient_b3_double_v137(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(Coarse_SN_Efficient_b3_double_v137, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x, cam):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        feature_cat0 = torch.cat([e4, csv_p], 1)
        feature_cat0 = self.cov_feature(feature_cat0)

        e4_cent = self.dblock(feature_cat0)

        # decoder layer1
        up_conv1 = self.decoder4(e4_cent)
        conv_e3 = self.cov_e3(e3)
        up_conv1_add = up_conv1 + conv_e3
        # decoder layer2
        up_conv2 = self.decoder3(up_conv1_add)  # [128,128,128]
        conv_e2 = self.cov_e2(e2)
        up_conv2_add = up_conv2 + conv_e2
        # decoder layer3
        up_conv3 = self.decoder2(up_conv2_add)
        conv_e1 = self.cov_e1(e1)
        up_conv3_add = up_conv3 + conv_e1
        # decoder layer4
        up_conv4 = self.decoder1(up_conv3_add)
        conv_e1 = self.cov_e0(e0)
        up_conv4_add = conv_e1 + up_conv4
        out = self.finalconv3(up_conv4_add)

        return F.sigmoid(out)


class CN_Efficient_b3_double_v137(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(CN_Efficient_b3_double_v137, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # classfication layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512, out_ch)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x, cam):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        csv_p = self.p_csv(cam)
        # csv_cov = self.conv_csv_x(csv_p)
        feature_cat0 = torch.cat([e4, csv_p], 1)
        feature_cat0 = self.cov_feature(feature_cat0)

        e4_cent = self.dblock(feature_cat0)

        # classfication layer
        cls_avg = self.avgpool(e4_cent)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out


class CN_Efficient_b3(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(CN_Efficient_b3, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=4, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # classfication layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(136, out_ch)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # classfication layer
        cls_avg = self.avgpool(e4)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out


class CN_Efficient_b3_without_SN(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(CN_Efficient_b3_without_SN, self).__init__()

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')
        filters = [64, 128, 256, 512]
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 136, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(137, 512, kernel_size=1, padding=0, stride=1)
        self.cov_e3 = nn.Conv2d(48, 256, kernel_size=1, padding=0, stride=1)
        self.cov_e2 = nn.Conv2d(32, 128, kernel_size=1, padding=0, stride=1)
        self.cov_e1 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)
        self.cov_e0 = nn.Conv2d(3, 32, kernel_size=1, padding=0, stride=1)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.my_CACV137 = DACblock137(256)  # 空洞卷积

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # classfication layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(136, out_ch)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)  # 解码部分

        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # classfication layer
        cls_avg = self.avgpool(e4)
        cls_flat = torch.flatten(cls_avg, 1)
        cls_out = self.fc(cls_flat)

        return cls_out


# if __name__ == '__main__':
#     from torchstat import stat
#
#     a = torch.rand((2, 3, 512, 512))
#     model = Coarse_SN_Efficient_b3_double_v137(3,2,bn=True, BatchNorm=False)
#     # model = H_Net_Efficient(2,3)
#
#     print(model(a))

class H_Net_Efficient(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b0', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(112)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(368, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(296, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(152, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(112, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) / 2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out), F.sigmoid(ave_out)


class H_Net_Efficient_b1(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b1, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b1', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(112)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(368, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(296, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(152, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(112, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) / 2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out), F.sigmoid(ave_out)


class H_Net_Efficient_b2(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b2', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(120)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(376, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(152, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(120, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class H_Net_Efficient_b2(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b2', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(120)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(376, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(152, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(120, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class H_Net_Efficient_b2_double_v137(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b2_double_v137, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b2', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(120)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(376, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(152, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_update = DACblock137(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(120, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC_update(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class H_Net_Efficient_b3(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b3, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(136, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.CAC(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class H_Net_Efficient_b3_double_v137(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b3_double_v137, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.my_CACV137 = DACblock137(256)  # 空洞卷积
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(136, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.my_CACV137(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class H_Net_Efficient_b4_double_v137(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_Efficient_b4_double_v137, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        ## up-sampping
        self.conv_up1 = M_Conv(112, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv_up2 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        ## efficient_net_encoders
        self.conv2_rx = M_Conv(3, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        self.eff = smp.encoders.get_encoder(name='efficientnet-b4', in_channels=3, depth=5, weights='imagenet')

        # ce_net encoder part

        filters = [64, 128, 256, 512]

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(136)  # 空洞卷积
        # self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        # self.e4_up = nn.ConvTranspose2d(112, 256, 3, stride=1,padding=1)
        self.e4_up = nn.ConvTranspose2d(392, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(304, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(160, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(104, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(67, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.my_CACV137 = DACblock137(256)  # 空洞卷积
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.e3_conv4 = M_Conv(40, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e2_conv4 = M_Conv(24, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e1_conv4 = M_Conv(32, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.e0_conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(67, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d4_conv = nn.Conv2d(136, 256, kernel_size=1, padding=0, stride=1, bias=True)
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)

        self.d3_down_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True)

        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(256, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x)  # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1)  # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3)  # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)
        rx = x
        ## efficient_net_encoders #(3, 32, 24, 40, 112, 320) (3, 5, 9, 16)
        rx1 = self.conv2_rx(rx)  # [2,32,512,512]

        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,32,256,256]
        e2 = eff_model[2]  # [2,24,128,128]
        e3 = eff_model[3]  # [2,40,64,64]
        e4 = eff_model[4]  # [2,112,32,32]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4_cent = self.dblock(e4)
        CAC_out = self.my_CACV137(out4)
        cent_cat = torch.cat([e4_cent, CAC_out], dim=1)  # [296,64,64]
        # CAC_out = self.conv_up1(e4) + CAC_out
        # the center part
        e4_up = self.e4_up(cent_cat)  # [256,64,64]

        # e3_out = self.e3_conv4(e3)  # [256,64,64]
        r1_cat = torch.cat([e3, e4_up], dim=1)  # [296,64,64]
        up_out = self.rc_up1(r1_cat)  # [256,128,128]
        up5 = self.up5(up_out)  # [128,128,128]

        # e2_out = self.e2_conv4(e2)
        r2_cat = torch.cat([e2, up5], dim=1)  # [152,128,128]
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        # up_out3 = self.rc_up4(r4_cat)
        # up8 = self.up8(up_out3)
        M_Net_out = self.side_8(r4_cat)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        e4_up_conv = self.d4_conv(e4_cent)
        e4_up_conv_out = e4_up_conv + out4
        d4 = self.decoder4(e4_up_conv_out)  # [256,64,64]
        # d4 = self.decoder4(e4) + self.d4_conv(out4) # [256,32,32]
        d3 = self.decoder3(d4)
        d3_down = self.d3_down_conv(d3) + self.d3_conv(out3)
        d2 = self.decoder2(d3_down) + self.d2_conv(out2)  # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1)  # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        # ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out), F.sigmoid(M_Net_out)  # ,F.sigmoid(ave_out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock134(nn.Module):
    def __init__(self, channel):
        super(DACblock134, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock137(nn.Module):
    def __init__(self, channel):
        super(DACblock137, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock139(nn.Module):
    def __init__(self, channel):
        super(DACblock139, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=9, padding=9)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class M_Net(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=512)
        # self.attentionblock6 = GridAttentionBlock(in_channels=256)
        # self.attentionblock7 = GridAttentionBlock(in_channels=128)
        # self.attentionblock8 = GridAttentionBlock(in_channels=64)

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        cet_out = self.center(out)

        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)

        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3, up_out], dim=1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2, up_out], dim=1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1, up_out], dim=1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5 + side_6 + side_7 + side_8) / 4
        return [ave_out, side_5, side_6, side_7, side_8]


class CACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(CACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.conv7x7 = nn.Conv2d(channel, channel, kernel_size=7, dilation=1, padding=3)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = nonlinearity(self.conv7x7(self.conv1x1(x)))
        dilate5_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class CACblock_with_inception(nn.Module):  # 1X1,3X3,5X5
    def __init__(self, channel):
        super(CACblock_with_inception, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        # self.conv7x7 = nn.Conv2d(channel, channel, kernel_size=7, dilation=1, padding=3)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        # dilate4_out = nonlinearity(self.conv7x7(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False,
                 BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels // num_groups == 0:
                num_groups = 1
            self.gn = nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm = BatchNorm
        if is_relu is False: self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackEncoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1,
                         groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1,
                         groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False,
                 num_groups=32):
        super(StackDecoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding,
                         dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1,
                         groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1,
                         groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear')
        # y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False,
                 BatchNorm=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv, pool
        else:
            return conv


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False,
                 BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv


class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv=False, bn=False,
                 BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation,
                             stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                             num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        out = F.upsample(x, size=(H, W), mode='bilinear')
        out = torch.cat([x_big, out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


class M_Decoder_my_10(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv=False, bn=False,
                 BatchNorm=False, num_groups=32):
        super(M_Decoder_my_10, self).__init__()
        padding = (dilation * kernel_size - 1) // 2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation,
                             stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding,
                             dilation=dilation, stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm,
                             num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                         stride=1, groups=1, is_bn=bn, BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x):
        x = self.decode(x)
        return x

# if __name__ == '__main__':
#     from torchstat import stat
#
#     a = torch.rand((2, 3, 512, 512))
#     model = H_Net_Efficient_b4_double_v137(3,2,bn=True, BatchNorm=False)
#     # model = H_Net_Efficient(2,3)
#
#     print(model(a))
