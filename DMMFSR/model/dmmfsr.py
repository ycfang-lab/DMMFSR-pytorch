from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_model(args, parent=False):
    return DMMFSR(args)


# define the basic component of RDB
class DB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kernel_size=3):
        super(DB_Conv, self).__init__()
        n_feats = inChannels
        rate = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(n_feats, rate, kernel_size, padding=(kernel_size - 1) // 2, stride=1),  # 如此操作，相当于大小没有变
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


# define the dense block (DB)
class DB(nn.Module):
    def __init__(self, args, n_layer):  # n_layer=8
        super(DB, self).__init__()
        n_feats = args.n_feats  # n_feats = 64
        rate = 64  # rate = 64
        kernel_size = 3  # kernel_size=3

        convs = []
        for n in range(n_layer):
            convs.append(DB_Conv(n_feats + n * rate, rate))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(n_feats + n_layer * rate, n_feats, 1, padding=0, stride=1)

    def forward(self, x):
        out = self.LFF(self.convs(x)) + x
        return out  # 64*64


# define the residual group (RG)
class RG(nn.Module):
    def __init__(self, n_feats, block, kernel_size=3, conv=common.default_conv):
        super(RG, self).__init__()
        n_resblock = block

        residual_group = []
        residual_group = [
            common.ResBlock(
                conv, n_feats, kernel_size
            ) for _ in range(n_resblock)]
        self.body = nn.Sequential(*residual_group)

    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out


class MSRB(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(MSRB, self).__init__()

        n_feats = 64
        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class DMMFSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DMMFSR, self).__init__()
        n_feats = args.n_feats  # 64
        kernel_size = 3
        block = 5  # 5
        n_layer = 8  # 8
        scale = args.scale[0]

        ## Stage I'
        # ----------------------------------------------------------
        # init keypoint feature extractor
        num_keypoint = 21
        self.key_head = [conv(num_keypoint, n_feats, kernel_size)]
        for i in range(block):
            att_name = 'm_key' + str(i + 1)
            self.__setattr__(att_name, conv(n_feats, n_feats, kernel_size))

        # init edge
        self.edge_head = [conv(3, n_feats, kernel_size)]
        for i in range(block):
            att_name = 'edge_body' + str(i + 1)
            self.__setattr__(att_name, MSRB())
        self.edge_tail = [
            nn.Conv2d(n_feats * (block + 1), n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False)]

        # init RIRN
        self.m_head = [conv(3, n_feats, kernel_size)]
        for i in range(block):
            att_name = 'm_body' + str(i + 1)
            self.__setattr__(att_name, conv(n_feats, n_feats, kernel_size))
        self.m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False)]

        # init ttc*5
        for i in range(block * 2):
            att_ttc_name = 'ttc' + str(i + 1)
            self.__setattr__(att_ttc_name, torch.nn.Parameter(torch.rand(1)))

        # -----------------------------------------------------------
        self.khead = nn.Sequential(*self.key_head)
        self.mhead = nn.Sequential(*self.m_head)
        self.mtail = nn.Sequential(*self.m_tail)
        self.ehead = nn.Sequential(*self.edge_head)
        self.etail = nn.Sequential(*self.edge_tail)
        self.etail_img = conv(n_feats, args.n_colors, kernel_size)

        ## Stage II
        self.image_feature = DB(args, n_layer)
        self.edge_feature = DB(args, n_layer)

        self.image_rg_1 = RG(n_feats, block)
        self.edge_rg_1 = RG(n_feats, block)

        self.image_rg_2 = RG(n_feats, block)
        self.edge_rg_2 = RG(n_feats, block)

        ## Stage III
        self.fusion = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.cat_rg_t = RG(n_feats, block)
        self.cat_rg_b = RG(n_feats, block)
        self.cat_rg_last = RG(n_feats, block)
        self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        # Stage I: Edge Reconstruction
        # print('Stage I')
        edge_out = []
        low_head = self.mhead(x)
        high_head = self.ehead(x)
        edge_out.append(high_head)
        low_1 = low_head + self.ttc1 * high_head
        high_1 = high_head + self.ttc2 * low_head
        edge_out.append(high_1)

        low_1 = self.m_body1(low_1)
        high_1 = self.edge_body1(high_1)
        low_2 = low_1 + self.ttc3 * high_1
        high_2 = high_1 + self.ttc4 * low_1
        edge_out.append(high_2)

        low_2 = self.m_body2(low_2)
        high_2 = self.edge_body2(high_2)
        low_3 = low_2 + self.ttc5 * high_2
        high_3 = high_2 + self.ttc6 * low_2
        edge_out.append(high_3)

        low_3 = self.m_body3(low_3)
        high_3 = self.edge_body3(high_3)
        low_4 = low_3 + self.ttc7 * high_3
        high_4 = high_3 + self.ttc8 * low_3
        edge_out.append(high_4)

        low_4 = self.m_body4(low_4)
        high_4 = self.edge_body4(high_4)
        low_5 = low_4 + self.ttc9 * high_4
        high_5 = high_4 + self.ttc10 * low_4
        edge_out.append(high_5)
        high = torch.cat(edge_out, 1)  # 2,384, 48, 48
        # ttc multi task
        low = self.mtail(low_5)
        high = self.etail(high)  # 2,64, 96, 96

        # Stage II: Feature Extraction
        # print('Stage II')
        image_feature_1 = self.image_feature(low)
        edge_feature_1 = self.edge_feature(high)

        image_feature_2 = self.image_rg_1(image_feature_1)
        edge_feature_2 = self.edge_rg_1(edge_feature_1)

        leve_1_bottom = image_feature_1 + edge_feature_1
        leve_1_bottom = self.cat_rg_b(leve_1_bottom)
        leve_2_top = image_feature_2 + edge_feature_2
        leve_2_top = self.cat_rg_t(leve_2_top)

        # Stage III: Edge Guidance Image Reconstruction
        # print('Stage III')
        cat_freature = torch.cat([leve_2_top, leve_1_bottom], 1)
        cat_fusion = self.fusion(cat_freature)
        cat_1 = self.cat_rg_last(cat_fusion)
        denoised = self.tail(cat_1)
        # print(denoised.shape)
        return self.etail_img(high), denoised


if __name__ == '__main__':
    from option import args
    args.scale = [2]
    net = DMMFSR(args)
    input = torch.rand(2, 3, 48, 48)
    net(input)
