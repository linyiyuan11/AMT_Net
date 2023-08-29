import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import Tensor
from torch import Tensor, reshape, stack
from .backbone import build_backbone
from .modules import TransformerDecoder, Transformer
from einops import rearrange
from torch.nn import Upsample


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelExchange(BaseModule):


    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p
        self.sam = SpatialAttention()
    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2


class SpatialExchange(BaseModule):


    def __init__(self, p=2):
        super(SpatialExchange, self).__init__()
        self.p = p
        self.sam = SpatialAttention()
    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2

class token_encoder(nn.Module):
    def __init__(self, in_chan = 32, token_len = 4, heads = 8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=1, heads=heads, dim_head=64, mlp_dim=64, dropout=0)
    def forward(self, x):
        b, c, h, w = x.shape
        
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)
        return x

class token_decoder(nn.Module):
    def __init__(self, in_chan = 32, size = 32, heads = 8):
        super(token_decoder, self).__init__()
        self.pos_embedding_decoder = nn.Parameter(torch.randn(1, in_chan, size, size))
        self.transformer_decoder = TransformerDecoder(dim=in_chan, depth=1, heads=heads, dim_head=True, mlp_dim=in_chan*2, dropout=0,softmax=in_chan)

    def forward(self, x, m):
        b, c, h, w = x.shape
        x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x


class context_aggregator(nn.Module):
    def __init__(self, in_chan=32, size=32):
        super(context_aggregator, self).__init__()
        self.token_encoder = token_encoder(in_chan=in_chan, token_len=4)
        self.token_decoder = token_decoder(in_chan = 32, size = size, heads = 8)
    def forward(self, feature):
        token = self.token_encoder(feature)
        out = self.token_decoder(feature, token)
        return out

class Classifier(nn.Module):
    def __init__(self, in_chan=32, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x
    
class Conv_stride(nn.Module):
    def __init__(self,in_channels=32):
        super(Conv_stride, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(32, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self,scale_factor=2):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchNorm = nn.BatchNorm2d(32, momentum=0.1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.upsample(x)
        return x

class CDNet(nn.Module):
    def __init__(self,  backbone='resnet50', output_stride=16, img_size = 512, img_chan=3, chan_num = 32, n_class =2):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)

        self.CA_s16 = context_aggregator(in_chan=chan_num, size=img_size//16)
        self.CA_s8 = context_aggregator(in_chan=chan_num, size=img_size//8)
        self.CA_s4 = context_aggregator(in_chan=chan_num, size=img_size//4)

        self.conv_s8 = nn.Conv2d(chan_num*2, chan_num, kernel_size=3, padding=1)
        self.conv_s4 = nn.Conv2d(chan_num*3, chan_num, kernel_size=3, padding=1)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode="bicubic", align_corners=True)

        self.EX = ChannelExchange()
        self.SE = SpatialExchange()

        self.liner1 = nn.Linear(1, 1)
        self.liner2 = nn.Linear(1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.classifier1 = Classifier(n_class = n_class)
        self.classifier2 = Classifier(n_class = n_class)
        self.classifier3 = Classifier(n_class = n_class)

        self.sam = SpatialAttention()

        self.conv_stride_1_1 = Conv_stride()
        self.conv_stride_1_2 = Conv_stride()
        self.conv_stride_1_3 = Conv_stride()
        self.conv_stride_2_1 = Conv_stride()
        self.conv_stride_2_2 = Conv_stride()
        self.conv_stride_2_3 = Conv_stride()

        self.upsample_1_1 = Upsample()
        self.upsample_1_2 = Upsample()
        self.upsample_1_3 = Upsample(scale_factor=4)
        self.upsample_2_1 = Upsample()
        self.upsample_2_2 = Upsample()
        self.upsample_2_3 = Upsample(scale_factor=4)
        
    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        out1_s16, out1_s8, out1_s4 = self.backbone(img1)
        out2_s16, out2_s8, out2_s4 = self.backbone(img2)
        
        out1_s16, out2_s16 = self.EX(out1_s16, out2_s16)
        out1_s8, out2_s8 = self.EX(out1_s8, out2_s8)
        out1_s4, out2_s4 = self.SE(out1_s4, out2_s4)

        out1_s16 = out1_s16*self.sam(out1_s16)
        out2_s16 = out2_s16 * self.sam(out2_s16)

        out1_s4 = out1_s4*self.sam(out1_s4)
        out2_s4 = out2_s4 * self.sam(out2_s4)

        out2_s8 = out2_s8*self.sam(out2_s8)
        out1_s8 = out1_s8 * self.sam(out1_s8)

        out1_s4_down_1 = self.conv_stride_1_1(out1_s4)
        out1_s4_down_2 = self.conv_stride_1_2(out1_s4_down_1)

        out1_s8_up = self.upsample_1_1(out1_s8)
        out1_s8_down = self.conv_stride_1_3(out1_s8)

        out1_s16_up_1 = self.upsample_1_2(out1_s16)
        out1_s16_up_2 = self.upsample_1_3(out1_s16)

        out1_s16 = out1_s16 + out1_s4_down_2 + out1_s8_down
        out1_s8 = out1_s8 + out1_s4_down_1 + out1_s16_up_1
        out1_s4 = out1_s4 + out1_s16_up_2 + out1_s8_up

        out2_s4_down_1 = self.conv_stride_2_1(out2_s4)
        out2_s4_down_2 = self.conv_stride_2_2(out2_s4_down_1)

        out2_s8_up = self.upsample_2_1(out2_s8)
        out2_s8_down = self.conv_stride_2_3(out2_s8)

        out2_s16_up_1 = self.upsample_2_2(out2_s16)
        out2_s16_up_2 = self.upsample_2_3(out2_s16)

        out2_s16 = out2_s16 + out2_s4_down_2 + out2_s8_down
        out2_s8 = out2_s8 + out2_s4_down_1 + out2_s16_up_1
        out2_s4 = out2_s4 + out2_s16_up_2 + out2_s8_up
        
        x1_s16 = self.CA_s16(out1_s16)
        x2_s16 = self.CA_s16(out2_s16)  # [8,32,32,32]

        x1_s8 = self.CA_s8(out1_s8)
        x2_s8 = self.CA_s8(out2_s8)  # [8,32,64,64]

        x1 = self.CA_s4(out1_s4)  # [8,32,128,128]
        x2 = self.CA_s4(out2_s4)

        x_s16 = x1_s16 + x2_s16
        weight_s16 = F.adaptive_avg_pool2d(x_s16, (1, 1))

        x_s8 = x1_s8 + x2_s8
        weight_s8 = F.adaptive_avg_pool2d(x_s8, (1, 1))

        x_s = x1 + x2
        weight_sx = F.adaptive_avg_pool2d(x_s, (1, 1))

        weight = weight_sx + weight_s8 + weight_s16

        weight = self.liner1(weight)
        weight = self.relu(weight)
        weight = self.liner2(weight)
        weight = self.sigmoid(weight)

        x1_s16 = weight * x1_s16
        x2_s16 = weight * x2_s16

        x1_s8 = weight * x1_s8
        x2_s8 = weight * x2_s8

        x1 = weight * x1
        x2 = weight * x2

        x16 = torch.cat([x1_s16, x2_s16], dim=1)
        x8 = torch.cat([x1_s8, x2_s8], dim=1)
        x = torch.cat([x1, x2], dim=1)

        x16 = F.interpolate(x16, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x8 = F.interpolate(x8, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x = F.interpolate(x, size=img1.shape[2:], mode='bicubic', align_corners=True)

        x = self.classifier3(x)
        x8 = self.classifier2(x8)
        x16 = self.classifier1(x16)

        return x, x8, x16

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
