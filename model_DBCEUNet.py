import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from .minet import miNet
from .resnet2020 import  Bottleneck, ResNetCt

from torch import Tensor


class Contrast(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Contrast, self).__init__()

        self.cvo_x = torch.Tensor([[[[-1, 1],
                                     [-2, 2]
                                     ]]])

        self.cvo_x = self.cvo_x.repeat(out_channels, in_channels, 1, 1)
        self.c_x = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
        self.c_x.weight.data = self.cvo_x

        self.cvo_y = torch.Tensor([[[[2, 1],
                                     [-2, -1]
                                     ]]])
        self.cvo_y = self.cvo_y.repeat(out_channels, in_channels, 1, 1)
        self.c_y = nn.Conv2d(in_channels, out_channels, (2, 2), stride=1, bias=False)
        self.c_y.weight.data = self.cvo_y

    def forward(self, x):
        with torch.no_grad():
            pad = nn.ZeroPad2d(padding=(0, 1, 0, 1))
            x = pad(x)
            x_x = self.c_x(x)
            x_y = self.c_y(x)
            out = (abs(x_x) + abs(x_y)) ** 0.5

            return out

class Conv_Contrast(nn.Module):
    def __init__(self,hidden_dim: int = 0):
        super().__init__()

        self.cvo = Contrast(in_channels = hidden_dim // 2,out_channels = hidden_dim // 2)
        self.conv33conv33conv11 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1)
        )
        self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=1)
        x = input_right + self.cvo(input_right)
        input_left = self.conv33conv33conv11(input_left)
        x = x.contiguous()
        output = torch.cat((input_left, x), dim=1)
        output = self.finalconv11(output).contiguous()
        return output + input

class MultiFeature(nn.Module):
    def __init__(self):
        super(MultiFeature, self).__init__()

        self.Conv_Contrast = Conv_Contrast(16)

        self.feaconv = nn.Conv2d(1, 16, 3, padding=1)  

        self.convf_11 = nn.Conv2d(16, 16, 3, padding=1)  
        self.bnf_11 = nn.BatchNorm2d(16)
        self.reluf_11 = nn.ReLU(inplace=True)
        self.convf_12 = nn.Conv2d(16, 16, 3, padding=1) 
        self.bnf_12 = nn.BatchNorm2d(16)
        self.reluf_12 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.convf_21 = nn.Conv2d(32, 32, 3, padding=1)  
        self.bnf_21 = nn.BatchNorm2d(32)
        self.reluf_21 = nn.ReLU(inplace=True)
        self.convf_22 = nn.Conv2d(32, 32, 3, padding=1)  
        self.bnf_22 = nn.BatchNorm2d(32)
        self.reluf_22 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.convf_31 = nn.Conv2d(64, 64, 3, padding=1)  
        self.bnf_31 = nn.BatchNorm2d(64)
        self.reluf_31 = nn.ReLU(inplace=True)
        self.convf_32 = nn.Conv2d(64, 64, 3, padding=1) 
        self.bnf_32 = nn.BatchNorm2d(64)
        self.reluf_32 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.convf_41 = nn.Conv2d(128, 128, 3, padding=1) 
        self.bnf_41 = nn.BatchNorm2d(128)
        self.reluf_41 = nn.ReLU(inplace=True)
        self.convf_42 = nn.Conv2d(128, 128, 3, padding=1) 
        self.bnf_42 = nn.BatchNorm2d(128)
        self.reluf_42 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # 求四个主要特征


        f1 = self.feaconv(x)
        f1 = self.Conv_Contrast(f1)

        f2 = self.reluf_11(self.bnf_11(self.convf_11(f1)))

        f2 = self.reluf_12(self.bnf_12(self.convf_12(f2)))
        f2 = torch.cat((f2, f1), 1)

        f2_ = self.pool1(f2)
        f3 = self.reluf_21(self.bnf_21(self.convf_21(f2_ )))
        f3 = self.reluf_22(self.bnf_22(self.convf_22(f3)))
        f3 = torch.cat((f3, f2_ ), 1)

        f3_ = self.pool2(f3)
        f4 = self.reluf_31(self.bnf_31(self.convf_31(f3_)))
        f4 = self.reluf_32(self.bnf_32(self.convf_32(f4)))
        f4 = torch.cat((f4, f3_), 1)

        f4_ = self.pool3(f4)
        f5 = self.reluf_41(self.bnf_41(self.convf_41(f4_)))
        f5 = self.reluf_42(self.bnf_42(self.convf_42(f5)))
        f5 = torch.cat((f5, f4_), 1)

        return f2, f3, f4, f5

class Down(nn.Module):
    def __init__(self,
                 inp_num = 1,
                 layers=[1, 2, 4, 8],
                 channels=[8, 16, 32, 64],
                 bottleneck_width=16,
                 stem_width=8,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 **kwargs
                 ):
        super(Down, self).__init__()

        # stemWidth = int(channels[0])
        stemWidth = int(8)
        self.stem = nn.Sequential(
            normLayer(1, affine=False),
            nn.Conv2d(1, stemWidth*2, kernel_size=3, stride=1, padding=1, bias=False),
            Conv_Contrast(stemWidth*2),
            normLayer(stemWidth*2),
            activate()
        )
        self.down = ResNetCt(Bottleneck, layers, inp_num=inp_num,
                       radix=2, groups=4, bottleneck_width=bottleneck_width,
                       deep_stem=True, stem_width=stem_width, avg_down=True,
                       avd=True, avd_first=False, layer_parms=channels, **kwargs)

    def forward(self, x):
        # ret = []
        x = self.stem(x)
        # ret.append(x)
        x = self.down(x)
        ret=x
        return ret

class UPCt(nn.Module):
    def __init__(self, channels=[],
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU
                 ):
        super(UPCt, self).__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0],
                      channels[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[1]),
            activate()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1],
                      channels[2],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[2]),
            activate()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2],
                      channels[3],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            normLayer(channels[3]),
            activate()
        )

    def forward(self, x):
        x1, x2, x3, x4 = x

        out = self.up1(x4)
        out = x3 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up2(out)
        out = x2 + F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.up3(out)
        out = x1 + F.interpolate(out, scale_factor=2, mode='bilinear')
        return out

class Head(nn.Module):
    def __init__(self, inpChannel, oupChannel,
                 normLayer=nn.BatchNorm2d,
                 activate=nn.ReLU,
                 # Dropout = 0.1
                 ):
        super(Head, self).__init__()
        interChannel = inpChannel // 4
        self.head = nn.Sequential(
            nn.Conv2d(inpChannel, interChannel,
                      kernel_size=3, padding=1,
                      bias=False),
            normLayer(interChannel),
            activate(),
            # nn.Dropout(),
            nn.Conv2d(interChannel, oupChannel,
                      kernel_size=1, padding=0,
                      bias=True)
        )

    def forward(self, x):
        return self.head(x)

class EDN(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256]):
        super(EDN, self).__init__()
        # it = lambda x: x

        # self.X1 = it
        # self.X2 = it
        # self.X3 = it
        self.block1 = Conv_Contrast(hidden_dim=channels[0])
        self.block2 = Conv_Contrast(hidden_dim=channels[1])
        self.block3 = Conv_Contrast(hidden_dim=channels[2])
        self.block4 = Conv_Contrast(hidden_dim=channels[3])

        self.convh1_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnh1_1 = nn.BatchNorm2d(64)
        self.reluh1_1 = nn.ReLU(inplace=True)

        self.convh2_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bnh2_1 = nn.BatchNorm2d(128)
        self.reluh2_1 = nn.ReLU(inplace=True)

        self.convh3_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bnh3_1 = nn.BatchNorm2d(256)
        self.reluh3_1 = nn.ReLU(inplace=True)

        self.convh4_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bnh4_1 = nn.BatchNorm2d(512)
        self.reluh4_1 = nn.ReLU(inplace=True)

    def forward(self, x,fea):
        x1 ,x2, x3, x4 = x
        f2, f3, f4, f5 = fea
        # print("1",x1.shape, x2.shape, x3.shape, x4.shape)
        # print("2",f2.shape, f3.shape, f4.shape, f5.shape)

        # h21 = self.reluh2_1(self.bnh2_1(self.convh2_1(torch.cat((h2, self.cvo2(h2)),1))))

        x1 = self.block1(x1)
        x1 = self.reluh1_1(self.bnh1_1(self.convh1_1(torch.cat((x1, f2),1))))
  
        x2 = self.block2(x2)
        x2 = self.reluh2_1(self.bnh2_1(self.convh2_1(torch.cat((x2, f3),1))))

        x3 = self.block3(x3)
        x3 = self.reluh3_1(self.bnh3_1(self.convh3_1(torch.cat((x3, f4),1))))

        x4 = self.block4(x4)
        x4 = self.reluh4_1(self.bnh4_1(self.convh4_1(torch.cat((x4, f5),1))))
        # print("3",x1.shape, x2.shape, x3.shape, x4.shape)
    
        return [x1, x2, x3, x4]

class DBCE_U_Net(miNet):
    def __init__(self, ):
        super(DBCE_U_Net, self).__init__()
        self.encoder = None
        self.decoder = None
        self.multiFeature = MultiFeature()
        self.down = Down(channels=[8, 16, 32, 64])
        #
        # self.up = UPCt(channels=[256,128,64,32])

        # self.down = Down(channels=[16, 32, 64, 128])
        # self.up = lambda x:x
        # self.up = UP(num_classes=num_classes, s=0.125)
        self.up = UPCt(channels=[512, 256,128,64])

        # self.head = Head(inpChannel=32, oupChannel=1)
        self.headDet = Head(inpChannel=64, oupChannel=1)

        self.headSeg = Head(inpChannel=64, oupChannel=1)
        # self.DN = DN()
        self.DN = EDN(channels=[32, 64, 128, 256])

    def funIndividual(self, x):
        fea = self.multiFeature(x)
        x1 = self.down(x) # x1是4个输出
        return x1,fea

    def funPallet(self, x):
        return x

    def funConbine(self, x):
        # ret = []
        # for i, j in zip(*x):
        #     ret.append(i+j)
        # return ret
        return x

    def funEncode(self, x):
        return x
        # return self.transformer(x)

    def funDecode(self, x,fea):
        x = self.DN(x,fea)
        x = self.up(x)
        return x

    def funOutput(self, x):
        # return self.head(x)
        # return torch.sigmoid(self.head(x))
        return torch.sigmoid(self.headSeg(x)) #torch.sigmoid(self.headDet(x)), 
    
    def forward(self, x):
        x,fea = self.funIndividual(x)
        x = self.funPallet(x)
        x = self.funConbine(x)
        x = self.funEncode(x)
        x = self.funDecode(x,fea)
        x = self.funOutput(x)
        return x

if __name__ == '__main__':
    # x = torch.rand((3,1,256,256))
    x = torch.rand((3 ,1 ,56, 776))
    x = x.to('cuda')
    model = DBCE_U_Net()
    model.to('cuda')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    out = model(x)
    print(out.shape)


# python train.py --model_names C_1_ISTDU_Net --dataset_names test0