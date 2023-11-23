import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from backbone.Res2Net_v1b import res2net50_v1b_26w_4s

class Net(nn.Module):
    def __init__(self, pretrained=None):
        super(Net, self).__init__()
        self.inc = double_conv(3,64)
        resnet50 =  res2net50_v1b_26w_4s(pretrained=True)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        
        self.epb = EPB()
        self.bpb = BPB()
        self.up = up(64,64,4)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_out_edge = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_out_loc = nn.Conv2d(64, 1, 3, padding=1)
        
        self.pgb3 = PGB(in_ch=512, out_ch=64)
        self.pgb4 = PGB(in_ch=1024, out_ch=64)
        self.pgb5 = PGB(in_ch=2048, out_ch=64)
        
        self.fab = nn.ModuleList([
                FAB(),
                FAB()
                ]) 
        self.outconv5 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv = nn.Conv2d(64, 1, 3, padding=1)
       

    def forward(self, x):
        x1 = self.inc(x)
        ct_stage1 = self.layer0(x)  # [-1, 64, h/2, w/2]
        ct_stage2 = self.layer1(ct_stage1)  # [-1, 256, h/4, w/4]
        ct_stage3 = self.layer2(ct_stage2)  # [-1, 512, h/8, w/8]
        ct_stage4 = self.layer3(ct_stage3)  # [-1, 1024, h/16, w/16]
        ct_stage5 = self.layer4(ct_stage4)  # [-1, 2048, h/32, w/32]
        
        ##PaD
        out_edge_fea = self.epb(ct_stage1, ct_stage2)
        out_location_fea = self.bpb(ct_stage3, ct_stage4, ct_stage5)
        
        out_edge = self.conv_out_edge(out_edge_fea)
        out_location1 = self.up(out_location_fea)* (out_edge_fea + self.up(out_location_fea))
        
        out_location = self.conv_out_loc(out_location1)

        ##IaD
        xf5 = self.pgb5(out_location, ct_stage5, kz=16)
        xf4 = self.pgb4(out_location, ct_stage4, kz=8)
        xf3 = self.pgb3(out_location, ct_stage3, kz=4)

        x_out5 = self.fab[0](xf4,xf5)
        x_out4 = self.fab[1](xf3,x_out5)
        x_out = self.up2(self.outconv4(x_out4)) + out_location

        return out_edge, out_location, x_out
#------->  CBR + PGB <---------#
class PGB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(PGB, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch,momentum=0.99),
        nn.ReLU(inplace=True),
        ) 
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2, kz):
        x2 = self.conv(x2)
        x = self.sigmoid(F.avg_pool2d(x1,kernel_size=(kz,kz),stride=kz))*x2+x2
        return x
#------->  FAB <---------#    
class FAB(nn.Module):
    def __init__(self, in_planes=64, out_planes=64):
        super(FAB, self).__init__()  
        self.conv = nn.Conv2d(in_planes, out_planes, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.catconv = double_conv(in_planes,out_planes)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc1 = nn.Conv2d(in_planes , in_planes // 16 , 1 , bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//16 , in_planes ,1, bias = False)
        self.saconv = nn.Sequential(
        nn.Conv2d(out_planes, 1, 3, padding=1),
        nn.BatchNorm2d(1,momentum=0.99),
        nn.ReLU(inplace=True),
        nn.Conv2d(1, 1, 3, padding=1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x4, x5):        
        x5 = self.up(x5)
        out = self.catconv(x4+x5)
        ca_out = self.fc2(self.relu1(self.fc1(self.avg_pool(out))))
        ca_out = self.softmax(ca_out)
        sa_out = self.saconv(out)
        sa_out = self.softmax(sa_out)
        
        out1 = x4 * ca_out * sa_out
        out2 = x5 * ca_out * sa_out     
        out_all = self.conv(out1+out2)
        return out_all
        
class up(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2, bilinear=True):
        super(up, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)    
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x         
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch,momentum=0.99),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch,momentum=0.99),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
#------->  EPB <---------#
class EPB(nn.Module):
    def __init__(self):
        super(EPB, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.up1 = up(in_ch=256, out_ch=256)
        self.up2 = up(in_ch=256, out_ch=256)
        self.dconv1 = double_conv(256+64, 64)
        self.dconv2 = double_conv(256+64, 256)
        self.conv = nn.Conv2d(256+64, 64, 3, padding=1)

    def forward(self, x1, x2):
        x11 = self.conv1(x1)
        x12 = self.conv12(F.avg_pool2d(x1,kernel_size=(2,2),stride=2))
        x21 = self.up1(x2)
        x22 = self.conv2(x2)
        x1_out = self.dconv1(torch.cat([x11, x21], dim=1))
        x2_out = self.dconv2(torch.cat([x12, x22], dim=1))
        x2_out = self.up2(x2_out)
        out = self.conv(torch.cat([x1_out, x2_out], dim=1))
        return out   
#------->  BPB <---------#
class BPB(nn.Module):
    def __init__(self):
        super(BPB, self).__init__()
        self.mfe5 = ASPP(2048, [1,3,5])
        self.mfe4 = ASPP(1024, [1,3,5])
        self.mfe3 = ASPP(512, [1,3,5])
        self.up5 = up(in_ch=256, out_ch=256, scale_factor=4)
        self.up4 = up(in_ch=256, out_ch=256, scale_factor=2)
        self.conv1 = double_conv(256, 64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x3, x4, x5):
        dx5 = self.mfe5(x5)
        dx4 = self.mfe4(x4)
        dx3 = self.mfe3(x3)
       
        cat345 = self.up5(dx5)+self.up4(dx4)+dx3
        out = self.conv2(self.conv1(cat345))
        return out
#------------------>ASPP <---------------------#
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))


        self.convs = nn.ModuleList(modules)
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)   