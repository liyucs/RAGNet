import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.padding import Conv2d

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num,fea_ch):
        super(up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        # partial conv
        self.cv = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch,track_running_stats=False)
        self.ac = nn.ReLU(inplace=True)
        self.pooling = torch.nn.AvgPool2d(3,stride=1,padding=1)
        self.mask_gene = MASK_GENE(fea_ch)
        if conv_num == 2:
            self.conv = double_conv(in_ch, out_ch) 
        if conv_num == 4:
            self.conv = forth_conv(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2-x3, x1], dim=1)
        mask, mask_e, mask_d = self.mask_gene(x3, x2, x1)     
        x = self.cv(x*mask) #mask = [mask_x2_e, mask_x1_d]
        mask_avg = torch.mean(self.pooling(mask),dim=1,keepdim=True)
        mask_avg[mask_avg ==0] = 1
        x = x*(1/mask_avg)
        x = self.bn(x)
        x = self.ac(x)
        
        x = self.conv(x)
        return x, mask, mask_e, mask_d
    

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class DOWN(nn.Module):
    def __init__(self):
        super(DOWN, self).__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x2 = self.downsample(x)
        x3 = self.downsample(x2)
        x4 = self.downsample(x3)
        return x,x2,x3,x4
   
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class forth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(forth_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch,track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self,encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder
        
    def forward(self,x):
        for index in range(0,4):
            x = self.encoder.features[index](x) 
        x1 = x  #torch.Size([1,64,224,224])
        for index in range(4,9):
            x = self.encoder.features[index](x)  
        x2 = x  #torch.Size([1,128,112,112])
        for index in range(9,18): 
            x = self.encoder.features[index](x)  
        x3 = x  #torch.Size([1,256,56,56])
        for index in range(18,27):
            x = self.encoder.features[index](x)
        x4 = x  #torch.Size([1,512,28,28])
        for index in range(27,36):
            x = self.encoder.features[index](x) 
        x5 = x  #torch.Size([1,512,14,14])
        return x1, x2, x3, x4, x5

class MASK_GENE(nn.Module):
    def __init__(self, io_ch):
        super(MASK_GENE, self).__init__()
        self.mask_conv0 = nn.Conv2d(io_ch*3, io_ch*3, 1)
        self.mask_ac = nn.ReLU(inplace=True)
        self.mask_conv1 = nn.Conv2d(io_ch*3, io_ch*2, 1)
        self.io_ch = io_ch
        
    def forward(self, x1, x2, x3):
        x = torch.cat([x3, x2, x1], dim=1)
        mask = F.sigmoid(self.mask_conv1(self.mask_ac(self.mask_conv0(x))))
        mask_e, mask_d = torch.split(mask,self.io_ch,dim=1)
        return mask,mask_e,mask_d
        
class GT_Model(nn.Module):
    def __init__(self, encoder_I, encoder_R):
        super(GT_Model, self).__init__()
        
        self.Encoder_I = Encoder(encoder_I)
        self.Encoder_R = Encoder(encoder_R)
        
        self.upy1 = up(1024, 256, 4, 512)  
        self.upy2 = up(512, 128, 4, 256)
        self.upy3 = up(256, 64, 2, 128)
        self.upy4 = up(128, 64, 2, 64) 
        self.outTc = outconv(64, 3)
        self.down = DOWN()
           
    def forward(self, input_I, input_R, map_thr=None, map_encoder=None):
        I_x1, I_x2, I_x3, I_x4, I_x5 = self.Encoder_I(input_I)
        R_x1, R_x2, R_x3, R_x4, R_x5 = self.Encoder_R(input_R)
             
        y4, mask4,mask4_e,mask4_d= self.upy1(I_x5, I_x4, R_x4)
        y3, mask3,mask3_e,mask3_d = self.upy2(y4, I_x3, R_x3) 
        y2, mask2,mask2_e,mask2_d = self.upy3(y3, I_x2, R_x2)
        y1, mask1,mask1_e,mask1_d = self.upy4(y2, I_x1, R_x1)
        y = self.outTc(y1)
        if not self.training:
            return F.sigmoid(y),
        
        map_1, map_2, map_3, map_4 = self.down(map_thr)
        map_1e, map_2e, map_3e, map_4e = self.down(map_encoder)
        list_map = [map_1, map_2, map_3, map_4]
        list_R = [R_x1, R_x2, R_x3, R_x4]
        list_mask = [mask1, mask2, mask3, mask4]
        list_mask_encoder = [mask1_e, mask2_e, mask3_e, mask4_e]
        list_map_encoder = [map_1e, map_2e, map_3e, map_4e]
        
        return F.sigmoid(y), list_map, list_R, list_mask, list_mask_encoder, list_map_encoder


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.layers=nn.Sequential(
            # layer_1: [batch, 256, 256, 3 * 2] => [batch, 128, 128, 64]
            Conv2d(6, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2,inplace=True),

            # layer_2: [batch, 128, 128, 64] => [batch, 64, 64, 64]
            Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            # layer_3: [batch, 64, 64, 64] => [batch, 32, 32, 64 ]
            Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            # layer_4: [batch, 32, 32, 64] => [batch, 32, 32, 1]
            Conv2d(64, 1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )

    def forward(self, blended, transmission):
        input = torch.cat([blended, transmission], dim=1)
        return self.layers(input)


















