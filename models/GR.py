import torch
import torch.nn as nn
import torch.nn.functional as F

class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num):
        super(up, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        if conv_num == 2:
            self.conv = double_conv(in_ch, out_ch)
        if conv_num == 4:
            self.conv = forth_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
   
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch,track_running_stats=False),
            nn.ReLU(inplace=True),
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
    
class GR_Model(nn.Module):
    def __init__(self, encoder):
        super(GR_Model, self).__init__()
        self.encoder = encoder
       
        self.upz1 = up(1024, 256, 4)  
        self.upz2 = up(512, 128, 4)
        self.upz3 = up(256, 64, 2)
        self.upz4 = up(128, 64, 2)  
        self.outzc = outconv(64, 3)        
        
    def forward(self, x):
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

        z4 = self.upz1(x5, x4)   
        z3 = self.upz2(z4, x3)   
        z2 = self.upz3(z3, x2)
        z1 = self.upz4(z2, x1)
        out_R = self.outzc(z1)
        
        return F.sigmoid(out_R)
    



















