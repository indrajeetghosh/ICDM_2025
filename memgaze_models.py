import torch
import torch.nn as nn
import torchvision.models as models

# -----------------------------------
# SpatialChannelAttention Module (CDBAM)
# -----------------------------------


class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.channel_fc(x)
        x = x * ch_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sp_att = self.spatial_conv(spatial_input)

        return x * sp_att

# -----------------------------------
# TeacherNet Model
# -----------------------------------

class TeacherNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)

        # Use intermediate layers
        self.layer1 = nn.Sequential(*list(base.children())[:5])  
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Attention-enhanced blocks
        self.att4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpatialChannelAttention(512)
        )
        self.att3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpatialChannelAttention(256)
        )
        self.att2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpatialChannelAttention(128)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        f1 = self.layer1(x)  
        f2 = self.layer2(f1) 
        f3 = self.layer3(f2) 
        f4 = self.layer4(f3) 
        a4 = self.att4(f4)   
        x = self.up3(a4)     

        a3 = self.att3(f3)   
        x = torch.cat([x, a3], dim=1)  
        x = self.up2(x)      

        a2 = self.att2(f2)   
        x = torch.cat([x, a2], dim=1) 
        x = self.up1(x)      

        heatmap = self.final_up(x)  
        score = self.global_head(a2)  

        return heatmap, score


# -----------------------------------
# StudentNet Model
# -----------------------------------



class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SpatialChannelAttention(64)
            ),
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SpatialChannelAttention(32)
            ),
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SpatialChannelAttention(16)
            ),
            nn.Sequential(
                nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                SpatialChannelAttention(8)
            )
        )

        self.decoder_heatmap = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

      
        self.decoder_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

     
        heatmap = self.decoder_heatmap(x)  
        score = self.decoder_score(x)    

        return heatmap, score
