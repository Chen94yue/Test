from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F


class resnet_swap_swaploss(nn.Module):
    def __init__(self, num_classes):
        super(resnet_swap_swaploss,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.stage1_img = nn.Sequential(*list(resnet50.children())[:5])
        self.stage2_img = nn.Sequential(*list(resnet50.children())[5:6])
        self.stage3_img = nn.Sequential(*list(resnet50.children())[6:7])
        self.stage4_img = nn.Sequential(*list(resnet50.children())[7])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, num_classes)
        self.classifier_swap = nn.Linear(2048, num_classes*2)
        self.Conv4toend = nn.Conv3d(1, 1, (1, 2, 2), (1, 2, 2), padding=0, bias=False)
        self.Conv3toend = nn.Conv3d(1, 1, (1, 4, 4), (1, 4, 4), padding=0, bias=False)
        self.Conv2toend = nn.Conv3d(1, 1, (1, 8, 8), (1, 8, 8), padding=0, bias=False)
        self.Convmask = nn.Conv2d(3840, 1, 1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)

        #x2end = x2.detach()
        #x2end = self.Conv2toend(x2end.view(-1,1,256,112,112))
        #x2end = x2end.view(-1, 256, 14, 14)
        #x2end = self.bn2(x2end)
        #x2end = self.relu(x2end)

        #x3end = x3.detach()
        #x3end = self.Conv3toend(x3end.view(-1,1,512,56,56))
        #x3end = x3end.view(-1, 512, 14, 14)
        #x3end = self.bn3(x3end)
        #x3end = self.relu(x3end)

        #x4end = x4.detach()
        #x4end = self.Conv4toend(x4end.view(-1,1,1024,28,28))
        #x4end = x4end.view(-1, 1024, 14, 14)
        #x4end = self.bn4(x4end)
        #x4end = self.relu(x4end)

        #mask = torch.cat((x5.detach(), x4end, x3end, x2end), 1)
        #mask = self.Convmask(mask)
        #mask = F.relu(mask)
        #mask = F.sigmoid(mask)

        #x5 = torch.cat((x5, x4end, x3end, x2end), 1)
        #x = x5*mask
        x = x5
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # out = []
        # out = self.classifier(x)
        out = self.classifier_swap(x)
        #print(out.size())

        return out
