from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F


class vgg(nn.Module):
    def __init__(self, num_classes):
        super(vgg,self).__init__()
        vgg16 = models.vgg16_bn(pretrained=True)
        self.stage1_img = nn.Sequential(*list(vgg16.features.children())[:14])
        self.stage2_img = nn.Sequential(*list(vgg16.features.children())[14:24])
        self.stage3_img = nn.Sequential(*list(vgg16.features.children())[24:34])
        self.stage4_img = nn.Sequential(*list(vgg16.features.children())[34:])
        self.cls1 = nn.Sequential(*list(vgg16.classifier.children())[:-1]) 
        self.cls2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)
        x = self.cls1(x5.view(x5.size(0),-1))
        # x5 = self.base(x)
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

        # x5 = torch.cat((x5, x4end, x3end, x2end), 1)
        # x = x5*mask
        #x = self.fc1(x5)
        #x = self.fc2(x)
        out = self.cls2(x)
        # print(x.size())
        # print(x5.size())
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        # out = self.classifier(x)

        return out
