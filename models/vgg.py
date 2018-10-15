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

        self.Conv5 = nn.Conv2d(512, 4096, 7, stride=3, padding=1)
        self.Conv5.weight = torch.nn.Parameter(vgg16.classifier[0].weight.view(4096,512,7,7))
        self.Conv5.bias = torch.nn.Parameter(vgg16.classifier[0].bias)
        self.BN5 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.Relu5 = nn.ReLU(inplace=True)

        self.Conv6 = nn.Conv2d(4096, 4096, 1, stride=1, padding=0)
        self.Conv6.weight = torch.nn.Parameter(vgg16.classifier[3].weight.view(4096,4096,1,1))
        self.Conv6.bias = torch.nn.Parameter(vgg16.classifier[3].bias)
        self.BN6 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.Relu6 = nn.ReLU(inplace=True)

        # print(vgg16.classifier)
        # self.cls = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        # self.base = nn.Sequential(*list(vgg16.children())[:-1])
        # print(self.cls)
        # print(*list(vgg16.features.children())[24:])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.fc1 = nn.Linear(100352, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        self.classifier = nn.Linear(4096, num_classes)
        # self.classifier_swap = nn.Linear(512, num_classes*2)

        # self.Conv4toend =i nn.Conv3d(1, 1, (1, 2, 2), (1, 2, 2), padding=0, bias=False)
        # self.Conv3toend = nn.Conv3d(1, 1, (1, 4, 4), (1, 4, 4), padding=0, bias=False)
        # self.Conv2toend = nn.Conv3d(1, 1, (1, 8, 8), (1, 8, 8), padding=0, bias=False)
        # self.Convmask = nn.Conv2d(3840, 1, 1, stride=1, padding=0, bias=False)

        # self.relu = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.bn4 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)
        x = self.Conv5(x5)
        x = self.BN5(x)
        x = self.Relu5(x)
        x = self.Conv6(x)
        x = self.BN6(x)
        x = self.Relu6(x)
        x = self.avgpool(x)
        # x5 = self.base(x)
        #v print(x5.size())
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
        x = x.view(x.size(0), -1)
        #x = self.fc1(x5)
        #x = self.fc2(x)
        out = self.classifier(x)
        # print(x.size())
        # print(x5.size())
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        # out = self.classifier(x)

        return out
