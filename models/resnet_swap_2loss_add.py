from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F


class resnet(nn.Module):
    def __init__(self, num_classes,cfg):
        super(resnet,self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.stage1_img = nn.Sequential(*list(resnet50.children())[:5])
        self.stage2_img = nn.Sequential(*list(resnet50.children())[5:6])
        self.stage3_img = nn.Sequential(*list(resnet50.children())[6:7])
        self.stage4_img = nn.Sequential(*list(resnet50.children())[7])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, num_classes)
        self.classifier_swap = nn.Linear(2048, num_classes*2)
        self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=False)
        self.avgpool2 = nn.AvgPool2d(2,stride=2)
        #self.indices = torch.tensor([2*i+1 for i in range(int(cfg['batch_size']))]).cuda()

    def forward(self, x, cfg):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)
        x = x5
        mask = self.Convmask(x)
        mask = self.avgpool2(mask)
        mask = F.tanh(mask)
        mask = mask.view(mask.size(0),-1)
        #print(mask.size())
        #mask = torch.index_select(mask,0,self.indices)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if cfg['swap'] and cfg['align']:
            out = []
            out.append(self.classifier(x))
            out.append(self.classifier_swap(x))
            out.append(mask)
        elif cfg['swap']:
            out = []
            out.append(self.classifier(x))
            out.append(self.classifier_swap(x))
        elif cfg['align']:
            out = []
            out.append(self.classifier(x))
            out.append(mask)
        else:
            out = self.classifier(x)
        return out
