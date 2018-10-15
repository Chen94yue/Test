import pandas as pd
import torch
from torchvision import datasets, models, transforms
from dataset.dataset import dataset, collate_fn_for_train_10, collate_fn_for_train_01, collate_fn_for_train_11, collate_fn_for_train_12, collate_fn_for_train_13, collate_fn_for_test


def ReadDatasetInf(data_name):
    if data_name == 'CUB':
        rawdata_root = './datasets/CUB_200_2011/all'
        train_pd = pd.read_csv("./datasets/CUB_200_2011/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
        test_pd = pd.read_csv("./datasets/CUB_200_2011/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
        numcls = 200
        numimage = 6033
    if data_name == 'CAR':
        rawdata_root = './datasets/st_car/all'
        train_pd = pd.read_csv("./datasets/st_car/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
        test_pd = pd.read_csv("./datasets/st_car/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
        numcls = 196
        numimage = 8144
    if data_name == 'BUT':
        rawdata_root = './datasets/butterfly_200/all'
        train_pd = pd.read_csv("./datasets/butterfly_200/train.txt",sep=" ",header=None, names=['ImageName', 'label'])
        test_pd = pd.read_csv("./datasets/butterfly_200/test.txt",sep=" ",header=None, names=['ImageName', 'label'])
        numcls = 200
        numimage = 10270
    print('Dataset:',data_name)
    print('train images:', train_pd.shape[0])
    print('test images:', test_pd.shape[0])
    print('num classes:', numcls)
    return rawdata_root, train_pd, test_pd, numcls, numimage


def DataTransform(cfg, rawdata_root, train_pd, test_pd):
    if True:
        data_transforms = {
            'swap': transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop((448,448)),
                transforms.RandomHorizontalFlip(),
                transforms.Randomswap((cfg['swap_num'],cfg['swap_num'])),
            ]),
            'unswap': transforms.Compose([
                transforms.Resize((512,512)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop((448,448)),
                transforms.RandomHorizontalFlip(),
            ]),
            'totensor': transforms.Compose([
                transforms.Resize((448,448)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'None': transforms.Compose([
                transforms.Resize((512,512)),
                transforms.CenterCrop((448,448)),
            ]),
        }
        data_set = {}
        data_set['train'] = dataset(
                cfg,
                imgroot=rawdata_root,
                anno_pd=train_pd,
                unswap=data_transforms["unswap"],
                swap=data_transforms["swap"],
                totensor=data_transforms["totensor"],
                centercrop = data_transforms["None"],
                train=True
                )
        data_set['val'] = dataset(
                cfg,
                imgroot=rawdata_root,
                anno_pd=test_pd,
                unswap=data_transforms["unswap"],
                swap=data_transforms["swap"],
                totensor=data_transforms["totensor"],
                centercrop = data_transforms["None"],
                train=False
                )
        dataloader = {}
        dataloader['train']=torch.utils.data.DataLoader(
                data_set['train'],
                batch_size=cfg['batch_size'],
                shuffle=True,
                num_workers=cfg['batch_size'],
                # select need origianl:swap ratio for training 1:0, 0:1, 1:1, 1:2, 1:3
                collate_fn=collate_fn_for_train_11
                )
        dataloader['val']=torch.utils.data.DataLoader(
                data_set['val'],
                batch_size=cfg['batch_size'],
                shuffle=True,
                num_workers=cfg['batch_size'],
                collate_fn=collate_fn_for_test
                )
    return data_set, dataloader

