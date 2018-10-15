# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat
class dataset(data.Dataset):
    def __init__(self, cfg, imgroot, anno_pd, unswap=None, swap=None, totensor=None, centercrop = None, train=False):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.unswap = unswap
        self.swap = swap
        self.totensor = totensor
        self.centercrop = centercrop
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        crop_num = [self.cfg['swap_num'], self.cfg['swap_num']]
        # num_regions = crop_num[0]*crop_num[1]
        if self.train:
            #original = original.rs.rr.rc.rh
            img_unswap = self.unswap(img)
            image_unswap_list = self.crop_image(img_unswap,crop_num)
            swap_law1 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]

            if self.cfg['swap']:
                #swap = original.swap
                img_swap = self.swap(img)

                #add for experiments
                img_swap1 = self.swap(img)
                img_swap2 = self.swap(img)

                image_swap_list = self.crop_image(img_swap,crop_num)

                #add for experiments
                image_swap_list1 = self.crop_image(img_swap1,crop_num)
                image_swap_list2 = self.crop_image(img_swap2,crop_num)

                unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
                swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]

                #add for experiments
                swap_stats1 = [sum(ImageStat.Stat(im).mean) for im in image_swap_list1]
                swap_stats2 = [sum(ImageStat.Stat(im).mean) for im in image_swap_list2]


                swap_law2 = []
                for swap_im in swap_stats:
                    distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                    index = distance.index(min(distance))
                    swap_law2.append((index-24)/49)
                    #print(swap_law2)

                #add for experiments
                swap_law3 = []
                for swap_im in swap_stats1:
                    distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                    index = distance.index(min(distance))
                    swap_law3.append((index-24)/49)
                    #print(swap_law3)
                swap_law4 = []
                for swap_im in swap_stats2:
                    distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                    index = distance.index(min(distance))
                    swap_law4.append((index-24)/49)
                    #print(swap_law4)

                img_swap = self.totensor(img_swap)

                #add for experiments
                img_swap1 = self.totensor(img_swap1)
                img_swap2 = self.totensor(img_swap2)

                img_unswap = self.totensor(img_unswap)
                label = self.labels[item]-1
                label_swap = label + self.cfg['numcls']
            else:
                img_swap = img_unswap
                img_swap = self.totensor(img_swap)
                img_swap1 = img_swap
                img_swap2 = img_swap
                img_unswap = self.totensor(img_unswap)
                label = self.labels[item]-1
                label_swap = label
                swap_law2 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]
                swap_law3 = swap_law2
                swap_law4 = swap_law2
        else:
            img_unswap = self.centercrop(img)
            img_swap = img_unswap
            img_swap1 = self.totensor(img_swap)
            img_swap2 = self.totensor(img_swap)
            img_swap = self.totensor(img_swap)
            img_unswap = self.totensor(img_unswap)
            label = self.labels[item]-1
            label_swap = label
            swap_law1 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]
            swap_law2 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]
            swap_law3 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]
            swap_law4 = [(i-24)/49 for i in range(crop_num[0]*crop_num[1])]
        return img_unswap, img_swap, label, label_swap, swap_law1, swap_law2, img_swap1, img_swap2, swap_law3, swap_law4

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


def collate_fn_for_train_11(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    #swap_flg = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        label_swap.append(sample[2])
        label_swap.append(sample[3])
        swap_law.append(sample[4])
        swap_law.append(sample[5])
        #swap_flg.append(0)
        #swap_flg.append(1)
    return torch.stack(imgs, 0), label, label_swap, swap_law #, swap_flg

def collate_fn_for_train_01(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[1])
        label.append(sample[2])
        label_swap.append(sample[3])
        swap_law.append(sample[5])
    return torch.stack(imgs, 0), label, label_swap, swap_law

def collate_fn_for_train_10(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[2])
        label_swap.append(sample[2])
        swap_law.append(sample[4])
    return torch.stack(imgs, 0), label, label_swap, swap_law


def collate_fn_for_train_12(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        imgs.append(sample[6])
        label.append(sample[2])
        label.append(sample[2])
        label.append(sample[2])
        label_swap.append(sample[2])
        label_swap.append(sample[3])
        label_swap.append(sample[3])
        swap_law.append(sample[4])
        swap_law.append(sample[5])
        swap_law.append(sample[8])
    return torch.stack(imgs, 0), label, label_swap, swap_law

def collate_fn_for_train_13(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        imgs.append(sample[6])
        imgs.append(sample[7])
        label.append(sample[2])
        label.append(sample[2])
        label.append(sample[2])
        label.append(sample[2])
        label_swap.append(sample[2])
        label_swap.append(sample[3])
        label_swap.append(sample[3])
        label_swap.append(sample[3])
        swap_law.append(sample[4])
        swap_law.append(sample[5])
        swap_law.append(sample[8])
        swap_law.append(sample[9])
    return torch.stack(imgs, 0), label, label_swap, swap_law


def collate_fn_for_test(batch):
    imgs = []
    label = []
    label_swap = []
    swap_law = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[2])
        label_swap.append(sample[2])
        swap_law.append(sample[4])
    return torch.stack(imgs, 0), label, label_swap, swap_law



