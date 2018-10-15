import os
import sys
import datetime
import yaml
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from torch.nn import CrossEntropyLoss
from models.resnet_swap_2loss_add import resnet
from datasetinf import ReadDatasetInf, DataTransform

setting = open('swap_align.yaml')
cfg = yaml.load(setting)
print(cfg)
time = datetime.datetime.now()

print('**********************************************')

print('Set cache dir')
filename = str(time.month) + str(time.day) + str(time.hour) + cfg['experiment_name'] + '_' + cfg['dataset']
save_dir = '../net_model/' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = save_dir + '/' + filename +'.log'
trainlog(logfile)
print('done')

print('*********************************************')


# Read information of dataset
rawdata_root, train_pd, test_pd, cfg['numcls'], cfg['numimage'] = ReadDatasetInf(cfg['dataset'])
print('Set transform')
data_set, dataloader = DataTransform(cfg, rawdata_root, train_pd, test_pd)
print('done')
print('*********************************************')


print('choose model and train set')
model = resnet(num_classes=cfg['numcls'],cfg = cfg)
base_lr = cfg['learning_rate']

resume = None
if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model.cuda()
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model, device_ids=[0,1])

# set new layer's lr
ignored_params1 = list(map(id, model.module.classifier.parameters()))
ignored_params2 = list(map(id, model.module.classifier_swap.parameters()))
ignored_params3 = list(map(id, model.module.Convmask.parameters()))
ignored_params = ignored_params1 + ignored_params2 + ignored_params3
print('the num of new layers:', len(ignored_params))
base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters())
optimizer = optim.SGD([{'params': base_params},
                       {'params': model.module.classifier.parameters(), 'lr': base_lr*10},
                       {'params': model.module.classifier_swap.parameters(), 'lr': base_lr*10},
                       {'params': model.module.Convmask.parameters(), 'lr': base_lr*10},
                      ], lr = base_lr, momentum=cfg['momentum'])
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
train(cfg,
      model,
      epoch_num=cfg['epoch'],
      start_epoch=0,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,
      val_inter=int(cfg['numimage']/cfg['batch_size']),)

