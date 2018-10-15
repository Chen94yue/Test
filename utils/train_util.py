#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
from torch.nn import L1Loss
from torch import nn



def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(cfg,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          val_inter=3500
          ):

    step = -1
    add_loss = L1Loss()
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode

        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            model.train(True)
            # print data
            inputs, labels, labels_swap, swap_law = data
            #print(inputs.size())
            #print(labels)
            #print(labels_swap)
            #print(swap_law[0], swap_law[1])
            #sys.pause()
            # print(labels_swap)
            # print(labels, labels_swap)
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())
            labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
            swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs, cfg)
            ##indices = torch.tensor([2*i+1 for i in range(int(outputs[2].size()[0])//2)]).cuda()
            
            #gt_swap = torch.index_select(swap_law,0,indices)
            #print(indices) 
            #print(torch.index_select(swap_law,0,indices).size())
            #print(torch.index_select(outputs[2],0,indices).size())     
            if cfg['swap'] and cfg['align']:
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels_swap)
                # loss += add_loss(torch.index_select(outputs[2],0,indices), torch.index_select(swap_law,0,indices))
                loss += add_loss(outputs[2],swap_law)
            elif cfg['swap']:
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels_swap)
            elif cfg['align']:
                loss = criterion(outputs[0], labels)
                loss += add_loss(outputs[1], gt_swap)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if step % val_inter == 0:
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects1 = 0
                val_corrects2 = 0
                val_corrects3 = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs,  labels, labels_swap, swap_law = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
                    labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).long().cuda())
                    # forward
                    if len(inputs)==1:
                        inputs = torch.cat((inputs,inputs))
                        labels = torch.cat((labels,labels))
                        labels_swap = torch.cat((labels_swap,labels_swap))
                    outputs = model(inputs,cfg)

                    if cfg['swap']:
                        loss = criterion(outputs[0], labels)
                        loss += criterion(outputs[1], labels_swap)
                        outputs1 = outputs[0] + outputs[1][:,0:cfg['numcls']] + outputs[1][:,cfg['numcls']:2*cfg['numcls']]
                        outputs2 = outputs[0]
                        outputs3 = outputs[1][:,0:cfg['numcls']] + outputs[1][:,cfg['numcls']:2*cfg['numcls']]

                    elif cfg['align']:
                        loss = criterion(outputs[0], labels)
                        outputs1 = outputs[0]
                        outputs2 = outputs1
                        outputs3 = outputs1
                    else:
                        loss = criterion(outputs, labels)
                        outputs1 = outputs
                        outputs2 = outputs1
                        outputs3 = outputs1

                    _, preds1 = torch.max(outputs1, 1)
                    _, preds2 = torch.max(outputs2, 1)
                    _, preds3 = torch.max(outputs3, 1)


                    # statistics
                    val_loss += loss.data.item()
                    # batch_corrects = torch.sum((preds == labels)).data.item()
                    # val_corrects += batch_corrects
                    batch_corrects1 = torch.sum((preds1 == labels)).data.item()
                    val_corrects1 += batch_corrects1
                    batch_corrects2 = torch.sum((preds2 == labels)).data.item()
                    val_corrects2 += batch_corrects2
                    batch_corrects3 = torch.sum((preds3 == labels)).data.item()
                    val_corrects3 += batch_corrects3


                val_loss = val_loss / val_size
                val_acc1 = val_corrects1 / len(data_set['val'])
                val_acc2 = val_corrects2 / len(data_set['val'])
                val_acc3 = val_corrects3 / len(data_set['val'])
                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f %.4f %.4f||time: %d'
                             % (dt(), epoch, val_loss, val_acc1, val_acc2, val_acc3, since))
                # save model
                save_path = os.path.join(save_dir,
                        'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_acc1))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)
            elif step % val_inter == 0:
                print(epoch)

