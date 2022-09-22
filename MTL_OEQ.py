import os
from collections import OrderedDict
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
from torch import optim
import torchvision.transforms as T

from MTL_Utils import Loss_DIoU, Class_Acc, Seg_mIoU, Seg_mIoU, Weight_DWA, Metrics_IoU
from dataload import H5Dataset, chunk_dset
from ResBlock import ResidualBlock



EPOCH = 270 # all epoches
START_TRANS = 120 # pretrain epoches
LOSS_PRE_WEIGHT = [0.5,0.5]
'''
the real batch size = BATCHSIZE * CHUNK_SIZE_ 
'''
BATCHSIZE = 3
CHUNK_SIZE_TRAIN = 13
CHUNK_SIZE_VAL = 41
CHUNK_SIZE_TEST = 41

''' write your data folder path here !!!'''
H5_SAVE_PATH = r'.data/'
FILE_SAVE_PATH = r'' # model save path

class MTL_Encoder(nn.Module):

    def __init__(self, res_mode, res_block_size=1):
        super(MTL_Encoder, self).__init__()
        self.res_block_mode = res_mode
        self.res_block_size = res_block_size

        self.one2one_size = 2

        self.num_tasks_encoder = 2
        self.num_layers = 6

        self.channels = [16*(2**i) for i in range(self.num_layers)]

        ## encoder part: 3 backbone for 3 task (binary, bbox, segmentation)
        self.encoder = nn.ModuleList([nn.ModuleList([
                nn.Sequential(OrderedDict([
                ('input_conv', nn.Conv2d(3,self.channels[0],kernel_size=7, stride=2, padding=3, bias=False)),
                ('input_bn', nn.BatchNorm2d(self.channels[0])),
                ('input_relu', nn.ReLU(True))
                ]))
        ]) for _ in range(self.num_tasks_encoder)])

        for i in range(self.num_tasks_encoder):
            for j in range(self.num_layers-1):
                self.encoder[i].append(ResidualBlock(self.channels[j],self.channels[j+1],size=self.res_block_size, sampling='down', mode=self.res_block_mode))
        
        self.pooler = nn.AdaptiveAvgPool2d((self.one2one_size,self.one2one_size))
        ## for binarys
        self.bin_classifier = nn.Sequential(OrderedDict([
                ('bin_classifier_fc1', nn.Linear(self.channels[-1]*(self.one2one_size**2), 200)),
                ('bin_classifier_fc2', nn.Linear(200, 80)),
                ('bin_classifier_fc3', nn.Linear(80, 2))
            ]))

        ## for bbox
        self.bbox_classifier = nn.Sequential(OrderedDict([
                ('bbox_classifier_fc1', nn.Linear(self.channels[-1]*(self.one2one_size**2), 200)),
                ('bbox_classifier_fc2', nn.Linear(200, 80)),
                ('bbox_classifier_fc3', nn.Linear(80, 4))
            ]))


        # self.cross_stich = nn.Parameter(torch.ones(((self.num_layers-1)*self.num_tasks_encoder, self.num_tasks_encoder)) / 2)
        self.cross_stich = nn.Parameter(torch.tensor([[0.8,0.2],[0.2,0.8]]))

    def forward(self, x, if_cross = True):
        ## input block
        x_1 = self.encoder[0][0](x)
        x_2 = self.encoder[1][0](x)

        '''
        encoder part: 3 tasks (binary, bbox, segmentation)
        '''
        for i in range(1, self.num_layers):
            ## cross stitch
            if if_cross:
                cs_x_1 = x_1 * self.cross_stich[0][0] + x_2 * self.cross_stich[0][1]
                cs_x_2 = x_1 * self.cross_stich[1][0] + x_2 * self.cross_stich[1][1]
            else:
                cs_x_1 = x_1 
                cs_x_2 = x_2

            ## resnet block
            x_1 = self.encoder[0][i](cs_x_1)
            x_2 = self.encoder[1][i](cs_x_2)

        pooled_x_1 = self.pooler(x_1)
        pooled_x_2 = self.pooler(x_2)

        '''
        task 1 (binary)
        '''
        ## binary task classifer
        bin = torch.flatten(pooled_x_1, 1)
        bin = self.bin_classifier(bin) 

        '''
        task 2 (bbox)
        '''
        bbox = torch.flatten(pooled_x_2, 1)
        bbox = self.bbox_classifier(bbox)

        '''
        return feture map for segmentation
        '''
        return [bin, bbox, x_1, x_2]

class InnerMTL(nn.Module):
    
    def __init__(self, res_block_mode, res_block_size):
        super(InnerMTL,self).__init__()

        self.res_block_mode = res_block_mode
        self.res_block_size = res_block_size
        self.inner_multiple = nn.Parameter(torch.ones((3,2))/2)

        self.encoder = nn.Sequential(
                  nn.Sequential(OrderedDict([
                      ('input_conv', nn.Conv2d(3,16,kernel_size=7, stride=2, padding=3, bias=False)),
                      ('input_bn', nn.BatchNorm2d(16)),
                      ('input_relu', nn.ReLU(True))
                      ])),
                  ResidualBlock(16,32,sampling='down',size=self.res_block_size, mode=self.res_block_mode),
                  ResidualBlock(32,64,sampling='down',size=self.res_block_size, mode=self.res_block_mode),
                  ResidualBlock(64,128,sampling='down',size=self.res_block_size, mode=self.res_block_mode),
                  ResidualBlock(128,256,sampling='down',size=self.res_block_size, mode=self.res_block_mode),
                  ResidualBlock(256,512,sampling='down',size=self.res_block_size, mode=self.res_block_mode)
            )
          

        self.multi_layer_1 = nn.ModuleList()
        for i in range(3):
            self.multi_layer_1.append(
                nn.Sequential(
                    ResidualBlock(512,256,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
                    ResidualBlock(256,128,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
                )
            )
        self.multi_layer_2 = nn.ModuleList()
        for i in range(2):
            self.multi_layer_2.append(
                nn.Sequential(
                    ResidualBlock(128,64,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
                    ResidualBlock(64,32,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
                )
            )

        self.decoder = nn.Sequential(
            ResidualBlock(32,16,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
            ResidualBlock(16,3,sampling='up',size=self.res_block_size, mode=self.res_block_mode),
            nn.Conv2d(3,2,kernel_size=3, stride=1,padding=1),
            nn.Sigmoid()
        )
        self.inner_multiple = nn.Parameter(torch.ones((3,2))/2)
        self.decoder_stitch_1 = nn.Parameter(torch.tensor([[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5]]))
        self.decoder_stitch_2 = nn.Parameter(torch.tensor([[0.5,0.5],[0.5,0.5]]))

    def forward(self, x, x_1, x_2):
        x_0 = self.encoder(x)
        
        ## first multiple
        x_0 = self.multi_layer_1[0][0](x_0)
        x_1 = self.multi_layer_1[1][0](x_1)
        x_2 = self.multi_layer_1[2][0](x_2)
        cs_x_0 = x_0 * self.decoder_stitch_1[0][0] + x_1 * self.decoder_stitch_1[0][1] + x_2 * self.decoder_stitch_1[0][2]
        cs_x_1 = x_0 * self.decoder_stitch_1[1][0] + x_1 * self.decoder_stitch_1[1][1] + x_2 * self.decoder_stitch_1[1][2]
        cs_x_2 = x_0 * self.decoder_stitch_1[2][0] + x_1 * self.decoder_stitch_1[2][1] + x_2 * self.decoder_stitch_1[2][2]
        x_0 = self.multi_layer_1[0][1](cs_x_0)
        x_1 = self.multi_layer_1[1][1](cs_x_1)
        x_2 = self.multi_layer_1[2][1](cs_x_2)
        x_1 = self.inner_multiple[0][0] * x_0 + self.inner_multiple[0][1] * x_1
        x_2 = self.inner_multiple[1][0] * x_0 + self.inner_multiple[1][1] * x_2

        ## second multiple
        x_1 = self.multi_layer_2[0][0](x_1)
        x_2 = self.multi_layer_2[1][0](x_2)
        cs_x_1 = x_1 * self.decoder_stitch_2[0][0] + x_2 * self.decoder_stitch_2[0][1]
        cs_x_2 = x_1 * self.decoder_stitch_2[1][0] + x_2 * self.decoder_stitch_2[1][1]
        x_1 = self.multi_layer_2[0][1](cs_x_1)
        x_2 = self.multi_layer_2[1][1](cs_x_2)

        x_0 = self.inner_multiple[2][0] * x_1 + self.inner_multiple[2][1] * x_2

        seg_output = self.decoder(x_0)

        return seg_output

'''
Load data
'''
t1 = time.time()
chunk_dset('train', H5_SAVE_PATH, CHUNK_SIZE_TRAIN)
chunk_dset('val', H5_SAVE_PATH, CHUNK_SIZE_VAL)
chunk_dset('test', H5_SAVE_PATH, CHUNK_SIZE_TEST)
t2 = time.time()
print('chunkize time: %.2f (min)' % ((t2-t1)/60))

train_set = H5Dataset(r'train', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_TRAIN)
val_set = H5Dataset(r'val', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_VAL)
test_set = H5Dataset(r'test', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_TEST)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)


'''
initialize the model, loss function, and optimizer
'''
trans = [ 
      T.RandomPerspective(distortion_scale=0.9),
      T.RandomRotation(degrees=(0, 30))]
applier = T.RandomApply(trans, p=0.4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion_bin = torch.nn.CrossEntropyLoss()
criterion_seg = torch.nn.CrossEntropyLoss()

res_block_mode = 'normal'
net_1 = MTL_Encoder(res_block_mode, 2).to(device)
net_2 = InnerMTL(res_block_mode, 2).to(device)
optimizer_1 = optim.Adam(net_1.parameters(), lr=0.001)
optimizer_2 = optim.Adam(net_2.parameters(), lr=0.001)

if torch.cuda.is_available():
    print(device, torch.cuda.get_device_name(device))
    torch.cuda.empty_cache()


'''
start training
'''

num_batches_train = len(train_loader)
num_batches_val = len(val_loader)

## train: all_loss, bin_loss, bbox_loss, seg_loss, bin_acc, ap_bbox, seg_miou
## vali: bin_loss, bbox_loss, seg_loss, bin_acc, ap_bbox, seg_miou
history = torch.zeros((EPOCH, 13))
## start ml encoder 
for epoch in range(EPOCH):  # loop over the dataset multiple times
    
    running_loss_all = torch.zeros(num_batches_train)
    running_loss_bin = torch.zeros(num_batches_train)
    running_loss_bbox = torch.zeros(num_batches_train)
    running_loss_seg = torch.zeros(num_batches_train)
    running_bin_acc = torch.zeros(num_batches_train)
    running_bbox_iou = torch.zeros(num_batches_train)
    running_seg_miou = torch.zeros(num_batches_train)

    for index, data in enumerate(train_loader, 0):
        inputs,labels_bin, labels_bbox, labels_seg = data
        
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
        inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
        if epoch < START_TRANS: inputs = applier(inputs)
        labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
        labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
        labels_bin = labels_bin.reshape(-1)[rand].to(device)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        net_outputs = net_1(inputs, if_cross=False)
        x_1,x_2 = net_outputs[2],net_outputs[3]
        outputs = net_2(inputs,x_1,x_2)
        loss_bin = criterion_bin(net_outputs[0], labels_bin)
        loss_bbox = Loss_DIoU(net_outputs[1], labels_bbox, device)
        loss_seg = criterion_seg(outputs, labels_seg)
        if torch.isnan(loss_bin) or torch.isnan(loss_bbox) or torch.isnan(loss_seg):
            raise Exception('loss nan: bin[{}] bbox[{}] seg[{}]'.format(loss_bin,loss_bbox,loss_seg))

        if index < 2:
            loss_weight = Weight_DWA(None, num_task=2, pre_weight=LOSS_PRE_WEIGHT)
        else:
            temp = torch.cat((running_loss_bin[index-2:index],running_loss_bbox[index-2:index]))
            loss_weight = Weight_DWA(temp.reshape((2,2)), num_task=2, pre_weight=LOSS_PRE_WEIGHT, temper=1)
            
        if loss_bin < 0.1:
            loss_2A = loss_bbox
        else:
            loss_2A = loss_weight[0]*loss_bin + loss_weight[1]*loss_bbox 

        if epoch < START_TRANS: 
            loss_2A.backward()
            optimizer_1.step()
        else:
            loss_seg.backward()
            optimizer_2.step()

        loss = loss_bin + loss_bbox + loss_seg
        running_bin_acc[index] = Class_Acc(net_outputs[0].argmax(dim=1), labels_bin)
        running_bbox_iou[index] = Metrics_IoU(net_outputs[1], labels_bbox, device)
        running_seg_miou[index] = Seg_mIoU(2, outputs.argmax(dim=1), labels_seg)
        running_loss_all[index] = loss.item()
        running_loss_bin[index] = loss_bin.item()
        running_loss_bbox[index] = loss_bbox.item()
        running_loss_seg[index] = loss_seg.item()
        
    history[epoch,0],history[epoch,1],history[epoch,2],history[epoch,3] = running_loss_all.mean(), running_loss_bin.mean(), running_loss_bbox.mean(), running_loss_seg.mean()
    history[epoch,4],history[epoch,5],history[epoch,6] = running_bin_acc.mean(), running_bbox_iou.mean(), running_seg_miou.mean()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_loss_bin = torch.zeros(num_batches_val)
    val_loss_bbox = torch.zeros(num_batches_val)
    val_loss_seg = torch.zeros(num_batches_val)
    val_bin_acc = torch.zeros(num_batches_val)
    val_bbox_iou = torch.zeros(num_batches_val)
    val_seg_miou = torch.zeros(num_batches_val)
    val_seg_iou = torch.zeros(num_batches_val)
    with torch.no_grad():
        for index, data in enumerate(val_loader, 0):
            inputs,labels_bin, labels_bbox, labels_seg = data

            # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
            rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
            inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
            labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
            labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
            labels_bin = labels_bin.reshape(-1)[rand].to(device)

            net_outputs = net_1(inputs, False)
            x_1,x_2 = net_outputs[2],net_outputs[3]
            outputs = net_2(inputs,x_1,x_2)
            loss_bin = criterion_bin(net_outputs[0], labels_bin)
            loss_bbox = Loss_DIoU(net_outputs[1], labels_bbox, device)
            loss_seg = criterion_seg(outputs, labels_seg)

            val_bin_acc[index] = Class_Acc(net_outputs[0].argmax(dim=1), labels_bin)
            val_bbox_iou[index] = Metrics_IoU(net_outputs[1], labels_bbox, device)
            val_seg_miou[index] = Seg_mIoU(2, outputs.argmax(dim=1), labels_seg)
            val_loss_bin[index] = loss_bin.item()
            val_loss_bbox[index] = loss_bbox.item()
            val_loss_seg[index] = loss_seg.item()

        
    history[epoch,7],history[epoch,8],history[epoch,9] = val_loss_bin.mean(), val_loss_bbox.mean(), val_loss_seg.mean()
    history[epoch,10],history[epoch,11],history[epoch,12]  = val_bin_acc.mean(), val_bbox_iou.mean(), val_seg_miou.mean()

    if epoch % 5 == 4:
        print('[==============Epoch: %d==============]' % (epoch + 1))
        print('Training: loss_all: %.5f, loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
            (history[epoch,0],history[epoch,1],history[epoch,2],history[epoch,3]))
        print('          acc_bin: %.5f,  iou_bbox: %.5f, miou_seg: %.5f' % \
            (history[epoch,4], history[epoch,5], history[epoch,6]))
        print('Validation: loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
            (history[epoch,7],history[epoch,8],history[epoch,9]))
        print('            acc_bin: %.5f,  iou_bbox: %.5f, miou_seg: %.5f' % \
            (history[epoch,10], history[epoch,11], history[epoch,12]))

'''
save model and history
'''
torch.save(history, os.path.join(FILE_SAVE_PATH, 'oeq _history.pt'))
torch.save(net_1, os.path.join(FILE_SAVE_PATH, 'oeq_encoder_model.pt'))
torch.save(net_2, os.path.join(FILE_SAVE_PATH, 'oeq_seg_model.pt'))

'''
start test
'''
num_batches_test = len(test_loader)

test_loss_bin = torch.zeros(num_batches_test)
test_loss_bbox = torch.zeros(num_batches_test)
test_loss_seg = torch.zeros(num_batches_test)
test_bin_acc = torch.zeros(num_batches_test)
test_bbox_iou = torch.zeros(num_batches_test)
test_seg_miou = torch.zeros(num_batches_test)
test_seg_iou = torch.zeros(num_batches_test)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
with torch.no_grad():
    for index, data in enumerate(test_loader, 0):
        inputs,labels_bin, labels_bbox, labels_seg = data

        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
        inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
        labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
        labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
        labels_bin = labels_bin.reshape(-1)[rand].to(device)

        net_outputs = net_1(inputs, False)
        x_1,x_2 = net_outputs[2],net_outputs[3]
        outputs = net_2(inputs,x_1,x_2)
        loss_bin = criterion_bin(net_outputs[0], labels_bin)
        loss_bbox = Loss_DIoU(net_outputs[1], labels_bbox, device)
        loss_seg = criterion_seg(outputs, labels_seg)

        test_seg_miou[index] = Seg_mIoU(2, outputs.argmax(dim=1), labels_seg)
        test_bin_acc[index] = Class_Acc(net_outputs[0].argmax(dim=1), labels_bin)
        test_bbox_iou[index] = Metrics_IoU(net_outputs[1], labels_bbox, device)
        test_loss_bin[index] = loss_bin.item()
        test_loss_bbox[index] = loss_bbox.item()
        test_loss_seg[index] = loss_seg.item()

print('[==============Test Performance:==============]')
print('loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
    (test_loss_bin.mean(),test_loss_bbox.mean(),test_loss_seg.mean()))
print('acc_bin: %.5f,  iou_bbox: %.5f, miou_seg: %.5f' % \
    (test_bin_acc.mean(), test_bbox_iou.mean(), test_seg_miou.mean()))


with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, bin, bbox, mask = data
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(img.shape[0] * img.shape[1])
        img = img.reshape(-1, 3, 256, 256)[rand]
        mask = mask.reshape(-1, 256, 256)[rand]
        bbox = bbox.reshape(-1, 4)[rand]
        bin = bin.reshape(-1)[rand]

        inputs = img.to(device)
        net_outputs = net_1(inputs)
        x_1,x_2 = net_outputs[2],net_outputs[3]
        outputs = net_2(inputs,x_1,x_2)

        test_index = torch.randint(len(bin), (1,))[0]
        pre_bbox = net_outputs[1][test_index].cpu()
        pre_mask = outputs[test_index].cpu().argmax(dim=0)

        plt.figure(figsize=(20,4))
        fig=plt.subplot(151)
        fig.imshow(img[test_index].permute(1, 2, 0))
        fig=plt.subplot(152)
        fig.imshow(img[test_index].permute(1, 2, 0))
        rect = patches.Rectangle((bbox[test_index][0],bbox[test_index][1]),\
                      bbox[test_index][2]-bbox[test_index][0],\
                      bbox[test_index][3]-bbox[test_index][1],\
                      linewidth=2,edgecolor='r', facecolor='none')
        fig.add_patch(rect)
        fig=plt.subplot(153)
        fig.imshow(img[test_index].permute(1, 2, 0))
        rect = patches.Rectangle((pre_bbox[0],pre_bbox[1]), pre_bbox[2]-pre_bbox[0], pre_bbox[3]-pre_bbox[1],\
            linewidth=2,edgecolor='r', facecolor='none')
        fig.add_patch(rect)
        fig=plt.subplot(154)
        fig.imshow(mask[test_index].reshape(256,256))
        fig=plt.subplot(155)
        fig.imshow(pre_mask.reshape(256,256))
        plt.savefig(os.path.join(FILE_SAVE_PATH, 'oeq_test.png'))
        break


train_set.close()
val_set.close()
test_set.close()

