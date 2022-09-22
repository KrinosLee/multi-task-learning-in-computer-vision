import os
from torch.utils.data import Dataset
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
from torch import optim

from MTL_Utils import Loss_DIoU, Class_Acc, Seg_mIoU, Seg_mIoU, Weight_DWA, Metrics_IoU
from dataload import H5Dataset, chunk_dset
from ResBlock import ResidualBlock

### load data
### chunkize original dataset

''' write your data folder path here !!!'''
H5_SAVE_PATH = r''   # data folder path
FILE_SAVE_PATH = r'' # model save path

BATCHSIZE = 3
CHUNK_SIZE_TRAIN = 10
CHUNK_SIZE_VAL = 10
CHUNK_SIZE_TEST = 10

t1 = time.time()
chunk_dset('train', H5_SAVE_PATH, CHUNK_SIZE_TRAIN)
chunk_dset('val', H5_SAVE_PATH, CHUNK_SIZE_VAL)
chunk_dset('test', H5_SAVE_PATH, CHUNK_SIZE_TEST)
t2 = time.time()
print('chunkize time: %.2f (min)' % ((t2-t1)/60))

train_set = H5Dataset(r'train', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_TRAIN)
val_set = H5Dataset(r'val', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_VAL)
test_set = H5Dataset(r'test', data_folder_path = H5_SAVE_PATH, chunk_size = CHUNK_SIZE_TEST)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True, num_workers=0)


### Main model
class MTL_ResNet(nn.Module):
    
    def __init__(self):
        super(MTL_ResNet,self).__init__()

        self.resblock_size = 2
        self.res_mode = 'normal'
        self.num_tasks_encoder = 3

        self.ini_conv = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride = 2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.res_block_t = nn.ModuleList([nn.ModuleList([self.ini_conv])])


        for j in range(self.num_tasks_encoder):
            if j < 2:
                self.res_block_t.append(nn.ModuleList([self.ini_conv]))
            
            for i in range(5):
                self.res_block_t[j].append(ResidualBlock(16*(2**i), 16*(2**(i+1)), size = self.resblock_size, mode=self.res_mode))
        
        # self.cross_stitch = nn.Parameter(torch.ones((5*self.num_tasks_encoder, self.num_tasks_encoder)))
        self.cross_stitch = nn.Parameter(torch.tensor([[0.6,0.3,0.1],[0.1,0.6,0.3],[0.1,0.3,0.6]]))

        self.cnn = nn.Sequential(
            nn.Conv2d(3,2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.Sigmoid())

        self.seg_task = nn.Sequential(
            ResidualBlock(512,256, sampling='up',size = self.resblock_size, mode=self.res_mode),
            ResidualBlock(256,128,sampling='up',size = self.resblock_size, mode=self.res_mode),
            ResidualBlock(128,64,sampling='up',size = self.resblock_size, mode=self.res_mode),
            ResidualBlock(64,32,sampling='up',size = self.resblock_size, mode=self.res_mode),
            ResidualBlock(32,16,sampling='up',size = self.resblock_size, mode=self.res_mode),
            ResidualBlock(16,3,sampling='up',size = self.resblock_size, mode=self.res_mode),
            self.cnn
        )

        self.adapool = nn.AdaptiveAvgPool2d((2,2))
        self.binary_task = nn.Sequential(
            nn.Linear(2048,200),
            nn.Linear(200,80),
            nn.Linear(80,2)
        )

        self.bbox_task = nn.Sequential(
            nn.Linear(2048,200),
            nn.Linear(200,80),
            nn.Linear(80,4)
        )

    def forward(self, x):

        ## input block
        x_1 = self.res_block_t[0][0](x)
        x_2 = self.res_block_t[1][0](x)
        x_3 = self.res_block_t[2][0](x)

        '''
        encoder part: 3 tasks (binary, bbox, segmentation)
        '''
        for i in range(1, 6):
            ## cross stitch
            # cs_index = (i-1) * self.num_tasks_encoder
            # cs_x_1 = x_1 * self.cross_stitch[cs_index][0] + x_2 * self.cross_stitch[cs_index][1] + x_3 * self.cross_stitch[cs_index][2]
            # cs_x_2 = x_1 * self.cross_stitch[cs_index+1][0] + x_2 * self.cross_stitch[cs_index+1][1] + x_3 * self.cross_stitch[cs_index+1][2]
            # cs_x_3 = x_1 * self.cross_stitch[cs_index+2][0] + x_2 * self.cross_stitch[cs_index+2][1] + x_3 * self.cross_stitch[cs_index+2][2]
            cs_index = 0
            cs_x_1 = x_1 * self.cross_stitch[cs_index][0] + x_2 * self.cross_stitch[cs_index][1] + x_3 * self.cross_stitch[cs_index][2]
            cs_x_2 = x_1 * self.cross_stitch[cs_index+1][0] + x_2 * self.cross_stitch[cs_index+1][1] + x_3 * self.cross_stitch[cs_index+1][2]
            cs_x_3 = x_1 * self.cross_stitch[cs_index+2][0] + x_2 * self.cross_stitch[cs_index+2][1] + x_3 * self.cross_stitch[cs_index+2][2]

            ## resnet block
            x_1 = self.res_block_t[0][i](cs_x_1)
            x_2 = self.res_block_t[1][i](cs_x_2)
            x_3 = self.res_block_t[2][i](cs_x_3)

        
        bbox_outut = self.bbox_task(torch.flatten(self.adapool(x_2),1))
        binary_output = self.binary_task(torch.flatten(self.adapool(x_1),1))
        seg_output = self.seg_task(x_3)

        return [binary_output,bbox_outut,seg_output]


### Initialize the net work
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion_bin = torch.nn.CrossEntropyLoss()
criterion_seg = torch.nn.CrossEntropyLoss()

net = MTL_ResNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
#print(summary(net,input_size = (3,256,256)))


### Train and Validation
EPOCH = 150
LOSS_PRE_WEIGHT = [0.3,1.,2.]

num_batches_train = len(train_loader)
num_batches_val = len(val_loader)

## train: all_loss, bin_loss, bbox_loss, seg_loss, bin_acc, ap_bbox, seg_miou
## vali: bin_loss, bbox_loss, seg_loss, bin_acc, ap_bbox, seg_miou
history = torch.zeros((EPOCH, 13), requires_grad=False)

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
        inputs, labels_bin, labels_bbox, labels_seg = data
        
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
        inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
        labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
        labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
        labels_bin = labels_bin.reshape(-1)[rand].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss_bin = criterion_bin(outputs[0], labels_bin)
        loss_bbox = Loss_DIoU(outputs[1], labels_bbox, device)
        loss_seg = criterion_seg(outputs[2], labels_seg)

        if index < 2:
            loss_weight = Weight_DWA(None, num_task=len(LOSS_PRE_WEIGHT), pre_weight=LOSS_PRE_WEIGHT)
        else:
            temp = torch.cat((running_loss_bin[index-2:index], running_loss_bbox[index-2:index], running_loss_seg[index-2:index]))
            loss_weight = Weight_DWA(temp.reshape((3,2)), num_task=len(LOSS_PRE_WEIGHT), pre_weight=LOSS_PRE_WEIGHT)

        if loss_bin < 0.2 and loss_bbox <0.2 :
            loss = loss_weight[2]*loss_seg
        elif loss_bin < 0.2 :
            loss = loss_weight[1]*loss_bbox + loss_weight[2]*loss_seg
        elif loss_bbox <0.2 :
            loss = loss_weight[0]*loss_bin + loss_weight[2]*loss_seg
        else:
            loss = loss_weight[0]*loss_bin + loss_weight[1]*loss_bbox + loss_weight[2]*loss_seg

        loss.backward()
        optimizer.step()

        running_bin_acc[index] = Class_Acc(outputs[0].argmax(dim=1), labels_bin)
        running_bbox_iou[index] = Metrics_IoU(outputs[1], labels_bbox, device)
        running_seg_miou[index] = Seg_mIoU(2, outputs[2].argmax(dim=1), labels_seg)
        running_loss_all[index] = loss.item()
        running_loss_bin[index] = loss_bin.item()
        running_loss_bbox[index] = loss_bbox.item()
        running_loss_seg[index] = loss_seg.item()
        
    history[epoch,0],history[epoch,1],history[epoch,2],history[epoch,3] = running_loss_all.mean(), running_loss_bin.mean(), running_loss_bbox.mean(), running_loss_seg.mean()
    history[epoch,4],history[epoch,5],history[epoch,6] = running_bin_acc.mean(), running_bbox_iou.mean(), running_seg_miou.mean()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        val_loss_bin = torch.zeros(num_batches_val)
        val_loss_bbox = torch.zeros(num_batches_val)
        val_loss_seg = torch.zeros(num_batches_val)
        val_bin_acc = torch.zeros(num_batches_val)
        val_bbox_iou = torch.zeros(num_batches_val)
        val_seg_miou = torch.zeros(num_batches_val)
        for index, data in enumerate(val_loader, 0):
            inputs,labels_bin, labels_bbox, labels_seg = data

            # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
            rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
            inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
            labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
            labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
            labels_bin = labels_bin.reshape(-1)[rand].to(device)

            outputs = net(inputs)
            loss_bin = criterion_bin(outputs[0], labels_bin)
            loss_bbox = Loss_DIoU(outputs[1], labels_bbox, device)
            loss_seg = criterion_seg(outputs[2], labels_seg)

            val_bin_acc[index] = Class_Acc(outputs[0].argmax(dim=1), labels_bin)
            val_bbox_iou[index] = Metrics_IoU(outputs[1], labels_bbox, device)
            val_seg_miou[index] = Seg_mIoU(2, outputs[2].argmax(dim=1), labels_seg)
            val_loss_bin[index] = loss_bin.item()
            val_loss_bbox[index] = loss_bbox.item()
            val_loss_seg[index] = loss_seg.item()

            if index%5==4:
                torch.cuda.empty_cache()
        
    history[epoch,7],history[epoch,8],history[epoch,9] = val_loss_bin.mean(), val_loss_bbox.mean(), val_loss_seg.mean()
    history[epoch,10],history[epoch,11],history[epoch,12]  = val_bin_acc.mean(), val_bbox_iou.mean(), val_seg_miou.mean()

    if epoch % 5 == 4:
        print('[==============Epoch: %d==============]' % (epoch + 1))
        print('Training: loss_all: %.5f, loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
            (history[epoch,0],history[epoch,1],history[epoch,2],history[epoch,3]))
        print('           acc_bin: %.5f, iou_bbox: %.5f, miou_seg: %.5f' % \
            (history[epoch,4], history[epoch,5], history[epoch,6]))
        print('Validation: loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
            (history[epoch,7],history[epoch,8],history[epoch,9]))
        print('             acc_bin: %.5f,  iou_bbox: %.5f, miou_seg: %.5f' % \
            (history[epoch,10], history[epoch,11], history[epoch,12]))


### Save history and model
torch.save(history, os.path.join(FILE_SAVE_PATH, 'main_history.pt'))
torch.save(net.state_dict(), os.path.join(FILE_SAVE_PATH, 'main_model.pt'))
torch.save(net, os.path.join(FILE_SAVE_PATH, 'main_model_lite.pt'))


### Test the model
num_batches_test = len(test_loader)

test_loss_bin = torch.zeros(num_batches_test)
test_loss_bbox = torch.zeros(num_batches_test)
test_loss_seg = torch.zeros(num_batches_test)
test_bin_acc = torch.zeros(num_batches_test)
test_bbox_iou = torch.zeros(num_batches_test)
test_seg_miou = torch.zeros(num_batches_test)
with torch.no_grad():
    for index, data in enumerate(test_loader, 0):
        inputs,labels_bin, labels_bbox, labels_seg = data

        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
        inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
        labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
        labels_bbox = labels_bbox.reshape(-1, 4)[rand].to(device)
        labels_bin = labels_bin.reshape(-1)[rand].to(device)

        outputs = net(inputs)
        loss_bin = criterion_bin(outputs[0], labels_bin)
        loss_bbox = Loss_DIoU(outputs[1], labels_bbox, device)
        loss_seg = criterion_seg(outputs[2], labels_seg)

        test_bin_acc[index] = Class_Acc(outputs[0].argmax(dim=1), labels_bin)
        test_bbox_iou = Metrics_IoU(outputs[1], labels_bbox, device)
        test_seg_miou[index] = Seg_mIoU(2, outputs[2].argmax(dim=1), labels_seg)
        test_loss_bin[index] = loss_bin.item()
        test_loss_bbox[index] = loss_bbox.item()
        test_loss_seg[index] = loss_seg.item()

    print('[==============Test Performance:==============]')
    print('loss_bin: %.5f, loss_bbox: %.5f, loss_seg: %.5f' % \
        (test_loss_bin.mean(),test_loss_bbox.mean(),test_loss_seg.mean()))
    print('acc_bin: %.5f,  iou_bbox: %.5f, miou_seg: %.5f' % \
        (test_bin_acc.mean(), test_bbox_iou.mean(), test_seg_miou.mean()))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


### Visualisation for the first image in test_loader
with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, bin, bbox, mask = data
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(img.shape[0] * img.shape[1])
        img = img.reshape(-1, 3, 256, 256)[rand]
        mask = mask.reshape(-1, 256, 256)[rand]
        bbox = bbox.reshape(-1, 4)[rand]

        inputs = img.to(device)
        outputs = net(inputs)
        
        test_index = torch.randint(len(img), (1, ))[0]
        pre_bbox = outputs[1][test_index].cpu()
        pre_mask = outputs[2][test_index].cpu().argmax(dim=0)

        plt.figure(figsize=(20,4))
        fig=plt.subplot(151)
        fig.imshow(img[test_index].permute(1, 2, 0))
        fig.set_title('Image')

        fig=plt.subplot(152)
        fig.imshow(img[test_index].permute(1, 2, 0))
        rect = patches.Rectangle((bbox[test_index][0],bbox[test_index][1]),\
                      bbox[test_index][2]-bbox[test_index][0],\
                      bbox[test_index][3]-bbox[test_index][1],\
                      linewidth=2,edgecolor='r', facecolor='none')
        fig.add_patch(rect)
        fig.set_title('Ground Truth bbox')

        fig=plt.subplot(153)
        fig.imshow(img[test_index].permute(1, 2, 0))
        rect = patches.Rectangle((pre_bbox[0],pre_bbox[1]), pre_bbox[2]-pre_bbox[0], pre_bbox[3]-pre_bbox[1],\
            linewidth=2,edgecolor='r', facecolor='none')
        fig.add_patch(rect)
        fig.set_title('Predicted bbox')

        fig=plt.subplot(154)
        fig.imshow(mask[test_index].reshape(256,256))
        fig.set_title('Ground Truth Mask')

        fig=plt.subplot(155)
        fig.imshow(pre_mask.reshape(256,256))
        fig.set_title('Predicted Mask')

        plt.savefig(os.path.join(FILE_SAVE_PATH, 'main_test.png'))

        break


### plot history
x = np.arange(EPOCH)

plt.figure()
plt.plot(x, history[:,3], label='train loss') # train loss
plt.plot(x, history[:,9], label='val loss') # val loss

plt.legend()
plt.savefig(os.path.join(FILE_SAVE_PATH, 'main_seg_loss.png'))

train_set.close()
val_set.close()
test_set.close()