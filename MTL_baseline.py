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


# import pytorch_model_summary as pms
''' write your data folder path here !!!'''
H5_SAVE_PATH = r''   # data folder path
FILE_SAVE_PATH = r'' # model save path

BATCHSIZE = 3
CHUNK_SIZE_TRAIN = 13
CHUNK_SIZE_VAL = 3
CHUNK_SIZE_TEST = 41


### load data
### chunkize original dataset
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


### Baseline model
class Seg_ResNet(nn.Module):
    
    def __init__(self):
        super(Seg_ResNet,self).__init__()

        self.res_mode = 'normal'
        self.num_tasks_encoder = 1

        self.ini_conv = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride = 2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.res_block_t = nn.ModuleList([self.ini_conv])
            
        for i in range(5):
            self.res_block_t.append(ResidualBlock(16*(2**i), 16*(2**(i+1)), mode=self.res_mode))
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3,2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.Sigmoid())

        self.seg_task = nn.Sequential(
            ResidualBlock(512,256,sampling='up', mode=self.res_mode),
            ResidualBlock(256,128,sampling='up', mode=self.res_mode),
            ResidualBlock(128,64,sampling='up', mode=self.res_mode),
            ResidualBlock(64,32,sampling='up', mode=self.res_mode),
            ResidualBlock(32,16,sampling='up', mode=self.res_mode),
            ResidualBlock(16,3,sampling='up', mode=self.res_mode),
            self.cnn
        )

        self.adapool = nn.AdaptiveAvgPool2d((2,2))

    def forward(self, x):

        ## input block
        x = self.res_block_t[0](x)

        '''
        encoder part
        '''
        for i in range(1, 6):
            ## resnet block
            x = self.res_block_t[i](x)
          
        seg_output = self.seg_task(x)

        return seg_output


### Initialize the net work
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion_seg = torch.nn.CrossEntropyLoss()

net = Seg_ResNet().to(device)
# print(pms.summary(net, torch.zeros((BATCHSIZE*CHUNK_SIZE_TRAIN, 3, 256, 256)).to(device), show_input=True, show_hierarchical=False))

optimizer = optim.Adam(net.parameters(), lr=0.001)

if torch.cuda.is_available():
    print(device, torch.cuda.get_device_name(device))
    torch.cuda.empty_cache()


### Train and Validation
EPOCH = 150

num_batches_train = len(train_loader)
num_batches_val = len(val_loader)

## train: seg_loss, seg_miou
## val: seg_loss, seg_miou
history = torch.zeros((EPOCH, 4), requires_grad=False)

## start ml encoder 
for epoch in range(EPOCH):  # loop over the dataset multiple times
    
    running_loss_seg = torch.zeros(num_batches_train)
    running_seg_miou = torch.zeros(num_batches_train)

    for index, data in enumerate(train_loader, 0):
        inputs, _, _, labels_seg = data
        
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(inputs.shape[0] * inputs.shape[1])
        inputs = inputs.reshape(-1, 3, 256, 256)[rand].to(device)
        labels_seg = labels_seg.reshape(-1, 256, 256)[rand].to(device)
        optimizer.zero_grad()

        output = net(inputs)
        loss_seg = criterion_seg(output, labels_seg)

        loss_seg.backward()
        optimizer.step()

        running_seg_miou[index] = Seg_mIoU(2, output.argmax(dim=1), labels_seg)
        running_loss_seg[index] = loss_seg.item()
        
    history[epoch, 0] = running_loss_seg.mean()
    history[epoch, 1] = running_seg_miou.mean()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        val_loss_seg = torch.zeros(num_batches_val)
        val_seg_miou = torch.zeros(num_batches_val)
        for index, data in enumerate(val_loader, 0):
            inputs, _, _, labels_seg = data

            # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
            inputs = inputs.reshape(-1, 3, 256, 256).to(device)
            labels_seg = labels_seg.reshape(-1, 256, 256).to(device)

            output = net(inputs)
            loss_seg = criterion_seg(output, labels_seg)

            val_seg_miou[index] = Seg_mIoU(2, output.argmax(dim=1), labels_seg)
            val_loss_seg[index] = loss_seg.item()

            if index%5==4:
                torch.cuda.empty_cache()
        
    history[epoch, 2] = val_loss_seg.mean()
    history[epoch, 3] = val_seg_miou.mean()

    if epoch % 5 == 4:
        print('[==============Epoch: %d==============]' % (epoch + 1))
        print('Training: loss_seg: %.5f, miou_seg: %.5f' % (history[epoch, 0], history[epoch, 1]))
        print('Validation: loss_seg: %.5f, miou_seg: %.5f' % (history[epoch, 2], history[epoch, 3]))
    

### Save history and model
torch.save(history, os.path.join(FILE_SAVE_PATH, 'baseline_history.pt'))
torch.save(net.state_dict(), os.path.join(FILE_SAVE_PATH, 'baseline_model_lite.pt'))
torch.save(net, os.path.join(FILE_SAVE_PATH, 'baseline_model.pt'))


### Test the model
num_batches_test = len(test_loader)

test_loss_seg = torch.zeros(num_batches_test)
test_seg_miou = torch.zeros(num_batches_test)

with torch.no_grad():
    for index, data in enumerate(test_loader, 0):
        inputs, _, _, labels_seg = data

        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        inputs = inputs.reshape(-1, 3, 256, 256).to(device)
        labels_seg = labels_seg.reshape(-1, 256, 256).to(device)

        output = net(inputs)
        loss_seg = criterion_seg(output, labels_seg)
        test_seg_miou[index] = Seg_mIoU(2, output.argmax(dim=1), labels_seg)
        test_loss_seg[index] = loss_seg.item()

    print('[==============Test Performance:==============]')
    print('loss_seg: %.5f, miou_seg: %.5f' % (test_loss_seg.mean(), test_seg_miou.mean()))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


### Visualisation for the first image in test_loader
with torch.no_grad():
    for i, data in enumerate(test_loader):
        
        img, _, _, mask = data
        # (batch_size, chunk_size, img) -> (batch_size*chunk_size, img) then shuffle
        rand = torch.randperm(img.shape[0] * img.shape[1])
        img = img.reshape(-1, 3, 256, 256)[rand]
        mask = mask.reshape(-1, 256, 256)[rand]

        inputs = img.to(device)
        output = net(inputs)
        pre_mask = output[0].cpu().argmax(dim=0)

        test_index = 0
        plt.figure(figsize=(12,4))
        fig=plt.subplot(131)
        fig.imshow(img[test_index].permute(1, 2, 0))
        fig.set_title('Image')
        
        fig=plt.subplot(132)
        fig.imshow(mask[test_index].reshape(256,256))
        fig.set_title('Ground Truth Mask')

        fig=plt.subplot(133)
        fig.imshow(pre_mask.reshape(256,256))
        fig.set_title('Predicted Mask')

        plt.savefig(os.path.join(FILE_SAVE_PATH, 'baseline_test.png'))

        break


### plot history
x = np.arange(EPOCH)

plt.figure()
plt.plot(x, history[:,0], label='train loss') # train loss
plt.plot(x, history[:,2], label='val loss') # val loss
plt.legend()
plt.savefig(os.path.join(FILE_SAVE_PATH, 'baseline_seg_loss.png'))

train_set.close()
val_set.close()
test_set.close()