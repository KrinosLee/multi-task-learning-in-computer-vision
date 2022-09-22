import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset

# open and close the h5 files
class H5Dataset(Dataset):
    def __init__(self, type, data_folder_path, chunk_size):
        self.chunk_size = chunk_size
        img_path = r'images.h5'
        mask_path = r'masks.h5' 
        bbox_path = r'bboxes.h5'
        bin_path = r'binary.h5'

        if type == 'train' or 'val' or 'test':
            self.h5_data_path = os.path.join(data_folder_path, type)
            
        self.img_file = h5py.File(os.path.join(self.h5_data_path, img_path), 'r')
        self.mask_file = h5py.File(os.path.join(self.h5_data_path, mask_path), 'r')
        self.bbox_file = h5py.File(os.path.join(self.h5_data_path, bbox_path), 'r')
        self.bin_file = h5py.File(os.path.join(self.h5_data_path, bin_path), 'r')

        self.img_key = 'images_chunk'
        self.mask_key = 'masks_chunk'
        self.bbox_key = 'bboxes_chunk'
        self.bin_key = 'binary_chunk'

        # read to existing numpy arrays
        self.img_chunk = np.empty((self.chunk_size, 256, 256, 3), dtype=np.float32)
        self.mask_chunk = np.empty((self.chunk_size, 256, 256, 1), dtype=np.float32)
        self.bbox_chunk = np.empty((self.chunk_size, 4), dtype=np.float32)
        self.bin_chunk = np.empty((self.chunk_size, 1), dtype=np.float32)

        self.length = self.img_file[self.img_key].shape[0]
        self.chunk_num = self.length // self.chunk_size
        

    def __len__(self):
        return self.chunk_num

    # only read the indexed sample into memory
    def __getitem__(self, index):
        s1 = index * self.chunk_size
        s2 = (index+1) * self.chunk_size

        self.img_file[self.img_key].read_direct(self.img_chunk, np.s_[s1:s2])
        self.mask_file[self.mask_key].read_direct(self.mask_chunk, np.s_[s1:s2])
        self.bbox_file[self.bbox_key].read_direct(self.bbox_chunk, np.s_[s1:s2])
        self.bin_file[self.bin_key].read_direct(self.bin_chunk, np.s_[s1:s2])

        img_chunk = torch.tensor(self.img_chunk, dtype=torch.float).permute(0, 3, 1, 2)/255 # (c, 3, 256, 256)
        mask_chunk = torch.tensor(self.mask_chunk, dtype=torch.long).squeeze() # (c, 256, 256)
        bbox_chunk = torch.tensor(self.bbox_chunk, dtype=torch.float) # (c, 4)
        bin_chunk = torch.tensor(self.bin_chunk, dtype=torch.long).reshape(-1) # (c, 1)

        # return img, mask, bbox, bin
        return img_chunk, bin_chunk, bbox_chunk, mask_chunk  

    def close(self):
        self.img_file.close()
        self.mask_file.close()
        self.bbox_file.close()
        self.bin_file.close()


def chunk_h5(key, chunk_size, filepath):
    chunk_key = key + '_chunk'

    with h5py.File(filepath, 'r+') as f:
        if chunk_key in list(f.keys()):
            del f[chunk_key] 

        data_size = f[key].shape[0]
        data_shape = f[key].shape
        chunk_shape = list(data_shape)
        chunk_shape[0] = chunk_size

        chunkset = f.create_dataset(chunk_key, shape=data_shape, dtype=np.float32, chunks=tuple(chunk_shape))
        for i in range(0, data_size, chunk_size):
            if data_size-i < chunk_size:
                chunkset[i:] = f[key][i:]
            else:
                chunkset[i:i+chunk_size] = f[key][i:i+chunk_size]

def chunk_dset(dset_type, h5_save_path, chunk_size):
    img_path = r'images.h5'
    mask_path = r'masks.h5' 
    bbox_path = r'bboxes.h5'
    bin_path = r'binary.h5'
    chunk_h5('images', chunk_size, os.path.join(h5_save_path, dset_type, img_path))
    chunk_h5('masks', chunk_size, os.path.join(h5_save_path, dset_type, mask_path))
    chunk_h5('bboxes', chunk_size, os.path.join(h5_save_path, dset_type, bbox_path))
    chunk_h5('binary', chunk_size, os.path.join(h5_save_path, dset_type, bin_path))

