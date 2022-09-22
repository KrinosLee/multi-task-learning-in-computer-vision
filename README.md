# multi-task-learning-in-computer-vision

A hard-structure visual multi-task learning system able for object recognition, segmentation and classification.

This repo contains 8 .py files:

1. In "dataload.py", there is a class "H5Dataset" defined, which is used to read and load the data of the H5 file.

2. In "MTL_Utils.py", there are several functions defined, which are used to calculate the metrics needed to measure model performance. 

3. In "ResBlock.py", there defined a residual block, which will be called in subsequent multi-task learning networks. When calling this block, the following attributes need to be specified: input channels, output channels, size (the number of blocks), sampling type ('down' or 'up'), and mode (whether the residual block has a bottleneck). According to different attributes, this class will generate various residual blocks, which will be an important component of each single task (bounding box, binary classification and segmentation) in multi-task learning.
    
4. The 5 files "MTL_baseline.py", "MTL_main.py", "MTL_ablation_bbox.py", "MTL_ablation_bin.py", "MTL_OEQ.py" contain 5 different models respectively:
    MTL_baseline: The baseline network capable of performing only the target task Segmentation.
    MTL_main: The network capable of performing the three tasks for minimum required project (MRP)
    MTL_ablation_bbox: The network capable of performing segmentation task and bounding box task
    MTL_ablation_bin: The network capable of performing segmentation task and binary classification task
    OEQ: The network capable of performing the three tasks for open-ended question (OEQ). (Improvements for MTL_main)
    These 5 files not only define the various models, but also train and test the models.

5. When you run the model files (any of these 5 files: "MTL_baseline.py", "MTL_main.py", "MTL_ablation_bbox.py", "MTL_ablation_bin.py" or "MTL_OEQ.py") to train and test the related network, you MUST import "dataload.py", "MTL_Utils.py" and "ResBlock.py".

6. Additional packages(except for packages in coursework1 environment) indicated in requirements.txt can be installed with the command line 'pip install -r requirements.txt'.

7. To import data, you need to change the 'H5_SAVE_PATH' in the first line after importing libraries into your own datapath, which should be a folder path that contains three folders named 
'train', 'val', 'test'. And your data should be stored in these three folders accordingly. 

8. You should also change the 'FILE_SAVE_PATH' to a specific file folder to save the running results of each model file, which including:
1) '*_history.pt': training and validation history for each epoch.
2) '*_model_lite.pt': saved model with only the learnable parameters.
3) '*_model.pt': entire saved model.
4) '*_seg_loss.png': training and validation segmentation loss over epochs.
5) '*_test.png': test result visualisation of a single instance.

9. Common errors
If you encounter "OSError: Unable to open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')", please clear and restart the runtime.
